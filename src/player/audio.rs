//! Plays TF2's sounds for the 3D player: decodes the game's WAV and MP3 files and
//! mixes them on a thread of its own into the computer's default output device.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;

/// The most sounds playing at once. Past this the oldest makes way.
const MAX_VOICES: usize = 96;
/// How long to wait for the output device to open.
const OPEN_TIMEOUT: Duration = Duration::from_secs(3);
/// Source's reference for how far sounds carry: a 60 dB sound at 36 units.
const REFERENCE_DB: f32 = 60.0;
const REFERENCE_DISTANCE: f32 = 36.0;
/// How far left or right a sound can pull, out of 1. Keeps hard-panned sounds audible in both ears.
const PAN_WIDTH: f32 = 0.8;
/// Room for many sounds at once: TF2's footsteps alone are 96 dB sounds, and a
/// fight has a dozen of them going at a time.
const MASTER_GAIN: f32 = 0.45;
/// Above this the mix is rounded off instead of cut, so loud moments don't crackle.
const LIMITER_KNEE: f32 = 0.6;

/// A decoded sound, mixed down to one channel.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub samples: Vec<f32>,
    pub rate: u32,
    /// Where a sound made to loop jumps back to (its WAV cue point).
    pub loop_start: Option<usize>,
}

/// Reads a sound file by its extension: WAV or MP3.
pub fn decode(path: &str, bytes: &[u8]) -> Option<Clip> {
    if path.ends_with(".mp3") {
        decode_mp3(bytes)
    } else {
        decode_wav(bytes)
    }
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// Reads 8 or 16-bit PCM or 32-bit float WAV. Compressed WAVs (ADPCM) give `None`.
pub fn decode_wav(bytes: &[u8]) -> Option<Clip> {
    if bytes.get(0..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
        return None;
    }
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    let mut loop_start = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4)? as usize;
        let body = &bytes[at + 8..(at + 8).saturating_add(size).min(bytes.len())];
        match id {
            b"fmt " => {
                format = u16_at(body, 0)?;
                channels = u16_at(body, 2)?;
                rate = u32_at(body, 4)?;
                bits = u16_at(body, 14)?;
            }
            b"data" => data = Some(body),
            // The first cue point is where a looping sound loops back to.
            b"cue " => {
                if u32_at(body, 0)? > 0 {
                    loop_start = u32_at(body, 4 + 20).map(|offset| offset as usize);
                }
            }
            _ => {}
        }
        at = at.saturating_add(8 + size + (size & 1));
    }
    let data = data?;
    let channels = usize::from(channels.max(1));
    let samples: Vec<f32> = match (format, bits) {
        (1, 8) => data.chunks_exact(channels).map(|frame| frame.iter().map(|&b| (f32::from(b) - 128.0) / 128.0).sum::<f32>() / channels as f32).collect(),
        (1, 16) => data
            .chunks_exact(2 * channels)
            .map(|frame| frame.chunks_exact(2).map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0).sum::<f32>() / channels as f32)
            .collect(),
        (3, 32) => data
            .chunks_exact(4 * channels)
            .map(|frame| frame.chunks_exact(4).map(|four| f32::from_le_bytes([four[0], four[1], four[2], four[3]])).sum::<f32>() / channels as f32)
            .collect(),
        _ => return None,
    };
    if samples.is_empty() || rate == 0 {
        return None;
    }
    let loop_start = loop_start.filter(|&start| start < samples.len());
    Some(Clip { samples, rate, loop_start })
}

/// Reads an MP3, which TF2 uses for its voice lines.
pub fn decode_mp3(bytes: &[u8]) -> Option<Clip> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let source = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(&hint, source, &FormatOptions::default(), &MetadataOptions::default())
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default()).ok()?;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut samples = Vec::new();
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else { continue };
        let spec = *decoded.spec();
        rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        samples.extend(buffer.samples().chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32));
    }
    (!samples.is_empty() && rate > 0).then_some(Clip { samples, rate, loop_start: None })
}

/// How loud a sound is in each ear: quieter with distance the way Source does it
/// (falling off with distance, slower for louder sounds), and to the side it's on.
/// A `level` of 0 is heard the same everywhere.
pub fn ear_gains(eye: Vec3, right: Vec3, origin: Vec3, level: f32, volume: f32) -> [f32; 2] {
    if level <= 0.0 {
        return [volume * std::f32::consts::FRAC_1_SQRT_2; 2];
    }
    let offset = origin - eye;
    let distance = offset.length();
    let distance_scale = 10f32.powf((REFERENCE_DB - level) / 20.0) / REFERENCE_DISTANCE;
    let relative = distance * distance_scale;
    let falloff = if relative > 0.1 { (1.0 / relative).min(1.0) } else { 1.0 };
    let pan = if distance > 1.0 { (offset / distance).dot(right).clamp(-1.0, 1.0) * PAN_WIDTH } else { 0.0 };
    let angle = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
    let gain = volume * falloff;
    [gain * angle.cos(), gain * angle.sin()]
}

/// One sound playing.
struct Voice {
    clip: Arc<Clip>,
    position: f64,
    /// Samples of the clip per output sample: its rate against the output's, times its pitch.
    step: f64,
    gains: [f32; 2],
    tag: u64,
    looping: bool,
}

#[derive(Default)]
struct Mixer {
    voices: Vec<Voice>,
    volume: f32,
    mix: Vec<f32>,
    /// A copy of everything played, left and right, while recording it.
    tap: Option<Vec<f32>>,
    /// While recording, keep the speakers quiet.
    silent: bool,
}

/// Leaves quiet sound alone and bends loud sound smoothly towards full scale.
fn limit(sample: f32) -> f32 {
    let size = sample.abs();
    if size <= LIMITER_KNEE {
        return sample;
    }
    let room = 1.0 - LIMITER_KNEE;
    (LIMITER_KNEE + room * ((size - LIMITER_KNEE) / room).tanh()).copysign(sample)
}

impl Mixer {
    /// Fills an interleaved output buffer of `channels` channels.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        let frames = out.len() / channels.max(1);
        self.mix.clear();
        self.mix.resize(frames * 2, 0.0);
        for voice in &mut self.voices {
            let samples = &voice.clip.samples;
            let length = samples.len() as f64;
            for frame in 0..frames {
                if voice.position >= length {
                    match (voice.looping, voice.clip.loop_start) {
                        (true, Some(start)) if (start as f64) < length => {
                            voice.position = start as f64 + (voice.position - length) % (length - start as f64);
                        }
                        _ => break,
                    }
                }
                let index = voice.position as usize;
                let fraction = (voice.position - index as f64) as f32;
                let now = samples[index];
                let next = samples.get(index + 1).copied().unwrap_or(now);
                let sample = now + (next - now) * fraction;
                self.mix[frame * 2] += sample * voice.gains[0];
                self.mix[frame * 2 + 1] += sample * voice.gains[1];
                voice.position += voice.step;
            }
        }
        self.voices.retain(|voice| voice.position < voice.clip.samples.len() as f64 || (voice.looping && voice.clip.loop_start.is_some()));
        let volume = self.volume * MASTER_GAIN;
        for (frame, out) in out.chunks_mut(channels.max(1)).enumerate() {
            let mut left = limit(self.mix[frame * 2] * volume);
            let mut right = limit(self.mix[frame * 2 + 1] * volume);
            if let Some(tap) = &mut self.tap {
                tap.extend_from_slice(&[left, right]);
                if self.silent {
                    (left, right) = (0.0, 0.0);
                }
            }
            match out {
                [mono] => *mono = (left + right) * 0.5,
                [l, r, rest @ ..] => {
                    *l = left;
                    *r = right;
                    rest.fill(0.0);
                }
                [] => {}
            }
        }
    }
}

/// The sound output: a mixer that the output device pulls from on its own thread.
pub struct Audio {
    mixer: Arc<Mutex<Mixer>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    rate: u32,
}

impl Audio {
    /// Opens the default output device. `None` when there's no sound device.
    pub fn start() -> Option<Self> {
        let mixer = Arc::new(Mutex::new(Mixer { volume: 1.0, ..Mixer::default() }));
        let stop = Arc::new(AtomicBool::new(false));
        let (opened, rate) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("demo player sound".into())
            .spawn({
                let mixer = Arc::clone(&mixer);
                let stop = Arc::clone(&stop);
                move || {
                    let stream = open_stream(mixer);
                    let _ = opened.send(stream.as_ref().map(|(_, rate)| *rate));
                    if stream.is_none() {
                        return;
                    }
                    // The stream plays for as long as it's kept.
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::park_timeout(Duration::from_millis(250));
                    }
                    drop(stream);
                }
            })
            .ok()?;
        let rate = rate.recv_timeout(OPEN_TIMEOUT).ok().flatten();
        let audio = Self { mixer, stop, thread: Some(thread), rate: rate.unwrap_or(0) };
        rate.map(|_| audio)
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Starts keeping a copy of everything played, to check by ear or by numbers.
    /// `silent` keeps it off the speakers meanwhile.
    pub fn start_recording(&self, silent: bool) {
        self.with_mixer(|mixer| {
            mixer.tap = Some(Vec::new());
            mixer.silent = silent;
        });
    }

    /// What was played since `start_recording`, left and right interleaved.
    pub fn take_recording(&self) -> Vec<f32> {
        let mut taken = Vec::new();
        self.with_mixer(|mixer| taken = mixer.tap.take().unwrap_or_default());
        taken
    }

    fn with_mixer(&self, change: impl FnOnce(&mut Mixer)) {
        if let Ok(mut mixer) = self.mixer.lock() {
            change(&mut mixer);
        }
    }

    /// Starts a sound. `tag` lets it be changed or stopped later; `from` skips into it, in seconds.
    pub fn play(&self, clip: Arc<Clip>, gains: [f32; 2], pitch: f32, tag: u64, looping: bool, from: f64) {
        let step = f64::from(clip.rate) / f64::from(self.rate.max(1)) * f64::from(pitch.clamp(0.25, 4.0));
        let position = from.max(0.0) * f64::from(clip.rate);
        self.with_mixer(|mixer| {
            if mixer.voices.len() >= MAX_VOICES {
                mixer.voices.remove(0);
            }
            mixer.voices.push(Voice { clip, position, step, gains, tag, looping });
        });
    }

    /// Changes how a playing sound is heard. False when nothing with that tag is playing.
    pub fn update(&self, tag: u64, gains: [f32; 2], pitch: f32) -> bool {
        let rate = f64::from(self.rate.max(1));
        let mut found = false;
        self.with_mixer(|mixer| {
            for voice in mixer.voices.iter_mut().filter(|voice| voice.tag == tag) {
                voice.gains = gains;
                voice.step = f64::from(voice.clip.rate) / rate * f64::from(pitch.clamp(0.25, 4.0));
                found = true;
            }
        });
        found
    }

    pub fn stop(&self, tag: u64) {
        self.with_mixer(|mixer| mixer.voices.retain(|voice| voice.tag != tag));
    }

    /// Stops every sound whose tag matches under `mask`.
    pub fn stop_matching(&self, tag: u64, mask: u64) {
        self.with_mixer(|mixer| mixer.voices.retain(|voice| voice.tag & mask != tag & mask));
    }

    pub fn stop_all(&self) {
        self.with_mixer(|mixer| mixer.voices.clear());
    }

    pub fn set_volume(&self, volume: f32) {
        self.with_mixer(|mixer| mixer.volume = volume.clamp(0.0, 1.0));
    }

}

impl Drop for Audio {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

/// Opens the default output device with the mixer feeding it, and its sample rate.
fn open_stream(mixer: Arc<Mutex<Mixer>>) -> Option<(cpal::Stream, u32)> {
    let device = cpal::default_host().default_output_device()?;
    let supported = device.default_output_config().ok()?;
    let rate = supported.sample_rate().0;
    let channels = usize::from(supported.channels());
    let config: cpal::StreamConfig = supported.config();
    let report = |error| eprintln!("sound output: {error}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config,
            move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if let Ok(mut mixer) = mixer.lock() {
                    mixer.fill(out, channels);
                } else {
                    out.fill(0.0);
                }
            },
            report,
            None,
        ),
        cpal::SampleFormat::I16 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &config,
                move |out: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    scratch.resize(out.len(), 0.0);
                    if let Ok(mut mixer) = mixer.lock() {
                        mixer.fill(&mut scratch, channels);
                    }
                    for (out, sample) in out.iter_mut().zip(&scratch) {
                        *out = (sample * 32767.0) as i16;
                    }
                },
                report,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &config,
                move |out: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    scratch.resize(out.len(), 0.0);
                    if let Ok(mut mixer) = mixer.lock() {
                        mixer.fill(&mut scratch, channels);
                    }
                    for (out, sample) in out.iter_mut().zip(&scratch) {
                        *out = ((sample * 32767.0) as i32 + 32768) as u16;
                    }
                },
                report,
                None,
            )
        }
        _ => return None,
    }
    .ok()?;
    stream.play().ok()?;
    Some((stream, rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format: u16, channels: u16, bits: u16, data: &[u8], cue: Option<u32>) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&format.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&22050u32.to_le_bytes());
        fmt.extend_from_slice(&0u32.to_le_bytes());
        fmt.extend_from_slice(&0u16.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut body = b"WAVE".to_vec();
        for (id, chunk) in [(b"fmt ", fmt), (b"data", data.to_vec())] {
            body.extend_from_slice(id);
            body.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            body.extend_from_slice(&chunk);
            if chunk.len() % 2 == 1 {
                body.push(0);
            }
        }
        if let Some(offset) = cue {
            let mut chunk = 1u32.to_le_bytes().to_vec();
            chunk.extend_from_slice(&[0; 20]);
            chunk.extend_from_slice(&offset.to_le_bytes());
            body.extend_from_slice(b"cue ");
            body.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            body.extend_from_slice(&chunk);
        }
        let mut file = b"RIFF".to_vec();
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);
        file
    }

    #[test]
    fn wav_files_decode_to_one_channel() {
        // 16-bit stereo: two frames, mixed down.
        let data: Vec<u8> = [16384i16, 0, -32768, -32768].iter().flat_map(|s| s.to_le_bytes()).collect();
        let clip = decode_wav(&wav(1, 2, 16, &data, Some(1))).unwrap();
        assert_eq!(clip.rate, 22050);
        assert_eq!(clip.samples, vec![0.25, -1.0]);
        assert_eq!(clip.loop_start, Some(1));
        // 8-bit is unsigned around 128, and an odd-sized chunk is padded.
        let clip = decode_wav(&wav(1, 1, 8, &[128, 255, 0], None)).unwrap();
        assert_eq!(clip.samples.len(), 3);
        assert_eq!(clip.samples[0], 0.0);
        assert!(clip.samples[2] == -1.0 && clip.samples[1] > 0.99);
        // Compressed WAVs aren't read.
        assert!(decode_wav(&wav(2, 1, 4, &[1, 2, 3, 4], None)).is_none());
        assert!(decode_wav(b"not a wav").is_none());
    }

    #[test]
    fn sounds_fade_with_distance_and_come_from_their_side() {
        let eye = Vec3::ZERO;
        let right = Vec3::new(0.0, -1.0, 0.0);
        let near = ear_gains(eye, right, Vec3::new(100.0, 0.0, 0.0), 75.0, 1.0);
        let far = ear_gains(eye, right, Vec3::new(2000.0, 0.0, 0.0), 75.0, 1.0);
        assert!(far[0] < near[0] * 0.2, "{near:?} {far:?}");
        assert!((near[0] - near[1]).abs() < 1e-5, "straight ahead is centred");
        let to_the_right = ear_gains(eye, right, Vec3::new(0.0, -300.0, 0.0), 75.0, 1.0);
        assert!(to_the_right[1] > to_the_right[0] * 2.0, "{to_the_right:?}");
        // A gunshot carries further than footsteps.
        let loud = ear_gains(eye, right, Vec3::new(2000.0, 0.0, 0.0), 94.0, 1.0);
        assert!(loud[0] > far[0] * 4.0);
        // Level 0 is everywhere at once.
        assert_eq!(ear_gains(eye, right, Vec3::splat(9000.0), 0.0, 1.0)[0], std::f32::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn the_mixer_plays_loops_and_ends_one_shots() {
        let clip = Arc::new(Clip { samples: vec![1.0, 1.0, 0.5, 0.5], rate: 4, loop_start: Some(2) });
        let mut mixer = Mixer { volume: 1.0, ..Mixer::default() };
        mixer.voices.push(Voice { clip: Arc::clone(&clip), position: 0.0, step: 1.0, gains: [1.0, 0.5], tag: 1, looping: false });
        mixer.voices.push(Voice { clip, position: 0.0, step: 1.0, gains: [1.0, 0.0], tag: 2, looping: true });
        let mut out = vec![0.0; 12];
        mixer.fill(&mut out, 2);
        // Frame 0: both voices at 1.0, so left is loud and rounded off below full scale.
        assert!(out[0] > 0.6 && out[0] < 1.0, "{}", out[0]);
        assert!((out[1] - 0.5 * MASTER_GAIN).abs() < 1e-6);
        // Frame 4: the one-shot has ended, the loop is back on its second half.
        assert!((out[8] - 0.5 * MASTER_GAIN).abs() < 1e-6 && out[9] == 0.0);
        assert_eq!(mixer.voices.len(), 1, "the one-shot is done");
        assert_eq!(mixer.voices[0].tag, 2);
    }
}

#[cfg(test)]
mod limiter_tests {
    use super::*;

    #[test]
    fn the_limiter_only_touches_loud_sound() {
        assert_eq!(limit(0.3), 0.3);
        assert_eq!(limit(-LIMITER_KNEE), -LIMITER_KNEE);
        let loud = limit(3.0);
        assert!(loud > 0.95 && loud < 1.0, "{loud}");
        assert!(limit(-3.0) == -loud);
        assert!(limit(0.9) > limit(0.8) && limit(0.8) > limit(0.7), "still gets louder");
    }
}
