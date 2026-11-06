//! Extraction of the voice chat recordings stored in a demo.
//!
//! TF2 demos contain every `svc_VoiceData` message the recorder received. Modern servers use the
//! "steam" voice codec: each message carries the speaker's SteamID64, followed by sub-packets with
//! Opus frames, and a CRC32 over everything before it. The frames are decoded per speaker and
//! concatenated into clips; pauses shorter than the configured gap are kept inside one clip
//! (as silence), longer pauses start a new clip.
//!
//! Legacy codecs (speex / celt without a SteamID in the payload) are not supported and are only
//! counted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::MessageHandler;
use tf_demo_parser::{Demo, DemoParser, MessageType, ParserState};

/// Name of the folder (next to the executable) that the exports are written to.
pub const VOICE_EXPORT_DIR: &str = "voicechats";
/// Default maximum gap between two recordings of one player that are still joined, in seconds.
pub const DEFAULT_MAX_GAP_SECS: f32 = 1.0;

const DEFAULT_TICK_INTERVAL: f32 = 0.015;
const DEFAULT_SAMPLE_RATE: u32 = 24_000;
const SUPPORTED_RATES: [u32; 5] = [8_000, 12_000, 16_000, 24_000, 48_000];
/// Lost frames up to this count are concealed, a larger jump resets the decoder instead.
const MAX_CONCEALED_FRAMES: u16 = 10;
/// Gaps up to this size are the tick jitter of consecutive messages, not a real pause.
const PAD_TOLERANCE_SECS: f64 = 0.1;

/// Longest Opus packet duration (120 ms) in samples at the given rate.
fn max_frame_samples(sample_rate: u32) -> usize {
    (sample_rate as usize * 12) / 100
}

/// Safe wrapper around a libopus mono decoder.
struct OpusDecoder {
    state: *mut unsafe_libopus::OpusDecoder,
}

impl OpusDecoder {
    fn new(sample_rate: u32) -> Option<Self> {
        let mut error = 0i32;
        // SAFETY: `error` is a valid out pointer; a null result is handled below.
        let state = unsafe { unsafe_libopus::opus_decoder_create(sample_rate as i32, 1, &mut error) };
        (error == 0 && !state.is_null()).then_some(Self { state })
    }

    /// Decodes one packet (or conceals one lost frame for an empty `packet`) into `pcm`.
    /// Returns the number of samples written.
    fn decode(&mut self, packet: &[u8], pcm: &mut [i16]) -> Option<usize> {
        let (data, len) = if packet.is_empty() { (std::ptr::null(), 0) } else { (packet.as_ptr(), packet.len() as i32) };
        // SAFETY: `state` is a live decoder owned by `self`; `data`/`len` describe `packet` (or are
        // null/0 for concealment) and `pcm` has room for `pcm.len()` samples.
        let samples = unsafe { unsafe_libopus::opus_decode(self.state, data, len, pcm.as_mut_ptr(), pcm.len() as i32, 0) };
        usize::try_from(samples).ok()
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        // SAFETY: `state` was created by `opus_decoder_create` and is destroyed exactly once.
        unsafe { unsafe_libopus::opus_decoder_destroy(self.state) };
    }
}

const SUBPACKET_SILENCE: u8 = 0;
const SUBPACKET_OPUS_PLC: u8 = 6;
const SUBPACKET_SAMPLE_RATE: u8 = 11;
const FRAME_RESET: u16 = 0xFFFF;

/// One recording of a single player.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceClip {
    /// Time the first sample was recorded at, in seconds since tick 0.
    pub start_secs: f64,
    pub sample_rate: u32,
    pub samples: Vec<i16>,
}

impl VoiceClip {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    fn end_secs(&self) -> f64 {
        self.start_secs + self.duration_secs()
    }
}

/// Everything that was exported or skipped for one demo.
#[derive(Debug, Clone, Default)]
pub struct VoiceExportSummary {
    pub directory: PathBuf,
    pub files: usize,
    pub players: usize,
    pub total_secs: f64,
    /// Voice messages that could not be used (legacy codec, corrupt packet, unknown sub-packet).
    pub skipped_messages: usize,
}

#[derive(Debug, Clone)]
enum VoiceItem {
    SampleRate(u32),
    Silence(usize),
    Frame { seq: u16, data: Vec<u8> },
    Reset,
}

#[derive(Debug)]
struct SteamVoicePacket {
    steam_id: u64,
    items: Vec<VoiceItem>,
}

struct RawVoiceMessage {
    tick: u32,
    data: Vec<u8>,
}

#[derive(Default)]
struct VoiceCollector {
    tick_interval: Option<f32>,
    messages: Vec<RawVoiceMessage>,
}

impl MessageHandler for VoiceCollector {
    type Output = Self;

    fn does_handle(message_type: MessageType) -> bool {
        matches!(message_type, MessageType::ServerInfo | MessageType::VoiceData)
    }

    fn handle_message(&mut self, message: &Message, tick: DemoTick, _parser_state: &ParserState) {
        match message {
            Message::ServerInfo(info) => {
                if info.interval_per_tick > 0.0 {
                    self.tick_interval = Some(info.interval_per_tick);
                }
            }
            Message::VoiceData(voice) => {
                let mut stream = voice.data.clone();
                let byte_count = stream.bits_left() / 8;
                if let Ok(bytes) = stream.read_bytes(byte_count) {
                    self.messages.push(RawVoiceMessage { tick: u32::from(tick), data: bytes.into_owned() });
                }
            }
            _ => {}
        }
    }

    fn into_output(self, _state: &ParserState) -> Self::Output {
        self
    }
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(!0u32, |crc, &b| CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8))
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn u8(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.bytes(2)?.try_into().ok()?))
    }

    fn bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        let slice = self.data.get(self.pos..self.pos.checked_add(len)?)?;
        self.pos += len;
        Some(slice)
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }
}

/// Parses a "steam" codec voice payload: `steamid64 | sub-packets | crc32`.
/// Returns `None` for anything else (legacy codecs, checksum mismatch, unknown sub-packets).
fn parse_steam_voice(data: &[u8]) -> Option<SteamVoicePacket> {
    if data.len() < 8 + 4 {
        return None;
    }
    let (body, crc_bytes) = data.split_at(data.len() - 4);
    if crc32(body) != u32::from_le_bytes(crc_bytes.try_into().ok()?) {
        return None;
    }

    let steam_id = u64::from_le_bytes(body[..8].try_into().ok()?);
    let mut reader = ByteReader { data: &body[8..], pos: 0 };
    let mut items = Vec::new();
    while !reader.is_empty() {
        match reader.u8()? {
            SUBPACKET_SAMPLE_RATE => items.push(VoiceItem::SampleRate(reader.u16()? as u32)),
            SUBPACKET_SILENCE => items.push(VoiceItem::Silence(reader.u16()? as usize)),
            SUBPACKET_OPUS_PLC => {
                let size = reader.u16()? as usize;
                let mut frames = ByteReader { data: reader.bytes(size)?, pos: 0 };
                while !frames.is_empty() {
                    let len = frames.u16()?;
                    if len == FRAME_RESET {
                        items.push(VoiceItem::Reset);
                        continue;
                    }
                    let seq = frames.u16()?;
                    items.push(VoiceItem::Frame { seq, data: frames.bytes(len as usize)?.to_vec() });
                }
            }
            _ => return None,
        }
    }
    Some(SteamVoicePacket { steam_id, items })
}

/// Decoder state and finished clips of one speaker.
struct PlayerTrack {
    sample_rate: u32,
    decoder: Option<OpusDecoder>,
    next_seq: Option<u16>,
    clips: Vec<VoiceClip>,
}

impl PlayerTrack {
    fn new() -> Self {
        Self { sample_rate: DEFAULT_SAMPLE_RATE, decoder: None, next_seq: None, clips: Vec::new() }
    }

    fn reset_decoder(&mut self) {
        self.decoder = None;
        self.next_seq = None;
    }

    /// Decodes the items of one message into PCM at the track's sample rate.
    fn decode(&mut self, items: &[VoiceItem]) -> Vec<i16> {
        let mut pcm = Vec::new();
        let mut scratch = Vec::new();
        for item in items {
            match item {
                VoiceItem::SampleRate(rate) => {
                    if *rate != self.sample_rate && SUPPORTED_RATES.contains(rate) {
                        // Audio already decoded at the previous rate must not be mixed in.
                        if !pcm.is_empty() {
                            break;
                        }
                        self.sample_rate = *rate;
                        self.reset_decoder();
                    }
                }
                VoiceItem::Silence(samples) => pcm.extend(std::iter::repeat(0i16).take(*samples)),
                VoiceItem::Reset => self.reset_decoder(),
                VoiceItem::Frame { seq, data } => {
                    if self.decoder.is_none() {
                        self.decoder = OpusDecoder::new(self.sample_rate);
                    }
                    let rate = self.sample_rate;
                    let missing = self.next_seq.map(|next| seq.wrapping_sub(next));
                    let Some(decoder) = self.decoder.as_mut() else { continue };
                    scratch.resize(max_frame_samples(rate), 0i16);
                    match missing {
                        Some(0) | None => {}
                        Some(lost) if lost <= MAX_CONCEALED_FRAMES => {
                            let plc_len = (rate / 50) as usize;
                            for _ in 0..lost {
                                if let Some(n) = decoder.decode(&[], &mut scratch[..plc_len]) {
                                    pcm.extend_from_slice(&scratch[..n]);
                                }
                            }
                        }
                        Some(_) => {
                            // Too much was lost to conceal: start over with a fresh decoder.
                            self.decoder = OpusDecoder::new(rate);
                        }
                    }
                    let Some(decoder) = self.decoder.as_mut() else { continue };
                    match decoder.decode(data, &mut scratch) {
                        Some(n) => {
                            pcm.extend_from_slice(&scratch[..n]);
                            self.next_seq = Some(seq.wrapping_add(1));
                        }
                        None => self.next_seq = None,
                    }
                }
            }
        }
        pcm
    }

    /// Appends audio that was received at `arrival_secs` (the time its last sample was recorded at).
    /// It continues the previous clip when the pause before it is at most `max_gap_secs`.
    fn push_audio(&mut self, arrival_secs: f64, pcm: Vec<i16>, max_gap_secs: f64) {
        if pcm.is_empty() {
            return;
        }
        let rate = self.sample_rate;
        let start = arrival_secs - pcm.len() as f64 / rate as f64;
        if let Some(clip) = self.clips.last_mut() {
            let gap = start - clip.end_secs();
            if clip.sample_rate == rate && gap <= max_gap_secs {
                if gap > PAD_TOLERANCE_SECS {
                    let silence = (gap * rate as f64).round() as usize;
                    clip.samples.extend(std::iter::repeat(0i16).take(silence));
                }
                clip.samples.extend_from_slice(&pcm);
                return;
            }
        }
        self.clips.push(VoiceClip { start_secs: start, sample_rate: rate, samples: pcm });
    }
}

/// Extracts the voice recordings of every speaker, ordered by SteamID64, from the demo bytes.
/// Returns the clips together with the number of unusable voice messages.
pub fn extract_voice_clips(
    demo_bytes: &[u8],
    max_gap_secs: f32,
) -> Result<(BTreeMap<u64, Vec<VoiceClip>>, usize), String> {
    let demo = Demo::new(demo_bytes);
    let parser = DemoParser::new_with_analyser(demo.get_stream(), VoiceCollector::default());
    let (_header, collector) = parser.parse().map_err(|e| format!("Failed to parse demo: {e}"))?;

    let tick_interval = collector.tick_interval.unwrap_or(DEFAULT_TICK_INTERVAL) as f64;
    let mut tracks: BTreeMap<u64, PlayerTrack> = BTreeMap::new();
    let mut skipped = 0;
    for message in &collector.messages {
        // Empty messages only tell the clients that the player stopped talking.
        if message.data.is_empty() {
            continue;
        }
        let Some(packet) = parse_steam_voice(&message.data) else {
            skipped += 1;
            continue;
        };
        let track = tracks.entry(packet.steam_id).or_insert_with(PlayerTrack::new);
        let pcm = track.decode(&packet.items);
        track.push_audio(message.tick as f64 * tick_interval, pcm, max_gap_secs.max(0.0) as f64);
    }

    let clips = tracks
        .into_iter()
        .filter(|(_, track)| !track.clips.is_empty())
        .map(|(steam_id, track)| (steam_id, track.clips))
        .collect();
    Ok((clips, skipped))
}

/// Encodes mono 16-bit PCM as a WAV file.
pub fn encode_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

fn clip_file_name(steam_id: u64, index: usize, clip_count: usize) -> String {
    if clip_count == 1 {
        format!("{steam_id}.wav")
    } else {
        format!("{steam_id}_{}.wav", index + 1)
    }
}

/// True for the files this export creates (`<steamid>.wav` / `<steamid>_<n>.wav`).
fn is_export_file(path: &Path) -> bool {
    let is_wav = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let (id, index) = stem.split_once('_').unwrap_or((stem, "0"));
    is_wav && !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit()) && index.bytes().all(|b| b.is_ascii_digit())
}

fn sanitize_folder_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ' | '(' | ')') { c } else { '_' })
        .collect();
    let cleaned = cleaned.trim().trim_end_matches('.').to_string();
    if cleaned.is_empty() { "demo".to_string() } else { cleaned }
}

/// Exports the voice chats of a demo to `<out_root>/<demo name>/<steamid>.wav`.
/// Files of an earlier export of the same demo are replaced.
pub fn export_voice_chats(
    demo_path: &Path,
    out_root: &Path,
    max_gap_secs: f32,
) -> Result<VoiceExportSummary, String> {
    let demo_bytes = std::fs::read(demo_path).map_err(|e| format!("Failed to read demo: {e}"))?;
    let (clips, skipped_messages) = extract_voice_clips(&demo_bytes, max_gap_secs)?;
    if clips.is_empty() {
        return Err(if skipped_messages > 0 {
            format!("No usable voice data ({skipped_messages} voice message(s) use an unsupported codec)")
        } else {
            "The demo contains no voice chat".to_string()
        });
    }

    let demo_name = demo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("demo");
    let directory = out_root.join(sanitize_folder_name(demo_name));
    std::fs::create_dir_all(&directory).map_err(|e| format!("Failed to create {}: {e}", directory.display()))?;
    if let Ok(entries) = std::fs::read_dir(&directory) {
        for path in entries.flatten().map(|e| e.path()).filter(|p| is_export_file(p)) {
            let _ = std::fs::remove_file(path);
        }
    }

    let mut summary = VoiceExportSummary { directory: directory.clone(), skipped_messages, ..Default::default() };
    for (steam_id, player_clips) in &clips {
        summary.players += 1;
        for (index, clip) in player_clips.iter().enumerate() {
            let path = directory.join(clip_file_name(*steam_id, index, player_clips.len()));
            std::fs::write(&path, encode_wav(clip.sample_rate, &clip.samples))
                .map_err(|e| format!("Failed to write {}: {e}", path.display()))?;
            summary.files += 1;
            summary.total_secs += clip.duration_secs();
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steam_packet(steam_id: u64, subpackets: &[u8]) -> Vec<u8> {
        let mut data = steam_id.to_le_bytes().to_vec();
        data.extend_from_slice(subpackets);
        let crc = crc32(&data);
        data.extend_from_slice(&crc.to_le_bytes());
        data
    }

    #[test]
    fn crc32_matches_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn parses_steam_voice_packet() {
        let mut sub = vec![SUBPACKET_SAMPLE_RATE];
        sub.extend_from_slice(&24_000u16.to_le_bytes());
        sub.push(SUBPACKET_SILENCE);
        sub.extend_from_slice(&480u16.to_le_bytes());
        // Two frames: (len 3, seq 7) and (len 1, seq 8)
        let mut frames = Vec::new();
        frames.extend_from_slice(&3u16.to_le_bytes());
        frames.extend_from_slice(&7u16.to_le_bytes());
        frames.extend_from_slice(&[1, 2, 3]);
        frames.extend_from_slice(&1u16.to_le_bytes());
        frames.extend_from_slice(&8u16.to_le_bytes());
        frames.push(9);
        sub.push(SUBPACKET_OPUS_PLC);
        sub.extend_from_slice(&(frames.len() as u16).to_le_bytes());
        sub.extend_from_slice(&frames);

        let packet = parse_steam_voice(&steam_packet(76561198000000000, &sub)).unwrap();
        assert_eq!(packet.steam_id, 76561198000000000);
        assert_eq!(packet.items.len(), 4);
        assert!(matches!(packet.items[0], VoiceItem::SampleRate(24_000)));
        assert!(matches!(packet.items[1], VoiceItem::Silence(480)));
        assert!(matches!(&packet.items[2], VoiceItem::Frame { seq: 7, data } if data == &[1, 2, 3]));
        assert!(matches!(&packet.items[3], VoiceItem::Frame { seq: 8, data } if data == &[9]));
    }

    #[test]
    fn rejects_corrupt_or_unknown_packets() {
        let mut packet = steam_packet(1, &[SUBPACKET_SAMPLE_RATE, 0x80, 0x5D]);
        assert!(parse_steam_voice(&packet).is_some());
        packet[9] ^= 1;
        assert!(parse_steam_voice(&packet).is_none());
        assert!(parse_steam_voice(&steam_packet(1, &[200, 0, 0])).is_none());
        assert!(parse_steam_voice(&[1, 2, 3]).is_none());
    }

    fn pcm(secs: f64) -> Vec<i16> {
        vec![100; (secs * DEFAULT_SAMPLE_RATE as f64) as usize]
    }

    #[test]
    fn joins_recordings_within_the_gap() {
        let mut track = PlayerTrack::new();
        track.push_audio(10.0, pcm(1.0), 1.0);
        // Starts 0.5 s after the first one ended: joined, with the pause kept as silence.
        track.push_audio(10.5 + 1.0, pcm(1.0), 1.0);
        assert_eq!(track.clips.len(), 1);
        let clip = &track.clips[0];
        assert_eq!(clip.samples.len(), 2 * 24_000 + 12_000);
        assert!((clip.start_secs - 9.0).abs() < 1e-9);

        // A 3 s pause starts a new clip.
        track.push_audio(14.5 + 1.0, pcm(1.0), 1.0);
        assert_eq!(track.clips.len(), 2);
        assert!((track.clips[1].start_secs - 14.5).abs() < 1e-9);
    }

    #[test]
    fn tick_jitter_is_not_padded() {
        let mut track = PlayerTrack::new();
        track.push_audio(10.0, pcm(0.02), 1.0);
        track.push_audio(10.0 + 0.02 + 0.015, pcm(0.02), 1.0);
        assert_eq!(track.clips.len(), 1);
        assert_eq!(track.clips[0].samples.len(), 2 * 480);
    }

    #[test]
    fn zero_gap_only_joins_continuous_audio() {
        let mut track = PlayerTrack::new();
        track.push_audio(10.0, pcm(1.0), 0.0);
        track.push_audio(11.0, pcm(1.0), 0.0);
        assert_eq!(track.clips.len(), 1);
        track.push_audio(12.5, pcm(1.0), 0.0);
        assert_eq!(track.clips.len(), 2);
    }

    #[test]
    fn wav_header_is_valid() {
        let wav = encode_wav(24_000, &[1, -2, 3]);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize, wav.len() - 8);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(wav.len(), 50);
    }

    #[test]
    fn export_file_names() {
        assert_eq!(clip_file_name(7656, 0, 1), "7656.wav");
        assert_eq!(clip_file_name(7656, 1, 3), "7656_2.wav");
        assert!(is_export_file(Path::new("7656.wav")));
        assert!(is_export_file(Path::new("7656_12.wav")));
        assert!(!is_export_file(Path::new("notes.wav")));
        assert!(!is_export_file(Path::new("7656.txt")));
        assert_eq!(sanitize_folder_name("a/b:c"), "a_b_c");
    }
}
