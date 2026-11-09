//! The sounds a demo carries: every `svc_Sounds` message the server sent the
//! recorder (pain, voice lines, other players' rockets and pipes, buildings,
//! pickups, the map's own sounds). Each message packs one or more sounds, each
//! written as the changes from the one before (Source's `SoundInfo_t::ReadDelta`).

use tf_demo_parser::demo::message::voice::ParseSoundsMessage;

const MAX_EDICT_BITS: u32 = 11;
const MAX_SOUND_INDEX_BITS: u32 = 14;
const SND_FLAG_BITS_ENCODE: u32 = 11;
const SOUND_SEQNUMBER_BITS: u32 = 10;
const MAX_SNDLVL_BITS: u32 = 9;
const MAX_SOUND_DELAY_MSEC_ENCODE_BITS: u32 = 13;
const SOUND_ORIGIN_BITS: u32 = 12;
const SOUND_ORIGIN_SCALE: f32 = 8.0;
const SOUND_DELAY_OFFSET: f32 = 0.1;

/// Sound flags (`SoundFlags_t`).
pub const SND_CHANGE_VOL: u32 = 1 << 0;
pub const SND_CHANGE_PITCH: u32 = 1 << 1;
pub const SND_STOP: u32 = 1 << 2;
pub const SND_STOP_LOOPING: u32 = 1 << 5;
/// The weapon channel (`CHAN_WEAPON`), which guns and swings play on.
pub const CHAN_WEAPON: u8 = 1;

/// One sound, as the server sent it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundInfo {
    pub entity: u32,
    /// Index into the demo's sound list (its `soundprecache` table).
    pub sound: u32,
    pub flags: u32,
    pub channel: u32,
    pub ambient: bool,
    /// A sentence: `sound` is a sentence number, not a file.
    pub sentence: bool,
    pub sequence: u32,
    pub volume: f32,
    /// How far it carries, in decibels: 75 is a normal sound, 0 is heard everywhere.
    pub level: u32,
    /// 100 is the sound's own pitch.
    pub pitch: u32,
    pub special_dsp: u32,
    pub delay: f32,
    pub origin: [f32; 3],
    pub speaker: i32,
}

impl Default for SoundInfo {
    /// What the first sound in a message is written against (`SoundInfo_t::SetDefault`).
    fn default() -> Self {
        Self {
            entity: 0,
            sound: 0,
            flags: 0,
            channel: 6, // CHAN_STATIC
            ambient: false,
            sentence: false,
            sequence: 0,
            volume: 1.0,
            level: 75,
            pitch: 100,
            special_dsp: 0,
            delay: 0.0,
            origin: [0.0; 3],
            speaker: -1,
        }
    }
}

/// Reads one `svc_Sounds` message. `None` when it doesn't decode cleanly, which
/// means the format isn't what we expect: better no sounds than wrong ones.
pub fn read_sounds(message: &ParseSoundsMessage) -> Option<Vec<SoundInfo>> {
    let mut stream = message.data.clone();
    let mut bits = Bits { stream: &mut stream };
    let mut previous = SoundInfo::default();
    let mut sounds = Vec::with_capacity(usize::from(message.num));
    for _ in 0..message.num {
        let sound = read_delta(&mut bits, &previous)?;
        sounds.push(sound);
        previous = sound;
    }
    // Every bit of the message should have been used.
    (bits.stream.bits_left() == 0).then_some(sounds)
}

struct Bits<'s, 'a> {
    stream: &'s mut tf_demo_parser::Stream<'a>,
}

impl Bits<'_, '_> {
    fn bit(&mut self) -> Option<bool> {
        self.stream.read_bool().ok()
    }

    fn unsigned(&mut self, count: u32) -> Option<u32> {
        self.stream.read_int::<u32>(count as usize).ok()
    }

    fn signed(&mut self, count: u32) -> Option<i32> {
        let raw = self.unsigned(count)?;
        let shift = 32 - count;
        Some(((raw << shift) as i32) >> shift)
    }

    /// A value that's either unchanged from the previous sound or sent in full.
    fn changed(&mut self, count: u32, previous: u32) -> Option<u32> {
        if self.bit()? {
            self.unsigned(count)
        } else {
            Some(previous)
        }
    }
}

fn read_delta(bits: &mut Bits, previous: &SoundInfo) -> Option<SoundInfo> {
    let mut sound = *previous;
    if bits.bit()? {
        sound.entity = if bits.bit()? { bits.unsigned(5)? } else { bits.unsigned(MAX_EDICT_BITS)? };
    }
    sound.sound = bits.changed(MAX_SOUND_INDEX_BITS, previous.sound)?;
    sound.flags = bits.changed(SND_FLAG_BITS_ENCODE, previous.flags)?;
    sound.channel = bits.changed(3, previous.channel)?;
    sound.ambient = bits.bit()?;
    sound.sentence = bits.bit()?;

    if sound.flags == SND_STOP {
        return Some(sound);
    }
    sound.sequence = if bits.bit()? {
        previous.sequence
    } else if bits.bit()? {
        previous.sequence + 1
    } else {
        bits.unsigned(SOUND_SEQNUMBER_BITS)?
    };
    if bits.bit()? {
        sound.volume = bits.unsigned(7)? as f32 / 127.0;
    }
    sound.level = bits.changed(MAX_SNDLVL_BITS, previous.level)?;
    sound.pitch = bits.changed(8, previous.pitch)?;
    sound.special_dsp = bits.changed(8, previous.special_dsp)?;
    if bits.bit()? {
        let mut delay = bits.signed(MAX_SOUND_DELAY_MSEC_ENCODE_BITS)? as f32 / 1000.0;
        if delay < 0.0 {
            delay *= 10.0;
        }
        sound.delay = delay - SOUND_DELAY_OFFSET;
    }
    for axis in 0..3 {
        if bits.bit()? {
            sound.origin[axis] = bits.signed(SOUND_ORIGIN_BITS)? as f32 * SOUND_ORIGIN_SCALE;
        }
    }
    if bits.bit()? {
        sound.speaker = bits.signed(MAX_EDICT_BITS + 1)?;
    }
    Some(sound)
}

/// A sound file's name as the game's list writes it, made into a path under `sound/`:
/// the list prefixes names with characters that say how to play them (`)`, `^`, `#`...).
pub fn sound_path(name: &str) -> String {
    let trimmed = name.trim_start_matches(|c: char| "*#@><^)(}$!?&~`+%".contains(c));
    format!("sound/{}", trimmed.replace('\\', "/").to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_from_the_sound_list_become_paths() {
        assert_eq!(sound_path(")weapons/rocket_shoot.wav"), "sound/weapons/rocket_shoot.wav");
        assert_eq!(sound_path("^Weapons\\Explode1.WAV"), "sound/weapons/explode1.wav");
        assert_eq!(sound_path("vo/medic_no01.mp3"), "sound/vo/medic_no01.mp3");
    }
}

#[cfg(test)]
mod survey {
    use super::*;
    use std::collections::BTreeMap;
    use tf_demo_parser::demo::data::DemoTick;
    use tf_demo_parser::demo::message::Message;
    use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
    use tf_demo_parser::demo::parser::MessageHandler;
    use tf_demo_parser::{Demo, DemoParser, MessageType, ParserState};

    #[derive(Default)]
    struct Scan {
        names: Vec<String>,
        decoded: usize,
        failed: usize,
        sounds: Vec<SoundInfo>,
    }

    impl MessageHandler for Scan {
        type Output = Self;
        fn does_handle(message_type: MessageType) -> bool {
            message_type == MessageType::ParseSounds
        }
        fn handle_message(&mut self, message: &Message, _tick: DemoTick, _state: &ParserState) {
            if let Message::ParseSounds(sounds) = message {
                match read_sounds(sounds) {
                    Some(list) => {
                        self.decoded += 1;
                        self.sounds.extend(list);
                    }
                    None => self.failed += 1,
                }
            }
        }
        fn handle_string_entry(&mut self, table: &str, index: usize, entry: &StringTableEntry, _state: &ParserState) {
            if table == "soundprecache" {
                if self.names.len() <= index {
                    self.names.resize(index + 1, String::new());
                }
                self.names[index] = entry.text().to_string();
            }
        }
        fn into_output(self, _state: &ParserState) -> Self::Output {
            self
        }
    }

    /// `DEMO=... cargo test --release --lib demo_sounds::survey -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn list_sounds_in_a_demo() {
        let Ok(path) = std::env::var("DEMO") else { return };
        let bytes = std::fs::read(path).unwrap();
        let demo = Demo::new(&bytes);
        let (_, scan) = DemoParser::new_with_analyser(demo.get_stream(), Scan::default()).parse().unwrap();
        println!("messages decoded {} failed {} sounds {} names {}", scan.decoded, scan.failed, scan.sounds.len(), scan.names.len());
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for sound in &scan.sounds {
            let name = if sound.sentence { format!("sentence {}", sound.sound) } else { scan.names.get(sound.sound as usize).cloned().unwrap_or_else(|| format!("#{}", sound.sound)) };
            *counts.entry(name).or_default() += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, count) in sorted.iter().take(70) {
            println!("SOUND {count:6} {name}");
        }
        for sound in scan.sounds.iter().take(8) {
            println!("SAMPLE {sound:?}");
        }
    }
}
