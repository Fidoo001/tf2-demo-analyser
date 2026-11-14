//! Lightweight lookup of the POV demo recorder (CLI `--recorder`).
//!
//! The recorder is identified by `SVC_ServerInfo::player_slot`, the recording client's own
//! slot (entity index = slot + 1, `userinfo` string table entry = slot), not by the header nick,
//! so duplicate names cannot cause a wrong match. Only ServerInfo messages and string tables are
//! handled (no entities), and parsing stops as soon as the slot's userinfo entry is known, which
//! is normally during signon.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result};
use bitbuffer::BitRead;
use serde::Serialize;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::{DemoTick, UserInfo};
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
use tf_demo_parser::demo::parser::{DemoHandler, MessageHandler, RawPacketStream};
use tf_demo_parser::{Demo, MessageType, ParserState};

#[derive(Debug, Clone, Serialize)]
pub struct RecorderInfo {
    /// Name from the recorder's userinfo entry (header nick when the entry is missing).
    pub name: String,
    /// SteamID64; `None` for SourceTV / bots or when the recorder's userinfo is missing.
    pub steam_id: Option<u64>,
}

#[derive(Default)]
struct RecorderFinder {
    nick: String,
    /// Entity index of the recording client (player_slot + 1).
    entity: Option<u32>,
    /// Latest userinfo per entity index seen so far (slot may arrive before or after ServerInfo).
    infos: Vec<(u32, String, String, bool)>,
    /// Set once the recorder's userinfo is known, so the caller can stop reading packets.
    done: Rc<Cell<bool>>,
}

impl RecorderFinder {
    fn found(&self) -> Option<&(u32, String, String, bool)> {
        let ent = self.entity?;
        self.infos.iter().find(|i| i.0 == ent)
    }
}

impl MessageHandler for RecorderFinder {
    type Output = RecorderInfo;

    fn does_handle(message_type: MessageType) -> bool {
        matches!(message_type, MessageType::ServerInfo)
    }

    fn handle_header(&mut self, header: &Header) {
        self.nick = header.nick.trim_matches('\0').trim().to_string();
    }

    fn handle_message(&mut self, message: &Message, _tick: DemoTick, _state: &ParserState) {
        if let Message::ServerInfo(info) = message {
            if self.entity.is_none() {
                self.entity = Some(info.player_slot as u32 + 1);
            }
            self.done.set(self.found().is_some());
        }
    }

    fn handle_string_entry(&mut self, table: &str, index: usize, entry: &StringTableEntry, _state: &ParserState) {
        if table != "userinfo" {
            return;
        }
        let text = entry.text.as_ref().map(|s| s.as_ref());
        let data = entry.extra_data.as_ref().map(|d| d.data.clone());
        if let Ok(Some(user_info)) = UserInfo::parse_from_string_table(index as u16, text, data) {
            let ent = u32::from(user_info.entity_id);
            let p = user_info.player_info;
            let is_bot = p.is_hl_tv != 0 || p.is_fake_player != 0 || p.is_replay != 0;
            self.infos.retain(|i| i.0 != ent);
            self.infos.push((ent, p.name, p.steam_id, is_bot));
            self.done.set(self.found().is_some());
        }
    }

    fn into_output(self, _state: &ParserState) -> RecorderInfo {
        match self.found() {
            Some((_, name, steam3, is_bot)) => RecorderInfo {
                name: name.clone(),
                steam_id: if *is_bot {
                    None
                } else {
                    SteamID::from_steam3(steam3).ok().map(u64::from)
                },
            },
            None => RecorderInfo { name: self.nick, steam_id: None },
        }
    }
}

/// Finds the demo recorder's name and SteamID64.
pub fn parse_recorder<P: AsRef<Path>>(path: P) -> Result<RecorderInfo> {
    let bytes = std::fs::read(path.as_ref())
        .with_context(|| format!("Failed to read demo file: {:?}", path.as_ref()))?;
    let demo = Demo::new(&bytes);
    let mut stream = demo.get_stream();
    let header = Header::read(&mut stream).with_context(|| "Failed to read demo header")?;
    let mut packets = RawPacketStream::new(stream);
    let done = Rc::new(Cell::new(false));
    let finder = RecorderFinder { done: done.clone(), ..Default::default() };
    let mut handler = DemoHandler::with_analyser(finder);
    handler.handle_header(&header);

    loop {
        match packets.next(&handler.state_handler) {
            Ok(Some(packet)) => {
                if handler.handle_packet(packet).is_err() {
                    break;
                }
            }
            // A malformed packet ends the stream; whatever was found so far is reported
            _ => break,
        }
        if done.get() {
            break;
        }
    }
    Ok(handler.into_output())
}
