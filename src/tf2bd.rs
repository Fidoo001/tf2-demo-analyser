//! TF2 Bot Detector (TF2BD) playerlist support (optional `tf2bd` feature).
//!
//! Reads and writes playerlists in the TF2BD format:
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/PazerOP/tf2_bot_detector/master/schemas/v3/playerlist.schema.json",
//!   "file_info": { "authors": [], "description": "", "title": "" },
//!   "players": [
//!     { "steamid": "[U:1:123]", "attributes": ["cheater"], "proof": ["..."],
//!       "last_seen": { "player_name": "name", "time": 1700000000 } }
//!   ]
//! }
//! ```
//!
//! The whole file is kept as JSON, so fields this program does not know about survive an edit.
//! The settings (playerlist sources, attribute order, colors and notifications) are stored in the
//! `tf2bd` section of params.json.

use std::collections::{HashMap, HashSet};
use std::io::Read;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use steamid_ng::SteamID;

use crate::lib::general_settings::{format_hex_rgb, parse_hex_rgb};
use crate::lib::parameters::{Parameter, Parameters};

pub const SCHEMA_URL: &str =
    "https://raw.githubusercontent.com/PazerOP/tf2_bot_detector/master/schemas/v3/playerlist.schema.json";
/// Largest remote playerlist that is downloaded.
const MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024;

/// Display settings of one attribute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributeSetting {
    pub name: String,
    /// Highlight color as `#RRGGBB`.
    pub color: String,
    /// Announce players with this attribute in party chat when they are on the server (live scanning).
    #[serde(default)]
    pub notify: bool,
}

impl AttributeSetting {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            color: format_hex_rgb(default_attribute_color(name)),
            notify: false,
        }
    }

    pub fn rgb(&self) -> [u8; 3] {
        parse_hex_rgb(&self.color).unwrap_or_else(|| default_attribute_color(&self.name))
    }
}

/// Default highlight color of an attribute.
pub fn default_attribute_color(name: &str) -> [u8; 3] {
    match name.to_ascii_lowercase().as_str() {
        "cheater" => [235, 70, 70],
        "suspicious" => [240, 200, 60],
        "exploiter" => [190, 110, 255],
        "racist" => [255, 140, 40],
        _ => [90, 170, 255],
    }
}

/// Public TF2BD playerlists offered in the playerlist settings, and used until the user picks
/// their own: (name, URL).
pub const SUGGESTED_PLAYERLISTS: [(&str, &str); 2] = [
    (
        "TF2BD Community Biglist (bots)",
        "https://gist.githubusercontent.com/wgetJane/0bc01bd46d7695362253c5a2fa49f2e9/raw/playerlist.biglist.json",
    ),
    ("TF2BD Trusted (cheaters)", "https://trusted.roto.lol/v1/steamids"),
];

/// The `tf2bd` section of params.json.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tf2bdSettings {
    pub enabled: bool,
    /// Local paths and URLs of the playerlists.
    pub playerlists: Vec<String>,
    /// Attributes in priority order: a player with several attributes is highlighted with the
    /// color of the first one in this list.
    pub attributes: Vec<AttributeSetting>,
}

impl Tf2bdSettings {
    pub const SECTION: &'static str = "tf2bd";
    pub const ENABLED_KEY: &'static str = "enabled";
    pub const PLAYERLISTS_KEY: &'static str = "playerlists";
    pub const ATTRIBUTES_KEY: &'static str = "attributes";

    /// Params.json only holds scalar values, so the lists are stored as JSON encoded strings.
    pub fn to_parameters(&self) -> Parameters {
        let mut map = HashMap::new();
        map.insert(Self::ENABLED_KEY.to_string(), Parameter::Bool(self.enabled));
        map.insert(
            Self::PLAYERLISTS_KEY.to_string(),
            Parameter::String(serde_json::to_string(&self.playerlists).unwrap_or_else(|_| "[]".to_string())),
        );
        map.insert(
            Self::ATTRIBUTES_KEY.to_string(),
            Parameter::String(serde_json::to_string(&self.attributes).unwrap_or_else(|_| "[]".to_string())),
        );
        map
    }

    pub fn from_parameters(params: &Parameters) -> Self {
        let json_list = |key: &str| match params.get(key) {
            Some(Parameter::String(s)) if !s.trim().is_empty() => serde_json::from_str::<Value>(s).ok(),
            _ => None,
        };
        let enabled = matches!(params.get(Self::ENABLED_KEY), Some(Parameter::Bool(true)));
        // Until playerlists are saved, the suggested community lists are used.
        let mut playerlists: Vec<String> = match params.get(Self::PLAYERLISTS_KEY) {
            None => SUGGESTED_PLAYERLISTS.iter().map(|(_, url)| url.to_string()).collect(),
            Some(_) => json_list(Self::PLAYERLISTS_KEY)
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default(),
        };
        playerlists.retain(|s| !s.trim().is_empty());
        let mut seen = HashSet::new();
        playerlists.retain(|s| seen.insert(s.clone()));

        let mut attributes: Vec<AttributeSetting> = json_list(Self::ATTRIBUTES_KEY)
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let mut seen = HashSet::new();
        attributes.retain(|a| !a.name.is_empty() && seen.insert(a.name.clone()));
        Self { enabled, playerlists, attributes }
    }

    pub fn from_config(config: &HashMap<String, Parameters>) -> Self {
        config
            .get(Self::SECTION)
            .map(Self::from_parameters)
            .unwrap_or_else(|| Self::from_parameters(&HashMap::new()))
    }

    pub fn attribute(&self, name: &str) -> Option<&AttributeSetting> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Position of the attribute in the priority order (unknown attributes come last).
    pub fn priority(&self, name: &str) -> usize {
        self.attributes.iter().position(|a| a.name == name).unwrap_or(usize::MAX)
    }

    /// Appends the attributes that are not configured yet to the bottom of the list.
    /// Returns whether any was added.
    pub fn add_attributes<'a>(&mut self, names: impl IntoIterator<Item = &'a str>) -> bool {
        let mut added = false;
        for name in names {
            if !name.is_empty() && self.attribute(name).is_none() {
                self.attributes.push(AttributeSetting::new(name));
                added = true;
            }
        }
        added
    }

    /// The attributes of a player sorted by priority, highest first.
    pub fn sort_by_priority(&self, attributes: &mut [String]) {
        attributes.sort_by_key(|a| self.priority(a));
    }
}

/// One player of a playerlist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlayerlistEntry {
    pub attributes: Vec<String>,
    pub proof: Vec<String>,
    pub last_seen_name: Option<String>,
}

/// Parses the `steamid` field of a player: SteamID3 (`[U:1:123]`), SteamID2 or a SteamID64
/// given as a string or a number.
pub fn parse_steam_id(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64().filter(|&id| id > 0),
        Value::String(s) => {
            let s = s.trim();
            if let Ok(id) = s.parse::<u64>() {
                return (id > 0).then_some(id);
            }
            SteamID::from_steam3(s)
                .or_else(|_| SteamID::from_steam2(s))
                .ok()
                .map(u64::from)
        }
        _ => None,
    }
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Null => None,
                other => Some(other.to_string()),
            })
            .collect(),
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn parse_entry(player: &Value) -> Option<(u64, PlayerlistEntry)> {
    let id = parse_steam_id(player.get("steamid")?)?;
    let last_seen_name = player
        .get("last_seen")
        .and_then(|l| l.get("player_name"))
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .map(str::to_string);
    Some((
        id,
        PlayerlistEntry {
            attributes: string_list(player.get("attributes")),
            proof: string_list(player.get("proof")),
            last_seen_name,
        },
    ))
}

/// A parsed TF2BD playerlist.
#[derive(Debug, Clone)]
pub struct Playerlist {
    json: Value,
    players: HashMap<u64, PlayerlistEntry>,
}

impl Playerlist {
    pub fn parse(text: &str) -> Result<Self, String> {
        let json: Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))
            .map_err(|e| format!("Invalid JSON: {e}"))?;
        if !json.is_object() {
            return Err("Not a TF2BD playerlist (the root is not an object)".to_string());
        }
        match json.get("players") {
            None | Some(Value::Array(_)) => {}
            Some(_) => return Err("Not a TF2BD playerlist (\"players\" is not an array)".to_string()),
        }
        let mut list = Self { json, players: HashMap::new() };
        list.reindex();
        Ok(list)
    }

    /// An empty playerlist with the given title.
    pub fn new_empty(title: &str) -> Self {
        let json = serde_json::json!({
            "$schema": SCHEMA_URL,
            "file_info": {
                "authors": [],
                "description": "",
                "title": title,
            },
            "players": [],
        });
        Self { json, players: HashMap::new() }
    }

    fn players_json(&self) -> &[Value] {
        self.json.get("players").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
    }

    fn players_json_mut(&mut self) -> &mut Vec<Value> {
        let root = self.json.as_object_mut().expect("playerlist root is an object");
        let players = root.entry("players").or_insert_with(|| Value::Array(Vec::new()));
        if !players.is_array() {
            *players = Value::Array(Vec::new());
        }
        players.as_array_mut().unwrap()
    }

    /// Rebuilds the lookup table. Duplicate entries of a player are merged.
    fn reindex(&mut self) {
        let mut players: HashMap<u64, PlayerlistEntry> = HashMap::new();
        for (id, entry) in self.players_json().iter().filter_map(parse_entry) {
            match players.get_mut(&id) {
                Some(existing) => {
                    for attr in entry.attributes {
                        if !existing.attributes.contains(&attr) {
                            existing.attributes.push(attr);
                        }
                    }
                    existing.proof.extend(entry.proof);
                    if existing.last_seen_name.is_none() {
                        existing.last_seen_name = entry.last_seen_name;
                    }
                }
                None => {
                    players.insert(id, entry);
                }
            }
        }
        self.players = players;
    }

    pub fn players(&self) -> &HashMap<u64, PlayerlistEntry> {
        &self.players
    }

    pub fn get(&self, steam_id: u64) -> Option<&PlayerlistEntry> {
        self.players.get(&steam_id)
    }

    pub fn title(&self) -> Option<&str> {
        self.json
            .get("file_info")
            .and_then(|f| f.get("title"))
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
    }

    /// Every attribute used in the list, in the order they first appear.
    pub fn attributes_in_order(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for player in self.players_json() {
            for attr in string_list(player.get("attributes")) {
                if seen.insert(attr.clone()) {
                    out.push(attr);
                }
            }
        }
        out
    }

    /// Adds or replaces the entry of a player. Other fields of an existing entry are kept and
    /// duplicate entries of the player are removed.
    pub fn set_player(&mut self, steam_id: u64, attributes: &[String], proof: &[String], name: Option<&str>) {
        let players = self.players_json_mut();
        let mut matching = players
            .iter()
            .enumerate()
            .filter(|(_, p)| p.get("steamid").and_then(parse_steam_id) == Some(steam_id))
            .map(|(i, _)| i);
        let index = matching.next();
        let duplicates: Vec<usize> = matching.collect();
        for i in duplicates.into_iter().rev() {
            players.remove(i);
        }
        let index = index.unwrap_or_else(|| {
            players.push(serde_json::json!({ "steamid": SteamID::from(steam_id).steam3() }));
            players.len() - 1
        });
        if !players[index].is_object() {
            players[index] = serde_json::json!({ "steamid": SteamID::from(steam_id).steam3() });
        }
        let entry = players[index].as_object_mut().unwrap();
        entry.insert("attributes".to_string(), serde_json::json!(attributes));
        if proof.is_empty() {
            entry.remove("proof");
        } else {
            entry.insert("proof".to_string(), serde_json::json!(proof));
        }
        if let Some(name) = name.filter(|n| !n.is_empty()) {
            let time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            entry.insert("last_seen".to_string(), serde_json::json!({ "player_name": name, "time": time }));
        }
        self.reindex();
    }

    /// Removes every entry of a player. Returns whether there was one.
    pub fn remove_player(&mut self, steam_id: u64) -> bool {
        let players = self.players_json_mut();
        let before = players.len();
        players.retain(|p| p.get("steamid").and_then(parse_steam_id) != Some(steam_id));
        let removed = players.len() != before;
        self.reindex();
        removed
    }

    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.json).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        std::fs::write(path, self.to_json_string()).map_err(|e| format!("Failed to write {}: {e}", path.display()))
    }
}

/// Whether a playerlist source is downloaded (an http(s) URL) instead of a local file.
pub fn is_remote_source(source: &str) -> bool {
    let lower = source.trim().to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Loads a playerlist from a local path or downloads it from a URL.
pub fn load_playerlist(source: &str) -> Result<Playerlist, String> {
    let source = source.trim();
    let text = if is_remote_source(source) {
        let resp = ureq::get(source)
            .timeout(std::time::Duration::from_secs(20))
            .call()
            .map_err(|e| format!("Download failed: {e}"))?;
        let mut text = String::new();
        resp.into_reader()
            .take(MAX_DOWNLOAD_BYTES)
            .read_to_string(&mut text)
            .map_err(|e| format!("Download failed: {e}"))?;
        text
    } else {
        std::fs::read_to_string(source).map_err(|e| format!("Failed to read {source}: {e}"))?
    };
    Playerlist::parse(&text)
}

/// Loading state of a configured playerlist.
#[derive(Debug, Clone)]
pub enum PlayerlistState {
    Loading,
    Loaded(Playerlist),
    Failed(String),
}

/// A configured playerlist source and its contents.
#[derive(Debug, Clone)]
pub struct LoadedPlayerlist {
    pub source: String,
    pub state: PlayerlistState,
}

impl LoadedPlayerlist {
    pub fn is_remote(&self) -> bool {
        is_remote_source(&self.source)
    }

    pub fn list(&self) -> Option<&Playerlist> {
        match &self.state {
            PlayerlistState::Loaded(list) => Some(list),
            _ => None,
        }
    }

    /// Display name: the file name of a local list, the URL of a remote one.
    pub fn display_name(&self) -> String {
        if self.is_remote() {
            return self.source.clone();
        }
        std::path::Path::new(&self.source)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.source.clone())
    }
}

/// Attributes of a player over all loaded lists, without duplicates, sorted by priority.
pub fn player_attributes(lists: &[LoadedPlayerlist], settings: &Tf2bdSettings, steam_id: u64) -> Vec<String> {
    let mut attributes: Vec<String> = Vec::new();
    for entry in lists.iter().filter_map(|l| l.list()?.get(steam_id)) {
        for attr in &entry.attributes {
            if !attributes.contains(attr) {
                attributes.push(attr.clone());
            }
        }
    }
    settings.sort_by_priority(&mut attributes);
    attributes
}

/// Every attribute used in the loaded lists, in the order they first appear.
pub fn found_attributes(lists: &[LoadedPlayerlist]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for list in lists.iter().filter_map(LoadedPlayerlist::list) {
        for attr in list.attributes_in_order() {
            if seen.insert(attr.clone()) {
                out.push(attr);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_community_lists_are_used_until_playerlists_are_saved() {
        let fresh = Tf2bdSettings::from_config(&HashMap::new());
        let suggested: Vec<String> = SUGGESTED_PLAYERLISTS.iter().map(|(_, url)| url.to_string()).collect();
        assert_eq!(fresh.playerlists, suggested);
        assert!(!fresh.enabled, "TF2BD itself stays off until turned on");

        // Once the user saves their own choice, even an empty one, it's kept.
        let saved = Tf2bdSettings { playerlists: Vec::new(), ..fresh };
        assert!(Tf2bdSettings::from_parameters(&saved.to_parameters()).playerlists.is_empty());
    }

    const LIST: &str = r#"{
        "$schema": "x",
        "file_info": { "title": "Test", "authors": ["a"] },
        "players": [
            { "steamid": "[U:1:100]", "attributes": ["cheater"], "proof": ["clip"], "custom": 1 },
            { "steamid": 76561197960265829, "attributes": ["suspicious", "racist"] },
            { "steamid": "76561197960265830", "attributes": ["exploiter"] },
            { "steamid": "[U:1:100]", "attributes": ["exploiter"] }
        ]
    }"#;

    #[test]
    fn parses_steam_id_formats_and_merges_duplicates() {
        let list = Playerlist::parse(LIST).unwrap();
        assert_eq!(list.players().len(), 3);
        let id = 76561197960265728 + 100;
        let entry = list.get(id).unwrap();
        assert_eq!(entry.attributes, vec!["cheater", "exploiter"]);
        assert_eq!(entry.proof, vec!["clip"]);
        assert!(list.get(76561197960265829).is_some());
        assert!(list.get(76561197960265830).is_some());
        assert_eq!(list.attributes_in_order(), vec!["cheater", "suspicious", "racist", "exploiter"]);
        assert_eq!(list.title(), Some("Test"));
    }

    #[test]
    fn set_and_remove_player_keep_other_fields() {
        let mut list = Playerlist::parse(LIST).unwrap();
        let id = 76561197960265728 + 100;
        list.set_player(id, &["suspicious".to_string()], &[], Some("name"));
        let entry = list.get(id).unwrap();
        assert_eq!(entry.attributes, vec!["suspicious"]);
        assert!(entry.proof.is_empty());
        assert_eq!(entry.last_seen_name.as_deref(), Some("name"));
        // The duplicate was dropped and the unknown field kept
        let json: Value = serde_json::from_str(&list.to_json_string()).unwrap();
        let players = json["players"].as_array().unwrap();
        assert_eq!(players.len(), 3);
        assert_eq!(players[0]["custom"], 1);

        let new_id = 76561197960265728 + 555;
        list.set_player(new_id, &["cheater".to_string()], &["note".to_string()], None);
        let json: Value = serde_json::from_str(&list.to_json_string()).unwrap();
        assert_eq!(json["players"][3]["steamid"], "[U:1:555]");

        assert!(list.remove_player(id));
        assert!(!list.remove_player(id));
        assert!(list.get(id).is_none());
        assert_eq!(list.players().len(), 3);
    }

    #[test]
    fn settings_roundtrip_and_priority() {
        let mut settings = Tf2bdSettings {
            enabled: true,
            playerlists: vec!["a.json".to_string(), "https://x/y.json".to_string()],
            attributes: Vec::new(),
        };
        assert!(settings.add_attributes(["suspicious", "cheater"]));
        assert!(!settings.add_attributes(["cheater"]));
        let restored = Tf2bdSettings::from_parameters(&settings.to_parameters());
        assert_eq!(restored, settings);

        let mut attrs = vec!["cheater".to_string(), "other".to_string(), "suspicious".to_string()];
        settings.sort_by_priority(&mut attrs);
        assert_eq!(attrs, vec!["suspicious", "cheater", "other"]);
    }
}
