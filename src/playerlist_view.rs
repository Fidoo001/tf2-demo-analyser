use std::collections::{BTreeMap, HashMap};
#[cfg(feature = "gui")]
use std::collections::HashSet;
use std::path::Path;
use anyhow::{Context, Result};
use bitbuffer::BitRead;

#[cfg(feature = "gui")]
use eframe::egui;
use serde::{Deserialize, Serialize};
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::packetentities::PacketEntity;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::datatable::{ParseSendTable, ServerClass, ServerClassName};
use tf_demo_parser::demo::packet::message::MessagePacketMeta;
use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
use tf_demo_parser::demo::parser::analyser::{Class, Team, UserId};
use tf_demo_parser::demo::parser::handler::BorrowMessageHandler;
use tf_demo_parser::demo::parser::{MessageHandler, RawPacketStream};
use tf_demo_parser::demo::message::packetentities::EntityId;
use tf_demo_parser::{Demo, MessageType, ParserState, ReadResult, Stream};

use crate::base::demo_handler_base::CheatDemoHandler;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EventKind {
    Connect { name: String, steam_id: String, address: String },
    Disconnect { reason: String },
    TeamChange { old_team: Team, new_team: Team },
    ClassChange { new_class: Class },
    NameChange { old_name: String, new_name: String },
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventKind::Connect { name, steam_id, address } => {
                if !address.is_empty() {
                    write!(f, "Connected: {} ({}) [{}]", name, steam_id, address)
                } else {
                    write!(f, "Connected: {} ({})", name, steam_id)
                }
            }
            EventKind::Disconnect { reason } => {
                if reason.is_empty() {
                    write!(f, "Disconnected")
                } else {
                    write!(f, "Disconnected: {}", reason)
                }
            }
            EventKind::TeamChange { old_team, new_team } => {
                write!(f, "Team changed: {} -> {}", format_team(*old_team), format_team(*new_team))
            }
            EventKind::ClassChange { new_class } => {
                write!(f, "Class changed: {}", format_class(*new_class))
            }
            EventKind::NameChange { old_name, new_name } => {
                write!(f, "Name changed: {} -> {}", old_name, new_name)
            }
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlayerEvent {
    pub tick: u32,
    pub kind: EventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DemoPlayer {
    pub user_id: UserId,
    pub steam_id: u64,
    pub steam3: String,
    pub name: String,
    /// Last known team (`Other` once the player has disconnected).
    pub team: Team,
    /// First RED/BLU team the player joined in this demo (Spectator if they never joined a
    /// team, `Other` if unknown). Used for the quick overview, since most players swap later.
    #[serde(default = "default_first_team")]
    pub first_team: Team,
    pub class: Class,
    pub entity_id: Option<EntityId>,
    pub events: Vec<PlayerEvent>,
    pub kills: u32,
    pub deaths: u32,
}

fn default_first_team() -> Team {
    Team::Other
}

impl DemoPlayer {
    /// Records `team` as the first joined team. Red/Blue always beats Spectator/Other, and the
    /// first Red/Blue seen is kept for the rest of the demo.
    fn note_first_team(&mut self, team: Team) {
        let is_playing = |t: Team| matches!(t, Team::Red | Team::Blue);
        if is_playing(self.first_team) {
            return;
        }
        if is_playing(team) || (team == Team::Spectator && self.first_team == Team::Other) {
            self.first_team = team;
        }
    }
}

pub fn format_team(team: Team) -> &'static str {
    match team {
        Team::Red => "RED",
        Team::Blue => "BLU",
        Team::Spectator => "Spectator",
        Team::Other => "Other",
    }
}

pub fn format_class(class: Class) -> &'static str {
    match class {
        Class::Scout => "Scout",
        Class::Sniper => "Sniper",
        Class::Soldier => "Soldier",
        Class::Demoman => "Demoman",
        Class::Medic => "Medic",
        Class::Heavy => "Heavy",
        Class::Pyro => "Pyro",
        Class::Spy => "Spy",
        Class::Engineer => "Engineer",
        Class::Other => "Unknown",
    }
}

#[derive(Default)]
pub struct PlayerListParser {
    pub players_by_id64: BTreeMap<u64, DemoPlayer>,
    pub userid_to_id64: HashMap<UserId, u64>,
    pub entid_to_userid: HashMap<EntityId, UserId>,
    pub class_names: Vec<ServerClassName>,
    pub current_tick: u32,
    pub unassigned_kd: HashMap<UserId, (u32, u32)>,
}

impl PlayerListParser {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_or_create_player(&mut self, steam_id: u64, user_id: UserId, name: &str, steam3: &str) -> &mut DemoPlayer {
        self.userid_to_id64.insert(user_id, steam_id);
        let unassigned = self.unassigned_kd.remove(&user_id);
        let tick = self.current_tick;
        let player = self.players_by_id64.entry(steam_id).or_insert_with(|| DemoPlayer {
            user_id,
            steam_id,
            steam3: steam3.to_string(),
            name: name.to_string(),
            team: Team::Other,
            first_team: Team::Other,
            class: Class::Other,
            entity_id: None,
            events: Vec::new(),
            kills: 0,
            deaths: 0,
        });
        if player.user_id != user_id {
            // Same account on a new connection (new userid): forget the previous session's team/class so
            // the first assignment of this session is not reported as a change from the old one.
            player.user_id = user_id;
            player.team = Team::Other;
            player.class = Class::Other;
        }
        if !name.is_empty() && player.name != name {
            let old_name = std::mem::replace(&mut player.name, name.to_string());
            player.events.push(PlayerEvent {
                tick,
                kind: EventKind::NameChange { old_name, new_name: name.to_string() },
            });
        }
        if let Some((k, d)) = unassigned {
            player.kills += k;
            player.deaths += d;
        }
        player
    }

    fn record_event(&mut self, steam_id: u64, kind: EventKind) {
        if let Some(player) = self.players_by_id64.get_mut(&steam_id) {
            // Avoid duplicate consecutive identical events at same tick
            if let Some(last) = player.events.last() {
                if last.tick == self.current_tick && last.kind == kind {
                    return;
                }
            }
            player.events.push(PlayerEvent {
                tick: self.current_tick,
                kind,
            });
        }
    }

    fn parse_user_info(
        &mut self,
        index: usize,
        text: Option<&str>,
        data: Option<Stream>,
    ) -> ReadResult<()> {
        let data_present = data.is_some();
        if let Some(user_info) =
            tf_demo_parser::demo::data::UserInfo::parse_from_string_table(index as u16, text, data)?
        {
            let pinfo = &user_info.player_info;
            let user_id = pinfo.user_id;
            let ent_id = user_info.entity_id;
            self.entid_to_userid.insert(ent_id, user_id);

            if let Ok(steam_id) = SteamID::from_steam3(&pinfo.steam_id) {
                let id64: u64 = steam_id.into();
                let player = self.get_or_create_player(id64, user_id, &pinfo.name, &pinfo.steam_id);
                player.entity_id = Some(ent_id);
                if !pinfo.steam_id.is_empty() {
                    player.steam3 = pinfo.steam_id.clone();
                }
            }
        } else if data_present {
            // The slot was emptied (player left): its entity must no longer resolve to the old player,
            // otherwise the next occupant's m_iTeam/m_iPlayerClass updates get attributed to them.
            let ent_id: EntityId = match text.map(|t| t.parse::<u32>().map(|id| id + 1)) {
                Some(Ok(id)) => id.into(),
                _ => (index as u32 + 1).into(),
            };
            self.entid_to_userid.remove(&ent_id);
        }
        Ok(())
    }

    fn handle_player_resource(&mut self, entity: &PacketEntity, parser_state: &ParserState) {
        for prop in entity.props(parser_state) {
            if let Some((table_name, prop_name)) = prop.identifier.names() {
                if let Ok(player_id) = prop_name.as_str().parse::<u32>() {
                    let entity_id = EntityId::from(player_id);
                    let user_id_opt = self.entid_to_userid.get(&entity_id).copied();
                    let steam_id_opt = user_id_opt.and_then(|uid| self.userid_to_id64.get(&uid).copied());

                    if let Some(steam_id) = steam_id_opt {
                        if let Some(player) = self.players_by_id64.get_mut(&steam_id) {
                            match table_name.as_str() {
                                "m_iTeam" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        let new_team = Team::new(val);
                                        player.note_first_team(new_team);
                                        if new_team != player.team && new_team != Team::Other {
                                            let old_team = player.team;
                                            player.team = new_team;
                                            if old_team != Team::Other {
                                                player.events.push(PlayerEvent {
                                                    tick: self.current_tick,
                                                    kind: EventKind::TeamChange { old_team, new_team },
                                                });
                                            }
                                        }
                                    }
                                }
                                "m_iPlayerClass" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        let new_class = Class::new(val);
                                        if new_class != player.class && new_class != Class::Other {
                                            player.class = new_class;
                                            player.events.push(PlayerEvent {
                                                tick: self.current_tick,
                                                kind: EventKind::ClassChange { new_class },
                                            });
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }
}

impl MessageHandler for PlayerListParser {
    type Output = Vec<DemoPlayer>;

    fn does_handle(message_type: MessageType) -> bool {
        matches!(
            message_type,
            MessageType::GameEvent | MessageType::PacketEntities | MessageType::NetTick
        )
    }

    fn handle_message(&mut self, message: &Message, tick: DemoTick, parser_state: &ParserState) {
        self.current_tick = tick.into();
        match message {
            Message::PacketEntities(msg) => {
                for entity in &msg.entities {
                    let class_name = self
                        .class_names
                        .get(usize::from(entity.server_class))
                        .map(|c| c.as_str())
                        .unwrap_or("");

                    if class_name == "CTFPlayerResource" {
                        self.handle_player_resource(entity, parser_state);
                    }
                }
            }
            Message::GameEvent(ge) => match &ge.event {
                GameEvent::PlayerConnect(e) => {
                    let net_id_str = e.network_id.to_string();
                    let name_str = e.name.to_string();
                    if net_id_str != "BOT" {
                        let id64 = SteamID::from_steam3(net_id_str.as_str())
                            .map(u64::from)
                            .unwrap_or(0);
                        if id64 > 0 {
                            let uid = UserId::from(e.user_id);
                            self.get_or_create_player(id64, uid, &name_str, &net_id_str);
                            self.record_event(
                                id64,
                                EventKind::Connect {
                                    name: name_str,
                                    steam_id: net_id_str,
                                    address: e.address.to_string(),
                                },
                            );
                        }
                    }
                }
                GameEvent::PlayerConnectClient(e) => {
                    let net_id_str = e.network_id.to_string();
                    let name_str = e.name.to_string();
                    if net_id_str != "BOT" {
                        let id64 = SteamID::from_steam3(net_id_str.as_str())
                            .map(u64::from)
                            .unwrap_or(0);
                        if id64 > 0 {
                            let uid = UserId::from(e.user_id);
                            self.entid_to_userid.insert(EntityId::from(e.index as u32), uid);
                            self.get_or_create_player(id64, uid, &name_str, &net_id_str);
                            self.record_event(
                                id64,
                                EventKind::Connect {
                                    name: name_str,
                                    steam_id: net_id_str,
                                    address: String::new(),
                                },
                            );
                        }
                    }
                }
                GameEvent::PlayerDisconnect(e) => {
                    let uid = UserId::from(e.user_id);
                    if let Some(&id64) = self.userid_to_id64.get(&uid) {
                        self.record_event(
                            id64,
                            EventKind::Disconnect {
                                reason: e.reason.to_string(),
                            },
                        );
                        self.entid_to_userid.retain(|_, u| *u != uid);
                        if let Some(player) = self.players_by_id64.get_mut(&id64) {
                            if player.user_id == uid {
                                player.team = Team::Other;
                                player.class = Class::Other;
                            }
                        }
                    }
                }
                GameEvent::PlayerTeam(e) => {
                    let uid = UserId::from(e.user_id);
                    let old_team = Team::new(e.old_team as i64);
                    let new_team = Team::new(e.team as i64);
                    if let Some(&id64) = self.userid_to_id64.get(&uid) {
                        if let Some(player) = self.players_by_id64.get_mut(&id64) {
                            player.team = new_team;
                            player.note_first_team(new_team);
                        }
                        self.record_event(id64, EventKind::TeamChange { old_team, new_team });
                    }
                }
                GameEvent::PlayerChangeClass(e) => {
                    let uid = UserId::from(e.user_id);
                    let new_class = Class::new(e.class as i64);
                    if let Some(&id64) = self.userid_to_id64.get(&uid) {
                        if let Some(player) = self.players_by_id64.get_mut(&id64) {
                            player.class = new_class;
                        }
                        self.record_event(id64, EventKind::ClassChange { new_class });
                    }
                }
                GameEvent::PlayerSpawn(e) => {
                    let uid = UserId::from(e.user_id);
                    let team = Team::new(e.team as i64);
                    let class = Class::new(e.class as i64);
                    if let Some(&id64) = self.userid_to_id64.get(&uid) {
                        if let Some(player) = self.players_by_id64.get_mut(&id64) {
                            if team != Team::Other {
                                player.team = team;
                                player.note_first_team(team);
                            }
                            if class != Class::Other {
                                player.class = class;
                            }
                        }
                    }
                }
                GameEvent::PlayerDeath(e) => {
                    const TF_DEATHFLAG_DEADRINGER: u16 = 0x0020;
                    if (e.death_flags & TF_DEATHFLAG_DEADRINGER) == 0 {
                        let victim_uid = UserId::from(e.user_id);
                        if let Some(&id64) = self.userid_to_id64.get(&victim_uid) {
                            if let Some(victim) = self.players_by_id64.get_mut(&id64) {
                                victim.deaths += 1;
                            }
                        } else {
                            self.unassigned_kd.entry(victim_uid).or_default().1 += 1;
                        }

                        if e.attacker != 0 && e.attacker != e.user_id {
                            let attacker_uid = UserId::from(e.attacker);
                            if let Some(&id64) = self.userid_to_id64.get(&attacker_uid) {
                                if let Some(attacker) = self.players_by_id64.get_mut(&id64) {
                                    attacker.kills += 1;
                                }
                            } else {
                                self.unassigned_kd.entry(attacker_uid).or_default().0 += 1;
                            }
                        }
                    }
                }

                _ => {}
            },
            _ => {}
        }
    }


    fn handle_string_entry(
        &mut self,
        table: &str,
        index: usize,
        entry: &StringTableEntry,
        _parser_state: &ParserState,
    ) {
        if table == "userinfo" {
            let _ = self.parse_user_info(
                index,
                entry.text.as_ref().map(|s| s.as_ref()),
                entry.extra_data.as_ref().map(|data| data.data.clone()),
            );
        }
    }

    fn handle_data_tables(
        &mut self,
        _parse_tables: &[ParseSendTable],
        server_classes: &[ServerClass],
        _parser_state: &ParserState,
    ) {
        self.class_names = server_classes.iter().map(|c| c.name.clone()).collect();
    }

    fn handle_packet_meta(
        &mut self,
        tick: DemoTick,
        _meta: &MessagePacketMeta,
        _parser_state: &ParserState,
    ) {
        self.current_tick = tick.into();
    }

    fn into_output(mut self, _state: &ParserState) -> Self::Output {
        for (uid, (k, d)) in self.unassigned_kd.drain() {
            if let Some(&id64) = self.userid_to_id64.get(&uid) {
                if let Some(player) = self.players_by_id64.get_mut(&id64) {
                    player.kills += k;
                    player.deaths += d;
                }
            }
        }
        self.players_by_id64.into_values().collect()
    }
}

impl BorrowMessageHandler for PlayerListParser {
    fn borrow_output(&self, _state: &ParserState) -> &Self::Output {
        // Output not stored as vec directly in self, into_output produces it
        static EMPTY: Vec<DemoPlayer> = Vec::new();
        &EMPTY
    }
}

impl crate::base::demo_handler_base::DemoHandlerAnalyser for PlayerListParser {}

pub fn parse_playerlist<P: AsRef<Path>>(path: P) -> Result<Vec<DemoPlayer>> {
    let file_bytes = std::fs::read(path.as_ref())
        .with_context(|| format!("Failed to read demo file: {:?}", path.as_ref()))?;

    let demo = Demo::new(&file_bytes);
    let mut stream = demo.get_stream();
    let header: Header = Header::read(&mut stream)
        .with_context(|| "Failed to read demo header")?;
    let mut packets = RawPacketStream::new(stream);

    let parser = PlayerListParser::new();
    let mut handler = CheatDemoHandler::with_analyser(parser);
    handler.handle_header(&header);

    loop {
        let packet = match packets.next(&handler.state_handler) {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(_) => continue,
        };
        handler.handle_packet(packet)?;
    }

    let mut players = handler.analyser.into_output(&handler.state_handler);
    // Sort players by name
    players.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    Ok(players)
}

/// Highlight of a player tagged in a TF2BD playerlist.
#[cfg(feature = "gui")]
#[derive(Debug, Clone)]
pub struct TagHighlight {
    pub color: egui::Color32,
    /// The player's attributes, highest priority first.
    pub label: String,
}

/// Which players to highlight in the playerlist, and with which colors.
#[cfg(feature = "gui")]
pub struct PlayerHighlights<'a> {
    pub reported_ids: &'a HashSet<u64>,
    /// Players whose Steam account no longer exists.
    pub deleted_ids: &'a HashSet<u64>,
    /// Players whose Steam account was checked and verified to exist.
    pub existing_ids: &'a HashSet<u64>,
    /// Color of players that are neither marked nor deleted; `None` uses the theme's text color.
    pub normal_color: Option<egui::Color32>,
    pub marked_color: egui::Color32,
    pub deleted_color: egui::Color32,
    /// Players tagged in TF2BD playerlists; their tag color takes precedence over the others.
    pub tags: &'a HashMap<u64, TagHighlight>,
    /// Whether players can be marked in a TF2BD playerlist (adds "Mark player" to the right click menu).
    pub can_mark: bool,
}

#[cfg(feature = "gui")]
pub struct PlayerlistWindow {
    pub players: Vec<DemoPlayer>,
    pub is_loading: bool,
    pub is_open: bool,
    pub error: Option<String>,
    pub expanded_player: Option<u64>,
    pub filter_query: String,
    pub current_file: Option<std::path::PathBuf>,
    /// (steam id, name) of a player the user chose to mark in a TF2BD playerlist.
    pub mark_request: Option<(u64, String)>,
    recv: Option<std::sync::mpsc::Receiver<Result<Vec<DemoPlayer>>>>,
}

#[cfg(feature = "gui")]
impl PlayerlistWindow {
    pub fn new() -> Self {
        Self {
            players: Vec::new(),
            is_loading: false,
            is_open: false,
            error: None,
            expanded_player: None,
            filter_query: String::new(),
            current_file: None,
            mark_request: None,
            recv: None,
        }
    }

    pub fn open_for_file<P: AsRef<Path>>(&mut self, path: P) {
        self.is_open = true;
        self.expanded_player = None;

        if self.current_file.as_deref() == Some(path.as_ref()) && !self.players.is_empty() {
            return;
        }

        self.current_file = Some(path.as_ref().to_path_buf());
        self.is_loading = true;
        self.players.clear();
        self.error = None;

        let path_buf = path.as_ref().to_path_buf();
        let (send, recv) = std::sync::mpsc::channel();
        self.recv = Some(recv);

        std::thread::spawn(move || {
            let result = parse_playerlist(&path_buf);
            let _ = send.send(result);
        });
    }

    pub fn open_for_player<P: AsRef<Path>>(&mut self, path: P, steam_id: u64) {
        self.is_open = true;
        self.expanded_player = Some(steam_id);

        if self.current_file.as_deref() == Some(path.as_ref()) && !self.players.is_empty() {
            return;
        }

        self.current_file = Some(path.as_ref().to_path_buf());
        self.is_loading = true;
        self.players.clear();
        self.error = None;

        let path_buf = path.as_ref().to_path_buf();
        let (send, recv) = std::sync::mpsc::channel();
        self.recv = Some(recv);

        std::thread::spawn(move || {
            let result = parse_playerlist(&path_buf);
            let _ = send.send(result);
        });
    }

    pub fn ui(&mut self, ctx: &egui::Context, highlights: &PlayerHighlights) {
        if !self.is_open {
            return;
        }

        if let Some(recv) = &self.recv {
            if let Ok(res) = recv.try_recv() {
                self.is_loading = false;
                match res {
                    Ok(players) => {
                        self.players = players;
                        self.error = None;
                    }
                    Err(e) => {
                        self.error = Some(format!("{:#}", e));
                    }
                }
                self.recv = None;
            }
        }

        let mut is_open = self.is_open;
        egui::Window::new("Demo Playerlist")
            .open(&mut is_open)
            .collapsible(false)
            .resizable(false)
            .default_width(540.0)
            .min_width(500.0)
            .max_width(540.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(235, 70, 70), format!("Error: {}", err));
                    return;
                }

                if self.is_loading {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Parsing demo players...");
                    });
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Filter:");
                    let prev_bg = ui.visuals().extreme_bg_color;
                    ui.visuals_mut().extreme_bg_color = ui.visuals().widgets.inactive.bg_fill;
                    ui.text_edit_singleline(&mut self.filter_query);
                    ui.visuals_mut().extreme_bg_color = prev_bg;

                    if !self.filter_query.is_empty() && ui.small_button("Clear").clicked() {
                        self.filter_query.clear();
                    }
                    ui.add_space(20.0);
                    ui.label(format!("Total players: {}", self.players.len()));
                });
                ui.separator();

                let red_players: Vec<&DemoPlayer> = self
                    .players
                    .iter()
                    .filter(|p| p.first_team == Team::Red && self.matches_filter(p))
                    .collect();
                let blu_players: Vec<&DemoPlayer> = self
                    .players
                    .iter()
                    .filter(|p| p.first_team == Team::Blue && self.matches_filter(p))
                    .collect();
                let other_players: Vec<&DemoPlayer> = self
                    .players
                    .iter()
                    .filter(|p| !matches!(p.first_team, Team::Red | Team::Blue) && self.matches_filter(p))
                    .collect();

                let mut player_to_toggle_expand = None;
                let mut mark_request = None;

                let max_scroll_h = (ui.ctx().screen_rect().height() - 150.0).max(200.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false, true])
                    .max_height(max_scroll_h)
                    .show(ui, |ui| {

                        ui.columns(3, |columns| {
                            // Column 0: RED
                            columns[0].vertical(|ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(255, 100, 100),
                                    format!("RED ({})", red_players.len()),
                                );
                                ui.separator();
                                for p in &red_players {
                                    self.render_player_item(
                                        ui,
                                        p,
                                        highlights,
                                        &mut player_to_toggle_expand,
                                        &mut mark_request,
                                    );
                                }
                            });

                            // Column 1: BLU
                            columns[1].vertical(|ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(100, 160, 255),
                                    format!("BLU ({})", blu_players.len()),
                                );
                                ui.separator();
                                for p in &blu_players {
                                    self.render_player_item(
                                        ui,
                                        p,
                                        highlights,
                                        &mut player_to_toggle_expand,
                                        &mut mark_request,
                                    );
                                }
                            });

                            // Column 2: OTHER
                            columns[2].vertical(|ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(180, 180, 180),
                                    format!("OTHER ({})", other_players.len()),
                                );
                                ui.separator();
                                for p in &other_players {
                                    self.render_player_item(
                                        ui,
                                        p,
                                        highlights,
                                        &mut player_to_toggle_expand,
                                        &mut mark_request,
                                    );
                                }
                            });
                        });
                    });

                if let Some(id) = player_to_toggle_expand {
                    if self.expanded_player == Some(id) {
                        self.expanded_player = None;
                    } else {
                        self.expanded_player = Some(id);
                    }
                }
                if mark_request.is_some() {
                    self.mark_request = mark_request;
                }

                ui.separator();
                ui.label("Double-click a player to inspect events, right-click for more options");
            });

        // Expanded player events sub-window
        let mut mark_request = None;
        if let Some(expanded_id) = self.expanded_player {
            if let Some(player) = self.players.iter().find(|p| p.steam_id == expanded_id) {
                let mut modal_open = true;
                egui::Window::new(format!("Player Events: {}", player.name))
                    .open(&mut modal_open)
                    .collapsible(false)
                    .resizable(false)
                    .default_size([480.0, 385.0])
                    .min_size([480.0, 385.0])
                    .max_size([480.0, 385.0])
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {

                        ui.horizontal(|ui| {
                            ui.strong("Name:");
                            ui.label(&player.name);
                            ui.add_space(10.0);
                            ui.strong("SteamID:");
                            ui.label(format!("{}", player.steam_id));
                        });
                        ui.horizontal(|ui| {
                            ui.strong("First Team:");
                            ui.label(format_team(player.first_team));
                            ui.add_space(10.0);
                            ui.strong("Last Team:");
                            ui.label(format_team(player.team));
                            ui.add_space(10.0);
                            ui.strong("Current Class:");
                            ui.label(format_class(player.class));
                        });
                        ui.horizontal(|ui| {
                            ui.strong("K/D:");
                            let ratio_str = if player.deaths == 0 {
                                if player.kills == 0 {
                                    "0.00".to_string()
                                } else {
                                    format!("{:.2}", player.kills as f32)
                                }
                            } else {
                                format!("{:.2}", player.kills as f32 / player.deaths as f32)
                            };
                            ui.label(format!("{} / {} ({})", player.kills, player.deaths, ratio_str));
                            ui.add_space(8.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(160, 160, 160),
                                format!("({} kills, {} deaths)", player.kills, player.deaths),
                            );
                        });
                        ui.separator();
                        ui.heading("Timeline of Events");

                        if player.events.is_empty() {
                            ui.label("No recorded events for this player.");
                        } else {
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                egui::Grid::new("player_events_grid")
                                    .striped(true)
                                    .num_columns(2)
                                    .spacing([15.0, 4.0])
                                    .show(ui, |ui| {
                                        ui.strong("Tick");
                                        ui.strong("Event");
                                        ui.end_row();

                                        for ev in &player.events {
                                            ui.label(format!("{}", ev.tick));
                                            ui.label(format!("{}", ev.kind));
                                            ui.end_row();
                                        }
                                    });
                            });
                        }

                        ui.add_space(8.0);
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("Copy SteamID").clicked() {
                                ui.ctx().copy_text(format!("{}", player.steam_id));
                            }
                            if ui.button("Copy Ingame Name").clicked() {
                                ui.ctx().copy_text(player.name.clone());
                            }
                            if ui.button("Open Steam Profile").clicked() {
                                let _ = opener::open_browser(format!(
                                    "https://steamcommunity.com/profiles/{}",
                                    player.steam_id
                                ));
                            }
                            if ui.button("Open SteamHistory").clicked() {
                                let _ = opener::open_browser(format!(
                                    "https://steamhistory.net/id/{}",
                                    player.steam_id
                                ));
                            }
                            if highlights.can_mark && ui.button("Mark player").clicked() {
                                mark_request = Some((player.steam_id, player.name.clone()));
                            }
                        });
                    });
                if !modal_open {
                    self.expanded_player = None;
                }
            }
        }
        if mark_request.is_some() {
            self.mark_request = mark_request;
        }

        self.is_open = is_open;
    }

    fn matches_filter(&self, p: &DemoPlayer) -> bool {
        if self.filter_query.is_empty() {
            return true;
        }
        let q = self.filter_query.to_lowercase();
        p.name.to_lowercase().contains(&q)
            || p.steam_id.to_string().contains(&q)
            || p.steam3.to_lowercase().contains(&q)
    }

    fn render_player_item(
        &self,
        ui: &mut egui::Ui,
        player: &DemoPlayer,
        highlights: &PlayerHighlights,
        player_to_toggle_expand: &mut Option<u64>,
        mark_request: &mut Option<(u64, String)>,
    ) {
        let tag = highlights.tags.get(&player.steam_id);
        let text_widget = if let Some(tag) = tag {
            egui::RichText::new(&player.name).color(tag.color)
        } else if highlights.reported_ids.contains(&player.steam_id) {
            egui::RichText::new(&player.name).color(highlights.marked_color)
        } else if highlights.deleted_ids.contains(&player.steam_id) {
            egui::RichText::new(&player.name).color(highlights.deleted_color)
        } else if highlights.existing_ids.contains(&player.steam_id) {
            if let Some(color) = highlights.normal_color {
                egui::RichText::new(&player.name).color(color)
            } else {
                egui::RichText::new(&player.name)
            }
        } else {
            egui::RichText::new(&player.name)
        };

        let is_selected = self.expanded_player == Some(player.steam_id);
        let selectable = egui::SelectableLabel::new(is_selected, text_widget);
        let mut response = ui.add_sized([ui.available_width(), 0.0], selectable);
        if let Some(tag) = tag {
            response = response.on_hover_text(format!("TF2BD: {}", tag.label));
        }

        if is_selected {
            response.scroll_to_me(Some(egui::Align::Center));
        }

        // Opening only: the list's toggle must not close an already open events window
        let already_open = self.expanded_player == Some(player.steam_id);
        if response.double_clicked() && !already_open {
            *player_to_toggle_expand = Some(player.steam_id);
        }

        response.context_menu(|ui| {
            if ui.button("View player events").clicked() {
                if !already_open {
                    *player_to_toggle_expand = Some(player.steam_id);
                }
                ui.close_menu();
            }
            if ui.button("Copy SteamHistory link").clicked() {
                ui.ctx().copy_text(format!("https://steamhistory.net/id/{}", player.steam_id));
                ui.close_menu();
            }
            if ui.button("Copy Steam profile link").clicked() {
                ui.ctx().copy_text(format!("https://steamcommunity.com/profiles/{}", player.steam_id));
                ui.close_menu();
            }
            if ui.button("Copy steamID").clicked() {
                ui.ctx().copy_text(format!("{}", player.steam_id));
                ui.close_menu();
            }
            if ui.button("Copy ingame name").clicked() {
                ui.ctx().copy_text(player.name.clone());
                ui.close_menu();
            }
            if highlights.can_mark && ui.button("Mark player").clicked() {
                *mark_request = Some((player.steam_id, player.name.clone()));
                ui.close_menu();
            }
        });
    }
}

