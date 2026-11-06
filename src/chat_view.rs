use std::collections::HashMap;
use std::path::Path;
use anyhow::{Context, Result};
use bitbuffer::BitRead;
#[cfg(feature = "gui")]
use eframe::egui;
use serde::{Deserialize, Serialize};
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::usermessage::{ChatMessageKind, UserMessage};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::message::MessagePacketMeta;
use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
use tf_demo_parser::demo::parser::handler::BorrowMessageHandler;
use tf_demo_parser::demo::parser::{MessageHandler, RawPacketStream};
use tf_demo_parser::demo::message::packetentities::{EntityId, PacketEntity};
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::packet::datatable::{ParseSendTable, ServerClass, ServerClassName};
use tf_demo_parser::demo::parser::analyser::{Team, UserId};

use tf_demo_parser::{Demo, MessageType, ParserState, ReadResult, Stream};

use crate::base::demo_handler_base::CheatDemoHandler;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub tick: u32,
    pub chat_type: String,
    pub from_name: String,
    pub from_steam_id: Option<u64>,
    pub message: String,
    /// Team of the sender at the moment the message was sent (`Team::Other` when unknown).
    #[serde(default)]
    pub team: Team,
}

#[derive(Default)]
pub struct ChatParser {
    pub messages: Vec<ChatMessage>,
    pub entid_to_name: HashMap<EntityId, String>,
    pub entid_to_steam_id: HashMap<EntityId, u64>,
    pub entid_to_userid: HashMap<EntityId, UserId>,
    /// Current team per player entity, kept up to date while parsing so each message
    /// gets the sender's team at the time it was sent.
    pub entid_to_team: HashMap<EntityId, Team>,
    pub class_names: Vec<ServerClassName>,
    pub current_tick: u32,
}

impl ChatParser {
    pub fn new() -> Self {
        Self::default()
    }

    fn set_team_of_user(&mut self, uid: UserId, team: Team) {
        let ents: Vec<EntityId> = self
            .entid_to_userid
            .iter()
            .filter(|(_, u)| **u == uid)
            .map(|(e, _)| *e)
            .collect();
        for ent in ents {
            self.entid_to_team.insert(ent, team);
        }
    }

    fn handle_player_resource(&mut self, entity: &PacketEntity, parser_state: &ParserState) {
        for prop in entity.props(parser_state) {
            let Some((table_name, prop_name)) = prop.identifier.names() else { continue };
            if table_name.as_str() != "m_iTeam" {
                continue;
            }
            let Ok(player_id) = prop_name.as_str().parse::<u32>() else { continue };
            if let Ok(val) = i64::try_from(&prop.value) {
                let team = Team::new(val);
                if team != Team::Other {
                    self.entid_to_team.insert(EntityId::from(player_id), team);
                }
            }
        }
    }

    fn parse_user_info(
        &mut self,
        index: usize,
        text: Option<&str>,
        data: Option<Stream>,
    ) -> ReadResult<()> {
        if let Some(user_info) =
            tf_demo_parser::demo::data::UserInfo::parse_from_string_table(index as u16, text, data)?
        {
            let ent_id = user_info.entity_id;
            if self.entid_to_userid.get(&ent_id) != Some(&user_info.player_info.user_id) {
                // A different player took over this slot: forget the previous occupant's team
                self.entid_to_team.remove(&ent_id);
            }
            self.entid_to_userid.insert(ent_id, user_info.player_info.user_id);
            let name = user_info.player_info.name;
            if !name.is_empty() {
                self.entid_to_name.insert(ent_id, name.clone());
            }
            if let Ok(sid) = SteamID::from_steam3(&user_info.player_info.steam_id) {
                let id64: u64 = sid.into();
                self.entid_to_steam_id.insert(ent_id, id64);
            }
        }
        Ok(())
    }
}

impl MessageHandler for ChatParser {
    type Output = Vec<ChatMessage>;

    fn does_handle(message_type: MessageType) -> bool {
        matches!(
            message_type,
            MessageType::UserMessage
                | MessageType::NetTick
                | MessageType::GameEvent
                | MessageType::PacketEntities
        )
    }

    fn handle_message(&mut self, message: &Message, tick: DemoTick, parser_state: &ParserState) {
        self.current_tick = tick.into();
        match message {
            Message::PacketEntities(msg) => {
                for entity in &msg.entities {
                    let is_resource = self
                        .class_names
                        .get(usize::from(entity.server_class))
                        .is_some_and(|c| c.as_str() == "CTFPlayerResource");
                    if is_resource {
                        self.handle_player_resource(entity, parser_state);
                    }
                }
            }
            Message::GameEvent(ge) => match &ge.event {
                GameEvent::PlayerTeam(e) => {
                    let uid = UserId::from(e.user_id);
                    self.set_team_of_user(uid, Team::new(e.team as i64));
                }
                GameEvent::PlayerSpawn(e) => {
                    let team = Team::new(e.team as i64);
                    if team != Team::Other {
                        self.set_team_of_user(UserId::from(e.user_id), team);
                    }
                }
                GameEvent::PlayerDisconnect(e) => {
                    let uid = UserId::from(e.user_id);
                    let ents: Vec<EntityId> = self
                        .entid_to_userid
                        .iter()
                        .filter(|(_, u)| **u == uid)
                        .map(|(e, _)| *e)
                        .collect();
                    for ent in ents {
                        self.entid_to_team.remove(&ent);
                    }
                }
                _ => {}
            },
            Message::UserMessage(user_msg) => match user_msg {
                UserMessage::SayText2(say) => {
                    let kind_str = format!("{:?}", say.kind);
                    let (chat_type, is_relevant) = match say.kind {
                        ChatMessageKind::ChatAll => ("All".to_string(), true),
                        ChatMessageKind::ChatAllDead => ("All (Dead)".to_string(), true),
                        ChatMessageKind::ChatTeam => ("Team".to_string(), true),
                        ChatMessageKind::ChatTeamDead => ("Team (Dead)".to_string(), true),
                        ChatMessageKind::ChatAllSpec => ("Spec".to_string(), true),
                        ChatMessageKind::Empty => {
                            // Some TF2 server mods use Empty with formatted text
                            ("All".to_string(), true)
                        }
                        _ => {
                            if kind_str.contains("All") {
                                ("All".to_string(), true)
                            } else if kind_str.contains("Team") {
                                ("Team".to_string(), true)
                            } else {
                                (kind_str, false)
                            }
                        }
                    };

                    if is_relevant {
                        let from_name = say.from.as_ref().map(|s| s.to_string()).unwrap_or_else(|| {
                            self.entid_to_name
                                .get(&say.client)
                                .cloned()
                                .unwrap_or_else(|| format!("Player {}", u32::from(say.client)))
                        });

                        let from_steam_id = self.entid_to_steam_id.get(&say.client).copied();

                        let mut text = say.text.to_string();
                        // Clean up leading color codes or control chars if any
                        if text.starts_with('\x01') || text.starts_with('\x02') || text.starts_with('\x03') || text.starts_with('\x04') {
                            text = text.trim_start_matches(|c: char| c.is_control() || c as u32 <= 4).to_string();
                        }

                        let team = self.entid_to_team.get(&say.client).copied().unwrap_or_default();

                        self.messages.push(ChatMessage {
                            tick: self.current_tick,
                            chat_type,
                            from_name,
                            from_steam_id,
                            message: text,
                            team,
                        });
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

    fn into_output(self, _state: &ParserState) -> Self::Output {
        self.messages
    }
}

impl BorrowMessageHandler for ChatParser {
    fn borrow_output(&self, _state: &ParserState) -> &Self::Output {
        &self.messages
    }
}

impl crate::base::demo_handler_base::DemoHandlerAnalyser for ChatParser {}

pub fn parse_chat<P: AsRef<Path>>(path: P) -> Result<Vec<ChatMessage>> {
    let file_bytes = std::fs::read(path.as_ref())
        .with_context(|| format!("Failed to read demo file: {:?}", path.as_ref()))?;

    let demo = Demo::new(&file_bytes);
    let mut stream = demo.get_stream();
    let header: Header = Header::read(&mut stream)
        .with_context(|| "Failed to read demo header")?;
    let mut packets = RawPacketStream::new(stream);

    let parser = ChatParser::new();
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

    let messages = handler.analyser.into_output(&handler.state_handler);
    Ok(messages)
}

/// Name color of a chat sender by team; `None` keeps the theme's text color.
#[cfg(feature = "gui")]
fn team_color(team: Team) -> Option<egui::Color32> {
    match team {
        Team::Red => Some(egui::Color32::from_rgb(255, 100, 100)),
        Team::Blue => Some(egui::Color32::from_rgb(100, 160, 255)),
        Team::Spectator => Some(egui::Color32::from_rgb(180, 180, 180)),
        Team::Other => None,
    }
}

#[cfg(feature = "gui")]
pub struct ChatWindow {
    pub messages: Vec<ChatMessage>,
    pub is_loading: bool,
    pub is_open: bool,
    pub error: Option<String>,
    pub filter_query: String,
    pub filter_type: String,
    recv: Option<std::sync::mpsc::Receiver<Result<Vec<ChatMessage>>>>,
}

#[cfg(feature = "gui")]
impl ChatWindow {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            is_loading: false,
            is_open: false,
            error: None,
            filter_query: String::new(),
            filter_type: "All Types".to_string(),
            recv: None,
        }
    }

    pub fn open_for_file<P: AsRef<Path>>(&mut self, path: P) {
        self.is_open = true;
        self.is_loading = true;
        self.messages.clear();
        self.error = None;

        let path_buf = path.as_ref().to_path_buf();
        let (send, recv) = std::sync::mpsc::channel();
        self.recv = Some(recv);

        std::thread::spawn(move || {
            let result = parse_chat(&path_buf);
            let _ = send.send(result);
        });
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        if !self.is_open {
            return;
        }

        if let Some(recv) = &self.recv {
            if let Ok(res) = recv.try_recv() {
                self.is_loading = false;
                match res {
                    Ok(msgs) => {
                        self.messages = msgs;
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
        egui::Window::new("Demo Chat Messages")
            .open(&mut is_open)
            .collapsible(false)
            .resizable(false)
            .default_size([540.0, 560.0])
            .min_size([500.0, 480.0])
            .max_size([540.0, 560.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(235, 70, 70), format!("Error: {}", err));
                    return;
                }

                if self.is_loading {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Parsing chat messages...");
                    });
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Search:");
                    let prev_bg = ui.visuals().extreme_bg_color;
                    ui.visuals_mut().extreme_bg_color = ui.visuals().widgets.inactive.bg_fill;
                    ui.text_edit_singleline(&mut self.filter_query);
                    ui.visuals_mut().extreme_bg_color = prev_bg;

                    if !self.filter_query.is_empty() && ui.small_button("Clear").clicked() {
                        self.filter_query.clear();
                    }

                    ui.add_space(10.0);
                    egui::ComboBox::from_label("Type")
                        .selected_text(&self.filter_type)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.filter_type, "All Types".to_string(), "All Types");
                            ui.selectable_value(&mut self.filter_type, "All".to_string(), "All Chat");
                            ui.selectable_value(&mut self.filter_type, "Team".to_string(), "Team Chat");
                        });

                    ui.add_space(10.0);
                    ui.label(format!("Total: {}", self.messages.len()));
                });
                ui.separator();

                let filtered: Vec<&ChatMessage> = self
                    .messages
                    .iter()
                    .filter(|m| {
                        let matches_q = if self.filter_query.is_empty() {
                            true
                        } else {
                            let q = self.filter_query.to_lowercase();
                            m.from_name.to_lowercase().contains(&q)
                                || m.message.to_lowercase().contains(&q)
                                || m.tick.to_string().contains(&q)
                        };

                        let matches_t = match self.filter_type.as_str() {
                            "All" => m.chat_type.starts_with("All"),
                            "Team" => m.chat_type.starts_with("Team"),
                            _ => true,
                        };

                        matches_q && matches_t
                    })
                    .collect();

                let available_height = ui.available_height() - 40.0;
                egui::ScrollArea::vertical()
                    .max_height(available_height)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        egui::Grid::new("chat_table_grid")
                            .striped(true)
                            .num_columns(4)
                            .spacing([8.0, 4.0])
                            .min_col_width(40.0)
                            .max_col_width(240.0)
                            .show(ui, |ui| {
                                ui.strong("Tick");
                                ui.strong("Type");
                                ui.strong("Name");
                                ui.strong("Message");
                                ui.end_row();

                                for msg in filtered {
                                    ui.label(format!("{}", msg.tick));

                                    let type_color = if msg.chat_type.starts_with("Team") {
                                        egui::Color32::from_rgb(100, 200, 255)
                                    } else {
                                        egui::Color32::from_rgb(220, 220, 220)
                                    };
                                    ui.colored_label(type_color, &msg.chat_type);

                                    let name_text = match team_color(msg.team) {
                                        Some(color) => egui::RichText::new(&msg.from_name).color(color),
                                        None => egui::RichText::new(&msg.from_name),
                                    };
                                    let name_label = ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| ui.selectable_label(false, name_text),
                                    ).inner;
                                    name_label.context_menu(|ui| {
                                        if ui.button("Copy Ingame Name").clicked() {
                                            ui.ctx().copy_text(msg.from_name.clone());
                                            ui.close_menu();
                                        }
                                        if let Some(sid) = msg.from_steam_id {
                                            if ui.button("Copy SteamID").clicked() {
                                                ui.ctx().copy_text(format!("{}", sid));
                                                ui.close_menu();
                                            }
                                            if ui.button("Open Steam Profile").clicked() {
                                                let _ = opener::open_browser(format!(
                                                    "https://steamcommunity.com/profiles/{}",
                                                    sid
                                                ));
                                                ui.close_menu();
                                            }
                                        }
                                        if ui.button("Copy Entire Message").clicked() {
                                            ui.ctx().copy_text(format!(
                                                "[{}] {}: {}",
                                                msg.chat_type, msg.from_name, msg.message
                                            ));
                                            ui.close_menu();
                                        }
                                    });

                                    ui.scope(|ui| {
                                        ui.set_max_width(240.0);
                                        let msg_label = ui.add(egui::Label::new(&msg.message).wrap());
                                        msg_label.context_menu(|ui| {
                                            if ui.button("Copy Message Content").clicked() {
                                                ui.ctx().copy_text(msg.message.clone());
                                                ui.close_menu();
                                            }
                                            if ui.button("Copy Ingame Name").clicked() {
                                                ui.ctx().copy_text(msg.from_name.clone());
                                                ui.close_menu();
                                            }
                                        });
                                    });

                                    ui.end_row();
                                }
                            });
                    });


                ui.separator();
                ui.label("Right-click any player or message to copy");
            });

        self.is_open = is_open;
    }
}

