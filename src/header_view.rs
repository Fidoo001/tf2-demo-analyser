use std::path::Path;
use anyhow::{Context, Result};
use bitbuffer::BitRead;
#[cfg(feature = "gui")]
use eframe::egui;

use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::Packet;
use tf_demo_parser::demo::parser::RawPacketStream;
use tf_demo_parser::{Demo, ParserState};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeaderData {
    pub file_name: String,
    pub demo_type: String,
    pub version: u32,
    pub protocol: u32,
    pub server: String,
    pub server_name: String,
    pub nick: String,
    pub map: String,
    pub game: String,
    pub duration: f32,
    pub ticks: u32,
    pub frames: u32,
    pub signon: u32,
    pub avg_tickrate: f32,
    pub avg_framerate: f32,
}

impl HeaderData {
    pub fn from_header(header: &Header, file_name: String, server_name: String) -> Self {
        let avg_tickrate = if header.duration > 0.0 {
            header.ticks as f32 / header.duration
        } else {
            0.0
        };
        let avg_framerate = if header.duration > 0.0 {
            header.frames as f32 / header.duration
        } else {
            0.0
        };

        Self {
            file_name,
            demo_type: header.demo_type.clone(),
            version: header.version,
            protocol: header.protocol,
            server: header.server.clone(),
            server_name,
            nick: header.nick.clone(),
            map: header.map.clone(),
            game: header.game.clone(),
            duration: header.duration,
            ticks: header.ticks,
            frames: header.frames,
            signon: header.signon,
            avg_tickrate,
            avg_framerate,
        }
    }

    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let file_name = path_ref
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let file_bytes = std::fs::read(path_ref)
            .with_context(|| format!("Failed to read demo file: {:?}", path_ref))?;

        Self::parse_bytes(&file_bytes, file_name)
    }

    /// Parses the header and the server name (from the first ServerInfo / hostname cvar),
    /// stopping as soon as the server name is found.
    pub fn parse_bytes(file_bytes: &[u8], file_name: String) -> Result<Self> {
        let demo = Demo::new(file_bytes);
        let mut stream = demo.get_stream();
        let header = Header::read(&mut stream)
            .with_context(|| "Failed to parse demo header from stream")?;

        let mut server_name = String::new();
        let parser_state = ParserState::new(header.protocol, |_| true, false);
        let mut packet_stream = RawPacketStream::new(stream);
        while let Ok(Some(packet)) = packet_stream.next(&parser_state) {
            match packet {
                Packet::Signon(p) | Packet::Message(p) => {
                    for msg in p.messages {
                        match msg {
                            Message::ServerInfo(si) => {
                                server_name = si.server_name.to_string();
                                break;
                            }
                            Message::SetConVar(cvar) => {
                                for cv in cvar.vars {
                                    if cv.key == "hostname" && server_name.is_empty() {
                                        server_name = cv.value;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    if !server_name.is_empty() {
                        break;
                    }
                }
                Packet::Stop(_) => break,
                _ => {}
            }
        }

        Ok(Self::from_header(&header, file_name, server_name))
    }

    /// Valve casual/competitive servers, same check as nocrex/oob_pitch.
    pub fn is_valve_server(&self) -> bool {
        self.server_name.trim().starts_with("Valve Matchmaking Server")
    }
}

#[cfg(feature = "gui")]
#[derive(Default)]
pub struct HeaderWindow {
    pub data: Option<HeaderData>,
    pub error: Option<String>,
    pub is_open: bool,
}

#[cfg(feature = "gui")]
impl HeaderWindow {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open_for_file<P: AsRef<Path>>(&mut self, path: P) {
        self.is_open = true;
        let path_ref = path.as_ref();
        let fname = path_ref
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        if let Some(data) = &self.data {
            if data.file_name == fname && self.error.is_none() {
                return;
            }
        }
        match HeaderData::parse_file(path_ref) {
            Ok(data) => {
                self.data = Some(data);
                self.error = None;
            }
            Err(e) => {
                self.data = None;
                self.error = Some(format!("{:#}", e));
            }
        }
    }

    pub fn open_with_header(&mut self, header: &Header, file_name: String, server_name: String) {
        self.is_open = true;
        self.data = Some(HeaderData::from_header(header, file_name, server_name));
        self.error = None;
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        if !self.is_open {
            return;
        }

        let mut is_open = self.is_open;
        egui::Window::new("Demo Header Info")
            .open(&mut is_open)
            .collapsible(false)
            .resizable(false)
            .default_size([540.0, 440.0])
            .min_size([540.0, 440.0])
            .max_size([540.0, 440.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {


                ui.spacing_mut().item_spacing.y = 6.0;

                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(235, 70, 70), format!("Error: {}", err));
                    return;
                }

                let Some(data) = &self.data else {
                    ui.label("No header data loaded.");
                    return;
                };

                ui.horizontal(|ui| {
                    ui.heading("Demo Information");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(&data.file_name)
                                .monospace()
                                .color(ui.visuals().weak_text_color()),
                        );
                    });
                });
                ui.separator();

                let hours = (data.duration / 3600.0).floor() as u32;
                let minutes = ((data.duration % 3600.0) / 60.0).floor() as u32;
                let seconds = (data.duration % 60.0).floor() as u32;
                let millis = ((data.duration % 1.0) * 1000.0).floor() as u32;
                let formatted_duration = format!(
                    "{:02}:{:02}:{:02}.{:03} ({:.2}s)",
                    hours, minutes, seconds, millis, data.duration
                );

                egui::Grid::new("header_info_grid")
                    .striped(true)
                    .num_columns(2)
                    .spacing([20.0, 6.0])
                    .show(ui, |ui| {
                        ui.strong("Map:");
                        ui.horizontal(|ui| {
                            ui.label(&data.map);
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(data.map.clone());
                            }
                        });
                        ui.end_row();

                        let s_name = if data.server_name.is_empty() {
                            "Unknown"
                        } else {
                            &data.server_name
                        };
                        ui.strong("Server Name:");
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(s_name).wrap());
                            if !data.server_name.is_empty() && ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(data.server_name.clone());
                            }
                        });
                        ui.end_row();

                        ui.strong("Server Address:");
                        ui.horizontal(|ui| {
                            ui.label(&data.server);
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(data.server.clone());
                            }
                        });
                        ui.end_row();

                        ui.strong("Recorder (Nick):");
                        ui.horizontal(|ui| {
                            ui.label(&data.nick);
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text(data.nick.clone());
                            }
                        });
                        ui.end_row();

                        ui.strong("Length / Duration:");
                        ui.label(formatted_duration);
                        ui.end_row();

                        ui.strong("Ticks:");
                        ui.label(format!("{} ticks", data.ticks));
                        ui.end_row();

                        ui.strong("Frames:");
                        ui.label(format!("{} frames", data.frames));
                        ui.end_row();

                        ui.strong("Average Tickrate:");
                        ui.label(format!("{:.2} ticks/sec", data.avg_tickrate));
                        ui.end_row();

                        ui.strong("Average Framerate:");
                        ui.label(format!("{:.2} fps", data.avg_framerate));
                        ui.end_row();

                        ui.strong("Game Directory:");
                        ui.label(&data.game);
                        ui.end_row();

                        ui.strong("Protocol Version:");
                        ui.label(format!("{}", data.protocol));
                        ui.end_row();

                        ui.strong("Demo Type / Version:");
                        ui.label(format!("{} (v{})", data.demo_type, data.version));
                        ui.end_row();

                        ui.strong("Signon Data:");
                        ui.label(format!("{} bytes", data.signon));
                        ui.end_row();
                    });
            });

        self.is_open = is_open;
    }
}
