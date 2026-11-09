//! GUI of the TF2BD feature: the settings category, the playerlist and attribute configuration
//! windows and the "Mark player" window.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{channel, Receiver, Sender};

use eframe::egui;

use crate::lib::general_settings::{format_hex_rgb, parse_hex_rgb};
use crate::lib::parameters::Parameters;
use crate::playerlist_view::TagHighlight;
use crate::tf2bd::{
    default_attribute_color, found_attributes, is_remote_source, load_playerlist, LoadedPlayerlist, Playerlist,
    PlayerlistState, Tf2bdSettings,
};

/// (load generation, source, result)
type LoadResult = (u64, String, Result<Playerlist, String>);

const ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(235, 70, 70);
const OK_COLOR: egui::Color32 = egui::Color32::from_rgb(90, 200, 110);

struct MarkDialog {
    steam_id: u64,
    name: String,
    /// Source of the selected local playerlist.
    list: Option<String>,
    /// Playerlist the fields were last filled from.
    filled_from: Option<String>,
    /// Attributes offered as checkboxes, in priority order, and whether they are selected.
    attributes: Vec<(String, bool)>,
    new_attribute: String,
    proof: String,
    error: Option<String>,
}

enum MarkAction {
    Save,
    Delete,
    Close,
}

pub struct Tf2bdUi {
    settings: Tf2bdSettings,
    lists: Vec<LoadedPlayerlist>,
    /// Incremented when the lists are (re)loaded or unloaded, so stale results are dropped.
    generation: u64,
    load_send: Sender<LoadResult>,
    load_recv: Receiver<LoadResult>,
    tags: HashMap<u64, TagHighlight>,
    /// Players announced in party chat when seen on the server: steam id → notified attributes.
    notify: HashMap<u64, String>,
    tags_dirty: bool,
    settings_changed: bool,
    notify_changed: bool,
    show_playerlists: bool,
    show_attributes: bool,
    new_source: String,
    playerlists_message: Option<Result<String, String>>,
    mark: Option<MarkDialog>,
    last_mark_list: Option<String>,
}

impl Tf2bdUi {
    pub fn new(settings: Tf2bdSettings) -> Self {
        let (load_send, load_recv) = channel();
        let mut ui = Self {
            settings,
            lists: Vec::new(),
            generation: 0,
            load_send,
            load_recv,
            tags: HashMap::new(),
            notify: HashMap::new(),
            tags_dirty: false,
            settings_changed: false,
            notify_changed: false,
            show_playerlists: false,
            show_attributes: false,
            new_source: String::new(),
            playerlists_message: None,
            mark: None,
            last_mark_list: None,
        };
        if ui.settings.enabled {
            ui.reload_all();
        }
        ui
    }

    pub fn enabled(&self) -> bool {
        self.settings.enabled
    }

    /// Tagged players with their highlight (empty while the feature is disabled).
    pub fn tags(&self) -> &HashMap<u64, TagHighlight> {
        &self.tags
    }

    pub fn any_modal_open(&self) -> bool {
        self.show_playerlists || self.show_attributes || self.mark.is_some()
    }

    /// The `tf2bd` params.json section, when the settings changed since the last call.
    pub fn take_settings_changed(&mut self) -> Option<Parameters> {
        std::mem::take(&mut self.settings_changed).then(|| self.settings.to_parameters())
    }

    /// The players to announce in party chat (steam id → attributes), when it changed since the last call.
    pub fn take_notify_changed(&mut self) -> Option<HashMap<u64, String>> {
        std::mem::take(&mut self.notify_changed).then(|| self.notify.clone())
    }

    fn reload_all(&mut self) {
        self.generation += 1;
        self.lists = self
            .settings
            .playerlists
            .iter()
            .map(|source| LoadedPlayerlist { source: source.clone(), state: PlayerlistState::Loading })
            .collect();
        for source in self.settings.playerlists.clone() {
            self.spawn_load(source);
        }
        self.tags_dirty = true;
    }

    fn spawn_load(&self, source: String) {
        let send = self.load_send.clone();
        let generation = self.generation;
        std::thread::spawn(move || {
            let result = load_playerlist(&source);
            let _ = send.send((generation, source, result));
        });
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.settings.enabled = enabled;
        self.settings_changed = true;
        if enabled {
            self.reload_all();
        } else {
            self.generation += 1;
            self.lists.clear();
            self.show_playerlists = false;
            self.show_attributes = false;
            self.mark = None;
            self.tags_dirty = true;
        }
    }

    /// Receives finished loads and rebuilds the highlights. Call once per frame.
    pub fn poll(&mut self, ctx: &egui::Context) {
        while let Ok((generation, source, result)) = self.load_recv.try_recv() {
            if generation != self.generation {
                continue;
            }
            if let Some(list) = self.lists.iter_mut().find(|l| l.source == source) {
                list.state = match result {
                    Ok(list) => PlayerlistState::Loaded(list),
                    Err(e) => PlayerlistState::Failed(e),
                };
                self.tags_dirty = true;
            }
        }
        if std::mem::take(&mut self.tags_dirty) {
            let found = found_attributes(&self.lists);
            if self.settings.add_attributes(found.iter().map(String::as_str)) {
                self.settings_changed = true;
            }
            self.rebuild_tags();
        }
        if self.lists.iter().any(|l| matches!(l.state, PlayerlistState::Loading)) {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }

    fn rebuild_tags(&mut self) {
        let mut attributes: HashMap<u64, Vec<String>> = HashMap::new();
        for list in self.lists.iter().filter_map(LoadedPlayerlist::list) {
            for (&id, entry) in list.players() {
                let attrs = attributes.entry(id).or_default();
                for attr in &entry.attributes {
                    if !attrs.contains(attr) {
                        attrs.push(attr.clone());
                    }
                }
            }
        }
        self.tags.clear();
        let mut notify = HashMap::new();
        for (id, mut attrs) in attributes {
            if attrs.is_empty() {
                continue;
            }
            self.settings.sort_by_priority(&mut attrs);
            let rgb = self
                .settings
                .attribute(&attrs[0])
                .map(|a| a.rgb())
                .unwrap_or_else(|| default_attribute_color(&attrs[0]));
            let notified: Vec<&str> = attrs
                .iter()
                .filter(|a| self.settings.attribute(a).is_some_and(|s| s.notify))
                .map(String::as_str)
                .collect();
            if !notified.is_empty() {
                notify.insert(id, notified.join(", "));
            }
            self.tags.insert(
                id,
                TagHighlight { color: egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), label: attrs.join(", ") },
            );
        }
        if notify != self.notify {
            self.notify = notify;
            self.notify_changed = true;
        }
    }

    /// The TF2BD category of the settings.
    pub fn settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.heading("TF2BD");
        ui.separator();
        let mut enabled = self.settings.enabled;
        if ui
            .checkbox(&mut enabled, "Enable TF2BD features")
            .on_hover_text("Load TF2 Bot Detector playerlists, highlight the tagged players and mark players from the right click menu")
            .changed()
        {
            self.set_enabled(enabled);
        }
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Configure playerlists").clicked() {
                    self.show_playerlists = true;
                    self.playerlists_message = None;
                }
                if ui.button("Configure attributes").clicked() {
                    self.show_attributes = true;
                }
            });
            if enabled {
                let loading = self.lists.iter().filter(|l| matches!(l.state, PlayerlistState::Loading)).count();
                let failed = self.lists.iter().filter(|l| matches!(l.state, PlayerlistState::Failed(_))).count();
                let mut status = format!("{} tagged players from {} playerlist(s)", self.tags.len(), self.lists.len());
                if loading > 0 {
                    status.push_str(&format!(", {loading} loading"));
                }
                if failed > 0 {
                    status.push_str(&format!(", {failed} failed to load"));
                }
                ui.weak(status);
            }
        });
    }

    /// Shows the open windows.
    pub fn show(&mut self, ctx: &egui::Context) {
        // Changes made this frame are applied by the next `poll`
        if self.tags_dirty || self.settings_changed {
            ctx.request_repaint();
        }
        if !self.settings.enabled {
            return;
        }
        if self.show_playerlists {
            self.playerlists_modal(ctx);
        }
        if self.show_attributes {
            self.attributes_modal(ctx);
        }
        if self.mark.is_some() {
            self.mark_modal(ctx);
        }
    }

    fn add_source(&mut self, source: &str) {
        let source = source.trim().trim_matches('"').trim().to_string();
        if source.is_empty() {
            return;
        }
        if self.settings.playerlists.contains(&source) {
            self.playerlists_message = Some(Err("This playerlist is already added".to_string()));
            return;
        }
        if !is_remote_source(&source) && !std::path::Path::new(&source).is_file() {
            self.playerlists_message = Some(Err(format!("File not found: {source}")));
            return;
        }
        self.settings.playerlists.push(source.clone());
        self.settings_changed = true;
        self.lists.push(LoadedPlayerlist { source: source.clone(), state: PlayerlistState::Loading });
        self.spawn_load(source);
        self.new_source.clear();
        self.playerlists_message = None;
    }

    fn create_local_playerlist(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("TF2BD playerlist", &["json"])
            .set_file_name("playerlist.json")
            .save_file()
        else {
            return;
        };
        let source = path.to_string_lossy().into_owned();
        if path.exists() {
            // Never overwrite an existing file: it is added as it is
            self.add_source(&source);
            if self.playerlists_message.is_none() {
                self.playerlists_message = Some(Ok("The file already exists, it was added without changes".to_string()));
            }
            return;
        }
        let title = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        match Playerlist::new_empty(&title).save(&path) {
            Ok(()) => {
                self.add_source(&source);
                if self.playerlists_message.is_none() {
                    self.playerlists_message = Some(Ok(format!("Created {}", path.display())));
                }
            }
            Err(e) => self.playerlists_message = Some(Err(e)),
        }
    }

    fn playerlists_modal(&mut self, ctx: &egui::Context) {
        let mut remove = None;
        let mut add = false;
        let mut browse = false;
        let mut create = false;
        let mut reload = false;
        let mut suggested: Option<&str> = None;
        let mut open = true;
        egui::Window::new("Playerlists")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(540.0)
            .min_width(540.0)
            .max_width(540.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
            ui.weak("TF2BD playerlists, as local files or URLs. URLs are downloaded again on every start.");
            ui.separator();
            if self.lists.is_empty() {
                ui.label("No playerlists added");
            }
            egui::ScrollArea::vertical().max_height(300.0).auto_shrink([false, true]).show(ui, |ui| {
                for (i, list) in self.lists.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Remove from the playerlists (the file is not deleted)")
                            .clicked()
                        {
                            remove = Some(i);
                        }
                        match &list.state {
                            PlayerlistState::Loading => {
                                ui.spinner();
                            }
                            PlayerlistState::Loaded(pl) => {
                                ui.colored_label(OK_COLOR, "✔").on_hover_text(format!("{} players", pl.players().len()));
                            }
                            PlayerlistState::Failed(e) => {
                                ui.colored_label(ERROR_COLOR, "✖").on_hover_text(e);
                            }
                        }
                        ui.weak(if list.is_remote() { "URL  " } else { "Local" });
                        let mut text = list.source.clone();
                        if let Some(pl) = list.list() {
                            text.push_str(&format!(" ({} players)", pl.players().len()));
                        }
                        ui.add(egui::Label::new(text).truncate()).on_hover_text(&list.source);
                    });
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut self.new_source)
                        .hint_text("Path or URL of a playerlist")
                        .desired_width(330.0),
                );
                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    add = true;
                }
                if ui.button("Browse...").clicked() {
                    browse = true;
                }
                if ui.add_enabled(!self.new_source.trim().is_empty(), egui::Button::new("Add")).clicked() {
                    add = true;
                }
            });
            ui.horizontal(|ui| {
                ui.weak("Community lists:");
                for (name, url) in crate::tf2bd::SUGGESTED_PLAYERLISTS {
                    let added = self.settings.playerlists.iter().any(|source| source == url);
                    if ui
                        .add_enabled(!added, egui::Button::new(format!("+ {name}")))
                        .on_hover_text(url)
                        .clicked()
                    {
                        suggested = Some(url);
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Create new local playerlist...").clicked() {
                    create = true;
                }
                if ui
                    .add_enabled(!self.lists.is_empty(), egui::Button::new("Reload all"))
                    .on_hover_text("Read the local playerlists and download the URLs again")
                    .clicked()
                {
                    reload = true;
                }
            });
            match &self.playerlists_message {
                Some(Ok(msg)) => {
                    ui.colored_label(OK_COLOR, msg);
                }
                Some(Err(msg)) => {
                    ui.colored_label(ERROR_COLOR, msg);
                }
                None => {}
            }
        });

        if let Some(url) = suggested {
            self.add_source(url);
        }
        if let Some(i) = remove {
            let source = self.lists.remove(i).source;
            self.settings.playerlists.retain(|s| *s != source);
            self.settings_changed = true;
            self.tags_dirty = true;
            self.playerlists_message = None;
        }
        if browse {
            if let Some(path) = rfd::FileDialog::new().add_filter("TF2BD playerlist", &["json"]).pick_file() {
                self.new_source = path.to_string_lossy().into_owned();
                add = true;
            }
        }
        if add {
            let source = self.new_source.clone();
            self.add_source(&source);
        }
        if create {
            self.create_local_playerlist();
        }
        if reload {
            self.reload_all();
            self.playerlists_message = None;
        }
        if !open {
            self.show_playerlists = false;
        }
    }

    fn attributes_modal(&mut self, ctx: &egui::Context) {
        let found: HashSet<String> = found_attributes(&self.lists).into_iter().collect();
        // Settings indexes of the attributes shown (those used in the loaded lists), in priority order
        let visible: Vec<usize> = (0..self.settings.attributes.len())
            .filter(|&i| found.contains(&self.settings.attributes[i].name))
            .collect();
        let loading = self.lists.iter().any(|l| matches!(l.state, PlayerlistState::Loading));
        let mut swap = None;
        let mut changed = false;
        let mut open = true;
        egui::Window::new("Attributes")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(500.0)
            .min_width(500.0)
            .max_width(500.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
            ui.weak("Players with several attributes are highlighted with the color of the highest one.");
            ui.separator();
            if loading {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Loading playerlists...");
                });
            }
            if visible.is_empty() {
                ui.label("No attributes found in the loaded playerlists");
            }
            egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, true]).show(ui, |ui| {
                egui::Grid::new("tf2bd_attributes_grid").num_columns(4).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
                    for (pos, &idx) in visible.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.add_enabled(pos > 0, egui::Button::new("⬆").small()).on_hover_text("Higher priority").clicked() {
                                swap = Some((idx, visible[pos - 1]));
                            }
                            if ui
                                .add_enabled(pos + 1 < visible.len(), egui::Button::new("⬇").small())
                                .on_hover_text("Lower priority")
                                .clicked()
                            {
                                swap = Some((idx, visible[pos + 1]));
                            }
                        });
                        let attribute = &mut self.settings.attributes[idx];
                        ui.label(&attribute.name);
                        ui.horizontal(|ui| {
                            let default = default_attribute_color(&attribute.name);
                            let mut rgb = parse_hex_rgb(&attribute.color).unwrap_or(default);
                            if ui.color_edit_button_srgb(&mut rgb).changed() {
                                attribute.color = format_hex_rgb(rgb);
                                changed = true;
                            }
                            if ui.add(egui::TextEdit::singleline(&mut attribute.color).desired_width(70.0).char_limit(7)).changed() {
                                changed = true;
                            }
                            let default_hex = format_hex_rgb(default);
                            let is_default = parse_hex_rgb(&attribute.color) == Some(default);
                            if ui
                                .add_enabled(!is_default, egui::Button::new("Default").small())
                                .on_hover_text(format!("Reset to default ({default_hex})"))
                                .clicked()
                            {
                                attribute.color = default_hex;
                                changed = true;
                            }
                        });
                        let notify_hover = if cfg!(feature = "rcon") {
                            "During live scanning, send a party chat notification when a player with this attribute joins the match"
                        } else {
                            "Party chat notifications need a build with the RCON feature"
                        };
                        if ui
                            .add_enabled(cfg!(feature = "rcon"), egui::Checkbox::new(&mut attribute.notify, "Notifications"))
                            .on_hover_text(notify_hover)
                            .on_disabled_hover_text(notify_hover)
                            .changed()
                        {
                            changed = true;
                        }
                        ui.end_row();
                    }
                });
            });
        });

        if let Some((a, b)) = swap {
            self.settings.attributes.swap(a, b);
            changed = true;
        }
        if changed {
            self.settings_changed = true;
            self.rebuild_tags();
        }
        if !open {
            self.show_attributes = false;
        }
    }

    /// Opens the "Mark player" modal for a player.
    pub fn open_mark(&mut self, steam_id: u64, name: &str) {
        if !self.settings.enabled {
            return;
        }
        let found: HashSet<String> = found_attributes(&self.lists).into_iter().collect();
        let attributes = self
            .settings
            .attributes
            .iter()
            .filter(|a| found.contains(&a.name))
            .map(|a| (a.name.clone(), false))
            .collect();
        let local: Vec<&LoadedPlayerlist> = self.local_lists().collect();
        // A list that already has the player, then the last used one, then the first one
        let list = local
            .iter()
            .find(|l| l.list().is_some_and(|pl| pl.get(steam_id).is_some()))
            .or_else(|| local.iter().find(|l| Some(&l.source) == self.last_mark_list.as_ref()))
            .or_else(|| local.first())
            .map(|l| l.source.clone());
        self.mark = Some(MarkDialog {
            steam_id,
            name: name.to_string(),
            list,
            filled_from: None,
            attributes,
            new_attribute: String::new(),
            proof: String::new(),
            error: None,
        });
    }

    /// Loaded local playerlists, the only ones players can be marked in.
    fn local_lists(&self) -> impl Iterator<Item = &LoadedPlayerlist> {
        self.lists.iter().filter(|l| !l.is_remote() && l.list().is_some())
    }

    fn mark_modal(&mut self, ctx: &egui::Context) {
        let Some(mut dialog) = self.mark.take() else { return };

        // Fill the fields from the player's entry when another list is selected. Without an entry
        // in that list, the attribute selection is kept so it can be saved there. The proof is
        // always replaced by that list's proof (empty without an entry) so it never carries over.
        if dialog.list != dialog.filled_from {
            dialog.filled_from = dialog.list.clone();
            let entry = dialog
                .list
                .as_ref()
                .and_then(|source| self.lists.iter().find(|l| l.source == *source))
                .and_then(|l| l.list())
                .and_then(|pl| pl.get(dialog.steam_id));
            dialog.proof = entry.map(|e| e.proof.join("\n")).unwrap_or_default();
            if let Some(entry) = entry {
                for (_, selected) in dialog.attributes.iter_mut() {
                    *selected = false;
                }
                let mut attrs = entry.attributes.clone();
                self.settings.sort_by_priority(&mut attrs);
                for attr in attrs {
                    match dialog.attributes.iter_mut().find(|(name, _)| *name == attr) {
                        Some((_, selected)) => *selected = true,
                        None => dialog.attributes.push((attr, true)),
                    }
                }
            }
        }

        let local: Vec<(String, String)> = self.local_lists().map(|l| (l.source.clone(), l.display_name())).collect();
        let has_entry = dialog
            .list
            .as_ref()
            .and_then(|source| self.lists.iter().find(|l| l.source == *source))
            .and_then(|l| l.list())
            .is_some_and(|pl| pl.get(dialog.steam_id).is_some());
        let mut action = None;
        let mut open = true;

        egui::Window::new("Mark player")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(460.0)
            .min_width(460.0)
            .max_width(460.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong(if dialog.name.is_empty() { "Unknown" } else { dialog.name.as_str() });
                ui.label(dialog.steam_id.to_string());
            });
            ui.separator();
            if local.is_empty() {
                ui.label("No local playerlist is loaded. Add or create one in \"Configure playerlists\".");
                return;
            }

            ui.horizontal(|ui| {
                ui.label("Playerlist:");
                let selected_name = dialog
                    .list
                    .as_ref()
                    .and_then(|s| local.iter().find(|(source, _)| source == s))
                    .map(|(_, name)| name.clone())
                    .unwrap_or_default();
                egui::ComboBox::from_id_salt("tf2bd_mark_list")
                    .selected_text(selected_name)
                    .width(300.0)
                    .show_ui(ui, |ui| {
                        for (source, name) in &local {
                            ui.selectable_value(&mut dialog.list, Some(source.clone()), name).on_hover_text(source);
                        }
                    });
            });
            if has_entry {
                ui.weak("The player is already marked in this playerlist");
            }

            ui.add_space(6.0);
            ui.label("Attributes:");
            ui.horizontal_wrapped(|ui| {
                for (name, selected) in dialog.attributes.iter_mut() {
                    ui.checkbox(selected, name.as_str());
                }
            });
            ui.horizontal(|ui| {
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut dialog.new_attribute)
                        .hint_text("New attribute")
                        .desired_width(160.0),
                );
                let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let new = dialog.new_attribute.trim().to_string();
                if (ui.add_enabled(!new.is_empty(), egui::Button::new("Add")).clicked() || enter) && !new.is_empty() {
                    match dialog.attributes.iter_mut().find(|(name, _)| *name == new) {
                        Some((_, selected)) => *selected = true,
                        None => dialog.attributes.push((new, true)),
                    }
                    dialog.new_attribute.clear();
                }
            });

            ui.add_space(6.0);
            ui.label("Proof / notes (one entry per line):");
            ui.add(
                egui::TextEdit::multiline(&mut dialog.proof)
                    .desired_rows(5)
                    .desired_width(f32::INFINITY),
            );

            if let Some(err) = &dialog.error {
                ui.colored_label(ERROR_COLOR, err);
            }
            ui.separator();
            ui.horizontal(|ui| {
                let any_selected = dialog.attributes.iter().any(|(_, s)| *s);
                if ui
                    .add_enabled(any_selected && dialog.list.is_some(), egui::Button::new("Save"))
                    .on_disabled_hover_text("Select at least one attribute")
                    .clicked()
                {
                    action = Some(MarkAction::Save);
                }
                if ui
                    .add_enabled(has_entry, egui::Button::new("Delete mark"))
                    .on_hover_text("Remove the player from this playerlist")
                    .clicked()
                {
                    action = Some(MarkAction::Delete);
                }
            });
        });

        if !open {
            action = Some(MarkAction::Close);
        }
        let keep_open = match action {
            None => true,
            Some(MarkAction::Close) => false,
            Some(action) => match self.apply_mark(&dialog, matches!(action, MarkAction::Delete)) {
                Ok(()) => false,
                Err(e) => {
                    dialog.error = Some(e);
                    true
                }
            },
        };
        if keep_open {
            self.mark = Some(dialog);
        }
    }

    /// Saves (or deletes) the player's entry in the selected playerlist file.
    fn apply_mark(&mut self, dialog: &MarkDialog, delete: bool) -> Result<(), String> {
        let source = dialog.list.clone().ok_or("No playerlist selected")?;
        let index = self
            .lists
            .iter()
            .position(|l| l.source == source && !l.is_remote())
            .ok_or("The playerlist is no longer configured")?;
        let mut list = self.lists[index].list().cloned().ok_or("The playerlist is not loaded")?;
        let attributes: Vec<String> = dialog.attributes.iter().filter(|(_, s)| *s).map(|(n, _)| n.clone()).collect();
        if delete {
            list.remove_player(dialog.steam_id);
        } else {
            let proof: Vec<String> = dialog
                .proof
                .lines()
                .map(str::trim_end)
                .filter(|l| !l.trim().is_empty())
                .map(str::to_string)
                .collect();
            list.set_player(dialog.steam_id, &attributes, &proof, Some(dialog.name.as_str()));
        }
        // Only replaced in memory once the file was written
        list.save(std::path::Path::new(&source))?;
        self.lists[index].state = PlayerlistState::Loaded(list);
        if !delete && self.settings.add_attributes(attributes.iter().map(String::as_str)) {
            self.settings_changed = true;
        }
        self.last_mark_list = Some(source);
        self.tags_dirty = true;
        Ok(())
    }
}
