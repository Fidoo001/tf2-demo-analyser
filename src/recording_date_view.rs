//! GUI part of the recording date estimate: a background worker that parses the opened demo
//! (and asks the Steam Web API about item drops when a key is set), and the date shown on the
//! right of the demo's file name row. See `lib::recording_date` for the methods.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};

use crate::lib::recording_date::{
    collect_item_facts, estimate_with_steam_api, format_date, format_date_time, DateMethod, RecordingDate,
    SteamApiLimiter,
};

/// The general settings key of the Steam Web API key.
pub const STEAM_API_KEY_SETTING: &str = "steam_api_key";

enum State {
    Pending,
    Done(Option<RecordingDate>),
    Failed(String),
}

struct Job {
    path: PathBuf,
    api_key: String,
    ctx: egui::Context,
}

pub struct RecordingDateView {
    jobs: Sender<Job>,
    results: Receiver<(PathBuf, String, Result<Option<RecordingDate>, String>)>,
    /// Finished estimates by (demo, API key used), so reopening a demo does not parse it again.
    cache: HashMap<(PathBuf, String), Result<Option<RecordingDate>, String>>,
    current: Option<(PathBuf, String)>,
    state: Option<State>,
}

impl Default for RecordingDateView {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordingDateView {
    pub fn new() -> Self {
        let (jobs, job_recv) = channel::<Job>();
        let (result_send, results) = channel();
        std::thread::Builder::new()
            .name("recording-date".to_string())
            .spawn(move || {
                let mut limiter = SteamApiLimiter::default();
                while let Ok(mut job) = job_recv.recv() {
                    // Only the newest request matters when demos are opened quickly one after another
                    while let Ok(newer) = job_recv.try_recv() {
                        job = newer;
                    }
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let bytes = std::fs::read(&job.path).map_err(|e| e.to_string())?;
                        let facts = collect_item_facts(&bytes).map_err(|e| e.to_string())?;
                        Ok(estimate_with_steam_api(&facts, &job.api_key, &mut limiter))
                    }))
                    .unwrap_or_else(|_| Err("the date estimate crashed".to_string()));
                    if result_send.send((job.path, job.api_key, result)).is_err() {
                        return;
                    }
                    job.ctx.request_repaint();
                }
            })
            .expect("Failed to spawn recording date thread");
        Self { jobs, results, cache: HashMap::new(), current: None, state: None }
    }

    /// Starts estimating the recording date of `path` (no-op when it is already shown or cached).
    pub fn request(&mut self, path: &Path, api_key: &str, ctx: &egui::Context) {
        let key = (path.to_path_buf(), api_key.trim().to_string());
        if self.current.as_ref() == Some(&key) && !matches!(self.state, Some(State::Failed(_))) {
            return;
        }
        self.current = Some(key.clone());
        if let Some(cached) = self.cache.get(&key) {
            self.state = Some(Self::state_of(cached.clone()));
            return;
        }
        self.state = Some(State::Pending);
        let _ = self.jobs.send(Job { path: key.0, api_key: key.1, ctx: ctx.clone() });
    }

    fn state_of(result: Result<Option<RecordingDate>, String>) -> State {
        match result {
            Ok(date) => State::Done(date),
            Err(e) => State::Failed(e),
        }
    }

    fn poll(&mut self) {
        while let Ok((path, key, result)) = self.results.try_recv() {
            let id = (path, key);
            if self.current.as_ref() == Some(&id) {
                self.state = Some(Self::state_of(result.clone()));
            }
            // A failed API lookup is not cached as final: it reruns when the demo is opened again
            let api_failed = matches!(&result, Ok(Some(d)) if d.api_note.as_deref().is_some_and(|n| n.contains("rate limited") || n.contains("failed")));
            if !api_failed {
                self.cache.insert(id, result);
            }
        }
    }

    /// Draws the date (right-aligned, with the method in small text below). Call inside a
    /// right-to-left layout at the end of the file name row.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        self.poll();
        let Some(state) = &self.state else { return };
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            match state {
                State::Pending => {
                    ui.label(egui::RichText::new("Estimating date...").weak());
                }
                State::Failed(e) => {
                    ui.label(egui::RichText::new("Date unknown").weak()).on_hover_text(e.as_str());
                }
                State::Done(None) => {
                    ui.label(egui::RichText::new("Date unknown").weak())
                        .on_hover_text("The demo has no items with real item IDs (e.g. only bots or stock items).");
                }
                State::Done(Some(date)) => {
                    let main = ui.label(egui::RichText::new(Self::date_text(date)).strong());
                    let mut method = Self::method_text(date).to_string();
                    if date.api_note.is_some() {
                        method.push_str(" ⚠");
                    }
                    let sub = ui.label(egui::RichText::new(method).small().weak());
                    let hover = Self::hover_text(date);
                    main.on_hover_text(&hover);
                    sub.on_hover_text(&hover);
                }
            }
        });
    }

    fn date_text(date: &RecordingDate) -> String {
        match (date.method, date.precision_days()) {
            (DateMethod::NewestItemId | DateMethod::Extrapolated, _) | (_, None) => {
                format!("{} or later", format_date(date.earliest))
            }
            (_, Some(days)) => format!("{} ±{:.0} d", format_date(date.estimate), days.round().max(1.0)),
        }
    }

    fn method_text(date: &RecordingDate) -> &'static str {
        match date.method {
            DateMethod::ItemDrop => "item drop (Steam API)",
            DateMethod::TradeHold if date.latest.is_none() => "trade hold (1 buyer), lower bound",
            DateMethod::TradeHold => "trade hold",
            DateMethod::NewestItemId => "newest item ID, lower bound",
            DateMethod::Extrapolated => "item ID extrapolation, lower bound",
        }
    }

    fn hover_text(date: &RecordingDate) -> String {
        let mut text = format!("Estimated recording date: {}\n", format_date_time(date.estimate));
        match date.latest {
            Some(latest) => text.push_str(&format!(
                "Window: {} to {}\n",
                format_date_time(date.earliest),
                format_date_time(latest)
            )),
            None => text.push_str(&format!("Not recorded before {}\n", format_date_time(date.earliest))),
        }
        text.push_str(&date.detail);
        if let Some(note) = &date.api_note {
            text.push_str("\n⚠ ");
            text.push_str(note);
        }
        text
    }
}
