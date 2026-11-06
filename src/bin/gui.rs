#![cfg_attr(all(not(debug_assertions), not(algo_cost_learning)), windows_subsystem = "windows")]

use std::collections::{HashMap, HashSet};

use analysis_template::{
    base::cheat_analyser_base::CheatAnalyser,
    chat_view::ChatWindow,
    header_view::HeaderWindow,
    playerlist_view::{PlayerHighlights, PlayerlistWindow, TagHighlight},
    lib::{
        algorithm::{analyse_bytes_with_settings, get_algorithms, is_experimental_algorithm_name, algorithm_tooltip, CheatAlgorithm, Detection},
        general_settings::{format_hex_rgb, parse_hex_rgb, ColorSettings, GeneralSettings, UiScaleSettings},
        live_scan::LiveRescan,
        parameters::{strip_session_only_parameters, Parameter, Parameters},
    },
};
#[cfg(feature = "rcon")]
use analysis_template::lib::live_scan::LiveScanGate;
#[cfg(feature = "rcon")]
use analysis_template::rcon::{
    ensure_vote_algorithm, vote_notifier::VOTE_ALGORITHM, RealTimeScanConfig, RealTimeScanEvent, RealTimeScanner,
};
#[cfg(feature = "tf2bd")]
use analysis_template::{tf2bd::Tf2bdSettings, tf2bd_view::Tf2bdUi};
use eframe::egui;
use itertools::Itertools;
use steamid_ng::SteamID;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    #[cfg(algo_cost_learning)]
    let (log_file, log_file_clone) = {
        let mut log_path = std::env::current_exe().unwrap_or_default();
        log_path.pop();
        let file = log_path.join("gui_debug.log");
        let _ = std::fs::write(&file, "main started\n");
        (Some(file.clone()), Some(file))
    };
    #[cfg(not(algo_cost_learning))]
    let (log_file, log_file_clone): (Option<std::path::PathBuf>, Option<std::path::PathBuf>) = (None, None);

    std::panic::set_hook({
        let log_file = log_file.clone();
        Box::new(move |info| {
            eprintln!("GUI PANIC: {:?}", info);
            if let Some(ref file) = log_file {
                let _ = std::fs::write(file, format!("PANIC: {:#?}\n", info));
            }
            let _ = std::fs::write("gui_panic.log", format!("{:#?}\n", info));
        })
    });

    println!("Initializing eframe...");
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../../assets/app_icon.png"))
        .ok();

    // The window is created before egui's zoom factor is set, so its size is given already scaled
    let startup_scale = Gui::load_ui_scale_percent() as f32 / 100.0;
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([BASE_WINDOW_SIZE.x * startup_scale, BASE_WINDOW_SIZE.y * startup_scale])
        .with_min_inner_size([BASE_WINDOW_SIZE.x * startup_scale, BASE_WINDOW_SIZE.y * startup_scale])
        .with_drag_and_drop(true);

    if let Some(icon) = icon {
        viewport = viewport.with_icon(icon);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    let res = eframe::run_native(
        "Demo Analysis",
        options,
        Box::new(move |cc| {
            install_fallback_fonts(&cc.egui_ctx);
            if let Some(ref file) = log_file_clone {
                let _ = std::fs::write(file, "creating Gui::new()\n");
            }
            println!("Calling Gui::new()...");
            let gui = Gui::new();
            cc.egui_ctx.set_zoom_factor(gui.applied_ui_scale as f32 / 100.0);
            if let Some(ref file) = log_file_clone {
                let _ = std::fs::write(file, "Gui::new() created successfully\n");
            }
            println!("Gui::new() finished");
            Ok(Box::new(gui))
        }),
    );
    if let Err(e) = res {
        eprintln!("GUI eframe error: {:#?}", e);
        if let Some(ref file) = log_file {
            let _ = std::fs::write(file, format!("eframe error: {:#?}\n", e));
        }
        let _ = std::fs::write("gui_error.log", format!("{:#?}\n", e));
    }
}

/// Player names can contain any Unicode (CJK, Cyrillic, Thai, Arabic, emoji, symbols...), but egui's bundled
/// fonts only cover Latin and a few symbols. Append whatever broad-coverage system fonts exist as fallbacks
/// (in order) so egui falls through to them for glyphs the default fonts lack.
fn install_fallback_fonts(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        // Windows
        "C:/Windows/Fonts/msyh.ttc",        // Chinese (Simplified) + Latin/kana
        "C:/Windows/Fonts/msjh.ttc",        // Chinese (Traditional)
        "C:/Windows/Fonts/YuGothM.ttc",     // Japanese
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/malgun.ttf",      // Korean
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/segoeui.ttf",     // Latin ext, Cyrillic, Greek, Arabic, Hebrew
        "C:/Windows/Fonts/LeelawUI.ttf",    // Thai / Lao
        "C:/Windows/Fonts/Nirmala.ttc",     // Indic
        "C:/Windows/Fonts/seguisym.ttf",    // symbols
        "C:/Windows/Fonts/seguiemj.ttf",    // emoji
        "C:/Windows/Fonts/seguihis.ttf",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        // macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ];

    let mut fonts = egui::FontDefinitions::default();
    for (i, path) in CANDIDATES.iter().enumerate() {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let key = format!("fallback_{}", i);
        fonts.font_data.insert(key.clone(), std::sync::Arc::new(egui::FontData::from_owned(bytes)));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push(key.clone());
        }
    }
    ctx.set_fonts(fonts);
}

#[cfg(windows)]
fn copy_file_to_clipboard(path: &std::path::Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;

    let full_path = if path.is_relative() {
        std::env::current_dir()
            .map(|cd| cd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    } else {
        path.to_path_buf()
    };

    let mut wide_path: Vec<u16> = full_path.as_os_str().encode_wide().collect();
    wide_path.push(0); // Null-terminator for string
    wide_path.push(0); // Extra null-terminator for double-null termination

    #[repr(C, packed(1))]
    struct DROPFILES {
        p_files: u32,
        pt_x: i32,
        pt_y: i32,
        f_nc: i32,
        f_wide: i32,
    }

    let dropfiles_size = std::mem::size_of::<DROPFILES>();
    let total_bytes = dropfiles_size + wide_path.len() * 2;

    #[link(name = "user32")]
    extern "system" {
        fn OpenClipboard(hwnd: *mut std::ffi::c_void) -> i32;
        fn CloseClipboard() -> i32;
        fn EmptyClipboard() -> i32;
        fn SetClipboardData(uFormat: u32, hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalAlloc(uFlags: u32, dwBytes: usize) -> *mut std::ffi::c_void;
        fn GlobalLock(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn GlobalUnlock(hMem: *mut std::ffi::c_void) -> i32;
        fn GlobalFree(hMem: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
    }

    const CF_HDROP: u32 = 15;
    const GMEM_MOVEABLE: u32 = 0x0002;

    unsafe {
        let h_global = GlobalAlloc(GMEM_MOVEABLE, total_bytes);
        if h_global.is_null() {
            return Err("Failed to allocate global memory".to_string());
        }

        let ptr = GlobalLock(h_global);
        if ptr.is_null() {
            GlobalFree(h_global);
            return Err("Failed to lock global memory".to_string());
        }

        let df = DROPFILES {
            p_files: dropfiles_size as u32,
            pt_x: 0,
            pt_y: 0,
            f_nc: 0,
            f_wide: 1, // Unicode (UTF-16)
        };

        std::ptr::copy_nonoverlapping(&df as *const _ as *const u8, ptr as *mut u8, dropfiles_size);
        std::ptr::copy_nonoverlapping(
            wide_path.as_ptr() as *const u8,
            (ptr as *mut u8).add(dropfiles_size),
            wide_path.len() * 2,
        );

        GlobalUnlock(h_global);

        let mut opened = false;
        for _ in 0..10 {
            if OpenClipboard(std::ptr::null_mut()) != 0 {
                opened = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        if !opened {
            GlobalFree(h_global);
            return Err("Failed to open clipboard".to_string());
        }

        EmptyClipboard();
        if SetClipboardData(CF_HDROP, h_global).is_null() {
            GlobalFree(h_global);
            CloseClipboard();
            return Err("Failed to set clipboard data".to_string());
        }
        CloseClipboard();
    }

    Ok(())
}

#[cfg(not(windows))]
fn copy_file_to_clipboard(_path: &std::path::Path) -> Result<(), String> {
    Err("Copying file to clipboard is only supported on Windows".to_string())
}


/// Whether a flagged player's Steam account still exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccountStatus {
    Exists,
    Deleted,
    /// The request failed or the response was not recognised; retried on the next analysis.
    Unknown,
}

/// Result of a single Steam profile request.
enum AccountLookup {
    Status(AccountStatus),
    /// Steam answered 429 / "too many requests": the same request has to be retried later.
    RateLimited,
    Failed,
}

/// Queries the public Steam community XML profile. Deleted accounts answer with
/// `<response><error>The specified profile could not be found.</error></response>`,
/// while every existing account (including never set up profiles) returns a `<profile>`.
fn query_steam_account(steam_id: u64) -> AccountLookup {
    let is_ratelimit = |text: &str| {
        let lower = text.to_ascii_lowercase();
        lower.contains("too many requests") || lower.contains("rate limit")
    };
    match analysis_template::util::helpers::request_steam_profile_xml(steam_id) {
        Ok(resp) => match resp.into_string() {
            Ok(body) if body.contains("<profile>") => AccountLookup::Status(AccountStatus::Exists),
            Ok(body)
                if body.contains("<response>")
                    && body.contains("The specified profile could not be found") =>
            {
                AccountLookup::Status(AccountStatus::Deleted)
            }
            Ok(body) if is_ratelimit(&body) => AccountLookup::RateLimited,
            _ => AccountLookup::Failed,
        },
        Err(ureq::Error::Status(code, resp)) => {
            if code == 429 || is_ratelimit(&resp.into_string().unwrap_or_default()) {
                AccountLookup::RateLimited
            } else {
                AccountLookup::Failed
            }
        }
        Err(ureq::Error::Transport(_)) => AccountLookup::Failed,
    }
}

/// Looks up a Steam account, waiting out rate limits. Steam keeps answering 429 while requests
/// continue to arrive, so the backoff blocks the whole checker queue and persists across accounts
/// (`backoff`) until a request succeeds again; one account gives up after a few rate limits so it
/// cannot hold up the rest of the queue indefinitely.
fn check_steam_account(steam_id: u64, backoff: &mut Option<std::time::Duration>) -> AccountStatus {
    const MAX_FAILURES: usize = 3;
    const MAX_RATE_LIMITS: usize = 3;
    let mut rate_limits = 0;
    const MIN_BACKOFF: std::time::Duration = std::time::Duration::from_secs(10);
    const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(300);
    let mut failures = 0;
    loop {
        match query_steam_account(steam_id) {
            AccountLookup::Status(status) => {
                *backoff = None;
                return status;
            }
            AccountLookup::RateLimited => {
                rate_limits += 1;
                if rate_limits >= MAX_RATE_LIMITS {
                    return AccountStatus::Unknown;
                }
                let wait = backoff.map_or(MIN_BACKOFF, |b| (b * 2).min(MAX_BACKOFF));
                *backoff = Some(wait);
                std::thread::sleep(wait);
            }
            AccountLookup::Failed => {
                failures += 1;
                if failures >= MAX_FAILURES {
                    return AccountStatus::Unknown;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
}

/// Maximum number of ticks listed per algorithm in the copied detections.
const COPY_MAX_TICKS: usize = 20;

fn escape_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' | '*' | '_' | '`' | '[' | ']' | '<' | '>' | '#' | '|' | '~' => {
                out.push('\\');
                out.push(c);
            }
            '\n' | '\r' => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

/// Formats detections as a compact markdown list: one block per player (most detections first)
/// with the detection count per algorithm and the ticks of its first `COPY_MAX_TICKS` detections.
/// Players in `reported_ids` get an "ALREADY MARKED" line at the top of their list.
fn format_detections_markdown<'a>(
    detections: &HashMap<u64, Vec<Detection>>,
    reported_ids: &HashSet<u64>,
    player_name: impl Fn(u64) -> Option<&'a str>,
) -> String {
    let mut out = String::new();
    for (&player, dets) in detections
        .iter()
        .filter(|(_, d)| !d.is_empty())
        .sorted_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)))
    {
        let mut per_algorithm: std::collections::BTreeMap<&str, Vec<u32>> = std::collections::BTreeMap::new();
        for d in dets {
            per_algorithm.entry(d.algorithm.as_str()).or_default().push(d.tick);
        }
        let name = player_name(player).filter(|n| !n.is_empty()).unwrap_or("Unknown");
        out.push_str(&format!("**{} - {}:**\n\n", escape_markdown(name), player));
        if reported_ids.contains(&player) {
            out.push_str("* ALREADY MARKED\n");
        }
        for (algorithm, mut ticks) in per_algorithm {
            ticks.sort_unstable();
            let listed = ticks.iter().take(COPY_MAX_TICKS).map(|t| t.to_string()).join(", ");
            // Algorithm names only contain letters, digits, `_` and `/`, which are safe unescaped
            out.push_str(&format!("* {} ({}) - {}\n", algorithm, ticks.len(), listed));
        }
        out.push('\n');
    }
    out.truncate(out.trim_end().len());
    out.push('\n');
    out
}

/// Formats every detection with its full data as pretty JSON, grouped by player (most detections first).
fn format_detections_json<'a>(
    detections: &HashMap<u64, Vec<Detection>>,
    player_name: impl Fn(u64) -> Option<&'a str>,
) -> String {
    let players: Vec<_> = detections
        .iter()
        .filter(|(_, d)| !d.is_empty())
        .sorted_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(b.0)))
        .map(|(&player, dets)| {
            let mut dets = dets.clone();
            dets.sort_by_key(|d| d.tick);
            serde_json::json!({
                "player": player,
                "name": player_name(player).filter(|n| !n.is_empty()).unwrap_or("Unknown"),
                "detections": dets,
            })
        })
        .collect();
    serde_json::to_string_pretty(&players).unwrap_or_else(|_| "[]".to_string())
}

/// Writes the text to `<temp>/demo_analysis_detections/<demo>_detections.<extension>` and returns the path.
/// The file is kept (and overwritten on the next copy of the same demo) so the clipboard
/// reference stays valid after pasting.
fn write_detections_file(demo_stem: &str, extension: &str, contents: &str) -> Result<std::path::PathBuf, String> {
    let safe_stem: String = demo_stem
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ') { c } else { '_' })
        .collect();
    let dir = std::env::temp_dir().join("demo_analysis_detections");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create temp directory: {e}"))?;
    let path = dir.join(format!("{safe_stem}_detections.{extension}"));
    std::fs::write(&path, contents).map_err(|e| format!("Failed to write detections file: {e}"))?;
    Ok(path)
}

/// Whether an algorithm is shown for the parameter search: by its name or the name of one of its parameters.
fn algo_matches_search(search: &str, name: &str, params: Option<&Parameters>) -> bool {
    search.is_empty()
        || name.to_lowercase().contains(search)
        || params.is_some_and(|p| p.keys().any(|k| k != "enabled" && k.to_lowercase().contains(search)))
}

fn rgb_color(rgb: [u8; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn luma(c: egui::Color32) -> f32 {
    (0.299 * c.r() as f32 + 0.587 * c.g() as f32 + 0.114 * c.b() as f32) / 255.0
}

/// Builds visuals from the custom text, background and selection colors. Widget fills and
/// strokes are blended between text and background so every element follows the custom scheme.
fn custom_visuals(text: egui::Color32, bg: egui::Color32, selected: egui::Color32) -> egui::Visuals {
    let is_dark = luma(bg) < 0.5;
    let mut v = if is_dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    let mix = |t: f32| bg.lerp_to_gamma(text, t);
    let strong = text.lerp_to_gamma(if is_dark { egui::Color32::WHITE } else { egui::Color32::BLACK }, 0.5);

    v.override_text_color = None;
    v.panel_fill = bg;
    v.window_fill = bg;
    v.faint_bg_color = mix(0.04);
    v.extreme_bg_color = bg.lerp_to_gamma(if is_dark { egui::Color32::BLACK } else { egui::Color32::WHITE }, 0.5);
    v.code_bg_color = mix(0.1);
    v.window_stroke.color = mix(0.25);
    v.selection.bg_fill = selected;
    // Text on a selected field: a strong tint of the selection color, like egui's defaults
    v.selection.stroke.color = if luma(selected) < 0.5 {
        selected.lerp_to_gamma(egui::Color32::WHITE, 0.75)
    } else {
        selected.lerp_to_gamma(egui::Color32::BLACK, 0.6)
    };

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = bg;
    w.noninteractive.weak_bg_fill = bg;
    w.noninteractive.bg_stroke.color = mix(0.2);
    w.noninteractive.fg_stroke.color = text;
    for (state, fill, stroke, fg) in [
        (&mut w.inactive, 0.15, None, text),
        (&mut w.hovered, 0.25, Some(0.5), strong),
        (&mut w.active, 0.35, Some(0.7), strong),
        (&mut w.open, 0.15, Some(0.3), text),
    ] {
        state.bg_fill = mix(fill);
        state.weak_bg_fill = mix(fill);
        if let Some(t) = stroke {
            state.bg_stroke.color = mix(t);
        }
        state.fg_stroke.color = fg;
    }
    v
}

/// Serialized parameters per algorithm/section, as written to params.json.
type SettingsSnapshot = HashMap<String, HashMap<String, serde_json::Value>>;

const UNSAVED_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 170, 40);
const MAX_RECENT_FILES: usize = 10;
/// Window size (in points) at 100% UI scale, which is also the minimum window size.
const BASE_WINDOW_SIZE: egui::Vec2 = egui::vec2(624.0, 800.0);

struct Gui {
    algos: HashMap<String, bool>,
    params: HashMap<String, Parameters>,
    algo_tooltips: HashMap<String, &'static str>,
    file: Option<std::path::PathBuf>,
    processing: bool,
    detections: HashMap<u64, Vec<Detection>>,
    selected_player: Option<u64>,
    selected_detection: Option<usize>,
    hovered_player: Option<u64>,
    reported_ids: HashSet<u64>,

    copied_demo_feedback: Option<std::time::Instant>,
    copied_detections_feedback: Option<std::time::Instant>,
    replay_feedback: Option<(std::time::Instant, Result<usize, String>)>,
    strip_feedback: Option<(std::time::Instant, Result<usize, String>)>,
    delete_replays_feedback: Option<(std::time::Instant, Result<usize, String>)>,
    #[cfg(feature = "experimental")]
    voice_export_feedback: Option<(std::time::Instant, Result<analysis_template::util::voice::VoiceExportSummary, String>)>,
    show_delete_replays_prompt: bool,
    save_feedback: Option<(std::time::Instant, Result<(), String>)>,
    prev_demo: Option<std::path::PathBuf>,
    next_demo: Option<std::path::PathBuf>,
    /// Error of the last analysis (or demo load), shown as a banner until dismissed.
    analysis_error: Option<String>,
    /// The running analysis was asked to stop; its result is discarded when it returns.
    cancel_requested: bool,
    recent_files: Vec<std::path::PathBuf>,
    param_search: String,
    /// Algorithm whose parameters the cogwheel asked to scroll to
    scroll_to_algo: Option<String>,
    /// Parameters and algorithm states as last saved, to highlight unsaved changes.
    saved_state: std::cell::RefCell<SettingsSnapshot>,
    show_exit_prompt: bool,
    allow_close: bool,

    analyser: Option<CheatAnalyser<'static>>,

    header_window: HeaderWindow,
    playerlist_window: PlayerlistWindow,
    chat_window: ChatWindow,
    #[cfg(feature = "rcon")]
    rt_enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(feature = "rcon")]
    rt_config: std::sync::Arc<std::sync::Mutex<RealTimeScanConfig>>,
    #[cfg(feature = "rcon")]
    rt_algos: std::sync::Arc<std::sync::Mutex<HashMap<String, bool>>>,
    #[cfg(feature = "rcon")]
    rt_params: std::sync::Arc<std::sync::Mutex<HashMap<String, Parameters>>>,
    #[cfg(feature = "rcon")]
    rt_reported_ids: std::sync::Arc<std::sync::Mutex<HashSet<u64>>>,
    /// Players announced in party chat when seen on the server (TF2BD notifications).
    #[cfg(feature = "rcon")]
    #[cfg_attr(not(feature = "tf2bd"), allow(dead_code))]
    rt_tagged_players: std::sync::Arc<std::sync::Mutex<HashMap<u64, String>>>,
    #[cfg(feature = "rcon")]
    rt_rx: std::sync::mpsc::Receiver<RealTimeScanEvent>,
    /// Pressing Analyse while Live is on rescans the whole demo; the live scans wait for it.
    #[cfg(feature = "rcon")]
    rt_gate: std::sync::Arc<LiveScanGate>,
    /// The running analysis is such a rescan.
    #[cfg(feature = "rcon")]
    live_rescan_running: bool,
    /// Live scans that started before the last rescan finished are outdated by it and ignored.
    #[cfg(feature = "rcon")]
    rt_min_scan_epoch: u64,
    #[cfg(feature = "rcon")]
    rt_status_message: Option<String>,
    #[cfg(feature = "rcon")]
    rcon_connected: bool,
    #[cfg(feature = "tf2bd")]
    tf2bd: Tf2bdUi,

    recv: std::sync::mpsc::Receiver<anyhow::Result<CheatAnalyser<'static>>>,
    send: std::sync::mpsc::Sender<anyhow::Result<CheatAnalyser<'static>>>,
    analysis_job_send: std::sync::mpsc::Sender<AnalysisJob>,
    reported_ids_recv: std::sync::mpsc::Receiver<HashSet<u64>>,

    account_status: HashMap<u64, AccountStatus>,
    deleted_accounts: HashSet<u64>,
    existing_accounts: HashSet<u64>,
    account_checker_send: std::sync::mpsc::Sender<u64>,
    /// Lets the account checker thread repaint the UI when a lookup finishes after the analysis.
    account_checker_ctx: std::sync::Arc<std::sync::OnceLock<egui::Context>>,
    shared_reported_ids: std::sync::Arc<std::sync::RwLock<HashSet<u64>>>,
    account_status_recv: std::sync::mpsc::Receiver<(u64, AccountStatus)>,
    applied_colors: Option<ColorSettings>,
    /// UI scale (percent) egui's zoom factor was last set to.
    applied_ui_scale: i32,
    /// Window size (in points) to restore once a changed zoom factor has taken effect.
    pending_window_resize: Option<egui::Vec2>,
}

struct AnalysisJob {
    file: std::path::PathBuf,
    algorithms: Vec<Box<dyn CheatAlgorithm<'static> + Send>>,
    general_settings: GeneralSettings,
    send: std::sync::mpsc::Sender<anyhow::Result<CheatAnalyser<'static>>>,
    repaint_ctx: egui::Context,
    /// Set when the demo is also being scanned live: the analysis then waits for the running live
    /// scan, hands its results to the live scanner and reads the demo itself.
    live_rescan: Option<LiveRescan>,
}


impl Gui {
    fn params_path() -> std::path::PathBuf {
        if let Ok(mut exe_path) = std::env::current_exe() {
            exe_path.pop();
            let p = exe_path.join("params.json");
            return p;
        }
        std::path::PathBuf::from("params.json")
    }

    fn reported_ids_path() -> std::path::PathBuf {
        if let Ok(mut exe_path) = std::env::current_exe() {
            exe_path.pop();
            let p = exe_path.join("reported_ids.txt");
            return p;
        }
        std::path::PathBuf::from("reported_ids.txt")
    }

    fn recent_files_path() -> std::path::PathBuf {
        let mut path = Self::params_path();
        path.set_file_name("recent_demos.txt");
        path
    }

    fn load_recent_files() -> Vec<std::path::PathBuf> {
        std::fs::read_to_string(Self::recent_files_path())
            .map(|content| {
                content
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(std::path::PathBuf::from)
                    .filter(|p| p.is_file())
                    .take(MAX_RECENT_FILES)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn add_recent_file(&mut self, path: &std::path::Path) {
        self.recent_files.retain(|p| p != path);
        self.recent_files.insert(0, path.to_path_buf());
        self.recent_files.truncate(MAX_RECENT_FILES);
        let content = self.recent_files.iter().map(|p| p.to_string_lossy()).join("
");
        let _ = std::fs::write(Self::recent_files_path(), content);
    }

    /// The parameters (with the algorithm enabled states merged in) as they are saved to params.json.
    fn snapshot_data(&self) -> SettingsSnapshot {
        let mut save_data = self.params.clone();
        for (algo_name, &enabled) in &self.algos {
            let p = save_data.entry(algo_name.clone()).or_default();
            p.insert("enabled".to_string(), Parameter::Bool(enabled));
        }
        strip_session_only_parameters(&mut save_data);
        save_data
            .into_iter()
            .map(|(section, params)| {
                let values = params
                    .into_iter()
                    .map(|(key, value)| (key, serde_json::to_value(&value).unwrap_or(serde_json::Value::Null)))
                    .collect();
                (section, values)
            })
            .collect()
    }

    /// (section, key) of every setting that differs from what was last saved.
    fn unsaved_settings(&self) -> HashSet<(String, String)> {
        let current = self.snapshot_data();
        let saved = self.saved_state.borrow();
        let mut dirty = HashSet::new();
        for (section, values) in &current {
            for (key, value) in values {
                if saved.get(section).and_then(|s| s.get(key)) != Some(value) {
                    dirty.insert((section.clone(), key.clone()));
                }
            }
        }
        dirty
    }

    fn save_params(&self) {
        let path = Self::params_path();
        let mut save_data = self.params.clone();
        for (algo_name, &enabled) in &self.algos {
            let p = save_data.entry(algo_name.clone()).or_default();
            p.insert("enabled".to_string(), Parameter::Bool(enabled));
        }
        *self.saved_state.borrow_mut() = self.snapshot_data();

        // Preserve any algorithm entries from the existing params.json that this build
        // doesn't know about (e.g. experimental algorithms when running the stable build).
        // We read the file first and keep keys that aren't present in `save_data`.
        let existing_data = std::fs::read_to_string(&path)
            .ok()
            .or_else(|| std::fs::read_to_string("params.json").ok());
        if let Some(existing) = existing_data {
            if let Ok(existing_params) = serde_json::from_str::<HashMap<String, Parameters>>(&existing) {
                for (algo_name, algo_params) in existing_params {
                    save_data.entry(algo_name).or_insert(algo_params);
                }
            }
        }
        strip_session_only_parameters(&mut save_data);

        if let Ok(json_data) = serde_json::to_vec_pretty(&save_data) {
            let _ = std::fs::write(&path, &json_data);
            if path != std::path::Path::new("params.json") {
                let _ = std::fs::write("params.json", &json_data);
            }
        }
        #[cfg(feature = "rcon")]
        self.sync_real_time_scanning();
    }

    /// Saves one section of the settings to params.json, leaving the other sections as they were
    /// last saved (unsaved changes elsewhere stay unsaved).
    #[cfg_attr(not(feature = "tf2bd"), allow(dead_code))]
    fn save_section(&self, section: &str) {
        let path = Self::params_path();
        let existing = std::fs::read_to_string(&path)
            .ok()
            .or_else(|| std::fs::read_to_string("params.json").ok())
            .and_then(|data| serde_json::from_str::<HashMap<String, Parameters>>(&data).ok());
        let (Some(mut save_data), Some(params)) = (existing, self.params.get(section)) else {
            self.save_params();
            return;
        };
        save_data.insert(section.to_string(), params.clone());
        strip_session_only_parameters(&mut save_data);
        if let Ok(json_data) = serde_json::to_vec_pretty(&save_data) {
            let _ = std::fs::write(&path, &json_data);
            if path != std::path::Path::new("params.json") {
                let _ = std::fs::write("params.json", &json_data);
            }
        }
        if let Some(values) = self.snapshot_data().remove(section) {
            self.saved_state.borrow_mut().insert(section.to_string(), values);
        }
    }

    /// Applies the TF2BD settings and notifications changed in its UI.
    #[cfg(feature = "tf2bd")]
    fn sync_tf2bd(&mut self) {
        if let Some(params) = self.tf2bd.take_settings_changed() {
            self.params.insert(Tf2bdSettings::SECTION.to_string(), params);
            self.save_section(Tf2bdSettings::SECTION);
        }
        #[cfg(feature = "rcon")]
        if let Some(tagged) = self.tf2bd.take_notify_changed() {
            if let Ok(mut shared) = self.rt_tagged_players.lock() {
                *shared = tagged;
            }
        }
    }

    /// The TF2BD highlight of a player, if they are tagged.
    fn player_tag(&self, _steam_id: u64) -> Option<&TagHighlight> {
        #[cfg(feature = "tf2bd")]
        return self.tf2bd.tags().get(&_steam_id);
        #[cfg(not(feature = "tf2bd"))]
        None
    }

    #[cfg(feature = "rcon")]
    fn sync_real_time_scanning(&self) {
        let rt_scan_enabled = match self.params.get("rcon").and_then(|p| p.get("real_time_scanning")) {
            Some(Parameter::Bool(b)) => *b,
            _ => false,
        };
        self.rt_enabled.store(rt_scan_enabled, std::sync::atomic::Ordering::Relaxed);

        let scan_on_death = match self.params.get("rcon").and_then(|p| p.get("scan_on_death")) {
            Some(Parameter::Bool(b)) => *b,
            _ => true,
        };
        let scan_periodic_too = match self.params.get("rcon").and_then(|p| p.get("scan_periodic_too")) {
            Some(Parameter::Bool(b)) => *b,
            _ => false,
        };
        let scan_interval = match self.params.get("rcon").and_then(|p| p.get("scan_interval_seconds")) {
            Some(Parameter::Int(i)) => (*i).max(5) as u64,
            _ => analysis_template::lib::live_scan::DEFAULT_LIVE_SCAN_INTERVAL_SECONDS,
        };
        let port = match self.params.get("rcon").and_then(|p| p.get("port")) {
            Some(Parameter::Int(i)) => *i,
            _ => 27015,
        };
        let password = match self.params.get("rcon").and_then(|p| p.get("password")) {
            Some(Parameter::String(s)) => s.clone(),
            _ => String::new(),
        };

        let party_chat_alerts = match self.params.get("rcon").and_then(|p| p.get("party_chat_alerts")) {
            Some(Parameter::Bool(b)) => *b,
            _ => true,
        };
        let warn_marked_cheaters = match self.params.get("rcon").and_then(|p| p.get("warn_marked_cheaters")) {
            Some(Parameter::Bool(b)) => *b,
            _ => true,
        };

        let vote_notifications = match self.params.get("rcon").and_then(|p| p.get("vote_notifications")) {
            Some(Parameter::Bool(b)) => *b,
            _ => false,
        };
        let ignore_recorder = match self.params.get("rcon").and_then(|p| p.get("ignore_recorder")) {
            Some(Parameter::Bool(b)) => *b,
            _ => true,
        };

        if let Ok(mut cfg) = self.rt_config.lock() {
            cfg.warn_marked_cheaters = warn_marked_cheaters;
            cfg.vote_notifications = vote_notifications;
            cfg.ignore_recorder = ignore_recorder;
            cfg.port = port;
            cfg.password = password;
            cfg.scan_on_death = scan_on_death;
            cfg.scan_periodic_too = scan_periodic_too;
            cfg.scan_interval_seconds = scan_interval;
            cfg.party_chat_alerts = party_chat_alerts;
        }
        if let Ok(mut a) = self.rt_algos.lock() {
            *a = self.algos.clone();
        }
        if let Ok(mut p) = self.rt_params.lock() {
            *p = self.params.clone();
        }
    }

    pub fn new() -> Self {
        let mut params: HashMap<String, Parameters> = HashMap::new();
        let mut algo_tooltips: HashMap<String, &'static str> = HashMap::new();
        for mut a in get_algorithms().drain(..) {
            let tip = a.tooltip();
            if !tip.is_empty() {
                algo_tooltips.insert(a.algorithm_name().to_string(), tip);
            }
            if a.params().is_some() {
                params.insert(a.algorithm_name().to_string(), a.params().cloned().unwrap());
            }
        }

        let params_path = Self::params_path();
        let data_opt = std::fs::read_to_string(&params_path)
            .ok()
            .or_else(|| std::fs::read_to_string("params.json").ok());

        if let Some(data) = data_opt {
            if let Ok(mut saved_params) = serde_json::from_str::<HashMap<String, Parameters>>(&data) {
                // Overrides saved by older builds must not survive a restart
                strip_session_only_parameters(&mut saved_params);
                for saved_algo in saved_params {
                    if let Some(algo) = params.get_mut(&saved_algo.0) {
                        for saved_param in saved_algo.1 {
                            if let Some(param) = algo.get_mut(saved_param.0.as_str()) {
                                match (&*param, saved_param.1) {
                                    (Parameter::String(_), Parameter::Int(i)) => {
                                        *param = Parameter::String(if i <= 0 { String::new() } else { i.to_string() });
                                    }
                                    (Parameter::String(_), Parameter::UInt(u)) => {
                                        *param = Parameter::String(if u == 0 { String::new() } else { u.to_string() });
                                    }
                                    (Parameter::String(_), Parameter::String(s)) => {
                                        *param = Parameter::String(s);
                                    }
                                    (Parameter::UInt(_), Parameter::Int(i)) if i >= 0 => {
                                        *param = Parameter::UInt(i as u64);
                                    }
                                    (Parameter::UInt(_), Parameter::UInt(u)) => {
                                        *param = Parameter::UInt(u);
                                    }
                                    (Parameter::UInt(_), Parameter::String(s)) => {
                                        if let Ok(u) = u64::try_from(&Parameter::String(s)) {
                                            *param = Parameter::UInt(u);
                                        }
                                    }
                                    (_, s) => {
                                        *param = s;
                                    }
                                }
                            }
                        }
                    } else {
                        params.insert(saved_algo.0, saved_algo.1);
                    }
                }
            }
        } else {
            // If params.json did not exist, generate it now!
            let mut save_data = params.clone();
            strip_session_only_parameters(&mut save_data);
            if let Ok(json_data) = serde_json::to_vec_pretty(&save_data) {
                let _ = std::fs::write(&params_path, &json_data);
                if params_path != std::path::Path::new("params.json") {
                    let _ = std::fs::write("params.json", &json_data);
                }
            }
        }

        // Fill in any general settings keys missing from an older params.json
        let gen_params = params.entry("general_settings".to_string()).or_default();
        for (key, value) in GeneralSettings::default()
            .to_parameters()
            .into_iter()
            .chain(ColorSettings::default().to_parameters())
            .chain(UiScaleSettings::default().to_parameters())
        {
            gen_params.entry(key).or_insert(value);
        }
        // Normalize the stored scale (hand-edited or wrongly typed values) to an in-range percentage
        let ui_scale = UiScaleSettings::from_parameters(gen_params).percent;
        gen_params.insert(UiScaleSettings::KEY.to_string(), Parameter::Int(ui_scale));
        gen_params
            .entry(Self::STRIP_DEMOS_KEY.to_string())
            .or_insert(Parameter::Bool(true));
        gen_params
            .entry(Self::COPY_DETAILED_KEY.to_string())
            .or_insert(Parameter::Bool(false));
        gen_params
            .entry(Self::OLD_REPLAY_KEY.to_string())
            .or_insert(Parameter::Bool(false));
        gen_params
            .entry(Self::SHOW_STRIP_KEY.to_string())
            .or_insert(Parameter::Bool(false));
        // Kept in the stable build too, so saving there does not drop the setting.
        gen_params
            .entry(Self::VOICE_GAP_KEY.to_string())
            .or_insert(Parameter::Float(Self::DEFAULT_VOICE_GAP_SECS));
        let tf2_dir =gen_params
            .entry(Self::TF2_DIR_KEY.to_string())
            .or_insert(Parameter::String(String::new()));
        if let Parameter::String(s) = tf2_dir {
            analysis_template::util::helpers::set_tf2_dir_override(Some(std::path::PathBuf::from(s.as_str())));
        }

        let algos =HashMap::from_iter(
            get_algorithms()
                .iter()
                .map(|a| {
                    let name = a.algorithm_name().to_string();
                    let enabled = params
                        .get(&name)
                        .and_then(|p| match p.get("enabled") {
                            Some(Parameter::Bool(b)) => Some(*b),
                            _ => None,
                        })
                        .unwrap_or_else(|| a.default());
                    (name, enabled)
                }),
        );
        let (send, recv) = std::sync::mpsc::channel();

        // Load initially known reported cheater IDs from local file if available
        let reported_path = Self::reported_ids_path();
        let mut initial_reported_ids = HashSet::new();
        if let Ok(content) = std::fs::read_to_string(&reported_path).or_else(|_| std::fs::read_to_string("reported_ids.txt")) {
            for line in content.lines() {
                if let Ok(id) = line.trim().parse::<u64>() {
                    initial_reported_ids.insert(id);
                }
            }
        }

        // Spawn background thread to download reported cheater IDs without blocking GUI
        let (reported_send, reported_ids_recv) = std::sync::mpsc::channel();
        let thread_reported_path = reported_path.clone();
        std::thread::spawn(move || {
            const URL: &str = "https://raw.githubusercontent.com/Nocrex/Tom/refs/heads/main/reported_ids.txt";
            if let Ok(resp) = ureq::get(URL).timeout(std::time::Duration::from_secs(10)).call() {
                if let Ok(body) = resp.into_string() {
                    let _ = std::fs::write(&thread_reported_path, &body);
                    if thread_reported_path != std::path::Path::new("reported_ids.txt") {
                        let _ = std::fs::write("reported_ids.txt", &body);
                    }
                    let mut ids = HashSet::new();
                    for line in body.lines() {
                        if let Ok(id) = line.trim().parse::<u64>() {
                            ids.insert(id);
                        }
                    }
                    let _ = reported_send.send(ids);
                    return;
                }
            }
            if let Ok(content) = std::fs::read_to_string(&thread_reported_path).or_else(|_| std::fs::read_to_string("reported_ids.txt")) {
                let mut ids = HashSet::new();
                for line in content.lines() {
                    if let Ok(id) = line.trim().parse::<u64>() {
                        ids.insert(id);
                    }
                }
                let _ = reported_send.send(ids);
            }
        });

        #[cfg(feature = "tf2bd")]
        let tf2bd = {
            let settings = Tf2bdSettings::from_config(&params);
            params.insert(Tf2bdSettings::SECTION.to_string(), settings.to_parameters());
            Tf2bdUi::new(settings)
        };

        #[cfg(feature = "rcon")]
        let (rt_enabled, rt_config, rt_algos, rt_params, rt_reported_ids, rt_tagged_players, rt_gate, rt_rx) = {
            let rcon_map = params.entry("rcon".to_string()).or_default();
            rcon_map.remove("enable_glow");
            rcon_map.remove("enable_sv_cheats");
            rcon_map.remove("rewind_ticks");
            let port = match rcon_map.get("port") {
                Some(Parameter::Int(p)) => *p,
                _ => {
                    rcon_map.insert("port".to_string(), Parameter::Int(27015));
                    27015
                }
            };
            let password = match rcon_map.get("password") {
                Some(Parameter::String(s)) => s.clone(),
                _ => {
                    rcon_map.insert("password".to_string(), Parameter::String(String::new()));
                    String::new()
                }
            };
            let scan_on_death = match rcon_map.get("scan_on_death") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("scan_on_death".to_string(), Parameter::Bool(true));
                    true
                }
            };
            let scan_periodic_too = match rcon_map.get("scan_periodic_too") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("scan_periodic_too".to_string(), Parameter::Bool(false));
                    false
                }
            };
            let scan_interval = match rcon_map.get("scan_interval_seconds") {
                Some(Parameter::Int(i)) => (*i).max(5) as u64,
                _ => {
                    let default = analysis_template::lib::live_scan::DEFAULT_LIVE_SCAN_INTERVAL_SECONDS;
                    rcon_map.insert("scan_interval_seconds".to_string(), Parameter::Int(default as i32));
                    default
                }
            };
            let party_chat_alerts = match rcon_map.get("party_chat_alerts") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("party_chat_alerts".to_string(), Parameter::Bool(true));
                    true
                }
            };
            let warn_marked_cheaters = match rcon_map.get("warn_marked_cheaters") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("warn_marked_cheaters".to_string(), Parameter::Bool(true));
                    true
                }
            };
            let vote_notifications = match rcon_map.get("vote_notifications") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("vote_notifications".to_string(), Parameter::Bool(false));
                    false
                }
            };
            let ignore_recorder = match rcon_map.get("ignore_recorder") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("ignore_recorder".to_string(), Parameter::Bool(true));
                    true
                }
            };
            let rt_scan_enabled = match rcon_map.get("real_time_scanning") {
                Some(Parameter::Bool(b)) => *b,
                _ => {
                    rcon_map.insert("real_time_scanning".to_string(), Parameter::Bool(false));
                    false
                }
            };

            let enabled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(rt_scan_enabled));
            let config = std::sync::Arc::new(std::sync::Mutex::new(RealTimeScanConfig {
                port,
                password,
                scan_on_death,
                scan_periodic_too,
                scan_interval_seconds: scan_interval,
                party_chat_alerts,
                warn_marked_cheaters,
                vote_notifications,
                ignore_recorder,
            }));
            let algos_arc = std::sync::Arc::new(std::sync::Mutex::new(algos.clone()));
            let params_arc = std::sync::Arc::new(std::sync::Mutex::new(params.clone()));
            let reported_arc = std::sync::Arc::new(std::sync::Mutex::new(initial_reported_ids.clone()));
            let tagged_arc = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
            let gate = std::sync::Arc::new(LiveScanGate::default());
            let (tx, rx) = std::sync::mpsc::channel();

            let scanner = RealTimeScanner::new(
                enabled.clone(),
                config.clone(),
                algos_arc.clone(),
                params_arc.clone(),
                reported_arc.clone(),
                tagged_arc.clone(),
                gate.clone(),
                tx,
            );
            scanner.spawn();

            (enabled, config, algos_arc, params_arc, reported_arc, tagged_arc, gate, rx)
        };

        let (account_status_send, account_status_recv) = std::sync::mpsc::channel();
        let (account_checker_send, account_checker_recv) = std::sync::mpsc::channel::<u64>();
        let shared_reported_ids = std::sync::Arc::new(std::sync::RwLock::new(initial_reported_ids.clone()));
        let account_checker_ctx = std::sync::Arc::new(std::sync::OnceLock::<egui::Context>::new());

        let (analysis_job_send, analysis_job_recv) = std::sync::mpsc::channel::<AnalysisJob>();
        std::thread::Builder::new()
            .name("analysis-runner".to_string())
            .spawn(move || {
                for job in analysis_job_recv {
                    let AnalysisJob {
                        file,
                        algorithms,
                        general_settings,
                        send,
                        repaint_ctx,
                        live_rescan,
                    } = job;
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Some(rescan) = live_rescan {
                            return rescan.run(&file, algorithms, &general_settings);
                        }
                        let file_bytes = std::fs::read(&file)
                            .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", file.display()))?;
                        analyse_bytes_with_settings(&file_bytes, algorithms, &general_settings)
                    }))
                    .unwrap_or_else(|panic| {
                        let msg = panic
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown error".to_string());
                        Err(anyhow::anyhow!("The analysis crashed: {msg}"))
                    });
                    analysis_template::set_detection_listener(None);
                    // A cancelled analysis must not make the next (live) scan abort as well
                    analysis_template::CANCEL_ANALYSIS.store(false, std::sync::atomic::Ordering::Relaxed);
                    let _ = send.send(result);
                    repaint_ctx.request_repaint();
                }
            })
            .expect("Failed to spawn analysis runner thread");

        {
            let status_send = account_status_send.clone();
            let reported = shared_reported_ids.clone();
            let repaint_ctx = account_checker_ctx.clone();
            std::thread::Builder::new()
                .name("account-checker".to_string())
                .spawn(move || {
                    // The listener sends a player once per detection, so the same id arrives many
                    // times. Resolved ids are never looked up again; failed ones (`Some(time)`) only
                    // after a cooldown, otherwise every queued duplicate would trigger a request.
                    let mut checked: HashMap<u64, Option<std::time::Instant>> = HashMap::new();
                    const REQUEST_INTERVAL: std::time::Duration = std::time::Duration::from_millis(300);
                    const UNKNOWN_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(300);
                    let mut last_request: Option<std::time::Instant> = None;
                    let mut backoff: Option<std::time::Duration> = None;
                    for id in account_checker_recv {
                        if let Ok(guard) = reported.read() {
                            if guard.contains(&id) {
                                continue;
                            }
                        }
                        match checked.get(&id) {
                            Some(None) => continue,
                            Some(Some(failed_at)) if failed_at.elapsed() < UNKNOWN_RETRY_AFTER => continue,
                            _ => {}
                        }
                        if let Some(wait) = last_request.and_then(|t| REQUEST_INTERVAL.checked_sub(t.elapsed())) {
                            std::thread::sleep(wait);
                        }
                        let status = check_steam_account(id, &mut backoff);
                        last_request = Some(std::time::Instant::now());
                        checked.insert(id, (status == AccountStatus::Unknown).then(std::time::Instant::now));
                        if status_send.send((id, status)).is_err() {
                            return;
                        }
                        if let Some(ctx) = repaint_ctx.get() {
                            ctx.request_repaint();
                        }
                    }
                })
                .expect("Failed to spawn account checker thread");
        }

        let gui = Self {
            algos,
            params,
            algo_tooltips,
            file: None,
            processing: false,
            detections: HashMap::new(),
            selected_player: None,
            selected_detection: None,
            hovered_player: None,
            reported_ids: initial_reported_ids,
            copied_demo_feedback: None,
            copied_detections_feedback: None,
            replay_feedback: None,
            strip_feedback: None,
            delete_replays_feedback: None,
            #[cfg(feature = "experimental")]
            voice_export_feedback: None,
            show_delete_replays_prompt: false,
            save_feedback: None,
            prev_demo: None,
            next_demo: None,
            analysis_error: None,
            cancel_requested: false,
            recent_files: Self::load_recent_files(),
            param_search: String::new(),
            scroll_to_algo: None,
            saved_state: std::cell::RefCell::new(SettingsSnapshot::new()),
            show_exit_prompt: false,
            allow_close: false,
            analyser: None,
            header_window: HeaderWindow::new(),
            playerlist_window: PlayerlistWindow::new(),
            chat_window: ChatWindow::new(),
            #[cfg(feature = "rcon")]
            rt_enabled,
            #[cfg(feature = "rcon")]
            rt_config,
            #[cfg(feature = "rcon")]
            rt_algos,
            #[cfg(feature = "rcon")]
            rt_params,
            #[cfg(feature = "rcon")]
            rt_reported_ids,
            #[cfg(feature = "rcon")]
            rt_tagged_players,
            #[cfg(feature = "rcon")]
            rt_rx,
            #[cfg(feature = "rcon")]
            rt_gate,
            #[cfg(feature = "rcon")]
            live_rescan_running: false,
            #[cfg(feature = "rcon")]
            rt_min_scan_epoch: 0,
            #[cfg(feature = "rcon")]
            rt_status_message: None,
            #[cfg(feature = "rcon")]
            rcon_connected: false,
            #[cfg(feature = "tf2bd")]
            tf2bd,
            recv,
            send,
            analysis_job_send,
            reported_ids_recv,
            account_status: HashMap::new(),
            deleted_accounts: HashSet::new(),
            existing_accounts: HashSet::new(),
            account_checker_send,
            account_checker_ctx,
            shared_reported_ids,
            account_status_recv,
            applied_colors: None,
            applied_ui_scale: ui_scale,
            pending_window_resize: None,
        };
        *gui.saved_state.borrow_mut() = gui.snapshot_data();
        gui
    }

    fn update_adjacent_demos(&mut self) {
        self.prev_demo = None;
        self.next_demo = None;
        let Some(current_file) = &self.file else {
            return;
        };
        let Some(dir) = current_file.parent() else {
            return;
        };
        let mut demos: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_file() {
                        let path = entry.path();
                        let is_dem = path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("dem"))
                            .unwrap_or(false);
                        if is_dem {
                            let modified = entry
                                .metadata()
                                .and_then(|m| m.modified())
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                            demos.push((path, modified));
                        }
                    }
                }
            }
        }
        demos.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.file_name().cmp(&b.0.file_name())));
        let Some(current_fname) = current_file.file_name() else {
            return;
        };
        if let Some(current_idx) = demos.iter().position(|(p, _)| p.file_name() == Some(current_fname)) {
            if current_idx > 0 {
                self.prev_demo = Some(demos[current_idx - 1].0.clone());
            }
            if current_idx + 1 < demos.len() {
                self.next_demo = Some(demos[current_idx + 1].0.clone());
            }
        }
    }

    fn select_demo(&mut self, path: std::path::PathBuf) {
        self.file = Some(path.clone());
        self.analyser = None;
        self.detections.clear();
        self.selected_player = None;
        self.selected_detection = None;
        self.replay_feedback = None;
        self.strip_feedback = None;
        self.delete_replays_feedback = None;
        #[cfg(feature = "experimental")]
        {
            self.voice_export_feedback = None;
        }
        self.analysis_error = None;
        self.add_recent_file(&path);
        self.update_adjacent_demos();

        // A manually opened demo must not be replaced by results from a live scan
        #[cfg(feature = "rcon")]
        {
            let live_on = matches!(
                self.params.get("rcon").and_then(|p| p.get("real_time_scanning")),
                Some(Parameter::Bool(true))
            );
            if live_on {
                self.params.entry("rcon".to_string()).or_default()
                    .insert("real_time_scanning".to_string(), Parameter::Bool(false));
                self.save_params();
            }
        }
    }



    /// Applies the custom color scheme, or restores the automatic light/dark scheme, when the
    /// color settings change.
    fn apply_color_scheme(&mut self, ctx: &egui::Context) {
        let colors = ColorSettings::from_config(&self.params);
        if self.applied_colors.as_ref() == Some(&colors) {
            return;
        }
        if colors.custom_colors_enabled {
            let visuals = custom_visuals(
                rgb_color(colors.text_color),
                rgb_color(colors.background_color),
                rgb_color(colors.selected_color),
            );
            ctx.set_theme(if visuals.dark_mode { egui::Theme::Dark } else { egui::Theme::Light });
            ctx.set_visuals_of(egui::Theme::Dark, visuals.clone());
            ctx.set_visuals_of(egui::Theme::Light, visuals);
        } else {
            ctx.set_visuals_of(egui::Theme::Dark, egui::Visuals::dark());
            ctx.set_visuals_of(egui::Theme::Light, egui::Visuals::light());
            ctx.set_theme(egui::ThemePreference::System);
        }
        self.applied_colors = Some(colors);
    }

    /// Reads the saved UI scale (percent) from params.json before the window exists.
    fn load_ui_scale_percent() -> i32 {
        std::fs::read_to_string(Self::params_path())
            .ok()
            .or_else(|| std::fs::read_to_string("params.json").ok())
            .and_then(|data| serde_json::from_str::<HashMap<String, Parameters>>(&data).ok())
            .map(|config| UiScaleSettings::from_config(&config).percent)
            .unwrap_or(UiScaleSettings::DEFAULT_PERCENT)
    }

    /// Applies the configured UI scale and keeps the window size matching it.
    ///
    /// The zoom factor only takes effect on the next frame, and viewport commands are converted to
    /// pixels with the zoom factor that is active at that time. So the window is resized one frame
    /// after the zoom change, back to the size (in points) it had before, which grows or shrinks it
    /// physically in proportion to the scale.
    fn apply_ui_scale(&mut self, ctx: &egui::Context) {
        let wanted = UiScaleSettings::from_config(&self.params).percent;
        let (inner_size, fixed_size) = ctx.input(|i| {
            let viewport = i.viewport();
            let fixed = viewport.maximized == Some(true) || viewport.fullscreen == Some(true);
            (viewport.inner_rect.map(|r| r.size()), fixed)
        });

        if let Some(size) = self.pending_window_resize {
            if (ctx.zoom_factor() * 100.0).round() as i32 != self.applied_ui_scale {
                ctx.request_repaint(); // The new zoom factor is not active yet
                return;
            }
            self.pending_window_resize = None;
            // Never below the base size: the minimum size is always the base size in points
            ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(BASE_WINDOW_SIZE));
            if !fixed_size {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size.max(BASE_WINDOW_SIZE)));
            }
            return;
        }

        // Applied once the slider is released, so the control does not move away from under the pointer
        if wanted != self.applied_ui_scale && !ctx.input(|i| i.pointer.any_down()) {
            if let Some(size) = inner_size {
                self.pending_window_resize = Some(size);
            }
            self.applied_ui_scale = wanted;
            ctx.set_zoom_factor(wanted as f32 / 100.0);
        }
    }

    /// Returns the (normal, marked, deleted) player name colors for the active color scheme.
    /// Normal players use the theme's default text color when no custom colors are set.
    fn player_colors(&self, ctx: &egui::Context) -> (Option<egui::Color32>, egui::Color32, egui::Color32) {
        match &self.applied_colors {
            Some(c) if c.custom_colors_enabled => (
                Some(rgb_color(c.normal_player_color)),
                rgb_color(c.marked_player_color),
                rgb_color(c.deleted_player_color),
            ),
            _ => {
                // Dimmed toward the background: darker than the default text in dark mode,
                // lighter in light mode, but still readable.
                let visuals = &ctx.style().visuals;
                let deleted = visuals
                    .widgets
                    .inactive
                    .fg_stroke
                    .color
                    .lerp_to_gamma(visuals.panel_fill, 0.45);
                (None, egui::Color32::from_rgb(235, 70, 70), deleted)
            }
        }
    }

    fn get_player_name(&self, steamid: u64) -> Option<&str> {
        let analyser = self.analyser.as_ref()?;
        analyser.state.get_player_name(steamid)
    }

    fn get_player_slot(&self, steamid: u64) -> Option<u32> {
        let analyser = self.analyser.as_ref()?;
        analyser.state.get_player_slot(steamid)
    }

    const STRIP_DEMOS_KEY: &'static str = "strip_demos_automatically";
    const TF2_DIR_KEY: &'static str = "tf2_dir_override";
    const COPY_DETAILED_KEY: &'static str = "copy_detailed_detections";
    const OLD_REPLAY_KEY: &'static str = "use_old_replay_conversion";
    const SHOW_STRIP_KEY: &'static str = "show_strip_demo_button";
    const VOICE_GAP_KEY: &'static str = "voice_export_max_gap_seconds";
    const DEFAULT_VOICE_GAP_SECS: f32 = 1.0;

    fn copy_detailed_detections(&self) -> bool {
        matches!(
            self.params.get("general_settings").and_then(|g| g.get(Self::COPY_DETAILED_KEY)),
            Some(Parameter::Bool(true))
        )
    }

    /// Longest pause between two recordings of a player that the voice export still joins, in seconds.
    #[cfg(feature = "experimental")]
    fn voice_export_gap_secs(&self) -> f32 {
        match self.params.get("general_settings").and_then(|g| g.get(Self::VOICE_GAP_KEY)) {
            Some(Parameter::Float(f)) if f.is_finite() => f.max(0.0),
            Some(Parameter::Int(i)) => (*i).max(0) as f32,
            Some(Parameter::UInt(u)) => *u as f32,
            _ => Self::DEFAULT_VOICE_GAP_SECS,
        }
    }

    /// Exports the voice chats of the demo to `voicechats/<demo name>` next to the executable.
    #[cfg(feature = "experimental")]
    fn export_voice_chats(&self, demo_path: &std::path::Path) -> Result<analysis_template::util::voice::VoiceExportSummary, String> {
        let mut out_root = Self::params_path();
        out_root.set_file_name(analysis_template::util::voice::VOICE_EXPORT_DIR);
        analysis_template::util::voice::export_voice_chats(demo_path, &out_root, self.voice_export_gap_secs())
    }

    fn show_strip_demo_button(&self) -> bool {
        matches!(
            self.params.get("general_settings").and_then(|g| g.get(Self::SHOW_STRIP_KEY)),
            Some(Parameter::Bool(true))
        )
    }

    const REPLAYS_INDEX_FILE: &'static str = "replays.dmx";
    const EMPTY_REPLAYS_INDEX: &'static str = "\"root\"
{
	\"version\"	\"0\"
}
";

    fn use_old_replay_conversion(&self) -> bool {
        matches!(
            self.params.get("general_settings").and_then(|g| g.get(Self::OLD_REPLAY_KEY)),
            Some(Parameter::Bool(true))
        )
    }

    fn strip_demos_automatically(&self) -> bool {
        matches!(
            self.params.get("general_settings").and_then(|g| g.get(Self::STRIP_DEMOS_KEY)),
            Some(Parameter::Bool(true))
        )
    }

    fn convert_demo_to_replay(&self, demo_path: &std::path::Path) -> Result<usize, String> {
        let replays_dir = analysis_template::util::helpers::find_tf2_replays_dir(demo_path)
            .ok_or_else(|| "Could not locate TF2 replay directory (tf/replay/client/replays)".to_string())?;
        std::fs::create_dir_all(&replays_dir)
            .map_err(|e| format!("Failed to create replays directory: {e}"))?;
        // The replay browser only lists replays when the replays.dmx index exists.
        let index_path = replays_dir.join(Self::REPLAYS_INDEX_FILE);
        if !index_path.exists() {
            std::fs::write(&index_path, Self::EMPTY_REPLAYS_INDEX)
                .map_err(|e| format!("Failed to create replays.dmx: {e}"))?;
        }

        let demo_filename = demo_path
            .file_name()
            .ok_or_else(|| "Invalid demo path".to_string())?
            .to_string_lossy()
            .to_string();

        let target_demo_path = replays_dir.join(&demo_filename);
        if demo_path != target_demo_path && !target_demo_path.exists() {
            if self.strip_demos_automatically() {
                // Strip in memory and write once, so a failure never leaves an unstripped copy behind.
                let data = std::fs::read(demo_path)
                    .map_err(|e| format!("Failed to read demo file: {e}"))?;
                let (stripped, _) = Self::strip_consolecmds_from_bytes(&data)?;
                std::fs::write(&target_demo_path, stripped)
                    .map_err(|e| format!("Failed to write demo to replays folder: {e}"))?;
            } else {
                std::fs::copy(demo_path, &target_demo_path)
                    .map_err(|e| format!("Failed to copy demo to replays folder: {e}"))?;
            }
        }

        if target_demo_path.exists() {
            let _ = analysis_template::util::helpers::sanitize_demo_signon_ticks(&target_demo_path);
        }

        let (map, duration) = if let Some(analyser) = &self.analyser {
            if let Some(h) = &analyser.header {
                (h.map.clone(), h.duration)
            } else {
                Self::read_demo_header_info(demo_path)?
            }
        } else {
            Self::read_demo_header_info(demo_path)?
        };

        let file_time = demo_path
            .metadata()
            .and_then(|m| m.created().or_else(|_| m.modified()))
            .unwrap_or_else(|_| std::time::SystemTime::now());

        let create_date: chrono::DateTime<chrono::Local> = chrono::DateTime::from(file_time);
        use chrono::{Datelike, Timelike};
        let kv_date = (create_date.day() - 1)
            | ((create_date.month() - 1) << 5)
            | ((create_date.year() as u32 - 2009) << 9);
        let kv_time =
            create_date.hour() | (create_date.minute() << 5) | (create_date.second() << 11);

        let edits_dir = replays_dir
            .parent()
            .map(|client_dir| client_dir.join("edits"))
            .unwrap_or_else(|| replays_dir.join("edits"));

        let mut used_handles = std::collections::HashSet::new();
        let allocate_handle = |used: &mut std::collections::HashSet<u32>| -> u32 {
            let mut handle: u32 = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u32) & 0x7FFF_FFFF;
            if handle == 0 {
                handle = 1;
            }
            while replays_dir.join(format!("replay_{handle}.dmx")).exists() || used.contains(&handle) {
                handle = (handle.wrapping_mul(1664525).wrapping_add(1013904223)) & 0x7FFF_FFFF;
                if handle == 0 {
                    handle = 1;
                }
            }
            used.insert(handle);
            handle
        };

        let is_analysed = self.analyser.is_some();
        let suspect_ids: Vec<u64> = if is_analysed {
            self.detections
                .iter()
                .filter(|(_, dets)| !dets.is_empty())
                .map(|(&sid, _)| sid)
                .collect()
        } else {
            Vec::new()
        };

        if suspect_ids.is_empty() {
            let replay_handle = allocate_handle(&mut used_handles);
            let generated = analysis_template::util::helpers::generate_replay_dmx_and_takes(
                replay_handle,
                &demo_filename,
                &map,
                duration,
                kv_date,
                kv_time,
                None,
                &[],
                None,
            );
            let dmx_path = replays_dir.join(format!("replay_{replay_handle}.dmx"));
            std::fs::write(&dmx_path, generated.main_dmx_content)
                .map_err(|e| format!("Failed to write replay DMX: {e}"))?;
            return Ok(1);
        }

        let all_detections: Vec<analysis_template::util::helpers::ReplayDetection> = self
            .detections
            .iter()
            .flat_map(|(&sid, dets)| {
                dets.iter().map(move |d| analysis_template::util::helpers::ReplayDetection {
                    steam_id: sid,
                    tick: d.tick,
                    algorithm: d.algorithm.clone(),
                })
            })
            .collect();

        let mut created_count = 0;
        for suspect_id in suspect_ids {
            let det_slot = self.get_player_slot(suspect_id)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-1".to_string());
            let player_name = self.get_player_name(suspect_id)
                .unwrap_or("unknown")
                .to_string();

            let mut suspect_dets: Vec<&Detection> = self.detections
                .get(&suspect_id)
                .map(|v| v.iter().collect())
                .unwrap_or_default();
            suspect_dets.sort_by_key(|d| d.tick);

            let mut suspect_takes = Vec::new();
            for det in suspect_dets {
                suspect_takes.push(analysis_template::util::helpers::ReplayTakeInput {
                    tick: det.tick,
                    player_name: player_name.clone(),
                    algorithm: det.algorithm.clone(),
                    player_slot: det_slot.clone(),
                });
            }

            let replay_handle = allocate_handle(&mut used_handles);
            let custom_title = format!("{suspect_id} ({demo_filename})");

            let generated = analysis_template::util::helpers::generate_replay_dmx_and_takes_with_detections(
                replay_handle,
                &demo_filename,
                &map,
                duration,
                kv_date,
                kv_time,
                Some(&det_slot),
                &suspect_takes,
                Some(&custom_title),
                &all_detections,
            );

            if !generated.takes.is_empty() {
                std::fs::create_dir_all(&edits_dir)
                    .map_err(|e| format!("Failed to create edits directory: {e}"))?;
            }

            for (take_filename, take_content) in &generated.takes {
                let take_path = edits_dir.join(take_filename);
                std::fs::write(&take_path, take_content)
                    .map_err(|e| format!("Failed to write take DMX ({take_filename}): {e}"))?;
            }

            let dmx_path = replays_dir.join(format!("replay_{replay_handle}.dmx"));
            std::fs::write(&dmx_path, generated.main_dmx_content)
                .map_err(|e| format!("Failed to write replay DMX: {e}"))?;

            created_count += 1;
        }

        Ok(created_count)
    }

    /// Deletes every `.dmx`/`.dem` file in the TF2 replays folder, plus the generated takes
    /// (`.dmx` files in the sibling `edits` folder). The `replays.dmx` index is reset to an empty
    /// one instead of being removed. Returns the number of deleted replay/take files.
    fn delete_all_replays(demo_path: &std::path::Path) -> Result<usize, String> {
        let replays_dir = analysis_template::util::helpers::find_tf2_replays_dir(demo_path)
            .ok_or_else(|| "Could not locate TF2 replay directory (tf/replay/client/replays)".to_string())?;
        let edits_dir = replays_dir.parent().map(|client_dir| client_dir.join("edits"));

        let mut deleted = 0;
        let mut errors = Vec::new();
        let mut purge = |dir: &std::path::Path, exts: &[&str]| {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            for entry in entries.flatten() {
                let path = entry.path();
                let matches = path.is_file()
                    && path
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| exts.iter().any(|x| e.eq_ignore_ascii_case(x)));
                if !matches || path.file_name().is_some_and(|n| n == Self::REPLAYS_INDEX_FILE) {
                    continue;
                }
                match std::fs::remove_file(&path) {
                    Ok(()) => deleted += 1,
                    Err(e) => errors.push(format!("{}: {e}", path.display())),
                }
            }
        };
        purge(&replays_dir, &["dmx", "dem"]);
        if let Some(edits_dir) = &edits_dir {
            purge(edits_dir, &["dmx"]);
        }

        let index_path = replays_dir.join(Self::REPLAYS_INDEX_FILE);
        if let Err(e) = std::fs::create_dir_all(&replays_dir)
            .and_then(|_| std::fs::write(&index_path, Self::EMPTY_REPLAYS_INDEX))
        {
            errors.push(format!("{}: {e}", index_path.display()));
        }

        if errors.is_empty() {
            Ok(deleted)
        } else {
            Err(format!("Deleted {deleted} files, {} failed: {}", errors.len(), errors.join("; ")))
        }
    }

    fn convert_demo_to_replay_old(&self, demo_path: &std::path::Path) -> Result<u32, String> {
        let (map, duration) = if let Some(analyser) = &self.analyser {
            if let Some(h) = &analyser.header {
                (Some(h.map.as_str()), Some(h.duration))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let header_info = match (map, duration) {
            (Some(m), Some(d)) => Some((m, d)),
            _ => None,
        };
        let handle = analysis_template::util::helpers::convert_demo_to_replay(demo_path, header_info)?;
        if self.strip_demos_automatically() {
            if let (Some(replays_dir), Some(name)) = (
                analysis_template::util::helpers::find_tf2_replays_dir(demo_path),
                demo_path.file_name(),
            ) {
                let target = replays_dir.join(name);
                // Only ever strip the copy in the replays folder, never the original.
                if target != demo_path && target.exists() {
                    Self::strip_demo_consolecmds(&target)?;
                }
            }
        }
        Ok(handle)
    }

    fn strip_demo_consolecmds(demo_path: &std::path::Path) -> Result<usize, String> {
        let data = std::fs::read(demo_path)
            .map_err(|e| format!("Failed to read demo file: {e}"))?;

        let (stripped_data, count) = Self::strip_consolecmds_from_bytes(&data)?;

        if count > 0 {
            std::fs::write(demo_path, &stripped_data)
                .map_err(|e| format!("Failed to write stripped demo: {e}"))?;
        }

        Ok(count)
    }

    fn strip_consolecmds_from_bytes(data: &[u8]) -> Result<(Vec<u8>, usize), String> {
        // Lenient: a crash-truncated demo must not fail the strip (and leave an unstripped copy behind).
        analysis_template::util::helpers::strip_consolecmds_lenient(data)
    }

    fn read_demo_header_info(path: &std::path::Path) -> Result<(String, f32), String> {
        use bitbuffer::BitRead;
        use tf_demo_parser::demo::header::Header;
        use tf_demo_parser::Demo;

        let file_bytes = std::fs::read(path).map_err(|e| format!("Failed to read demo: {e}"))?;
        let demo = Demo::new(&file_bytes);
        let mut stream = demo.get_stream();
        let header = Header::read(&mut stream).map_err(|e| format!("Failed to read demo header: {e}"))?;
        Ok((header.map, header.duration))
    }


    fn analyse(&mut self, ctx: &egui::Context) {
        if self.file.is_none() {
            return;
        }
        self.selected_detection = None;
        self.selected_player = None;
        self.update_adjacent_demos();
        let mut algorithms = get_algorithms();
        algorithms.retain(|a| self.algos[a.algorithm_name()]);

        for a in algorithms.iter_mut() {
            if let Some(p) = self.params.get(a.algorithm_name()) {
                if let Some(algo_params) = a.params() {
                    algo_params.clone_from(p);
                }
            }
        }

        let general_settings = GeneralSettings::from_config(&self.params);
        let file = self.file.clone().unwrap();
        let send = self.send.clone();
        analysis_template::reset_progress(1);
        analysis_template::CANCEL_ANALYSIS.store(false, std::sync::atomic::Ordering::Relaxed);
        let _ = self.account_checker_ctx.set(ctx.clone());
        analysis_template::set_detection_listener(Some(self.account_checker_send.clone()));
        self.analysis_error = None;
        self.cancel_requested = false;

        #[cfg(feature = "rcon")]
        let live_rescan = self.begin_live_rescan(&mut algorithms);
        #[cfg(not(feature = "rcon"))]
        let live_rescan = None;

        let repaint_ctx = ctx.clone();
        let _ = self.analysis_job_send.send(AnalysisJob {
            file,
            algorithms,
            general_settings,
            send,
            repaint_ctx,
            live_rescan,
        });
        self.processing = true;
    }

    /// While Live is on, analysing rescans the whole demo for the live scanner too: its scans
    /// wait until the rescan has finished, its detections replace the live scanner's and the next
    /// live scan continues from where the rescan ended.
    #[cfg(feature = "rcon")]
    fn begin_live_rescan(
        &mut self,
        algorithms: &mut Vec<Box<dyn CheatAlgorithm<'static> + Send>>,
    ) -> Option<LiveRescan> {
        if !self.rt_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // Holds the live scans back from now on, also while the job waits in the queue
        let guard = self.rt_gate.begin_rescan();
        // Same algorithms as the live scanner runs, otherwise its next scan would start over
        let votes_enabled_by_user = algorithms.iter().any(|a| a.algorithm_name() == VOTE_ALGORITHM);
        let announces_votes = self.rt_config.lock().map_or(false, |c| c.vote_notifications);
        if announces_votes {
            ensure_vote_algorithm(algorithms);
        }
        self.live_rescan_running = true;
        self.rt_status_message = Some("Scanning: manual rescan, live scans paused".to_string());
        let hidden = (announces_votes && !votes_enabled_by_user).then(|| VOTE_ALGORITHM.to_string());
        Some(LiveRescan::new(guard, hidden))
    }

    pub fn update_ui(&mut self, ctx: &egui::Context) {

        while let Ok(ids) = self.reported_ids_recv.try_recv() {
            #[cfg(feature = "rcon")]
            if let Ok(mut shared) = self.rt_reported_ids.lock() {
                *shared = ids.clone();
            }
            if let Ok(mut shared) = self.shared_reported_ids.write() {
                *shared = ids.clone();
            }
            self.reported_ids = ids;
        }

        while let Ok((id, status)) = self.account_status_recv.try_recv() {
            match status {
                AccountStatus::Exists => {
                    self.existing_accounts.insert(id);
                }
                AccountStatus::Deleted => {
                    self.deleted_accounts.insert(id);
                }
                AccountStatus::Unknown => {}
            }
            self.account_status.insert(id, status);
            ctx.request_repaint();
        }

        #[cfg(feature = "tf2bd")]
        {
            self.tf2bd.poll(ctx);
            self.sync_tf2bd();
        }

        self.apply_color_scheme(ctx);
        self.apply_ui_scale(ctx);
        let (normal_color, marked_color, deleted_color) = self.player_colors(ctx);

        // Before the live scan results: a scan that started after a rescan finished has the
        // rescan's detections and has to be the one shown.
        if self.processing {
            if let Ok(result) = self.recv.try_recv() {
                self.processing = false;
                #[cfg(feature = "rcon")]
                let live_rescan = std::mem::take(&mut self.live_rescan_running);
                if std::mem::take(&mut self.cancel_requested) {
                    // Cancelled: discard whatever came back
                    #[cfg(feature = "rcon")]
                    if live_rescan {
                        self.rt_status_message = Some("Rescan cancelled, live scans resumed".to_string());
                    }
                } else {
                    match result {
                        Ok(mut analyser) => {
                            analyser.algorithms.clear();
                            analyser.state.user_cmds.clear();
                            analyser.state.user_cmds.shrink_to_fit();
                            analyser.state.console_cmds.clear();
                            analyser.state.console_cmds.shrink_to_fit();
                            self.detections.clear();
                            for det in &analyser.detections {
                                self.detections.entry(det.player).or_default().push(det.clone());
                            }
                            analyser.print_detection_summary();
                            #[cfg(feature = "rcon")]
                            if live_rescan {
                                // The live scanner continues from this rescan; its scans that
                                // started before the rescan finished are outdated by it.
                                self.rt_min_scan_epoch = self.rt_gate.epoch();
                                let now = chrono::Local::now().format("%H:%M:%S");
                                self.rt_status_message = Some(format!(
                                    "Scanned {now}: manual rescan ({} total)",
                                    analyser.detections.len()
                                ));
                            }
                            self.analyser = Some(analyser);
                        }
                        Err(e) => {
                            #[cfg(feature = "rcon")]
                            if live_rescan {
                                self.rt_status_message = Some("Rescan failed, live scans resumed".to_string());
                            }
                            self.analyser = None;
                            self.detections.clear();
                            self.analysis_error = Some(format!("{e:#}"));
                        }
                    }
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }

        #[cfg(feature = "rcon")]
        {
            while let Ok(event) = self.rt_rx.try_recv() {
                match event {
                    RealTimeScanEvent::DemoDetected { demo_path, recorder_name } => {
                        // Select the latest demo right away instead of when the first scan finishes
                        if !self.processing && self.file.as_ref() != Some(&demo_path) {
                            self.file = Some(demo_path);
                            self.analyser = None;
                            self.detections.clear();
                            self.selected_player = None;
                            self.selected_detection = None;
                            self.analysis_error = None;
                        }
                        let rec = recorder_name.as_deref().unwrap_or("Unknown");
                        self.rt_status_message = Some(format!("Recording: {rec}"));
                    }
                    RealTimeScanEvent::DeathDetected { victim: _ } => {
                        self.rt_status_message = Some("Death detected, scanning...".to_string());
                    }
                    RealTimeScanEvent::RecordingStopped(_) => {
                        self.rt_status_message = Some("Recording stopped, waiting for a new demo".to_string());
                    }
                    RealTimeScanEvent::ScanStarted(_) => {
                        self.rt_status_message = Some("Scanning...".to_string());
                    }
                    RealTimeScanEvent::ScanFinished { demo_path, mut analyser, new_notifications, alerts_sent, epoch } => {
                        if epoch < self.rt_min_scan_epoch {
                            // Started before the last manual rescan finished, which has all of this and more
                            continue;
                        }
                        let count = new_notifications.len();
                        let total_dets = analyser.detections.len();
                        analyser.algorithms.clear();
                        analyser.state.user_cmds.clear();
                        analyser.state.user_cmds.shrink_to_fit();
                        analyser.state.console_cmds.clear();
                        analyser.state.console_cmds.shrink_to_fit();
                        // The scanner keeps the detections of the earlier scans of this demo, so
                        // replacing the results here drops nothing found before.
                        self.file = Some(demo_path);
                        self.analyser = Some(analyser);
                        self.detections.clear();
                        if let Some(a) = self.analyser.as_ref() {
                            for det in &a.detections {
                                self.detections.entry(det.player).or_default().push(det.clone());
                            }
                        }
                        // Live scans run without the detection listener; queue the flagged players
                        // for the account check here (already resolved ones are skipped).
                        let _ = self.account_checker_ctx.set(ctx.clone());
                        for &player in self.detections.keys().filter(|&&p| p != 0) {
                            let _ = self.account_checker_send.send(player);
                        }
                        let now = chrono::Local::now().format("%H:%M:%S");
                        self.rt_status_message = if count > 0 {
                            if alerts_sent {
                                Some(format!("Scanned {now}: {count} new alert(s) sent ({total_dets} total)"))
                            } else {
                                Some(format!("Scanned {now}: {count} new found ({total_dets} total)"))
                            }
                        } else {
                            Some(format!("Scanned {now}: nothing new ({total_dets} total)"))
                        };
                    }
                    RealTimeScanEvent::RconStatus(s) => {
                        self.rcon_connected = s.starts_with("Connected");
                    }
                    RealTimeScanEvent::MarkedCheatersOnServer(names) => {
                        let now = chrono::Local::now().format("%H:%M:%S");
                        let shown = match names.as_slice() {
                            [] => String::new(),
                            [only] => only.clone(),
                            [first, rest @ ..] => format!("{first} +{}", rest.len()),
                        };
                        self.rt_status_message = Some(format!("Marked on server {now}: {shown}"));
                    }
                    RealTimeScanEvent::RconConnected(conn) => {
                        self.rcon_connected = conn;
                    }
                }
                ctx.request_repaint();
            }

            let is_live_active = self.params.get("rcon")
                .and_then(|p| p.get("real_time_scanning"))
                .map_or(false, |p| match p {
                    Parameter::Bool(b) => *b,
                    _ => false,
                });
            if is_live_active {
                ctx.request_repaint_after(std::time::Duration::from_millis(500));
            }
        }

        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close && !self.unsaved_settings().is_empty() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.show_exit_prompt = true;
        }

        #[allow(unused_mut)]
        let mut has_open_subwindow =
            self.header_window.is_open || self.playerlist_window.is_open || self.chat_window.is_open;
        #[cfg(feature = "tf2bd")]
        {
            has_open_subwindow |= self.tf2bd.any_modal_open();
        }

        if !has_open_subwindow && !self.processing && !ctx.input(|i| i.raw.hovered_files.is_empty()) {
            egui::Window::new("Hover")
                .movable(false)
                .title_bar(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, &[0.0, 0.0])
                .show(ctx, |ui| {
                    ui.heading("Drop to analyze");
                });
        }
        let search = self.param_search.trim().to_lowercase();
        // The stable build has few algorithms: the search only filters the parameters there
        let algo_search = if self.algos.keys().any(|name| is_experimental_algorithm_name(name)) {
            search.clone()
        } else {
            String::new()
        };
        let dirty = self.unsaved_settings();
        let has_unsaved = !dirty.is_empty();
        // Setting label, in the unsaved color when it differs from params.json
        let setting_label = |section: &str, key: &str, text: &str| -> egui::RichText {
            let text = egui::RichText::new(text);
            if dirty.contains(&(section.to_string(), key.to_string())) {
                text.color(UNSAVED_COLOR)
            } else {
                text
            }
        };
        // Whether a setting is shown with the current search
        let search_matches = |texts: &[&str]| -> bool {
            search.is_empty() || texts.iter().any(|t| t.to_lowercase().contains(&search))
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            if has_open_subwindow {
                ui.disable();
            }
            ui.horizontal(|ui|{
                let mut algo_changed = false;
                let mut goto_algo: Option<String> = None;
                // Fixed (45% of the minimum window height): only the results area below grows with the window
                let top_section_height = 360.0;
                ui.vertical(|ui|{
                    ui.set_width(200.0);
                    ui.heading("Algorithms");
                    egui::ScrollArea::vertical()
                        .id_salt("algo_list")
                        .auto_shrink([false, false])
                        .max_height(top_section_height - 30.0)
                        .min_scrolled_height(top_section_height - 30.0)
                        .show(ui, |ui| {
                        for mut algo in self
                            .algos
                            .iter_mut()
                            .filter(|(name, _)| !is_experimental_algorithm_name(name))
                            .filter(|(name, _)| algo_matches_search(&algo_search, name, self.params.get(*name)))
                            .sorted_by_key(|a| a.0)
                        {
                            let has_params = self.params.get(algo.0).map_or(false, |p| p.keys().any(|k| k != "enabled"));
                            let enabled = *algo.1;
                            ui.horizontal(|ui| {
                                let mut cb = ui.checkbox(&mut algo.1, algo.0);
                                if let Some(tip) = self.algo_tooltips.get(algo.0).copied().or_else(|| algorithm_tooltip(algo.0)) {
                                    cb = cb.on_hover_text(tip);
                                }
                                if cb.changed() {
                                    algo_changed = true;
                                }
                                if enabled && has_params && ui.small_button("⚙").on_hover_text("Go to parameters").clicked() {
                                    goto_algo = Some(algo.0.clone());
                                }
                            });
                        }
                        let has_experimental = self.algos.iter().any(|(name, _)| is_experimental_algorithm_name(name));
                        if has_experimental {
                            ui.separator();
                            for mut algo in self
                                .algos
                                .iter_mut()
                                .filter(|(name, _)| is_experimental_algorithm_name(name))
                                .filter(|(name, _)| algo_matches_search(&algo_search, name, self.params.get(*name)))
                                .sorted_by_key(|a| a.0)
                            {
                                let has_params = self.params.get(algo.0).map_or(false, |p| p.keys().any(|k| k != "enabled"));
                                let enabled = *algo.1;
                                ui.horizontal(|ui| {
                                    let mut cb = ui.checkbox(&mut algo.1, algo.0);
                                    if let Some(tip) = self.algo_tooltips.get(algo.0).copied().or_else(|| algorithm_tooltip(algo.0)) {
                                        cb = cb.on_hover_text(tip);
                                    }
                                    if cb.changed() {
                                        algo_changed = true;
                                    }
                                    if enabled && has_params && ui.small_button("⚙").on_hover_text("Go to parameters").clicked() {
                                        goto_algo = Some(algo.0.clone());
                                    }
                                });
                            }
                        }
                    });
                });
                if let Some(name) = goto_algo {
                    self.param_search.clear();
                    self.scroll_to_algo = Some(name);
                }
                if algo_changed {
                    self.save_params();
                }
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);
                ui.vertical(|ui|{
                        ui.horizontal(|ui|{
                            ui.heading("Parameters");
                            let save_btn = if has_unsaved {
                                egui::Button::new(egui::RichText::new("Save *").color(UNSAVED_COLOR))
                            } else {
                                egui::Button::new("Save")
                            };
                            if ui.add(save_btn).on_hover_text(if has_unsaved {
                                "Unsaved changes (highlighted) - click to save to params.json"
                            } else {
                                "All settings are saved"
                            }).clicked(){
                                self.save_params();
                            }
                            // Right-aligned to the window edge (the scroll content can be wider than the window)
                            let right_edge = ui.ctx().screen_rect().max.x - 24.0;
                            let width = (right_edge - ui.cursor().min.x).max(60.0);
                            ui.allocate_ui_with_layout(
                                egui::vec2(width, ui.spacing().interact_size.y),
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.param_search)
                                            .hint_text("Search...")
                                            .desired_width(130.0),
                                    );
                                    if !self.param_search.is_empty() && ui.small_button("✕").on_hover_text("Clear search").clicked() {
                                        self.param_search.clear();
                                    }
                                },
                            );
                        });
                        ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("param_list")
                    .auto_shrink([false, false])
                    .max_height(top_section_height - 30.0)
                    .min_scrolled_height(top_section_height - 30.0)
                    .show(ui, |ui|{
                    ui.vertical(|ui|{
                        if let Some(gen_params) = self.params.get_mut("general_settings") {
                            let gs = "general_settings";
                            let general_texts = [
                                "general settings",
                                "global sliding window enabled",
                                "sliding window ticks",
                                "sliding window detections",
                                "per algorithm",
                                "strip demos automatically",
                                "copy detailed detections",
                                "use old replay conversion",
                                "show strip demo button",
                                "tf2 folder override",
                                "ui scale",
                                "zoom",
                                "custom colors",
                                "color",
                            ];
                            if search_matches(&general_texts) {
                            ui.add_space(10.0);
                            ui.heading("General Settings");
                            ui.separator();
                            if let Some(Parameter::Bool(enabled)) = gen_params.get_mut("global_sliding_window_enabled") {
                                if search_matches(&["global sliding window enabled"]) {
                                    ui.checkbox(enabled, setting_label(gs, "global_sliding_window_enabled", "Global sliding window enabled"));
                                }
                            }
                            if let Some(Parameter::Int(ticks)) = gen_params.get_mut("sliding_window_ticks") {
                                if search_matches(&["sliding window ticks"]) {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::DragValue::new(ticks).speed(100).range(0..=1_000_000));
                                        ui.label(setting_label(gs, "sliding_window_ticks", "Sliding window ticks"));
                                    });
                                }
                            }
                            if let Some(Parameter::Int(dets)) = gen_params.get_mut("sliding_window_detections") {
                                if search_matches(&["sliding window detections"]) {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::DragValue::new(dets).speed(1).range(1..=100));
                                        ui.label(setting_label(gs, "sliding_window_detections", "Sliding window detections"));
                                    });
                                }
                            }
                            if let Some(Parameter::Bool(per_algo)) = gen_params.get_mut("per_algorithm") {
                                if search_matches(&["per algorithm"]) {
                                    ui.checkbox(per_algo, setting_label(gs, "per_algorithm", "per algorithm"));
                                }
                            }
                            if let Some(Parameter::Bool(strip)) = gen_params.get_mut(Self::STRIP_DEMOS_KEY) {
                                if search_matches(&["strip demos automatically"]) {
                                    ui.checkbox(strip, setting_label(gs, Self::STRIP_DEMOS_KEY, "Strip demos automatically"))
                                        .on_hover_text("Strip consolecmds from the copy placed in the replays folder when creating a replay. The original demo is not modified.");
                                }
                            }
                            if let Some(Parameter::Bool(detailed)) = gen_params.get_mut(Self::COPY_DETAILED_KEY) {
                                if search_matches(&["copy detailed detections"]) {
                                    ui.checkbox(detailed, setting_label(gs, Self::COPY_DETAILED_KEY, "Copy detailed detections"))
                                        .on_hover_text("When enabled, Copy detections copies a .json file with every detection and its details instead of the short markdown summary.");
                                }
                            }
                            if let Some(Parameter::Bool(old_replay)) = gen_params.get_mut(Self::OLD_REPLAY_KEY) {
                                if search_matches(&["use old replay conversion"]) {
                                    ui.checkbox(old_replay, setting_label(gs, Self::OLD_REPLAY_KEY, "Use old replay conversion"))
                                        .on_hover_text("Convert to replay uses the basic conversion without takes or suspect tracking.");
                                }
                            }
                            if let Some(Parameter::Bool(show_strip)) = gen_params.get_mut(Self::SHOW_STRIP_KEY) {
                                if search_matches(&["show strip demo button"]) {
                                    ui.checkbox(show_strip, setting_label(gs, Self::SHOW_STRIP_KEY, "Show Strip demo button"))
                                        .on_hover_text("Show the Strip demo button, which removes all consolecmds from the demo file itself.");
                                }
                            }
                            #[cfg(feature = "experimental")]
                            if let Some(Parameter::Float(gap)) = gen_params.get_mut(Self::VOICE_GAP_KEY) {
                                if search_matches(&["voice export gap", "voice chat"]) {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::DragValue::new(gap).speed(0.05).range(0.0..=60.0).suffix(" s"))
                                            .on_hover_text("Recordings of the same player with a pause of at most this long are joined into one file when exporting voice chats.");
                                        ui.label(setting_label(gs, Self::VOICE_GAP_KEY, "Voice export gap"));
                                    });
                                }
                            }
                            if let Some(Parameter::String(dir)) = gen_params.get_mut(Self::TF2_DIR_KEY) {
                                if search_matches(&["tf2 folder override"]) {
                                    ui.horizontal(|ui| {
                                        let tf2_dir_unsaved = dirty.contains(&(gs.to_string(), Self::TF2_DIR_KEY.to_string()));
                                        let mut edit = egui::TextEdit::singleline(dir)
                                            .desired_width(190.0)
                                            .hint_text("TF2 folder (auto-detect)");
                                        if tf2_dir_unsaved {
                                            edit = edit.text_color(UNSAVED_COLOR);
                                        }
                                        let mut changed = ui
                                            .add(edit)
                                            .on_hover_text("Manually select the Team Fortress 2 'tf' (or 'tf/demos') folder. Leave empty to auto-detect.")
                                            .changed();
                                        if ui.button("Browse...").clicked() {
                                            let mut dialog = rfd::FileDialog::new();
                                            if std::path::Path::new(dir.as_str()).is_dir() {
                                                dialog = dialog.set_directory(dir.as_str());
                                            }
                                            if let Some(path) = dialog.pick_folder() {
                                                *dir = path.to_string_lossy().into_owned();
                                                changed = true;
                                            }
                                        }
                                        if ui.add_enabled(!dir.is_empty(), egui::Button::new("Clear")).clicked() {
                                            dir.clear();
                                            changed = true;
                                        }
                                        if changed {
                                            analysis_template::util::helpers::set_tf2_dir_override(Some(
                                                std::path::PathBuf::from(dir.as_str()),
                                            ));
                                        }
                                    });
                                }
                            }
                            if let Some(Parameter::Int(percent)) = gen_params.get_mut(UiScaleSettings::KEY) {
                                if search_matches(&["ui scale", "zoom"]) {
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Slider::new(percent, UiScaleSettings::MIN_PERCENT..=UiScaleSettings::MAX_PERCENT)
                                                .suffix("%")
                                                .step_by(1.0),
                                        )
                                        .on_hover_text("Scales the whole interface and the window size. Applied when the slider is released.");
                                        if ui
                                            .add_enabled(*percent != UiScaleSettings::DEFAULT_PERCENT, egui::Button::new("Default").small())
                                            .on_hover_text(format!("Reset to default ({}%)", UiScaleSettings::DEFAULT_PERCENT))
                                            .clicked()
                                        {
                                            *percent = UiScaleSettings::DEFAULT_PERCENT;
                                        }
                                        ui.label(setting_label(gs, UiScaleSettings::KEY, "UI scale"));
                                    });
                                }
                            }
                            if search_matches(&["custom colors", "color"]) {
                                ui.add_space(5.0);
                                let mut custom_colors = false;
                                if let Some(Parameter::Bool(custom)) = gen_params.get_mut(ColorSettings::ENABLED_KEY) {
                                    ui.checkbox(custom, setting_label(gs, ColorSettings::ENABLED_KEY, "Custom colors (disables automatic light/dark mode)"));
                                    custom_colors = *custom;
                                }
                                let defaults = ColorSettings::default();
                                ui.add_enabled_ui(custom_colors, |ui| {
                                    for (key, label, default) in [
                                        (ColorSettings::TEXT_KEY, "Text / element color", defaults.text_color),
                                        (ColorSettings::BACKGROUND_KEY, "Background color", defaults.background_color),
                                        (ColorSettings::SELECTED_KEY, "Selected field color", defaults.selected_color),
                                        (ColorSettings::NORMAL_KEY, "Normal players color", defaults.normal_player_color),
                                        (ColorSettings::MARKED_KEY, "Marked players color", defaults.marked_player_color),
                                        (ColorSettings::DELETED_KEY, "Deleted players color", defaults.deleted_player_color),
                                    ] {
                                        if let Some(Parameter::String(hex)) = gen_params.get_mut(key) {
                                            ui.horizontal(|ui| {
                                                let mut rgb = parse_hex_rgb(hex).unwrap_or(default);
                                                if ui.color_edit_button_srgb(&mut rgb).changed() {
                                                    *hex = format_hex_rgb(rgb);
                                                }
                                                ui.add(egui::TextEdit::singleline(hex).desired_width(70.0).char_limit(7));
                                                let default_hex = format_hex_rgb(default);
                                                let is_default = parse_hex_rgb(hex) == Some(default);
                                                if ui
                                                    .add_enabled(!is_default, egui::Button::new("Default").small())
                                                    .on_hover_text(format!("Reset to default ({default_hex})"))
                                                    .clicked()
                                                {
                                                    *hex = default_hex;
                                                }
                                                ui.label(setting_label(gs, key, label));
                                            });
                                        }
                                    }
                                });
                            }
                            }
                        }
                        #[cfg(feature = "rcon")]
                        {
                            let mut rcon_changed = false;
                            let rcon_visible = search_matches(&[
                                "rcon settings", "rcon port", "rcon password", "scan on death", "also scan periodically",
                                "scan interval", "party chat alerts", "warn about marked cheaters on server",
                                "vote notifications", "ignore recorder",
                            ]);
                            if let Some(rcon_params) = self.params.get_mut("rcon").filter(|_| rcon_visible) {
                                ui.add_space(10.0);
                                ui.heading("RCON Settings");
                                ui.separator();

                                let mut port = match rcon_params.get("port") {
                                    Some(Parameter::Int(p)) => *p,
                                    _ => 27015,
                                };
                                ui.horizontal(|ui| {
                                    ui.label("RCON Port:");
                                    if ui.add(egui::DragValue::new(&mut port).range(1..=65535)).changed() {
                                        rcon_params.insert("port".to_string(), Parameter::Int(port));
                                        rcon_changed = true;
                                    }
                                });

                                let mut password = match rcon_params.get("password") {
                                    Some(Parameter::String(p)) => p.clone(),
                                    _ => String::new(),
                                };
                                ui.horizontal(|ui| {
                                    ui.label("RCON Password:");
                                    if ui.add(egui::TextEdit::singleline(&mut password).password(true).hint_text("rcon_password")).changed() {
                                        rcon_params.insert("password".to_string(), Parameter::String(password));
                                        rcon_changed = true;
                                    }
                                });

                                let mut scan_on_death = match rcon_params.get("scan_on_death") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => true,
                                };
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut scan_on_death, "Scan on death")
                                        .on_hover_text("Automatically run demo scan every time the recorder dies (to avoid lag during gameplay)")
                                        .changed()
                                    {
                                        rcon_params.insert("scan_on_death".to_string(), Parameter::Bool(scan_on_death));
                                        rcon_changed = true;
                                    }
                                });

                                let mut scan_periodic_too = match rcon_params.get("scan_periodic_too") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => false,
                                };
                                ui.horizontal(|ui| {
                                    if ui.add_enabled(scan_on_death, egui::Checkbox::new(&mut scan_periodic_too, "Also scan periodically"))
                                        .on_hover_text("With 'scan on death', also scan every N seconds. Whichever happens first triggers the scan (a scan on death restarts the timer)")
                                        .changed()
                                    {
                                        rcon_params.insert("scan_periodic_too".to_string(), Parameter::Bool(scan_periodic_too));
                                        rcon_changed = true;
                                    }
                                });

                                let periodic_active = !scan_on_death || scan_periodic_too;
                                let mut interval = match rcon_params.get("scan_interval_seconds") {
                                    Some(Parameter::Int(i)) => *i,
                                    _ => analysis_template::lib::live_scan::DEFAULT_LIVE_SCAN_INTERVAL_SECONDS as i32,
                                };
                                ui.horizontal(|ui| {
                                    let drag = ui.add_enabled(periodic_active, egui::DragValue::new(&mut interval).range(5..=3600).speed(1));
                                    let label = ui.label("Scan interval (seconds)");
                                    if periodic_active {
                                        label.on_hover_text("Interval in seconds between periodic scans");
                                    } else {
                                        label.on_hover_text("Disabled because 'scan on death' is active without periodic scanning");
                                    }
                                    if drag.changed() {
                                        rcon_params.insert("scan_interval_seconds".to_string(), Parameter::Int(interval));
                                        rcon_changed = true;
                                    }
                                });

                                let mut party_chat_alerts = match rcon_params.get("party_chat_alerts") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => true,
                                };
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut party_chat_alerts, "Party chat alerts")
                                        .on_hover_text("Send cheat detection notifications into the TF2 party chat via RCON")
                                        .changed()
                                    {
                                        rcon_params.insert("party_chat_alerts".to_string(), Parameter::Bool(party_chat_alerts));
                                        rcon_changed = true;
                                    }
                                });

                                let mut warn_marked_cheaters = match rcon_params.get("warn_marked_cheaters") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => true,
                                };
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut warn_marked_cheaters, "Warn about marked cheaters on server")
                                        .on_hover_text("Periodically run 'status' and warn in the TF2 party chat when a player from the marked cheaters list is on the server")
                                        .changed()
                                    {
                                        rcon_params.insert("warn_marked_cheaters".to_string(), Parameter::Bool(warn_marked_cheaters));
                                        rcon_changed = true;
                                    }
                                });

                                let mut vote_notifications = match rcon_params.get("vote_notifications") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => false,
                                };
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut vote_notifications, "Vote notifications")
                                        .on_hover_text("Announce each votekick in the TF2 party chat once voting has finished: caller, target and reason with the yes/no counts, plus one line each of who voted yes and no (long lists are cut off). At most 3 messages per votekick; votes from before the scanner started watching the demo are skipped")
                                        .changed()
                                    {
                                        rcon_params.insert("vote_notifications".to_string(), Parameter::Bool(vote_notifications));
                                        rcon_changed = true;
                                    }
                                });

                                let mut ignore_recorder = match rcon_params.get("ignore_recorder") {
                                    Some(Parameter::Bool(b)) => *b,
                                    _ => true,
                                };
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut ignore_recorder, "Ignore recorder")
                                        .on_hover_text("Don't send notifications about yourself (the recorder): detections, marked cheater warnings and TF2BD playerlist warnings. The recorder is the account logged in to Steam, or the player named like the demo's recorder")
                                        .changed()
                                    {
                                        rcon_params.insert("ignore_recorder".to_string(), Parameter::Bool(ignore_recorder));
                                        rcon_changed = true;
                                    }
                                });
                            }

                            if rcon_changed {
                                self.save_params();
                            }
                        }
                        #[cfg(feature = "tf2bd")]
                        if search_matches(&["tf2bd", "tf2 bot detector", "playerlists", "attributes", "mark player"]) {
                            self.tf2bd.settings_ui(ui);
                        }
                        for (name, params) in self.params.iter_mut().sorted_by_key(|a|a.0) {
                            if !self.algos.get(name).copied().unwrap_or(false) || params.iter().all(|(k, _)| k == "enabled") {
                                continue;
                            }
                            // With a search, an algorithm shows all its parameters if its name matches,
                            // otherwise only the matching ones
                            let name_matches = search.is_empty() || name.to_lowercase().contains(&search);
                            if !name_matches && !params.keys().any(|k| k != "enabled" && k.to_lowercase().contains(&search)) {
                                continue;
                            }
                            ui.add_space(10.0);
                            let mut heading = ui.heading(name);
                            if self.scroll_to_algo.as_deref() == Some(name.as_str()) {
                                heading.scroll_to_me(Some(egui::Align::TOP));
                                self.scroll_to_algo = None;
                            }
                            if let Some(tip) = self.algo_tooltips.get(name).copied().or_else(|| algorithm_tooltip(name)) {
                                heading = heading.on_hover_text(tip);
                            }
                            ui.separator();
                            for param in params.iter_mut().sorted_by_key(|p|p.0){
                                if param.0 == "enabled" {
                                    continue;
                                }
                                if !name_matches && !param.0.to_lowercase().contains(&search) {
                                    continue;
                                }
                                let label = setting_label(name, param.0, param.0);
                                if param.0 == "recorder_id" || param.0 == "target_player" {
                                    match param.1 {
                                        Parameter::Int(i) => *param.1 = Parameter::String(if *i <= 0 { String::new() } else { i.to_string() }),
                                        Parameter::UInt(u) => *param.1 = Parameter::String(if *u == 0 { String::new() } else { u.to_string() }),
                                        _ => {}
                                    }
                                }
                                match param.1 {
                                    Parameter::Float(f) => {
                                        ui.horizontal(|ui|{
                                            ui.add(egui::DragValue::new(f).speed(0.001).max_decimals(50));
                                            ui.label(label.clone());
                                        });
                                    }
                                    Parameter::Int(i) => {
                                        ui.horizontal(|ui|{
                                            ui.add(egui::DragValue::new(i).speed(1).max_decimals(0));
                                            ui.label(label.clone());
                                        });
                                    }
                                    Parameter::UInt(u) => {
                                        ui.horizontal(|ui|{
                                            let id = ui.make_persistent_id((name.as_str(), param.0));
                                            let mut text: String = ui.data_mut(|d| {
                                                d.get_temp(id).unwrap_or_else(|| {
                                                    if *u == 0 {
                                                        String::new()
                                                    } else {
                                                        u.to_string()
                                                    }
                                                })
                                            });
                                            let response = ui.add(
                                                egui::TextEdit::singleline(&mut text)
                                                    .hint_text("0 (Auto-detect)")
                                                    .desired_width(170.0),
                                            );
                                            if response.changed() {
                                                let trimmed = text.trim();
                                                if trimmed.is_empty() || trimmed == "0" {
                                                    *u = 0;
                                                } else if let Ok(val) = trimmed.parse::<u64>() {
                                                    *u = val;
                                                } else if let Ok(steam) = SteamID::from_steam3(trimmed) {
                                                    *u = steam.into();
                                                } else if let Ok(steam) = SteamID::from_steam2(trimmed) {
                                                    *u = steam.into();
                                                }
                                                ui.data_mut(|d| d.insert_temp(id, text));
                                            } else if !response.has_focus() {
                                                let current_str = if *u == 0 {
                                                    String::new()
                                                } else {
                                                    u.to_string()
                                                };
                                                if text != current_str {
                                                    ui.data_mut(|d| d.insert_temp(id, current_str));
                                                }
                                            }
                                            ui.label(label.clone());
                                        });
                                    }
                                    Parameter::String(s) => {
                                        ui.horizontal(|ui|{
                                            if param.0 == "recorder_id" {
                                                ui.add(egui::TextEdit::singleline(s).hint_text("Auto-detect").desired_width(170.0));
                                            } else if param.0 == "target_player" {
                                                ui.add(egui::TextEdit::singleline(s).hint_text("All players").desired_width(170.0));
                                            } else {
                                                ui.text_edit_singleline(s);
                                            }
                                            ui.label(label.clone());
                                        });
                                    }
                                    Parameter::Bool(b) => {
                                        ui.horizontal(|ui|{
                                            ui.checkbox(b, label.clone());
                                        });
                                    }
                                }
                            }
                        }
                    });
                });
                });
            });
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(!self.processing, egui::Button::new("Open...")).clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Demos", &["dem"])
                        .pick_file()
                    {
                        self.select_demo(path);
                    }
                }
                ui.add_enabled_ui(self.file.is_some(), |ui|{
                    let analyse_text = if !self.processing {
                        "Analyse"
                    } else if self.cancel_requested {
                        "Cancelling..."
                    } else {
                        "Cancel"
                    };
                    let analyse_btn = ui.add_enabled(!self.cancel_requested, egui::Button::new(analyse_text));
                    if analyse_btn.clicked() {
                        if self.processing {
                            self.cancel_requested = true;
                            analysis_template::CANCEL_ANALYSIS.store(true, std::sync::atomic::Ordering::Relaxed);
                        } else {
                            self.analyse(ui.ctx());
                        }
                    }
                    let save_text = match &self.save_feedback {
                        Some((t, Ok(()))) if t.elapsed().as_secs_f32() < 1.5 => "Saved ✓",
                        Some((t, Err(_))) if t.elapsed().as_secs_f32() < 3.0 => "Failed to save ✗",
                        _ => "Save detections",
                    };
                    let mut save_btn = ui.add_enabled(
                        self.analyser.is_some() && !self.processing,
                        egui::Button::new(save_text),
                    );
                    if let Some((t, Err(err_msg))) = &self.save_feedback {
                        if t.elapsed().as_secs_f32() < 3.0 {
                            save_btn = save_btn.on_hover_text(err_msg);
                        }
                    }
                    if save_btn.clicked() {
                        // Same content and format as "Copy detections"
                        let (extension, contents) = if self.copy_detailed_detections() {
                            ("json", format_detections_json(&self.detections, |id| self.get_player_name(id)))
                        } else {
                            ("md", format_detections_markdown(&self.detections, &self.reported_ids, |id| self.get_player_name(id)))
                        };
                        let demo_stem = self
                            .file
                            .as_ref()
                            .and_then(|p| p.file_stem())
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "demo".to_string());
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name(format!("{demo_stem}_detections.{extension}"))
                            .add_filter(extension, &[extension])
                            .save_file()
                        {
                            let result = std::fs::write(&path, contents)
                                .map_err(|e| format!("Failed to write {}: {e}", path.display()));
                            self.save_feedback = Some((std::time::Instant::now(), result));
                            ui.ctx().request_repaint_after(std::time::Duration::from_millis(3100));
                        }
                    }
                    let copy_dets_text = if self.copied_detections_feedback.as_ref().map_or(false, |t| t.elapsed().as_secs_f32() < 1.5) {
                        "Copied detections ✓"
                    } else {
                        "Copy detections"
                    };
                    let has_detections = self.detections.values().any(|d| !d.is_empty());
                    if ui
                        .add_enabled(has_detections && !self.processing, egui::Button::new(copy_dets_text))
                        .on_hover_text("Copy the detections as a .md file (player, count per algorithm, first 20 ticks), ready to paste into Discord. With \"Copy detailed detections\" enabled in the general settings, copies a .json file with all detections and their details instead")
                        .clicked()
                    {
                        let detailed = self.copy_detailed_detections();
                        let (extension, markdown) = if detailed {
                            ("json", format_detections_json(&self.detections, |id| self.get_player_name(id)))
                        } else {
                            ("md", format_detections_markdown(&self.detections, &self.reported_ids, |id| self.get_player_name(id)))
                        };
                        let demo_stem = self
                            .file
                            .as_ref()
                            .and_then(|p| p.file_stem())
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "demo".to_string());
                        match write_detections_file(&demo_stem, extension, &markdown)
                            .and_then(|path| copy_file_to_clipboard(&path))
                        {
                            Ok(()) => {}
                            Err(e) => {
                                // Fall back to plain text so the copy still does something useful
                                eprintln!("Failed to copy detections file to clipboard: {e}");
                                ui.ctx().copy_text(markdown);
                            }
                        }
                        self.copied_detections_feedback = Some(std::time::Instant::now());
                    }
                    if ui.button("View header").clicked() {
                        if let Some(path) = &self.file {
                            self.header_window.open_for_file(path);
                        }
                    }

                    if ui.button("Playerlist").clicked() {
                        if let Some(path) = &self.file {
                            self.playerlist_window.open_for_file(path);
                        }
                    }
                    if ui.button("Chat").clicked() {
                        if let Some(path) = &self.file {
                            self.chat_window.open_for_file(path);
                        }
                    }
                    #[cfg(feature = "experimental")]
                    {
                        let voice_btn_text = match &self.voice_export_feedback {
                            Some((t, Ok(_))) if t.elapsed().as_secs_f32() < 1.5 => "Exported VC ✓",
                            Some((t, Err(_))) if t.elapsed().as_secs_f32() < 3.0 => "No VC exported ✗",
                            _ => "Export VC",
                        };
                        let mut voice_btn = ui.button(voice_btn_text);
                        voice_btn = match &self.voice_export_feedback {
                            Some((_, Err(err_msg))) => voice_btn.on_hover_text(err_msg),
                            Some((_, Ok(summary))) => voice_btn.on_hover_text(format!(
                                "Exported {} recording(s) of {} player(s), {:.1} s in total, to {}{}",
                                summary.files,
                                summary.players,
                                summary.total_secs,
                                summary.directory.display(),
                                if summary.skipped_messages > 0 {
                                    format!("\n{} voice message(s) skipped (unsupported codec or corrupt)", summary.skipped_messages)
                                } else {
                                    String::new()
                                },
                            )),
                            None => voice_btn.on_hover_text(
                                "Export all voice chats of the demo as .wav files into voicechats/<demo name>, named by SteamID. Recordings of a player with a pause shorter than the voice export gap (general settings) are joined.",
                            ),
                        };
                        if voice_btn.clicked() {
                            if let Some(path) = self.file.clone() {
                                let result = self.export_voice_chats(&path);
                                if let Err(e) = &result {
                                    eprintln!("Failed to export voice chats: {e}");
                                }
                                self.voice_export_feedback = Some((std::time::Instant::now(), result));
                            }
                        }
                    }
                });
            });
            ui.add_space(4.0);
            ui.add_enabled_ui(self.file.is_some() && !self.processing, |ui| {
                ui.horizontal(|ui| {
                    if ui.add_enabled(self.prev_demo.is_some(), egui::Button::new("Previous demo")).clicked() {
                        self.update_adjacent_demos();
                        if let Some(prev_path) = self.prev_demo.clone() {
                            self.select_demo(prev_path);
                        }
                    }
                    if ui.add_enabled(self.next_demo.is_some(), egui::Button::new("Next demo")).clicked() {
                        self.update_adjacent_demos();
                        if let Some(next_path) = self.next_demo.clone() {
                            self.select_demo(next_path);
                        }
                    }
                    let copy_btn_text = if self.copied_demo_feedback.as_ref().map_or(false, |t| t.elapsed().as_secs_f32() < 1.5) {
                        "Copied demo ✓"
                    } else {
                        "Copy demo"
                    };
                    if ui.button(copy_btn_text).clicked() {
                        if let Some(path) = &self.file {
                            if let Err(e) = copy_file_to_clipboard(path) {
                                eprintln!("Failed to copy demo to clipboard: {e}");
                            } else {
                                self.copied_demo_feedback = Some(std::time::Instant::now());
                            }
                        }
                    }
                    let use_old_replay = self.use_old_replay_conversion();
                    let replay_btn_text = match &self.replay_feedback {
                        // Stays checked until another demo is loaded (select_demo clears it), so a
                        // second click can't create duplicate replays of the same demo
                        Some((_, Ok(count))) => {
                            if *count > 1 {
                                "Converted (multiple replays) ✓"
                            } else {
                                "Converted to replay ✓"
                            }
                        }
                        Some((t, Err(_))) if t.elapsed().as_secs_f32() < 3.0 => "Failed to convert ✗",
                        _ => "Convert to replay",
                    };
                    let already_converted = matches!(&self.replay_feedback, Some((_, Ok(_))));
                    let mut replay_btn = ui.add_enabled(!already_converted, egui::Button::new(replay_btn_text));
                    if let Some((_, Err(err_msg))) = &self.replay_feedback {
                        replay_btn = replay_btn.on_hover_text(err_msg);
                    } else if let Some((_, Ok(count))) = &self.replay_feedback {
                        replay_btn = replay_btn.on_disabled_hover_text(format!("Created {count} replay(s); load another demo to convert again"));
                    } else if use_old_replay {
                        replay_btn = replay_btn.on_hover_text("Basic replay conversion without takes or suspect tracking (old method, enabled in the general settings)");
                    } else {
                        replay_btn = replay_btn.on_hover_text("Convert demo to replay(s) with grouped takes for each suspect");
                    }
                    if replay_btn.clicked() {
                        if let Some(path) = &self.file {
                            let result = if use_old_replay {
                                self.convert_demo_to_replay_old(path).map(|_| 1)
                            } else {
                                self.convert_demo_to_replay(path)
                            };
                            match result {
                                Ok(count) => {
                                    self.replay_feedback = Some((std::time::Instant::now(), Ok(count)));
                                }
                                Err(e) => {
                                    eprintln!("Failed to convert demo to replay: {e}");
                                    self.replay_feedback = Some((std::time::Instant::now(), Err(e)));
                                }
                            }
                        }
                    }
                    let delete_btn_text = match &self.delete_replays_feedback {
                        Some((t, Ok(count))) if t.elapsed().as_secs_f32() < 1.5 => {
                            if *count == 0 { "No replays found ✓" } else { "Deleted replays ✓" }
                        }
                        Some((t, Err(_))) if t.elapsed().as_secs_f32() < 3.0 => "Failed to delete ✗",
                        _ => "Delete all replays",
                    };
                    let mut delete_btn = ui.button(delete_btn_text);
                    if let Some((_, Err(err_msg))) = &self.delete_replays_feedback {
                        delete_btn = delete_btn.on_hover_text(err_msg);
                    } else if let Some((_, Ok(count))) = &self.delete_replays_feedback {
                        delete_btn = delete_btn.on_hover_text(format!("Deleted {count} file(s)"));
                    } else {
                        delete_btn = delete_btn.on_hover_text("Delete all .dmx and .dem files (including generated takes) from the TF2 replay folder");
                    }
                    if delete_btn.clicked() {
                        self.show_delete_replays_prompt = true;
                    }
                    if self.show_strip_demo_button() {
                    let strip_btn_text = match &self.strip_feedback {
                        Some((t, Ok(count))) if t.elapsed().as_secs_f32() < 1.5 => {
                            if *count == 0 {
                                "No consolecmds found ✓"
                            } else {
                                "Stripped demo ✓"
                            }
                        }
                        Some((t, Err(_))) if t.elapsed().as_secs_f32() < 3.0 => "Failed to strip ✗",
                        _ => "Strip demo",
                    };
                    let mut strip_btn = ui.button(strip_btn_text);
                    if let Some((_, Err(err_msg))) = &self.strip_feedback {
                        strip_btn = strip_btn.on_hover_text(err_msg);
                    } else if let Some((_, Ok(count))) = &self.strip_feedback {
                        strip_btn = strip_btn.on_hover_text(format!("Stripped {count} consolecmds"));
                    } else {
                        strip_btn = strip_btn.on_hover_text("Strip all consolecmds directly in the demo to prevent playback interference");
                    }
                    if strip_btn.clicked() {
                        if let Some(path) = &self.file {
                            match Self::strip_demo_consolecmds(path) {
                                Ok(count) => {
                                    self.strip_feedback = Some((std::time::Instant::now(), Ok(count)));
                                }
                                Err(e) => {
                                    eprintln!("Failed to strip demo consolecmds: {e}");
                                    self.strip_feedback = Some((std::time::Instant::now(), Err(e)));
                                }
                            }
                        }
                    }
                    }
                });
            });
            #[cfg(feature = "rcon")]
            {
                let rcon_bool = |key: &str, default: bool| match self.params.get("rcon").and_then(|p| p.get(key)) {
                    Some(Parameter::Bool(b)) => *b,
                    _ => default,
                };
                let mut rt_enabled = rcon_bool("real_time_scanning", false);
                let scan_on_death = rcon_bool("scan_on_death", true);
                let scan_periodic_too = rcon_bool("scan_periodic_too", false);
                let scan_interval = match self.params.get("rcon").and_then(|p| p.get("scan_interval_seconds")) {
                    Some(Parameter::Int(i)) => (*i).max(5),
                    _ => analysis_template::lib::live_scan::DEFAULT_LIVE_SCAN_INTERVAL_SECONDS as i32,
                };
                // Short description of when scans happen
                let mode = if scan_on_death && scan_periodic_too {
                    format!("on death / {scan_interval}s")
                } else if scan_on_death {
                    "on death".to_string()
                } else {
                    format!("every {scan_interval}s")
                };

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut rt_enabled, "Live")
                        .on_hover_text("Monitor the TF2 demo being recorded and scan it in real time")
                        .changed()
                    {
                        self.params.entry("rcon".to_string()).or_default()
                            .insert("real_time_scanning".to_string(), Parameter::Bool(rt_enabled));
                        self.save_params();
                    }
                    if !rt_enabled {
                        return;
                    }
                    let msg = self.rt_status_message.as_deref();
                    if msg.is_some_and(|m| m.starts_with("Scanning") || m.starts_with("Death")) {
                        ui.spinner();
                    }
                    let amber = egui::Color32::from_rgb(255, 175, 60);
                    let (icon, color, text, hover) = if !self.rcon_connected {
                        let hover = format!("RCON is not connected: scanning every {scan_interval}s, party chat alerts are disabled");
                        match msg {
                            Some(m) if m.starts_with("Scanning") => ("🔄", egui::Color32::from_rgb(255, 205, 80), format!("{m} (RCON offline)"), hover),
                            Some(m) if m.starts_with("Scanned") => ("✅", egui::Color32::from_rgb(90, 200, 110), format!("{m} (RCON offline)"), hover),
                            Some(m) if m.starts_with("Recording") => ("⚠", amber, format!("{m} (RCON offline)"), hover),
                            _ => ("⚠", amber, format!("RCON offline, scanning every {scan_interval}s"), hover),
                        }
                    } else {
                        let hover = format!("RCON connected: scanning {mode}");
                        match msg {
                            Some(m) if m.starts_with("Scanning") => ("🔄", egui::Color32::from_rgb(255, 205, 80), m.to_string(), hover),
                            Some(m) if m.starts_with("Death") => ("💀", egui::Color32::from_rgb(245, 120, 120), m.to_string(), hover),
                            Some(m) if m.starts_with("Scanned") => ("✅", egui::Color32::from_rgb(90, 200, 110), m.to_string(), hover),
                            Some(m) if m.starts_with("Recording") => ("🔴", egui::Color32::from_rgb(235, 85, 85), format!("{m} ({mode})"), hover),
                            Some(m) => ("ℹ", egui::Color32::from_rgb(160, 200, 240), m.to_string(), hover),
                            None => ("📡", egui::Color32::from_rgb(160, 200, 240), format!("Watching for demo ({mode})"), hover),
                        }
                    };
                    ui.add(egui::Label::new(egui::RichText::new(format!("{icon} {text}")).color(color)).truncate())
                        .on_hover_text(hover);
                });
            }
            if self.processing {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Analysing...");
                    let (avg_ticks, total_ticks) = analysis_template::get_average_progress();
                    let (pct, text) = if total_ticks == 0 {
                        if avg_ticks > 0 {
                            // Ongoing match / corrupted header demo: fallback to 100% with "-"
                            (1.0, "-".to_string())
                        } else {
                            // Initial startup / loading: 0% empty bar without blue flash
                            (0.0, "0%".to_string())
                        }
                    } else {
                        let pct = (avg_ticks as f32 / total_ticks as f32).clamp(0.0, 1.0);
                        (pct, format!("{:.0}%", pct * 100.0))
                    };
                    let right_margin = 8.0;
                    let bar_width = (ui.ctx().screen_rect().max.x - right_margin - ui.cursor().min.x).max(50.0);
                    ui.add(
                        egui::widgets::ProgressBar::new(pct)
                            .text(text)
                            .desired_width(bar_width),
                    );
                });
            }
            ui.add_space(10.0);
            if let Some(err) = self.analysis_error.clone() {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("⚠ Analysis failed: {err}"))
                                .color(egui::Color32::from_rgb(235, 70, 70)),
                        )
                        .wrap(),
                    );
                    if ui.small_button("Dismiss").clicked() {
                        self.analysis_error = None;
                    }
                });
                ui.add_space(4.0);
            }
            if let Some(p) = &self.file {
                ui.heading(p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
                ui.label("Doubleclick steamid to open profile");
            }
            ui.separator();
            let mut recent_to_open = None;
            if self.file.is_none() {
                ui.add_space(20.0);
                let right_edge = ui.ctx().screen_rect().max.x - 8.0;
                let width = (right_edge - ui.cursor().min.x).max(50.0);
                ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.heading("Drop a .dem file here or click Open...");
                    if !self.recent_files.is_empty() {
                        ui.add_space(20.0);
                        ui.label(egui::RichText::new("Recent demos").strong());
                        ui.add_space(4.0);
                        for path in &self.recent_files {
                            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            if ui
                                .link(name)
                                .on_hover_text(path.to_string_lossy())
                                .clicked()
                            {
                                recent_to_open = Some(path.clone());
                            }
                        }
                    }
                });
            }
            if self.file.is_some() {
            ui.horizontal_top(|ui| {
                egui::ScrollArea::vertical()
                    .id_salt("players")
                    .show(ui, |ui| {
                        ui.set_min_width(180.0);
                        ui.vertical(|ui| {
                            let mut newly_hovered_player = None;
                            let mut player_to_open_in_playerlist = None;
                            #[cfg(feature = "tf2bd")]
                            let mut player_to_mark: Option<u64> = None;
                            for player in self.detections.iter().sorted_by_key(|d| d.1.len()).rev()
                            {
                                let label_text = if self.hovered_player == Some(*player.0) {
                                    format!("{} ({})", player.0, player.1.len())
                                } else if let Some(name) = self.get_player_name(*player.0) {
                                    format!("{} ({})", name, player.1.len())
                                } else {
                                    format!("{} ({})", player.0, player.1.len())
                                };

                                let is_marked = self.reported_ids.contains(player.0);
                                let is_deleted = !is_marked && self.deleted_accounts.contains(player.0);
                                let is_existing = !is_marked && self.existing_accounts.contains(player.0);
                                let tag = self.player_tag(*player.0);

                                let widget_text = if let Some(tag) = tag {
                                    egui::RichText::new(label_text).color(tag.color)
                                } else if is_marked {
                                    egui::RichText::new(label_text).color(marked_color)
                                } else if is_deleted {
                                    egui::RichText::new(label_text).color(deleted_color)
                                } else if is_existing {
                                    if let Some(color) = normal_color {
                                        egui::RichText::new(label_text).color(color)
                                    } else {
                                        egui::RichText::new(label_text)
                                    }
                                } else {
                                    egui::RichText::new(label_text)
                                };

                                let button = egui::SelectableLabel::new(
                                    self.selected_player.is_some_and(|u| u == *player.0),
                                    widget_text,
                                );
                                let mut res = ui.add_sized([180.0, 0.0], button);
                                if let Some(tag) = tag {
                                    res = res.on_hover_text(format!("TF2BD: {}", tag.label));
                                } else if is_deleted {
                                    res = res.on_hover_text("Steam account no longer exists");
                                }

                                res.context_menu(|ui| {
                                    if ui.button("View player info in playerlist").clicked() {
                                        player_to_open_in_playerlist = Some(*player.0);
                                        ui.close_menu();
                                    }
                                    if ui.button("Copy SteamHistory link").clicked() {
                                        ui.ctx().copy_text(format!("https://steamhistory.net/id/{}", player.0));
                                        ui.close_menu();
                                    }
                                    if ui.button("Copy Steam profile link").clicked() {
                                        ui.ctx().copy_text(format!("https://steamcommunity.com/profiles/{}", player.0));
                                        ui.close_menu();
                                    }
                                    if ui.button("Copy steamID").clicked() {
                                        ui.ctx().copy_text(format!("{}", player.0));
                                        ui.close_menu();
                                    }
                                    if ui.button("Copy ingame name").clicked() {
                                        let name = self.get_player_name(*player.0).unwrap_or("");
                                        ui.ctx().copy_text(name.to_string());
                                        ui.close_menu();
                                    }
                                    #[cfg(feature = "tf2bd")]
                                    if self.tf2bd.enabled() && ui.button("Mark player").clicked() {
                                        player_to_mark = Some(*player.0);
                                        ui.close_menu();
                                    }
                                });

                                if res.hovered() {
                                    newly_hovered_player = Some(*player.0);
                                }
                                if res.clicked() {
                                    self.selected_player = Some(*player.0);
                                    self.selected_detection = None;
                                }
                                if res.double_clicked() {
                                    let _ = opener::open_browser(format!(
                                        "https://steamhistory.net/id/{}",
                                        player.0
                                    ));
                                }
                            }
                            self.hovered_player = newly_hovered_player;

                            #[cfg(feature = "tf2bd")]
                            if let Some(steam_id) = player_to_mark {
                                let name = self.get_player_name(steam_id).unwrap_or("").to_string();
                                self.tf2bd.open_mark(steam_id, &name);
                            }

                            if let Some(steam_id) = player_to_open_in_playerlist {
                                if let Some(path) = &self.file {
                                    self.playerlist_window.open_for_player(path, steam_id);
                                }
                            }
                        });
                    });
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("detections")
                    .show(ui, |ui| {
                        ui.vertical(|ui| {
                            if let Some(detections) =
                                self.selected_player.and_then(|p| self.detections.get(&p))
                            {
                                for (i, det) in detections.iter().enumerate() {
                                    let resp = ui.selectable_label(
                                        self.selected_detection.is_some_and(|si| si == i),
                                        format!("{}: {}", det.tick, det.algorithm),
                                    );
                                    if resp.clicked() {
                                        self.selected_detection = Some(i);
                                    }
                                }
                            }
                        });
                    });
                ui.separator();
                let right_margin = 8.0;
                let available_for_details = (ui.ctx().screen_rect().max.x - right_margin - ui.cursor().min.x).max(50.0);
                egui::ScrollArea::vertical()
                    .id_salt("details")
                    .max_width(available_for_details)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let scrollbar_reserve = (ui.spacing().scroll.allocated_width() + 8.0).max(16.0);
                        let wrap_width = (available_for_details - scrollbar_reserve).max(50.0);
                        if let Some(det) = self
                            .selected_player
                            .and_then(|p| self.detections.get(&p))
                            .and_then(|dets| self.selected_detection.and_then(|di| dets.get(di)))
                        {
                            let text = serde_json::to_string_pretty(&det.data).unwrap();
                            let mut layout_job = egui::WidgetText::from(text)
                                .into_layout_job(ui.style(), egui::FontSelection::Default, egui::Align::Min);
                            layout_job.wrap.max_width = wrap_width;
                            layout_job.wrap.break_anywhere = true;
                            let galley = ui.fonts(|f| f.layout_job(layout_job));
                            ui.add(egui::Label::new(galley));
                        }
                    });
            });
            }
            if let Some(path) = recent_to_open {
                if path.is_file() {
                    self.select_demo(path);
                } else {
                    self.recent_files.retain(|p| *p != path);
                    self.analysis_error = Some(format!("{} no longer exists", path.display()));
                }
            }
        });

        // Render sub-windows
        self.header_window.ui(ctx);
        #[cfg(feature = "tf2bd")]
        let (tags, can_mark) = (self.tf2bd.tags(), self.tf2bd.enabled());
        #[cfg(not(feature = "tf2bd"))]
        let (tags, can_mark) = (&HashMap::new(), false);
        self.playerlist_window.ui(
            ctx,
            &PlayerHighlights {
                reported_ids: &self.reported_ids,
                deleted_ids: &self.deleted_accounts,
                existing_ids: &self.existing_accounts,
                normal_color,
                marked_color,
                deleted_color,
                tags,
                can_mark,
            },
        );
        self.chat_window.ui(ctx);
        #[cfg(feature = "tf2bd")]
        {
            if let Some((steam_id, name)) = self.playerlist_window.mark_request.take() {
                self.tf2bd.open_mark(steam_id, &name);
            }
            self.tf2bd.show(ctx);
            self.sync_tf2bd();
        }

        if !has_open_subwindow && !self.processing {
            if let Some(path) = ctx.input(|i| i.raw.dropped_files.first().and_then(|f| f.path.clone())) {
                self.select_demo(path);
                self.analyse(ctx);
            }
        }

        if self.show_delete_replays_prompt {
            egui::Window::new("Delete all replays")
                .collapsible(false)
                .resizable(false)
                .pivot(egui::Align2::CENTER_CENTER)
                .default_pos(ctx.screen_rect().center())
                .show(ctx, |ui| {
                    ui.label("Delete all .dmx and .dem files from the TF2 replay folder,
including the takes generated for the replays?");
                    ui.label("This cannot be undone.");
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Delete all").clicked() {
                            self.show_delete_replays_prompt = false;
                            if let Some(path) = self.file.clone() {
                                let result = Self::delete_all_replays(&path);
                                if let Err(e) = &result {
                                    eprintln!("Failed to delete replays: {e}");
                                }
                                self.delete_replays_feedback = Some((std::time::Instant::now(), result));
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_delete_replays_prompt = false;
                        }
                    });
                });
        }

        if self.show_exit_prompt {
            egui::Window::new("Unsaved settings")
                .collapsible(false)
                .resizable(false)
                .pivot(egui::Align2::CENTER_CENTER)
                .default_pos(ctx.screen_rect().center())
                .show(ctx, |ui| {
                    ui.label("Some settings were changed but not saved.");
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save and exit").clicked() {
                            self.save_params();
                            self.allow_close = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Exit without saving").clicked() {
                            self.allow_close = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_exit_prompt = false;
                        }
                    });
                });
        }
    }
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_ui(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitbuffer::BitRead;

    #[test]
    fn test_format_detections_markdown() {
        let det = |tick: u32, algorithm: &str, player: u64| Detection {
            tick,
            algorithm: algorithm.to_string(),
            player,
            data: serde_json::json!({}),
        };
        let mut detections: HashMap<u64, Vec<Detection>> = HashMap::new();
        detections.insert(
            1,
            vec![
                det(12514, "autostrafe", 1),
                det(2186, "autostrafe", 1),
                det(15261, "autostrafe", 1),
                det(4089, "fidoo/psilent5", 1),
                det(3228, "fidoo/psilent5", 1),
            ],
        );
        detections.insert(2, (0..25).map(|i| det(i * 10, "bhop_a", 2)).collect());
        detections.insert(3, vec![det(5, "x", 3)]);

        let names = |id: u64| match id {
            1 => Some("ANGRY_TOBUSCUS"),
            2 => Some(""),
            _ => None,
        };
        let reported: HashSet<u64> = [1].into_iter().collect();
        let md = format_detections_markdown(&detections, &reported, names);
        let ticks_20 = (0..20).map(|i| (i * 10).to_string()).collect::<Vec<_>>().join(", ");
        let expected = format!(
            "**Unknown - 2:**\n\n* bhop_a (25) - {ticks_20}\n\n\
             **ANGRY\\_TOBUSCUS - 1:**\n\n* ALREADY MARKED\n* autostrafe (3) - 2186, 12514, 15261\n\
             * fidoo/psilent5 (2) - 3228, 4089\n\n\
             **Unknown - 3:**\n\n* x (1) - 5\n"
        );
        assert_eq!(md, expected);
    }

    #[test]
    fn test_strip_consolecmds() {
        let demo_path = std::path::Path::new("reference_demos/autostrafe.dem");
        if !demo_path.exists() {
            return;
        }

        let original_bytes = std::fs::read(demo_path).expect("Read autostrafe.dem");
        let (stripped_bytes, count) =
            Gui::strip_consolecmds_from_bytes(&original_bytes).expect("Strip consolecmds");

        assert!(count > 0, "Should have stripped at least one consolecmd");
        assert!(
            stripped_bytes.len() < original_bytes.len(),
            "Stripped demo should be smaller"
        );

        // Second pass should strip 0 consolecmds and produce identical bytes
        let (second_pass, second_count) =
            Gui::strip_consolecmds_from_bytes(&stripped_bytes).expect("Second strip pass");
        assert_eq!(second_count, 0, "Second pass should find 0 consolecmds");
        assert_eq!(
            second_pass, stripped_bytes,
            "Second pass should produce identical bytes"
        );

        // Verify the stripped demo can be parsed cleanly by tf-demo-parser
        let demo = tf_demo_parser::Demo::new(&stripped_bytes);
        let mut stream = demo.get_stream();
        let header = tf_demo_parser::demo::header::Header::read(&mut stream).expect("Parse header");
        assert_eq!(header.demo_type, "HL2DEMO");

        let state = tf_demo_parser::ParserState::new(header.protocol, |_| true, false);
        let mut packets = tf_demo_parser::demo::parser::RawPacketStream::new(stream);
        let mut consolecmd_found = false;
        let mut usercmd_found = false;
        while let Ok(Some(packet)) = packets.next(&state) {
            match packet {
                tf_demo_parser::demo::packet::Packet::ConsoleCmd(_) => consolecmd_found = true,
                tf_demo_parser::demo::packet::Packet::UserCmd(_) => usercmd_found = true,
                _ => {}
            }
        }
        assert!(
            !consolecmd_found,
            "Stripped demo should contain zero ConsoleCmd packets"
        );
        assert!(usercmd_found, "Stripped demo should keep its UserCmd packets");
    }

    #[test]
    fn test_strip_demo_consolecmds_inplace() {
        let demo_path = std::path::Path::new("reference_demos/autostrafe.dem");
        if !demo_path.exists() {
            return;
        }

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("test_strip_inplace_{}.dem", std::process::id()));
        std::fs::copy(demo_path, &temp_file).expect("Copy to temp");

        let original_size = std::fs::metadata(&temp_file).unwrap().len();

        let count = Gui::strip_demo_consolecmds(&temp_file).expect("Strip in-place");
        assert!(count > 0);

        let new_size = std::fs::metadata(&temp_file).unwrap().len();
        assert!(new_size < original_size, "File should be smaller after in-place strip");

        // Running again on the same file should succeed with 0 stripped
        let count2 = Gui::strip_demo_consolecmds(&temp_file).expect("Strip in-place again");
        assert_eq!(count2, 0);

        let final_size = std::fs::metadata(&temp_file).unwrap().len();
        assert_eq!(new_size, final_size);

        let _ = std::fs::remove_file(&temp_file);
    }
}

