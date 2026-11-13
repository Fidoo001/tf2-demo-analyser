//! Real-Time Demo Scanning & Death Detection Module
//!
//! Monitors TF2 demo recording activity in real-time, triggers demo scans on
//! player death or configurable intervals, and notifies the in-game party chat
//! via RCON when cheats are detected.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::base::cheat_analyser_base::CheatAnalyser;
use crate::lib::algorithm::{analyse_sequential_from_tick, get_algorithms, CheatAlgorithm, Demo};
use crate::lib::live_scan::{
    analysed_until, LiveScanGate, LiveScanProgress, DEFAULT_LIVE_SCAN_INTERVAL_SECONDS,
};
use crate::lib::general_settings::GeneralSettings;
use crate::lib::parameters::{Parameter, Parameters};
use crate::util::helpers::find_default_tf2_dir;
use super::client::RconClient;
use super::vote_notifier::{VoteNotifier, VOTE_ALGORITHM};
use crate::debug_log;

/// Prints a live scanner message and writes it to the debug log (`--debug`).
macro_rules! live_log {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        debug_log!("{line}");
    }};
}

/// Like `live_log!`, on stderr.
macro_rules! live_elog {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        eprintln!("{line}");
        debug_log!("{line}");
    }};
}

#[derive(Debug, Clone)]
pub struct RealTimeScanConfig {
    pub port: i32,
    pub password: String,
    pub scan_on_death: bool,
    /// With `scan_on_death`, also scan every `scan_interval_seconds`; whichever comes first.
    pub scan_periodic_too: bool,
    pub scan_interval_seconds: u64,
    pub party_chat_alerts: bool,
    pub warn_marked_cheaters: bool,
    /// Announce each votekick (caller, target, yes/no voters) in party chat once it has finished.
    pub vote_notifications: bool,
    /// Skip the notifications about the recorder (detections, marked cheater and playerlist warnings).
    pub ignore_recorder: bool,
}

impl Default for RealTimeScanConfig {
    fn default() -> Self {
        Self {
            port: 27015,
            password: String::new(),
            scan_on_death: true,
            scan_periodic_too: false,
            scan_interval_seconds: DEFAULT_LIVE_SCAN_INTERVAL_SECONDS,
            party_chat_alerts: true,
            warn_marked_cheaters: true,
            vote_notifications: false,
            ignore_recorder: true,
        }
    }
}

/// How often the `status` command is issued to look for marked cheaters on the server.
const STATUS_POLL_INTERVAL: Duration = Duration::from_secs(20);
/// How long to wait for the server's `status` reply to show up in `console.log`.
const STATUS_REPLY_DELAY: Duration = Duration::from_millis(900);
const STEAM_ID64_BASE: u64 = 76561197960265728;

/// SteamID64 of the account logged in to Steam, which is the one recording the live demo
/// (`HKCU\Software\Valve\Steam\ActiveProcess\ActiveUser`, 0 while logged out).
#[cfg(windows)]
pub fn local_steam_id() -> Option<u64> {
    #[link(name = "advapi32")]
    extern "system" {
        fn RegGetValueW(
            hkey: isize,
            sub_key: *const u16,
            value: *const u16,
            flags: u32,
            value_type: *mut u32,
            data: *mut std::ffi::c_void,
            data_len: *mut u32,
        ) -> i32;
    }
    const HKEY_CURRENT_USER: isize = 0x8000_0001u32 as i32 as isize;
    const RRF_RT_REG_DWORD: u32 = 0x18;
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let (sub_key, value) = (wide(r"Software\Valve\Steam\ActiveProcess"), wide("ActiveUser"));
    let mut account = 0u32;
    let mut len = std::mem::size_of::<u32>() as u32;
    // SAFETY: both strings are NUL-terminated and `account` holds the requested DWORD.
    let status = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            sub_key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            (&mut account as *mut u32).cast(),
            &mut len,
        )
    };
    (status == 0 && account != 0).then(|| STEAM_ID64_BASE + u64::from(account))
}

#[cfg(not(windows))]
pub fn local_steam_id() -> Option<u64> {
    None
}

/// The recorder's SteamID64 from the demo: the only player named like the header nick.
fn demo_recorder_id(analyser: &CheatAnalyser) -> Option<u64> {
    let nick = analyser.state.header.as_ref()?.nick.trim_matches('\0').trim().to_string();
    if nick.is_empty() {
        return None;
    }
    let mut ids = analyser
        .state
        .player_names
        .iter()
        .filter(|(_, name)| name.trim() == nick)
        .map(|(&id, _)| id);
    match (ids.next(), ids.next()) {
        (Some(id), None) => Some(id),
        _ => None,
    }
}

/// Parses the player rows of a `status` output into `(steam_id64, name)` pairs.
/// Rows look like: `#    123 "Some Name"   [U:1:12345678]   12:34   50  0 active`.
/// Bots (`BOT` unique id) and the header row are skipped.
pub fn parse_status_players(text: &str) -> Vec<(u64, String)> {
    let mut players = Vec::new();
    for line in text.lines() {
        let clean = strip_console_timestamp(line);
        if !clean.starts_with('#') {
            continue;
        }
        let (Some(name_start), Some(name_end)) = (clean.find('"'), clean.rfind('"')) else {
            continue;
        };
        if name_end <= name_start {
            continue;
        }
        let name = &clean[name_start + 1..name_end];
        let Some(uniqueid) = clean[name_end + 1..].split_whitespace().next() else {
            continue;
        };
        let Some(account) = uniqueid
            .strip_prefix("[U:1:")
            .and_then(|s| s.strip_suffix(']'))
            .and_then(|s| s.parse::<u64>().ok())
        else {
            continue;
        };
        players.push((STEAM_ID64_BASE + account, name.to_string()));
    }
    players
}

/// Reads the bytes appended to `console.log` since `from_pos` (shared read mode).
fn read_log_since(path: &Path, from_pos: u64) -> Option<String> {
    let mut file = open_recording_file(path).ok()?;
    let len = file.metadata().ok()?.len();
    let start = if len < from_pos { 0 } else { from_pos };
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

pub enum RealTimeScanEvent {
    DemoDetected {
        demo_path: PathBuf,
        recorder_name: Option<String>,
    },
    DeathDetected {
        victim: String,
    },
    /// The active demo was closed (or went stale); it is no longer scanned.
    RecordingStopped(PathBuf),
    ScanStarted(PathBuf),
    ScanFinished {
        demo_path: PathBuf,
        analyser: CheatAnalyser<'static>,
        new_notifications: Vec<(String, String, u32)>, // (player_name, algorithm, tick)
        alerts_sent: bool,
        /// `LiveScanGate::epoch` when the scan started: the results are outdated by a manual rescan
        /// that finished after that.
        epoch: u64,
    },
    RconStatus(String),
    RconConnected(bool),
    /// Marked cheaters found on the current server and announced in party chat.
    MarkedCheatersOnServer(Vec<String>),
}

/// Resolves the TF2 game directory (`tf`).
pub fn find_tf2_root_dir() -> Option<PathBuf> {
    if let Some(demos_dir) = find_default_tf2_dir() {
        if demos_dir
            .file_name()
            .map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("demos"))
        {
            if let Some(parent) = demos_dir.parent() {
                return Some(parent.to_path_buf());
            }
        }
        Some(demos_dir)
    } else {
        None
    }
}

/// Searches the standard TF2 demo directories for a `.dem` file actively being recorded.
/// A file is considered "actively being recorded" if it:
/// 1. Has at least 1072 bytes (valid header size)
/// 2. Starts with `HL2DEMO\0`
/// 3. Was modified within the last 60 seconds
/// 4. Has not been finished (see `is_demo_recording_finished`)
pub fn find_recording_demo() -> Option<PathBuf> {
    find_recording_demo_traced(&mut |_| {})
}

/// `find_recording_demo`, reporting the folders searched and why recent demos were skipped.
pub fn find_recording_demo_traced(trace: &mut dyn FnMut(String)) -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(demos_dir) = find_default_tf2_dir() {
        search_dirs.push(demos_dir.clone());
        if demos_dir
            .file_name()
            .map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("demos"))
        {
            if let Some(parent) = demos_dir.parent() {
                search_dirs.push(parent.to_path_buf());
            }
        } else {
            let demos_sub = demos_dir.join("demos");
            if demos_sub.is_dir() {
                search_dirs.push(demos_sub);
            }
        }
    }

    let mut newest_candidate: Option<(PathBuf, std::time::SystemTime)> = None;

    if search_dirs.is_empty() {
        trace("Recording search: TF2 folder not found".to_string());
    }
    for dir in search_dirs {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                trace(format!("Recording search: cannot list {}: {e}", dir.display()));
                continue;
            }
        };
        trace(format!("Recording search: looking in {}", dir.display()));

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .map_or(false, |ext| ext.eq_ignore_ascii_case("dem"))
            {
                if let Ok(meta) = path.metadata() {
                    // Only demos changed in the last 10 minutes are reported, the others are old ones
                    if meta.modified().ok().and_then(|m| m.elapsed().ok()).map_or(false, |e| e.as_secs() <= 600) {
                        trace(format!(
                            "  {}: {} bytes, modified {}s ago, finished={}, shared open={}",
                            path.display(),
                            meta.len(),
                            meta.modified().ok().and_then(|m| m.elapsed().ok()).map_or(0, |e| e.as_secs()),
                            is_demo_recording_finished(&path),
                            match open_recording_file(&path) {
                                Ok(_) => "OK".to_string(),
                                Err(e) => format!("FAILED ({e})"),
                            }
                        ));
                    }
                    if meta.len() >= 1072 && !is_demo_recording_finished(&path) {
                        if let Ok(modified) = meta.modified() {
                            if let Ok(elapsed) = modified.elapsed() {
                                if elapsed.as_secs() <= 60 {
                                    // Verify header magic
                                    if let Ok(mut f) = open_recording_file(&path) {
                                        let mut magic = [0u8; 8];
                                        if f.read_exact(&mut magic).is_ok() && &magic == b"HL2DEMO\0" {
                                            match &newest_candidate {
                                                Some((_, prev_mod)) if modified > *prev_mod => {
                                                    newest_candidate = Some((path, modified));
                                                }
                                                None => {
                                                    newest_candidate = Some((path, modified));
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
            }
        }
    }

    newest_candidate.map(|(p, _)| p)
}

/// Whether TF2 has stopped recording the demo. The engine writes the header with zero playback
/// time, ticks and frames when the recording starts and fills them in (and appends the stop
/// frame) only when it stops, so they stay zero while it is being recorded (and after a crash).
pub fn is_demo_recording_finished(path: &Path) -> bool {
    let Ok(mut file) = open_recording_file(path) else {
        return false;
    };
    let mut header = [0u8; 1072];
    if file.read_exact(&mut header).is_err() || &header[0..8] != b"HL2DEMO\0" {
        return false;
    }
    // playback_time f32 at 1056, playback_ticks i32 at 1060, playback_frames i32 at 1064
    let ticks = i32::from_le_bytes(header[1060..1064].try_into().unwrap());
    let frames = i32::from_le_bytes(header[1064..1068].try_into().unwrap());
    ticks > 0 || frames > 0
}

/// Reads the player (recorder) nickname stored inside the demo header.
pub fn get_demo_recorder_name(path: &Path) -> Option<String> {
    let mut file = open_recording_file(path).ok()?;
    let mut header = [0u8; 1072];
    file.read_exact(&mut header).ok()?;
    if &header[0..8] != b"HL2DEMO\0" {
        return None;
    }
    // client_name is at byte offset 276..536 (260 bytes)
    let client_name_slice = &header[276..536];
    let end = client_name_slice.iter().position(|&b| b == 0).unwrap_or(260);
    let name = String::from_utf8_lossy(&client_name_slice[..end])
        .trim()
        .to_string();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Opens a file in shared read mode so TF2's active write lock does not fail on Windows.
pub fn open_recording_file(path: &Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_SHARE_READ (1) | FILE_SHARE_WRITE (2) | FILE_SHARE_DELETE (4) = 7
        std::fs::OpenOptions::new()
            .read(true)
            .share_mode(7)
            .open(path)
    }
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)
    }
}

/// Reads all available bytes from a recording demo file with shared read permissions.
pub fn read_recording_demo_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = open_recording_file(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Follows `console.log` to know which demo TF2 is recording: demo support (and `record`) prints
/// `Recording to demos\<name>.dem...` when a recording starts and `Completed demo, ...` when it ends.
/// Unlike the demo file's modification time this is correct right away, also when the recording
/// has been running for a long time before the scanner was turned on.
#[derive(Default)]
struct ConsoleDemoTracker {
    /// `console.log` position up to which the lines were processed (None: not read yet)
    pos: Option<u64>,
    /// Demo the console says is being recorded (path as printed by the game)
    recording: Option<String>,
}

impl ConsoleDemoTracker {
    /// Largest part of the log looked through when the scanner starts.
    const MAX_INITIAL_WINDOW: u64 = 64 * 1024 * 1024;

    /// Processes what was added to the log since the last call; returns the demo that is being
    /// recorded, if the file exists, is a demo and has not been finished.
    fn poll(&mut self, tf_dir: &Path) -> Option<PathBuf> {
        let log_path = tf_dir.join("console.log");
        let len = std::fs::metadata(&log_path).ok()?.len();
        match self.pos {
            Some(pos) if len >= pos => {
                if len > pos {
                    if let Some(text) = read_log_since(&log_path, pos) {
                        // Only whole lines; the game may be in the middle of writing one
                        let complete = text.rfind('\n').map_or(0, |i| i + 1);
                        self.process(&text[..complete]);
                        self.pos = Some(pos + text[..complete].len() as u64);
                    }
                }
            }
            _ => {
                // First read (or the log was replaced): look back until a recording event shows up
                self.recording = None;
                let mut window: u64 = 1024 * 1024;
                loop {
                    let start = len.saturating_sub(window);
                    let Some(text) = read_log_since(&log_path, start) else { break };
                    // Skip the (possibly cut) first line
                    let begin = if start > 0 { text.find('\n').map_or(text.len(), |i| i + 1) } else { 0 };
                    let complete = text.rfind('\n').map_or(0, |i| i + 1).max(begin);
                    let found = self.process(&text[begin..complete]);
                    if found || start == 0 || window >= Self::MAX_INITIAL_WINDOW {
                        self.pos = Some(start + complete as u64);
                        break;
                    }
                    window *= 4;
                }
            }
        }
        let name = self.recording.as_deref()?;
        let path = tf_dir.join(name);
        let meta = path.metadata().ok()?;
        (meta.len() >= 1072 && has_demo_magic(&path) && !is_demo_recording_finished(&path)).then_some(path)
    }

    /// Returns whether a recording event was seen.
    fn process(&mut self, text: &str) -> bool {
        let mut found = false;
        for line in text.lines() {
            let line = strip_console_timestamp(line);
            if let Some(rest) = line.strip_prefix("Recording to ") {
                if let Some(name) = rest.trim_end().strip_suffix("...") {
                    if name.to_ascii_lowercase().ends_with(".dem") {
                        self.recording = Some(name.trim().to_string());
                        found = true;
                    }
                }
            } else if line.starts_with("Completed demo,") {
                self.recording = None;
                found = true;
            }
        }
        found
    }
}

fn has_demo_magic(path: &Path) -> bool {
    let Ok(mut f) = open_recording_file(path) else { return false };
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).is_ok() && &magic == b"HL2DEMO\0"
}

/// Strips optional console timestamps (e.g. `10/12/2026 - 01:55:17: `) from log lines.
pub fn strip_console_timestamp(line: &str) -> &str {
    let trimmed = line.trim();
    // Checked on bytes: lines can contain multi-byte characters (player names, chat), and slicing
    // the str at a byte index inside one of them panics.
    let bytes = trimmed.as_bytes();
    if bytes.len() >= 23
        && bytes[21] == b':'
        && bytes[22] == b' '
        && bytes[..10].contains(&b'/')
        && bytes[..20].contains(&b'-')
    {
        // Byte 22 is an ASCII space, so byte 23 is always a char boundary
        trimmed[23..].trim()
    } else {
        trimmed
    }
}

/// Parses the victim player name from kill or suicide messages in `console.log`.
/// Examples:
/// - `"Kim Shii killed LESNUK with sniperrifle."` -> `Some("LESNUK")`
/// - `"Shotgun killed ChuvachokGB with rocketlauncher_directhit. (crit)"` -> `Some("ChuvachokGB")`
/// - `"Kim Shii suicided."` -> `Some("Kim Shii")`
/// - `"Kim Shii died."` -> `Some("Kim Shii")`
pub fn parse_death_victim(clean_line: &str) -> Option<&str> {
    let s = clean_line.trim();
    if let Some(killed_idx) = s.find(" killed ") {
        let after_killed = &s[killed_idx + 8..];
        if let Some(with_idx) = after_killed.rfind(" with ") {
            let victim = after_killed[..with_idx].trim();
            if !victim.is_empty() {
                return Some(victim);
            }
        }
    }
    if let Some(suicided) = s.strip_suffix(" suicided.") {
        let victim = suicided.trim();
        if !victim.is_empty() {
            return Some(victim);
        }
    }
    if let Some(died) = s.strip_suffix(" died.") {
        let victim = died.trim();
        if !victim.is_empty() {
            return Some(victim);
        }
    }
    None
}

/// Tracks TF2 `console.log` from the current file end to detect player death events.
pub struct DeathTracker {
    console_log_path: PathBuf,
    last_seek_pos: u64,
}

impl DeathTracker {
    pub fn new(console_log_path: PathBuf) -> Self {
        let current_len = std::fs::metadata(&console_log_path)
            .map(|m| m.len())
            .unwrap_or(0);
        Self {
            console_log_path,
            last_seek_pos: current_len,
        }
    }

    pub fn reset_to_end(&mut self) {
        if let Ok(meta) = std::fs::metadata(&self.console_log_path) {
            self.last_seek_pos = meta.len();
        }
    }

    /// Reads newly appended lines and returns `true` if `recorder_name` died.
    pub fn check_death(&mut self, recorder_name: &str) -> bool {
        let mut file = match open_recording_file(&self.console_log_path) {
            Ok(f) => f,
            Err(_) => return false,
        };

        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        if file_len < self.last_seek_pos {
            // Log file was truncated/cleared
            self.last_seek_pos = 0;
        }

        if file.seek(SeekFrom::Start(self.last_seek_pos)).is_err() {
            return false;
        }

        let mut buf = Vec::new();
        if file.read_to_end(&mut buf).is_err() {
            return false;
        }

        self.last_seek_pos += buf.len() as u64;

        if buf.is_empty() {
            return false;
        }

        let text = String::from_utf8_lossy(&buf);
        let mut died = false;

        for line in text.lines() {
            let clean = strip_console_timestamp(line);
            if let Some(victim) = parse_death_victim(clean) {
                let v = victim.trim().trim_matches('"');
                let r = recorder_name.trim().trim_matches('"');
                if v.eq_ignore_ascii_case(r) || v == r {
                    died = true;
                    break;
                }
            }
        }

        died
    }
}

/// Builds algorithm instances based on the GUI selection and parameter overrides, without the
/// algorithms live scans skip.
pub fn build_configured_algorithms(
    algos_map: &std::collections::HashMap<String, bool>,
    params_map: &std::collections::HashMap<String, Parameters>,
) -> Vec<Box<dyn CheatAlgorithm<'static> + Send>> {
    let mut algorithms = Vec::new();
    for mut algo in get_algorithms() {
        let name = algo.algorithm_name().to_string();
        let enabled = algos_map.get(&name).copied().unwrap_or_else(|| algo.default_enabled());
        if !enabled || algo.skipped_in_live_scan() {
            continue;
        }
        if let Some(algo_params) = algo.params() {
            if let Some(p) = params_map.get(&name) {
                algo_params.clone_from(p);
            }
        }
        algorithms.push(algo);
    }
    algorithms
}

/// Makes sure the vote algorithm runs in the scan, even if the user has not enabled it.
pub fn ensure_vote_algorithm(algorithms: &mut Vec<Box<dyn CheatAlgorithm<'static> + Send>>) {
    if !algorithms.iter().any(|a| a.algorithm_name() == VOTE_ALGORITHM) {
        match get_algorithms().into_iter().find(|a| a.algorithm_name() == VOTE_ALGORITHM) {
            Some(algo) => algorithms.push(algo),
            None => {
                live_log!("[Live Scanner] {VOTE_ALGORITHM} is not part of this build — no vote notifications");
                return;
            }
        }
    }
    if let Some(params) = algorithms
        .iter_mut()
        .find(|a| a.algorithm_name() == VOTE_ALGORITHM)
        .and_then(|a| a.params())
    {
        params.insert("enabled".to_string(), Parameter::Bool(true));
    }
}

/// Sends each line to the party chat, spaced out so the game does not drop any.
fn send_party_lines(rcon_client: &mut RconClient, lines: &[String]) {
    if !rcon_client.is_connected() {
        return;
    }
    for line in lines {
        live_log!("[Live Scanner] Party alert sent: {line}");
        if let Err(e) = rcon_client.send_party_chat(line) {
            debug_log!("[Live Scanner] Party chat send failed: {e}");
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

/// `--debug`: checks that RCON commands really run in TF2 by echoing a unique line into its console,
/// then looks for the line in the RCON reply and in `console.log` (which death detection, recording
/// detection and `status` parsing read).
fn verify_rcon_console(rcon_client: &mut RconClient) {
    let nonce = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    );
    let marker = format!("[Demo Analysis] RCON debug check {nonce}");
    let log_path = find_tf2_root_dir().map(|d| d.join("console.log"));
    let start_pos = log_path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len());
    debug_log!("[RCON check] Echoing \"{marker}\" into the TF2 console");
    match rcon_client.send_command(&format!("echo \"{marker}\"")) {
        Ok(reply) => debug_log!(
            "[RCON check] RCON reply: {:?} -> {}",
            reply.trim(),
            if reply.contains(&nonce) {
                "contains the marker: TF2 ran the command"
            } else {
                "marker NOT in the reply (TF2 may still have run it, see console.log below)"
            }
        ),
        Err(e) => {
            debug_log!("[RCON check] FAILED to send the echo command: {e}");
            return;
        }
    }
    let Some(log_path) = log_path else {
        debug_log!("[RCON check] TF2 folder not found, console.log cannot be checked");
        return;
    };
    let started = Instant::now();
    loop {
        if read_log_since(&log_path, start_pos.unwrap_or(0)).map_or(false, |t| t.contains(&nonce)) {
            debug_log!(
                "[RCON check] OK: the marker appeared in {} after {} ms (RCON commands run in TF2 and console.log is written)",
                log_path.display(),
                started.elapsed().as_millis()
            );
            return;
        }
        if started.elapsed() >= Duration::from_secs(3) {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    debug_log!(
        "[RCON check] FAILED: the marker did not appear in {} within 3 s ({}). Death detection, recording detection and marked cheater warnings will not work; \
         check that this is the folder of the running TF2 or add -condebug to the launch options.",
        log_path.display(),
        match std::fs::metadata(&log_path) {
            Ok(m) => format!("file is {} bytes, was {:?}", m.len(), start_pos),
            Err(e) => format!("file not available: {e}"),
        }
    );
}

/// Background worker managing real-time demo detection, death tracking, scanning, and RCON notifications.
pub struct RealTimeScanner {
    enabled: Arc<AtomicBool>,
    config: Arc<Mutex<RealTimeScanConfig>>,
    algos: Arc<Mutex<std::collections::HashMap<String, bool>>>,
    params: Arc<Mutex<std::collections::HashMap<String, Parameters>>>,
    reported_ids: Arc<Mutex<HashSet<u64>>>,
    /// Players announced in party chat when they are on the server: steam id → their tags
    /// (e.g. attributes from TF2BD playerlists).
    tagged_players: Arc<Mutex<std::collections::HashMap<u64, String>>>,
    /// Holds the scans back while a manual rescan of the whole demo is running and receives its results.
    gate: Arc<LiveScanGate>,
    tx: std::sync::mpsc::Sender<RealTimeScanEvent>,
}

impl RealTimeScanner {
    pub fn new(
        enabled: Arc<AtomicBool>,
        config: Arc<Mutex<RealTimeScanConfig>>,
        algos: Arc<Mutex<std::collections::HashMap<String, bool>>>,
        params: Arc<Mutex<std::collections::HashMap<String, Parameters>>>,
        reported_ids: Arc<Mutex<HashSet<u64>>>,
        tagged_players: Arc<Mutex<std::collections::HashMap<u64, String>>>,
        gate: Arc<LiveScanGate>,
        tx: std::sync::mpsc::Sender<RealTimeScanEvent>,
    ) -> Self {
        Self {
            enabled,
            config,
            algos,
            params,
            reported_ids,
            tagged_players,
            gate,
            tx,
        }
    }

    /// Takes over the results of a manual rescan of `demo_path`, if there are any: they replace all
    /// detections found so far and the next scan starts where the rescan ended.
    fn adopt_manual_rescan(&self, live_progress: &mut LiveScanProgress, demo_path: &Path) {
        if let Some(rescan) = self.gate.take_rescan() {
            if rescan.demo_path == demo_path {
                println!(
                    "[Live Scanner] Manual rescan adopted ({} detections, up to tick {})",
                    rescan.detections.len(),
                    rescan.analysed_until
                );
                live_progress.adopt_rescan(rescan);
            }
        }
    }

    /// Spawns the background scanner thread.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("live-scanner".to_string())
            .spawn(move || {
                // A panic must not end live scanning and RCON for the rest of the session: the loop
                // starts over (reconnecting RCON, picking the recording up again) after a pause.
                loop {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_loop())) {
                        Ok(()) => break,
                        Err(panic) => {
                            let msg = panic
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| panic.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| "unknown error".to_string());
                            live_elog!("[Live Scanner] Crashed ({msg}), restarting");
                            let _ = self.tx.send(RealTimeScanEvent::RconConnected(false));
                            let _ = self
                                .tx
                                .send(RealTimeScanEvent::RconStatus(format!("Live scanner crashed, restarting: {msg}")));
                            std::thread::sleep(Duration::from_secs(2));
                        }
                    }
                }
            })
            .expect("Failed to spawn live scanner thread")
    }

    fn run_loop(&self) {
        let mut rcon_client = RconClient::default();
        let mut prev_rcon_connected: Option<bool> = None;
        let mut active_demo: Option<PathBuf> = None;
        let mut recorder_name: Option<String> = None;
        // Recorder resolved from the scanned demo, used when Steam's logged in account is unknown
        let mut recorder_id_from_demo: Option<u64> = None;
        let mut death_tracker: Option<DeathTracker> = None;
        let mut notified_detections: HashSet<(u64, String)> = HashSet::new();
        let mut live_progress = LiveScanProgress::default();
        // Only exists while vote notifications are enabled
        let mut vote_notifier: Option<VoteNotifier> = None;
        let mut last_scan_time = Instant::now();
        let mut last_death_scan_time = Instant::now();
        // A scan was due while a manual rescan was running and starts as soon as that has finished
        let mut scan_pending = false;
        let mut last_reconnect_attempt = Instant::now() - Duration::from_secs(60);
        let mut reconnect_backoff = Duration::from_secs(5);
        // Marked cheaters and tagged players currently on the server that were already announced.
        let mut warned_marked: HashSet<u64> = HashSet::new();
        let mut warned_tagged: HashSet<u64> = HashSet::new();
        let mut last_status_request = Instant::now() - STATUS_POLL_INTERVAL;
        // (console.log position before `status` was sent, time it was sent)
        let mut pending_status: Option<(u64, Instant)> = None;
        let mut console_demo_tracker = ConsoleDemoTracker::default();
        // `--debug`: last logged states, so the log only gets a line when something changes
        let mut dbg_enabled: Option<bool> = None;
        let mut dbg_config = String::new();
        let mut dbg_tf_dir: Option<Option<PathBuf>> = None;
        let mut dbg_console_log: Option<String> = None;
        let mut dbg_console_demo: Option<Option<PathBuf>> = None;
        let mut dbg_fallback_demo: Option<Option<PathBuf>> = None;
        let mut dbg_no_recorder: Option<PathBuf> = None;
        let mut dbg_last_search_trace = Instant::now() - Duration::from_secs(3600);
        debug_log!("[Live Scanner] Thread started");

        while !std::thread::panicking() {
            std::thread::sleep(Duration::from_millis(600));

            let enabled = self.enabled.load(Ordering::Relaxed);
            if dbg_enabled != Some(enabled) {
                dbg_enabled = Some(enabled);
                debug_log!("[Live Scanner] Live scanning {}", if enabled { "ENABLED" } else { "disabled" });
            }
            if !enabled {
                if rcon_client.is_connected() {
                    rcon_client.disconnect();
                    let _ = self.tx.send(RealTimeScanEvent::RconStatus("Disconnected".to_string()));
                    let _ = self.tx.send(RealTimeScanEvent::RconConnected(false));
                }
                prev_rcon_connected = Some(false);
                active_demo = None;
                recorder_name = None;
                recorder_id_from_demo = None;
                death_tracker = None;
                notified_detections.clear();
                live_progress = LiveScanProgress::default();
                scan_pending = false;
                let _ = self.gate.take_rescan();
                vote_notifier = None;
                warned_marked.clear();
                warned_tagged.clear();
                pending_status = None;
                console_demo_tracker = ConsoleDemoTracker::default();
                reconnect_backoff = Duration::from_secs(5);
                last_reconnect_attempt = Instant::now() - Duration::from_secs(60);
                continue;
            }

            let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
            if crate::util::debug_log::enabled() {
                let summary = format!(
                    "port={} password={} scan_on_death={} scan_periodic_too={} interval={}s party_chat_alerts={} \
                     warn_marked_cheaters={} vote_notifications={} ignore_recorder={}",
                    cfg.port,
                    if cfg.password.trim().is_empty() { "empty".to_string() } else { format!("{} chars", cfg.password.chars().count()) },
                    cfg.scan_on_death,
                    cfg.scan_periodic_too,
                    cfg.scan_interval_seconds,
                    cfg.party_chat_alerts,
                    cfg.warn_marked_cheaters,
                    cfg.vote_notifications,
                    cfg.ignore_recorder
                );
                if summary != dbg_config {
                    debug_log!("[Live Scanner] Settings: {summary}");
                    dbg_config = summary;
                }
                let tf_dir = find_tf2_root_dir();
                if dbg_tf_dir.as_ref() != Some(&tf_dir) {
                    debug_log!("[Live Scanner] TF2 folder: {:?}", tf_dir);
                    dbg_tf_dir = Some(tf_dir.clone());
                }
                let log_state = match tf_dir.as_ref().map(|d| d.join("console.log")) {
                    None => "no TF2 folder".to_string(),
                    Some(p) => match open_recording_file(&p) {
                        Ok(_) => "readable".to_string(),
                        Err(e) => format!("NOT readable: {e}"),
                    },
                };
                if dbg_console_log.as_ref() != Some(&log_state) {
                    debug_log!("[Live Scanner] console.log: {log_state}");
                    dbg_console_log = Some(log_state);
                }
            }
            let ignored_recorder = if cfg.ignore_recorder {
                local_steam_id().or(recorder_id_from_demo)
            } else {
                None
            };

            // Turning the setting on mid-demo only announces the votes from then on.
            if cfg.vote_notifications {
                vote_notifier.get_or_insert_with(VoteNotifier::new);
            } else {
                vote_notifier = None;
            }

            // Maintain RCON connection
            let target_addr = format!("127.0.0.1:{}", cfg.port);
            if rcon_client.config.server_address != target_addr
                || rcon_client.config.password != cfg.password
            {
                rcon_client.disconnect();
                rcon_client.config.server_address = target_addr;
                rcon_client.config.password = cfg.password.clone();
                reconnect_backoff = Duration::from_secs(5);
                last_reconnect_attempt = Instant::now() - Duration::from_secs(60);
            }

            if !rcon_client.is_connected() {
                if cfg.password.trim().is_empty() {
                    if prev_rcon_connected != Some(false) {
                        debug_log!("[Live Scanner] No RCON password configured: not connecting (scans fall back to the interval)");
                        let _ = self.tx.send(RealTimeScanEvent::RconStatus("No password configured".to_string()));
                    }
                } else if last_reconnect_attempt.elapsed() >= reconnect_backoff {
                    last_reconnect_attempt = Instant::now();
                    match rcon_client.connect() {
                        Ok(()) => {
                            debug_log!("[Live Scanner] RCON connected to {}", rcon_client.config.server_address);
                            if let Err(e) = rcon_client.send_command("con_logfile console.log") {
                                debug_log!("[Live Scanner] Sending con_logfile failed: {e}");
                            }
                            if crate::util::debug_log::enabled() && rcon_client.is_connected() {
                                verify_rcon_console(&mut rcon_client);
                            }
                            let _ = self.tx.send(RealTimeScanEvent::RconStatus("Connected ✓".to_string()));
                            reconnect_backoff = Duration::from_secs(5);
                        }
                        Err(e) if e.contains("invalid password") => {
                            // TF2 bans 127.0.0.1 from RCON (until it is restarted) after a few wrong
                            // passwords, so a wrong one is only retried once the password or port changes.
                            reconnect_backoff = Duration::MAX;
                            live_elog!("[Live Scanner] {e} — not retrying until the RCON settings change");
                            let _ = self.tx.send(RealTimeScanEvent::RconStatus(format!(
                                "Disconnected: {e} (not retried until the RCON settings change)"
                            )));
                        }
                        Err(e) => {
                            reconnect_backoff = reconnect_backoff.saturating_mul(2).min(Duration::from_secs(30));
                            live_elog!("[Live Scanner] RCON connection failed: {e}");
                            let _ = self.tx.send(RealTimeScanEvent::RconStatus(format!(
                                "Disconnected (retry in {}s): {}",
                                reconnect_backoff.as_secs(),
                                e
                            )));
                        }
                    }
                }
            }

            let currently_connected = rcon_client.is_connected();
            if prev_rcon_connected != Some(currently_connected) {
                prev_rcon_connected = Some(currently_connected);
                let _ = self.tx.send(RealTimeScanEvent::RconConnected(currently_connected));
            }

            // Warn about marked cheaters and tagged players present on the server
            let watch_tagged = !self.tagged_players.lock().unwrap_or_else(|e| e.into_inner()).is_empty();
            if !currently_connected || !(cfg.warn_marked_cheaters || watch_tagged) {
                warned_marked.clear();
                warned_tagged.clear();
                pending_status = None;
            } else if let Some(log_path) = find_tf2_root_dir().map(|d| d.join("console.log")) {
                if let Some((pos, sent_at)) = pending_status {
                    if sent_at.elapsed() >= STATUS_REPLY_DELAY {
                        pending_status = None;
                        let status_text = read_log_since(&log_path, pos);
                        debug_log!(
                            "[Live Scanner] status reply: {}",
                            match &status_text {
                                Some(t) => format!("{} bytes in console.log, {} players parsed", t.len(), parse_status_players(t).len()),
                                None => format!("could not read {}", log_path.display()),
                            }
                        );
                        if let Some(text) = status_text {
                            let players = parse_status_players(&text);
                            if !players.is_empty() {
                                let marked = if cfg.warn_marked_cheaters {
                                    self.reported_ids.lock().unwrap_or_else(|e| e.into_inner()).clone()
                                } else {
                                    HashSet::new()
                                };
                                let tags: Vec<Option<String>> = {
                                    let tagged = self.tagged_players.lock().unwrap_or_else(|e| e.into_inner());
                                    players.iter().map(|(id, _)| tagged.get(id).cloned()).collect()
                                };
                                let present: HashSet<u64> = players.iter().map(|(id, _)| *id).collect();
                                // Forget players who left so they are announced again if they rejoin.
                                warned_marked.retain(|id| present.contains(id));
                                warned_tagged.retain(|id| present.contains(id));
                                let mut announced = Vec::new();
                                for ((id, name), tag) in players.iter().zip(tags) {
                                    if ignored_recorder == Some(*id) {
                                        continue;
                                    }
                                    let is_marked = marked.contains(id) && warned_marked.insert(*id);
                                    let tag = tag.filter(|_| warned_tagged.insert(*id));
                                    let alert = match (is_marked, tag) {
                                        (true, Some(tag)) => format!("[Analyser] Marked cheater on server ({tag}): {name}"),
                                        (true, None) => format!("[Analyser] Marked cheater on server: {name}"),
                                        // TF2BD playerlist entries are not necessarily cheaters
                                        (false, Some(tag)) => format!("[Analyser] Marked player on server ({tag}): {name}"),
                                        (false, None) => continue,
                                    };
                                    live_log!("[Live Scanner] Party alert sent: {alert}");
                                    if let Err(e) = rcon_client.send_party_chat(&alert) {
                                        debug_log!("[Live Scanner] Party chat send failed: {e}");
                                    }
                                    announced.push(name.clone());
                                    std::thread::sleep(Duration::from_millis(300));
                                }
                                if !announced.is_empty() {
                                    let _ = self.tx.send(RealTimeScanEvent::MarkedCheatersOnServer(announced));
                                }
                            }
                        }
                    }
                } else if last_status_request.elapsed() >= STATUS_POLL_INTERVAL {
                    last_status_request = Instant::now();
                    let pos = std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
                    match rcon_client.send_command("status") {
                        Ok(_) => pending_status = Some((pos, Instant::now())),
                        Err(e) => debug_log!("[Live Scanner] status command failed: {e}"),
                    }
                }
            }

            // The demo the console says is being recorded (right from the start, whatever the file's age)
            let console_demo = find_tf2_root_dir().and_then(|dir| console_demo_tracker.poll(&dir));
            if dbg_console_demo.as_ref() != Some(&console_demo) {
                debug_log!(
                    "[Live Scanner] console.log says recording: {:?} (last 'Recording to' line: {:?})",
                    console_demo,
                    console_demo_tracker.recording
                );
                dbg_console_demo = Some(console_demo.clone());
            }

            // Check if active demo is still being recorded
            let still_recording = if let Some(ref path) = active_demo {
                if let Ok(meta) = path.metadata() {
                    // A newer recording announced in the console replaces the active one
                    console_demo.as_ref().map_or(true, |c| c == path)
                        && meta.len() >= 1072
                        && (console_demo.as_ref() == Some(path)
                            || meta.modified().map_or(false, |m| m.elapsed().map_or(false, |e| e.as_secs() <= 90)))
                        && !is_demo_recording_finished(path)
                } else {
                    false
                }
            } else {
                false
            };

            if !still_recording {
                if let Some(path) = active_demo.take() {
                    live_log!("[Live Scanner] Recording stopped: {} — no longer scanning it", path.display());
                    // Announce a votekick that was still going on when the recording ended
                    if let Some(notifier) = vote_notifier.as_mut() {
                        let lines = notifier.flush(&live_progress.raw_detections);
                        send_party_lines(&mut rcon_client, &lines);
                    }
                    vote_notifier = None;
                    recorder_name = None;
                    recorder_id_from_demo = None;
                    death_tracker = None;
                    notified_detections.clear();
                    live_progress = LiveScanProgress::default();
                    scan_pending = false;
                    let _ = self.tx.send(RealTimeScanEvent::RecordingStopped(path));
                }
            }

            let current_demo = if still_recording {
                active_demo.clone()
            } else {
                // Console output first, the newest demo file being written as fallback
                console_demo.clone().or_else(|| {
                    let trace_now = crate::util::debug_log::enabled()
                        && dbg_last_search_trace.elapsed() >= Duration::from_secs(30);
                    let found = if trace_now {
                        dbg_last_search_trace = Instant::now();
                        find_recording_demo_traced(&mut |line| debug_log!("[Live Scanner] {line}"))
                    } else {
                        find_recording_demo()
                    };
                    if dbg_fallback_demo.as_ref() != Some(&found) {
                        debug_log!("[Live Scanner] Recording demo by file time: {:?}", found);
                        dbg_fallback_demo = Some(found.clone());
                    }
                    found
                })
            };

            let demo_path = match current_demo {
                Some(p) => p,
                None => {
                    active_demo = None;
                    recorder_name = None;
                    continue;
                }
            };

            // If a new recording demo was started:
            if active_demo.as_ref() != Some(&demo_path) {
                active_demo = Some(demo_path.clone());
                notified_detections.clear();
                live_progress = LiveScanProgress::default();
                scan_pending = false;
                vote_notifier = cfg.vote_notifications.then(VoteNotifier::new);
                recorder_name = get_demo_recorder_name(&demo_path);
                recorder_id_from_demo = None;

                if let Some(tf_dir) = find_tf2_root_dir() {
                    let log_path = tf_dir.join("console.log");
                    death_tracker = Some(DeathTracker::new(log_path));
                }

                last_scan_time = Instant::now();
                last_death_scan_time = Instant::now();
                live_log!("[Live Scanner] Detected recording demo: {} (Recorder: {:?})", demo_path.display(), recorder_name);
                let _ = self.tx.send(RealTimeScanEvent::DemoDetected {
                    demo_path: demo_path.clone(),
                    recorder_name: recorder_name.clone(),
                });
            }

            // A manual rescan of the whole demo (Analyse button) replaces everything found so far.
            // The next scan then covers what was recorded after the point the rescan ended at.
            self.adopt_manual_rescan(&mut live_progress, &demo_path);

            // Check if scan should trigger
            let mut should_scan = scan_pending;
            let interval = cfg.scan_interval_seconds.max(5);

            if cfg.scan_on_death && currently_connected {
                // When scan_on_death is enabled and RCON is working as intended:
                // Wait for player death; periodic scanning only runs if `scan_periodic_too` is set
                // (a death scan resets the interval timer, so whichever comes first wins).
                if cfg.scan_periodic_too && last_scan_time.elapsed().as_secs() >= interval {
                    should_scan = true;
                }
                if let Some(ref mut dt) = death_tracker {
                    let rec_name = recorder_name.as_deref().unwrap_or("");
                    if rec_name.is_empty() && dbg_no_recorder.as_ref() != Some(&demo_path) {
                        debug_log!("[Live Scanner] Recorder name unknown (empty demo header nick): death detection cannot work");
                        dbg_no_recorder = Some(demo_path.clone());
                    }
                    if !rec_name.is_empty() && dt.check_death(rec_name) {
                        // Debounce death triggers with 5-second cooldown
                        if last_death_scan_time.elapsed().as_secs() >= 5 {
                            live_log!("[Live Scanner] Player death detected for '{rec_name}' — triggering scan");
                            let _ = self.tx.send(RealTimeScanEvent::DeathDetected { victim: rec_name.to_string() });
                            should_scan = true;
                            last_death_scan_time = Instant::now();
                        }
                    }
                }
            } else {
                // Either scan_on_death is disabled OR RCON is not connected:
                // Fall back to periodic scanning on the configured interval.
                if last_scan_time.elapsed().as_secs() >= interval {
                    if cfg.scan_on_death && !currently_connected {
                        live_log!("[Live Scanner] RCON not connected — falling back to periodic scan (every {}s)", interval);
                    }
                    should_scan = true;
                }
            }

            if should_scan {
                // Live scans wait while a manual rescan of the whole demo is running
                let Some(_scan_permit) = self.gate.try_begin_live_scan() else {
                    scan_pending = true;
                    continue;
                };
                scan_pending = false;
                last_scan_time = Instant::now();
                // A rescan that finished since the check above
                self.adopt_manual_rescan(&mut live_progress, &demo_path);
                let scan_epoch = self.gate.epoch();

                let bytes = match read_recording_demo_bytes(&demo_path) {
                    Ok(b) if b.len() >= 1072 => b,
                    Ok(b) => {
                        debug_log!("[Live Scanner] Demo too short to scan yet ({} bytes): {}", b.len(), demo_path.display());
                        continue;
                    }
                    Err(e) => {
                        debug_log!("[Live Scanner] Reading {} FAILED: {e} (kind {:?}, os error {:?})", demo_path.display(), e.kind(), e.raw_os_error());
                        continue;
                    }
                };

                let general_settings = {
                    let p_map = self.params.lock().unwrap_or_else(|e| e.into_inner());
                    GeneralSettings::from_config(&p_map)
                };

                let (mut algorithms, votes_enabled_by_user) = {
                    let a_map = self.algos.lock().unwrap_or_else(|e| e.into_inner());
                    let p_map = self.params.lock().unwrap_or_else(|e| e.into_inner());
                    (
                        build_configured_algorithms(&a_map, &p_map),
                        a_map.get(VOTE_ALGORITHM).copied().unwrap_or(false),
                    )
                };
                if vote_notifier.is_some() {
                    ensure_vote_algorithm(&mut algorithms);
                }

                if algorithms.is_empty() {
                    debug_log!("[Live Scanner] No algorithms enabled for live scans, nothing to scan");
                    continue;
                }

                let mut algorithm_names: Vec<String> =
                    algorithms.iter().map(|a| a.algorithm_name().to_string()).collect();
                algorithm_names.sort();
                // These run on the whole demo in every scan, the others only on the part since the last
                let whole_demo_algorithms: HashSet<String> = algorithms
                    .iter()
                    .filter(|a| a.needs_whole_demo())
                    .map(|a| a.algorithm_name().to_string())
                    .collect();
                if bytes.len() < live_progress.analysed_bytes {
                    // The file shrank: a new recording replaced the old one under the same name.
                    live_progress = LiveScanProgress::default();
                    notified_detections.clear();
                    vote_notifier = cfg.vote_notifications.then(VoteNotifier::new);
                }
                let from_tick = live_progress.next_start_tick(&algorithm_names);

                println!(
                    "[Live Scanner] Scanning live demo: {} ({} KB) from tick {}",
                    demo_path.display(),
                    bytes.len() / 1024,
                    from_tick
                );
                let _ = self.tx.send(RealTimeScanEvent::ScanStarted(demo_path.clone()));

                // Live scans run single-threaded on this thread so they don't compete with the game
                // (or a manual analysis) for the CPU; only manual analyses use the worker pool.
                let scan_started = Instant::now();
                let scan_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let demo = Demo::new(&bytes);
                    analyse_sequential_from_tick(&demo, algorithms, from_tick).map(|mut analyser| {
                        analyser.algorithms.clear();
                        analyser
                    })
                }));

                match &scan_res {
                    Ok(Ok(_)) => debug_log!("[Live Scanner] Analysis took {} ms", scan_started.elapsed().as_millis()),
                    Ok(Err(e)) => debug_log!("[Live Scanner] Analysis FAILED after {} ms: {e}", scan_started.elapsed().as_millis()),
                    Err(panic) => debug_log!(
                        "[Live Scanner] Analysis PANICKED after {} ms: {}",
                        scan_started.elapsed().as_millis(),
                        panic
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown error".to_string())
                    ),
                }
                if let Ok(Ok(mut analyser)) = scan_res {
                    // Still in the signon data: the next scan starts from the beginning again
                    let scanned_until = analysed_until(&analyser);
                    if let Some(until) = scanned_until {
                        live_progress.analysed_until =
                            Some(live_progress.analysed_until.map_or(until, |t| t.max(until)));
                    }
                    let analysed_until = scanned_until.unwrap_or(0);
                    live_progress.analysed_bytes = bytes.len();
                    live_progress.algorithm_names = algorithm_names;
                    if recorder_id_from_demo.is_none() {
                        recorder_id_from_demo = demo_recorder_id(&analyser);
                    }
                    let ignored_recorder = ignored_recorder.or(recorder_id_from_demo.filter(|_| cfg.ignore_recorder));
                    let all_detections = live_progress.merge(
                        std::mem::take(&mut analyser.detections),
                        from_tick,
                        &whole_demo_algorithms,
                    );
                    let vote_lines = vote_notifier
                        .as_mut()
                        .map(|notifier| {
                            let lines = notifier.collect(all_detections, analysed_until);
                            // Every individual vote goes to the console only, never the party chat
                            for line in notifier.console_lines(all_detections) {
                                live_log!("[Live Scanner] {line}");
                            }
                            lines
                        })
                        .unwrap_or_default();
                    analyser.detections = general_settings.filter_detections(all_detections);
                    if !votes_enabled_by_user {
                        // Only run for the notifications, the votes are not flags
                        analyser.detections.retain(|d| d.algorithm != VOTE_ALGORITHM);
                    }

                    let mut new_notifications = Vec::new();
                    for det in analyser
                        .detections
                        .iter()
                        .filter(|d| d.algorithm != VOTE_ALGORITHM && ignored_recorder != Some(d.player))
                    {
                        let key = (det.player, det.algorithm.clone());
                        if !notified_detections.contains(&key) {
                            notified_detections.insert(key);
                            let player_name = analyser
                                .state
                                .get_player_name(det.player)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| det.player.to_string());
                            new_notifications.push((player_name, det.algorithm.clone(), det.tick));
                        }
                    }

                    // Send party chat alerts through RCON ONLY if enabled and RCON is connected
                    let alerts_sent = if cfg.party_chat_alerts && !new_notifications.is_empty() && rcon_client.is_connected() {
                        for (player_name, algo, _tick) in &new_notifications {
                            let alert = format!("[Analyser] {} flagged for {}", player_name, algo);
                            live_log!("[Live Scanner] Party alert sent: {alert}");
                            if let Err(e) = rcon_client.send_party_chat(&alert) {
                                        debug_log!("[Live Scanner] Party chat send failed: {e}");
                                    }
                            std::thread::sleep(Duration::from_millis(300));
                        }
                        true
                    } else {
                        false
                    };
                    send_party_lines(&mut rcon_client, &vote_lines);

                    println!(
                        "[Live Scanner] Scan finished ({} detections, {} new notifications, alerts_sent={})",
                        analyser.detections.len(),
                        new_notifications.len(),
                        alerts_sent
                    );
                    let _ = self.tx.send(RealTimeScanEvent::ScanFinished {
                        demo_path: demo_path.clone(),
                        analyser,
                        new_notifications,
                        alerts_sent,
                        epoch: scan_epoch,
                    });
                }

                // Scans are sequential, so one that is due while another is running is skipped:
                // the interval restarts when this scan has finished (not when it started, which
                // would start the next one right away after a slow scan) and deaths that happened
                // meanwhile don't trigger a scan either. The next scan continues from where this
                // one ended, so nothing is lost.
                last_scan_time = Instant::now();
                last_death_scan_time = Instant::now();
                if let (Some(dt), Some(rec_name)) = (death_tracker.as_mut(), recorder_name.as_deref()) {
                    if !rec_name.is_empty() {
                        let _ = dt.check_death(rec_name);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_console_timestamp() {
        let input = "10/12/2026 - 01:55:17: Kim Shii killed LESNUK with sniperrifle.";
        let expected = "Kim Shii killed LESNUK with sniperrifle.";
        assert_eq!(strip_console_timestamp(input), expected);

        let input_no_ts = "Kim Shii killed LESNUK with sniperrifle.";
        assert_eq!(strip_console_timestamp(input_no_ts), expected);

        // Multi-byte characters around the byte positions checked must not panic
        // A 2-byte char at bytes 19-20 (the reported panic), and one at bytes 9-10
        let at_20 = format!("{}é: hello", "a".repeat(19));
        assert_eq!(at_20.as_bytes()[21], b':');
        assert_eq!(strip_console_timestamp(&at_20), at_20);
        let at_10 = format!("{}é{}: hello", "a".repeat(9), "-".repeat(10));
        assert_eq!(at_10.as_bytes()[21], b':');
        assert_eq!(strip_console_timestamp(&at_10), at_10);
        for line in ["ééééééééééé: x", "a/éééééééé-ééé: yyyyyyyyyyy", "🙂🙂🙂🙂🙂-🙂🙂/ : zzzz"] {
            let _ = strip_console_timestamp(line);
        }
    }

    #[test]
    fn test_demo_recorder_id() {
        let mut analyser = CheatAnalyser::default();
        assert_eq!(demo_recorder_id(&analyser), None);
        analyser.state.header = Some(tf_demo_parser::demo::header::Header {
            demo_type: "HL2DEMO".to_string(),
            version: 3,
            protocol: 24,
            server: String::new(),
            nick: "Kim Shii\0".to_string(),
            map: String::new(),
            game: "tf".to_string(),
            duration: 0.0,
            ticks: 0,
            frames: 0,
            signon: 0,
        });
        analyser.state.player_names.insert(1, "LESNUK".to_string());
        analyser.state.player_names.insert(2, "Kim Shii".to_string());
        assert_eq!(demo_recorder_id(&analyser), Some(2));
        // A name stealer makes the recorder ambiguous
        analyser.state.player_names.insert(3, "Kim Shii".to_string());
        assert_eq!(demo_recorder_id(&analyser), None);
    }

    #[test]
    fn test_is_demo_recording_finished() {
        let path = std::env::temp_dir().join(format!("live_scan_test_{}.dem", std::process::id()));
        let mut header = vec![0u8; 1100];
        header[..8].copy_from_slice(b"HL2DEMO\0");
        header[1068..1072].copy_from_slice(&350000i32.to_le_bytes()); // signon length
        std::fs::write(&path, &header).unwrap();
        // Still recording: playback ticks and frames are zero
        assert!(!is_demo_recording_finished(&path));

        header[1060..1064].copy_from_slice(&4000i32.to_le_bytes());
        header[1064..1068].copy_from_slice(&3900i32.to_le_bytes());
        std::fs::write(&path, &header).unwrap();
        assert!(is_demo_recording_finished(&path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_parse_status_players() {
        let status = "hostname: Test\nplayers : 2 humans, 1 bots (32 max)\n\
            # userid name                uniqueid            connected ping loss state\n\
            #      2 \"Kim \"Shii\"\"       [U:1:1234]          12:34       50    0 active\n\
            #      3 \"Bot\"               BOT                                     active\n\
            10/12/2026 - 01:55:17: #      4 \"LESNUK\"            [U:1:4294967295]    00:10       80    0 active\n";
        let players = parse_status_players(status);
        assert_eq!(players.len(), 2);
        assert_eq!(players[0], (STEAM_ID64_BASE + 1234, "Kim \"Shii\"".to_string()));
        assert_eq!(players[1].0, STEAM_ID64_BASE + 4294967295);
    }

    #[test]
    fn test_parse_death_victim() {
        assert_eq!(
            parse_death_victim("Kim Shii killed LESNUK with sniperrifle."),
            Some("LESNUK")
        );
        assert_eq!(
            parse_death_victim("Shotgun killed ChuvachokGB with rocketlauncher_directhit. (crit)"),
            Some("ChuvachokGB")
        );
        assert_eq!(
            parse_death_victim("Kim Shii suicided."),
            Some("Kim Shii")
        );
        assert_eq!(
            parse_death_victim("rlaxodnd died."),
            Some("rlaxodnd")
        );
        assert_eq!(
            parse_death_victim("Connected to 127.0.0.1:27015"),
            None
        );
    }
}
