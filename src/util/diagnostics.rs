//! Startup checks written to the debug log (`--debug`): where the TF2 folder was looked for, whether
//! its demos and `console.log` can actually be read (and the demos folder written), and the TF2
//! launch options / configs that RCON and live scanning depend on. Secrets are never logged: the
//! RCON password is only reported as set / matching or not.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::debug_log;
use crate::util::helpers::{find_default_tf2_dir_traced, get_steam_install_path};

/// Header size of a demo file.
const DEMO_HEADER_LEN: usize = 1072;

/// Opens a file in shared mode, like the live scanner does, so TF2's write handle does not block it.
fn open_shared(path: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        std::fs::OpenOptions::new().read(true).share_mode(7).open(path)
    }
    #[cfg(not(windows))]
    {
        std::fs::File::open(path)
    }
}

fn age(time: std::io::Result<SystemTime>) -> String {
    match time.map(|t| t.elapsed()) {
        Ok(Ok(d)) => {
            let s = d.as_secs();
            if s < 120 {
                format!("{s}s ago")
            } else if s < 7200 {
                format!("{}min ago", s / 60)
            } else if s < 172_800 {
                format!("{}h ago", s / 3600)
            } else {
                format!("{}d ago", s / 86_400)
            }
        }
        Ok(Err(_)) => "in the future".to_string(),
        Err(e) => format!("unknown ({e})"),
    }
}

/// Program version, executable, working folder and build features.
pub fn log_environment() {
    debug_log!("=== Demo Analysis {} debug log ===", env!("CARGO_PKG_VERSION"));
    debug_log!("Started: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S %z"));
    debug_log!("OS: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    debug_log!("Arguments: {:?}", std::env::args().skip(1).collect::<Vec<_>>());
    debug_log!("Executable: {:?}", std::env::current_exe());
    debug_log!("Working folder: {:?}", std::env::current_dir());
    debug_log!(
        "Features: rcon={} tf2bd={} experimental={} parallel={}",
        cfg!(feature = "rcon"),
        cfg!(feature = "tf2bd"),
        cfg!(feature = "experimental"),
        cfg!(feature = "parallel")
    );
    debug_log!("CPU threads: {}", num_cpus::get());
}

/// The `tf` folder for a demos folder found by `find_default_tf2_dir` (which returns `tf/demos`
/// when it exists, else `tf`).
pub fn tf_dir_of(demos_dir: &Path) -> PathBuf {
    if demos_dir.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("demos")) {
        demos_dir.parent().map_or_else(|| demos_dir.to_path_buf(), Path::to_path_buf)
    } else {
        demos_dir.to_path_buf()
    }
}

/// Logs the TF2 folder search, read/write access to the demos and `console.log`, and the TF2
/// launch options and configs relevant to RCON. `rcon_password` is the GUI's setting (compared,
/// never logged).
pub fn log_tf2_setup(rcon_password: Option<&str>) {
    debug_log!("--- TF2 folder detection ---");
    let demos_dir = find_default_tf2_dir_traced(&mut |line| debug_log!("  {line}"));
    let Some(demos_dir) = demos_dir else {
        debug_log!("RESULT: TF2 folder NOT found. Demo browsing and live scanning cannot work; set the TF2 folder manually in the settings.");
        log_steam_launch_options();
        return;
    };
    let tf_dir = tf_dir_of(&demos_dir);
    debug_log!("RESULT: demos folder = {}", demos_dir.display());
    debug_log!("RESULT: tf folder    = {}", tf_dir.display());
    if demos_dir == tf_dir {
        debug_log!("  note: no tf\\demos folder, demos are looked for in tf itself");
    }

    debug_log!("--- Folder access ---");
    check_folder(&tf_dir, "tf folder", false);
    if demos_dir != tf_dir {
        check_folder(&demos_dir, "demos folder", true);
    } else {
        check_write_access(&demos_dir, "tf folder");
    }

    debug_log!("--- Demo files ---");
    check_demo_files(&demos_dir, &tf_dir);

    debug_log!("--- console.log (live scanning: death detection, recording detection, status) ---");
    check_console_log(&tf_dir);

    debug_log!("--- TF2 configs ---");
    check_configs(&tf_dir, rcon_password);

    log_steam_launch_options();
}

fn check_folder(dir: &Path, label: &str, check_write: bool) {
    match std::fs::metadata(dir) {
        Ok(meta) => debug_log!("{label}: exists, read-only attribute={}", meta.permissions().readonly()),
        Err(e) => {
            debug_log!("{label}: metadata FAILED: {e}");
            return;
        }
    }
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let (mut ok, mut failed) = (0usize, 0usize);
            for entry in entries {
                match entry {
                    Ok(_) => ok += 1,
                    Err(e) => {
                        if failed == 0 {
                            debug_log!("{label}: listing entry FAILED: {e}");
                        }
                        failed += 1;
                    }
                }
            }
            debug_log!("{label}: listing OK ({ok} entries, {failed} unreadable)");
        }
        Err(e) => debug_log!("{label}: listing FAILED: {e}"),
    }
    if check_write {
        check_write_access(dir, label);
    }
}

/// Creates and removes a small file: the GUI renames flagged demos in this folder.
fn check_write_access(dir: &Path, label: &str) {
    let probe = dir.join(".demo_analysis_write_test.tmp");
    match std::fs::write(&probe, b"test") {
        Ok(()) => {
            let removed = std::fs::remove_file(&probe);
            debug_log!(
                "{label}: write OK (test file {})",
                match removed {
                    Ok(()) => "removed".to_string(),
                    Err(e) => format!("could NOT be removed: {e}"),
                }
            );
        }
        Err(e) => debug_log!("{label}: write FAILED: {e} (renaming flagged demos will not work)"),
    }
}

fn check_demo_files(demos_dir: &Path, tf_dir: &Path) {
    let mut dirs = vec![demos_dir.to_path_buf()];
    if tf_dir != demos_dir {
        dirs.push(tf_dir.to_path_buf());
    }
    let mut demos: Vec<(PathBuf, u64, std::io::Result<SystemTime>)> = Vec::new();
    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let before = demos.len();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e.eq_ignore_ascii_case("dem")) {
                match entry.metadata() {
                    Ok(meta) => demos.push((path, meta.len(), meta.modified())),
                    Err(e) => debug_log!("  {}: metadata FAILED: {e}", path.display()),
                }
            }
        }
        debug_log!("{}: {} .dem files", dir.display(), demos.len() - before);
    }
    if demos.is_empty() {
        debug_log!("No demos found. Live scanning needs TF2 to record demos (e.g. ds_enable 2 / automatic demo recording).");
        return;
    }
    demos.sort_by_key(|(_, _, m)| std::cmp::Reverse(m.as_ref().ok().copied()));
    debug_log!("Newest demos:");
    for (path, len, modified) in demos.iter().take(3) {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let header = read_demo_header(path);
        debug_log!("  {name}: {len} bytes, modified {}, {header}", age(clone_time(modified)));
    }
}

fn clone_time(t: &std::io::Result<SystemTime>) -> std::io::Result<SystemTime> {
    match t {
        Ok(t) => Ok(*t),
        Err(e) => Err(std::io::Error::new(e.kind(), e.to_string())),
    }
}

/// Reads the header both in shared mode (as the live scanner) and normally (as the analysis).
fn read_demo_header(path: &Path) -> String {
    let shared = match open_shared(path) {
        Ok(mut f) => {
            let mut header = vec![0u8; DEMO_HEADER_LEN];
            match f.read_exact(&mut header) {
                Ok(()) if &header[0..8] == b"HL2DEMO\0" => {
                    let ticks = i32::from_le_bytes(header[1060..1064].try_into().unwrap());
                    let map_end = header[536..796].iter().position(|&b| b == 0).unwrap_or(260);
                    let map = String::from_utf8_lossy(&header[536..536 + map_end]).into_owned();
                    format!(
                        "shared read OK (map {map}, {})",
                        if ticks > 0 { format!("{ticks} ticks, finished") } else { "still recording or crashed".to_string() }
                    )
                }
                Ok(()) => "shared read OK but NOT a demo (bad magic)".to_string(),
                Err(e) => format!("shared read of header FAILED: {e}"),
            }
        }
        Err(e) => format!("shared open FAILED: {e}"),
    };
    let normal = match std::fs::File::open(path) {
        Ok(_) => "normal open OK".to_string(),
        Err(e) => format!("normal open FAILED: {e}"),
    };
    format!("{shared}, {normal}")
}

fn check_console_log(tf_dir: &Path) {
    let log_path = tf_dir.join("console.log");
    let meta = match std::fs::metadata(&log_path) {
        Ok(m) => m,
        Err(e) => {
            debug_log!(
                "{}: NOT available ({e}). TF2 writes it with the -condebug launch option or `con_logfile console.log` \
                 (sent automatically once RCON connects).",
                log_path.display()
            );
            return;
        }
    };
    debug_log!("{}: {} bytes, modified {}", log_path.display(), meta.len(), age(meta.modified()));
    match open_shared(&log_path) {
        Ok(mut f) => {
            let tail_len = meta.len().min(64 * 1024);
            let _ = f.seek(SeekFrom::Start(meta.len() - tail_len));
            let mut buf = Vec::new();
            match f.read_to_end(&mut buf) {
                Ok(n) => {
                    let text = String::from_utf8_lossy(&buf);
                    let recordings = text.lines().filter(|l| l.contains("Recording to ")).count();
                    let completed = text.lines().filter(|l| l.contains("Completed demo,")).count();
                    let kills = text.lines().filter(|l| l.contains(" killed ")).count();
                    debug_log!(
                        "  shared read OK ({n} bytes of the tail): {recordings} 'Recording to' lines, {completed} 'Completed demo' lines, {kills} kill lines"
                    );
                }
                Err(e) => debug_log!("  shared read FAILED: {e}"),
            }
        }
        Err(e) => debug_log!("  shared open FAILED: {e}"),
    }
}

/// Values of console commands set in a cfg file: `(command, rest of the line)`, comments removed.
fn cfg_commands(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split("//").next().unwrap_or("");
        for statement in line.split(';') {
            let statement = statement.trim();
            let mut parts = statement.splitn(2, char::is_whitespace);
            if let Some(cmd) = parts.next().filter(|c| !c.is_empty()) {
                let value = parts.next().unwrap_or("").trim().trim_matches('"').to_string();
                out.push((cmd.to_ascii_lowercase(), value));
            }
        }
    }
    out
}

fn check_configs(tf_dir: &Path, rcon_password: Option<&str>) {
    let mut cfgs = vec![tf_dir.join("cfg").join("autoexec.cfg")];
    if let Ok(entries) = std::fs::read_dir(tf_dir.join("custom")) {
        for entry in entries.flatten() {
            let cfg = entry.path().join("cfg").join("autoexec.cfg");
            if cfg.is_file() {
                cfgs.push(cfg);
            }
        }
    }
    let gui_password = rcon_password.map(str::trim).filter(|p| !p.is_empty());
    debug_log!(
        "GUI RCON password: {}",
        gui_password.map_or("empty".to_string(), |p| format!("set ({} characters)", p.chars().count()))
    );
    for cfg in cfgs {
        let text = match std::fs::read(&cfg) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(e) => {
                debug_log!("{}: not readable ({e})", cfg.display());
                continue;
            }
        };
        let commands = cfg_commands(&text);
        let mut found = Vec::new();
        for (cmd, value) in &commands {
            match cmd.as_str() {
                "rcon_password" => found.push(format!(
                    "rcon_password set ({} characters, {})",
                    value.chars().count(),
                    match gui_password {
                        Some(p) if p == value.as_str() => "MATCHES the GUI password",
                        Some(_) => "DIFFERS from the GUI password",
                        None => "GUI password empty",
                    }
                )),
                "net_start" => found.push("net_start".to_string()),
                "ip" | "hostport" | "con_logfile" | "ds_enable" | "ds_dir" | "con_timestamp" => {
                    found.push(format!("{cmd} {value}"))
                }
                _ => {}
            }
        }
        debug_log!(
            "{}: {}",
            cfg.display(),
            if found.is_empty() { "no RCON / demo related commands".to_string() } else { found.join(", ") }
        );
    }
}

/// TF2's launch options from Steam (`userdata/<id>/config/localconfig.vdf`, app 440): live
/// scanning over RCON needs `-usercon`; the RCON password, if given here, is masked.
fn log_steam_launch_options() {
    debug_log!("--- Steam launch options (TF2, app 440) ---");
    let Some(steam) = get_steam_install_path() else {
        debug_log!("Steam install path not found");
        return;
    };
    let userdata = steam.join("userdata");
    let entries = match std::fs::read_dir(&userdata) {
        Ok(e) => e,
        Err(e) => {
            debug_log!("{}: not readable ({e})", userdata.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let config = entry.path().join("config").join("localconfig.vdf");
        let Ok(bytes) = std::fs::read(&config) else { continue };
        let text = String::from_utf8_lossy(&bytes);
        let account = entry.file_name().to_string_lossy().into_owned();
        match launch_options_440(&text) {
            Some(options) => {
                let lower = options.to_ascii_lowercase();
                debug_log!(
                    "account {account}: \"{}\" (-usercon {}, -condebug {})",
                    mask_password(&options),
                    if lower.contains("-usercon") { "present" } else { "MISSING" },
                    if lower.contains("-condebug") { "present" } else { "missing" },
                );
            }
            None => debug_log!("account {account}: no launch options for TF2 (-usercon MISSING)"),
        }
    }
}

/// The `LaunchOptions` value inside the `"440"` app block of a localconfig.vdf.
fn launch_options_440(vdf: &str) -> Option<String> {
    let lines: Vec<&str> = vdf.lines().map(str::trim).collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i] == "\"440\"" && lines.get(i + 1) == Some(&"{") {
            let mut depth = 0i32;
            for line in &lines[i + 1..] {
                if *line == "{" {
                    depth += 1;
                } else if *line == "}" {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                } else if depth == 1 {
                    if let Some(rest) = line.strip_prefix("\"LaunchOptions\"") {
                        let rest = rest.trim();
                        let value = rest.strip_prefix('"').unwrap_or(rest);
                        let value = value.strip_suffix('"').unwrap_or(value);
                        return Some(value.replace("\\\"", "\"").replace("\\\\", "\\"));
                    }
                }
            }
        }
        i += 1;
    }
    None
}

/// Replaces the value after `rcon_password` with `***`.
fn mask_password(options: &str) -> String {
    let mut out = Vec::new();
    let mut mask_next = false;
    for word in options.split_whitespace() {
        if mask_next {
            out.push("***".to_string());
            mask_next = false;
        } else {
            mask_next = word.trim_start_matches('+').eq_ignore_ascii_case("rcon_password");
            out.push(word.to_string());
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_tf2_launch_options() {
        let vdf = "\"apps\"\n{\n\"420\"\n{\n\"LaunchOptions\"\t\t\"-x\"\n}\n\"440\"\n{\n\"cloud\"\n{\n\"LaunchOptions\"\t\t\"nested\"\n}\n\"LaunchOptions\"\t\t\"-novid -usercon +rcon_password \\\"pw\\\"\"\n}\n}";
        let opts = launch_options_440(vdf).unwrap();
        assert_eq!(opts, "-novid -usercon +rcon_password \"pw\"");
        assert_eq!(mask_password(&opts), "-novid -usercon +rcon_password ***");
    }

    #[test]
    fn parses_cfg_commands() {
        let cmds = cfg_commands("rcon_password \"abc\" // comment\nnet_start; ip 0.0.0.0\n");
        assert_eq!(cmds[0], ("rcon_password".to_string(), "abc".to_string()));
        assert_eq!(cmds[1].0, "net_start");
        assert_eq!(cmds[2], ("ip".to_string(), "0.0.0.0".to_string()));
    }
}
