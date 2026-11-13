//! Opt-in debug log, turned on by launching the GUI with `--debug`.
//!
//! Without the flag nothing is written anywhere: `debug_log!` only checks an atomic flag. With it,
//! every launch truncates the log file and writes timestamped lines from all threads, flushed
//! right away so the log survives a crash or a killed process.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub const DEBUG_FLAG: &str = "--debug";
pub const LOG_FILE_NAME: &str = "demo_analysis_debug.log";

static ENABLED: AtomicBool = AtomicBool::new(false);
static FILE: Mutex<Option<File>> = Mutex::new(None);
static PATH: OnceLock<PathBuf> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();

/// Whether debug logging is on (cheap; checked by `debug_log!` before formatting anything).
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Where the log is being written, once `init` has succeeded.
pub fn path() -> Option<&'static Path> {
    PATH.get().map(|p| p.as_path())
}

/// Turns debug logging on. The log goes next to the executable or, when that folder cannot be
/// written (e.g. Program Files), into the temp folder. An existing log is overwritten.
pub fn init() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join(LOG_FILE_NAME));
    }
    candidates.push(std::env::temp_dir().join(LOG_FILE_NAME));

    let mut errors = Vec::new();
    for candidate in candidates {
        match File::create(&candidate) {
            Ok(file) => {
                *FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some(file);
                let _ = PATH.set(candidate.clone());
                START.get_or_init(Instant::now);
                ENABLED.store(true, Ordering::Relaxed);
                if !errors.is_empty() {
                    write(format_args!("Could not create the log at: {}", errors.join("; ")));
                }
                return Ok(candidate);
            }
            Err(e) => errors.push(format!("{} ({e})", candidate.display())),
        }
    }
    Err(errors.join("; "))
}

/// Writes one line; use `debug_log!` instead, which skips the formatting when logging is off.
pub fn write(args: std::fmt::Arguments) {
    if !enabled() {
        return;
    }
    let elapsed = START.get().map_or(0.0, |s| s.elapsed().as_secs_f64());
    let now = chrono::Local::now().format("%H:%M:%S%.3f");
    let thread = std::thread::current();
    let thread_name = thread.name().unwrap_or("unnamed");
    let mut guard = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(file) = guard.as_mut() {
        let _ = writeln!(file, "[{now} +{elapsed:9.3}s] [{thread_name}] {args}");
        let _ = file.flush();
    }
}

/// Writes a line to the debug log when the program was started with `--debug`.
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if $crate::util::debug_log::enabled() {
            $crate::util::debug_log::write(format_args!($($arg)*));
        }
    };
}
