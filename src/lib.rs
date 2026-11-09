use std::sync::atomic::AtomicU32;
pub mod base {
    pub mod cheat_analyser_base;
    pub mod demo_handler_base;
}

pub mod header_view;
pub mod playerlist_view;
pub mod chat_view;
#[cfg(feature = "player")]
pub mod player;
#[cfg(feature = "rcon")]
pub mod rcon;
#[cfg(feature = "tf2bd")]
pub mod tf2bd;
#[cfg(all(feature = "tf2bd", feature = "gui"))]
pub mod tf2bd_view;


pub mod algorithms {
    // Stable and experimental modules are auto-discovered by build.rs at compile time.
    // To add, remove, or move an algorithm between folders, simply place its .rs file
    // in src/algorithms/, src/algorithms/fidoo/, src/algorithms/nocrex/,
    // src/algorithms/experimental/, or src/algorithms/fidoo_experimental/
    // — no manual module declarations or code changes needed here.
    include!(concat!(env!("OUT_DIR"), "/stable_mods.rs"));
    include!(concat!(env!("OUT_DIR"), "/experimental_mods.rs"));
}

pub mod util {
    pub mod bsp;
    pub mod datatable_validator;
    pub mod helpers;
    pub mod schema_equip_regions;
    #[cfg(any(feature = "experimental", feature = "player"))]
    pub mod voice;
    pub mod nocrex {
        pub mod jankguard;
    }
}

pub mod lib {
    pub mod algorithm;
    pub mod algorithm_costs;
    pub mod excluded_algorithms_stable;
    #[cfg(has_experimental_exclusions)]
    pub mod excluded_algorithms_experimental;
    pub mod general_settings;
    pub mod live_scan;
    pub mod parameters;
}

pub static SILENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub const MAX_WORKERS: usize = 32;
pub static WORKER_TICKS: [AtomicU32; MAX_WORKERS] = [
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
    AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0),
];
pub static WORKER_COUNT: AtomicU32 = AtomicU32::new(1);
pub static PROGRESS_CURRENT: AtomicU32 = AtomicU32::new(0);
pub static PROGRESS_TOTAL: AtomicU32 = AtomicU32::new(0);

/// Set by the GUI to make a running analysis stop at the next packet; the analysis then returns an error.
pub static CANCEL_ANALYSIS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static DETECTION_LISTENER: std::sync::Mutex<Option<std::sync::mpsc::Sender<u64>>> =
    std::sync::Mutex::new(None);
static DETECTION_LISTENER_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets (or clears) a channel that receives the SteamID of every flagged player as soon as an
/// algorithm produces a detection, before the sliding window filter. Used by the GUI to start
/// per-player lookups while the analysis is still running.
pub fn set_detection_listener(listener: Option<std::sync::mpsc::Sender<u64>>) {
    let active = listener.is_some();
    *DETECTION_LISTENER.lock().unwrap_or_else(|e| e.into_inner()) = listener;
    DETECTION_LISTENER_ACTIVE.store(active, std::sync::atomic::Ordering::Relaxed);
}

/// Forwards the players of new detections to the detection listener, if one is set.
pub fn notify_detections(detections: &[lib::algorithm::Detection]) {
    if !DETECTION_LISTENER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Ok(guard) = DETECTION_LISTENER.lock() {
        if let Some(listener) = guard.as_ref() {
            for d in detections.iter().filter(|d| d.player != 0) {
                let _ = listener.send(d.player);
            }
        }
    }
}

pub fn reset_progress(num_workers: usize) {
    let workers = num_workers.clamp(1, MAX_WORKERS) as u32;
    WORKER_COUNT.store(workers, std::sync::atomic::Ordering::Relaxed);
    PROGRESS_CURRENT.store(0, std::sync::atomic::Ordering::Relaxed);
    PROGRESS_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    for i in 0..MAX_WORKERS {
        WORKER_TICKS[i].store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn get_average_progress() -> (u32, u32) {
    let count = WORKER_COUNT.load(std::sync::atomic::Ordering::Relaxed).max(1) as usize;
    let total = PROGRESS_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
    let mut sum: u64 = 0;
    for i in 0..count.min(MAX_WORKERS) {
        sum += WORKER_TICKS[i].load(std::sync::atomic::Ordering::Relaxed) as u64;
    }
    let avg = (sum / count as u64) as u32;
    (avg, total)
}

#[macro_export]
macro_rules! dev_print {
    ($($arg:tt)*) => {
        if !crate::SILENT.load(std::sync::atomic::Ordering::Relaxed) {
            println!($($arg)*);
        }
    }
}
