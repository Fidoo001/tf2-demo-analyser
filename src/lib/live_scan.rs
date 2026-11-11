use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::base::cheat_analyser_base::CheatAnalyser;
use crate::lib::algorithm::{analyse_bytes_from_tick, is_skipped_in_live_scan, CheatAlgorithm, Detection};
use crate::lib::general_settings::GeneralSettings;

/// Seconds between periodic live scans unless the user sets another interval.
pub const DEFAULT_LIVE_SCAN_INTERVAL_SECONDS: u64 = 30;

/// A scan ends with detections that a spawn or teleport right after it may still retract (60
/// ticks) and launches that still wait for their lookaround. Each live scan therefore takes over
/// from this many ticks before the end of the previous one and redoes them.
pub const LIVE_SCAN_OVERLAP_TICKS: u32 = 64;
/// Ticks of history an algorithm gets before the tick its scan takes over from, unless it says
/// otherwise with `live_warmup_ticks()`: the 60 ticks JankGuard algorithms ignore after a spawn
/// or teleport, plus some margin.
pub const DEFAULT_LIVE_WARMUP_TICKS: u32 = 130;

/// Tick a scan analysed the demo until, for the next scan to continue from. `None` while the
/// scan has not got past the signon data: its frames carry the server tick at connection time
/// (thousands), while the demo ticks start at 0 after it, so the next scan would skip them.
pub fn analysed_until(analyser: &CheatAnalyser) -> Option<u32> {
    analyser.past_signon().then(|| u32::from(analyser.tick))
}

/// Results of the previous live scans of the demo being recorded. Each scan only runs the
/// algorithms on the part recorded since the last one; their detections are added to these.
#[derive(Default)]
pub struct LiveScanProgress {
    /// Last tick analysed by the previous scan.
    pub analysed_until: Option<u32>,
    /// Demo size at the previous scan, a smaller file means the recording was restarted.
    pub analysed_bytes: usize,
    /// Enabled algorithms at the previous scan; newly enabled ones need the whole demo.
    pub algorithm_names: Vec<String>,
    /// Detections of all scans so far, before the sliding window filter.
    pub raw_detections: Vec<Detection>,
}

impl LiveScanProgress {
    /// Tick the next scan takes over from: the detections from here on are the new scan's. Every
    /// algorithm starts its own `live_warmup_ticks()` earlier, so it has the history it needs.
    pub fn next_start_tick(&self, algorithm_names: &[String]) -> u32 {
        match self.analysed_until {
            Some(until) if algorithm_names == self.algorithm_names => {
                until.saturating_sub(LIVE_SCAN_OVERLAP_TICKS)
            }
            _ => 0,
        }
    }

    /// Adds the detections of a scan that took over from `from_tick` (see `next_start_tick`) and
    /// returns all detections so far.
    ///
    /// A scan reports everything it has when it ends, but what comes after it can still retract
    /// the last detections (a spawn or teleport) or change them (a launch waiting for its
    /// lookaround). And an algorithm that ran on a part of the demo only matches a scan of the
    /// whole demo once it is warmed up. The new scan therefore replaces the previous results from
    /// the provisional ticks at the end of the previous scan on, and its earlier detections are
    /// ignored (the previous scan found those with all of their history).
    /// Algorithms that need the whole demo ran on all of it so far, so the new scan replaces all
    /// of their previous results.
    pub fn merge(
        &mut self,
        detections: Vec<Detection>,
        from_tick: u32,
        whole_demo_algorithms: &HashSet<String>,
    ) -> &[Detection] {
        let partial = |detection: &Detection| !whole_demo_algorithms.contains(&detection.algorithm);

        self.raw_detections
            .retain(|d| partial(d) && d.tick < from_tick);
        self.raw_detections
            .extend(detections.into_iter().filter(|d| !partial(d) || d.tick >= from_tick));
        self.sort_detections();
        &self.raw_detections
    }

    /// Continues from a manual analysis of the whole demo: its detections replace everything found
    /// so far (the next scans only add to them) and the next scan starts where it ended.
    pub fn adopt_rescan(&mut self, rescan: RescanResult) {
        self.analysed_until = Some(rescan.analysed_until);
        self.analysed_bytes = rescan.analysed_bytes;
        self.algorithm_names = rescan.algorithm_names;
        self.raw_detections = rescan.detections;
        self.sort_detections();
    }

    fn sort_detections(&mut self) {
        self.raw_detections.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.player.cmp(&b.player))
                .then_with(|| a.algorithm.cmp(&b.algorithm))
        });
    }
}

/// What a manual analysis of the whole demo hands over to the live scanner.
pub struct RescanResult {
    pub demo_path: PathBuf,
    /// Detections before the sliding window filter.
    pub detections: Vec<Detection>,
    pub analysed_until: u32,
    pub analysed_bytes: usize,
    /// Sorted names of the algorithms that ran.
    pub algorithm_names: Vec<String>,
}

/// Makes the live scanner and a manual rescan of the demo being recorded take turns: a rescan
/// holds the live scanner back until it has finished and then hands its results over.
#[derive(Default)]
pub struct LiveScanGate {
    rescan_active: AtomicBool,
    live_scan_running: AtomicBool,
    /// Number of rescans that handed over their results.
    epoch: AtomicU64,
    handover: Mutex<Option<RescanResult>>,
}

impl LiveScanGate {
    /// Stops new live scans from starting until the returned guard is dropped.
    pub fn begin_rescan(self: &Arc<Self>) -> RescanGuard {
        self.rescan_active.store(true, Ordering::SeqCst);
        RescanGuard { gate: self.clone() }
    }

    /// Live scanner: `None` while a rescan is running (try again later), otherwise the permit to
    /// scan until it is dropped, which a rescan waits for.
    pub fn try_begin_live_scan(&self) -> Option<LiveScanPermit<'_>> {
        self.live_scan_running.store(true, Ordering::SeqCst);
        if self.rescan_active.load(Ordering::SeqCst) {
            self.live_scan_running.store(false, Ordering::SeqCst);
            return None;
        }
        Some(LiveScanPermit { gate: self })
    }

    /// Number of finished rescans. A live scan that started before a rescan finished has an older
    /// number than the rescan and its results are outdated by it.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Live scanner: the results of the last rescan, if it has not picked them up yet.
    pub fn take_rescan(&self) -> Option<RescanResult> {
        self.handover().take()
    }

    fn handover(&self) -> MutexGuard<'_, Option<RescanResult>> {
        self.handover.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct LiveScanPermit<'a> {
    gate: &'a LiveScanGate,
}

impl Drop for LiveScanPermit<'_> {
    fn drop(&mut self) {
        self.gate.live_scan_running.store(false, Ordering::SeqCst);
    }
}

/// Keeps the live scanner from scanning while it exists, even if the rescan fails or panics.
pub struct RescanGuard {
    gate: Arc<LiveScanGate>,
}

impl RescanGuard {
    /// Waits for the live scan that is still going on, if any. Fails when the analysis is cancelled.
    pub fn wait_for_live_scan(&self) -> anyhow::Result<()> {
        while self.gate.live_scan_running.load(Ordering::SeqCst) {
            if crate::CANCEL_ANALYSIS.load(Ordering::Relaxed) {
                anyhow::bail!("Analysis cancelled");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(())
    }

    /// Hands the results to the live scanner and lets it scan again.
    fn finish(self, result: RescanResult) {
        *self.gate.handover() = Some(result);
        self.gate.epoch.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for RescanGuard {
    fn drop(&mut self) {
        // A cancellation belongs to the rescan, the live scans must not abort because of it
        crate::CANCEL_ANALYSIS.store(false, Ordering::Relaxed);
        self.gate.rescan_active.store(false, Ordering::SeqCst);
    }
}

/// A manual analysis of the whole demo that is being recorded and scanned live.
pub struct LiveRescan {
    guard: RescanGuard,
    /// Algorithm that only runs for the live scanner's notifications and is not part of the results.
    hidden_algorithm: Option<String>,
}

impl LiveRescan {
    pub fn new(guard: RescanGuard, hidden_algorithm: Option<String>) -> Self {
        Self { guard, hidden_algorithm }
    }

    /// Analyses the demo in its current state once the running live scan has finished. The live
    /// scanner then continues from this analysis: with its detections, from the tick it ended at.
    pub fn run(
        self,
        file: &Path,
        algorithms: Vec<Box<dyn CheatAlgorithm<'static> + Send>>,
        settings: &GeneralSettings,
    ) -> anyhow::Result<CheatAnalyser<'static>> {
        self.guard.wait_for_live_scan()?;
        let bytes = std::fs::read(file)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", file.display()))?;
        // The live scanner continues without the algorithms it skips
        let mut algorithm_names: Vec<String> = algorithms
            .iter()
            .filter(|a| !a.skipped_in_live_scan())
            .map(|a| a.algorithm_name().to_string())
            .collect();
        algorithm_names.sort();

        let mut analyser = analyse_bytes_from_tick(&bytes, algorithms, 0)?;
        let result = RescanResult {
            demo_path: file.to_path_buf(),
            detections: analyser
                .detections
                .iter()
                .filter(|d| !is_skipped_in_live_scan(&d.algorithm))
                .cloned()
                .collect(),
            // Before the end of the signon data, the next scan has to start from the beginning
            analysed_until: analysed_until(&analyser).unwrap_or(0),
            analysed_bytes: bytes.len(),
            algorithm_names,
        };
        analyser.apply_sliding_window(settings);
        if let Some(hidden) = &self.hidden_algorithm {
            analyser.detections.retain(|d| &d.algorithm != hidden);
        }
        self.guard.finish(result);
        Ok(analyser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(tick: u32, player: u64, algorithm: &str) -> Detection {
        Detection { tick, algorithm: algorithm.to_string(), player, data: serde_json::Value::Null }
    }

    fn ticks(detections: &[Detection], algorithm: &str) -> Vec<u32> {
        detections.iter().filter(|d| d.algorithm == algorithm).map(|d| d.tick).collect()
    }

    #[test]
    fn test_next_start_tick() {
        let names = vec!["a".to_string()];
        let mut progress = LiveScanProgress::default();
        assert_eq!(progress.next_start_tick(&names), 0);

        progress.analysed_until = Some(2000);
        progress.algorithm_names = names.clone();
        assert_eq!(progress.next_start_tick(&names), 2000 - LIVE_SCAN_OVERLAP_TICKS);
        // A newly enabled algorithm needs the whole demo
        assert_eq!(progress.next_start_tick(&["a".to_string(), "b".to_string()]), 0);
        // A short demo is scanned from the start again
        progress.analysed_until = Some(LIVE_SCAN_OVERLAP_TICKS - 10);
        assert_eq!(progress.next_start_tick(&names), 0);
    }

    #[test]
    fn test_no_analysed_tick_before_the_end_of_the_signon_data() {
        let mut analyser = CheatAnalyser::default();
        analyser.tick = 6836.into();
        assert_eq!(analysed_until(&analyser), None);
    }

    // The skipped algorithms are all experimental
    #[cfg(feature = "experimental")]
    #[test]
    fn test_skipped_algorithms() {
        for name in [
            "fidoo/demoswap_pvs",
            "fidoo/recorder_crithack",
            "recorder_aim_assist",
            "fidoo/tampered",
            "fidoo/lboxaimbot",
            "j_backtrack",
            "amalgam_triggerbot",
        ] {
            assert!(is_skipped_in_live_scan(name), "{name}");
        }
        for name in ["fidoo/psilent5", "autostrafe", "fidoo/votekick_votes", "nocrex/aimsnap"] {
            assert!(!is_skipped_in_live_scan(name), "{name}");
        }
    }

    #[test]
    fn test_first_scan_keeps_everything() {
        let mut progress = LiveScanProgress::default();
        let all = progress.merge(
            vec![detection(500, 1, "a"), detection(100, 2, "a")],
            0,
            &HashSet::new(),
        );
        assert_eq!(ticks(all, "a"), vec![100, 500]);
    }

    #[test]
    fn test_next_scan_replaces_the_provisional_end_of_the_previous_one() {
        let none = HashSet::new();
        let mut progress = LiveScanProgress::default();
        // The previous scan ended at tick 2000: 1950 may still be retracted
        progress.merge(vec![detection(500, 1, "a"), detection(1500, 1, "a"), detection(1950, 1, "a")], 0, &none);

        let from_tick = 2000 - LIVE_SCAN_OVERLAP_TICKS;
        // The new scan retracted 1950, and a detection before the tick it took over from (1900,
        // found by its algorithms while warming up) is ignored
        let all = progress.merge(
            vec![detection(1900, 1, "a"), detection(1940, 1, "a"), detection(2100, 1, "a")],
            from_tick,
            &none,
        );
        assert_eq!(ticks(all, "a"), vec![500, 1500, 1940, 2100]);

        // Nothing new: the end of the scan is redone, the rest stays
        let all = progress.merge(vec![detection(1940, 1, "a")], from_tick, &none);
        assert_eq!(ticks(all, "a"), vec![500, 1500, 1940]);
    }

    #[test]
    fn test_whole_demo_algorithms_are_replaced_as_a_whole() {
        let whole: HashSet<String> = ["w".to_string()].into();
        let mut progress = LiveScanProgress::default();
        progress.merge(
            vec![detection(1000, 1, "w"), detection(1950, 1, "w"), detection(1950, 1, "a")],
            0,
            &whole,
        );

        let from_tick = 2000 - LIVE_SCAN_OVERLAP_TICKS;
        // It ran on the whole demo: what it found earlier is in the result, what it no longer
        // finds is gone, whatever the tick
        let all = progress.merge(
            vec![detection(300, 1, "w"), detection(1950, 1, "w"), detection(2100, 1, "w")],
            from_tick,
            &whole,
        );
        assert_eq!(ticks(all, "w"), vec![300, 1950, 2100]);
        // The partial algorithm found nothing in the new scan, its provisional detection is gone
        assert!(ticks(all, "a").is_empty());
    }

    fn rescan_result(detections: Vec<Detection>, analysed_until: u32) -> RescanResult {
        RescanResult {
            demo_path: PathBuf::from("live.dem"),
            detections,
            analysed_until,
            analysed_bytes: 5000,
            algorithm_names: vec!["a".to_string()],
        }
    }

    #[test]
    fn test_rescan_replaces_everything_and_scans_continue_from_its_end() {
        let none = HashSet::new();
        let whole: HashSet<String> = ["w".to_string()].into();
        let mut progress = LiveScanProgress::default();
        progress.merge(vec![detection(100, 1, "a"), detection(300, 2, "a"), detection(400, 3, "w")], 0, &whole);
        progress.analysed_until = Some(1000);
        progress.algorithm_names = vec!["a".to_string()];

        // The rescan reached tick 2000, found something else and lost the detections of player 2
        progress.adopt_rescan(rescan_result(
            vec![detection(100, 1, "a"), detection(1500, 4, "a"), detection(1950, 1, "a")],
            2000,
        ));
        assert_eq!(progress.analysed_until, Some(2000));
        assert_eq!(progress.analysed_bytes, 5000);
        assert_eq!(progress.next_start_tick(&["a".to_string()]), 2000 - LIVE_SCAN_OVERLAP_TICKS);
        let kept: Vec<_> = progress.raw_detections.iter().map(|d| (d.tick, d.player)).collect();
        assert_eq!(kept, vec![(100, 1), (1500, 4), (1950, 1)]);

        // The next scan only covers the new part; the rescan's detections stay
        let from_tick = progress.next_start_tick(&["a".to_string()]);
        let all = progress.merge(vec![detection(1940, 1, "a"), detection(2100, 5, "a")], from_tick, &none);
        assert_eq!(all.iter().map(|d| (d.tick, d.player)).collect::<Vec<_>>(),
            vec![(100, 1), (1500, 4), (1940, 1), (2100, 5)]);
    }

    #[test]
    fn test_live_scans_wait_for_a_rescan() {
        let gate = Arc::new(LiveScanGate::default());
        assert!(gate.try_begin_live_scan().is_some());

        let guard = gate.begin_rescan();
        assert!(gate.try_begin_live_scan().is_none());
        // The refused attempt must not look like a running scan to the rescan
        guard.wait_for_live_scan().unwrap();

        drop(guard);
        assert!(gate.try_begin_live_scan().is_some());
    }

    #[test]
    fn test_rescan_waits_for_the_running_live_scan() {
        let gate = Arc::new(LiveScanGate::default());
        let permit = gate.try_begin_live_scan().unwrap();
        let guard = gate.begin_rescan();
        let (done_send, done_recv) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            guard.wait_for_live_scan().unwrap();
            done_send.send(()).unwrap();
        });
        assert!(done_recv.recv_timeout(Duration::from_millis(100)).is_err());
        drop(permit);
        done_recv.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn test_finished_rescan_is_handed_over_once() {
        let gate = Arc::new(LiveScanGate::default());
        assert_eq!(gate.epoch(), 0);
        assert!(gate.take_rescan().is_none());

        // A failed or cancelled rescan hands nothing over
        drop(gate.begin_rescan());
        assert!(gate.take_rescan().is_none());
        assert_eq!(gate.epoch(), 0);

        let guard = gate.begin_rescan();
        guard.finish(rescan_result(vec![detection(5, 1, "a")], 10));
        assert_eq!(gate.epoch(), 1);
        assert!(gate.try_begin_live_scan().is_some());
        assert_eq!(gate.take_rescan().unwrap().analysed_until, 10);
        assert!(gate.take_rescan().is_none());
    }

    #[test]
    fn test_full_rescan_replaces_partial_algorithms() {
        let none = HashSet::new();
        let mut progress = LiveScanProgress::default();
        progress.merge(vec![detection(100, 1, "a"), detection(900, 1, "a")], 0, &none);
        let all = progress.merge(vec![detection(100, 1, "a")], 0, &none);
        assert_eq!(ticks(all, "a"), vec![100]);
    }
}
