//! Party chat summaries of the votekicks found in the demo being recorded.
//!
//! A votekick can have up to 23 voters, so announcing every cast would flood the chat. Instead
//! each votekick is announced once, after voting has finished, with at most three lines:
//!
//! ```text
//! [Analyser] Votekick: Caller -> Target (cheating) | 14 yes, 3 no
//! [Analyser] Yes (14): a, b, c, d, +10 more
//! [Analyser] No (3): e, f, g
//! ```
//!
//! The votes come from the `fidoo/votekick_votes` detections.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::lib::algorithm::Detection;

pub const VOTE_ALGORITHM: &str = "fidoo/votekick_votes";

const TICK_RATE: f32 = 66.67;
/// A votekick lasts 15 seconds; it is announced once this long has passed since it was called.
const VOTE_SETTLE_SECONDS: f32 = 20.0;
/// Votes called shortly before the scanner started watching the demo are not announced either.
const WATCH_MARGIN_SECONDS: f32 = 5.0;
const MAX_LINE_CHARS: usize = 120;
const MAX_NAME_CHARS: usize = 20;
/// Room kept free at the end of a voter list for its `, +NN more` tail.
const MORE_TAIL_CHARS: usize = 10;

#[derive(Default)]
struct Vote {
    called: bool,
    caller: String,
    target: String,
    reason: String,
    yes: Vec<String>,
    no: Vec<String>,
    voters: HashSet<u64>,
}

impl Vote {
    fn messages(&self) -> Vec<String> {
        let header = format!(
            "[Analyser] Votekick: {} -> {} ({}) | {} yes, {} no",
            shorten(&self.caller),
            shorten(&self.target),
            shorten(&self.reason),
            self.yes.len(),
            self.no.len()
        );
        let mut lines = vec![truncate_line(header)];
        lines.extend(voter_line("Yes", &self.yes));
        lines.extend(voter_line("No", &self.no));
        lines
    }
}

fn shorten(text: &str) -> String {
    if text.chars().count() > MAX_NAME_CHARS {
        let mut short: String = text.chars().take(MAX_NAME_CHARS - 1).collect();
        short.push('…');
        short
    } else {
        text.to_string()
    }
}

fn truncate_line(line: String) -> String {
    if line.chars().count() > MAX_LINE_CHARS {
        line.chars().take(MAX_LINE_CHARS).collect()
    } else {
        line
    }
}

/// One line listing the voters of an option, cut off with `+N more` when it would get too long.
fn voter_line(label: &str, voters: &[String]) -> Option<String> {
    if voters.is_empty() {
        return None;
    }
    let mut line = format!("[Analyser] {label} ({}): ", voters.len());
    let mut shown = 0;
    for (i, voter) in voters.iter().enumerate() {
        let name = shorten(voter);
        let sep = if i == 0 { "" } else { ", " };
        let has_more = i + 1 < voters.len();
        let reserve = if has_more { MORE_TAIL_CHARS } else { 0 };
        if i > 0 && line.chars().count() + sep.len() + name.chars().count() + reserve > MAX_LINE_CHARS {
            break;
        }
        line.push_str(sep);
        line.push_str(&name);
        shown += 1;
    }
    if shown < voters.len() {
        line.push_str(&format!(", +{} more", voters.len() - shown));
    }
    Some(truncate_line(line))
}

/// Groups the vote detections by votekick, in the order they were called. A votekick is
/// identified by its vote id and the tick it was called on, since ids restart on a new server.
fn collect_votes(detections: &[Detection]) -> Vec<((u32, u32), Vote)> {
    let mut votes: Vec<((u32, u32), Vote)> = Vec::new();
    for det in detections.iter().filter(|d| d.algorithm == VOTE_ALGORITHM) {
        let data = &det.data;
        let vote_id = data["vote_id"].as_u64().unwrap_or(0) as u32;
        let (key, is_call) = match data["action"].as_str() {
            Some("call_vote") => ((vote_id, det.tick), true),
            Some("cast_vote") => ((vote_id, data["vote_called_tick"].as_u64().unwrap_or(0) as u32), false),
            _ => continue,
        };
        let index = match votes.iter().position(|(k, _)| *k == key) {
            Some(index) => index,
            None => {
                votes.push((key, Vote::default()));
                votes.len() - 1
            }
        };
        let vote = &mut votes[index].1;
        let name_of = |field: &str, player: u64| match data[field].as_str() {
            Some(name) if !name.is_empty() => name.to_string(),
            _ if player != 0 => player.to_string(),
            _ => "?".to_string(),
        };
        if is_call {
            vote.called = true;
            vote.caller = name_of("caller", det.player);
            vote.target = data["target"].as_str().unwrap_or("?").to_string();
            vote.reason = data["reason"].as_str().unwrap_or("").to_string();
        } else if vote.voters.insert(det.player) {
            let voter = name_of("voter", det.player);
            match data["decision"].as_str() {
                Some("yes") => vote.yes.push(voter),
                Some("no") => vote.no.push(voter),
                _ => {}
            }
        }
    }
    votes.sort_by_key(|(key, _)| key.1);
    votes
}

/// Decides which votekicks of the demo being recorded are announced, and when.
pub struct VoteNotifier {
    watching_since: Instant,
    /// Votekicks called before this tick were already in the demo when the scanner started
    /// watching it. Set by the first scan.
    watch_from_tick: Option<u32>,
    announced: HashSet<(u32, u32)>,
    /// Calls (player 0) and casts already written to the console: (vote id, call tick, player).
    logged: HashSet<(u32, u32, u64)>,
}

impl VoteNotifier {
    pub fn new() -> Self {
        Self {
            watching_since: Instant::now(),
            watch_from_tick: None,
            announced: HashSet::new(),
            logged: HashSet::new(),
        }
    }

    /// Returns one console line per vote call and per individual vote not logged yet. These are
    /// never sent to the party chat. Call after `collect`, which sets where watching started.
    pub fn console_lines(&mut self, detections: &[Detection]) -> Vec<String> {
        let from = self.watch_from_tick.unwrap_or(0);
        let mut lines = Vec::new();
        for det in detections.iter().filter(|d| d.algorithm == VOTE_ALGORITHM) {
            let data = &det.data;
            let vote_id = data["vote_id"].as_u64().unwrap_or(0) as u32;
            let text = |field: &str| data[field].as_str().filter(|s| !s.is_empty()).unwrap_or("?").to_string();
            match data["action"].as_str() {
                Some("call_vote") if det.tick >= from => {
                    if self.logged.insert((vote_id, det.tick, 0)) {
                        lines.push(format!(
                            "Vote #{vote_id} (tick {}): {} called a votekick on {} ({})",
                            det.tick,
                            text("caller"),
                            text("target"),
                            text("reason")
                        ));
                    }
                }
                Some("cast_vote") => {
                    let called = data["vote_called_tick"].as_u64().unwrap_or(0) as u32;
                    if called >= from && self.logged.insert((vote_id, called, det.player)) {
                        let ticks = data["interval_ticks"].as_u64().unwrap_or(0);
                        lines.push(format!(
                            "Vote #{vote_id} (tick {}): {} voted {} on {} (+{} ticks, {:.1}s after the call)",
                            det.tick,
                            text("voter"),
                            text("decision"),
                            text("target"),
                            ticks,
                            ticks as f32 / TICK_RATE
                        ));
                    }
                }
                _ => {}
            }
        }
        lines
    }

    /// Returns the chat lines of the votekicks that finished since the last call. `tick` is the
    /// last tick of the demo scanned so far.
    pub fn collect(&mut self, detections: &[Detection], tick: u32) -> Vec<String> {
        let watching_for = self.watching_since.elapsed();
        let from = *self
            .watch_from_tick
            .get_or_insert_with(|| watch_start_tick(tick, watching_for));
        let settle = (VOTE_SETTLE_SECONDS * TICK_RATE) as u32;
        self.take(detections, from, |called| tick >= called.saturating_add(settle))
    }

    /// Returns the chat lines of all votekicks not announced yet, e.g. when the recording ended
    /// while one was still going on.
    pub fn flush(&mut self, detections: &[Detection]) -> Vec<String> {
        let from = self.watch_from_tick.unwrap_or(0);
        self.take(detections, from, |_| true)
    }

    fn take(&mut self, detections: &[Detection], from: u32, ready: impl Fn(u32) -> bool) -> Vec<String> {
        let mut lines = Vec::new();
        for (key, vote) in collect_votes(detections) {
            if !vote.called || key.1 < from || self.announced.contains(&key) || !ready(key.1) {
                continue;
            }
            self.announced.insert(key);
            lines.extend(vote.messages());
        }
        lines
    }
}

fn watch_start_tick(tick: u32, watching_for: Duration) -> u32 {
    let watched_ticks = ((watching_for.as_secs_f32() + WATCH_MARGIN_SECONDS) * TICK_RATE) as u32;
    tick.saturating_sub(watched_ticks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(tick: u32, vote_id: u32, caller: &str, target: &str) -> Detection {
        Detection {
            tick,
            algorithm: VOTE_ALGORITHM.to_string(),
            player: 1,
            data: json!({"action": "call_vote", "vote_id": vote_id, "caller": caller, "target": target, "reason": "cheating"}),
        }
    }

    fn cast(tick: u32, vote_id: u32, called: u32, player: u64, voter: &str, decision: &str) -> Detection {
        Detection {
            tick,
            algorithm: VOTE_ALGORITHM.to_string(),
            player,
            data: json!({"action": "cast_vote", "vote_id": vote_id, "vote_called_tick": called, "voter": voter, "decision": decision}),
        }
    }

    fn notifier_watching_from(tick: u32) -> VoteNotifier {
        let mut notifier = VoteNotifier::new();
        notifier.watch_from_tick = Some(tick);
        notifier
    }

    #[test]
    fn summarises_a_finished_vote_in_three_lines() {
        let dets = vec![
            call(1000, 7, "Jimtex", "freeway_cola"),
            cast(1000, 7, 1000, 1, "Jimtex", "yes"),
            cast(1000, 7, 1000, 2, "freeway_cola", "no"),
            cast(1090, 7, 1000, 3, "marley", "yes"),
        ];
        let mut notifier = notifier_watching_from(0);
        // Still within the voting time: nothing yet
        assert!(notifier.collect(&dets, 1500).is_empty());

        let lines = notifier.collect(&dets, 1000 + 2000);
        assert_eq!(
            lines,
            vec![
                "[Analyser] Votekick: Jimtex -> freeway_cola (cheating) | 2 yes, 1 no",
                "[Analyser] Yes (2): Jimtex, marley",
                "[Analyser] No (1): freeway_cola",
            ]
        );
        // Announced only once
        assert!(notifier.collect(&dets, 1000 + 3000).is_empty());
    }

    #[test]
    fn long_voter_lists_are_cut_off() {
        let mut dets = vec![call(1000, 1, "Caller", "Target")];
        for i in 0..23u64 {
            dets.push(cast(1100 + i as u32, 1, 1000, 100 + i, &format!("PlayerName{i:02}"), "yes"));
        }
        let lines = notifier_watching_from(0).collect(&dets, 10_000);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].ends_with("| 23 yes, 0 no"));
        assert!(lines[1].starts_with("[Analyser] Yes (23): PlayerName00, PlayerName01"));
        assert!(lines[1].contains(", +"));
        assert!(lines[1].ends_with(" more"));
        assert!(lines.iter().all(|l| l.chars().count() <= MAX_LINE_CHARS));
    }

    #[test]
    fn votes_from_before_watching_and_repeated_ids_are_handled() {
        let dets = vec![
            call(500, 3, "Old", "Vote"),
            call(9000, 3, "New", "Server"), // same id on a later server
            cast(9100, 3, 9000, 5, "Voter", "no"),
        ];
        let mut notifier = notifier_watching_from(5000);
        let lines = notifier.collect(&dets, 20_000);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("New -> Server"));
        assert_eq!(lines[1], "[Analyser] No (1): Voter");
    }

    #[test]
    fn console_lines_log_every_vote_once() {
        let mut dets = vec![
            call(1000, 7, "Jimtex", "freeway_cola"),
            cast(1000, 7, 1000, 1, "Jimtex", "yes"),
            cast(1200, 7, 1000, 3, "marley", "no"),
            call(100, 6, "Old", "Vote"),
        ];
        dets[2].data["interval_ticks"] = json!(200);
        dets[1].data["target"] = json!("freeway_cola");
        dets[2].data["target"] = json!("freeway_cola");
        let mut notifier = notifier_watching_from(500);
        let lines = notifier.console_lines(&dets);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "Vote #7 (tick 1000): Jimtex called a votekick on freeway_cola (cheating)");
        assert_eq!(lines[1], "Vote #7 (tick 1000): Jimtex voted yes on freeway_cola (+0 ticks, 0.0s after the call)");
        assert_eq!(lines[2], "Vote #7 (tick 1200): marley voted no on freeway_cola (+200 ticks, 3.0s after the call)");
        assert!(notifier.console_lines(&dets).is_empty());
    }

    #[test]
    fn flush_announces_unfinished_votes() {
        let dets = vec![call(1000, 2, "A", "B")];
        let mut notifier = notifier_watching_from(0);
        assert!(notifier.collect(&dets, 1100).is_empty());
        assert_eq!(notifier.flush(&dets).len(), 1);
        assert!(notifier.flush(&dets).is_empty());
    }

    #[test]
    fn watch_start_accounts_for_elapsed_time() {
        // 10s watched + 5s margin at 66.67 ticks/s
        assert_eq!(watch_start_tick(10_000, Duration::from_secs(10)), 10_000 - 1000);
        assert_eq!(watch_start_tick(100, Duration::from_secs(10)), 0);
    }
}
