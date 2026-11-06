use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gameevent_gen::GameEvent;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::analyser::Class;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{angle_delta, handle_to_entid};
use crate::util::nocrex::jankguard::JankGuard;

const ATTACK_HISTORY_MAX_TICKS: u32 = 40;
const SNAPSHOT_HISTORY_MAX_TICKS: u32 = 66;
const READY_LOG_MAX_TICKS: u32 = 200;
/// Every stab (and silent flick) is judged on its own; it is reported when its score reaches this.
/// Isolated detections are removed later by the global sliding window.
const DETECTION_SCORE_THRESHOLD: u32 = 100;
const STRONG_EVIDENCE_SCORE: u32 = 100;
const WEAK_EVIDENCE_SCORE: u32 = 50;
/// Knife hits are evaluated this many ticks after the hurt event so the spy's
/// angles around the attack tick are available.
const STAB_EVAL_DELAY_TICKS: u32 = 4;
/// Largest victim lag (in ticks) searched for a geometrically valid backstab.
const MAX_STAB_LAG_TICKS: u32 = 66;
/// Window (ticks before the hurt event) in which a raised `m_bReadyToBackstab` counts.
const READY_WINDOW_LOOKBACK_TICKS: u32 = 60;
/// Pitch samples [attack-6, attack-2] are compared against the attack-tick pitch.
const PITCH_SNAP_LOOKBACK_TICKS: u32 = 6;

// CTFKnife swing trace: 48 HU forward ray swept with an 18 HU half-size hull.
const KNIFE_RANGE: f32 = 48.0;
const KNIFE_HULL: f32 = 18.0;
// Slack for position quantization and interpolation of the recorded snapshots.
const KNIFE_HULL_TOLERANCE: f32 = 4.0;
const BACKSTAB_DOT_TOLERANCE: f32 = 0.05;

#[derive(Clone, Debug)]
struct PlayerSnapshot {
    tick: u32,
    position: Vector,
    view_angle: f32, // yaw
    pitch_angle: f32,
}

#[derive(Clone, Debug, Default)]
struct KnifeInfo {
    owner_entity: Option<EntityId>,
    ready_to_backstab: bool,
    last_update_tick: u32,
}

#[derive(Clone, Debug, Default)]
struct SpyBackstabTracker {
    ready_to_backstab: bool,
    ready_start_tick: Option<u32>,
    last_closed_window: Option<(u32, u32)>, // (start_tick, end_tick)
    recent_attacks: VecDeque<u32>,
    /// `m_bReadyToBackstab` transitions (tick, new value), kept for READY_LOG_MAX_TICKS.
    ready_log: VecDeque<(u32, bool)>,
}

impl SpyBackstabTracker {
    fn record_attack(&mut self, tick: u32) {
        if self.recent_attacks.back().copied() != Some(tick) {
            self.recent_attacks.push_back(tick);
        }
        while self
            .recent_attacks
            .front()
            .is_some_and(|&t| tick.saturating_sub(t) > ATTACK_HISTORY_MAX_TICKS)
        {
            self.recent_attacks.pop_front();
        }
    }
}

/// A knife hit waiting for STAB_EVAL_DELAY_TICKS before it is evaluated.
#[derive(Clone, Debug)]
struct PendingStab {
    tick: u32,
    spy: u64,
    victim_uid: u32,
    victim_class: Class,
    is_backstab: bool,
    damage: u16,
    spy_ping: u16,
    spy_duck: bool,
    victim_duck: bool,
    spy_pos: Vector,
    reaction_ticks: Option<u32>,
    /// Victim snapshots up to 66 ticks before the hurt event, plus the state at the hurt event.
    victim_hist: Vec<PlayerSnapshot>,
}

#[derive(Clone, Debug)]
struct SuspiciousRecord {
    tick: u32,
    player: u64,
    score: u32,
    #[allow(dead_code)]
    detection_type: String,
    data: serde_json::Value,
}

pub struct AutoBackstab {
    params: Parameters,
    jg: JankGuard,
    spy_states: HashMap<u64, SpyBackstabTracker>,
    knife_entities: HashMap<EntityId, KnifeInfo>,
    player_histories: HashMap<u64, VecDeque<PlayerSnapshot>>,
    recent_attacks: HashMap<u64, VecDeque<u32>>,
    pending_stabs: Vec<PendingStab>,
    logged_detections: HashSet<(u64, u32)>,
    detections: Vec<Detection>,
}

impl Default for AutoBackstab {
    fn default() -> Self {
        Self::new()
    }
}

fn find_snapshot(hist: &[PlayerSnapshot], tick: u32) -> Option<&PlayerSnapshot> {
    hist.binary_search_by_key(&tick, |s| s.tick).ok().map(|i| &hist[i])
}

fn forward(yaw: f32, pitch: f32) -> [f32; 3] {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    [p.cos() * y.cos(), p.cos() * y.sin(), -p.sin()]
}

/// Slab test of the segment `p0 + t * d` (t in [0, 1]) against an AABB.
fn segment_hits_box(p0: [f32; 3], d: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for i in 0..3 {
        if d[i].abs() < 1e-9 {
            if p0[i] < bmin[i] || p0[i] > bmax[i] {
                return false;
            }
        } else {
            let mut a = (bmin[i] - p0[i]) / d[i];
            let mut b = (bmax[i] - p0[i]) / d[i];
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// Whether the knife swing trace from `spy` reaches a victim standing at `victim_pos`.
/// A hull sweep against an AABB equals a ray against the AABB grown by the hull size.
fn knife_trace_hits(spy: &PlayerSnapshot, spy_duck: bool, victim_pos: &Vector, victim_duck: bool) -> bool {
    let eye = [spy.position.x, spy.position.y, spy.position.z + if spy_duck { 45.0 } else { 68.0 }];
    let f = forward(spy.view_angle, spy.pitch_angle);
    let d = [f[0] * KNIFE_RANGE, f[1] * KNIFE_RANGE, f[2] * KNIFE_RANGE];
    let h = KNIFE_HULL + KNIFE_HULL_TOLERANCE;
    let top = if victim_duck { 62.0 } else { 82.0 };
    let bmin = [victim_pos.x - 24.0 - h, victim_pos.y - 24.0 - h, victim_pos.z - h];
    let bmax = [victim_pos.x + 24.0 + h, victim_pos.y + 24.0 + h, victim_pos.z + top + h];
    segment_hits_box(eye, d, bmin, bmax)
}

/// CTFKnife::IsBehindAndFacingTarget dot products: (behind, facing, view).
/// The game requires behind > 0, facing > 0.5 and view > -0.3.
fn backstab_dots(spy_pos: &Vector, spy_yaw: f32, victim_pos: &Vector, victim_yaw: f32) -> (f32, f32, f32) {
    let (tx, ty) = (victim_pos.x - spy_pos.x, victim_pos.y - spy_pos.y);
    let n = (tx * tx + ty * ty).sqrt().max(1e-6);
    let (tx, ty) = (tx / n, ty / n);
    let (vy, sy) = (victim_yaw.to_radians(), spy_yaw.to_radians());
    let (vfx, vfy) = (vy.cos(), vy.sin());
    let (sfx, sfy) = (sy.cos(), sy.sin());
    (tx * vfx + ty * vfy, tx * sfx + ty * sfy, vfx * sfx + vfy * sfy)
}

/// Start of the final cluster of attack ticks at or before the hurt tick
/// (the hurt event is usually reported 1-3 ticks after the attack itself).
fn attack_tick(attacks: Option<&VecDeque<u32>>, hurt_tick: u32) -> u32 {
    let mut atk: Vec<u32> = attacks
        .map(|a| a.iter().copied().filter(|&t| t + 6 >= hurt_tick && t <= hurt_tick).collect())
        .unwrap_or_default();
    atk.sort_unstable();
    atk.dedup();
    let Some(&last) = atk.last() else {
        return hurt_tick;
    };
    let mut start = last;
    for &t in atk.iter().rev() {
        if start - t <= 2 {
            start = t;
        } else {
            break;
        }
    }
    start
}

impl AutoBackstab {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("preswing_check_ticks".to_string(), Parameter::Int(3)),
                ("lag_excess_strong_ticks".to_string(), Parameter::Float(20.0)),
                ("lag_excess_medium_ticks".to_string(), Parameter::Float(14.0)),
                ("min_victim_history_ticks".to_string(), Parameter::Int(30)),
                ("pitch_snap_max_pitch".to_string(), Parameter::Float(0.5)),
                ("pitch_snap_min_jump".to_string(), Parameter::Float(3.0)),
            ]),
            jg: JankGuard::default(),
            spy_states: HashMap::new(),
            knife_entities: HashMap::new(),
            player_histories: HashMap::new(),
            recent_attacks: HashMap::new(),
            pending_stabs: Vec::new(),
            logged_detections: HashSet::new(),
            detections: Vec::new(),
        }
    }

    fn update_spy_ready_state(&mut self, sid: u64, is_ready: bool, tick: u32) {
        let tracker = self.spy_states.entry(sid).or_default();
        if is_ready != tracker.ready_to_backstab {
            tracker.ready_log.push_back((tick, is_ready));
            while tracker
                .ready_log
                .front()
                .is_some_and(|&(t, _)| tick.saturating_sub(t) > READY_LOG_MAX_TICKS)
            {
                tracker.ready_log.pop_front();
            }
        }
        if is_ready && !tracker.ready_to_backstab {
            tracker.ready_to_backstab = true;
            tracker.ready_start_tick = Some(tick);
        } else if !is_ready && tracker.ready_to_backstab {
            if let Some(t_ready) = tracker.ready_start_tick {
                tracker.last_closed_window = Some((t_ready, tick));
            }
            tracker.ready_to_backstab = false;
            tracker.ready_start_tick = None;
        }
    }

    fn reset_spy_state(&mut self, sid: u64) {
        if let Some(tracker) = self.spy_states.get_mut(&sid) {
            tracker.ready_to_backstab = false;
            tracker.ready_start_tick = None;
            tracker.last_closed_window = None;
            tracker.recent_attacks.clear();
        }
    }

    fn record_attack(&mut self, sid: u64, tick: u32) {
        let tracker = self.spy_states.entry(sid).or_default();
        tracker.record_attack(tick);

        let attacks = self.recent_attacks.entry(sid).or_default();
        if attacks.back().copied() != Some(tick) {
            attacks.push_back(tick);
        }
        while attacks
            .front()
            .is_some_and(|&t| tick.saturating_sub(t) > ATTACK_HISTORY_MAX_TICKS)
        {
            attacks.pop_front();
        }
    }

    fn record_player_snapshots(&mut self, state: &CheatAnalyserState) {
        let tick = u32::from(state.tick);
        for player in &state.players {
            if !player.in_pvs || player.state != PlayerState::Alive {
                continue;
            }
            let Some(sid) = player_id(player) else {
                continue;
            };

            let snapshot = PlayerSnapshot {
                tick,
                position: player.position,
                view_angle: player.view_angle,
                pitch_angle: player.pitch_angle,
            };

            let history = self.player_histories.entry(sid).or_default();
            if history.back().is_some_and(|s| s.tick == tick) {
                history.pop_back();
            }
            history.push_back(snapshot);
        }

        self.player_histories.retain(|_, history| {
            while history
                .front()
                .is_some_and(|s| tick.saturating_sub(s.tick) > SNAPSHOT_HISTORY_MAX_TICKS)
            {
                history.pop_front();
            }
            !history.is_empty()
        });
    }

    /// Report a record on its own if its score reaches DETECTION_SCORE_THRESHOLD.
    fn emit_record(&mut self, rec: SuspiciousRecord) -> Option<Detection> {
        if rec.score < DETECTION_SCORE_THRESHOLD || !self.logged_detections.insert((rec.player, rec.tick)) {
            return None;
        }
        let det = Detection {
            tick: rec.tick,
            algorithm: self.algorithm_name().to_string(),
            player: rec.player,
            data: rec.data,
        };
        self.detections.push(det.clone());
        Some(det)
    }

    /// Smallest victim lag (ticks before the attack) at which the stab is geometrically
    /// possible. With `check_dots` the backstab angle conditions must hold as well,
    /// otherwise only the swing trace range is tested.
    fn min_valid_lag(&self, p: &PendingStab, t_attack: u32, check_dots: bool) -> Option<u32> {
        let spy_hist = self.player_histories.get(&p.spy)?;
        let spy_snaps: Vec<&PlayerSnapshot> = [0i64, -1, 1]
            .iter()
            .filter_map(|&k| {
                let t = (t_attack as i64 + k) as u32;
                spy_hist.iter().rev().find(|s| s.tick == t)
            })
            .collect();
        if spy_snaps.is_empty() {
            return None;
        }
        // Victim eye angles are not lag compensated, so the victim's current yaw is used.
        let victim_yaws: Vec<f32> = [t_attack, t_attack.saturating_sub(1), t_attack + 1, p.tick]
            .iter()
            .filter_map(|&t| find_snapshot(&p.victim_hist, t).map(|s| s.view_angle))
            .collect();
        if victim_yaws.is_empty() {
            return None;
        }

        for lag in 0..=MAX_STAB_LAG_TICKS.min(t_attack) {
            let Some(victim) = find_snapshot(&p.victim_hist, t_attack - lag) else {
                continue;
            };
            for spy in &spy_snaps {
                if !knife_trace_hits(spy, p.spy_duck, &victim.position, p.victim_duck) {
                    continue;
                }
                if !check_dots {
                    return Some(lag);
                }
                for &victim_yaw in &victim_yaws {
                    let (behind, facing, view) = backstab_dots(&spy.position, spy.view_angle, &victim.position, victim_yaw);
                    if behind > -BACKSTAB_DOT_TOLERANCE
                        && facing > 0.5 - BACKSTAB_DOT_TOLERANCE
                        && view > -0.3 - BACKSTAB_DOT_TOLERANCE
                    {
                        return Some(lag);
                    }
                }
            }
        }
        None
    }

    /// Melee aimbots (e.g. Amalgam) aim at the victim's origin with the target height clamped
    /// to the spy's eye level, so on level ground the attack tick has a perfectly flat pitch.
    /// Returns (attack pitch, closest preceding pitch delta) when such a snap happened.
    fn level_pitch_snap(&self, spy: u64, t_attack: u32) -> Option<(f32, f32)> {
        let hist = self.player_histories.get(&spy)?;
        let pitch = hist.iter().rev().find(|s| s.tick == t_attack)?.pitch_angle;
        let max_pitch = get_parameter_value::<f32>(&self.params, "pitch_snap_max_pitch");
        let min_jump = get_parameter_value::<f32>(&self.params, "pitch_snap_min_jump");
        if pitch.abs() >= max_pitch {
            return None;
        }
        let min_delta = hist
            .iter()
            .filter(|s| s.tick + PITCH_SNAP_LOOKBACK_TICKS >= t_attack && s.tick + 2 <= t_attack)
            .map(|s| (s.pitch_angle - pitch).abs())
            .fold(f32::MAX, f32::min);
        (min_delta != f32::MAX && min_delta >= min_jump).then_some((pitch, min_delta))
    }

    fn evaluate_stab(&mut self, p: PendingStab) -> Option<Detection> {
        let t_attack = attack_tick(self.recent_attacks.get(&p.spy), p.tick);

        let had_ready_window = self.spy_states.get(&p.spy).is_some_and(|tr| {
            tr.ready_log
                .iter()
                .any(|&(t, ready)| ready && t + READY_WINDOW_LOOKBACK_TICKS >= p.tick)
        });

        // Lag compensation validity: legit stabs are possible against a victim record no older
        // than the spy's latency plus interpolation; backtrack cheats reach much older records.
        let ping_ticks = p.spy_ping as f32 / 15.0;
        let min_hist = get_parameter_value::<i32>(&self.params, "min_victim_history_ticks").max(0) as usize;
        let strong = get_parameter_value::<f32>(&self.params, "lag_excess_strong_ticks");
        let medium = get_parameter_value::<f32>(&self.params, "lag_excess_medium_ticks");

        let mut score = 0u32;
        let mut types: Vec<&str> = Vec::new();
        let mut valid_lag = None;
        let mut range_lag = None;
        if p.is_backstab && p.victim_hist.len() > min_hist {
            valid_lag = self.min_valid_lag(&p, t_attack, true);
            range_lag = self.min_valid_lag(&p, t_attack, false);
            let excess = |lag: Option<u32>| lag.map(|l| l as f32 - ping_ticks);
            match (excess(valid_lag), excess(range_lag)) {
                (None, None) => {
                    score += STRONG_EVIDENCE_SCORE;
                    types.push("impossible_backstab");
                }
                (None, Some(er)) if er > strong => {
                    score += STRONG_EVIDENCE_SCORE;
                    types.push("extended_backtrack_stab");
                }
                (None, Some(_)) => {
                    score += WEAK_EVIDENCE_SCORE;
                    types.push("facestab_backtrack");
                }
                (Some(e), _) if e > strong => {
                    score += STRONG_EVIDENCE_SCORE;
                    types.push("extended_backtrack_stab");
                }
                (Some(e), _) if e > medium => {
                    score += WEAK_EVIDENCE_SCORE;
                    types.push("backtrack_stab");
                }
                _ => {}
            }
        }

        let pitch_snap = self.level_pitch_snap(p.spy, t_attack);
        if pitch_snap.is_some() {
            score += STRONG_EVIDENCE_SCORE;
            types.push("level_pitch_snap");
        }

        if score < DETECTION_SCORE_THRESHOLD {
            return None;
        }

        let lag_excess = valid_lag.map(|l| l as f32 - ping_ticks);
        let round1 = |v: f32| (v * 10.0).round() / 10.0;
        let victim_now = p.victim_hist.last().map(|s| s.position).unwrap_or(p.spy_pos);
        let (dx, dy) = (victim_now.x - p.spy_pos.x, victim_now.y - p.spy_pos.y);
        let data = json!({
            "attack_tick": t_attack,
            "reaction_ticks": p.reaction_ticks,
            "had_ready_window": had_ready_window,
            "min_valid_lag": valid_lag,
            "min_range_lag": range_lag,
            "ping_ticks": round1(ping_ticks),
            "lag_excess": lag_excess.map(round1),
            "attack_pitch": pitch_snap.map(|(p, _)| round1(p)),
            "pitch_jump": pitch_snap.map(|(_, j)| round1(j)),
            "victim_id": p.victim_uid,
            "victim_class": format_class_name(p.victim_class),
            "spy_pos": [round1(p.spy_pos.x), round1(p.spy_pos.y), round1(p.spy_pos.z)],
            "victim_pos": [round1(victim_now.x), round1(victim_now.y), round1(victim_now.z)],
            "dist2d": round1((dx * dx + dy * dy).sqrt()),
            "damage": p.damage,
            "detection_types": types,
            "score": score,
        });
        let rec = SuspiciousRecord {
            tick: p.tick,
            player: p.spy,
            score,
            detection_type: types.join(","),
            data,
        };
        self.emit_record(rec)
    }

    /// Check for 1-tick silent aimbot flicks
    fn check_silent_flick(
        &mut self,
        sid: u64,
        _tick: u32,
        state: &CheatAnalyserState,
        player: &Player,
    ) -> Option<SuspiciousRecord> {
        let history = self.player_histories.get(&sid)?;
        let n = history.len();
        if n < 3 {
            return None;
        }

        let s0 = &history[n - 3];
        let s1 = &history[n - 2];
        let s2 = &history[n - 1];
        // Must be within 2..=3 ticks (allowing 1 dropped demo packet tick)
        let dt = s2.tick.saturating_sub(s0.tick);
        if dt < 2 || dt > 3 {
            return None;
        }
        if s1.tick <= s0.tick || s2.tick <= s1.tick {
            return None;
        }

        // Teleport safety
        if self.jg.teleported(&sid, s1.tick) <= 20 {
            return None;
        }

        // Pitch safety (exclude clamping / flipping glitches)
        if s0.pitch_angle.abs() >= 87.0
            || s1.pitch_angle.abs() >= 87.0
            || s2.pitch_angle.abs() >= 87.0
        {
            return None;
        }
        if (s1.pitch_angle - s0.pitch_angle).abs() >= 80.0
            || (s2.pitch_angle - s1.pitch_angle).abs() >= 80.0
        {
            return None;
        }

        let p01 = ((s1.position.x - s0.position.x).powi(2)
            + (s1.position.y - s0.position.y).powi(2)
            + (s1.position.z - s0.position.z).powi(2))
        .sqrt();
        let p12 = ((s2.position.x - s1.position.x).powi(2)
            + (s2.position.y - s1.position.y).powi(2)
            + (s2.position.z - s1.position.z).powi(2))
        .sqrt();
        if p01 > 50.0 || p12 > 50.0 {
            return None;
        }

        if self.jg.spawned(&sid, s1.tick) <= 10 || self.jg.teleported(&sid, s1.tick) <= 20 {
            return None;
        }

        if player.class != Class::Spy
            || player.is_taunting()
            || player.health == 0
            || player.state != PlayerState::Alive
        {
            return None;
        }

        let d_in = angle_delta((s0.view_angle, s0.pitch_angle), (s1.view_angle, s1.pitch_angle));
        let d_out = angle_delta((s1.view_angle, s1.pitch_angle), (s2.view_angle, s2.pitch_angle));
        let d_base = angle_delta((s0.view_angle, s0.pitch_angle), (s2.view_angle, s2.pitch_angle));

        if d_in < 30.0 || d_out < 30.0 {
            return None;
        }

        let ratio = if (d_in + d_out) > 0.001 {
            d_base / (d_in + d_out)
        } else {
            1.0
        };

        // Attack correlation: check if an attack happened within [-1, +3] ticks of the flick
        let had_attack = if let Some(attacks) = self.recent_attacks.get(&sid) {
            attacks
                .iter()
                .any(|&atk_t| atk_t >= s1.tick.saturating_sub(1) && atk_t <= s1.tick + 3)
        } else {
            false
        };
        let attack_correlated = had_attack || self.jg.fired(&sid, s1.tick) <= 3;

        // Scale baseline threshold for demo gaps (dt=2: 16.0 deg, dt=3: 24.0 deg)
        let d_base_threshold = 16.0 + (dt.saturating_sub(2)) as f32 * 8.0;

        // If attack correlated, allow standard silent flick drift.
        // If attack was a miss (no damage/anim event), require pristine silent flick snap-back (d_base <= 8 deg, ratio <= 0.10).
        if attack_correlated {
            if ratio > 0.30 || d_base > d_base_threshold || d_in < 40.0 || d_out < 40.0 {
                return None;
            }
        } else {
            if ratio > 0.10 || d_base > 8.0 || d_in < 50.0 || d_out < 50.0 {
                return None;
            }
        }

        // Target verification: view ray at tick s1 must align with an ENEMY within knife range (current or backtracked)
        let max_fov = 35.0;
        let mut target_enemy = None;
        for enemy in &state.players {
            if enemy.team == player.team {
                continue;
            }
            let Some(esid) = player_id(enemy) else {
                continue;
            };

            // Check enemy backtracked history at or near s1.tick (even if victim died from the stab at current tick)
            if let Some(ehist) = self.player_histories.get(&esid) {
                for esnap in ehist.iter() {
                    let lag = s1.tick.saturating_sub(esnap.tick);
                    if lag <= 25 {
                        let edx = esnap.position.x - s1.position.x;
                        let edy = esnap.position.y - s1.position.y;
                        let edz = esnap.position.z - s1.position.z;
                        let edist = (edx * edx + edy * edy + edz * edz).sqrt();
                        if edist <= 380.0 {
                            let aim_yaw = edy.atan2(edx).to_degrees();
                            let aim_fov =
                                ((aim_yaw - s1.view_angle + 180.0).rem_euclid(360.0) - 180.0).abs();
                            if aim_fov <= max_fov {
                                target_enemy = Some((esid, edist, aim_fov, lag));
                                break;
                            }
                        }
                    }
                }
            } else if enemy.state == PlayerState::Alive {
                let dx = enemy.position.x - s1.position.x;
                let dy = enemy.position.y - s1.position.y;
                let dz = enemy.position.z - s1.position.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= 380.0 {
                    let aim_yaw = dy.atan2(dx).to_degrees();
                    let aim_fov =
                        ((aim_yaw - s1.view_angle + 180.0).rem_euclid(360.0) - 180.0).abs();
                    if aim_fov <= max_fov {
                        target_enemy = Some((esid, dist, aim_fov, 0u32));
                        break;
                    }
                }
            }
            if target_enemy.is_some() {
                break;
            }
        }

        let (target_sid, target_dist, target_fov, target_lag) = target_enemy?;

        let mut flick_score = if (d_in >= 45.0 && d_out >= 45.0) || d_in + d_out >= 110.0 {
            100
        } else if d_in >= 35.0 && d_out >= 35.0 {
            60
        } else {
            40
        };

        if attack_correlated {
            flick_score += 20;
        }
        if target_fov <= 15.0 {
            flick_score += 20;
        }
        if attack_correlated {
            flick_score += 30;
        }

        let data_json = json!({
            "tick": s1.tick,
            "d_in": (d_in * 10.0).round() / 10.0,
            "d_out": (d_out * 10.0).round() / 10.0,
            "d_base": (d_base * 10.0).round() / 10.0,
            "ratio": (ratio * 100.0).round() / 100.0,
            "target_player": target_sid,
            "target_dist": (target_dist * 10.0).round() / 10.0,
            "target_fov": (target_fov * 10.0).round() / 10.0,
            "target_lag": target_lag,
            "attack_correlated": attack_correlated,
            "detection_types": ["silent_aimbot_flick"],
            "score": flick_score
        });

        Some(SuspiciousRecord {
            tick: s1.tick,
            player: sid,
            score: flick_score,
            detection_type: "silent_aimbot_flick".to_string(),
            data: data_json,
        })
    }
}

impl<'a> CheatAlgorithm<'a> for AutoBackstab {
    fn default(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/auto_backstab"
    }

    fn tooltip(&self) -> &'static str {
        "Detects suspicious backstabs, including impossible face-stabs and extended backtrack stabs."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        let mut types = self.jg.handled_messages()?;
        if !types.contains(&MessageType::PacketEntities) {
            types.push(MessageType::PacketEntities);
        }
        if !types.contains(&MessageType::GameEvent) {
            types.push(MessageType::GameEvent);
        }
        if !types.contains(&MessageType::TempEntities) {
            types.push(MessageType::TempEntities);
        }
        types.sort_by_key(|a| format!("{:?}", a));
        types.dedup();
        Ok(types)
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);
        let ticknum = u32::from(tick);

        match message {
            Message::PacketEntities(msg) => {
                for entity in &msg.entities {
                    if entity.update_type == UpdateType::Delete {
                        self.knife_entities.remove(&entity.entity_index);
                        continue;
                    }

                    let class_name: &str = parser_state
                        .server_classes
                        .get(usize::from(entity.server_class))
                        .map(|class| class.name.as_str())
                        .unwrap_or("");

                    let is_knife_class = class_name == "CTFKnife"
                        || class_name.contains("Knife")
                        || class_name == "CTFWeaponKnife";

                    let mut ready_prop = None;
                    let mut owner_prop = None;

                    for prop in entity.props(parser_state) {
                        if let Some((_, prop_name)) = prop.identifier.names() {
                            match prop_name.as_str() {
                                "m_bReadyToBackstab" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        ready_prop = Some(val != 0);
                                    }
                                }
                                "m_hOwnerEntity" | "m_hOwner" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        let handle = val as u32;
                                        let ent_id = handle_to_entid(handle);
                                        if u32::from(ent_id) != 0x7FF && u32::from(ent_id) != 0 {
                                            owner_prop = Some(ent_id);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    if is_knife_class || ready_prop.is_some() {
                        let knife_entry = self.knife_entities.entry(entity.entity_index).or_default();
                        if let Some(owner) = owner_prop {
                            knife_entry.owner_entity = Some(owner);
                        }
                        if let Some(ready) = ready_prop {
                            knife_entry.ready_to_backstab = ready;
                            knife_entry.last_update_tick = ticknum;

                            let owner_sid = knife_entry.owner_entity.and_then(|owner_ent| {
                                state.players.iter().find(|p| p.entity == owner_ent).and_then(player_id)
                            });

                            if let Some(sid) = owner_sid {
                                self.update_spy_ready_state(sid, ready, ticknum);
                            }
                        }
                    }
                }
            }
            Message::TempEntities(msg) => {
                for event in &msg.events {
                    if let Some(server_class) = parser_state.server_classes.get(usize::from(event.class_id)) {
                        if server_class.name.as_str() == "CTEPlayerAnimEvent" {
                            const ANIM_PLAYER: SendPropIdentifier =
                                SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_hPlayer");
                            if let Some(prop) = event.props.iter().find(|p| p.identifier == ANIM_PLAYER) {
                                if let Some(id64) = i64::try_from(&prop.value).ok()
                                    .and_then(|id| id.try_into().ok())
                                    .map(|id| handle_to_entid(id))
                                    .and_then(|id| state.entid_to_userid.get(&id))
                                    .and_then(|uid| state.userid_to_id64.get(uid))
                                {
                                    self.record_attack(*id64, ticknum);
                                }
                            }
                        }
                    }
                }
            }
            Message::GameEvent(event_msg) => match &event_msg.event {
                GameEvent::PlayerSpawn(spawn) => {
                    if let Some(sid) = state.get_id64_from_userid(spawn.user_id.into()) {
                        self.reset_spy_state(sid);
                    }
                }
                GameEvent::PostInventoryApplication(app) => {
                    if let Some(sid) = state.get_id64_from_userid(app.user_id.into()) {
                        self.reset_spy_state(sid);
                    }
                }
                GameEvent::PlayerHurt(hurt) => self.on_player_hurt(hurt, state, ticknum),
                _ => {}
            },
            _ => {}
        }

        Ok(Vec::new())
    }

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        self.record_player_snapshots(state);
        let ticknum = u32::from(state.tick);
        let mut new_detections = Vec::new();

        for player in &state.players {
            if player.state != PlayerState::Alive {
                continue;
            }
            let Some(sid) = player_id(player) else {
                continue;
            };

            if player.class == Class::Spy {
                if let Some(active_weapon_ent) = player.active_weapon {
                    let ready_state = self
                        .knife_entities
                        .get(&active_weapon_ent)
                        .map(|k| k.ready_to_backstab);
                    if let Some(ready) = ready_state {
                        if let Some(knife_info) = self.knife_entities.get_mut(&active_weapon_ent) {
                            knife_info.owner_entity = Some(player.entity);
                        }
                        self.update_spy_ready_state(sid, ready, ticknum);
                    }
                }

                if self.jg.fired(&sid, ticknum) <= 1 {
                    self.record_attack(sid, ticknum);
                }

                if let Some(flick_record) = self.check_silent_flick(sid, ticknum, state, player) {
                    new_detections.extend(self.emit_record(flick_record));
                }
            }
        }

        let (due, waiting): (Vec<PendingStab>, Vec<PendingStab>) = std::mem::take(&mut self.pending_stabs)
            .into_iter()
            .partition(|p| ticknum >= p.tick + STAB_EVAL_DELAY_TICKS);
        self.pending_stabs = waiting;
        for p in due {
            new_detections.extend(self.evaluate_stab(p));
        }

        Ok(new_detections)
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let mut out = Vec::new();
        for p in std::mem::take(&mut self.pending_stabs) {
            out.extend(self.evaluate_stab(p));
        }
        Ok(out)
    }
}

impl AutoBackstab {
    /// Queue knife hits by spies for delayed evaluation.
    fn on_player_hurt(
        &mut self,
        hurt: &tf_demo_parser::demo::gameevent_gen::PlayerHurtEvent,
        state: &CheatAnalyserState,
        ticknum: u32,
    ) {
        let victim_uid = u32::from(hurt.user_id);
        let attacker_uid = u32::from(hurt.attacker);
        let weapon_id = hurt.weapon_id as u16;
        let is_knife_hit = weapon_id == 7 || hurt.custom == 2;
        if attacker_uid == victim_uid || hurt.damage_amount == 0 || !is_knife_hit {
            return;
        }

        let find = |uid: u32| {
            state.players.iter().find(|p| p.info.as_ref().is_some_and(|i| u32::from(i.user_id) == uid))
        };
        let (Some(attacker), Some(victim)) = (find(attacker_uid), find(victim_uid)) else {
            return;
        };
        let (Some(attacker_sid), Some(victim_sid)) = (player_id(attacker), player_id(victim)) else {
            return;
        };

        let fresh = |p: &Player, sid: u64| {
            p.in_pvs
                || self
                    .player_histories
                    .get(&sid)
                    .and_then(|h| h.back())
                    .is_some_and(|s| ticknum.saturating_sub(s.tick) <= 2)
        };
        if !fresh(attacker, attacker_sid) || !fresh(victim, victim_sid) || attacker.is_taunting() {
            return;
        }

        self.record_attack(attacker_sid, ticknum);
        if self.jg.teleported(&attacker_sid, ticknum) <= 15 || self.jg.teleported(&victim_sid, ticknum) <= 15 {
            return;
        }

        // Ready-window reaction time, reported for context.
        let tracker = self.spy_states.entry(attacker_sid).or_default();
        let t_ready = if let Some(t_r) = tracker.ready_start_tick {
            Some(t_r)
        } else if let Some((t_r, t_closed)) = tracker.last_closed_window {
            (ticknum.saturating_sub(t_closed) <= 6).then_some(t_r)
        } else {
            None
        };
        tracker.last_closed_window = None;
        let preswing_ticks = get_parameter_value::<i32>(&self.params, "preswing_check_ticks").max(0) as u32;
        let was_preswinging = t_ready.is_some_and(|t_r| {
            tracker
                .recent_attacks
                .iter()
                .any(|&a| a >= t_r.saturating_sub(preswing_ticks) && a < t_r)
        });
        let reaction_ticks = t_ready.filter(|_| !was_preswinging).map(|t_r| ticknum.saturating_sub(t_r));

        let mut victim_hist: Vec<PlayerSnapshot> = self
            .player_histories
            .get(&victim_sid)
            .map(|h| {
                h.iter()
                    .filter(|s| s.tick < ticknum && ticknum - s.tick <= SNAPSHOT_HISTORY_MAX_TICKS)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        victim_hist.push(PlayerSnapshot {
            tick: ticknum,
            position: victim.position,
            view_angle: victim.view_angle,
            pitch_angle: victim.pitch_angle,
        });

        self.pending_stabs.push(PendingStab {
            tick: ticknum,
            spy: attacker_sid,
            victim_uid,
            victim_class: victim.class,
            is_backstab: hurt.custom == 2,
            damage: hurt.damage_amount,
            spy_ping: attacker.ping,
            spy_duck: attacker.is_ducking(),
            victim_duck: victim.is_ducking(),
            spy_pos: attacker.position,
            reaction_ticks,
            victim_hist,
        });
    }
}

fn player_id(player: &Player) -> Option<u64> {
    let info = player.info.as_ref()?;
    if info.steam_id == "BOT" {
        return None;
    }
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

fn format_class_name(class: Class) -> &'static str {
    match class {
        Class::Scout => "Scout",
        Class::Sniper => "Sniper",
        Class::Soldier => "Soldier",
        Class::Demoman => "Demoman",
        Class::Medic => "Medic",
        Class::Heavy => "Heavy",
        Class::Pyro => "Pyro",
        Class::Spy => "Spy",
        Class::Engineer => "Engineer",
        Class::Other => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tf_demo_parser::demo::gameevent_gen::PlayerHurtEvent;
    use tf_demo_parser::demo::message::gameevent::GameEventMessage;
    use tf_demo_parser::demo::parser::analyser::{Team, UserId, UserInfo};

    const SPY_STEAM: &str = "[U:1:12345678]";
    const VICTIM_STEAM: &str = "[U:1:87654321]";

    fn create_test_player(entity_id: u32, user_id: u32, steam_id: &str, class: Class, team: Team, pos: Vector) -> Player {
        Player {
            entity: EntityId::from(entity_id),
            position: pos,
            health: 125,
            max_health: 125,
            class,
            team,
            view_angle: 0.0,
            pitch_angle: 0.0,
            state: PlayerState::Alive,
            info: Some(UserInfo {
                classes: Default::default(),
                name: "Test".to_string(),
                user_id: UserId::from(user_id as u16),
                steam_id: steam_id.to_string(),
                entity_id: EntityId::from(entity_id),
                team,
            }),
            charge: 0,
            simtime: 0,
            ping: 30,
            in_pvs: true,
            active_weapon: None,
            cond: 0,
            cond_ex: 0,
            cond_ex2: 0,
            cond_ex3: 0,
            invis_change_complete_time: 0.0,
            flags: 1,
        }
    }

    fn hurt_message() -> Message<'static> {
        Message::GameEvent(GameEventMessage {
            event: GameEvent::PlayerHurt(PlayerHurtEvent {
                user_id: 6,
                attacker: 1,
                health: 0,
                damage_amount: 750,
                crit: true,
                weapon_id: 7,
                bonus_effect: 0,
                custom: 2,
                show_disguised_crit: false,
                mini_crit: false,
                all_see_crit: false,
            }),
            event_type_id: unsafe { std::mem::zeroed() },
            event_type: tf_demo_parser::demo::gameevent_gen::GameEventType::PlayerHurt,
        })
    }

    fn spy_sid() -> u64 {
        u64::from(SteamID::from_steam3(SPY_STEAM).unwrap())
    }

    /// Drives the algorithm tick by tick; `setup` positions the spy and victim each tick.
    struct Sim {
        algo: AutoBackstab,
        state: CheatAnalyserState,
        parser_state: ParserState,
        tick: u32,
        detections: Vec<Detection>,
    }

    impl Sim {
        fn new() -> Self {
            let mut state = CheatAnalyserState::default();
            state.set_entid_to_userid(EntityId::from(1u32), UserId::from(1u16));
            state.set_entid_to_userid(EntityId::from(2u32), UserId::from(6u16));
            Sim { algo: AutoBackstab::new(), state, parser_state: ParserState::new(24, |_| false, false), tick: 1000, detections: Vec::new() }
        }

        fn step(&mut self, spy: (Vector, f32, f32), victim: (Vector, f32)) {
            self.tick += 1;
            self.state.tick = self.tick.into();
            let mut s = create_test_player(1, 1, SPY_STEAM, Class::Spy, Team::Blue, spy.0);
            s.view_angle = spy.1;
            s.pitch_angle = spy.2;
            let mut v = create_test_player(2, 6, VICTIM_STEAM, Class::Medic, Team::Red, victim.0);
            v.view_angle = victim.1;
            self.state.players = vec![s, v];
            let d = self.algo.on_tick(&self.state, &self.parser_state).unwrap();
            self.detections.extend(d);
        }

        fn stab(&mut self) {
            let d = self.algo.on_message(&hurt_message(), &self.state, &self.parser_state, self.tick.into()).unwrap();
            self.detections.extend(d);
        }

        fn finish(&mut self) {
            let d = self.algo.finish().unwrap();
            self.detections.extend(d);
        }
    }

    fn v(x: f32, y: f32, z: f32) -> Vector {
        Vector { x, y, z }
    }

    #[test]
    fn test_geometry_helpers() {
        let spy = PlayerSnapshot { tick: 0, position: v(-50.0, 0.0, 0.0), view_angle: 0.0, pitch_angle: 0.0 };
        assert!(knife_trace_hits(&spy, false, &v(0.0, 0.0, 0.0), false));
        assert!(!knife_trace_hits(&spy, false, &v(200.0, 0.0, 0.0), false));

        let (behind, facing, view) = backstab_dots(&spy.position, 0.0, &v(0.0, 0.0, 0.0), 0.0);
        assert!(behind > 0.99 && facing > 0.99 && view > 0.99);
        // Victim facing the spy: not behind.
        let (behind, _, view) = backstab_dots(&spy.position, 0.0, &v(0.0, 0.0, 0.0), 180.0);
        assert!(behind < 0.0 && view < -0.3);
    }

    #[test]
    fn test_attack_tick_uses_start_of_final_cluster() {
        let attacks: VecDeque<u32> = VecDeque::from(vec![80, 97, 98, 100]);
        assert_eq!(attack_tick(Some(&attacks), 100), 97);
        assert_eq!(attack_tick(None, 100), 100);
    }

    #[test]
    fn test_legit_backstab_not_flagged() {
        let mut sim = Sim::new();
        for i in 0..40 {
            sim.step((v(-150.0 + i as f32 * 2.5, 0.0, 0.0), 0.0, 15.0), (v(0.0, 0.0, 0.0), 0.0));
        }
        sim.stab();
        for _ in 0..5 {
            sim.step((v(-50.0, 0.0, 0.0), 0.0, 15.0), (v(0.0, 0.0, 0.0), 0.0));
        }
        sim.finish();
        assert!(sim.detections.is_empty());
    }

    #[test]
    fn test_impossible_backstabs_flagged_individually() {
        let mut sim = Sim::new();
        for _ in 0..2 {
            // Victim never comes within knife reach in the recorded history.
            for _ in 0..40 {
                sim.step((v(0.0, 0.0, 0.0), 0.0, 10.0), (v(300.0, 0.0, 0.0), 0.0));
            }
            sim.stab();
        }
        for _ in 0..5 {
            sim.step((v(0.0, 0.0, 0.0), 0.0, 10.0), (v(300.0, 0.0, 0.0), 0.0));
        }
        let spy_dets: Vec<_> = sim.detections.iter().filter(|d| d.player == spy_sid()).collect();
        assert_eq!(spy_dets.len(), 2);
        assert_eq!(spy_dets[0].data["detection_types"][0], "impossible_backstab");
    }

    #[test]
    fn test_medium_lag_backstab_not_flagged() {
        let mut sim = Sim::new();
        // Victim was within reach 16-18 ticks before the stab (ping 30 ms = 2 ticks),
        // giving a lag excess between the medium and strong thresholds.
        for i in 0..40 {
            let x = if i < 22 { 0.0 } else { 300.0 };
            sim.step((v(-50.0, 0.0, 0.0), 0.0, 10.0), (v(x, 0.0, 0.0), 0.0));
        }
        sim.stab();
        for _ in 0..5 {
            sim.step((v(-50.0, 0.0, 0.0), 0.0, 10.0), (v(300.0, 0.0, 0.0), 0.0));
        }
        sim.finish();
        assert!(sim.detections.is_empty());
    }

    #[test]
    fn test_level_pitch_snaps_flagged() {
        let mut sim = Sim::new();
        for _ in 0..3 {
            for _ in 0..40 {
                sim.step((v(-50.0, 0.0, 0.0), 0.0, 20.0), (v(0.0, 0.0, 0.0), 0.0));
            }
            // Attack tick: pitch snaps to exactly level, then returns.
            sim.step((v(-50.0, 0.0, 0.0), 0.0, 0.0), (v(0.0, 0.0, 0.0), 0.0));
            sim.stab();
        }
        for _ in 0..5 {
            sim.step((v(-50.0, 0.0, 0.0), 0.0, 20.0), (v(0.0, 0.0, 0.0), 0.0));
        }
        let spy_dets: Vec<_> = sim.detections.iter().filter(|d| d.player == spy_sid()).collect();
        assert_eq!(spy_dets.len(), 3);
        assert!(spy_dets.iter().all(|d| d.data["detection_types"].as_array().unwrap().iter().any(|t| t == "level_pitch_snap")));
    }
}
