//! Learned auto backstab detection on spectated spies' backstabs. See docs/ml_autobackstab.md.
//!
//! Melee aimbots with auto backstab (Amalgam `AimbotMelee`, RijiN `melee_aimbot`) swing on the first tick a
//! backstab is possible, aim at the victim's origin (Amalgam: height clamped to the spy's eye, i.e. level),
//! often through a backtracked record, and with silent aim the eye angles flick onto the victim for the
//! attack tick only. Every backstab of a spectated spy keeps the spy's and the victim's samples around it;
//! per stab the aim error to the victim, the snap into and out of the attack tick, how long the stab had
//! already been possible, the knife readiness, lag compensation validity and movement are computed. The
//! stabs of a spy are aggregated over the demo, together with the knife swing / hit counts, and scored by
//! a model trained on the demo database (`assets/ml_autobackstab_model.json`). Spies at or above
//! `min_score` are reported at their highest scoring stabs (a second, per-stab model).
//!
//! `examples/dump_ml_autobackstab_features.rs` dumps the same features for training.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::demo::parser::analyser::Class;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

use crate::util::helpers::resolve_recorder_sid;
use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{handle_to_entid, FxHashMap};
use crate::util::nocrex::jankguard::JankGuard;

/// Stabs are evaluated this many ticks after the hurt event (samples after the attack)
const EVAL_DELAY: u32 = 10;
/// Largest victim lag searched for a geometrically valid backstab, in ticks
const MAX_STAB_LAG: u32 = 66;
/// Spy samples before the attack used for the run / snap features
const LOOKBACK: u32 = 60;
const HISTORY_TICKS: u32 = LOOKBACK + MAX_STAB_LAG + EVAL_DELAY + 20;
/// Knife swing events closer than this belong to one swing
const SWING_MERGE: u32 = 10;
/// Swings, readiness transitions and attack events kept per spy
const LOG_TICKS: u32 = 400;
const MIN_SINCE_SPAWN: u32 = 30;
const MIN_SINCE_TELEPORT: u32 = 16;
/// The spy's sample at the attack tick may be at most this far from it
const MAX_ATTACK_SAMPLE_GAP: u32 = 2;
/// A victim sample older than this (relative to the wanted tick) does not count as a record
const MAX_RECORD_GAP: u32 = 4;
/// Floor for step sizes in ratios (about half the angle quantization)
const MIN_STEP: f32 = 0.18;
const FL_DUCKING: u32 = 1 << 1;
/// Networked pitch is quantized in ~0.7 degree steps, so a level view reads as up to 0.35
const LEVEL_PITCH: f32 = 0.36;
/// Victim lags searched for the aim point the spy's view matches best
const MAX_AIM_LAG: u32 = 30;
const ANIM_ATTACK_PRIMARY: i64 = 0;
const ANIM_CUSTOM_GESTURE: i64 = 20;

// CTFKnife swing trace: 48 HU forward ray swept with an 18 HU half-size hull, with slack for quantized and
// interpolated positions. CTFKnife::IsBehindAndFacingTarget dot tolerance.
const KNIFE_RANGE: f32 = 48.0;
const KNIFE_HULL: f32 = 18.0 + 4.0;
const DOT_TOLERANCE: f32 = 0.05;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Per-stab features, in the order of `STAB_FEATURES`
pub const STAB_FEATURES: [&str; 35] = [
    "yaw_err", "pitch_err", "pitch_abs", "yaw_err_m1", "yaw_err_m4", "yaw_err_p2", "step_atk", "pre_med",
    "stillness", "post_step", "post_ratio", "valid_run", "valid_frac", "range_run", "react_ready", "had_ready",
    "lag_excess", "range_excess", "impossible", "behind_dot", "facing_dot", "view_dot", "spy_speed",
    "victim_speed", "move_cos", "rel_lat", "dist2d", "swings_before", "since_prev_swing", "yaw_err_best",
    "pitch_err_best", "best_lag", "step_m1", "turn_8", "ping_ticks",
];
/// Stab features that are aggregated into the spy features (all but the context `ping_ticks`)
const AGGREGATED: usize = 34;
const IDX_YAW_ERR: usize = 0;
const IDX_PITCH_ERR: usize = 1;
const IDX_STEP_ATK: usize = 6;
const IDX_STILLNESS: usize = 8;
const IDX_POST_STEP: usize = 9;
const IDX_VALID_RUN: usize = 11;
const IDX_RANGE_RUN: usize = 13;
const IDX_REACT_READY: usize = 14;
const IDX_HAD_READY: usize = 15;
const IDX_LAG_EXCESS: usize = 16;
const IDX_IMPOSSIBLE: usize = 18;
const IDX_DIST2D: usize = 26;
const IDX_YAW_ERR_BEST: usize = 29;
const IDX_PITCH_ABS: usize = 2;

/// Spy feature names: `<stab feature>_med`, `_p10`, `_p90` for the aggregated stab features, then the
/// shares and per-demo knife counts
pub fn unit_feature_names() -> Vec<String> {
    let mut names = Vec::new();
    for suffix in ["med", "p10", "p90"] {
        names.extend(STAB_FEATURES[..AGGREGATED].iter().map(|f| format!("{f}_{suffix}")));
    }
    for n in UNIT_EXTRA {
        names.push(n.to_string());
    }
    names
}

const UNIT_EXTRA: [&str; 10] = [
    "instant_share", "aim_share", "snap_share", "bt_share", "unready_share", "stabs_per_swing", "backstab_share",
    "log_stabs", "level_share", "best_aim_share",
];

#[derive(Clone, Copy)]
struct Sample {
    tick: u32,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
}

impl Sample {
    fn eye(&self) -> [f32; 3] {
        [self.pos[0], self.pos[1], self.pos[2] + if self.ducking { 45.0 } else { 75.0 }]
    }
}

struct PendingStab {
    tick: u32,
    spy: u64,
    victim: u64,
    ping_ticks: f32,
}

/// Per spy knife state of the demo
#[derive(Default)]
struct SpyLog {
    /// Primary attack animation events
    attacks: VecDeque<u32>,
    /// Knife swings (merged attack events while the knife is out)
    swings: VecDeque<u32>,
    /// `m_bReadyToBackstab` transitions (tick, new value)
    ready: VecDeque<(u32, bool)>,
    ready_now: bool,
    n_swings: usize,
    n_knife_hits: usize,
    n_backstabs: usize,
}

#[derive(Clone, Debug)]
pub struct StabRecord {
    pub tick: u32,
    pub attack_tick: u32,
    pub spy: u64,
    pub victim: u64,
    pub features: [f32; STAB_FEATURES.len()],
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_stabs: usize,
    pub n_swings: usize,
    pub n_knife_hits: usize,
    pub features: Vec<f32>,
    pub score: f32,
}

/// One LightGBM tree, flattened. Child index >= 0 is a node, < 0 is leaf `!index`.
#[derive(Deserialize)]
struct Tree {
    feature: Vec<u16>,
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    default_left: Vec<bool>,
    /// 0 none (NaN -> 0), 1 zero, 2 NaN
    missing: Vec<u8>,
    leaf: Vec<f64>,
}

impl Tree {
    fn predict(&self, x: &[f32]) -> f64 {
        if self.feature.is_empty() {
            return self.leaf.first().copied().unwrap_or(0.0);
        }
        let mut node = 0i32;
        while node >= 0 {
            let n = node as usize;
            let mut v = f64::from(x[usize::from(self.feature[n])]);
            let missing = self.missing[n];
            let go_default = match missing {
                2 => v.is_nan(),
                1 => v.is_nan() || v.abs() <= 1e-35,
                _ => false,
            };
            if missing != 2 && v.is_nan() {
                v = 0.0;
            }
            let left = if go_default { self.default_left[n] } else { v <= self.threshold[n] };
            node = if left { self.left[n] } else { self.right[n] };
        }
        self.leaf[!node as usize]
    }
}

#[derive(Deserialize)]
struct Booster {
    features: Vec<String>,
    trees: Vec<Tree>,
}

impl Booster {
    fn probability(&self, x: &[f32]) -> f32 {
        let raw: f64 = self.trees.iter().map(|t| t.predict(x)).sum();
        (1.0 / (1.0 + (-raw).exp())) as f32
    }
}

#[derive(Deserialize)]
struct Model {
    unit: Booster,
    stab: Booster,
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_autobackstab_model.json")).ok()?;
            let stab_names: Vec<String> = STAB_FEATURES.iter().map(|s| s.to_string()).collect();
            (model.unit.features == unit_feature_names() && model.stab.features == stab_names).then_some(model)
        })
        .as_ref()
}

fn player_sid(player: &Player) -> Option<u64> {
    let info = player.info.as_ref()?;
    if info.steam_id == "BOT" {
        return None;
    }
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

fn is_knife(state: &CheatAnalyserState, player: &Player) -> bool {
    player
        .active_weapon
        .and_then(|w| state.weapons.get(&w))
        .is_some_and(|w| w.class_name.contains("Knife"))
}

type V3 = [f64; 3];

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Angle between two vectors in degrees, precise for tiny angles
fn angle_between(a: V3, b: V3) -> f64 {
    let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    norm(c).atan2(d).to_degrees()
}

fn view_dir(yaw: f32, pitch: f32) -> V3 {
    let (y, p) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
    [p.cos() * y.cos(), p.cos() * y.sin(), -p.sin()]
}

fn wrap180(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

/// Linear interpolation quantile of the non-NaN values (pandas' default), NaN when there are none
fn quantile(values: &mut Vec<f64>, q: f64) -> f64 {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * (pos - lo as f64)
}

fn median(values: &[f64]) -> f64 {
    quantile(&mut values.to_vec(), 0.5)
}

/// Slab test of the segment `p0 + t * d` (t in [0, 1]) against an AABB
fn segment_hits_box(p0: [f32; 3], d: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for i in 0..3 {
        if d[i].abs() < 1e-9 {
            if p0[i] < bmin[i] || p0[i] > bmax[i] {
                return false;
            }
        } else {
            let (mut a, mut b) = ((bmin[i] - p0[i]) / d[i], (bmax[i] - p0[i]) / d[i]);
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

/// Whether the knife swing trace of `spy` reaches the victim record `v` (hull sweep = ray against the
/// grown box)
fn knife_reaches(spy: &Sample, v: &Sample) -> bool {
    let eye = spy.eye();
    let f = view_dir(spy.yaw, spy.pitch);
    let d = [f[0] as f32 * KNIFE_RANGE, f[1] as f32 * KNIFE_RANGE, f[2] as f32 * KNIFE_RANGE];
    let top = if v.ducking { 62.0 } else { 82.0 };
    let bmin = [v.pos[0] - 24.0 - KNIFE_HULL, v.pos[1] - 24.0 - KNIFE_HULL, v.pos[2] - KNIFE_HULL];
    let bmax = [v.pos[0] + 24.0 + KNIFE_HULL, v.pos[1] + 24.0 + KNIFE_HULL, v.pos[2] + top + KNIFE_HULL];
    segment_hits_box(eye, d, bmin, bmax)
}

/// CTFKnife::IsBehindAndFacingTarget dot products (behind, facing, view); the game requires
/// behind > 0, facing > 0.5 and view > -0.3
fn backstab_dots(spy: &Sample, v: &Sample, victim_yaw: f32) -> (f32, f32, f32) {
    let (tx, ty) = (v.pos[0] - spy.pos[0], v.pos[1] - spy.pos[1]);
    let n = (tx * tx + ty * ty).sqrt().max(1e-6);
    let (tx, ty) = (tx / n, ty / n);
    let (vy, sy) = (victim_yaw.to_radians(), spy.yaw.to_radians());
    let (vfx, vfy, sfx, sfy) = (vy.cos(), vy.sin(), sy.cos(), sy.sin());
    (tx * vfx + ty * vfy, tx * sfx + ty * sfy, vfx * sfx + vfy * sfy)
}

fn dots_ok((behind, facing, view): (f32, f32, f32)) -> bool {
    behind > -DOT_TOLERANCE && facing > 0.5 - DOT_TOLERANCE && view > -0.3 - DOT_TOLERANCE
}

/// Latest sample at or before `tick`, if it is at most `MAX_RECORD_GAP` older
fn record_at(samples: &[Sample], tick: i64) -> Option<&Sample> {
    let i = samples.partition_point(|s| i64::from(s.tick) <= tick).checked_sub(1)?;
    (tick - i64::from(samples[i].tick) <= i64::from(MAX_RECORD_GAP)).then(|| &samples[i])
}

/// Yaw / pitch error of the spy's view to the melee aimbot aim point of a victim record: the victim's origin
/// with the height clamped to the spy's eye into the victim's hull
fn aim_error(spy: &Sample, v: &Sample) -> (f32, f32) {
    let eye = spy.eye();
    let top = if v.ducking { 62.0 } else { 82.0 };
    let tz = eye[2].clamp(v.pos[2], v.pos[2] + top);
    let (dx, dy, dz) = (v.pos[0] - eye[0], v.pos[1] - eye[1], tz - eye[2]);
    let yaw = dy.atan2(dx).to_degrees();
    let pitch = -dz.atan2((dx * dx + dy * dy).sqrt()).to_degrees();
    (wrap180(spy.yaw - yaw).abs(), (spy.pitch - pitch).abs())
}

/// Context of one stab handed to `stab_features`
struct StabContext<'a> {
    spy: &'a [Sample],
    victim: &'a [Sample],
    attack_tick: u32,
    hurt_tick: u32,
    ping_ticks: f32,
    /// Knife readiness transitions of the spy
    ready: &'a VecDeque<(u32, bool)>,
    swings: &'a VecDeque<u32>,
}

fn horizontal_velocity(s: &[Sample], i: usize) -> Option<[f32; 2]> {
    let j = (0..i).rev().find(|&j| s[i].tick - s[j].tick >= 4)?;
    let dt = (s[i].tick - s[j].tick) as f32 / 66.67;
    (s[i].tick - s[j].tick <= 12).then(|| [(s[i].pos[0] - s[j].pos[0]) / dt, (s[i].pos[1] - s[j].pos[1]) / dt])
}

/// Features of one backstab, or `None` when the samples around it are unusable
fn stab_features(c: &StabContext) -> Option<[f32; STAB_FEATURES.len()]> {
    let s = c.spy;
    let ta = c.attack_tick;
    let h = s.partition_point(|x| x.tick <= ta).checked_sub(1)?;
    if ta - s[h].tick > MAX_ATTACK_SAMPLE_GAP || h < 4 {
        return None;
    }
    let victim_yaw_at = |t: u32| record_at(c.victim, i64::from(t)).map(|v| v.yaw);
    let vyaw = victim_yaw_at(ta)?;

    // lag compensation: smallest victim record lag at which the stab is possible (with / without the angle
    // conditions), from the spy's samples around the attack tick
    let spy_att: Vec<&Sample> = [h, h.saturating_sub(1), (h + 1).min(s.len() - 1)]
        .iter()
        .map(|&i| &s[i])
        .filter(|x| x.tick + 1 >= ta && x.tick <= ta + 1)
        .collect();
    let victim_yaws: Vec<f32> =
        [ta, ta.saturating_sub(1), ta + 1, c.hurt_tick].iter().filter_map(|&t| victim_yaw_at(t)).collect();
    let mut valid_lag = None;
    let mut range_lag = None;
    for v in c.victim.iter().rev().filter(|v| v.tick <= ta && ta - v.tick <= MAX_STAB_LAG) {
        let lag = ta - v.tick;
        for sp in &spy_att {
            if !knife_reaches(sp, v) {
                continue;
            }
            range_lag.get_or_insert(lag);
            if victim_yaws.iter().any(|&y| dots_ok(backstab_dots(sp, v, y))) {
                valid_lag.get_or_insert(lag);
            }
        }
        if valid_lag.is_some() {
            break;
        }
    }
    let lag = valid_lag.or(range_lag).unwrap_or(c.ping_ticks.round().max(0.0) as u32);
    let rec = |t: u32| record_at(c.victim, i64::from(t) - i64::from(lag));
    let v0 = rec(s[h].tick)?;

    let (yaw_err, pitch_err) = aim_error(&s[h], v0);
    let err_at = |i: usize| rec(s[i].tick).map_or(f32::NAN, |v| aim_error(&s[i], v).0);
    let back = |ticks: u32| (0..h).rev().find(|&i| s[i].tick + ticks <= s[h].tick);
    let yaw_err_m1 = err_at(h - 1);
    let yaw_err_m4 = back(4).map_or(f32::NAN, err_at);
    let yaw_err_p2 = (h + 1..s.len()).find(|&i| s[i].tick >= s[h].tick + 2).map_or(f32::NAN, err_at);

    // aim point the view matches best over the victim's recent records (the record a backtracking aimbot
    // aimed at is unknown)
    let (mut yaw_err_best, mut pitch_err_best, mut best_lag) = (f32::NAN, f32::NAN, f32::NAN);
    for v in c.victim.iter().filter(|v| v.tick <= s[h].tick && s[h].tick - v.tick <= MAX_AIM_LAG) {
        let (ye, pe) = aim_error(&s[h], v);
        if !(ye >= yaw_err_best) {
            (yaw_err_best, pitch_err_best, best_lag) = (ye, pe, (s[h].tick - v.tick) as f32);
        }
    }

    // view motion
    let dirs: Vec<V3> = s.iter().map(|x| view_dir(x.yaw, x.pitch)).collect();
    let rate = |i: usize| (angle_between(dirs[i + 1], dirs[i]) / f64::from((s[i + 1].tick - s[i].tick).max(1))) as f32;
    let step_atk = rate(h - 1);
    let pre: Vec<f64> = (0..h - 1).filter(|&i| s[i].tick + 12 >= s[h].tick).map(|i| f64::from(rate(i))).collect();
    let pre_med = if pre.is_empty() { f32::NAN } else { median(&pre) as f32 };
    let post: Vec<f32> = (h..(h + 3).min(s.len() - 1)).map(rate).collect();
    let post_step = post.iter().copied().fold(f32::NAN, f32::max);
    let stillness = step_atk / pre_med.max(MIN_STEP);
    let step_m1 = rate(h - 2);
    let turn_8 = back(8).map_or(f32::NAN, |i| angle_between(dirs[h], dirs[i]) as f32);
    let post_ratio = post_step / step_atk.max(MIN_STEP);

    // how long the stab had been possible (at the same victim lag) and in range before the attack
    let valid_at = |i: usize, dots: bool| {
        let Some(v) = rec(s[i].tick) else { return false };
        knife_reaches(&s[i], v) && (!dots || victim_yaw_at(s[i].tick).is_some_and(|y| dots_ok(backstab_dots(&s[i], v, y))))
    };
    let run = |dots: bool| {
        if !valid_at(h, dots) {
            return f32::NAN;
        }
        let mut j = h;
        while j > 0 && s[j - 1].tick + LOOKBACK >= ta && valid_at(j - 1, dots) {
            j -= 1;
        }
        (ta - s[j].tick) as f32
    };
    let valid_run = run(true);
    let range_run = run(false);
    let window: Vec<usize> = (0..h).filter(|&i| s[i].tick + 30 >= ta).collect();
    let valid_frac = if window.is_empty() {
        f32::NAN
    } else {
        window.iter().filter(|&&i| valid_at(i, true)).count() as f32 / window.len() as f32
    };

    // knife readiness: ticks since m_bReadyToBackstab rose (when raised at the attack)
    let ready_at = |t: u32| c.ready.iter().rev().find(|&&(rt, _)| rt <= t).is_some_and(|&(_, r)| r);
    let had_ready = c.ready.iter().any(|&(t, r)| r && t + 60 >= c.hurt_tick && t <= c.hurt_tick)
        || ready_at(c.hurt_tick.saturating_sub(60));
    // the swing lowers the knife, so the readiness may already read false on the attack tick
    let react_ready = c.ready.iter().rev().find(|&&(t, r)| r && t <= ta).map_or(f32::NAN, |&(rise, _)| {
        let fall = c.ready.iter().find(|&&(t, r)| !r && t > rise).map(|&(t, _)| t);
        if fall.is_none_or(|f| f + 2 >= ta) { (ta - rise) as f32 } else { f32::NAN }
    });

    let excess = |l: Option<u32>| l.map_or(f32::NAN, |l| l as f32 - c.ping_ticks);
    let (behind, facing, view) = backstab_dots(&s[h], v0, vyaw);

    // movement
    let vi = c.victim.iter().position(|x| std::ptr::eq(x, v0))?;
    let spy_v = horizontal_velocity(s, h);
    let vic_v = horizontal_velocity(c.victim, vi);
    let speed = |v: Option<[f32; 2]>| v.map_or(f32::NAN, |v| v[0].hypot(v[1]));
    let (tx, ty) = (v0.pos[0] - s[h].pos[0], v0.pos[1] - s[h].pos[1]);
    let dist2d = tx.hypot(ty);
    let (ux, uy) = (tx / dist2d.max(1e-3), ty / dist2d.max(1e-3));
    let move_cos = spy_v.map_or(f32::NAN, |v| (v[0] * ux + v[1] * uy) / v[0].hypot(v[1]).max(1.0));
    let rel_lat = match (spy_v, vic_v) {
        (Some(a), Some(b)) => ((a[0] - b[0]) * -uy + (a[1] - b[1]) * ux).abs(),
        _ => f32::NAN,
    };

    let swings_before = c.swings.iter().filter(|&&t| t + 132 >= ta && t + 6 < ta).count() as f32;
    let since_prev_swing = c.swings.iter().rev().find(|&&t| t + 6 < ta).map_or(400.0, |&t| (ta - t).min(400) as f32);

    let impossible = (valid_lag.is_none() && range_lag.is_none()) as u8 as f32;
    Some([
        yaw_err, pitch_err, s[h].pitch.abs(), yaw_err_m1, yaw_err_m4, yaw_err_p2, step_atk, pre_med, stillness,
        post_step, post_ratio, valid_run, valid_frac, range_run, react_ready, had_ready as u8 as f32,
        excess(valid_lag), excess(range_lag), impossible, behind, facing, view, speed(spy_v), speed(vic_v),
        move_cos, rel_lat, dist2d, swings_before, since_prev_swing, yaw_err_best, pitch_err_best, best_lag, step_m1,
        turn_8, c.ping_ticks,
    ])
}

/// Spy features from the spy's stab features (see `unit_feature_names`)
fn unit_features(stabs: &[&StabRecord], log: &SpyLog) -> Vec<f32> {
    let mut out = Vec::with_capacity(AGGREGATED * 3 + UNIT_EXTRA.len());
    for q in [0.5, 0.1, 0.9] {
        for f in 0..AGGREGATED {
            let mut col: Vec<f64> = stabs.iter().map(|s| f64::from(s.features[f])).collect();
            out.push(quantile(&mut col, q) as f32);
        }
    }
    let n = stabs.len() as f32;
    let share = |pred: &dyn Fn(&[f32; STAB_FEATURES.len()]) -> bool| {
        stabs.iter().filter(|s| pred(&s.features)).count() as f32 / n
    };
    out.push(share(&|f| f[IDX_VALID_RUN] <= 1.0));
    out.push(share(&|f| f[IDX_YAW_ERR] <= 1.0 && f[IDX_PITCH_ERR] <= 1.0));
    out.push(share(&|f| f[IDX_STILLNESS] >= 6.0 && f[IDX_STEP_ATK] >= 5.0));
    out.push(share(&|f| f[IDX_LAG_EXCESS] > 14.0 || f[IDX_IMPOSSIBLE] > 0.5));
    out.push(share(&|f| f[IDX_HAD_READY] < 0.5));
    out.push(log.n_backstabs as f32 / log.n_swings.max(log.n_backstabs).max(1) as f32);
    out.push(log.n_backstabs as f32 / log.n_knife_hits.max(log.n_backstabs).max(1) as f32);
    out.push((1.0 + n).ln());
    out.push(share(&|f| f[IDX_PITCH_ABS] <= LEVEL_PITCH));
    out.push(share(&|f| f[IDX_YAW_ERR_BEST] <= 1.0));
    out
}

pub struct MlAutoBackstab {
    params: Parameters,
    jg: JankGuard,
    history: FxHashMap<u64, VecDeque<Sample>>,
    last_sim: FxHashMap<u64, i64>,
    knife_ready: FxHashMap<EntityId, bool>,
    logs: FxHashMap<u64, SpyLog>,
    pending: Vec<PendingStab>,
    recorder: Option<u64>,
    /// Usable backstabs of the demo, in evaluation order
    pub stabs: Vec<StabRecord>,
    /// Spies with enough stabs, filled by `finish`
    pub units: Vec<UnitRecord>,
}

impl Default for MlAutoBackstab {
    fn default() -> Self {
        Self::new()
    }
}

impl MlAutoBackstab {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("min_score".to_string(), Parameter::Float(0.7)),
                ("min_stabs".to_string(), Parameter::Int(3)),
                ("max_detections".to_string(), Parameter::Int(3)),
            ]),
            jg: JankGuard::default(),
            history: FxHashMap::default(),
            last_sim: FxHashMap::default(),
            knife_ready: FxHashMap::default(),
            logs: FxHashMap::default(),
            pending: Vec::new(),
            recorder: None,
            stabs: Vec::new(),
            units: Vec::new(),
        }
    }

    /// Evaluates the pending stabs whose window has passed (all of them when `all`)
    fn flush(&mut self, now: u32, all: bool) {
        let mut i = 0;
        while i < self.pending.len() {
            if !all && self.pending[i].tick + EVAL_DELAY > now {
                i += 1;
                continue;
            }
            let p = self.pending.swap_remove(i);
            let (Some(a), Some(v), Some(log)) = (self.history.get(&p.spy), self.history.get(&p.victim), self.logs.get(&p.spy))
            else {
                continue;
            };
            let attack_tick = attack_tick(&log.attacks, p.tick);
            let lo = attack_tick.saturating_sub(LOOKBACK + 4);
            let spy: Vec<Sample> = a.iter().filter(|s| s.tick >= lo && s.tick <= p.tick + EVAL_DELAY).copied().collect();
            let vic: Vec<Sample> = v
                .iter()
                .filter(|s| s.tick + MAX_STAB_LAG + LOOKBACK + 4 >= attack_tick && s.tick <= p.tick + EVAL_DELAY)
                .copied()
                .collect();
            let ctx = StabContext {
                spy: &spy,
                victim: &vic,
                attack_tick,
                hurt_tick: p.tick,
                ping_ticks: p.ping_ticks,
                ready: &log.ready,
                swings: &log.swings,
            };
            if let Some(features) = stab_features(&ctx) {
                self.stabs.push(StabRecord { tick: p.tick, attack_tick, spy: p.spy, victim: p.victim, features, score: f32::NAN });
            }
        }
    }

    fn on_anim_events(&mut self, message: &Message, state: &CheatAnalyserState, parser_state: &ParserState, tick: u32) {
        const ANIM_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_hPlayer");
        const ANIM_EVENT: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_iEvent");
        let Message::TempEntities(msg) = message else { return };
        // events are delta encoded against the previous event of the same class
        let mut prev_class = None;
        let (mut handle, mut event_id) = (0u32, 0i64);
        for event in &msg.events {
            if prev_class != Some(event.class_id) {
                (handle, event_id) = (0, 0);
                prev_class = Some(event.class_id);
            }
            if parser_state.server_classes.get(usize::from(event.class_id)).map(|c| c.name.as_str()) != Some("CTEPlayerAnimEvent") {
                continue;
            }
            for prop in &event.props {
                if prop.identifier == ANIM_PLAYER {
                    if let Some(h) = i64::try_from(&prop.value).ok().and_then(|h| u32::try_from(h).ok()) {
                        handle = h;
                    }
                } else if prop.identifier == ANIM_EVENT {
                    if let Ok(e) = i64::try_from(&prop.value) {
                        event_id = e;
                    }
                }
            }
            // PLAYERANIMEVENT_ATTACK_PRIMARY; a backstab sends PLAYERANIMEVENT_CUSTOM_GESTURE instead
            // (CTFKnife::SendPlayerAnimEvent)
            if event_id != ANIM_ATTACK_PRIMARY && event_id != ANIM_CUSTOM_GESTURE {
                continue;
            }
            let entity = handle_to_entid(handle);
            let Some(player) = state.players.iter().find(|p| p.entity == entity) else { continue };
            if player.class != Class::Spy {
                continue;
            }
            let Some(sid) = player_sid(player) else { continue };
            let knife = is_knife(state, player);
            if event_id == ANIM_CUSTOM_GESTURE && !knife {
                continue;
            }
            let log = self.logs.entry(sid).or_default();
            if log.attacks.back() != Some(&tick) {
                log.attacks.push_back(tick);
            }
            if knife && log.swings.back().is_none_or(|&t| t + SWING_MERGE <= tick) {
                log.swings.push_back(tick);
                log.n_swings += 1;
            }
        }
    }
}

/// Start of the final cluster of attack events at or before the hurt tick (the hurt event usually arrives
/// 1-3 ticks after the swing)
fn attack_tick(attacks: &VecDeque<u32>, hurt_tick: u32) -> u32 {
    let mut atk: Vec<u32> = attacks.iter().copied().filter(|&t| t + 6 >= hurt_tick && t <= hurt_tick).collect();
    atk.sort_unstable();
    atk.dedup();
    let Some(&last) = atk.last() else { return hurt_tick };
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

impl<'a> CheatAlgorithm<'a> for MlAutoBackstab {
    fn default(&self) -> bool {
        false
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/ml_autobackstab"
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    /// The scores aggregate all backstabs of the demo
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        let mut m = self.jg.handled_messages()?;
        for t in [MessageType::GameEvent, MessageType::TempEntities, MessageType::PacketEntities] {
            if !m.contains(&t) {
                m.push(t);
            }
        }
        Ok(m)
    }

    fn init(&mut self) -> Result<(), Error> {
        let params = std::mem::take(&mut self.params);
        *self = Self::new();
        self.params = params;
        Ok(())
    }

    fn on_message(&mut self, message: &Message, state: &CheatAnalyserState, parser_state: &ParserState, tick: DemoTick) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);
        let tick = u32::from(tick);
        match message {
            Message::PacketEntities(msg) => {
                for entity in &msg.entities {
                    if entity.update_type == UpdateType::Delete {
                        self.knife_ready.remove(&entity.entity_index);
                        continue;
                    }
                    let is_knife_class = parser_state
                        .server_classes
                        .get(usize::from(entity.server_class))
                        .is_some_and(|c| c.name.as_str().contains("Knife"));
                    if !is_knife_class {
                        continue;
                    }
                    for prop in entity.props(parser_state) {
                        if prop.identifier.names().is_some_and(|(_, n)| n.as_str() == "m_bReadyToBackstab") {
                            if let Ok(v) = i64::try_from(&prop.value) {
                                self.knife_ready.insert(entity.entity_index, v != 0);
                            }
                        }
                    }
                }
            }
            Message::TempEntities(_) => self.on_anim_events(message, state, parser_state, tick),
            Message::GameEvent(GameEventMessage { event: GameEvent::PlayerHurt(h), .. }) => {
                let knife_hit = h.weapon_id == 7 || h.custom == 2;
                if !knife_hit || h.damage_amount == 0 {
                    return Ok(vec![]);
                }
                let (Some(a), Some(v)) = (state.get_id64_from_userid(h.attacker.into()), state.get_id64_from_userid(h.user_id.into())) else {
                    return Ok(vec![]);
                };
                if a == v || a == 0 || Some(a) == self.recorder {
                    return Ok(vec![]);
                }
                let Some(ap) = state.players.iter().find(|p| player_sid(p) == Some(a) && p.state == PlayerState::Alive) else {
                    return Ok(vec![]);
                };
                let Some(vp) = state.players.iter().find(|p| player_sid(p) == Some(v)) else { return Ok(vec![]) };
                if ap.class != Class::Spy || ap.team == vp.team || !ap.in_pvs || ap.is_taunting() {
                    return Ok(vec![]);
                }
                let log = self.logs.entry(a).or_default();
                log.n_knife_hits += 1;
                if h.custom != 2 {
                    return Ok(vec![]);
                }
                log.n_backstabs += 1;
                if self.jg.spawned(&a, tick) < MIN_SINCE_SPAWN
                    || self.jg.teleported(&a, tick) < MIN_SINCE_TELEPORT
                    || self.jg.teleported(&v, tick) < MIN_SINCE_TELEPORT
                {
                    return Ok(vec![]);
                }
                self.pending.push(PendingStab { tick, spy: a, victim: v, ping_ticks: f32::from(ap.ping) / 15.0 });
            }
            _ => {}
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _parser_state: &ParserState) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        if self.recorder.is_none() {
            self.recorder = resolve_recorder_sid(state);
        }
        let tick = u32::from(state.tick);
        for p in &state.players {
            if !p.in_pvs || p.state != PlayerState::Alive {
                continue;
            }
            let Some(sid) = player_sid(p) else { continue };
            if p.class == Class::Spy {
                if let Some(&ready) = p.active_weapon.and_then(|w| self.knife_ready.get(&w)) {
                    let log = self.logs.entry(sid).or_default();
                    if ready != log.ready_now {
                        log.ready_now = ready;
                        log.ready.push_back((tick, ready));
                    }
                }
            }
            let sim = state.sim_tick(p);
            if self.last_sim.insert(sid, sim) == Some(sim) {
                continue;
            }
            let ducking = p.flags & FL_DUCKING != 0;
            let h = self.history.entry(sid).or_default();
            h.push_back(Sample { tick, pos: [p.position.x, p.position.y, p.position.z], yaw: p.view_angle, pitch: p.pitch_angle, ducking });
            while h.front().is_some_and(|s| s.tick + HISTORY_TICKS < tick) {
                h.pop_front();
            }
        }
        if tick % 64 == 0 {
            for log in self.logs.values_mut() {
                while log.attacks.front().is_some_and(|&t| t + LOG_TICKS < tick) {
                    log.attacks.pop_front();
                }
                while log.swings.front().is_some_and(|&t| t + LOG_TICKS < tick) {
                    log.swings.pop_front();
                }
                // keep the latest transition so the current state stays known
                while log.ready.len() > 1 && log.ready[1].0 + LOG_TICKS < tick {
                    log.ready.pop_front();
                }
            }
        }
        if !self.pending.is_empty() {
            self.flush(tick, false);
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        self.flush(u32::MAX, true);
        let min_score = get_parameter_value::<f32>(&self.params, "min_score");
        let min_stabs = get_parameter_value::<i32>(&self.params, "min_stabs").max(1) as usize;
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let model = model();
        if let Some(m) = model {
            for s in &mut self.stabs {
                s.score = m.stab.probability(&s.features);
            }
        }
        let mut spies: Vec<u64> = self.stabs.iter().map(|s| s.spy).collect();
        spies.sort_unstable();
        spies.dedup();
        let mut detections = Vec::new();
        let empty = SpyLog::default();
        for spy in spies {
            let stabs: Vec<&StabRecord> = self.stabs.iter().filter(|s| s.spy == spy).collect();
            if stabs.len() < min_stabs {
                continue;
            }
            let log = self.logs.get(&spy).unwrap_or(&empty);
            let features = unit_features(&stabs, log);
            let score = model.map_or(f32::NAN, |m| m.unit.probability(&features));
            self.units.push(UnitRecord {
                player: spy,
                n_stabs: stabs.len(),
                n_swings: log.n_swings,
                n_knife_hits: log.n_knife_hits,
                features,
                score,
            });
            if !(score >= min_score) {
                continue;
            }
            let mut top = stabs.clone();
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            for s in top.into_iter().take(max_detections) {
                let f = &s.features;
                let opt = |v: f32| if v.is_nan() { None } else { Some(v) };
                detections.push(Detection {
                    tick: s.tick,
                    algorithm: "fidoo/ml_autobackstab".to_string(),
                    player: spy,
                    data: json!({
                        "score": score,
                        "stab_score": s.score,
                        "stabs": stabs.len(),
                        "swings": log.n_swings,
                        "target": s.victim,
                        "attack_tick": s.attack_tick,
                        "yaw_error_deg": f[IDX_YAW_ERR],
                        "pitch_error_deg": f[IDX_PITCH_ERR],
                        "snap_deg_per_tick": f[IDX_STEP_ATK],
                        "return_deg_per_tick": opt(f[IDX_POST_STEP]),
                        "possible_for_ticks": opt(f[IDX_VALID_RUN]),
                        "in_range_for_ticks": opt(f[IDX_RANGE_RUN]),
                        "knife_ready_ticks": opt(f[IDX_REACT_READY]),
                        "lag_excess_ticks": opt(f[IDX_LAG_EXCESS]),
                        "distance": f[IDX_DIST2D].round(),
                    }),
                });
            }
        }
        detections.sort_by_key(|d| (d.tick, d.player));
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tick: u32, x: f32, yaw: f32, pitch: f32) -> Sample {
        Sample { tick, pos: [x, 0.0, 0.0], yaw, pitch, ducking: false }
    }

    #[test]
    fn aim_error_zero_at_origin_level() {
        let spy = sample(0, -50.0, 0.0, 0.0);
        let v = sample(0, 0.0, 0.0, 0.0);
        let (y, p) = aim_error(&spy, &v);
        assert!(y < 1e-4 && p < 1e-4);
        assert!(knife_reaches(&spy, &v));
        assert!(dots_ok(backstab_dots(&spy, &v, 0.0)));
        assert!(!dots_ok(backstab_dots(&spy, &v, 180.0)));
    }

    #[test]
    fn instant_stab_has_zero_valid_run() {
        // spy walks up behind a static victim; the view turns onto the back only on the attack tick
        let spy: Vec<Sample> = (0..=70u32).map(|t| sample(t, -50.0, if t < 70 { 90.0 } else { 0.0 }, 0.0)).collect();
        let victim: Vec<Sample> = (0..=80u32).map(|t| sample(t, 0.0, 0.0, 0.0)).collect();
        let ready = VecDeque::new();
        let swings = VecDeque::new();
        let c = StabContext { spy: &spy, victim: &victim, attack_tick: 70, hurt_tick: 71, ping_ticks: 2.0, ready: &ready, swings: &swings };
        let f = stab_features(&c).unwrap();
        assert_eq!(f[IDX_VALID_RUN], 0.0);
        assert!(f[IDX_YAW_ERR] < 1e-3);
        assert!(f[IDX_STEP_ATK] > 50.0);
        assert_eq!(f[IDX_HAD_READY], 0.0);
        assert_eq!(f.len(), STAB_FEATURES.len());
    }

    #[test]
    fn unit_features_match_names() {
        let s = StabRecord { tick: 0, attack_tick: 0, spy: 1, victim: 2, features: [0.0; STAB_FEATURES.len()], score: 0.0 };
        assert_eq!(unit_features(&[&s], &SpyLog::default()).len(), unit_feature_names().len());
    }
}
