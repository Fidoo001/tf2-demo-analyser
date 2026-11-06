//! Learned sniper triggerbot detection. See docs/ml_triggerbot.md.
//!
//! A triggerbot user aims by hand and lets the cheat fire when the head is under the crosshair: the
//! crosshair is swept or zigzagged slowly over the head, often repeatedly, and the shot lands while it is
//! still moving. A scoped sniper's `CSniperDot` sits at the end of the server-side aim trace with 1/32 unit
//! precision, so `dot - eye` is the exact aim of every scoped tick. For every scoped rifle shot (hit or
//! miss) the crosshair's path over the 100 ticks before the shot is measured against the target's head at
//! the sniper's lag compensation offset (distance, acquisition, speed into the shot) and its shape over
//! the scoped stretch leading into the shot (head crossings, direction reversals, smoothness, hovering,
//! holding height, snaps). The shots of a sniper are aggregated over the demo into player features and
//! scored by a logistic regression (`assets/ml_triggerbot_model.json`) trained on the confirmed triggerbot
//! cheaters of the reference demos and reference-like reported snipers no aimbot algorithm flags, against
//! unreported and aimbot-flagged snipers. Snipers at or above `min_score` are reported at their highest
//! scoring hits (a per-shot LightGBM model).
//!
//! `examples/dump_ml_triggerbot_features.rs` dumps the same features for training.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::packetentities::UpdateType;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::analyser::{Class, Team};
use tf_demo_parser::demo::sendprop::{SendPropIdentifier, SendPropValue};
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;

/// Shots are evaluated this many ticks after being fired, so the hurt event has arrived
const POST_TICKS: u32 = 8;
/// Ticks before the shot whose aim is kept
const AIM_WINDOW: i32 = 100;
/// Largest lag compensation offset tried (interp + latency, in ticks)
const MAX_K: i32 = 19;
/// Offset used when a sniper has too few hits to estimate it (100 ms interp + ~50 ms ping)
const DEFAULT_K: i32 = 10;
const MIN_HITS_FOR_K: usize = 4;
/// Ticks of target positions kept per shot
const TRACK_WINDOW: u32 = AIM_WINDOW as u32 + MAX_K as u32 + 2;
const HISTORY_TICKS: usize = 160;
const HEAD_RADIUS: f32 = 6.0;
/// Per-hit errors are clamped to this many head radii when estimating `k`
const K_ERROR_CLAMP: f32 = 10.0;
/// Candidate targets must be within this angle of the (coarse) eye angles at the shot
const TARGET_FOV: f32 = 15.0;
/// Max gap (ticks) bridged when interpolating a target's position
const MAX_INTERP_GAP: i32 = 4;
/// Ticks before the shot used for the crosshair path features
const PATH_WINDOW: i32 = 60;
/// Sniper rifle refire time (1.5 s) and the no-headshot time after zooming in (0.2 s), in ticks
const REFIRE_TICKS: i64 = 100;
const HEADSHOT_UNLOCK_TICKS: i64 = 14;
/// Caps for "long ago" timings
const MAX_SINCE_PREV: f32 = 300.0;
const MAX_ZOOM_DELAY: u32 = 66;
const MAX_READY_LATENCY: f32 = 66.0;
/// A miss within this many head radii of a head is a near miss
const NEAR_MISS_RADIUS: f32 = 4.0;
const TF_COND_ZOOMED: u32 = 1 << 1;

const DOT_ORIGIN: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_vecOrigin");
const DOT_OWNER: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_hOwnerEntity");
const BULLETS_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
const BULLETS_WEAPON: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iWeaponID");

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Per-shot features, in the order of `SHOT_FEATURES`. Distances are in head radii, `a*` are distances
/// from the final aim point (relative to the target, so head model errors cancel), speeds per tick.
pub const SHOT_FEATURES: [&str; 59] = [
    "d0", "d1", "d2", "d4", "d8", "a1", "a2", "a4", "a8", "acq1", "acq2", "acq4", "v0", "v1", "v2", "decel",
    "ang_speed0", "approach_cos", "min_d_pre", "dwell_frac", "tgt_speed", "k_dev", "since_prev", "zoom_delay",
    "ready_latency", "ready_hold", "tgt_dir_cos", "n_near", "shooter_speed", "scope_len", "near_frac", "path_near",
    "cross", "reversals", "turn_mean", "max_step", "med_step", "smooth", "speed_cv", "y_std_near", "x_std_near",
    "v_shot", "moving_shot", "slide", "lat_shot", "near_time", "revisit", "revisit_ready", "pass_close", "entry",
    "on_before", "d_prev", "slow_slide", "hit", "headshot", "dist", "lag", "cover", "gap0",
];
/// Path features (`scope_len` .. `slow_slide`), see `ShotRecord::path_features`
const PATH_FEATURES: usize = 24;
const IDX_PATH: usize = 29;
const IDX_A1: usize = 5;
const IDX_DECEL: usize = 15;
const IDX_SCOPE_LEN: usize = 29;
const IDX_NEAR_FRAC: usize = 30;
const IDX_PATH_NEAR: usize = 31;
const IDX_CROSS: usize = 32;
const IDX_REVERSALS: usize = 33;
const IDX_TURN_MEAN: usize = 34;
const IDX_MAX_STEP: usize = 35;
const IDX_MED_STEP: usize = 36;
const IDX_SMOOTH: usize = 37;
const IDX_SPEED_CV: usize = 38;
const IDX_Y_STD: usize = 39;
const IDX_V_SHOT: usize = 41;
const IDX_NEAR_TIME: usize = 45;
const IDX_REVISIT: usize = 46;
const IDX_ON_BEFORE: usize = 50;
const IDX_D_PREV: usize = 51;
const IDX_SLOW_SLIDE: usize = 52;
/// Shot features that are aggregated into the player features (all but the trailing context)
const AGGREGATED: usize = 53;
const IDX_D0: usize = 0;
const IDX_ACQ2: usize = 10;
const IDX_ZOOM_DELAY: usize = 23;
#[cfg(test)]
const IDX_READY_LATENCY: usize = 24;
const IDX_READY_HOLD: usize = 25;
const IDX_HIT: usize = 53;
const IDX_HEADSHOT: usize = 54;
const IDX_DIST: usize = 55;
const IDX_LAG: usize = 56;
/// Player features, see `unit_features`
pub const UNIT_FEATURES: [&str; 29] = [
    "zig_share", "snap_share", "moving_share", "long_share", "hover_share", "slide_share", "reversals_mean",
    "cross_mean", "revisit_mean", "near_frac_med", "path_near_med", "smooth_med", "speed_cv_med", "turn_mean_med",
    "y_std_med", "med_step_med", "v_shot_med", "scope_med", "near_time_med", "headshot_rate", "hit_d0_med",
    "hit_on_before_med", "hit_rate", "conversion", "unlock_share", "hit_a1_med", "hit_decel_med", "flick_hit_share",
    "quick_share",
];

const IDX_UNIT_SCOPE_MED: usize = 17;
/// Snipers whose median scoped stretch before a shot is shorter than this (ticks) are not scored
const MIN_SCOPE_MED: f32 = 8.0;

pub fn unit_feature_names() -> Vec<String> {
    UNIT_FEATURES.iter().map(|s| s.to_string()).collect()
}

#[derive(Clone, Copy, Debug)]
struct Snap {
    tick: u32,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
    alive: bool,
    in_pvs: bool,
    team: Team,
    class: Class,
    cond: u32,
}

#[derive(Clone, Copy, Debug)]
struct Dot {
    owner: u32,
    pos: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Shot {
    tick: u32,
    ent: u32,
    weapon_id: i64,
    /// Previous rifle shot of the same shooter
    prev_tick: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
struct Hurt {
    tick: u32,
    attacker_uid: u16,
    victim_uid: u16,
    custom: u16,
}

#[derive(Clone, Copy, Debug)]
struct AimSample {
    j: i32,
    eye: [f32; 3],
    yaw: f32,
    pitch: f32,
}

/// Head positions of a candidate target, by tick offset from the shot
#[derive(Clone, Debug)]
struct TargetTrack {
    sid: u64,
    head: Vec<(i32, [f32; 3])>,
}

impl TargetTrack {
    /// Head position at a tick offset, linearly interpolated over short gaps
    fn head_at(&self, j: i32) -> Option<[f32; 3]> {
        let idx = self.head.partition_point(|(t, _)| *t < j);
        let (t1, p1) = *self.head.get(idx)?;
        if t1 == j {
            return Some(p1);
        }
        let (t0, p0) = *self.head.get(idx.checked_sub(1)?)?;
        if t1 - t0 > MAX_INTERP_GAP {
            return None;
        }
        let a = (j - t0) as f32 / (t1 - t0) as f32;
        Some([p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a, p0[2] + (p1[2] - p0[2]) * a])
    }
}

/// Offset of the crosshair from a target's head, in head radii
#[derive(Clone, Copy, Debug)]
struct Rel {
    x: f32,
    y: f32,
    dist: f32,
}

impl Rel {
    fn len(&self) -> f32 {
        self.x.hypot(self.y)
    }

    fn sub(&self, o: &Rel) -> f32 {
        (self.x - o.x).hypot(self.y - o.y)
    }
}

#[derive(Clone, Debug)]
struct ShotRecord {
    tick: u32,
    weapon: String,
    headshot: bool,
    hit: bool,
    /// Index into `targets` of the hurt player, when they were trackable
    victim: Option<usize>,
    /// Ticks since zooming in, when the zoom started within the kept history
    zoom_delay: Option<u32>,
    /// Ticks since zooming in is at least this (the whole kept history was zoomed)
    zoom_long: bool,
    prev_tick: Option<u32>,
    shooter_speed: f32,
    /// Exact aim from the sniper dot, ordered by tick offset
    aim: Vec<AimSample>,
    targets: Vec<TargetTrack>,
}

impl ShotRecord {
    fn aim_at(&self, j: i32) -> Option<&AimSample> {
        let idx = self.aim.partition_point(|a| a.j < j);
        self.aim.get(idx).filter(|a| a.j == j)
    }

    /// Crosshair offset at tick offset `j` from where `target` was `k` ticks earlier
    fn rel(&self, j: i32, target: &TargetTrack, k: i32) -> Option<Rel> {
        let aim = self.aim_at(j)?;
        let head = target.head_at(j - k)?;
        let (hy, hp) = dir_angles(aim.eye, head);
        let d = dist(aim.eye, head);
        let r = (HEAD_RADIUS / d.max(1.0)).atan().to_degrees();
        Some(Rel { x: wrap180(aim.yaw - hy) * hp.to_radians().cos() / r, y: (aim.pitch - hp) / r, dist: d })
    }

    /// The hurt player for hits, else the head closest to the crosshair at the shot
    fn target(&self, k: i32) -> Option<&TargetTrack> {
        if self.hit {
            return self.targets.get(self.victim?);
        }
        self.targets
            .iter()
            .filter_map(|t| Some((self.rel(0, t, k)?.len(), t)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, t)| t)
    }

    /// First tick from which the rifle could fire a headshot
    fn ready_offset(&self) -> Option<i64> {
        let refire = self.prev_tick.map_or(i64::MIN, |p| i64::from(p) + REFIRE_TICKS - i64::from(self.tick));
        let unlock = match self.zoom_delay {
            Some(z) => HEADSHOT_UNLOCK_TICKS - i64::from(z),
            None if self.zoom_long => i64::MIN,
            None => return None,
        };
        Some(refire.max(unlock))
    }

    /// Shot features at the sniper's lag offset `k`, `None` without aim or target at the shot.
    /// Dot samples are often only every second tick (and missing out of PVS), so the features walk the
    /// available samples instead of fixed tick offsets.
    fn features(&self, k: i32) -> Option<[f32; SHOT_FEATURES.len()]> {
        let t = self.target(k)?;
        let r0 = self.rel(0, t, k)?;
        let nan = f32::NAN;
        // crosshair offsets before the shot, latest first
        let before: Vec<(i32, Rel)> =
            self.aim.iter().rev().filter(|a| a.j < 0).filter_map(|a| Some((a.j, self.rel(a.j, t, k)?))).collect();
        // latest sample at most 2 ticks before `j`
        let at = |j: i32| before.iter().find(|(jj, _)| *jj <= j && *jj >= j - 2).map(|x| x.1);
        let d = |j: i32| at(j).map_or(nan, |r| r.len());
        let a = |j: i32| at(j).map_or(nan, |r| r.sub(&r0));
        // ticks since the crosshair was further than `radius` from the final aim point
        let acq = |radius: f32| {
            before.iter().find(|(_, r)| r.sub(&r0) > radius).map_or((AIM_WINDOW + 1) as f32, |(j, _)| -j as f32)
        };
        // speed between consecutive samples (per tick), from the shot backwards
        let seq: Vec<(i32, Rel)> = std::iter::once((0, r0)).chain(before.iter().copied()).collect();
        let v = |i: usize| match (seq.get(i), seq.get(i + 1)) {
            (Some(c), Some(p)) if c.0 - p.0 <= MAX_INTERP_GAP => c.1.sub(&p.1) / (c.0 - p.0) as f32,
            _ => nan,
        };
        let (v0, v1, v2, v3) = (v(0), v(1), v(2), v(3));
        let prior: Vec<f32> = [v1, v2, v3].into_iter().filter(|x| !x.is_nan()).collect();
        let decel = if prior.is_empty() || v0.is_nan() {
            nan
        } else {
            v0 / (prior.iter().sum::<f32>() / prior.len() as f32).max(0.05)
        };
        let prev = before.first().filter(|(j, _)| -j <= MAX_INTERP_GAP).copied();
        let gap0 = prev.map_or(nan, |(j, _)| -j as f32);
        let ang_speed0 = match (prev.and_then(|(j, _)| self.aim_at(j)), self.aim_at(0)) {
            (Some(p), Some(c)) => {
                (wrap180(c.yaw - p.yaw) * c.pitch.to_radians().cos()).hypot(c.pitch - p.pitch) / (c.j - p.j) as f32
            }
            _ => nan,
        };
        // motion into the shot against the direction to the head centre
        let approach_cos = match prev {
            Some((_, p)) if p.len() > 1e-3 && r0.sub(&p) > 1e-3 => {
                let (mx, my) = (r0.x - p.x, r0.y - p.y);
                (-(mx * p.x + my * p.y)) / (mx.hypot(my) * p.len())
            }
            _ => nan,
        };
        let pre: Vec<f32> = before.iter().filter(|(j, _)| *j >= -15).map(|(_, r)| r.len()).collect();
        let min_d_pre = if pre.is_empty() { nan } else { pre.iter().copied().fold(f32::INFINITY, f32::min) };
        let window: Vec<f32> = before.iter().filter(|(j, _)| *j >= -30).map(|(_, r)| r.len()).collect();
        let dwell_frac = if window.is_empty() {
            nan
        } else {
            window.iter().filter(|&&x| x < 1.5).count() as f32 / window.len() as f32
        };
        // target's angular motion at the shot, in head radii per tick
        let head_dir = |j: i32| -> Option<(f32, f32)> {
            let aim = self.aim_at(j)?;
            let h = t.head_at(j - k)?;
            let (hy, hp) = dir_angles(aim.eye, h);
            let r = (HEAD_RADIUS / dist(aim.eye, h).max(1.0)).atan().to_degrees();
            Some((hy / r, hp / r))
        };
        let tgt_speed = match (prev.and_then(|(j, _)| Some((j, head_dir(j)?))), head_dir(0)) {
            (Some((j, p)), Some(c)) => (wrap180(c.0 - p.0) * c.1.to_radians().cos()).hypot(c.1 - p.1) / -j as f32,
            _ => nan,
        };
        // crosshair motion vs target motion over the last ~4 ticks
        let s4 = before.iter().filter(|(j, _)| (-6..=-3).contains(j)).min_by_key(|(j, _)| (j + 4).abs()).map(|x| x.0);
        let tgt_dir_cos = match (s4.and_then(|j| self.aim_at(j)), self.aim_at(0)) {
            (Some(a0), Some(a1)) => match (t.head_at(a0.j - k), t.head_at(-k)) {
                (Some(h0), Some(h1)) => {
                    let (y0, p0) = dir_angles(a0.eye, h0);
                    let (y1, p1) = dir_angles(a1.eye, h1);
                    let c = p1.to_radians().cos();
                    let tm = (wrap180(y1 - y0) * c, p1 - p0);
                    let am = (wrap180(a1.yaw - a0.yaw) * c, a1.pitch - a0.pitch);
                    let (tl, al) = (tm.0.hypot(tm.1), am.0.hypot(am.1));
                    if tl < 0.02 || al < 0.02 {
                        nan
                    } else {
                        (tm.0 * am.0 + tm.1 * am.1) / (tl * al)
                    }
                }
                _ => nan,
            },
            _ => nan,
        };
        // per-shot lag offset fit, compared with the sniper's
        let k_dev = if self.hit {
            (0..=MAX_K)
                .filter_map(|kk| Some((self.rel(0, t, kk)?.len(), kk)))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map_or(nan, |(_, kk)| (kk - k) as f32)
        } else {
            nan
        };
        let since_prev = self.prev_tick.map_or(MAX_SINCE_PREV, |p| ((self.tick - p) as f32).min(MAX_SINCE_PREV));
        let zoom_delay = match self.zoom_delay {
            Some(z) if z <= MAX_ZOOM_DELAY => z as f32,
            _ => nan,
        };
        let ready = self.ready_offset();
        let ready_latency = ready.map_or(nan, |r| (-(r as f32)).clamp(-MAX_READY_LATENCY, MAX_READY_LATENCY));
        // ticks the crosshair already sat on the final aim point while the rifle was ready
        let ready_hold = match ready {
            Some(r) => {
                let (mut hold, mut last) = (0, 0);
                for &(j, rel) in &before {
                    if i64::from(j) < r || last - j > 3 || rel.sub(&r0) >= 1.0 {
                        break;
                    }
                    (hold, last) = (-j, j);
                }
                hold as f32
            }
            None => nan,
        };
        let n_near = self.targets.iter().filter_map(|x| self.rel(0, x, k)).filter(|r| r.len() < NEAR_MISS_RADIUS).count();
        let cover = self.aim.iter().filter(|a| a.j >= -30).count() as f32 / 31.0;
        let path = self.path_features(t, k, (!ready_latency.is_nan()).then_some(-ready_latency));
        let mut out = [nan; SHOT_FEATURES.len()];
        out[..IDX_PATH].copy_from_slice(&[
            r0.len(), d(-1), d(-2), d(-4), d(-8), a(-1), a(-2), a(-4), a(-8), acq(1.0), acq(2.0), acq(4.0), v0, v1,
            v2, decel, ang_speed0, approach_cos, min_d_pre, dwell_frac, tgt_speed, k_dev, since_prev, zoom_delay,
            ready_latency, ready_hold, tgt_dir_cos, n_near as f32, self.shooter_speed,
        ]);
        out[IDX_PATH..AGGREGATED].copy_from_slice(&path);
        out[AGGREGATED..].copy_from_slice(&[
            f32::from(u8::from(self.hit)), f32::from(u8::from(self.headshot)), r0.dist, k as f32, cover, gap0,
        ]);
        Some(out)
    }

    /// Shape of the crosshair path relative to the target's head (head radii) over the scoped stretch leading
    /// into the shot (dot samples at most 4 ticks apart, last `PATH_WINDOW` ticks): how long the sniper was
    /// scoped, how much the crosshair wandered, crossed and reversed around the head, how smooth it moved, and
    /// how it was moving when the shot fired. A triggerbot user sweeps the crosshair over the head and lets
    /// the cheat fire. `ready` is the tick offset from which the rifle could fire a headshot.
    fn path_features(&self, t: &TargetTrack, k: i32, ready: Option<f32>) -> [f32; PATH_FEATURES] {
        let nan = f32::NAN;
        let mut out = [nan; PATH_FEATURES];
        let mut p: Vec<(f32, f32, f32)> =
            self.aim.iter().filter_map(|a| self.rel(a.j, t, k).map(|r| (a.j as f32, r.x, r.y))).collect();
        if p.last().map_or(true, |x| x.0 != 0.0) {
            return out;
        }
        let mut s = p.len() - 1;
        while s > 0 && p[s].0 - p[s - 1].0 <= MAX_INTERP_GAP as f32 {
            s -= 1;
        }
        out[0] = -p[s].0;
        let from = p[s].0.max(-PATH_WINDOW as f32);
        p.retain(|x| x.0 >= from);
        let n = p.len();
        if n < 4 {
            return out;
        }
        let d: Vec<f32> = p.iter().map(|x| x.1.hypot(x.2)).collect();
        let (mut vx, mut vy, mut sp, mut dt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for w in p.windows(2) {
            let g = w[1].0 - w[0].0;
            vx.push((w[1].1 - w[0].1) / g);
            vy.push((w[1].2 - w[0].2) / g);
            sp.push(((w[1].1 - w[0].1) / g).hypot((w[1].2 - w[0].2) / g));
            dt.push(g);
        }
        let m = sp.len();
        let near_frac = d.iter().filter(|&&x| x < 3.0).count() as f32 / n as f32;
        let path_near: f32 = (0..m).filter(|&i| d[i + 1] < 6.0).map(|i| sp[i] * dt[i]).sum();
        let sign = |v: f32| if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 };
        // crossings of the head: x changes sign at head height, y changes sign in the head column
        let cross = (0..m)
            .map(|i| {
                let (a, b) = (p[i], p[i + 1]);
                u32::from(sign(b.1) != sign(a.1) && b.2.abs() < 2.0 && a.2.abs() < 2.0)
                    + u32::from(sign(b.2) != sign(a.2) && b.1.abs() < 2.0 && a.1.abs() < 2.0)
            })
            .sum::<u32>();
        // direction reversals while moving near the head
        let (mut reversals, mut last, mut turns) = (0, None::<usize>, Vec::new());
        for i in 0..m {
            if sp[i] <= 0.08 || d[i + 1] > 6.0 {
                continue;
            }
            if let Some(l) = last {
                let c = (vx[i] * vx[l] + vy[i] * vy[l]) / (sp[i] * sp[l]);
                turns.push(c);
                if c < -0.5 {
                    reversals += 1;
                }
            }
            last = Some(i);
        }
        let turn_mean = if turns.is_empty() { nan } else { turns.iter().sum::<f32>() / turns.len() as f32 };
        let max_step = sp.iter().copied().fold(f32::MIN, f32::max);
        let med_step = quantile(&mut sp.iter().map(|&x| f64::from(x)).collect(), 0.5) as f32;
        let mean_step = sp.iter().sum::<f32>() / m as f32;
        let std_step = (sp.iter().map(|x| (x - mean_step).powi(2)).sum::<f32>() / m as f32).sqrt();
        let std = |v: &[f32]| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let near: Vec<usize> = (0..n).filter(|&i| d[i] < 6.0).collect();
        let (y_std, x_std) = if near.len() >= 3 {
            (std(&near.iter().map(|&i| p[i].2).collect::<Vec<_>>()), std(&near.iter().map(|&i| p[i].1).collect::<Vec<_>>()))
        } else {
            (nan, nan)
        };
        let v_shot = sp[m - 1];
        let moving_shot = f32::from(u8::from(v_shot > 0.1 && d[n - 1] < 2.0));
        let (mut slide, mut lat_shot) = (nan, nan);
        if m >= 3 {
            let cos = |a: usize, b: usize| (vx[a] * vx[b] + vy[a] * vy[b]) / (sp[a] * sp[b]).max(1e-9);
            let c = cos(m - 1, m - 2).min(cos(m - 1, m - 3));
            let min3 = sp[m - 3..].iter().copied().fold(f32::MAX, f32::min);
            slide = f32::from(u8::from(c > 0.7 && v_shot > 0.05 && min3 > 0.03));
            let (px, py) = (p[n - 2].1, p[n - 2].2);
            let pl = px.hypot(py);
            if pl > 1e-3 && v_shot > 1e-3 {
                lat_shot = (vx[m - 1] * py - vy[m - 1] * px).abs() / (pl * v_shot);
            }
        }
        // ticks since the crosshair came (and stayed) within 3 radii
        let mut i = n - 1;
        while i > 0 && d[i - 1] < 3.0 {
            i -= 1;
        }
        let near_time = -p[i].0;
        // earlier visits of the final aim point, separated from the shot by leaving it
        let rel: Vec<f32> = p.iter().map(|x| (x.1 - p[n - 1].1).hypot(x.2 - p[n - 1].2)).collect();
        let is_ready = |j: f32| ready.map_or(false, |r| j >= r);
        let (mut left, mut revisit, mut revisit_ready) = (false, 0, 0);
        for t in (0..n - 1).rev() {
            if rel[t] > 1.5 {
                left = true;
            } else if left && rel[t] < 0.5 {
                revisit += 1;
                if is_ready(p[t].0) {
                    revisit_ready += 1;
                }
            }
        }
        // closest pass over the head while ready, before the final approach
        let mut lo = n - 1;
        while lo > 0 && rel[lo - 1] <= 1.5 {
            lo -= 1;
        }
        let pass_close = (0..lo)
            .filter(|&t| ready.is_none() || is_ready(p[t].0))
            .map(|t| d[t])
            .fold(f32::INFINITY, f32::min);
        let pass_close = if pass_close.is_finite() { pass_close } else { nan };
        // offset from the head centre along the motion at the shot (negative = still on the entry side), ticks
        // already on the head before the shot
        let entry = if v_shot > 1e-3 { (p[n - 1].1 * vx[m - 1] + p[n - 1].2 * vy[m - 1]) / v_shot } else { nan };
        let mut i = n as isize - 2;
        while i >= 0 && d[i as usize] < 1.2 {
            i -= 1;
        }
        let on_before = if ((i + 1) as usize) < n - 1 { -p[(i + 1) as usize].0 } else { 0.0 };
        let last4 = sp[m.saturating_sub(4)..].iter().copied().fold(f32::MIN, f32::max);
        let slow_slide = f32::from(u8::from((0.08..=1.0).contains(&v_shot) && last4 < 1.5 && d[n - 1] < 2.0));
        out[1..].copy_from_slice(&[
            near_frac, path_near, cross as f32, reversals as f32, turn_mean, max_step, med_step,
            max_step / med_step.max(0.05), std_step / mean_step.max(0.05), y_std, x_std, v_shot, moving_shot, slide,
            lat_shot, near_time, revisit as f32, revisit_ready as f32, pass_close, entry, on_before, d[n - 2],
            slow_slide,
        ]);
        out
    }
}

#[derive(Clone, Debug)]
pub struct ShotFeatures {
    pub tick: u32,
    pub shooter: u64,
    pub victim: u64,
    pub weapon: String,
    pub features: [f32; SHOT_FEATURES.len()],
    pub score: f32,
    /// Crosshair offset from the target's head (head radii) per dot sample, for inspection
    pub path: Vec<(i32, f32, f32)>,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_hits: usize,
    pub n_shots: usize,
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

/// Logistic regression on standardised features. Player features over few shots are noisy (a lucky match
/// looks extreme), so each is first shrunk toward the legit population value `prior` with the weight
/// `n / (n + shrink_shots)` of the sniper's `n` shots; unknown values are replaced by `fill`.
#[derive(Deserialize)]
struct Linear {
    features: Vec<String>,
    prior: Vec<f64>,
    shrink_shots: f64,
    fill: Vec<f64>,
    mean: Vec<f64>,
    scale: Vec<f64>,
    coef: Vec<f64>,
    intercept: f64,
}

impl Linear {
    fn probability(&self, x: &[f32], shots: usize) -> f32 {
        let w = shots as f64 / (shots as f64 + self.shrink_shots);
        let raw: f64 = self.intercept
            + x.iter()
                .enumerate()
                .map(|(i, &v)| {
                    let v = if v.is_nan() { self.fill[i] } else { w * f64::from(v) + (1.0 - w) * self.prior[i] };
                    (v - self.mean[i]) / self.scale[i] * self.coef[i]
                })
                .sum::<f64>();
        (1.0 / (1.0 + (-raw).exp())) as f32
    }
}

#[derive(Deserialize)]
struct Model {
    unit: Linear,
    shot: Booster,
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_triggerbot_model.json")).ok()?;
            let shot_names: Vec<String> = SHOT_FEATURES.iter().map(|s| s.to_string()).collect();
            let n = UNIT_FEATURES.len();
            let unit_ok = model.unit.features == unit_feature_names()
                && [model.unit.prior.len(), model.unit.fill.len(), model.unit.mean.len(), model.unit.scale.len(), model.unit.coef.len()]
                    == [n; 5];
            (unit_ok && model.shot.features == shot_names).then_some(model)
        })
        .as_ref()
}

fn player_sid(player: &Player, state: &CheatAnalyserState) -> Option<u64> {
    let info = player.info.as_ref()?;
    if info.steam_id == "BOT" {
        return None;
    }
    if let Some(sid) = state.get_id64_from_userid(info.user_id) {
        if sid != 0 {
            return Some(sid);
        }
    }
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

/// Head hitbox centre above the origin, calibrated on confirmed sniper headshots
fn head_z(class: Class, ducking: bool) -> f32 {
    if ducking {
        return 48.0;
    }
    match class {
        Class::Scout | Class::Soldier | Class::Demoman => 68.0,
        Class::Pyro => 67.0,
        Class::Heavy => 76.0,
        Class::Engineer => 66.0,
        Class::Medic => 73.0,
        Class::Sniper => 74.0,
        Class::Spy | Class::Other => 72.0,
    }
}

fn sniper_eye(snap: &Snap) -> [f32; 3] {
    [snap.pos[0], snap.pos[1], snap.pos[2] + if snap.ducking { 45.0 } else { 75.0 }]
}

fn wrap180(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

/// (yaw, pitch) of the direction from `from` to `to`, Source conventions (pitch positive = down)
fn dir_angles(from: [f32; 3], to: [f32; 3]) -> (f32, f32) {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    (dy.atan2(dx).to_degrees(), -(dz.atan2((dx * dx + dy * dy).sqrt())).to_degrees())
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn snap_at(hist: &VecDeque<Snap>, tick: u32) -> Option<&Snap> {
    let idx = hist.partition_point(|s| s.tick < tick);
    hist.get(idx).filter(|s| s.tick == tick)
}

/// TF_WEAPON_SNIPERRIFLE, TF_WEAPON_SNIPERRIFLE_DECAP, TF_WEAPON_SNIPERRIFLE_CLASSIC (from the bullet,
/// the active weapon may already have changed when the shot is evaluated)
fn is_rifle(weapon_id: i64) -> bool {
    matches!(weapon_id, 17 | 77 | 99)
}

/// Weapon name from the bullet's weapon id, when the active weapon is no longer known
fn rifle_name(weapon_id: i64) -> &'static str {
    match weapon_id {
        77 => "Bazaar Bargain",
        99 => "Classic",
        _ => "Sniper Rifle",
    }
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

/// Share of the non-NaN values of column `f` satisfying `pred`, NaN when there are none
fn share(rows: &[&ShotFeatures], f: usize, pred: impl Fn(f32) -> bool) -> f32 {
    let known: Vec<f32> = rows.iter().map(|r| r.features[f]).filter(|v| !v.is_nan()).collect();
    if known.is_empty() {
        return f32::NAN;
    }
    known.iter().filter(|&&v| pred(v)).count() as f32 / known.len() as f32
}

/// Lag compensation offset minimizing the median crosshair-to-head error of the sniper's hits
fn estimate_k(hits: &[&ShotRecord]) -> i32 {
    if hits.len() < MIN_HITS_FOR_K {
        return DEFAULT_K;
    }
    let mut best: Option<(f32, i32)> = None;
    for k in 0..=MAX_K {
        let mut errors: Vec<f32> = hits
            .iter()
            .filter_map(|s| s.rel(0, s.targets.get(s.victim?)?, k))
            .map(|r| r.len().min(K_ERROR_CLAMP))
            .collect();
        if errors.len() < 3 {
            continue;
        }
        errors.sort_by(|a, b| a.total_cmp(b));
        let median = errors[errors.len() / 2];
        if best.map_or(true, |(m, _)| median < m) {
            best = Some((median, k));
        }
    }
    best.map_or(DEFAULT_K, |(_, k)| k)
}

/// Player features from the sniper's shot features (see `UNIT_FEATURES`): how often the crosshair zigzags,
/// snaps, hovers or slides into shots, typical path shape and timing, the outcome of the hits, and how often
/// hits arrive in one big step or right as the scope comes up (aimbot snaps, which a triggerbot does not need)
fn unit_features(shots: &[&ShotFeatures]) -> Vec<f32> {
    let hits: Vec<&ShotFeatures> = shots.iter().copied().filter(|s| s.features[IDX_HIT] > 0.5).collect();
    let misses: Vec<&ShotFeatures> = shots.iter().copied().filter(|s| s.features[IDX_HIT] < 0.5).collect();
    let n = shots.len().max(1) as f32;
    // share of all shots satisfying `pred` (unknown values count as not satisfying it)
    let all = |pred: &dyn Fn(&[f32; SHOT_FEATURES.len()]) -> bool| shots.iter().filter(|s| pred(&s.features)).count() as f32 / n;
    let col = |rows: &[&ShotFeatures], f: usize| {
        let mut c: Vec<f64> = rows.iter().map(|r| f64::from(r.features[f])).collect();
        quantile(&mut c, 0.5) as f32
    };
    let mean = |f: usize| {
        let known: Vec<f32> = shots.iter().map(|s| s.features[f]).filter(|v| !v.is_nan()).collect();
        if known.is_empty() { f32::NAN } else { known.iter().sum::<f32>() / known.len() as f32 }
    };
    let near_misses = misses.iter().filter(|s| s.features[IDX_D0] < NEAR_MISS_RADIUS).count();
    vec![
        all(&|f| f[IDX_REVERSALS] >= 2.0 || f[IDX_CROSS] >= 2.0),
        all(&|f| f[IDX_MAX_STEP] >= 2.5),
        all(&|f| f[IDX_V_SHOT] > 0.1),
        all(&|f| f[IDX_SCOPE_LEN] >= 40.0),
        all(&|f| f[IDX_NEAR_TIME] >= 10.0),
        mean(IDX_SLOW_SLIDE),
        mean(IDX_REVERSALS),
        mean(IDX_CROSS),
        mean(IDX_REVISIT),
        col(shots, IDX_NEAR_FRAC),
        col(shots, IDX_PATH_NEAR),
        col(shots, IDX_SMOOTH),
        col(shots, IDX_SPEED_CV),
        col(shots, IDX_TURN_MEAN),
        col(shots, IDX_Y_STD),
        col(shots, IDX_MED_STEP),
        col(shots, IDX_V_SHOT),
        col(shots, IDX_SCOPE_LEN),
        col(shots, IDX_NEAR_TIME),
        share(&hits, IDX_HEADSHOT, |v| v > 0.5),
        col(&hits, IDX_D0),
        col(&hits, IDX_ON_BEFORE),
        hits.len() as f32 / n,
        hits.len() as f32 / (hits.len() + near_misses).max(1) as f32,
        share(shots, IDX_ZOOM_DELAY, |v| (14.0..=16.0).contains(&v)),
        col(&hits, IDX_A1),
        col(&hits, IDX_DECEL),
        hits.iter().filter(|s| s.features[IDX_D_PREV] > 4.0).count() as f32 / hits.len().max(1) as f32,
        all(&|f| f[IDX_SCOPE_LEN] <= 3.0),
    ]
}

#[derive(Default)]
struct ShooterData {
    shots: Vec<ShotRecord>,
}

pub struct MlTriggerbot {
    params: Parameters,
    history: FxHashMap<u32, VecDeque<Snap>>,
    dots: FxHashMap<u32, Dot>,
    dot_history: FxHashMap<u32, VecDeque<(u32, [f32; 3])>>,
    pending: Vec<Shot>,
    last_shot: FxHashMap<u32, u32>,
    hurts: VecDeque<Hurt>,
    shooters: FxHashMap<u64, ShooterData>,
    /// Shot features of the demo, filled by `finish`
    pub shots: Vec<ShotFeatures>,
    /// Snipers with enough hits, filled by `finish`
    pub units: Vec<UnitRecord>,
}

impl Default for MlTriggerbot {
    fn default() -> Self {
        Self::new()
    }
}

impl MlTriggerbot {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("min_score".to_string(), Parameter::Float(0.33)),
                ("min_shots".to_string(), Parameter::Int(8)),
                ("max_detections".to_string(), Parameter::Int(3)),
            ]),
            history: FxHashMap::default(),
            dots: FxHashMap::default(),
            dot_history: FxHashMap::default(),
            pending: Vec::new(),
            last_shot: FxHashMap::default(),
            hurts: VecDeque::new(),
            shooters: FxHashMap::default(),
            shots: Vec::new(),
            units: Vec::new(),
        }
    }

    fn record_shot(&mut self, shot: Shot, state: &CheatAnalyserState) {
        let Some(shooter_hist) = self.history.get(&shot.ent) else { return };
        let Some(s0) = snap_at(shooter_hist, shot.tick).copied() else { return };
        if s0.class != Class::Sniper || !is_rifle(shot.weapon_id) {
            return;
        }
        let Some(shooter) = state.players.iter().find(|p| u32::from(p.entity) == shot.ent) else { return };
        let Some(sid) = player_sid(shooter, state) else { return };

        let shooter_uid = shooter.info.as_ref().map_or(0, |i| u16::from(i.user_id));
        let hurt = self
            .hurts
            .iter()
            .find(|h| h.attacker_uid == shooter_uid && h.victim_uid != shooter_uid && h.tick >= shot.tick && h.tick <= shot.tick + 2)
            .copied();
        let victim_ent = hurt.and_then(|h| {
            state
                .players
                .iter()
                .find(|p| p.info.as_ref().is_some_and(|i| u16::from(i.user_id) == h.victim_uid))
                .map(|p| u32::from(p.entity))
        });

        let dot_hist = self.dot_history.get(&shot.ent);
        let mut aim = Vec::new();
        for j in -AIM_WINDOW..=0 {
            let tick = (i64::from(shot.tick) + i64::from(j)) as u32;
            let Some(snap) = snap_at(shooter_hist, tick) else { continue };
            let Some(&(_, dot)) = dot_hist.and_then(|h| h.iter().find(|(t, _)| *t == tick)) else { continue };
            let eye = sniper_eye(snap);
            if dist(eye, dot) < 8.0 {
                continue;
            }
            let (yaw, pitch) = dir_angles(eye, dot);
            aim.push(AimSample { j, eye, yaw, pitch });
        }
        if aim.last().map_or(true, |a| a.j != 0) {
            return;
        }

        let eye0 = sniper_eye(&s0);
        let mut targets = Vec::new();
        let mut victim = None;
        for (&ent, hist) in &self.history {
            if ent == shot.ent {
                continue;
            }
            let Some(t0) = snap_at(hist, shot.tick) else { continue };
            let is_victim = Some(ent) == victim_ent;
            if t0.team == s0.team || !t0.in_pvs || (!t0.alive && !is_victim) {
                continue;
            }
            let head0 = [t0.pos[0], t0.pos[1], t0.pos[2] + head_z(t0.class, t0.ducking)];
            let (hy, hp) = dir_angles(eye0, head0);
            let off = (wrap180(s0.yaw - hy) * hp.to_radians().cos()).hypot(s0.pitch - hp);
            if off > TARGET_FOV && !is_victim {
                continue;
            }
            let sid = state
                .players
                .iter()
                .find(|p| u32::from(p.entity) == ent)
                .and_then(|p| player_sid(p, state))
                .unwrap_or(0);
            let head = hist
                .iter()
                .filter(|s| s.tick + TRACK_WINDOW >= shot.tick && s.tick <= shot.tick && s.in_pvs)
                .map(|s| (s.tick as i32 - shot.tick as i32, [s.pos[0], s.pos[1], s.pos[2] + head_z(s.class, s.ducking)]))
                .collect();
            if is_victim {
                victim = Some(targets.len());
            }
            targets.push(TargetTrack { sid, head });
        }

        // start of the current zoom
        let (mut zoom_delay, mut zoom_long) = (None, false);
        if s0.cond & TF_COND_ZOOMED != 0 {
            let mut start = None;
            let mut ended = false;
            for s in shooter_hist.iter().rev().filter(|s| s.tick <= shot.tick) {
                if s.cond & TF_COND_ZOOMED == 0 {
                    ended = true;
                    break;
                }
                start = Some(shot.tick - s.tick);
            }
            if ended {
                zoom_delay = start;
            } else {
                zoom_long = start.is_some_and(|z| z > MAX_ZOOM_DELAY);
            }
        }
        let shooter_speed = snap_at(shooter_hist, shot.tick.saturating_sub(1))
            .map_or(f32::NAN, |p| dist(p.pos, s0.pos));

        self.shooters.entry(sid).or_default().shots.push(ShotRecord {
            tick: shot.tick,
            weapon: match state.get_player_weapon(shooter) {
                w if w.to_lowercase().contains("unknown") => rifle_name(shot.weapon_id).to_string(),
                w => w,
            },
            headshot: hurt.is_some_and(|h| h.custom == 1),
            hit: hurt.is_some(),
            victim,
            zoom_delay,
            zoom_long,
            prev_tick: shot.prev_tick,
            shooter_speed,
            aim,
            targets,
        });
    }
}

impl<'a> CheatAlgorithm<'a> for MlTriggerbot {
    fn default(&self) -> bool {
        false
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/ml_triggerbot"
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    /// The scores aggregate all shots of the demo
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::TempEntities, MessageType::GameEvent, MessageType::PacketEntities])
    }

    fn init(&mut self) -> Result<(), Error> {
        let params = std::mem::take(&mut self.params);
        *self = Self::new();
        self.params = params;
        Ok(())
    }

    fn on_message(&mut self, message: &Message, _state: &CheatAnalyserState, parser_state: &ParserState, tick: DemoTick) -> Result<Vec<Detection>, Error> {
        let ticknum: u32 = tick.into();
        match message {
            Message::TempEntities(msg) => {
                for event in &msg.events {
                    let class = &parser_state.server_classes[usize::from(event.class_id)].name;
                    if class.as_str() != "CTEFireBullets" {
                        continue;
                    }
                    let prop = |id: SendPropIdentifier| {
                        event.props.iter().find(|p| p.identifier == id).and_then(|p| i64::try_from(&p.value).ok())
                    };
                    if let (Some(player), Some(weapon_id)) = (prop(BULLETS_PLAYER), prop(BULLETS_WEAPON)) {
                        // m_iPlayer is the entity index - 1
                        let ent = player as u32 + 1;
                        if !is_rifle(weapon_id) {
                            continue;
                        }
                        let prev_tick = self.last_shot.insert(ent, ticknum).filter(|&t| t < ticknum);
                        self.pending.push(Shot { tick: ticknum, ent, weapon_id, prev_tick });
                    }
                }
            }
            Message::PacketEntities(msg) => {
                for e in &msg.entities {
                    let id = u32::from(e.entity_index);
                    if matches!(e.update_type, UpdateType::Delete | UpdateType::Leave) {
                        self.dots.remove(&id);
                        continue;
                    }
                    let entering = e.update_type == UpdateType::Enter;
                    let is_dot = if entering {
                        parser_state
                            .server_classes
                            .get(usize::from(e.server_class))
                            .is_some_and(|c| c.name.as_str() == "CSniperDot")
                    } else {
                        self.dots.contains_key(&id)
                    };
                    if !is_dot {
                        continue;
                    }
                    let dot = self.dots.entry(id).or_insert(Dot { owner: 0, pos: [0.0; 3] });
                    let mut apply = |identifier: SendPropIdentifier, value: &SendPropValue| match identifier {
                        DOT_ORIGIN => {
                            if let Ok(v) = Vector::try_from(value) {
                                dot.pos = [v.x, v.y, v.z];
                            }
                        }
                        DOT_OWNER => {
                            if let Ok(v) = i64::try_from(value) {
                                dot.owner = (v as u32) & 0x7FF;
                            }
                        }
                        _ => {}
                    };
                    if entering {
                        for prop in e.props(parser_state) {
                            apply(prop.identifier, &prop.value);
                        }
                    } else {
                        for prop in &e.props {
                            apply(prop.identifier, &prop.value);
                        }
                    }
                }
            }
            Message::GameEvent(ev) => {
                if let GameEvent::PlayerHurt(h) = &ev.event {
                    self.hurts.push_back(Hurt { tick: ticknum, attacker_uid: h.attacker, victim_uid: h.user_id, custom: h.custom });
                }
            }
            _ => {}
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        let ticknum = u32::from(state.tick);
        for p in &state.players {
            let hist = self.history.entry(u32::from(p.entity)).or_default();
            if hist.back().is_some_and(|s| s.tick > ticknum) {
                hist.clear(); // signon packets carry unrelated ticks
            }
            if hist.back().is_some_and(|s| s.tick == ticknum) {
                continue;
            }
            hist.push_back(Snap {
                tick: ticknum,
                pos: [p.position.x, p.position.y, p.position.z],
                yaw: p.view_angle,
                pitch: p.pitch_angle,
                ducking: p.is_ducking(),
                alive: p.state == PlayerState::Alive,
                in_pvs: p.in_pvs,
                team: p.team,
                class: p.class,
                cond: p.cond,
            });
            while hist.len() > HISTORY_TICKS {
                hist.pop_front();
            }
        }
        for dot in self.dots.values() {
            let hist = self.dot_history.entry(dot.owner).or_default();
            if hist.back().is_some_and(|(t, _)| *t > ticknum) {
                hist.clear();
            }
            if hist.back().is_some_and(|(t, _)| *t == ticknum) {
                continue;
            }
            hist.push_back((ticknum, dot.pos));
            while hist.len() > HISTORY_TICKS {
                hist.pop_front();
            }
        }
        while self.hurts.front().is_some_and(|h| h.tick + 32 < ticknum) {
            self.hurts.pop_front();
        }
        if !self.pending.is_empty() {
            let (ready, rest): (Vec<Shot>, Vec<Shot>) =
                std::mem::take(&mut self.pending).into_iter().partition(|s| s.tick + POST_TICKS <= ticknum);
            self.pending = rest;
            for shot in ready {
                self.record_shot(shot, state);
            }
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let min_score = get_parameter_value::<f32>(&self.params, "min_score");
        let min_shots = get_parameter_value::<i32>(&self.params, "min_shots").max(1) as usize;
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let model = model();
        let mut sids: Vec<u64> = self.shooters.keys().copied().collect();
        sids.sort_unstable();
        let mut detections = Vec::new();
        for sid in sids {
            let data = &self.shooters[&sid];
            let hits: Vec<&ShotRecord> = data.shots.iter().filter(|s| s.hit && s.victim.is_some()).collect();
            let k = estimate_k(&hits);
            let first = self.shots.len();
            for s in &data.shots {
                let Some(features) = s.features(k) else { continue };
                let score = model.map_or(f32::NAN, |m| m.shot.probability(&features));
                let victim = s.target(k).map_or(0, |t| t.sid);
                let path = s.target(k).map_or_else(Vec::new, |t| {
                    s.aim.iter().filter_map(|a| s.rel(a.j, t, k).map(|r| (a.j, r.x, r.y))).collect()
                });
                self.shots.push(ShotFeatures { tick: s.tick, shooter: sid, victim, weapon: s.weapon.clone(), features, score, path });
            }
            let shots: Vec<&ShotFeatures> = self.shots[first..].iter().collect();
            let n_hits = shots.iter().filter(|s| s.features[IDX_HIT] > 0.5).count();
            if shots.len() < min_shots || n_hits == 0 {
                continue;
            }
            let features = unit_features(&shots);
            // without a scoped crosshair path there is nothing to judge
            let score = if features[IDX_UNIT_SCOPE_MED] >= MIN_SCOPE_MED {
                model.map_or(f32::NAN, |m| m.unit.probability(&features, shots.len()))
            } else {
                f32::NAN
            };
            self.units.push(UnitRecord { player: sid, n_hits, n_shots: shots.len(), features, score });
            if !(score >= min_score) {
                continue;
            }
            let mut top: Vec<&ShotFeatures> = shots.iter().copied().filter(|s| s.features[IDX_HIT] > 0.5).collect();
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            for s in top.into_iter().take(max_detections) {
                let f = &s.features;
                let opt = |v: f32| if v.is_nan() { None } else { Some(v) };
                detections.push(Detection {
                    tick: s.tick,
                    algorithm: "fidoo/ml_triggerbot".to_string(),
                    player: sid,
                    data: json!({
                        "score": score,
                        "shot_score": s.score,
                        "hits": n_hits,
                        "shots": shots.len(),
                        "weapon": s.weapon,
                        "headshot": f[IDX_HEADSHOT] > 0.5,
                        "target": s.victim,
                        "distance": f[IDX_DIST].round(),
                        "head_error_radii": f[IDX_D0],
                        "scoped_ticks": f[IDX_SCOPE_LEN],
                        "near_head_ticks": opt(f[IDX_NEAR_TIME]),
                        "head_crossings": opt(f[IDX_CROSS]),
                        "reversals": opt(f[IDX_REVERSALS]),
                        "speed_at_shot_radii": opt(f[IDX_V_SHOT]),
                        "acquisition_ticks": f[IDX_ACQ2],
                        "ready_hold_ticks": opt(f[IDX_READY_HOLD]),
                        "zoom_delay_ticks": opt(f[IDX_ZOOM_DELAY]),
                        "lag_ticks": f[IDX_LAG],
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

    /// A static target at 1000 units; the crosshair sweeps onto its head at 0.6 radii per tick and stops
    /// at `stop_at`.
    fn sweep_shot(stop_at: i32) -> ShotRecord {
        let eye = [0.0, 0.0, 75.0];
        let head = [1000.0, 0.0, 75.0];
        let radius = (HEAD_RADIUS / 1000.0).atan().to_degrees();
        let (yaw, pitch) = dir_angles(eye, head);
        let aim = (-AIM_WINDOW..=0)
            .map(|j| AimSample { j, eye, yaw: yaw + (stop_at - j.min(stop_at)) as f32 * 0.6 * radius, pitch })
            .collect();
        ShotRecord {
            tick: 1000,
            weapon: "Sniper Rifle".to_string(),
            headshot: true,
            hit: true,
            victim: Some(0),
            zoom_delay: None,
            zoom_long: true,
            prev_tick: None,
            shooter_speed: 0.0,
            aim,
            targets: vec![TargetTrack { sid: 1, head: (-70..=0).map(|j| (j, head)).collect() }],
        }
    }

    #[test]
    fn instant_shot_has_short_ready_hold() {
        let f = sweep_shot(0).features(DEFAULT_K).unwrap();
        assert!(f[IDX_D0] < 1e-3);
        assert_eq!(f[IDX_ACQ2], 4.0);
        // within 1 radius of the aim point only on the tick before
        assert_eq!(f[IDX_READY_HOLD], 1.0);
        assert!((f[12] - 0.6).abs() < 1e-2); // v0
    }

    #[test]
    fn held_shot_counts_ready_hold() {
        let f = sweep_shot(-10).features(DEFAULT_K).unwrap();
        assert_eq!(f[IDX_ACQ2], 14.0);
        // on the aim point from -10 (and within 1 radius at -11)
        assert_eq!(f[IDX_READY_HOLD], 11.0);
        let mut s = sweep_shot(-10);
        // rifle ready only 3 ticks before the shot
        s.prev_tick = Some(1000 - 103);
        let f = s.features(DEFAULT_K).unwrap();
        assert_eq!(f[IDX_READY_HOLD], 3.0);
        assert_eq!(f[IDX_READY_LATENCY], 3.0);
    }

    #[test]
    fn unit_features_match_names() {
        let s = sweep_shot(0);
        let f = s.features(DEFAULT_K).unwrap();
        let row = ShotFeatures { tick: 0, shooter: 0, victim: 1, weapon: String::new(), features: f, score: 0.0, path: vec![] };
        assert_eq!(unit_features(&[&row]).len(), unit_feature_names().len());
    }
}
