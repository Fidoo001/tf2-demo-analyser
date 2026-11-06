//! Learned low FOV hitscan silent aim detection on spectated players' eye angles. See docs/ml_silent_aim.md.
//!
//! A silent aimbot writes the aim into the one usercmd that fires (Amalgam `AimbotHitscan.cpp`, AimType Silent:
//! `pCmd->viewangles = vAngles` only when `G::Attacking == 1`) and the next usercmd carries the player's own view
//! again. In the networked eye angles a shot's update leaves the mouse path for one update and comes back (A -> B -> C).
//! With backtrack the shot also sets `tick_count` to an older record, so B points at where the victim was then.
//!
//! Every bullet of a spectated player is measured against the mouse path; a bullet that hurt an enemy is also measured
//! against the victim at every lag compensation rewind: how close is B to the victim, and could the path without B have
//! hit at all. Lag, batched usercmds and jitter are not excused by rules. They are features of the shot (update gaps,
//! the player's own impulses before the shot), and the player's impulses on shots are compared with their impulses on
//! ordinary updates. A shot model scores the suspicious hits, a player model scores the demo.
//!
//! `examples/dump_ml_silent_aim_features.rs` dumps the same features for training.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

use crate::algorithms::fidoo::ml_backtrack::is_precise;
use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{resolve_recorder_sid, FxHashMap};
use crate::util::nocrex::jankguard::JankGuard;

const BULLETS_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
const BULLETS_WEAPON: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iWeaponID");

/// Largest rewind tried, in ticks (sv_maxunlag 1 s plus a little)
pub const MAX_LAG: usize = 70;
/// Rewinds considered for the typical rewind
const LAGS: usize = 61;
/// Attacker updates used around a shot
const PRE_TICKS: u32 = 40;
const POST_TICKS: u32 = 14;
const HISTORY_TICKS: u32 = MAX_LAG as u32 + PRE_TICKS + POST_TICKS + 12;
/// The visible flick can be off the bullet tick by this many ticks (docs/ml_sniperaim.md 7.4)
const OFFSET_TICKS: i64 = 2;
/// Largest gap (sim ticks) between updates that still counts as one mouse path
const MAX_GAP: i64 = 8;
const MAX_VICTIM_GAP: i64 = 4;
/// Hurt events arrive up to 8 ticks after the bullet, or just before it in the same tick
const HURT_AFTER: u32 = 8;
const HURT_BEFORE: u32 = 1;
/// Hits of one attacker closer than this are scored once (sprays)
const MIN_HIT_SPACING: u32 = 8;
const MIN_SINCE_SPAWN: u32 = 60;
const MIN_DISTANCE: f64 = 50.0;
/// Only hits with an impulse of at least this many degrees are scored by the shot model (one yaw step is 0.35)
pub const CANDIDATE_IMPULSE: f32 = 0.3;
/// Impulses above this are not low FOV, but a large silent flick is still a silent flick: kept, capped
const MAX_ANGLE: f32 = 60.0;
const MIN_NOISE: f32 = 0.35;
const MAX_RATIO: f32 = 50.0;
const BODY_HALF_WIDTH: f64 = 24.0;
const BODY_BOTTOM: f64 = 5.0;
const BODY_TOP_STANDING: f64 = 80.0;
const BODY_TOP_DUCKING: f64 = 60.0;
/// Networked eye angle precision (degrees) added to the target's size
const EYE_TOLERANCE: f64 = 0.35;
const NEAR: i64 = 2;
const TF_DMG_CUSTOM_BURNING: u16 = 3;
const TF_DMG_CUSTOM_BLEEDING: u16 = 34;
/// Impulse levels of the per-player counters (degrees)
const LEVELS: [f32; 3] = [0.7, 1.5, 3.0];
/// Added to the rates in the shot / ordinary update ratio
const RATE_FLOOR: f32 = 0.005;
/// Lowest ordinary update rate assumed in the binomial scores
const BASE_FLOOR: f32 = 0.002;
const MAX_BINOM: f64 = 300.0;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Per-hit features. p0..p4 are consecutive updates, p2 the update with the largest impulse near the bullet.
pub const SHOT_FEATURES: [&str; 65] = [
    "impulse", "impulse0", "offset", "out", "back", "span", "exact_ret", "back_cos", "in_cos", "dev_lin", "seg",
    "pred_pre", "pred_post", "detour", "noise_med", "noise_p90", "noise_max", "ratio", "near_count", "v_pre", "v_post",
    "still_pre", "still_post", "gap_pre", "gap_post", "gap_med", "gap_max", "yaw_share", "pitch", "family", "precise",
    "on_ground", "ducking", "has_target", "headshot", "dist", "ang_size", "err_b", "err_b_hu", "k_b", "err_a_kb",
    "err_c_kb", "gain_a", "gain_c", "path_err", "path_hu", "gain_path", "dir_cos", "head_hu", "n_ok", "v_target",
    "k_off", "err_b_k0", "err_a_k0", "err_c_k0", "gain_k0", "near_ok", "path_near_ok", "k0", "path_kb", "b_off", "a_off",
    "c_off", "path_off", "gain_rel",
];
const IDX_IMPULSE: usize = 0;
const IDX_OFFSET: usize = 2;
const IDX_OUT: usize = 3;
const IDX_BACK: usize = 4;
const IDX_SPAN: usize = 5;
const IDX_RATIO: usize = 17;
const IDX_GAP_MED: usize = 25;
const IDX_DIST: usize = 35;
const IDX_ERR_B_HU: usize = 38;
const IDX_K_B: usize = 39;
const IDX_PATH_HU: usize = 45;
const IDX_K_OFF: usize = 51;
const IDX_ERR_B_K0: usize = 52;
const IDX_ERR_A_K0: usize = 53;
const IDX_ERR_C_K0: usize = 54;
const IDX_GAIN_K0: usize = 55;
const IDX_NEAR_OK: usize = 56;
const IDX_PATH_NEAR_OK: usize = 57;
const IDX_K0: usize = 58;
const IDX_B_OFF: usize = 60;

/// Player features, see `unit_features`. scripts/ml_silent_aim_train.py has a copy (`unit_row`).
pub const UNIT_FEATURES: [&str; 39] = [
    "n_hits", "n_shots", "n_cand", "cand_share", "k3", "k5", "k7", "k9", "s5", "s9", "top1", "top2", "top3", "top5",
    "sum_log", "sum_p", "h07", "h15", "h30", "m07", "m15", "m30", "b07", "b15", "b30", "ex07", "ex15", "ex30", "gap_med",
    "imp_med_k5", "k_need", "s_need", "k_bt", "s_bt", "k_exact", "ratio_p90", "bs07", "bs15", "bs30",
];

pub fn unit_feature_names() -> Vec<String> {
    UNIT_FEATURES.iter().map(|s| s.to_string()).collect()
}

fn shot_feature_names() -> Vec<String> {
    SHOT_FEATURES.iter().map(|s| s.to_string()).collect()
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
    shot: Booster,
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_silent_aim_model.json")).ok()?;
            (model.unit.features == unit_feature_names() && model.shot.features == shot_feature_names()).then_some(model)
        })
        .as_ref()
}

/// One update of a player: a new simulation tick or a view change
#[derive(Clone, Copy, Debug)]
struct Update {
    tick: u32,
    sim: i64,
    yaw: f32,
    pitch: f32,
    pos: [f32; 3],
    eye_height: f32,
    head_height: f32,
    ducking: bool,
    on_ground: bool,
}

#[derive(Clone, Debug)]
struct PendingShot {
    tick: u32,
    #[allow(dead_code)]
    ent: u32,
    player: u64,
    weapon: String,
    family: u8,
}

#[derive(Clone, Copy, Debug)]
struct Hurt {
    tick: u32,
    victim: u64,
    headshot: bool,
}

/// One scored hit
#[derive(Clone, Debug)]
pub struct ShotRecord {
    pub tick: u32,
    pub shooter: u64,
    pub victim: u64,
    pub weapon: String,
    pub features: [f32; SHOT_FEATURES.len()],
    pub score: f32,
    curves: Option<Curves>,
}

#[derive(Clone, Debug)]
struct Curves {
    b: Vec<f32>,
    a: Vec<f32>,
    c: Vec<f32>,
    path: Vec<f32>,
    ang_size: f32,
}

/// Per-player counters of the demo: impulses on the bullet update of hits and misses and on ordinary updates
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub shots: u32,
    pub hits: u32,
    pub misses: u32,
    pub base: u32,
    pub hit_levels: [u32; 3],
    pub miss_levels: [u32; 3],
    pub base_levels: [u32; 3],
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub k0: usize,
    pub features: Vec<f32>,
    pub score: f32,
}

fn wrap180(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

/// `to` relative to `from` in mouse space: wrapped yaw scaled to arc length at the pitch of `from`, and pitch
fn rel(from: &Update, to: &Update) -> (f32, f32) {
    (wrap180(to.yaw - from.yaw) * from.pitch.to_radians().cos(), to.pitch - from.pitch)
}

fn hypot(a: (f32, f32)) -> f32 {
    a.0.hypot(a.1)
}

fn cos2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (la, lb) = (hypot(a), hypot(b));
    if la < 1e-3 || lb < 1e-3 {
        return f32::NAN;
    }
    (a.0 * b.0 + a.1 * b.1) / (la * lb)
}

fn point_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let l2 = ab.0 * ab.0 + ab.1 * ab.1;
    let t = if l2 > 0.0 { (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - a.0 - ab.0 * t).hypot(p.1 - a.1 - ab.1 * t)
}

fn gap_of(a: &Update, b: &Update) -> i64 {
    let sim = b.sim - a.sim;
    if (1..=200).contains(&sim) {
        sim
    } else {
        i64::from(b.tick.saturating_sub(a.tick)).max(1)
    }
}

/// The out-and-back measurement of update `i`
#[derive(Clone, Copy, Debug)]
struct Triangle {
    impulse: f32,
    out: f32,
    back: f32,
    span: f32,
    exact_ret: bool,
    back_cos: f32,
    in_cos: f32,
    dev_lin: f32,
    seg: f32,
    pred_pre: f32,
    pred_post: f32,
    still_pre: f32,
    still_post: f32,
    v_pre: f32,
    v_post: f32,
    gap_pre: i64,
    gap_post: i64,
    yaw_share: f32,
}

/// `None` when update `i` has no neighbour close enough. The impulse is the distance of p2 from every smooth
/// explanation of the mouse path: the segment p1-p3, the motion before and after extrapolated, and the two legs
/// (an aim that did not leave, or stays).
fn triangle(u: &[Update], i: usize) -> Option<Triangle> {
    if i == 0 || i + 1 >= u.len() {
        return None;
    }
    let (gap_pre, gap_post) = (gap_of(&u[i - 1], &u[i]), gap_of(&u[i], &u[i + 1]));
    if gap_pre > MAX_GAP || gap_post > MAX_GAP {
        return None;
    }
    let p2 = &u[i];
    let p1 = rel(p2, &u[i - 1]);
    let p3 = rel(p2, &u[i + 1]);
    let o = (0.0, 0.0);
    let (out, back) = (hypot(p1), hypot(p3));
    let span = (p3.0 - p1.0).hypot(p3.1 - p1.1);
    let seg = point_segment(o, p1, p3);
    let t = gap_pre as f32 / (gap_pre + gap_post) as f32;
    let dev_lin = (p1.0 + (p3.0 - p1.0) * t).hypot(p1.1 + (p3.1 - p1.1) * t);
    let prior = (i >= 2).then(|| (rel(p2, &u[i - 2]), gap_of(&u[i - 2], &u[i - 1]))).filter(|(_, g)| *g <= MAX_GAP);
    let next = u.get(i + 2).map(|n| (rel(p2, n), gap_of(&u[i + 1], n))).filter(|(_, g)| *g <= MAX_GAP);
    let (pred_pre, still_pre, in_cos, v_pre) = match prior {
        Some((p0, g)) => {
            let v = ((p1.0 - p0.0) / g as f32, (p1.1 - p0.1) / g as f32);
            let at = (p1.0 + v.0 * gap_pre as f32, p1.1 + v.1 * gap_pre as f32);
            (hypot(at), hypot((p1.0 - p0.0, p1.1 - p0.1)), cos2((p1.0 - p0.0, p1.1 - p0.1), (-p1.0, -p1.1)), hypot(v))
        }
        None => (f32::NAN, f32::NAN, f32::NAN, f32::NAN),
    };
    let (pred_post, still_post, v_post) = match next {
        Some((p4, g)) => {
            let v = ((p4.0 - p3.0) / g as f32, (p4.1 - p3.1) / g as f32);
            let at = (p3.0 - v.0 * gap_post as f32, p3.1 - v.1 * gap_post as f32);
            (hypot(at), hypot((p4.0 - p3.0, p4.1 - p3.1)), hypot(v))
        }
        None => (f32::NAN, f32::NAN, f32::NAN),
    };
    let mut impulse = out.min(back).min(seg);
    if !pred_pre.is_nan() {
        impulse = impulse.min(pred_pre);
    }
    if !pred_post.is_nan() {
        impulse = impulse.min(pred_post);
    }
    Some(Triangle {
        impulse,
        out,
        back,
        span,
        exact_ret: u[i - 1].yaw == u[i + 1].yaw && u[i - 1].pitch == u[i + 1].pitch,
        back_cos: cos2((-p1.0, -p1.1), p3),
        in_cos,
        dev_lin,
        seg,
        pred_pre,
        pred_post,
        still_pre,
        still_post,
        v_pre,
        v_post,
        gap_pre,
        gap_post,
        yaw_share: if out > 1e-3 { p1.0.abs() / out } else { f32::NAN },
    })
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

fn eye_of(a: &Update) -> V3 {
    [f64::from(a.pos[0]), f64::from(a.pos[1]), f64::from(a.pos[2] + a.eye_height)]
}

/// Point of the body column of `v` closest to the view height along the ray, relative to the eye
fn column_point(eye: V3, d: V3, v: &Update) -> V3 {
    let vpos = v.pos.map(f64::from);
    let top = if v.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING };
    let rel = [vpos[0] - eye[0], vpos[1] - eye[1]];
    let dh = rel[0].hypot(rel[1]);
    let dxy = d[0].hypot(d[1]).max(1e-6);
    let aim_z = eye[2] + d[2] / dxy * dh;
    let tz = aim_z.clamp(vpos[2] + BODY_BOTTOM, vpos[2] + top);
    [rel[0], rel[1], tz - eye[2]]
}

/// Error (degrees) from a view to the body column of the victim at every rewind, NaN where the victim wasn't sampled
fn error_curve(eye: V3, yaw: f32, pitch: f32, at: &[Option<Update>]) -> Vec<f32> {
    let d = view_dir(yaw, pitch);
    at.iter().map(|v| v.as_ref().map_or(f32::NAN, |v| angle_between(d, column_point(eye, d, v)) as f32)).collect()
}

/// The victim's sample at every rewind of the bullet tick
fn victim_at(vic: &[Update], tick: u32) -> Vec<Option<Update>> {
    (0..=MAX_LAG)
        .map(|lag| {
            let q = i64::from(tick) - lag as i64;
            let i = vic.partition_point(|s| i64::from(s.tick) <= q);
            (i > 0 && q - i64::from(vic[i - 1].tick) <= MAX_VICTIM_GAP).then(|| vic[i - 1])
        })
        .collect()
}

fn median(values: &mut [f32]) -> f32 {
    quantile(values, 0.5)
}

/// Linear interpolation quantile (numpy's default), NaN when empty
fn quantile(values: &mut [f32], q: f64) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let pos = q * (values.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    (f64::from(values[lo]) + (f64::from(values[hi]) - f64::from(values[lo])) * (pos - lo as f64)) as f32
}

/// Index of the update on the bullet tick and of the one with the largest impulse near it
fn shot_updates(u: &[Update], tick: u32) -> Option<(usize, usize)> {
    let at = u.iter().rposition(|s| s.tick <= tick)?;
    let mut best = at;
    let mut best_impulse = triangle(u, at).map_or(0.0, |t| t.impulse);
    for (i, s) in u.iter().enumerate() {
        if i != at && (i64::from(s.tick) - i64::from(tick)).abs() <= OFFSET_TICKS {
            let impulse = triangle(u, i).map_or(0.0, |t| t.impulse);
            if impulse > best_impulse {
                best = i;
                best_impulse = impulse;
            }
        }
    }
    Some((at, best))
}

/// Features of one hit that need no knowledge of the player's other hits; the rewind-relative ones are NaN until `finish`
fn shot_features(u: &[Update], tick: u32, hit: Option<(&[Update], bool)>, family: u8, precise: bool) -> Option<([f32; SHOT_FEATURES.len()], Option<Curves>)> {
    let (at, best) = shot_updates(u, tick)?;
    let tri = triangle(u, best)?;
    let impulse0 = triangle(u, at).map_or(0.0, |t| t.impulse);
    let cap = |v: f32| v.min(MAX_ANGLE);

    // the player's own impulses before the bullet, on the same measure
    let mut noise: Vec<f32> = (0..u.len())
        .filter(|&i| u[i].tick + 3 < tick && u[i].tick + PRE_TICKS >= tick)
        .filter_map(|i| triangle(u, i).map(|t| t.impulse))
        .collect();
    let noise_max = noise.iter().copied().fold(f32::NAN, f32::max);
    let noise_med = median(&mut noise);
    let noise_p90 = quantile(&mut noise, 0.9);
    let near_count = (0..u.len())
        .filter(|&i| i != best && u[i].tick + 8 >= tick && u[i].tick <= tick + 8)
        .filter_map(|i| triangle(u, i))
        .filter(|t| t.impulse >= 0.5 * tri.impulse.max(0.35))
        .count();
    let mut gaps: Vec<f32> = u.windows(2).map(|w| gap_of(&w[0], &w[1]) as f32).collect();
    let gap_max = gaps.iter().copied().fold(0.0, f32::max);
    let gap_med = median(&mut gaps);
    let detour = (tri.out + tri.back) / tri.span.max(MIN_NOISE);
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let p2 = &u[best];
    let (a, c) = (&u[best - 1], &u[best + 1]);

    let mut f = [f32::NAN; SHOT_FEATURES.len()];
    f[..33].copy_from_slice(&[
        cap(tri.impulse),
        cap(impulse0),
        (i64::from(p2.tick) - i64::from(tick)) as f32,
        cap(tri.out),
        cap(tri.back),
        cap(tri.span),
        flag(tri.exact_ret),
        tri.back_cos,
        tri.in_cos,
        cap(tri.dev_lin),
        cap(tri.seg),
        cap(tri.pred_pre),
        cap(tri.pred_post),
        detour.min(20.0),
        noise_med,
        noise_p90,
        noise_max,
        (tri.impulse / noise_p90.max(MIN_NOISE)).min(MAX_RATIO),
        near_count as f32,
        tri.v_pre,
        tri.v_post,
        cap(tri.still_pre),
        cap(tri.still_post),
        tri.gap_pre as f32,
        tri.gap_post as f32,
        gap_med,
        gap_max,
        tri.yaw_share,
        p2.pitch.abs(),
        f32::from(family),
        flag(precise),
        flag(p2.on_ground),
        flag(p2.ducking),
    ]);
    f[33] = 0.0;

    let mut curves = None;
    if let Some((vic, headshot)) = hit {
        f[34] = flag(headshot);
        let at_lag = victim_at(vic, tick);
        let eye = eye_of(p2);
        let b = error_curve(eye, p2.yaw, p2.pitch, &at_lag);
        if let Some(k_b) = (0..=MAX_LAG).filter(|&k| !b[k].is_nan()).min_by(|&x, &y| b[x].total_cmp(&b[y])) {
            let v = at_lag[k_b].as_ref().unwrap();
            let db = view_dir(p2.yaw, p2.pitch);
            let target = column_point(eye, db, v);
            let dist = norm(target);
            if dist >= MIN_DISTANCE {
                let hu = |deg: f32| (f64::from(deg).to_radians().tan() * dist) as f32;
                let ca = error_curve(eye_of(a), a.yaw, a.pitch, &at_lag);
                let cc = error_curve(eye_of(c), c.yaw, c.pitch, &at_lag);
                // the mouse path without the excursion: the chord between the updates around it
                let mut path = ca.clone();
                let (dy, dp) = (wrap180(c.yaw - a.yaw), c.pitch - a.pitch);
                for n in 1..=4 {
                    let t = n as f32 / 4.0;
                    let e = error_curve(eye, a.yaw + dy * t, a.pitch + dp * t, &at_lag);
                    for (o, v) in path.iter_mut().zip(&e) {
                        *o = o.min(*v);
                    }
                }
                let path_err = path.iter().copied().fold(f32::INFINITY, f32::min);
                let ang_size = BODY_HALF_WIDTH.atan2(dist).to_degrees() as f32;
                let err_b = b[k_b];
                // flick direction against the direction from A to the victim at that rewind
                let (yaw_t, pitch_t) = (target[1].atan2(target[0]).to_degrees() as f32, (-target[2]).atan2(target[0].hypot(target[1])).to_degrees() as f32);
                let scale = a.pitch.to_radians().cos();
                let to_target = (wrap180(yaw_t - a.yaw) * scale, pitch_t - a.pitch);
                let flick = rel(a, p2);
                let head = [f64::from(v.pos[0]) - eye[0], f64::from(v.pos[1]) - eye[1], f64::from(v.pos[2] + v.head_height) - eye[2]];
                let head_hu = (angle_between(db, head).to_radians().tan() * norm(head)) as f32;
                let tol = ang_size + EYE_TOLERANCE as f32;
                let n_ok = b.iter().filter(|e| **e <= tol).count();
                let v_target = at_lag.get(k_b + 2).and_then(|s| s.as_ref()).map_or(f32::NAN, |older| {
                    let t0 = column_point(eye, db, older);
                    (angle_between(t0, target) / 2.0) as f32
                });
                f[33] = 1.0;
                f[35] = dist as f32;
                f[36] = ang_size;
                f[37] = err_b;
                f[38] = hu(err_b);
                f[39] = k_b as f32;
                f[40] = ca[k_b];
                f[41] = cc[k_b];
                f[42] = ca[k_b] - err_b;
                f[43] = cc[k_b] - err_b;
                f[44] = path_err;
                f[45] = hu(path_err);
                f[46] = path_err - err_b;
                f[47] = cos2(flick, to_target);
                f[48] = head_hu;
                f[49] = n_ok as f32;
                f[50] = v_target;
                f[59] = path[k_b];
                curves = Some(Curves { b, a: ca, c: cc, path, ang_size });
            }
        }
    }
    Some((f, curves))
}

/// Bits of the rewinds at which the curve is on target
fn mask(curve: &[f32], ang_size: f32) -> (u64, bool) {
    let tol = ang_size + EYE_TOLERANCE as f32;
    let (mut ok, mut valid) = (0u64, 0);
    for (lag, &e) in curve.iter().take(LAGS).enumerate() {
        if !e.is_nan() {
            valid += 1;
            if e <= tol {
                ok |= 1 << lag;
            }
        }
    }
    let n = ok.count_ones() as usize;
    (ok, n > 0 && n * 2 <= valid)
}

fn span(lo: i64, hi: i64) -> u64 {
    let lo = lo.max(0) as u32;
    let hi = hi.min(LAGS as i64 - 1);
    if hi < i64::from(lo) {
        return 0;
    }
    let hi = hi as u32;
    (if hi >= 63 { u64::MAX } else { (1u64 << (hi + 1)) - 1 }) & !((1u64 << lo) - 1)
}

/// The rewind that explains most of a player's informative hits
fn typical_rewind(masks: &[u64]) -> usize {
    let cover = |l: i64| masks.iter().filter(|&&m| m & span(l - 1, l + 1) != 0).count();
    (0..LAGS as i64).fold(0i64, |b, l| if cover(l) > cover(b) { l } else { b }) as usize
}

/// Fills the features relative to the player's typical rewind
fn fill_rewind(f: &mut [f32; SHOT_FEATURES.len()], c: &Curves, k0: usize) {
    let k = k0 as i64;
    let near = span(k - NEAR, k + NEAR);
    let (b, a, cc) = (c.b[k0], c.a[k0], c.c[k0]);
    f[IDX_K_OFF] = f[IDX_K_B] - k0 as f32;
    f[IDX_ERR_B_K0] = b;
    f[IDX_ERR_A_K0] = a;
    f[IDX_ERR_C_K0] = cc;
    f[IDX_GAIN_K0] = a.min(cc) - b;
    f[IDX_NEAR_OK] = if mask(&c.b, c.ang_size).0 & near != 0 { 1.0 } else { 0.0 };
    f[IDX_PATH_NEAR_OK] = if mask(&c.path, c.ang_size).0 & near != 0 { 1.0 } else { 0.0 };
    f[IDX_K0] = k0 as f32;
    // in units of the target's size: at long range a small flick is what it takes to get on target
    let near_path = (k0.saturating_sub(NEAR as usize)..=(k0 + NEAR as usize).min(MAX_LAG)).map(|l| c.path[l]).fold(f32::INFINITY, f32::min);
    f[IDX_B_OFF] = b / c.ang_size;
    f[IDX_B_OFF + 1] = a / c.ang_size;
    f[IDX_B_OFF + 2] = cc / c.ang_size;
    f[IDX_B_OFF + 3] = near_path / c.ang_size;
    f[IDX_B_OFF + 4] = (a.min(cc) - b) / c.ang_size;
}

/// -log10 P(X >= x) for X ~ Binomial(n, p)
fn binom_score(x: u32, n: u32, p: f64) -> f64 {
    if x == 0 || n == 0 {
        return 0.0;
    }
    let ln_pmf = |i: u32| {
        let lc: f64 = (0..i).map(|j| f64::from(n - j).ln() - f64::from(j + 1).ln()).sum();
        lc + f64::from(i) * p.ln() + f64::from(n - i) * (1.0 - p).ln()
    };
    let terms: Vec<f64> = (x..=n).map(ln_pmf).collect();
    let m = terms.iter().cloned().fold(f64::MIN, f64::max);
    let ln_tail = m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln();
    (-ln_tail / std::f64::consts::LN_10).clamp(0.0, MAX_BINOM)
}

/// Player features of one player's scored hits and counters; `unit_row` in the training script is a copy
pub fn unit_features(cands: &[&ShotRecord], n: &Counters) -> Vec<f32> {
    let mut p: Vec<f32> = cands.iter().map(|c| c.score).collect();
    p.sort_by(|a, b| b.total_cmp(a));
    let k = |t: f32| p.iter().filter(|v| **v >= t).count() as f32;
    let top = |m: usize| {
        let m = m.min(p.len());
        p[..m].iter().sum::<f32>() / m.max(1) as f32
    };
    let hits = n.hits.max(1) as f32;
    let rate = |c: u32, of: u32| c as f32 / of.max(1) as f32;
    let h: Vec<f32> = n.hit_levels.iter().map(|&c| rate(c, n.hits)).collect();
    let m: Vec<f32> = n.miss_levels.iter().map(|&c| rate(c, n.misses)).collect();
    let b: Vec<f32> = n.base_levels.iter().map(|&c| rate(c, n.base)).collect();
    let ex = |i: usize| (h[i] + RATE_FLOOR) / (b[i] + RATE_FLOOR);
    let col = |i: usize| cands.iter().map(|c| c.features[i]).collect::<Vec<f32>>();
    let mut strong: Vec<f32> = cands.iter().filter(|c| c.score >= 0.5).map(|c| c.features[IDX_IMPULSE]).collect();
    // the excursion hit the victim at the typical rewind and the path without it could not
    let need = cands
        .iter()
        .filter(|c| c.score >= 0.3 && c.features[IDX_ERR_B_HU] <= 24.0 && c.features[IDX_PATH_HU] > 36.0)
        .count() as f32;
    // on target only away from the typical rewind, and not by the path
    let bt = cands
        .iter()
        .filter(|c| c.score >= 0.3 && c.features[IDX_NEAR_OK] == 0.0 && c.features[IDX_PATH_NEAR_OK] == 0.0 && c.features[IDX_K_OFF].abs() >= 3.0)
        .count() as f32;
    let exact = cands.iter().filter(|c| c.score >= 0.3 && c.features[6] == 1.0).count() as f32;
    let sum_log: f64 = p.iter().map(|v| -(1.0 - f64::from(v.min(0.999))).log10()).sum();
    vec![
        n.hits as f32,
        n.shots as f32,
        p.len() as f32,
        p.len() as f32 / hits,
        k(0.3),
        k(0.5),
        k(0.7),
        k(0.9),
        k(0.5) / hits,
        k(0.9) / hits,
        top(1),
        top(2),
        top(3),
        top(5),
        sum_log as f32,
        p.iter().sum(),
        h[0],
        h[1],
        h[2],
        m[0],
        m[1],
        m[2],
        b[0],
        b[1],
        b[2],
        ex(0),
        ex(1),
        ex(2),
        quantile(&mut col(IDX_GAP_MED), 0.5),
        quantile(&mut strong, 0.5),
        need,
        need / hits,
        bt,
        bt / hits,
        exact,
        quantile(&mut col(IDX_RATIO), 0.9),
        binom_score(n.hit_levels[0], n.hits, f64::from(b[0].clamp(BASE_FLOOR, 0.5))) as f32,
        binom_score(n.hit_levels[1], n.hits, f64::from(b[1].clamp(BASE_FLOOR, 0.5))) as f32,
        binom_score(n.hit_levels[2], n.hits, f64::from(b[2].clamp(BASE_FLOOR, 0.5))) as f32,
    ]
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

/// TF_WEAPON ids of CTEFireBullets: 0 sniper rifles; 1 shotguns, scatterguns, pistols, revolvers; 2 minigun and SMGs
fn family_of(weapon_id: i64) -> Option<u8> {
    match weapon_id {
        17 | 77 | 99 => Some(0),
        12..=16 | 41..=43 | 71 | 75 | 76 | 85 => Some(1),
        18 | 19 | 103 => Some(2),
        _ => None,
    }
}

fn eye_height(player: &Player, ducking: bool) -> f32 {
    if ducking {
        return 45.0;
    }
    match player.class_name() {
        "scout" => 65.0,
        "sniper" | "medic" | "heavy" | "spy" => 75.0,
        _ => 68.0,
    }
}

fn head_height(player: &Player, ducking: bool) -> f32 {
    if ducking {
        return 48.0;
    }
    match player.class_name() {
        "heavy" => 76.0,
        "sniper" => 74.0,
        "medic" => 73.0,
        "spy" => 72.0,
        "pyro" => 67.0,
        "engineer" => 66.0,
        _ => 68.0,
    }
}

pub struct MlSilentAim {
    params: Parameters,
    jg: JankGuard,
    history: FxHashMap<u64, VecDeque<Update>>,
    pending: Vec<PendingShot>,
    /// Recent bullets by shooter
    fires: FxHashMap<u64, VecDeque<u32>>,
    hurts: FxHashMap<u64, VecDeque<Hurt>>,
    last_row: FxHashMap<u64, u32>,
    recorder: Option<u64>,
    counters: FxHashMap<u64, Counters>,
    /// Rewind masks of all informative hits by shooter
    masks: FxHashMap<u64, Vec<u64>>,
    /// Hits with a candidate impulse, filled while parsing (`score` and the units by `finish`)
    pub shots: Vec<ShotRecord>,
    /// Scored players, filled by `finish`
    pub units: Vec<UnitRecord>,
    /// Counters of all players, filled by `finish` for the dumper
    pub all_counters: Vec<(u64, Counters)>,
}

impl Default for MlSilentAim {
    fn default() -> Self {
        Self::new()
    }
}

impl MlSilentAim {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("min_score".to_string(), Parameter::Float(0.83)),
                ("min_hits".to_string(), Parameter::Int(3)),
                ("min_shot_score".to_string(), Parameter::Float(0.65)),
                ("max_detections".to_string(), Parameter::Int(20)),
            ]),
            jg: JankGuard::default(),
            history: FxHashMap::default(),
            pending: Vec::new(),
            fires: FxHashMap::default(),
            hurts: FxHashMap::default(),
            last_row: FxHashMap::default(),
            recorder: None,
            counters: FxHashMap::default(),
            masks: FxHashMap::default(),
            shots: Vec::new(),
            units: Vec::new(),
            all_counters: Vec::new(),
        }
    }

    /// The hurt event this bullet caused, if any: a later bullet takes the events from its own tick on
    fn hurt_of(&self, shot: &PendingShot) -> Option<Hurt> {
        let later = self.fires.get(&shot.player).and_then(|f| f.iter().copied().find(|&t| t > shot.tick));
        self.hurts.get(&shot.player)?.iter().copied().find(|h| {
            h.tick + HURT_BEFORE >= shot.tick && h.tick <= shot.tick + HURT_AFTER && later.is_none_or(|t| h.tick + HURT_BEFORE < t)
        })
    }

    fn evaluate(&mut self, shot: PendingShot) {
        let Some(history) = self.history.get(&shot.player) else { return };
        let u: Vec<Update> = history
            .iter()
            .filter(|s| s.tick + PRE_TICKS >= shot.tick && s.tick <= shot.tick + POST_TICKS)
            .copied()
            .collect();
        let hurt = self.hurt_of(&shot);
        let Some(at) = u.iter().rposition(|s| s.tick <= shot.tick) else { return };
        let impulse0 = triangle(&u, at).map_or(0.0, |t| t.impulse);
        let counters = self.counters.entry(shot.player).or_default();
        counters.shots += 1;
        let (levels, total) = match hurt {
            Some(_) => (&mut counters.hit_levels, &mut counters.hits),
            None => (&mut counters.miss_levels, &mut counters.misses),
        };
        *total += 1;
        for (c, l) in levels.iter_mut().zip(LEVELS) {
            *c += u32::from(impulse0 >= l);
        }
        let Some(hurt) = hurt else { return };
        if self.last_row.get(&shot.player).is_some_and(|&t| shot.tick < t + MIN_HIT_SPACING) {
            return;
        }
        self.last_row.insert(shot.player, shot.tick);
        let vic: Vec<Update> = self
            .history
            .get(&hurt.victim)
            .map(|v| v.iter().filter(|s| s.tick <= shot.tick).copied().collect())
            .unwrap_or_default();
        let precise = is_precise(&shot.weapon);
        let Some((features, curves)) = shot_features(&u, shot.tick, Some((&vic, hurt.headshot)), shot.family, precise) else { return };
        if let Some(c) = &curves {
            let (ok, informative) = mask(&c.b, c.ang_size);
            if informative {
                self.masks.entry(shot.player).or_default().push(ok);
            }
        }
        if features[IDX_IMPULSE] >= CANDIDATE_IMPULSE {
            self.shots.push(ShotRecord {
                tick: shot.tick,
                shooter: shot.player,
                victim: hurt.victim,
                weapon: shot.weapon,
                features,
                score: f32::NAN,
                curves,
            });
        }
    }

    /// Counts the impulse of the update before the newest one, unless a bullet is close to it
    fn count_base(&mut self, sid: u64) {
        let Some(h) = self.history.get(&sid) else { return };
        let n = h.len();
        if n < 4 {
            return;
        }
        let u: Vec<Update> = h.range(n - 4..).copied().collect();
        let Some(t) = triangle(&u, 2) else { return };
        let tick = u[2].tick;
        if self.fires.get(&sid).is_some_and(|f| f.iter().any(|&b| b + OFFSET_TICKS as u32 >= tick && b <= tick + OFFSET_TICKS as u32)) {
            return;
        }
        let c = self.counters.entry(sid).or_default();
        c.base += 1;
        for (l, v) in c.base_levels.iter_mut().zip(LEVELS) {
            *l += u32::from(t.impulse >= v);
        }
    }
}

impl<'a> CheatAlgorithm<'a> for MlSilentAim {
    fn default(&self) -> bool {
        false
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/ml_silent_aim"
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    /// The scores aggregate all hits of the demo
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        let mut m = self.jg.handled_messages()?;
        for t in [MessageType::GameEvent, MessageType::TempEntities] {
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
        if let Message::TempEntities(te) = message {
            for event in &te.events {
                if parser_state.server_classes.get(usize::from(event.class_id)).is_none_or(|c| c.name.as_str() != "CTEFireBullets") {
                    continue;
                }
                let prop = |id: SendPropIdentifier| event.props.iter().find(|p| p.identifier == id).and_then(|p| i64::try_from(&p.value).ok());
                let (Some(player), Some(weapon_id)) = (prop(BULLETS_PLAYER), prop(BULLETS_WEAPON)) else { continue };
                let Some(family) = family_of(weapon_id) else { continue };
                // m_iPlayer is the entity index - 1
                let ent = player as u32 + 1;
                let Some(p) = state.players.iter().find(|p| u32::from(p.entity) == ent) else { continue };
                if !p.in_pvs || p.state != PlayerState::Alive || p.is_taunting() {
                    continue;
                }
                let Some(sid) = player_sid(p, state) else { continue };
                if Some(sid) == self.recorder {
                    continue;
                }
                let f = self.fires.entry(sid).or_default();
                f.push_back(tick);
                while f.front().is_some_and(|&t| t + 40 < tick) {
                    f.pop_front();
                }
                if self.jg.spawned(&sid, tick).min(self.jg.teleported(&sid, tick)) < MIN_SINCE_SPAWN {
                    continue;
                }
                self.pending.push(PendingShot { tick, ent, player: sid, weapon: state.get_player_weapon(p), family });
            }
            return Ok(vec![]);
        }
        let Message::GameEvent(GameEventMessage { event: GameEvent::PlayerHurt(h), .. }) = message else { return Ok(vec![]) };
        let (Some(a), Some(v)) = (state.get_id64_from_userid(h.attacker.into()), state.get_id64_from_userid(h.user_id.into())) else {
            return Ok(vec![]);
        };
        // burn and bleed ticks are credited to the active weapon
        if a == v || a == 0 || Some(a) == self.recorder || h.custom == TF_DMG_CUSTOM_BLEEDING || h.custom == TF_DMG_CUSTOM_BURNING {
            return Ok(vec![]);
        }
        let team = |sid: u64| state.players.iter().find(|p| player_sid(p, state) == Some(sid)).map(|p| p.team);
        if team(a) == team(v) {
            return Ok(vec![]);
        }
        if self.jg.spawned(&v, tick).min(self.jg.teleported(&v, tick)) < MIN_SINCE_SPAWN {
            return Ok(vec![]);
        }
        let list = self.hurts.entry(a).or_default();
        list.push_back(Hurt { tick, victim: v, headshot: h.custom == 1 });
        while list.front().is_some_and(|x| x.tick + 40 < tick) {
            list.pop_front();
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        if self.recorder.is_none() {
            self.recorder = resolve_recorder_sid(state);
        }
        let tick = u32::from(state.tick);
        for p in &state.players {
            if !p.in_pvs || p.state != PlayerState::Alive || p.is_taunting() {
                continue;
            }
            let Some(sid) = player_sid(p, state) else { continue };
            let pitch = if p.pitch_angle > 180.0 { p.pitch_angle - 360.0 } else { p.pitch_angle };
            let sim = state.sim_tick(p);
            let h = self.history.entry(sid).or_default();
            if h.back().is_some_and(|s| s.tick > tick) {
                h.clear(); // signon packets carry unrelated ticks
            }
            if h.back().is_some_and(|s| s.sim == sim && s.yaw == p.view_angle && s.pitch == pitch) {
                continue;
            }
            let ducking = p.is_ducking();
            h.push_back(Update {
                tick,
                sim,
                yaw: p.view_angle,
                pitch,
                pos: [p.position.x, p.position.y, p.position.z],
                eye_height: eye_height(p, ducking),
                head_height: head_height(p, ducking),
                ducking,
                on_ground: p.is_on_ground(),
            });
            while h.front().is_some_and(|s| s.tick + HISTORY_TICKS < tick) {
                h.pop_front();
            }
            if Some(sid) != self.recorder {
                self.count_base(sid);
            }
        }
        if !self.pending.is_empty() {
            let (ready, rest): (Vec<PendingShot>, Vec<PendingShot>) =
                std::mem::take(&mut self.pending).into_iter().partition(|s| s.tick + POST_TICKS <= tick);
            self.pending = rest;
            for s in ready {
                self.evaluate(s);
            }
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        for s in std::mem::take(&mut self.pending) {
            self.evaluate(s);
        }
        let min_score = get_parameter_value::<f32>(&self.params, "min_score");
        let min_hits = get_parameter_value::<i32>(&self.params, "min_hits").max(1) as u32;
        let min_shot_score = get_parameter_value::<f32>(&self.params, "min_shot_score");
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let model = model();

        let mut k0s: FxHashMap<u64, usize> = FxHashMap::default();
        for (player, masks) in &self.masks {
            k0s.insert(*player, typical_rewind(masks));
        }
        for shot in &mut self.shots {
            if let (Some(c), Some(&k0)) = (shot.curves.take(), k0s.get(&shot.shooter)) {
                fill_rewind(&mut shot.features, &c, k0);
            }
            shot.score = model.map_or(f32::NAN, |m| m.shot.probability(&shot.features));
        }
        let mut players: Vec<u64> = self.counters.iter().filter(|(_, c)| c.hits >= min_hits).map(|(p, _)| *p).collect();
        players.sort_unstable();
        self.all_counters = players.iter().map(|p| (*p, self.counters[p])).collect();

        let mut detections = Vec::new();
        for player in players {
            let cands: Vec<&ShotRecord> = self.shots.iter().filter(|s| s.shooter == player).collect();
            if cands.is_empty() {
                continue;
            }
            let counters = self.counters[&player];
            let features = unit_features(&cands, &counters);
            let score = model.map_or(f32::NAN, |m| m.unit.probability(&features));
            self.units.push(UnitRecord { player, k0: k0s.get(&player).copied().unwrap_or(0), features, score });
            if !(score >= min_score) {
                continue;
            }
            let mut top = cands;
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            let n_top = top.iter().filter(|s| s.score >= min_shot_score).count().clamp(1, max_detections);
            let round = |v: f32| if v.is_nan() { None } else { Some((v * 100.0).round() / 100.0) };
            for s in top.into_iter().take(n_top) {
                let f = &s.features;
                detections.push(Detection {
                    tick: s.tick,
                    algorithm: "fidoo/ml_silent_aim".to_string(),
                    player,
                    data: json!({
                        "score": score,
                        "shot_score": s.score,
                        "hits": counters.hits,
                        "weapon": s.weapon,
                        "target": s.victim,
                        "distance": f[IDX_DIST].round(),
                        "impulse_angle": round(f[IDX_IMPULSE]),
                        "out_angle": round(f[IDX_OUT]),
                        "back_angle": round(f[IDX_BACK]),
                        "return_gap_angle": round(f[IDX_SPAN]),
                        "flick_tick_offset": f[IDX_OFFSET],
                        "flick_miss_hu": round(f[IDX_ERR_B_HU]),
                        "path_miss_hu": round(f[IDX_PATH_HU]),
                        "rewind_ticks": f[IDX_K_B],
                        "typical_rewind_ticks": f[IDX_K0],
                        "backtrack_flick": f[IDX_NEAR_OK] == 0.0 && f[IDX_PATH_NEAR_OK] == 0.0 && f[IDX_K_OFF].abs() >= 3.0,
                        "jitter_ratio": round(f[IDX_RATIO]),
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

    fn update(tick: u32, yaw: f32, pitch: f32) -> Update {
        Update { tick, sim: i64::from(tick), yaw, pitch, pos: [0.0; 3], eye_height: 68.0, head_height: 68.0, ducking: false, on_ground: true }
    }

    fn path(views: &[(f32, f32)]) -> Vec<Update> {
        views.iter().enumerate().map(|(i, v)| update(100 + i as u32, v.0, v.1)).collect()
    }

    #[test]
    fn out_and_back_has_an_impulse() {
        let u = path(&[(10.0, 0.0), (10.0, 0.0), (12.0, 1.0), (10.0, 0.0), (10.0, 0.0)]);
        let t = triangle(&u, 2).unwrap();
        assert!((t.impulse - 5f32.sqrt()).abs() < 0.01, "{t:?}");
        assert!(t.exact_ret);
        assert!(t.back_cos < -0.99);
    }

    #[test]
    fn smooth_sweep_has_no_impulse() {
        let u = path(&[(10.0, 0.0), (11.0, 0.0), (12.0, 0.0), (13.0, 0.0), (14.0, 0.0)]);
        assert!(triangle(&u, 2).unwrap().impulse < 0.01);
    }

    #[test]
    fn held_snap_has_no_impulse() {
        let u = path(&[(10.0, 0.0), (10.0, 0.0), (13.0, 0.0), (13.0, 0.0), (13.0, 0.0)]);
        assert!(triangle(&u, 2).unwrap().impulse < 0.01);
    }

    #[test]
    fn sweeping_silent_shot_is_measured_off_the_sweep() {
        let u = path(&[(10.0, 0.0), (11.0, 0.0), (12.0, 2.0), (13.0, 0.0), (14.0, 0.0)]);
        let t = triangle(&u, 2).unwrap();
        assert!(t.impulse > 1.9 && t.impulse < 2.1, "{t:?}");
    }

    #[test]
    fn wide_gaps_are_not_a_path() {
        let mut u = path(&[(10.0, 0.0), (12.0, 0.0), (10.0, 0.0)]);
        u[2].tick += 20;
        u[2].sim += 20;
        assert!(triangle(&u, 1).is_none());
    }

    #[test]
    fn flick_to_a_backtracked_position_is_off_the_typical_rewind() {
        // victim walking along y: 10 units per tick, 1000 units away; A and C aim at where it is now
        let vic: Vec<Update> = (0..=120u32)
            .map(|t| {
                let mut v = update(t, 0.0, 0.0);
                v.pos = [1000.0, t as f32 * 10.0 - 1200.0, 0.0];
                v.eye_height = 68.0;
                v
            })
            .collect();
        // rewinds of the victim at the shot tick 120: y = -10 * k. B aims at k = 25.
        let yaw_b = (-250.0f32).atan2(1000.0).to_degrees();
        let u = path(&[(0.0, 0.0), (0.0, 0.0), (yaw_b, 0.0), (0.0, 0.0), (0.0, 0.0)])
            .into_iter()
            .enumerate()
            .map(|(i, mut s)| {
                s.tick = 118 + i as u32;
                s.sim = s.tick as i64;
                s
            })
            .collect::<Vec<_>>();
        let (mut f, c) = shot_features(&u, 120, Some((&vic, false)), 1, true).unwrap();
        let c = c.unwrap();
        assert_eq!(f[IDX_K_B] as i32, 25);
        fill_rewind(&mut f, &c, 2);
        assert_eq!(f[IDX_K_OFF] as i32, 23);
        assert_eq!(f[IDX_NEAR_OK], 0.0);
        assert_eq!(f[IDX_PATH_NEAR_OK], 1.0);
    }
}
