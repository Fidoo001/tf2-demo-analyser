//! Learned projectile silent aim detection. See docs/ml_psilent.md.
//!
//! A pSilent cheat writes the aim into the one usercmd that fires and chokes it (Amalgam `G::PSilentAngles`,
//! `Ticks.cpp ManagePacket`), so the following cmd restores the player's view and only that one is networked in
//! `m_angEyeAngles`. The projectile keeps the fired direction in `m_vInitialVelocity`.
//!
//! The server derives a rocket-like launch direction from the fired view only (`GetProjectileFireSetup`):
//! `forward * (d - x) - right * y - up * z`, where (x, y, z) is the muzzle offset of the weapon and d the length of
//! the eye trace (200..2000 units; a shorter trace falls back to 2000). The eye position cancels out, so the distance
//! between a view and the launch needs no position, eye height or lag correction.
//!
//! Every launch is compared with the player's networked view path around it. Lag, batched usercmds and hidden flicks
//! are not excused by rules: their measurements (update gaps, turn speed, reach) are features of the launch.
//! `examples/dump_ml_psilent_features.rs` dumps the same features for training.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::analyser::Class;
use tf_demo_parser::demo::sendprop::{SendProp, SendPropIdentifier};
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{handle_to_entid, resolve_recorder_sid, FxHashMap};

/// Ticks of view kept before and after a launch
const PRE_TICKS: i32 = 40;
const POST_TICKS: i32 = 36;
const HISTORY_TICKS: usize = (PRE_TICKS + POST_TICKS + 8) as usize;
/// Largest update gap (ticks) whose chord still counts as a view path
const MAX_CHORD_GAP: i64 = 16;
const MAX_OWNER_DISTANCE: f32 = 115.0;
/// Eye trace lengths of `GetProjectileFireSetup`: the end point is used when the trace fraction is above 0.1
const MIN_TRACE: f32 = 200.0;
const MAX_TRACE: f32 = 2000.0;
/// Random part of a pipe's launch velocity (FirePipeBomb)
const PIPE_UP: f32 = 200.0;
const PIPE_NOISE: f32 = 10.0;
/// Only launches at least this far (degrees) off the chord of the update pair around them are scored
const CANDIDATE_DEVIATION: f32 = 0.5;
/// Deviation of a launch with no measurable path
const MAX_DEVIATION: f32 = 180.0;
/// Launch spread (degrees) of projectiles whose direction comes from `m_angRotation`
const ROTATION_NOISE: f32 = 3.0;
const MIN_LAUNCH_SPEED: f32 = 300.0;
const MAX_RATIO: f32 = 50.0;
const MIN_DIVISOR: f32 = 0.25;
/// A launch key seen this many ticks ago belongs to an earlier projectile (Delete events can be missed)
const KEY_EXPIRY_TICKS: u32 = 4000;

const KIND_ROCKET: u8 = 0;
const KIND_PIPE: u8 = 1;
const KIND_STICKY: u8 = 2;
const KIND_FLARE: u8 = 3;
const KIND_ARROW: u8 = 4;
const KIND_BOLT: u8 = 5;
const KIND_ENERGY_BALL: u8 = 6;
const KIND_ENERGY_RING: u8 = 7;
const KIND_ORB: u8 = 8;
const KIND_FIREBALL: u8 = 9;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Per-launch features, in the order computed by `features`. A = last view update before the batch that delivered
/// the projectile, B = first update at or after the launch tick. Angles in degrees, gaps in sim ticks.
pub const SHOT_FEATURES: [&str; 49] = [
    "kind", "launch_pitch", "view_pitch", "dev_a", "dev_b", "dev_ab", "dev_pa", "dev_bn", "dev_tight", "dev_w5",
    "dev_w10", "dev_w18", "dev_w32", "raw_b", "span_ab", "gap_ab", "gap_pre", "gap_post", "b_delay", "out", "back",
    "back_cos", "in_cos", "ab_cos", "next_move", "v_pre", "v_post", "reach_ratio", "noise", "dev_ratio", "gap_med",
    "gap_max", "update_share", "ping_ticks", "on_ground", "ducking", "speed_h", "dvz", "owner_dist", "origin_err",
    "origin_side", "spike", "miss_launch", "miss_view", "enemy_dist", "enemy_ang_launch", "enemy_ang_view",
    "speed", "n_pre",
];

/// Player features, see `unit_features`
pub const UNIT_FEATURES: [&str; 26] = [
    "n_launches", "n_cand", "cand_share", "k3", "k5", "k7", "k9", "s5", "s9", "top1", "top2", "top3", "top5", "sum_log",
    "sum_p", "dev_med_k5", "dev_max", "k_d2", "k_d6", "s_d2", "k_w6", "s_w6", "gap_med", "gap_ab_med", "ratio_p90",
    "down_share",
];
const IDX_KIND: usize = 0;
const IDX_LAUNCH_PITCH: usize = 1;
const IDX_DEV_AB: usize = 5;
const IDX_DEV_TIGHT: usize = 8;
const IDX_DEV_W32: usize = 12;
const IDX_GAP_AB: usize = 15;
const IDX_REACH: usize = 27;
const IDX_DEV_RATIO: usize = 29;
const IDX_GAP_MED: usize = 30;

pub fn unit_feature_names() -> Vec<String> {
    UNIT_FEATURES.iter().map(|s| s.to_string()).collect()
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    tick: u32,
    sim: i64,
    yaw: f32,
    pitch: f32,
    pos: [f32; 3],
    on_ground: bool,
    ducking: bool,
    usable: bool,
}

/// A view change or a new simulation tick of the player
#[derive(Clone, Copy, Debug)]
struct ViewUpdate {
    tick: u32,
    sim: i64,
    yaw: f32,
    pitch: f32,
}

#[derive(Clone, Debug)]
pub struct LaunchFeatures {
    pub tick: u32,
    pub player: u64,
    pub weapon: String,
    pub recorder: bool,
    pub features: [f32; SHOT_FEATURES.len()],
    pub score: f32,
    /// (tick offset, sim offset, yaw - launch yaw, pitch - launch pitch), only with `MlPsilent::dump_paths`
    pub path: Vec<(i32, i32, f32, f32)>,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_launches: usize,
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
    shot: Booster,
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_psilent_model.json")).ok()?;
            let shot_names: Vec<String> = SHOT_FEATURES.iter().map(|s| s.to_string()).collect();
            (model.unit.features == unit_feature_names() && model.shot.features == shot_names).then_some(model)
        })
        .as_ref()
}

fn wrap180(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn len3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// Forward, right and up of a view (Source's AngleVectors, roll 0)
fn angle_vectors(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.to_radians().sin_cos();
    let (sp, cp) = pitch.to_radians().sin_cos();
    ([cp * cy, cp * sy, -sp], [sy, -cy, 0.0], [sp * cy, sp * sy, cp])
}

/// Source's angles of a direction (pitch positive = down)
fn vector_angles(v: [f32; 3]) -> (f32, f32) {
    (v[1].atan2(v[0]).to_degrees(), (-v[2]).atan2((v[0] * v[0] + v[1] * v[1]).sqrt()).to_degrees())
}

fn view_angle(a: (f32, f32), b: (f32, f32)) -> f32 {
    let fa = angle_vectors(a.0, a.1).0;
    let fb = angle_vectors(b.0, b.1).0;
    dot3(fa, fb).clamp(-1.0, 1.0).acos().to_degrees()
}

/// View change in mouse space: wrapped yaw scaled to arc length at the mean pitch, and pitch
fn mouse(from: (f32, f32), to: (f32, f32)) -> (f32, f32) {
    let scale = ((from.1 + to.1) * 0.5).to_radians().cos();
    (wrap180(to.0 - from.0) * scale, to.1 - from.1)
}

fn cos2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (la, lb) = (a.0.hypot(a.1), b.0.hypot(b.1));
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

/// Muzzle offset of a projectile: forward, right candidates (flipped view models, centre fire), up (standing, ducked)
fn muzzle(kind: u8) -> (f32, &'static [f32], [f32; 2]) {
    match kind {
        KIND_ROCKET => (23.5, &[12.0, -12.0, 0.0], [-3.0, 8.0]),
        KIND_FLARE => (23.5, &[12.0, -12.0], [-3.0, 8.0]),
        KIND_ARROW | KIND_BOLT => (23.5, &[-8.0, 8.0, 0.0], [-3.0, -2.0]),
        KIND_ENERGY_BALL | KIND_ENERGY_RING => (23.5, &[-8.0, 8.0], [-3.0, 8.0]),
        KIND_PIPE | KIND_STICKY => (16.0, &[8.0, -8.0], [-6.0, -6.0]),
        _ => (23.5, &[0.0], [0.0, 0.0]),
    }
}

/// Launch line of one projectile
#[derive(Clone, Copy, Debug)]
struct Launch {
    kind: u8,
    dir: [f32; 3],
    speed: f32,
    angles: (f32, f32),
}

impl Launch {
    /// Smallest angle (degrees) between the launch and any launch direction the view can produce
    fn deviation(&self, yaw: f32, pitch: f32) -> f32 {
        let (f, r, u) = angle_vectors(yaw, pitch);
        let along = dot3(self.dir, f);
        if along < 0.5 {
            return along.clamp(-1.0, 1.0).acos().to_degrees();
        }
        let p = (dot3(self.dir, r) / along, dot3(self.dir, u) / along);
        let distance = match self.kind {
            KIND_PIPE | KIND_STICKY => {
                let forward = (self.speed * self.speed - PIPE_UP * PIPE_UP).max(1.0).sqrt();
                let noise = PIPE_NOISE / forward;
                let up = PIPE_UP / forward;
                let dx = (p.0.abs() - noise).max(0.0);
                let dy = ((p.1 - up).abs() - noise).max(0.0);
                dx.hypot(dy)
            }
            // networked as a coarse rotation, spread over about +-2.5 degrees in both axes
            KIND_ORB | KIND_FIREBALL => {
                let noise = ROTATION_NOISE.to_radians().tan();
                (p.0.abs() - noise).max(0.0).hypot((p.1.abs() - noise).max(0.0))
            }
            kind => {
                let (x, sides, ups) = muzzle(kind);
                let mut best = f32::INFINITY;
                for &y in sides {
                    for z in ups {
                        let far = (-y / (MAX_TRACE - x), -z / (MAX_TRACE - x));
                        let near = (-y / (MIN_TRACE - x), -z / (MIN_TRACE - x));
                        best = best.min(point_segment(p, near, far));
                    }
                }
                best
            }
        };
        distance.atan().to_degrees()
    }

    /// Smallest deviation along the straight path between two views
    fn chord(&self, a: &ViewUpdate, b: &ViewUpdate) -> f32 {
        let (dy, dp) = (wrap180(b.yaw - a.yaw), b.pitch - a.pitch);
        let span = dy.abs().max(dp.abs());
        let steps = ((span / 0.25).ceil() as usize).clamp(1, 32);
        (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                self.deviation(a.yaw + dy * t, a.pitch + dp * t)
            })
            .fold(f32::INFINITY, f32::min)
    }
}

/// Everything the features of one launch need; samples are complete once the post-launch window has passed
#[derive(Clone, Debug)]
struct Pending {
    tick: u32,
    player: u64,
    ent: u32,
    weapon: String,
    recorder: bool,
    launch: Launch,
    origin: [f32; 3],
    /// Lead-corrected centres of the enemies in PVS at the launch
    targets: Vec<[f32; 3]>,
    ping: u16,
    view_height: f32,
}

fn gap_of(a: &ViewUpdate, b: &ViewUpdate) -> i64 {
    let sim = b.sim - a.sim;
    if (1..=200).contains(&sim) {
        sim
    } else {
        i64::from(b.tick.saturating_sub(a.tick)).max(1)
    }
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let m = values.len() / 2;
    if values.len() % 2 == 1 {
        values[m]
    } else {
        (values[m - 1] + values[m]) * 0.5
    }
}

/// Miss distance (units) of a ray from a target centre; gravity projectiles only count the horizontal miss
fn ray_miss(origin: [f32; 3], dir: [f32; 3], target: [f32; 3], gravity: bool) -> f32 {
    let (mut w, mut d) = (sub3(target, origin), dir);
    if gravity {
        w[2] = 0.0;
        d[2] = 0.0;
        let l = len3(d).max(1e-3);
        d = [d[0] / l, d[1] / l, 0.0];
    }
    let t = dot3(w, d).max(0.0);
    len3(sub3(w, [d[0] * t, d[1] * t, d[2] * t]))
}

fn features(p: &Pending, samples: &[Sample]) -> Option<([f32; SHOT_FEATURES.len()], Vec<(i32, i32, f32, f32)>)> {
    let window: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.tick + PRE_TICKS as u32 >= p.tick && s.tick <= p.tick + POST_TICKS as u32)
        .collect();
    let at_launch = window.iter().rposition(|s| s.tick <= p.tick)?;
    // only the contiguous usable stretch around the launch (death, respawn, PVS gaps end the path)
    if !window[at_launch].usable {
        return None;
    }
    let mut start = at_launch;
    while start > 0 && window[start - 1].usable {
        start -= 1;
    }
    let mut end = at_launch;
    while end + 1 < window.len() && window[end + 1].usable {
        end += 1;
    }
    let stretch = &window[start..=end];
    let mut updates: Vec<ViewUpdate> = Vec::new();
    for s in stretch {
        if let Some(last) = updates.last() {
            if last.sim == s.sim && (last.yaw - s.yaw).abs() < 0.001 && (last.pitch - s.pitch).abs() < 0.001 {
                continue;
            }
        }
        updates.push(ViewUpdate { tick: s.tick, sim: s.sim, yaw: s.yaw, pitch: s.pitch });
    }
    let b = updates.iter().position(|u| u.tick >= p.tick)?;
    if b == 0 {
        return None;
    }
    let l = &p.launch;
    let (ub, ua) = (updates[b], updates[b - 1]);
    let dev = |u: &ViewUpdate| l.deviation(u.yaw, u.pitch);
    let up = (b >= 2).then(|| updates[b - 2]);
    let un = updates.get(b + 1).copied();
    let chords: Vec<f32> = updates
        .windows(2)
        .map(|w| if gap_of(&w[0], &w[1]) <= MAX_CHORD_GAP { l.chord(&w[0], &w[1]) } else { f32::INFINITY })
        .collect();
    let dev_ab = chords[b - 1];
    let dev_pa = if b >= 2 { chords[b - 2] } else { f32::NAN };
    let dev_bn = chords.get(b).copied().unwrap_or(f32::NAN);
    let dev_tight = [dev_ab, dev_pa, dev_bn].into_iter().filter(|v| !v.is_nan()).fold(f32::INFINITY, f32::min);
    let dev_window = |n: i32| {
        updates
            .windows(2)
            .zip(&chords)
            .filter(|(w, _)| w[1].tick + n as u32 >= p.tick && w[0].tick <= p.tick + n as u32)
            .map(|(_, c)| *c)
            .fold(f32::INFINITY, f32::min)
    };
    let view = |u: &ViewUpdate| (u.yaw, u.pitch);
    let span = view_angle(view(&ua), view(&ub));
    let gap = gap_of(&ua, &ub);
    let (out, back) = (view_angle(view(&ua), l.angles), view_angle(l.angles, view(&ub)));
    let m_out = mouse(view(&ua), l.angles);
    let m_back = mouse(l.angles, view(&ub));
    let in_cos = up.map_or(f32::NAN, |u| cos2(mouse(view(&u), view(&ua)), m_out));
    let ab_cos = cos2(mouse(view(&ua), view(&ub)), m_out);
    let speeds: Vec<f32> = updates.windows(2).map(|w| view_angle(view(&w[0]), view(&w[1])) / gap_of(&w[0], &w[1]) as f32).collect();
    let v_pre = speeds[b.saturating_sub(4)..b - 1].iter().copied().fold(0.0, f32::max);
    let v_post = speeds[b..(b + 3).min(speeds.len())].iter().copied().fold(0.0, f32::max);
    let local = v_pre.max(v_post);
    let reach = if local > 1e-3 { ((out + back) / (local * gap as f32)).min(MAX_RATIO) } else { MAX_RATIO };
    let mut around: Vec<f32> = speeds
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != b - 1 && (i as i64 - b as i64).abs() <= 12)
        .map(|(_, s)| *s)
        .collect();
    let noise = median(&mut around);
    let dev_ratio = (dev_tight.min(MAX_DEVIATION) / (noise.max(0.0) * gap as f32).max(MIN_DIVISOR)).min(MAX_RATIO);
    let mut gaps: Vec<f32> = updates.windows(2).map(|w| gap_of(&w[0], &w[1]) as f32).collect();
    let gap_max = gaps.iter().copied().fold(0.0, f32::max);
    let gap_med = median(&mut gaps);
    let tick_span = stretch.last()?.tick.saturating_sub(stretch.first()?.tick).max(1);
    let update_share = updates.len().saturating_sub(1) as f32 / tick_span as f32;

    // a visible one-update flick to the launch direction (silent aim without a choked packet)
    let mut spike = 0.0f32;
    for i in 1..updates.len().saturating_sub(1) {
        if updates[i].tick.abs_diff(p.tick) > 2 || dev(&updates[i]) > 2.0 {
            continue;
        }
        let (prev, this, next) = (view(&updates[i - 1]), view(&updates[i]), view(&updates[i + 1]));
        let leg = view_angle(prev, this).min(view_angle(this, next));
        spike = spike.max(leg - view_angle(prev, next));
    }

    // movement at the launch; dvz separates rocket jumps
    let s0 = window[at_launch];
    let z_speed = |from: i64, to: i64| -> f32 {
        let pick = |t: i64| stretch.iter().find(|s| i64::from(s.tick) >= t);
        match (pick(i64::from(p.tick) + from), pick(i64::from(p.tick) + to)) {
            (Some(a), Some(b)) if b.tick > a.tick => (b.pos[2] - a.pos[2]) / (b.tick - a.tick) as f32 * 66.0,
            _ => f32::NAN,
        }
    };
    let dvz = z_speed(3, 12) - z_speed(-6, 0);
    let horizontal = stretch
        .iter()
        .find(|s| s.tick + 4 >= p.tick)
        .filter(|a| a.tick < s0.tick)
        .map_or(f32::NAN, |a| (s0.pos[0] - a.pos[0]).hypot(s0.pos[1] - a.pos[1]) / (s0.tick - a.tick) as f32 * 66.0);

    // muzzle consistency of the projectile's first origin at the launch's own view (perpendicular to the flight)
    let eye = [s0.pos[0], s0.pos[1], s0.pos[2] + if s0.ducking { 45.0 } else { p.view_height }];
    let (_, lr, lu) = angle_vectors(l.angles.0, l.angles.1);
    let to_origin = sub3(p.origin, eye);
    let origin_side = dot3(to_origin, lr);
    let (_, sides, ups) = muzzle(l.kind);
    let mut origin_err = f32::INFINITY;
    for &y in sides {
        for z in ups {
            origin_err = origin_err.min((origin_side - y).hypot(dot3(to_origin, lu) - z));
        }
    }

    // nearest enemy to the launch line and to the view
    let gravity = matches!(l.kind, KIND_PIPE | KIND_STICKY | KIND_ARROW | KIND_BOLT);
    let fb = angle_vectors(ub.yaw, ub.pitch).0;
    let best = p
        .targets
        .iter()
        .map(|t| (ray_miss(p.origin, l.dir, *t, gravity), *t))
        .min_by(|a, b| a.0.total_cmp(&b.0));
    let (miss_launch, miss_view, enemy_dist, ang_launch, ang_view) = match best {
        Some((miss, target)) => {
            let w = sub3(target, p.origin);
            let d = len3(w).max(1.0);
            let unit = [w[0] / d, w[1] / d, w[2] / d];
            let angle = |dir: [f32; 3]| {
                if gravity {
                    wrap180(vector_angles(dir).0 - vector_angles(unit).0).abs()
                } else {
                    dot3(dir, unit).clamp(-1.0, 1.0).acos().to_degrees()
                }
            };
            (miss, ray_miss(p.origin, fb, target, gravity), d, angle(l.dir), angle(fb))
        }
        None => (f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN),
    };

    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let cap = |v: f32| v.min(MAX_DEVIATION);
    let out_features = [
        f32::from(l.kind),
        l.angles.1,
        ub.pitch,
        dev(&ua),
        dev(&ub),
        cap(dev_ab),
        cap(dev_pa),
        cap(dev_bn),
        cap(dev_tight),
        cap(dev_window(5)),
        cap(dev_window(10)),
        cap(dev_window(18)),
        cap(dev_window(32)),
        view_angle(l.angles, view(&ub)),
        span,
        gap as f32,
        up.map_or(f32::NAN, |u| gap_of(&u, &ua) as f32),
        un.map_or(f32::NAN, |u| gap_of(&ub, &u) as f32),
        ub.tick.saturating_sub(p.tick) as f32,
        out,
        back,
        cos2(m_out, m_back),
        in_cos,
        ab_cos,
        un.map_or(f32::NAN, |u| view_angle(view(&ub), view(&u))),
        v_pre,
        v_post,
        reach,
        noise,
        dev_ratio,
        gap_med,
        gap_max,
        update_share,
        (f32::from(p.ping) / 15.0).round(),
        flag(s0.on_ground),
        flag(s0.ducking),
        horizontal,
        dvz,
        len3(to_origin),
        origin_err,
        origin_side,
        spike,
        miss_launch,
        miss_view,
        enemy_dist,
        ang_launch,
        ang_view,
        l.speed,
        b as f32,
    ];
    let path = updates
        .iter()
        .map(|u| (u.tick as i32 - p.tick as i32, (u.sim - ub.sim) as i32, wrap180(u.yaw - l.angles.0), u.pitch - l.angles.1))
        .collect();
    Some((out_features, path))
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

/// Features of one player from their `n_launches` launches, of which `candidates` were scored.
/// scripts/ml_psilent_train.py has a copy (`unit_row`), which trains on out-of-fold launch scores.
fn unit_features(candidates: &[&LaunchFeatures], n_launches: usize) -> Vec<f32> {
    let n = n_launches as f32;
    let mut p: Vec<f32> = candidates.iter().map(|c| c.score).collect();
    p.sort_by(|a, b| b.total_cmp(a));
    let k = |t: f32| p.iter().filter(|v| **v >= t).count() as f32;
    let top = |m: usize| {
        let m = m.min(p.len());
        p[..m].iter().sum::<f32>() / m.max(1) as f32
    };
    let column = |i: usize| candidates.iter().map(|c| c.features[i]).collect::<Vec<f32>>();
    let dev = column(IDX_DEV_TIGHT);
    let k_d = |t: f32| dev.iter().filter(|v| **v >= t).count() as f32;
    let k_w6 = column(IDX_DEV_W32).iter().filter(|v| **v >= 6.0).count() as f32;
    let mut strong: Vec<f32> = candidates.iter().filter(|c| c.score >= 0.5).map(|c| c.features[IDX_DEV_TIGHT]).collect();
    let down = column(IDX_LAUNCH_PITCH).iter().filter(|v| **v >= 45.0).count() as f32;
    let sum_log: f64 = p.iter().map(|v| -(1.0 - f64::from(v.min(0.999))).log10()).sum();
    vec![
        n,
        p.len() as f32,
        p.len() as f32 / n,
        k(0.3),
        k(0.5),
        k(0.7),
        k(0.9),
        k(0.5) / n,
        k(0.9) / n,
        top(1),
        top(2),
        top(3),
        top(5),
        sum_log as f32,
        p.iter().sum(),
        quantile(&mut strong, 0.5),
        dev.iter().copied().fold(0.0, f32::max),
        k_d(2.0),
        k_d(6.0),
        k_d(2.0) / n,
        k_w6,
        k_w6 / n,
        quantile(&mut column(IDX_GAP_MED), 0.5),
        quantile(&mut column(IDX_GAP_AB), 0.5),
        quantile(&mut column(IDX_DEV_RATIO), 0.9),
        down / p.len().max(1) as f32,
    ]
}

fn kind_name(kind: f32) -> &'static str {
    match kind as u8 {
        KIND_ROCKET => "rocket",
        KIND_PIPE => "pipe",
        KIND_STICKY => "sticky",
        KIND_FLARE => "flare",
        KIND_ARROW => "arrow",
        KIND_BOLT => "bolt",
        KIND_ENERGY_BALL => "energy_ball",
        KIND_ENERGY_RING => "energy_ring",
        KIND_ORB => "orb",
        _ => "fireball",
    }
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

fn prop_vector(props: &[SendProp], ids: &[(&str, &str)]) -> Option<Vector> {
    ids.iter().find_map(|(table, name)| {
        let id = SendPropIdentifier::new(table, name);
        props.iter().find(|p| p.identifier == id).and_then(|p| Vector::try_from(&p.value).ok())
    })
}

fn prop_int(props: &[SendProp], ids: &[(&str, &str)]) -> Option<i64> {
    ids.iter().find_map(|(table, name)| {
        let id = SendPropIdentifier::new(table, name);
        props.iter().find(|p| p.identifier == id).and_then(|p| i64::try_from(&p.value).ok())
    })
}

fn projectile_kind(class: &str, props: &[SendProp]) -> Option<u8> {
    Some(match class {
        "CTFProjectile_Rocket" => KIND_ROCKET,
        "CTFGrenadePipebombProjectile" => match prop_int(props, &[("DT_TFProjectile_Pipebomb", "m_iType")]) {
            Some(1) | Some(2) => KIND_STICKY,
            _ => KIND_PIPE,
        },
        "CTFProjectile_Flare" => KIND_FLARE,
        "CTFProjectile_Arrow" => KIND_ARROW,
        "CTFProjectile_HealingBolt" => KIND_BOLT,
        "CTFProjectile_EnergyBall" => KIND_ENERGY_BALL,
        "CTFProjectile_EnergyRing" => KIND_ENERGY_RING,
        "CTFProjectile_MechanicalArmOrb" => KIND_ORB,
        "CTFProjectile_BallOfFire" => KIND_FIREBALL,
        _ => return None,
    })
}

fn launch_velocity(kind: u8, props: &[SendProp]) -> Option<[f32; 3]> {
    if !matches!(kind, KIND_ORB | KIND_FIREBALL) {
        let v = prop_vector(props, &[("DT_TFBaseRocket", "m_vInitialVelocity"), ("DT_TFWeaponBaseGrenadeProj", "m_vInitialVelocity")]);
        if let Some(v) = v {
            let v = [v.x, v.y, v.z];
            if len3(v) > 1.0 {
                return Some(v);
            }
        }
    }
    let rot = prop_vector(props, &[("DT_TFBaseRocket", "m_angRotation"), ("DT_BaseEntity", "m_angRotation")])?;
    let f = angle_vectors(rot.y, rot.x).0;
    let speed = if kind == KIND_FIREBALL { 3000.0 } else { 1100.0 };
    Some([f[0] * speed, f[1] * speed, f[2] * speed])
}

fn owner_of(props: &[SendProp], state: &CheatAnalyserState) -> Option<u32> {
    let thrower = prop_int(props, &[("DT_BaseGrenade", "m_hThrower")]).map(|h| u32::from(handle_to_entid(h as u32)));
    let owner = prop_int(props, &[("DT_BaseEntity", "m_hOwnerEntity")]).map(|h| u32::from(handle_to_entid(h as u32)));
    [thrower, owner].into_iter().flatten().find(|&e| state.players.iter().any(|p| u32::from(p.entity) == e))
}

fn launcher_of(props: &[SendProp]) -> Option<EntityId> {
    let handle = prop_int(
        props,
        &[("DT_TFProjectile_Pipebomb", "m_hLauncher"), ("DT_TFBaseRocket", "m_hLauncher"), ("DT_BaseProjectile", "m_hOriginalLauncher")],
    )?;
    let ent = handle_to_entid(handle as u32);
    (u32::from(ent) != 0 && u32::from(ent) != 0x7FF).then_some(ent)
}

/// Projectiles of another class are reflected or not this player's launch
fn class_fires(class: Class, kind: u8) -> bool {
    match kind {
        KIND_ROCKET | KIND_ENERGY_BALL => class == Class::Soldier,
        KIND_PIPE | KIND_STICKY => class == Class::Demoman,
        KIND_FLARE | KIND_FIREBALL => class == Class::Pyro,
        KIND_ARROW => class == Class::Sniper,
        KIND_BOLT => class == Class::Medic,
        KIND_ENERGY_RING => matches!(class, Class::Soldier | Class::Engineer),
        KIND_ORB => class == Class::Engineer,
        _ => false,
    }
}

fn is_excluded_weapon(name: &str) -> bool {
    name.contains("Beggar") || name.contains("Rocket Jumper") || name.contains("Sticky Jumper")
}

fn view_height(player: &Player) -> f32 {
    match player.class {
        Class::Scout => 65.0,
        Class::Sniper | Class::Medic | Class::Spy => 75.0,
        _ => 68.0,
    }
}

pub struct MlPsilent {
    params: Parameters,
    history: FxHashMap<u32, VecDeque<Sample>>,
    velocity: FxHashMap<u32, [f32; 3]>,
    seen: FxHashMap<(u32, u32), u32>,
    pending: Vec<Pending>,
    recorder: Option<u64>,
    /// Launch features of the demo, filled while parsing (`score` and the units by `finish`)
    pub launches: Vec<LaunchFeatures>,
    /// Scored players, filled by `finish`
    pub units: Vec<UnitRecord>,
    /// Fill `LaunchFeatures::path` (research dumps)
    pub dump_paths: bool,
}

impl Default for MlPsilent {
    fn default() -> Self {
        Self::new()
    }
}

impl MlPsilent {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_score".to_string(), Parameter::Float(0.7)),
                ("min_launches".to_string(), Parameter::Int(1)),
                ("min_shot_score".to_string(), Parameter::Float(0.5)),
                ("max_detections".to_string(), Parameter::Int(20)),
            ]),
            history: FxHashMap::default(),
            velocity: FxHashMap::default(),
            seen: FxHashMap::default(),
            pending: Vec::new(),
            recorder: None,
            launches: Vec::new(),
            units: Vec::new(),
            dump_paths: false,
        }
    }

    fn evaluate(&mut self, p: Pending) {
        let Some(history) = self.history.get(&p.ent) else { return };
        let samples: Vec<Sample> = history.iter().copied().collect();
        let Some((features, path)) = features(&p, &samples) else { return };
        self.launches.push(LaunchFeatures {
            tick: p.tick,
            player: p.player,
            weapon: p.weapon,
            recorder: p.recorder,
            features,
            score: f32::NAN,
            path: if self.dump_paths { path } else { Vec::new() },
        });
    }

    fn capture(&mut self, props: &[SendProp], class: &str, key: (u32, u32), state: &CheatAnalyserState) {
        let tick = u32::from(state.tick);
        let Some(kind) = projectile_kind(class, props) else { return };
        let Some(owner_ent) = owner_of(props, state) else { return };
        let Some(player) = state.players.iter().find(|p| u32::from(p.entity) == owner_ent) else { return };
        if !class_fires(player.class, kind) {
            return;
        }
        let Some(sid) = player_sid(player, state) else { return };
        if !player.in_pvs || player.state != PlayerState::Alive || player.is_taunting() || player.cond & (1 << 7) != 0 {
            return;
        }
        if prop_int(props, &[("DT_TFBaseRocket", "m_iDeflected"), ("DT_TFWeaponBaseGrenadeProj", "m_iDeflected")]).is_some_and(|d| d > 0)
            || prop_int(props, &[("DT_TFProjectile_Pipebomb", "m_bTouched")]).is_some_and(|t| t > 0)
        {
            return;
        }
        // the launcher must be in hand: a projectile laid earlier can enter the PVS after a weapon switch
        if let Some(launcher) = launcher_of(props) {
            if player.active_weapon != Some(launcher) {
                return;
            }
        }
        let weapon = state.get_player_weapon(player);
        if is_excluded_weapon(&weapon) {
            return;
        }
        let Some(origin) = prop_vector(props, &[("DT_TFBaseRocket", "m_vecOrigin"), ("DT_TFWeaponBaseGrenadeProj", "m_vecOrigin"), ("DT_BaseEntity", "m_vecOrigin")]) else {
            return;
        };
        let origin = [origin.x, origin.y, origin.z];
        if len3(sub3(origin, [player.position.x, player.position.y, player.position.z])) > MAX_OWNER_DISTANCE {
            return;
        }
        let Some(velocity) = launch_velocity(kind, props) else { return };
        let speed = len3(velocity);
        if speed < MIN_LAUNCH_SPEED {
            return;
        }
        let dir = [velocity[0] / speed, velocity[1] / speed, velocity[2] / speed];
        // a key seen recently: the projectile re-entered the PVS
        if self.seen.get(&key).is_some_and(|&first| tick.saturating_sub(first) < KEY_EXPIRY_TICKS) {
            return;
        }
        self.seen.insert(key, tick);

        let targets = state
            .players
            .iter()
            .filter(|e| e.team != player.team && e.state == PlayerState::Alive && e.in_pvs && u32::from(e.entity) != owner_ent)
            .map(|e| {
                let pos = [e.position.x, e.position.y, e.position.z + 40.0];
                let flight = len3(sub3(pos, origin)) / speed.max(1.0);
                let v = self.velocity.get(&u32::from(e.entity)).copied().unwrap_or([0.0; 3]);
                [pos[0] + v[0] * flight, pos[1] + v[1] * flight, pos[2] + v[2] * flight]
            })
            .collect();
        self.pending.push(Pending {
            tick,
            player: sid,
            ent: owner_ent,
            weapon,
            recorder: self.recorder == Some(sid),
            launch: Launch { kind, dir, speed, angles: vector_angles(dir) },
            origin,
            targets,
            ping: player.ping,
            view_height: view_height(player),
        });
    }
}

impl<'a> CheatAlgorithm<'a> for MlPsilent {
    fn algorithm_name(&self) -> &str {
        "fidoo/ml_psilent"
    }

    fn tooltip(&self) -> &'static str {
        "Scores each player's projectile launches over the whole demo with a learned model of projectile silent aim: a launch direction that no view on the player's networked view path can produce, accounting for the muzzle offset, with update gaps and turn speed deciding what lag or a hidden flick can explain."
    }

    // Never hidden by the global sliding window: aggregates its evidence per player over the whole demo
    fn excluded_from_sliding_window(&self) -> bool {
        true
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    /// The scores aggregate all launches of the demo
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::PacketEntities])
    }

    fn init(&mut self) -> Result<(), Error> {
        let params = std::mem::take(&mut self.params);
        let dump_paths = self.dump_paths;
        *self = Self::new();
        self.params = params;
        self.dump_paths = dump_paths;
        Ok(())
    }

    fn on_message(&mut self, message: &Message, state: &CheatAnalyserState, parser_state: &ParserState, _: DemoTick) -> Result<Vec<Detection>, Error> {
        let Message::PacketEntities(message) = message else {
            return Ok(vec![]);
        };
        if self.recorder.is_none() {
            self.recorder = resolve_recorder_sid(state);
        }
        for entity in &message.entities {
            let key = (u32::from(entity.entity_index), entity.serial_number);
            match entity.update_type {
                UpdateType::Delete => {
                    self.seen.remove(&key);
                }
                UpdateType::Enter => {
                    let Some(class) = parser_state.server_classes.get(usize::from(entity.server_class)).map(|c| c.name.as_str()) else {
                        continue;
                    };
                    if !class.starts_with("CTF") || !class.contains("Projectile") {
                        continue;
                    }
                    let props: Vec<_> = entity.props(parser_state).collect();
                    self.capture(&props, class, key, state);
                }
                _ => {}
            }
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        let tick = u32::from(state.tick);
        for p in &state.players {
            let ent = u32::from(p.entity);
            let hist = self.history.entry(ent).or_default();
            if hist.back().is_some_and(|s| s.tick > tick) {
                hist.clear(); // signon packets carry unrelated ticks
            }
            if hist.back().is_some_and(|s| s.tick == tick) {
                continue;
            }
            let pos = [p.position.x, p.position.y, p.position.z];
            let pitch = if p.pitch_angle > 180.0 { p.pitch_angle - 360.0 } else { p.pitch_angle };
            if let Some(old) = hist.iter().rev().nth(3) {
                let dt = tick.saturating_sub(old.tick).max(1) as f32 / 66.0;
                self.velocity.insert(ent, [(pos[0] - old.pos[0]) / dt, (pos[1] - old.pos[1]) / dt, (pos[2] - old.pos[2]) / dt]);
            }
            hist.push_back(Sample {
                tick,
                sim: state.sim_tick(p),
                yaw: p.view_angle,
                pitch,
                pos,
                on_ground: p.is_on_ground(),
                ducking: p.is_ducking(),
                usable: p.state == PlayerState::Alive && p.in_pvs && !p.is_taunting() && p.cond & (1 << 7) == 0,
            });
            while hist.len() > HISTORY_TICKS {
                hist.pop_front();
            }
        }
        if !self.pending.is_empty() {
            let (ready, rest): (Vec<Pending>, Vec<Pending>) =
                std::mem::take(&mut self.pending).into_iter().partition(|p| p.tick + POST_TICKS as u32 <= tick);
            self.pending = rest;
            for p in ready {
                self.evaluate(p);
            }
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        for p in std::mem::take(&mut self.pending) {
            self.evaluate(p);
        }
        let min_score = get_parameter_value::<f32>(&self.params, "min_score");
        let min_launches = get_parameter_value::<i32>(&self.params, "min_launches").max(1) as usize;
        let min_shot_score = get_parameter_value::<f32>(&self.params, "min_shot_score");
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let model = model();
        for launch in &mut self.launches {
            if launch.features[IDX_DEV_AB] >= CANDIDATE_DEVIATION {
                launch.score = model.map_or(f32::NAN, |m| m.shot.probability(&launch.features));
            }
        }
        let mut players: Vec<u64> = self.launches.iter().map(|l| l.player).collect();
        players.sort_unstable();
        players.dedup();
        let mut detections = Vec::new();
        for player in players {
            let mine: Vec<&LaunchFeatures> = self.launches.iter().filter(|l| l.player == player).collect();
            let candidates: Vec<&LaunchFeatures> = mine.iter().copied().filter(|l| l.features[IDX_DEV_AB] >= CANDIDATE_DEVIATION).collect();
            if candidates.is_empty() || mine.len() < min_launches {
                continue;
            }
            let features = unit_features(&candidates, mine.len());
            let score = model.map_or(f32::NAN, |m| m.unit.probability(&features));
            self.units.push(UnitRecord { player, n_launches: mine.len(), features, score });
            if !(score >= min_score) {
                continue;
            }
            // every launch the launch model finds silent-aim-like (at least the most suspicious one)
            let mut top = candidates.clone();
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            let n_top = top.iter().filter(|l| l.score >= min_shot_score).count().clamp(1, max_detections);
            let round = |v: f32| if v.is_nan() { None } else { Some((v * 100.0).round() / 100.0) };
            for l in top.into_iter().take(n_top) {
                let f = &l.features;
                detections.push(Detection {
                    tick: l.tick,
                    algorithm: "fidoo/ml_psilent".to_string(),
                    player,
                    data: json!({
                        "score": score,
                        "launch_score": l.score,
                        "launches": mine.len(),
                        "weapon": l.weapon,
                        "projectile": kind_name(f[IDX_KIND]),
                        "deviation": round(f[IDX_DEV_TIGHT]),
                        "update_pair_deviation": round(f[IDX_DEV_AB]),
                        "wide_window_deviation": round(f[IDX_DEV_W32]),
                        "launch_pitch": round(f[IDX_LAUNCH_PITCH]),
                        "update_gap": f[IDX_GAP_AB],
                        "typical_update_gap": round(f[IDX_GAP_MED]),
                        "reach_ratio": round(f[IDX_REACH]),
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

    fn launch_from_view(kind: u8, yaw: f32, pitch: f32, trace: f32) -> Launch {
        let (f, r, u) = angle_vectors(yaw, pitch);
        let (x, sides, ups) = muzzle(kind);
        let (y, z) = (sides[0], ups[0]);
        let d = trace - x;
        let v = [f[0] * d - r[0] * y - u[0] * z, f[1] * d - r[1] * y - u[1] * z, f[2] * d - r[2] * y - u[2] * z];
        let l = len3(v);
        let dir = [v[0] / l, v[1] / l, v[2] / l];
        Launch { kind, dir, speed: 1100.0, angles: vector_angles(dir) }
    }

    #[test]
    fn deviation_is_zero_for_a_launch_from_the_view() {
        for trace in [200.0, 700.0, 2000.0] {
            let launch = launch_from_view(KIND_ROCKET, 40.0, 25.0, trace);
            assert!(launch.deviation(40.0, 25.0) < 0.01, "{trace}");
        }
    }

    #[test]
    fn deviation_measures_a_view_off_the_launch() {
        let launch = launch_from_view(KIND_ROCKET, 40.0, 0.0, 2000.0);
        let dev = launch.deviation(50.0, 0.0);
        assert!(dev > 5.0 && dev < 11.0, "{dev}");
    }

    #[test]
    fn pipe_deviation_removes_the_upward_boost() {
        let (f, _, u) = angle_vectors(10.0, -5.0);
        let v = [f[0] * 1200.0 + u[0] * 200.0, f[1] * 1200.0 + u[1] * 200.0, f[2] * 1200.0 + u[2] * 200.0];
        let speed = len3(v);
        let dir = [v[0] / speed, v[1] / speed, v[2] / speed];
        let launch = Launch { kind: KIND_PIPE, dir, speed, angles: vector_angles(dir) };
        assert!(launch.deviation(10.0, -5.0) < 0.01);
        assert!(launch.deviation(10.0, 5.0) > 5.0);
    }

    #[test]
    fn chord_finds_a_launch_passed_by_the_view() {
        let launch = launch_from_view(KIND_ROCKET, 40.0, 0.0, 2000.0);
        let update = |yaw: f32| ViewUpdate { tick: 0, sim: 0, yaw, pitch: 0.0 };
        assert!(launch.chord(&update(30.0), &update(50.0)) < 0.1);
        assert!(launch.chord(&update(50.0), &update(60.0)) > 5.0);
    }

    #[test]
    fn quantile_matches_numpy() {
        assert_eq!(quantile(&mut [1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
        assert!(quantile(&mut [], 0.5).is_nan());
    }
}
