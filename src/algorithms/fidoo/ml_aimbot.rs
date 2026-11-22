//! Learned aimbot detection on spectated players' hitscan hits. See docs/ml_aimbot.md.
//!
//! Every hitscan hit on an enemy keeps the attacker's eye samples from 1 s before to 0.25 s after it and
//! the victim's positions. Per hit, the aim error to the victim's body column is computed at the best
//! lag compensation offset, together with snap (step size, stillness, jerk), tracking and approach-shape
//! features. Single hits are too coarse at the networked angle resolution, so the hits of a player are
//! aggregated over the demo (median / 10th / 90th percentile) and scored by a gradient boosted tree
//! model (`assets/ml_aimbot_model.json`, trained with LightGBM on the demo database). Players at or
//! above `min_score` are reported at their highest scoring hits (a second, per-hit model).
//!
//! `examples/dump_ml_aimbot_features.rs` dumps the same features for training.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::{MessageType, ParserState};

use crate::util::helpers::resolve_recorder_sid;
use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;
use crate::util::nocrex::jankguard::JankGuard;

/// Attacker samples kept before / after the hit tick
const WINDOW_BEFORE: u32 = 66;
const WINDOW_AFTER: u32 = 16;
/// Samples before the hit searched for the acquisition (how the aim got onto the target). Aimbots that wait for a
/// headshot or for charge (Amalgam `ShouldFire`) snap on and hold the target, often more than a second before the hit.
const ACQ_WINDOW_BEFORE: u32 = 198;
/// Largest lag compensation offset tried, in ticks
const MAX_LAG: usize = 24;
const HISTORY_TICKS: u32 = ACQ_WINDOW_BEFORE + MAX_LAG as u32 + WINDOW_AFTER + 8;
/// The error before the acquiring update is the largest one in this many ticks before it (a snap over two updates)
const ACQ_FROM_TICKS: i64 = 8;
/// The hold only ends where the error leaves this multiple of the on-target limit, so a tracking wobble just outside
/// the limit is not taken for the acquisition
const ACQ_EXIT: f64 = 2.5;
/// Hits of one attacker closer than this are merged into the first one (pellets, sprays)
const MIN_HIT_SPACING: u32 = 8;
const MIN_WINDOW_SAMPLES: usize = 8;
/// Hits right after a spawn / teleport, at point blank or across merged updates are not used
const MIN_SINCE_SPAWN: u32 = 60;
const MIN_DISTANCE: f32 = 50.0;
const MAX_SAMPLE_GAP: f32 = 3.0;
/// Half width of the body used for the target's angular size
const BODY_HALF_WIDTH: f32 = 24.0;
const BODY_BOTTOM: f32 = 5.0;
const BODY_TOP_STANDING: f32 = 80.0;
const BODY_TOP_DUCKING: f32 = 60.0;
/// Floor for step sizes in ratios (about half the yaw quantization)
const MIN_STEP: f32 = 0.18;
const FL_DUCKING: u32 = 1 << 1;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

/// Per-hit features, in the order of `HIT_FEATURES`
pub const HIT_FEATURES: [&str; 38] = [
    "err0", "err0_rel", "err0_hu", "err_m1", "err_m2", "err_m4", "err_m8", "err_p1", "err_p2", "err_p3",
    "best_lag_gain", "step_hit", "pre_med", "stillness", "max_step_10", "post_step", "post_ratio", "t_on",
    "pre_err_max", "jerk", "trk_resid", "trk_tgt", "trk_ratio", "appr_len", "appr_drop", "appr_ratio_med",
    "appr_ratio_mad", "appr_mono", "appr_step_cv", "acq_from", "acq_step", "acq_frac", "acq_land_rel", "acq_t",
    "hold_err_rel", "hold_resid", "dist", "lag",
];
/// Hit features measured on the near-hit window (`WINDOW_BEFORE`)
const NEAR: usize = 29;
/// Acquisition features (`acquisition_features`, on `ACQ_WINDOW_BEFORE`)
const ACQ: usize = 7;
/// Hit features that are aggregated into the player features (all but the context `dist` and `lag`)
const AGGREGATED: usize = NEAR + ACQ;
pub const IDX_ERR0_REL: usize = 1;
pub const IDX_STEP_HIT: usize = 11;
pub const IDX_STILLNESS: usize = 13;
pub const IDX_JERK: usize = 19;
pub const IDX_ACQ_STEP: usize = 30;
pub const IDX_ACQ_T: usize = 33;
pub const IDX_DIST: usize = AGGREGATED;
pub const IDX_LAG: usize = AGGREGATED + 1;

/// Player feature names: `<hit feature>_med`, `_p10`, `_p90` for the aggregated hit features, then the
/// snap and precise shares
pub fn unit_feature_names() -> Vec<String> {
    let mut names = Vec::new();
    for suffix in ["med", "p10", "p90"] {
        names.extend(HIT_FEATURES[..AGGREGATED].iter().map(|f| format!("{f}_{suffix}")));
    }
    names.push("snap_share".to_string());
    names.push("precise_share".to_string());
    names
}

#[derive(Clone, Copy)]
struct Sample {
    tick: u32,
    pos: [f32; 3],
    eye: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
}

struct PendingHit {
    tick: u32,
    attacker: u64,
    victim: u64,
    weapon: String,
    since_spawn: u32,
}

#[derive(Clone, Debug)]
pub struct HitRecord {
    pub tick: u32,
    pub attacker: u64,
    pub victim: u64,
    pub weapon: String,
    pub features: [f32; HIT_FEATURES.len()],
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_hits: usize,
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
    hit: Booster,
    /// Evidence weighting (scripts/ml_aimbot_train.py `shrink`): the player score is pulled towards `prior` on the
    /// logit scale by min(1, hits / full_hits), so a few aimbot-like hits cannot give a high score
    #[serde(default)]
    prior: Option<f64>,
    #[serde(default)]
    full_hits: Option<f64>,
}

impl Model {
    fn player_score(&self, features: &[f32], hits: usize) -> f32 {
        let p = f64::from(self.unit.probability(features));
        let (Some(prior), Some(full)) = (self.prior, self.full_hits.filter(|f| *f > 0.0)) else { return p as f32 };
        let logit = |q: f64| {
            let q = q.clamp(1e-9, 1.0 - 1e-9);
            (q / (1.0 - q)).ln()
        };
        let z0 = logit(prior);
        let z = z0 + (logit(p) - z0) * (hits as f64 / full).min(1.0);
        (1.0 / (1.0 + (-z).exp())) as f32
    }
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_aimbot_model.json")).ok()?;
            let hit_names: Vec<String> = HIT_FEATURES.iter().map(|s| s.to_string()).collect();
            (model.unit.features == unit_feature_names() && model.hit.features == hit_names).then_some(model)
        })
        .as_ref()
}

/// Per-hit model probability of a hit's features (`HitRecord::features`), `None` without a model. fidoo/plain_aimbot
/// uses it to flag single hits.
pub fn hit_probability(features: &[f32]) -> Option<f32> {
    model().map(|m| m.hit.probability(features))
}

/// Same exclusions as fidoo/plain_aimbot (excluded and projectile weapons) plus melee and other non-guns
pub fn is_hitscan(name: &str) -> bool {
    let lw = name.to_lowercase();
    ![
        "builder", "pda", "toolbox", "destruction", "disguise", "sapper", "red-tape", "invis watch",
        "cloak and dagger", "dead ringer", "wrangler", "giger counter", "medi gun", "medigun", "kritzkrieg",
        "quick-fix", "vaccinator", "jumper", "jarate", "mad milk", "gas passer", "mutated milk", "self-aware",
        "grenade launcher", "iron bomber", "loch-n-load", "loose cannon", "stickybomb", "scottish resistance",
        "quickiebomb", "flame", "degreaser", "backburner", "phlog", "rainblower", "nostromo", "dragon's fury",
        "crossbow", "direct hit", "liberty launcher", "rocket launcher", "black box", "air strike", "beggar",
        "mangler", "original", "righteous bison", "pomson", "manmelter", "flare gun", "detonator", "scorch shot",
        "rescue ranger", "huntsman", "fortified compound", "syringe gun", "blutsauger", "overdose",
        "knife", "wrench", "bat", "bottle", "fist", "gloves", "shovel", "axe", "kukri", "bushwacka", "saw",
        "eyelander", "katana", "pan", "sword", "skullcutter", "equalizer", "escape plan", "jag", "gunslinger",
        "spy-cicle", "eternal reward", "kunai", "sharp dresser", "powerjack", "third degree", "candy cane",
        "sandman", "ubersaw", "vita-saw", "amputator", "persuader", "caber", "zatoichi", "claidheamh",
        "market gardener", "disciplinary", "holiday punch", "eviction notice", "warrior", "big earner",
        "wrap assassin", "atomizer", "fan o'war", "unarmed combat", "homewrecker", "maul", "neon annihilator",
        "postal pummeler", "back scratcher", "lollichop", "ham shank", "conscientious", "freedom staff",
        "memory maker", "black rose", "bread bite", "necro smasher", "crossing guard", "prinny machete",
        "nessie", "golf club", "saxxy", "solemn vow", "southern hospitality", "boston basher", "sun-on-a-stick",
        "scotsman", "tribalman", "shahanshah", "melee", "lunchbox", "sandvich", "banana", "chocolate",
        "steak", "cleaver", "guillotine", "jar", "drink", "bonk", "cola", "milk", "gas", "banner", "backup",
        "concheror", "battalion", "parachute", "shield", "targe", "booties", "bootlegger", "razorback",
        "darwin", "cozy", "spellbook", "grappling", "passtime", "contract", "tool", "wearable", "unknown",
    ]
    .iter()
    .any(|w| lw.contains(w))
}

fn view_height(player: &Player, ducking: bool) -> f32 {
    if ducking {
        return 45.0;
    }
    match player.class_name() {
        "scout" => 65.0,
        "sniper" | "medic" | "heavy" | "spy" => 75.0,
        _ => 68.0,
    }
}

fn player_sid(player: &Player) -> Option<u64> {
    let info = player.info.as_ref()?;
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

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

/// Aim error (degrees) from `eye` along `d` to the victim's body column at `v`, and the aimed point relative to the eye
fn column_error(eye: V3, d: V3, v: &Sample) -> (f64, V3) {
    let vpos = v.pos.map(f64::from);
    let top = if v.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING } as f64;
    let rel = sub(vpos, eye);
    let dh = (rel[0] * rel[0] + rel[1] * rel[1]).sqrt();
    let dxy = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
    let aim_z = eye[2] + d[2] / dxy * dh;
    let tz = aim_z.clamp(vpos[2] + f64::from(BODY_BOTTOM), vpos[2] + top);
    let t = [vpos[0] - eye[0], vpos[1] - eye[1], tz - eye[2]];
    (angle_between(d, t), t)
}

/// The victim sample at `q` ticks relative to the hit (the last one at or before it)
fn victim_at<'v>(vic: &'v [Sample], vdt: &[i64], q: i64) -> &'v Sample {
    &vic[vdt.partition_point(|&t| t <= q).saturating_sub(1)]
}

/// Near-hit features of one hit (the first `NEAR` of `HIT_FEATURES`, then `dist` and `lag`) from the attacker's eye
/// samples and the victim's position samples, or `None` when the window is unusable. The attacker samples are ordered
/// by tick, `hit_tick` is the hurt tick.
fn hit_features(att: &[Sample], vic: &[Sample], hit_tick: u32) -> Option<[f32; NEAR + 2]> {
    let n = att.len();
    let adt: Vec<i64> = att.iter().map(|s| i64::from(s.tick) - i64::from(hit_tick)).collect();
    let h = adt.iter().rposition(|&t| t <= 0)?;
    if h < 4 || vic.len() < 4 {
        return None;
    }
    let vdt: Vec<i64> = vic.iter().map(|s| i64::from(s.tick) - i64::from(hit_tick)).collect();
    let dirs: Vec<V3> = att.iter().map(|s| view_dir(s.yaw, s.pitch)).collect();
    let eyes: Vec<V3> = att.iter().map(|s| s.eye.map(f64::from)).collect();

    // aim error to the victim's body column for every lag and sample
    let mut err = vec![vec![0.0f64; n]; MAX_LAG + 1];
    let mut tgt = vec![vec![[0.0f64; 3]; n]; MAX_LAG + 1];
    for lag in 0..=MAX_LAG {
        for i in 0..n {
            let (e, t) = column_error(eyes[i], dirs[i], victim_at(vic, &vdt, adt[i] - lag as i64));
            err[lag][i] = e;
            tgt[lag][i] = t;
        }
    }
    // best lag: smallest mean error over the samples leading into the hit
    let lo = h.saturating_sub(3);
    let means: Vec<f64> = err.iter().map(|e| e[lo..=h].iter().sum::<f64>() / (h + 1 - lo) as f64).collect();
    let k = (0..=MAX_LAG).min_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap()).unwrap();
    let best_lag_gain = means.iter().cloned().fold(f64::MIN, f64::max) - means[k];
    let e = &err[k];
    let dist = norm(tgt[k][h]);
    if dist < 1.0 {
        return None;
    }
    let ang_size = f64::from(BODY_HALF_WIDTH).atan2(dist).to_degrees();

    let gaps: Vec<f64> = adt.windows(2).map(|w| (w[1] - w[0]).max(1) as f64).collect();
    let rate: Vec<f64> = (0..n - 1).map(|i| angle_between(dirs[i + 1], dirs[i]) / gaps[i]).collect();
    let tdir: Vec<V3> = tgt[k].iter().map(|t| {
        let l = norm(*t).max(1e-9);
        [t[0] / l, t[1] / l, t[2] / l]
    }).collect();
    let tsteps: Vec<f64> = (0..n - 1).map(|i| angle_between(tdir[i + 1], tdir[i]) / gaps[i]).collect();
    // view motion not explained by the target's motion
    let resid: Vec<f64> = (0..n - 1)
        .map(|i| {
            let dv = sub(dirs[i + 1], dirs[i]);
            let dt = sub(tdir[i + 1], tdir[i]);
            norm(sub(dv, dt)).to_degrees() / gaps[i]
        })
        .collect();

    let at = |arr: &[f64], i: isize| if i >= 0 && (i as usize) < arr.len() { arr[i as usize] } else { f64::NAN };
    let hi = h as isize;
    let pre = &rate[h.saturating_sub(12)..h - 1];
    let pre_med = if pre.is_empty() { f64::NAN } else { median(pre) };
    let step_hit = rate[h - 1];
    let max_step_10 = rate[h.saturating_sub(10)..h].iter().cloned().fold(f64::MIN, f64::max);
    let post = &rate[h.min(rate.len())..(h + 3).min(rate.len())];
    let post_step = if post.is_empty() { f64::NAN } else { post.iter().cloned().fold(f64::MIN, f64::max) };
    // time on target before the hit
    let on_limit = ang_size.max(1.0);
    let mut i = hi;
    while i >= 0 && e[i as usize] <= on_limit {
        i -= 1;
    }
    let t_on = if i < hi { (adt[h] - adt[(i + 1) as usize]) as f64 } else { 0.0 };
    let pre_err_max = e[..((i + 1).max(1) as usize)].iter().cloned().fold(f64::MIN, f64::max);
    // second difference of the view around the hit
    let w = &dirs[h.saturating_sub(8)..(h + 4).min(n)];
    let jerk = if w.len() >= 3 {
        (0..w.len() - 2)
            .map(|j| norm([w[j + 2][0] - 2.0 * w[j + 1][0] + w[j][0], w[j + 2][1] - 2.0 * w[j + 1][1] + w[j][1],
                w[j + 2][2] - 2.0 * w[j + 1][2] + w[j][2]]).to_degrees())
            .fold(f64::MIN, f64::max)
    } else {
        f64::NAN
    };
    let tw = h.saturating_sub(10)..h;
    let trk_resid = median(&resid[tw.clone()]);
    let trk_tgt = median(&tsteps[tw]);

    // approach shape: from the largest error in the ~30 ticks before acquisition down to it
    let acq = ((i + 1) as usize).min(h);
    let a0 = adt.partition_point(|&t| t < adt[acq] - 30);
    let seg_lo = if acq > a0 {
        // first index of the largest error
        (a0..=acq).max_by(|&x, &y| e[x].partial_cmp(&e[y]).unwrap().then(y.cmp(&x))).unwrap()
    } else {
        acq
    };
    let es = &e[seg_lo..=acq];
    let ts = &adt[seg_lo..=acq];
    let appr_len = (ts[ts.len() - 1] - ts[0]) as f64;
    let appr_drop = es[0] - es[es.len() - 1];
    let (mut ratio_med, mut ratio_mad, mut mono, mut step_cv) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    if es.len() >= 4 && es[0] >= 2.0 {
        let dts: Vec<f64> = ts.windows(2).map(|w| (w[1] - w[0]).max(1) as f64).collect();
        let r: Vec<f64> = (0..es.len() - 1)
            .filter(|&j| es[j] > 0.5)
            .map(|j| (es[j + 1].max(1e-3) / es[j].max(1e-3)).powf(1.0 / dts[j]))
            .collect();
        if r.len() >= 3 {
            let dsteps: Vec<f64> = (0..es.len() - 1).map(|j| -(es[j + 1] - es[j]) / dts[j]).collect();
            ratio_med = median(&r);
            let dev: Vec<f64> = r.iter().map(|x| (x - ratio_med).abs()).collect();
            ratio_mad = median(&dev);
            mono = (0..es.len() - 1).filter(|&j| es[j + 1] < es[j]).count() as f64 / (es.len() - 1) as f64;
            let mean = dsteps.iter().sum::<f64>() / dsteps.len() as f64;
            let std = (dsteps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dsteps.len() as f64).sqrt();
            step_cv = std / mean.abs().max(0.05);
        }
    }
    let stillness = if pre_med.is_nan() { f64::NAN } else { step_hit / pre_med.max(f64::from(MIN_STEP)) };
    let post_ratio = post_step / step_hit.max(f64::from(MIN_STEP));
    let trk_ratio = trk_resid / trk_tgt.max(0.05);
    if gaps.is_empty() || median(&gaps) > f64::from(MAX_SAMPLE_GAP) || dist < f64::from(MIN_DISTANCE) {
        return None;
    }
    let f = [
        e[h], e[h] / ang_size, e[h].to_radians() * dist, at(e, hi - 1), at(e, hi - 2), at(e, hi - 4), at(e, hi - 8),
        at(e, hi + 1), at(e, hi + 2), at(e, hi + 3), best_lag_gain, step_hit, pre_med, stillness, max_step_10,
        post_step, post_ratio, t_on, pre_err_max, jerk, trk_resid, trk_tgt, trk_ratio, appr_len, appr_drop,
        ratio_med, ratio_mad, mono, step_cv, dist, k as f64,
    ];
    Some(f.map(|x| x as f32))
}

/// How the aim got onto the target, in the order of `HIT_FEATURES[NEAR..AGGREGATED]`, on the long window at the hit's
/// lag `k`. The hold is the last run of samples within `ACQ_EXIT` times the on-target limit (`max(ang_size, 1°)`) that
/// lasts until the hit; the acquisition is its first sample on the target, the acquiring update the step into it. All
/// NaN when the aim is not on target at the hit, or already held at the start of the window.
/// - `acq_from`: the largest error in the `ACQ_FROM_TICKS` before the acquisition (the approach starts there)
/// - `acq_step`: the largest view step of one update in the approach, degrees per tick
/// - `acq_frac`: the largest share of `acq_from` that one update of the approach removed (an aimbot snap: ~1)
/// - `acq_land_rel`: error right after it, in units of the target's angular size
/// - `acq_t`: ticks from the acquisition to the hit
/// - `hold_err_rel`: median error from the acquisition to the hit, in units of the target's angular size
/// - `hold_resid`: median view motion not explained by the target's motion during the hold, degrees per tick
fn acquisition_features(att: &[Sample], vic: &[Sample], hit_tick: u32, k: usize, ang_size: f64) -> [f32; ACQ] {
    let nan = [f32::NAN; ACQ];
    let adt: Vec<i64> = att.iter().map(|s| i64::from(s.tick) - i64::from(hit_tick)).collect();
    let Some(h) = adt.iter().rposition(|&t| t <= 0) else { return nan };
    if vic.is_empty() {
        return nan;
    }
    let vdt: Vec<i64> = vic.iter().map(|s| i64::from(s.tick) - i64::from(hit_tick)).collect();
    let dirs: Vec<V3> = att.iter().map(|s| view_dir(s.yaw, s.pitch)).collect();
    let (e, tgt): (Vec<f64>, Vec<V3>) =
        (0..=h).map(|i| column_error(att[i].eye.map(f64::from), dirs[i], victim_at(vic, &vdt, adt[i] - k as i64))).unzip();
    let on_limit = ang_size.max(1.0);
    if e[h] > on_limit {
        return nan;
    }
    let Some(left) = (0..h).rev().find(|&i| e[i] > ACQ_EXIT * on_limit) else { return nan };
    let Some(acq) = (left + 1..=h).find(|&i| e[i] <= on_limit) else { return nan };
    // the approach: from the (first) largest error in the ticks before the acquisition to it
    let a0 = (0..acq).find(|&i| adt[i] >= adt[acq] - ACQ_FROM_TICKS).unwrap_or(acq - 1);
    let j = (a0..acq).fold(acq - 1, |b, i| if e[i] > e[b] || (e[i] == e[b] && i < b) { i } else { b });
    let from = e[j];
    let (mut step, mut frac) = (0.0f64, 0.0f64);
    for i in j..acq {
        let gap = (adt[i + 1] - adt[i]).max(1) as f64;
        step = step.max(angle_between(dirs[i + 1], dirs[i]) / gap);
        frac = frac.max((e[i] - e[i + 1]) / from.max(1e-6));
    }
    let unit = |t: V3| {
        let l = norm(t).max(1e-9);
        [t[0] / l, t[1] / l, t[2] / l]
    };
    let resid: Vec<f64> = (acq..h)
        .map(|i| {
            let g = (adt[i + 1] - adt[i]).max(1) as f64;
            norm(sub(sub(dirs[i + 1], dirs[i]), sub(unit(tgt[i + 1]), unit(tgt[i])))).to_degrees() / g
        })
        .collect();
    let f = [
        from,
        step,
        frac,
        e[acq] / ang_size,
        (adt[h] - adt[acq]) as f64,
        median(&e[acq..=h]) / ang_size,
        if resid.is_empty() { f64::NAN } else { median(&resid) },
    ];
    f.map(|x| x as f32)
}

/// Player features from the player's hit features (see `unit_feature_names`)
fn unit_features(hits: &[&HitRecord]) -> Vec<f32> {
    let mut out = Vec::with_capacity(AGGREGATED * 3 + 2);
    for q in [0.5, 0.1, 0.9] {
        for f in 0..AGGREGATED {
            let mut col: Vec<f64> = hits.iter().map(|h| f64::from(h.features[f])).collect();
            out.push(quantile(&mut col, q) as f32);
        }
    }
    let n = hits.len() as f32;
    let snap = hits.iter().filter(|h| {
        let f = &h.features;
        f[IDX_STILLNESS] >= 6.0 && f[IDX_STEP_HIT] >= 2.0 && f[IDX_ERR0_REL] <= 0.6
    });
    out.push(snap.count() as f32 / n);
    out.push(hits.iter().filter(|h| h.features[IDX_ERR0_REL] <= 0.25).count() as f32 / n);
    out
}

pub struct MlAimbot {
    params: Parameters,
    jg: JankGuard,
    history: FxHashMap<u64, VecDeque<Sample>>,
    last_sim: FxHashMap<u64, i64>,
    last_hit: FxHashMap<u64, u32>,
    pending: Vec<PendingHit>,
    recorder: Option<u64>,
    /// Usable hits of the demo, in hit order
    pub hits: Vec<HitRecord>,
    /// Players with enough hits, filled by `finish`
    pub units: Vec<UnitRecord>,
}

impl Default for MlAimbot {
    fn default() -> Self {
        Self::new()
    }
}

impl MlAimbot {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_score".to_string(), Parameter::Float(0.33)),
                ("min_hits".to_string(), Parameter::Int(5)),
                ("max_detections".to_string(), Parameter::Int(3)),
            ]),
            jg: JankGuard::default(),
            history: FxHashMap::default(),
            last_sim: FxHashMap::default(),
            last_hit: FxHashMap::default(),
            pending: Vec::new(),
            recorder: None,
            hits: Vec::new(),
            units: Vec::new(),
        }
    }

    /// Evaluates every pending hit into `hits` (end of the demo), for users of the hit features outside `finish`
    pub fn flush_pending(&mut self) {
        self.flush(u32::MAX, true);
    }

    /// Evaluates the pending hits whose window has passed (all of them when `all`)
    fn flush(&mut self, now: u32, all: bool) {
        let mut i = 0;
        while i < self.pending.len() {
            if !all && self.pending[i].tick + WINDOW_AFTER > now {
                i += 1;
                continue;
            }
            let hit = self.pending.swap_remove(i);
            let (Some(a), Some(v)) = (self.history.get(&hit.attacker), self.history.get(&hit.victim)) else { continue };
            let lo = hit.tick.saturating_sub(WINDOW_BEFORE);
            let hi = hit.tick + WINDOW_AFTER;
            let att: Vec<Sample> = a.iter().filter(|s| s.tick >= lo && s.tick <= hi).copied().collect();
            if att.len() < MIN_WINDOW_SAMPLES || hit.since_spawn < MIN_SINCE_SPAWN {
                continue;
            }
            let vic: Vec<Sample> = v.iter().filter(|s| s.tick + MAX_LAG as u32 >= lo && s.tick <= hi).copied().collect();
            if let Some(near) = hit_features(&att, &vic, hit.tick) {
                // the acquisition window starts after the attacker's spawn or teleport
                let acq_lo = hit.tick.saturating_sub(ACQ_WINDOW_BEFORE.min(hit.since_spawn));
                let att_long: Vec<Sample> = a.iter().filter(|s| s.tick >= acq_lo && s.tick <= hi).copied().collect();
                let vic_long: Vec<Sample> =
                    v.iter().filter(|s| s.tick + MAX_LAG as u32 >= acq_lo && s.tick <= hi).copied().collect();
                let k = near[NEAR + 1] as usize;
                let ang_size = f64::from(BODY_HALF_WIDTH).atan2(f64::from(near[NEAR])).to_degrees();
                let acq = acquisition_features(&att_long, &vic_long, hit.tick, k, ang_size);
                let mut features = [f32::NAN; HIT_FEATURES.len()];
                features[..NEAR].copy_from_slice(&near[..NEAR]);
                features[NEAR..AGGREGATED].copy_from_slice(&acq);
                features[IDX_DIST] = near[NEAR];
                features[IDX_LAG] = near[NEAR + 1];
                self.hits.push(HitRecord {
                    tick: hit.tick,
                    attacker: hit.attacker,
                    victim: hit.victim,
                    weapon: hit.weapon,
                    features,
                    score: f32::NAN,
                });
            }
        }
    }
}

impl<'a> CheatAlgorithm<'a> for MlAimbot {
    fn algorithm_name(&self) -> &str {
        "fidoo/ml_aimbot"
    }

    fn tooltip(&self) -> &'static str {
        "Scores each player's hitscan hits over the whole demo with a learned model (snaps, jerk, stillness, tracking, and how the aim got onto the target in the last 3 s, for aimbots that wait for a headshot or charge) and marks the most aimbot-like hits of high-scoring players."
    }

    // Never hidden by the global sliding window: aggregates its evidence per player over the whole demo
    fn excluded_from_sliding_window(&self) -> bool {
        true
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
        if !m.contains(&MessageType::GameEvent) {
            m.push(MessageType::GameEvent);
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
        let Message::GameEvent(GameEventMessage { event: GameEvent::PlayerHurt(h), .. }) = message else { return Ok(vec![]) };
        let tick = u32::from(tick);
        let (Some(a), Some(v)) = (state.get_id64_from_userid(h.attacker.into()), state.get_id64_from_userid(h.user_id.into())) else {
            return Ok(vec![]);
        };
        if a == v || a == 0 || Some(a) == self.recorder {
            return Ok(vec![]);
        }
        let Some(ap) = state.players.iter().find(|p| player_sid(p) == Some(a) && p.state == PlayerState::Alive) else { return Ok(vec![]) };
        let Some(vp) = state.players.iter().find(|p| player_sid(p) == Some(v)) else { return Ok(vec![]) };
        if ap.team == vp.team || !ap.in_pvs {
            return Ok(vec![]);
        }
        let weapon = state.get_player_weapon(ap);
        if !is_hitscan(&weapon) || self.last_hit.get(&a).is_some_and(|&t| tick < t + MIN_HIT_SPACING) {
            return Ok(vec![]);
        }
        self.last_hit.insert(a, tick);
        let since_spawn = self.jg.spawned(&a, tick).min(self.jg.teleported(&a, tick));
        self.pending.push(PendingHit { tick, attacker: a, victim: v, weapon, since_spawn });
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
            let sim = state.sim_tick(p);
            if self.last_sim.insert(sid, sim) == Some(sim) {
                continue;
            }
            let ducking = p.flags & FL_DUCKING != 0;
            let pos = [p.position.x, p.position.y, p.position.z];
            let eye = [pos[0], pos[1], pos[2] + view_height(p, ducking)];
            let h = self.history.entry(sid).or_default();
            h.push_back(Sample { tick, pos, eye, yaw: p.view_angle, pitch: p.pitch_angle, ducking });
            while h.front().is_some_and(|s| s.tick + HISTORY_TICKS < tick) {
                h.pop_front();
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
        let min_hits = get_parameter_value::<i32>(&self.params, "min_hits").max(1) as usize;
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let model = model();
        if let Some(m) = model {
            for hit in &mut self.hits {
                hit.score = m.hit.probability(&hit.features);
            }
        }
        let mut players: Vec<u64> = self.hits.iter().map(|h| h.attacker).collect();
        players.sort_unstable();
        players.dedup();
        let mut detections = Vec::new();
        for player in players {
            let hits: Vec<&HitRecord> = self.hits.iter().filter(|h| h.attacker == player).collect();
            if hits.len() < min_hits {
                continue;
            }
            let features = unit_features(&hits);
            let score = model.map_or(f32::NAN, |m| m.player_score(&features, hits.len()));
            self.units.push(UnitRecord { player, n_hits: hits.len(), features, score });
            if !(score >= min_score) {
                continue;
            }
            let mut top = hits.clone();
            top.sort_by(|a, b| b.score.total_cmp(&a.score));
            for h in top.into_iter().take(max_detections) {
                let f = &h.features;
                detections.push(Detection {
                    tick: h.tick,
                    algorithm: "fidoo/ml_aimbot".to_string(),
                    player,
                    data: json!({
                        "score": score,
                        "hit_score": h.score,
                        "hits": hits.len(),
                        "weapon": h.weapon,
                        "target": h.victim,
                        "distance": f[IDX_DIST].round(),
                        "error_deg": f[0],
                        "snap_deg_per_tick": f[IDX_STEP_HIT],
                        "stillness": f[IDX_STILLNESS],
                        "jerk_deg": f[IDX_JERK],
                        "lag_ticks": f[IDX_LAG],
                        "acquire_deg_per_tick": f[IDX_ACQ_STEP],
                        "acquired_ticks_before_hit": f[IDX_ACQ_T],
                    }),
                });
            }
        }
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tick: u32, yaw: f32) -> Sample {
        Sample { tick, pos: [0.0, 0.0, 0.0], eye: [0.0, 0.0, 68.0], yaw, pitch: 0.0, ducking: false }
    }

    fn victim(tick: u32) -> Sample {
        Sample { tick, pos: [1000.0, 0.0, 0.0], eye: [1000.0, 0.0, 68.0], yaw: 0.0, pitch: 0.0, ducking: false }
    }

    #[test]
    fn snap_and_hold_is_one_acquiring_update() {
        // 10 deg off a still target, one-update snap onto it at tick 60, held until the hit at tick 100
        let att: Vec<Sample> = (0..=100).map(|t| sample(t, if t < 60 { 10.0 } else { 0.0 })).collect();
        let vic: Vec<Sample> = (0..=100).map(victim).collect();
        let ang = f64::from(BODY_HALF_WIDTH).atan2(1000.0).to_degrees();
        let f = acquisition_features(&att, &vic, 100, 0, ang);
        assert!((f[0] - 10.0).abs() < 0.01, "{f:?}");
        assert!((f[1] - 10.0).abs() < 0.01, "{f:?}");
        assert!(f[2] > 0.99, "{f:?}");
        assert!(f[3] < 0.01, "{f:?}");
        assert_eq!(f[4], 40.0);
        assert!(f[5] < 0.01 && f[6] < 0.01, "{f:?}");
    }

    #[test]
    fn few_hits_pull_the_score_towards_the_prior() {
        let empty = || Booster { features: Vec::new(), trees: Vec::new() };
        let m = Model { unit: empty(), hit: empty(), prior: Some(0.1), full_hits: Some(15.0) };
        assert!((m.player_score(&[], 15) - 0.5).abs() < 1e-6);
        assert!((m.player_score(&[], 40) - 0.5).abs() < 1e-6);
        assert!((m.player_score(&[], 0) - 0.1).abs() < 1e-6);
        // a third of the evidence: a third of the way from the prior's logit to the model's
        let z = (0.1f64 / 0.9).ln() * (2.0 / 3.0);
        assert!((f64::from(m.player_score(&[], 5)) - 1.0 / (1.0 + (-z).exp())).abs() < 1e-6);
        let unweighted = Model { unit: empty(), hit: empty(), prior: None, full_hits: None };
        assert!((unweighted.player_score(&[], 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn already_held_target_has_no_acquisition() {
        let att: Vec<Sample> = (0..=100).map(|t| sample(t, 0.2)).collect();
        let vic: Vec<Sample> = (0..=100).map(victim).collect();
        assert!(acquisition_features(&att, &vic, 100, 0, 1.37).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn wobble_just_outside_the_target_does_not_end_the_hold() {
        // snap at tick 30, then one sample 1.5x the limit off at tick 70: the hold continues
        let att: Vec<Sample> = (0..=100).map(|t| sample(t, if t < 30 { 12.0 } else if t == 70 { 2.0 } else { 0.0 })).collect();
        let vic: Vec<Sample> = (0..=100).map(victim).collect();
        let ang = f64::from(BODY_HALF_WIDTH).atan2(1000.0).to_degrees();
        assert_eq!(acquisition_features(&att, &vic, 100, 0, ang)[4], 70.0);
    }
}
