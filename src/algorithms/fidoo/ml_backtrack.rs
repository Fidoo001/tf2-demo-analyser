//! Hitscan backtrack detection on spectated players' hits. See docs/ml_backtrack.md.
//!
//! The server lag-compensates a hitscan shot by rewinding the victim to `tick_count - lerp`. A legit
//! player's rewind (`k` ticks before the hurt tick) is their round trip latency plus interpolation, so it
//! stays roughly constant and follows the scoreboard ping. Backtrack cheats choose an older record per
//! shot (up to 200 ms around the "correct" time), and fake latency / fake interp push the whole window
//! back by up to sv_maxunlag (1 s) while a ping reducer keeps the scoreboard ping low.
//!
//! Per hit of a weapon without spread (sniper rifles, revolvers), the attacker's view ray (the sniper dot
//! when scoped) is compared with the victim's body column (head sphere for headshots) at every rewind
//! 0..=MAX_LAG ticks, which gives the rewinds the shot could have hit. Per player, the rewind that explains
//! most hits is the typical one (k0). Hits only possible at a rewind 3..15 ticks away from k0 are
//! backtrack-like. Occasional lag makes a few of them (usually with gaps in the attacker's updates);
//! constant high latency only moves k0. The player features are scored by a learned model.

use std::collections::VecDeque;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::packetentities::{PacketEntitiesMessage, UpdateType};
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::demo::sendprop::{SendPropIdentifier, SendPropValue};
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::util::helpers::resolve_recorder_sid;
use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;
use crate::util::nocrex::jankguard::JankGuard;

/// Largest rewind tried, in ticks (sv_maxunlag 1 s plus a little)
pub const MAX_LAG: usize = 70;
/// Attacker samples kept before the shot. The aim is the last one (the shot tick): an aimbot snaps onto the
/// backtracked record on that tick only, and the mean over several samples turns flicks into fake rewinds.
/// The previous sample and the mean of three are kept for research.
const AIM_SAMPLES: usize = 3;
const HISTORY_TICKS: u32 = MAX_LAG as u32 + 40;
/// Hits of one attacker closer than this are merged into the first one (pellets, sprays)
const MIN_HIT_SPACING: u32 = 8;
/// Hits right after a spawn / teleport of either player are not used
const MIN_SINCE_SPAWN: u32 = 60;
const MIN_DISTANCE: f32 = 50.0;
/// Largest gap between the rewound time and the victim sample used for it
const MAX_VICTIM_GAP: i64 = 4;
const MAX_AIM_GAP: i64 = 4;
/// Largest gap between the shot and the attacker sample used as its aim
const SHOT_AIM_GAP: u32 = 2;
const BODY_HALF_WIDTH: f32 = 24.0;
const BODY_BOTTOM: f32 = 5.0;
const BODY_TOP_STANDING: f32 = 80.0;
const BODY_TOP_DUCKING: f32 = 60.0;
/// Head sphere radius used for headshots (generous: the head moves with the animation)
const HEAD_RADIUS: f32 = 12.0;
const FL_DUCKING: u32 = 1 << 1;
/// Rewinds considered for the typical rewind (sv_maxunlag)
pub const LAGS: usize = 61;
/// Networked eye angle precision, and the eye height model error with the sniper dot (degrees)
const EYE_TOLERANCE: f32 = 0.35;
const DOT_TOLERANCE: f32 = 0.1;
/// Hits possible within k0 +- NEAR are explained by the typical rewind; backtrack reaches FAR beyond it
const NEAR: usize = 2;
const FAR: usize = 15;
/// Legit share of backtrack-like hits, for the binomial score (unmarked players, shot-tick aim)
const BASE_RATE: f64 = 0.034;
/// Attacker update gap (ticks) in the 32 ticks before the hit that counts as lag
const LAG_GAP: u32 = 6;
const GAP_WINDOW: u32 = 32;
/// A hit needs a CTEFireBullets of the attacker at most this many ticks before / after the hurt event
/// (bleed, afterburn and spells hurt without a shot and are credited to the active weapon). The hurt
/// event often arrives 1..8 ticks after the shot, so the shot tick anchors the aim and the rewinds.
const FIRE_BEFORE: u32 = 8;
const FIRE_AFTER: u32 = 1;
const TF_DMG_CUSTOM_BLEEDING: u16 = 34;
const FL_ONGROUND: u32 = 1;
/// Victim window (ticks before the hit) checked for standing on a moving platform
const LIFT_WINDOW: u32 = 40;
const BULLETS_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
const DOT_ORIGIN: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_vecOrigin");
const DOT_OWNER: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_hOwnerEntity");

#[derive(Clone, Copy)]
struct Sample {
    tick: u32,
    pos: [f32; 3],
    eye: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
    /// Head centre height above the origin
    head: f32,
    /// The view angles come from the sniper dot (exact) instead of the networked eye angles
    dot: bool,
    on_ground: bool,
}

struct Dot {
    owner: u32,
    pos: [f32; 3],
}

struct PendingHit {
    tick: u32,
    attacker: u64,
    victim: u64,
    weapon: String,
    ping: u16,
    headshot: bool,
    custom: u16,
    ent: u32,
}

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

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
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_backtrack_model.json")).ok()?;
            let names: Vec<String> = UNIT_FEATURES.iter().map(|s| s.to_string()).collect();
            (model.unit.features == names).then_some(model)
        })
        .as_ref()
}

/// Player feature names, in the order of `unit_features`
pub const UNIT_FEATURES: [&str; 18] = [
    "n", "k0", "cov", "ping", "excess", "bt_share", "older_share", "newer_share", "bt_score", "none_share",
    "dot_share", "hs_share", "bt_offset_med", "bt_gap_share", "gap_share", "cov2", "ping_iqr", "bt_spread",
];

#[derive(Clone, Debug)]
pub struct HitRecord {
    pub tick: u32,
    pub attacker: u64,
    pub victim: u64,
    pub weapon: String,
    /// Attacker's scoreboard ping (ms) at the hit
    pub ping: u16,
    pub dist: f32,
    /// Angular half width of the victim's body, degrees
    pub ang_size: f32,
    /// Aim error (degrees) at the shot to the victim's body rewound by 0..=MAX_LAG ticks, NaN where the victim wasn't sampled
    pub err: Vec<f32>,
    /// Headshots: aim error to the rewound head sphere's surface (negative inside), else empty
    pub head_err: Vec<f32>,
    /// The aim at the shot came from the sniper dot
    pub dot: bool,
    /// Largest gap between the attacker's updates in the 32 ticks before the hit
    pub att_gap: u32,
    /// Bit L: the hit was possible at rewind L (0..LAGS)
    pub ok: u64,
    /// Possible at some rewind but not at most of them, with the victim sampled over 0..=40 ticks
    pub informative: bool,
    pub custom: u16,
    /// Ticks from the hurt event to the shot (its CTEFireBullets; i32::MAX: none, only kept for research)
    pub fire_dt: i32,
    /// View rotation into the hit sample, degrees per tick
    pub step_hit: f32,
    /// Largest vertical speed (HU/tick) of the victim while standing still on the ground before the hit
    pub lift_vz: f32,
    /// Masks from the last aim sample alone, the one before it, and the mean of the last three
    pub ok_last: u64,
    pub ok_prev: u64,
    pub ok_mean: u64,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub k0: usize,
    pub features: Vec<f32>,
    pub score: f32,
}

/// Same exclusions as fidoo/plain_aimbot, keeping only weapons without spread
pub fn is_precise(name: &str) -> bool {
    let lw = name.to_lowercase();
    if lw.contains("huntsman") || lw.contains("fortified compound") || lw.contains("smg") || lw.contains("carbine") {
        return false;
    }
    [
        "sniper rifle", "machina", "hitman", "bazaar", "classic", "sydney", "awper", "shooting star", "ap-sap",
        "revolver", "ambassador", "enforcer", "diamondback", "l'etranger", "big kill",
    ]
    .iter()
    .any(|w| lw.contains(w))
}

/// Rewinds at which a hit was possible, and whether the hit tells rewinds apart
fn hittable(err: &[f32], head_err: &[f32], ang_size: f32, dot: bool) -> (u64, bool) {
    let q = if dot { DOT_TOLERANCE } else { EYE_TOLERANCE };
    let (curve, tol) = if head_err.is_empty() { (err, ang_size + q) } else { (head_err, q) };
    let mut ok = 0u64;
    let mut valid = 0;
    for (lag, &e) in curve.iter().take(LAGS).enumerate() {
        if !e.is_nan() {
            valid += 1;
            if e <= tol {
                ok |= 1 << lag;
            }
        }
    }
    let n_ok = ok.count_ones() as usize;
    let informative = n_ok > 0 && n_ok * 2 <= valid && curve.iter().take(41).all(|e| !e.is_nan());
    (ok, informative)
}

/// Bits lo..=hi (clamped to the considered rewinds)
fn span(lo: isize, hi: isize) -> u64 {
    let lo = lo.max(0) as u32;
    let hi = hi.min(LAGS as isize - 1);
    if hi < lo as isize {
        return 0;
    }
    let hi = hi as u32;
    (if hi >= 63 { u64::MAX } else { (1u64 << (hi + 1)) - 1 }) & !((1u64 << lo) - 1)
}

/// -log10 P(X >= x) for X ~ Binomial(n, p)
fn binom_score(x: usize, n: usize, p: f64) -> f64 {
    if x == 0 {
        return 0.0;
    }
    let ln_pmf = |i: usize| {
        let lc: f64 = (0..i).map(|j| ((n - j) as f64).ln() - ((j + 1) as f64).ln()).sum();
        lc + i as f64 * p.ln() + (n - i) as f64 * (1.0 - p).ln()
    };
    let terms: Vec<f64> = (x..=n).map(ln_pmf).collect();
    let m = terms.iter().cloned().fold(f64::MIN, f64::max);
    let ln_tail = m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln();
    -ln_tail / std::f64::consts::LN_10
}

fn quantile(values: &mut Vec<f64>, q: f64) -> f64 {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (values.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    values[lo] + (values[hi] - values[lo]) * (pos - lo as f64)
}

/// Typical rewind of a player's informative hits and whether each hit is backtrack-like (older, newer)
pub fn classify(hits: &[&HitRecord]) -> (usize, f64, Vec<(bool, bool)>) {
    let inf: Vec<u64> = hits.iter().filter(|h| h.informative).map(|h| h.ok).collect();
    let n = inf.len().max(1) as f64;
    let cov: Vec<f64> = (0..LAGS as isize).map(|l| inf.iter().filter(|&&o| o & span(l - 1, l + 1) != 0).count() as f64 / n).collect();
    let k0 = (0..LAGS).fold(0, |b, l| if cov[l] > cov[b] { l } else { b });
    let k = k0 as isize;
    let (near, older, newer) = (span(k - NEAR as isize, k + NEAR as isize), span(k + NEAR as isize + 1, k + FAR as isize),
        span(k - FAR as isize, k - NEAR as isize - 1));
    let kinds = hits
        .iter()
        .map(|h| {
            let bt = h.informative && h.ok & near == 0;
            (bt && h.ok & older != 0, bt && h.ok & newer != 0)
        })
        .collect();
    (k0, cov[k0], kinds)
}

/// Player features (see `UNIT_FEATURES`) of all precise hits of one player, or `None` with fewer than
/// `min_hits` informative hits
pub fn unit_features(hits: &[&HitRecord], min_hits: usize) -> Option<(usize, Vec<f32>, Vec<(bool, bool)>)> {
    let n_inf = hits.iter().filter(|h| h.informative).count();
    if n_inf < min_hits {
        return None;
    }
    let (k0, cov, kinds) = classify(hits);
    let bt: Vec<bool> = kinds.iter().map(|&(o, w)| o || w).collect();
    let n_bt = bt.iter().filter(|&&b| b).count();
    let n = n_inf as f64;
    let share = |f: &dyn Fn(usize) -> bool| (0..hits.len()).filter(|&i| hits[i].informative && f(i)).count() as f64 / n;
    let mut pings: Vec<f64> = hits.iter().map(|h| f64::from(h.ping)).collect();
    let ping = quantile(&mut pings.clone(), 0.5);
    let ping_iqr = quantile(&mut pings.clone(), 0.75) - quantile(&mut pings, 0.25);
    let k = k0 as isize;
    let mut offsets: Vec<f64> = (0..hits.len())
        .filter(|&i| bt[i])
        .map(|i| (0..LAGS as isize).filter(|&l| hits[i].ok >> l & 1 == 1).map(|l| (l - k).abs()).min().unwrap_or(0) as f64)
        .collect();
    let bt_gap = (0..hits.len()).filter(|&i| bt[i] && hits[i].att_gap >= LAG_GAP).count() as f64;
    let mut minutes: Vec<u32> = (0..hits.len()).filter(|&i| bt[i]).map(|i| hits[i].tick / 3960).collect();
    minutes.dedup();
    let inf: Vec<u64> = hits.iter().filter(|h| h.informative).map(|h| h.ok).collect();
    let cov2 = (0..LAGS as isize)
        .filter(|l| (l - k).abs() >= 5)
        .map(|l| inf.iter().filter(|&&o| o & span(l - 1, l + 1) != 0).count() as f64 / n)
        .fold(0.0, f64::max);
    let all = hits.len() as f64;
    let f = [
        n,
        k0 as f64,
        cov,
        ping,
        k0 as f64 - ping * 66.67 / 1000.0,
        n_bt as f64 / n,
        share(&|i| kinds[i].0),
        share(&|i| kinds[i].1),
        binom_score(n_bt, n_inf, BASE_RATE),
        hits.iter().filter(|h| h.ok == 0).count() as f64 / all,
        hits.iter().filter(|h| h.dot).count() as f64 / all,
        hits.iter().filter(|h| !h.head_err.is_empty()).count() as f64 / all,
        quantile(&mut offsets, 0.5),
        if n_bt > 0 { bt_gap / n_bt as f64 } else { f64::NAN },
        share(&|i| hits[i].att_gap >= LAG_GAP),
        cov2,
        ping_iqr,
        if n_bt > 0 { minutes.len() as f64 / n_bt as f64 } else { f64::NAN },
    ];
    Some((k0, f.iter().map(|&x| x as f32).collect(), kinds))
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

/// Head centre above the origin
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

fn player_sid(player: &Player) -> Option<u64> {
    let info = player.info.as_ref()?;
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
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

/// Error from the view ray of `a` to the body column of victim sample `v`, and the distance
fn body_error(a: &Sample, v: &Sample) -> (f64, f64) {
    let eye = a.eye.map(f64::from);
    let vpos = v.pos.map(f64::from);
    let d = view_dir(a.yaw, a.pitch);
    let top = if v.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING } as f64;
    let rel = [vpos[0] - eye[0], vpos[1] - eye[1]];
    let dh = (rel[0] * rel[0] + rel[1] * rel[1]).sqrt();
    let dxy = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
    let aim_z = eye[2] + d[2] / dxy * dh;
    let tz = aim_z.clamp(vpos[2] + f64::from(BODY_BOTTOM), vpos[2] + top);
    let t = [rel[0], rel[1], tz - eye[2]];
    (angle_between(d, t), norm(t))
}

/// Angle from the view ray of `a` to the surface of the head sphere of victim sample `v` (negative inside)
fn head_error(a: &Sample, v: &Sample) -> f64 {
    let eye = a.eye.map(f64::from);
    let t = [f64::from(v.pos[0]) - eye[0], f64::from(v.pos[1]) - eye[1], f64::from(v.pos[2] + v.head) - eye[2]];
    let d = norm(t).max(f64::from(HEAD_RADIUS) + 1.0);
    angle_between(view_dir(a.yaw, a.pitch), t) - (f64::from(HEAD_RADIUS) / d).asin().to_degrees()
}

/// Mean error over the aim samples at every rewind, NaN where the victim wasn't sampled
fn curve(aim: &[Sample], vic: &[Sample], err_fn: impl Fn(&Sample, &Sample) -> f64) -> Vec<f32> {
    let mut err = vec![f32::NAN; MAX_LAG + 1];
    for (lag, e) in err.iter_mut().enumerate() {
        let mut sum = 0.0;
        for a in aim {
            let q = i64::from(a.tick) - lag as i64;
            let vi = vic.partition_point(|s| i64::from(s.tick) <= q);
            if vi == 0 || q - i64::from(vic[vi - 1].tick) > MAX_VICTIM_GAP {
                sum = f64::NAN;
                break;
            }
            sum += err_fn(a, &vic[vi - 1]);
        }
        *e = (sum / aim.len() as f64) as f32;
    }
    err
}

struct HitCurve {
    err: Vec<f32>,
    head_err: Vec<f32>,
    dist: f32,
    ang_size: f32,
    dot: bool,
    att_gap: u32,
    ok_last: (u64, bool),
    ok_prev: (u64, bool),
    ok_mean: (u64, bool),
    step_hit: f32,
    lift_vz: f32,
}

/// Error curves of one hit, or `None` when the aim samples are unusable
fn hit_curve(att: &[Sample], vic: &[Sample], hit_tick: u32, headshot: bool) -> Option<HitCurve> {
    let h = att.iter().rposition(|s| s.tick <= hit_tick)?;
    if h + 1 < AIM_SAMPLES {
        return None;
    }
    let aim = &att[h + 1 - AIM_SAMPLES..=h];
    let att_gap = att[..=h]
        .windows(2)
        .filter(|w| w[1].tick + GAP_WINDOW >= hit_tick)
        .map(|w| w[1].tick - w[0].tick)
        .max()
        .unwrap_or(0);
    // the aim must be the shot tick's (it is in 99.7% of the hits)
    if hit_tick - aim[aim.len() - 1].tick > SHOT_AIM_GAP
        || i64::from(hit_tick) - i64::from(aim[0].tick) > MAX_AIM_GAP + AIM_SAMPLES as i64
    {
        return None;
    }
    let last = aim[aim.len() - 1];
    let vi = vic.partition_point(|s| s.tick <= last.tick);
    if vi == 0 {
        return None;
    }
    let dist = body_error(&last, &vic[vi - 1]).1;
    if !(dist >= f64::from(MIN_DISTANCE)) {
        return None;
    }
    let ang_size = f64::from(BODY_HALF_WIDTH).atan2(dist).to_degrees() as f32;
    let masks = |set: &[Sample]| {
        let err = curve(set, vic, |a, v| body_error(a, v).0);
        let head_err = if headshot { curve(set, vic, head_error) } else { Vec::new() };
        hittable(&err, &head_err, ang_size, set.iter().all(|a| a.dot))
    };
    let n = aim.len();
    let (a, b) = (&aim[n - 2], &aim[n - 1]);
    let step_hit = angle_between(view_dir(a.yaw, a.pitch), view_dir(b.yaw, b.pitch)) / f64::from((b.tick - a.tick).max(1));
    // victim standing still on the ground while its height changes: a lift or moving platform
    let lift_vz = vic
        .windows(2)
        .filter(|w| w[1].tick + LIFT_WINDOW >= hit_tick && w[0].on_ground && w[1].on_ground)
        .filter_map(|w| {
            let dt = (w[1].tick - w[0].tick).max(1) as f32;
            let dh = (w[1].pos[0] - w[0].pos[0]).hypot(w[1].pos[1] - w[0].pos[1]) / dt;
            (dh < 3.0).then(|| (w[1].pos[2] - w[0].pos[2]).abs() / dt)
        })
        .fold(0.0f32, f32::max);
    let shot = &aim[n - 1..];
    Some(HitCurve {
        err: curve(shot, vic, |a, v| body_error(a, v).0),
        head_err: if headshot { curve(shot, vic, head_error) } else { Vec::new() },
        dist: dist as f32,
        ang_size,
        dot: aim[n - 1].dot,
        att_gap,
        ok_last: masks(&aim[n - 1..]),
        ok_prev: masks(&aim[n - 2..n - 1]),
        ok_mean: masks(aim),
        step_hit: step_hit as f32,
        lift_vz,
    })
}

pub struct MlBacktrack {
    params: Parameters,
    jg: JankGuard,
    history: FxHashMap<u64, VecDeque<Sample>>,
    last_sim: FxHashMap<u64, i64>,
    last_hit: FxHashMap<u64, u32>,
    pending: Vec<PendingHit>,
    /// Sniper dots by entity index
    dots: FxHashMap<u32, Dot>,
    /// Recent CTEFireBullets ticks by shooter entity index
    fires: FxHashMap<u32, VecDeque<u32>>,
    recorder: Option<u64>,
    /// Usable hits of the demo, in hit order
    pub hits: Vec<HitRecord>,
    /// Players with enough informative hits, filled by `finish`
    pub units: Vec<UnitRecord>,
}

impl Default for MlBacktrack {
    fn default() -> Self {
        Self::new()
    }
}

impl MlBacktrack {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("min_score".to_string(), Parameter::Float(0.3)),
                ("min_hits".to_string(), Parameter::Int(8)),
                ("max_detections".to_string(), Parameter::Int(20)),
                // research: keep hurt events without a shot (dumped with fire_dt)
                ("keep_unfired".to_string(), Parameter::Bool(false)),
            ]),
            jg: JankGuard::default(),
            history: FxHashMap::default(),
            last_sim: FxHashMap::default(),
            last_hit: FxHashMap::default(),
            pending: Vec::new(),
            dots: FxHashMap::default(),
            fires: FxHashMap::default(),
            recorder: None,
            hits: Vec::new(),
            units: Vec::new(),
        }
    }

    fn track_dots(&mut self, msg: &PacketEntitiesMessage, parser_state: &ParserState) {
        for e in &msg.entities {
            let id = u32::from(e.entity_index);
            if matches!(e.update_type, UpdateType::Delete | UpdateType::Leave) {
                self.dots.remove(&id);
                continue;
            }
            let entering = e.update_type == UpdateType::Enter;
            let is_dot = if entering {
                parser_state.server_classes.get(usize::from(e.server_class)).is_some_and(|c| c.name.as_str() == "CSniperDot")
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

    /// Evaluates the pending hits from before `now` (all of them when `all`)
    fn flush(&mut self, now: u32, all: bool) {
        let (ready, waiting): (Vec<PendingHit>, Vec<PendingHit>) =
            std::mem::take(&mut self.pending).into_iter().partition(|h| all || h.tick + FIRE_AFTER < now);
        self.pending = waiting;
        let keep_unfired = get_parameter_value::<bool>(&self.params, "keep_unfired");
        for hit in ready {
            // the shot: the attacker's latest CTEFireBullets up to FIRE_BEFORE ticks before the hurt event,
            // which often arrives a few ticks after the shot
            let fire = self
                .fires
                .get(&hit.ent)
                .and_then(|f| f.iter().copied().filter(|&t| t + FIRE_BEFORE >= hit.tick && t <= hit.tick + FIRE_AFTER).max());
            if fire.is_none() && !keep_unfired {
                continue;
            }
            let fire_dt = fire.map_or(i32::MAX, |t| t as i32 - hit.tick as i32);
            let shot = fire.unwrap_or(hit.tick);
            let (Some(a), Some(v)) = (self.history.get(&hit.attacker), self.history.get(&hit.victim)) else { continue };
            let lo = shot.saturating_sub(AIM_SAMPLES as u32 + MAX_AIM_GAP as u32 + 2);
            let glo = shot.saturating_sub(GAP_WINDOW + 1);
            let att: Vec<Sample> = a.iter().filter(|s| s.tick >= glo && s.tick <= shot).copied().collect();
            let vlo = lo.saturating_sub(MAX_LAG as u32 + MAX_VICTIM_GAP as u32);
            let vic: Vec<Sample> = v.iter().filter(|s| s.tick >= vlo && s.tick <= shot).copied().collect();
            if let Some(c) = hit_curve(&att, &vic, shot, hit.headshot) {
                let (ok, informative) = c.ok_last;
                self.hits.push(HitRecord {
                    tick: shot,
                    attacker: hit.attacker,
                    victim: hit.victim,
                    weapon: hit.weapon,
                    ping: hit.ping,
                    dist: c.dist,
                    ang_size: c.ang_size,
                    err: c.err,
                    head_err: c.head_err,
                    dot: c.dot,
                    att_gap: c.att_gap,
                    ok,
                    informative,
                    custom: hit.custom,
                    fire_dt,
                    step_hit: c.step_hit,
                    lift_vz: c.lift_vz,
                    ok_last: c.ok_last.0,
                    ok_prev: c.ok_prev.0,
                    ok_mean: c.ok_mean.0,
                });
            }
        }
    }
}

impl<'a> CheatAlgorithm<'a> for MlBacktrack {
    fn algorithm_name(&self) -> &str {
        "fidoo/ml_backtrack"
    }

    fn tooltip(&self) -> &'static str {
        "Scores each player's sniper rifle and revolver hits over the whole demo with a learned model of hitscan backtrack use (hits only possible at a lag compensation rewind several ticks away from the player's typical one, typical rewind far above the scoreboard ping) and marks the most backtracked hits."
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
        for t in [MessageType::GameEvent, MessageType::PacketEntities, MessageType::TempEntities] {
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
        if let Message::PacketEntities(msg) = message {
            self.track_dots(msg, parser_state);
            return Ok(vec![]);
        }
        if let Message::TempEntities(te) = message {
            let tick = u32::from(tick);
            for event in &te.events {
                if parser_state.server_classes.get(usize::from(event.class_id)).is_none_or(|c| c.name.as_str() != "CTEFireBullets") {
                    continue;
                }
                let player = event.props.iter().find(|p| p.identifier == BULLETS_PLAYER).and_then(|p| i64::try_from(&p.value).ok());
                if let Some(player) = player {
                    // m_iPlayer is the entity index - 1
                    let f = self.fires.entry(player as u32 + 1).or_default();
                    f.push_back(tick);
                    while f.front().is_some_and(|&t| t + 16 < tick) {
                        f.pop_front();
                    }
                }
            }
            return Ok(vec![]);
        }
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
        if ap.team == vp.team || !ap.in_pvs || !vp.in_pvs {
            return Ok(vec![]);
        }
        let weapon = state.get_player_weapon(ap);
        // bleed ticks are credited to the active weapon and come every 0.5 s
        if h.custom == TF_DMG_CUSTOM_BLEEDING || !is_precise(&weapon) || self.last_hit.get(&a).is_some_and(|&t| tick < t + MIN_HIT_SPACING) {
            return Ok(vec![]);
        }
        self.last_hit.insert(a, tick);
        let since = |p: &u64| self.jg.spawned(p, tick).min(self.jg.teleported(p, tick));
        if since(&a) < MIN_SINCE_SPAWN || since(&v) < MIN_SINCE_SPAWN.max(MAX_LAG as u32 + 8) {
            return Ok(vec![]);
        }
        self.pending.push(PendingHit { tick, attacker: a, victim: v, weapon, ping: ap.ping, headshot: h.custom == 1, custom: h.custom, ent: u32::from(ap.entity) });
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
            let ent = u32::from(p.entity);
            let dot = self.dots.values().find(|d| d.owner == ent).map(|d| {
                let t = [d.pos[0] - eye[0], d.pos[1] - eye[1], d.pos[2] - eye[2]];
                (t[1].atan2(t[0]).to_degrees(), (-t[2]).atan2(t[0].hypot(t[1])).to_degrees())
            });
            let (yaw, pitch) = dot.unwrap_or((p.view_angle, p.pitch_angle));
            let head = head_height(p, ducking);
            let h = self.history.entry(sid).or_default();
            h.push_back(Sample { tick, pos, eye, yaw, pitch, ducking, head, dot: dot.is_some(), on_ground: p.flags & FL_ONGROUND != 0 });
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
        let mut players: Vec<u64> = self.hits.iter().map(|h| h.attacker).collect();
        players.sort_unstable();
        players.dedup();
        let mut detections = Vec::new();
        for player in players {
            let hits: Vec<&HitRecord> = self.hits.iter().filter(|h| h.attacker == player).collect();
            let Some((k0, features, kinds)) = unit_features(&hits, min_hits) else { continue };
            let score = model.map_or(f32::NAN, |m| m.unit.probability(&features));
            self.units.push(UnitRecord { player, k0, features: features.clone(), score });
            if !(score >= min_score) {
                continue;
            }
            // the backtrack-like hits, farthest from the typical rewind first
            let k = k0 as isize;
            let mut bt: Vec<(usize, isize)> = (0..hits.len())
                .filter(|&i| kinds[i].0 || kinds[i].1)
                .map(|i| {
                    let off = (0..LAGS as isize).filter(|&l| hits[i].ok >> l & 1 == 1).min_by_key(|&l| (l - k).abs()).unwrap_or(k);
                    (i, off - k)
                })
                .collect();
            bt.sort_by_key(|&(i, off)| (std::cmp::Reverse(off.abs()), hits[i].tick));
            let n_bt = bt.len();
            for (i, off) in bt.into_iter().take(max_detections) {
                let h = hits[i];
                detections.push(Detection {
                    tick: h.tick,
                    algorithm: "fidoo/ml_backtrack".to_string(),
                    player,
                    data: json!({
                        "score": score,
                        "backtrack_hits": n_bt,
                        "hits": features[0],
                        "typical_rewind_ticks": k0,
                        "hit_rewind_offset_ticks": off,
                        "ping": h.ping,
                        "excess_ticks": features[4],
                        "weapon": h.weapon,
                        "target": h.victim,
                        "distance": h.dist.round(),
                        "headshot": !h.head_err.is_empty(),
                        "sniper_dot": h.dot,
                        "update_gap_ticks": h.att_gap,
                    }),
                });
            }
        }
        Ok(detections)
    }
}
