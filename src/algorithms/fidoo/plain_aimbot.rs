// Detection of plain (visible) aimbot in POV demos.
//
// Plain aimbot writes the aim angles into both the usercmd and the engine view angles, so the
// networked eye angles of the cheater visibly move onto the target and stay there (no snapback,
// unlike silent aim). Amalgam's "Plain" mode snaps on attack ticks, "Locking" snaps and tracks
// every tick (including while waiting for headshot/charge), "Smooth" lerps towards the target by
// a constant fraction every tick, and RijiN lerps by `1 - smoothing / 100`.
//
// Signatures (see docs/plain_aimbot.md):
// - snap:      an impulsive one-update jump of the view (rate-normalized stillness before and
//              after, gap-aware for 33fps/high-ping demos) with joint (target, lag) landing and
//              follow verification.
// - shot_snap: a hitscan hit (`PlayerHurt`) whose victim was acquired by an impulsive snap onto
//              that exact victim — either on the shot tick (dead-still low-FOV or mouse1-bound
//              snap-and-leave) or up to ~2 s earlier followed by a tight hold onto the hit
//              ("wait for headshot" / "wait for charge").
// - corner:    two consecutive large view steps in sharply different directions landing on a
//              target, or a sharp switch directly between two distinct targets (including back
//              and forth between two enemies).
// - lock:      sustained tight aim on a moving target (gap-resilient): either high view motion
//              at tiny error (classic lock) or target-correlated tracking with low residual while
//              firing (long-range / smooth-tracking lock).
// - hit:       a single hitscan hit that the fidoo/ml_aimbot per-hit model scores as very
//              aimbot-like.
//
// Projectile weapons are aimed at the target's predicted position (extrapolated by the projectile's
// flight time), splash weapons at the feet, arcing projectiles against the flat ballistic arc.
//
// Eye angles of other players are networked with ~0.35° yaw / ~0.71° pitch precision and only
// change when the player's simulation time advances (typically every 2–5 ticks), so all checks
// operate on simtime samples.

use std::collections::{BTreeMap, HashMap, VecDeque};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::algorithms::fidoo::ml_aimbot::{self, HitRecord, MlAimbot};
use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::resolve_recorder_sid;
use crate::util::nocrex::jankguard::JankGuard;

const SAMPLE_HISTORY: usize = 128;
/// Samples after a jump used to check where the view went next
const FOLLOW_SAMPLES: usize = 3;
const POSITION_HISTORY: usize = 160;
const TELEPORT_DIST_SQ: f32 = 256.0 * 256.0;
// Vertical extent of the aimable body column above the origin
const BODY_BOTTOM: f32 = 5.0;
const BODY_TOP_STANDING: f32 = 80.0;
const BODY_TOP_DUCKING: f32 = 60.0;
const BODY_HALF_WIDTH: f32 = 24.0;
/// Splash weapons are aimed at the feet / the ground below the target
const SPLASH_BOTTOM: f32 = -24.0;
const FL_DUCKING: u32 = 1 << 1;
const TICKS_PER_SECOND: f32 = 66.67;
/// Ticks over which a target's velocity is estimated for projectile prediction
const VELOCITY_TICKS: i64 = 6;
/// Longest projectile flight time that is extrapolated
const MAX_FLIGHT_TICKS: f32 = 100.0;
const WORLD_GRAVITY: f32 = 800.0;
const MAX_INTERP_SPAN: u32 = 8;
/// Delay after a PlayerHurt before evaluating `shot_snap` so post-shot samples are present
const HIT_FOLLOW_TICKS: u32 = 6;
const MIN_PENDING_HIT_SPACING: u32 = 6;
const TF_DMG_CUSTOM_HEADSHOT: u16 = 1;
const TF_DMG_CUSTOM_HEADSHOT_DECAP: u16 = 51;
const TF_DMG_CUSTOM_BURNING: u16 = 3;
const TF_DMG_CUSTOM_BLEEDING: u16 = 34;

#[derive(Clone, Copy, Debug, PartialEq)]
enum AimModel {
    Hitscan,
    /// Aimed at the predicted position. `gravity`: fraction of world gravity acting on the projectile
    /// (aimed higher, along the ballistic arc). `splash`: aimed at the feet. `heals_team`: also shot
    /// at teammates.
    Projectile { speed: f32, gravity: f32, splash: bool, heals_team: bool },
    /// Flame throwers (legit pyros look the same at this resolution), medi guns, tools, throwables and
    /// grenade launchers (steep arc, aim too ambiguous)
    Excluded,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    tick: u32,
    /// Decoded simulation tick
    sim_tick: i64,
    yaw: f32,
    pitch: f32,
    /// `view_dir(yaw, pitch)`, computed once per sample
    dir: (f32, f32, f32),
    eye: Vector,
    team: u8,
    fired: bool,
    model: AimModel,
    melee: bool,
}

#[derive(Clone, Copy, Debug)]
struct PositionRecord {
    tick: u32,
    position: Vector,
    team: u8,
    ducking: bool,
    class: u8,
    life_id: u32,
}

struct PendingShotHit {
    tick: u32,
    attacker: u64,
    victim: u64,
    headshot: bool,
    weapon: String,
    class: &'static str,
}

#[derive(Default)]
struct PlayerData {
    samples: VecDeque<Sample>,
    positions: VecDeque<PositionRecord>,
    last_sim_tick: Option<i64>,
    last_seen_tick: u32,
    /// Lock runs keyed by (target, lag). Ordered maps keep tie-breaks between equally long runs
    /// (lowest lag wins) and the order of finished locks the same on every run.
    locks: BTreeMap<(u64, i8), LockRun>,
    /// Last detection tick per detection type
    last_detection: HashMap<&'static str, u32>,
    /// Longest qualifying ended run per target, emitted once no run on that target is alive
    pending_locks: BTreeMap<u64, (i8, LockRun)>,
    /// `hit` detections of this player in the demo (capped at `hit_max_detections`, never reset)
    hit_detections: u32,
    /// Last queued `PendingShotHit` tick per victim to merge pellets/sprays
    last_pending_hit: HashMap<u64, u32>,
    /// Incremented whenever samples reset (death, teleport, PVS gap) so `position_at` never
    /// interpolates across a respawn/teleport while keeping pre-death positions for delayed checks
    life_id: u32,
}

#[derive(Clone, Copy, Default)]
struct LockRun {
    start_tick: u32,
    last_tick: u32,
    samples: u32,
    moving_samples: u32,
    view_path: f32,
    target_path: f32,
    resid_sum: f32,
    last_tdir: Option<(f32, f32, f32)>,
    error_sum: f32,
    max_error: f32,
    distance_sum: f32,
    fired: bool,
}

/// A possible target and its recent position records.
struct Target<'p> {
    sid: u64,
    ally: bool,
    positions: &'p VecDeque<PositionRecord>,
}

/// Best joint landing + follow match of a snap at one sample.
struct SnapMatch {
    error: f32,
    error_before: f32,
    target: usize,
    lag: i32,
    distance: f32,
    head_miss_hu: f32,
    follow: Vec<f32>,
    score: f32,
}

/// Best `shot_snap` match for a hitscan `PlayerHurt`.
struct ShotSnapMatch {
    tier: &'static str,
    snap_tick: u32,
    dt_to_hit: u32,
    step: f32,
    step_rate: f32,
    pre: f32,
    post: f32,
    err_b: f32,
    err_a: f32,
    err_hit: f32,
    hold_med_err: f32,
    head_miss_hu: f32,
    distance: f32,
    lag: i32,
    tgap: u32,
    sgap: i64,
    rank: f32,
}

#[derive(Default)]
pub struct PlainAimbot {
    params: Parameters,
    jg: JankGuard,
    // Ordered by SteamID so targets are always tried in the same order (ties keep the first).
    players: BTreeMap<u64, PlayerData>,
    /// Aim model and melee flag per weapon name (both depend on the name only)
    weapon_models: HashMap<String, (AimModel, bool)>,
    /// Hit features of fidoo/ml_aimbot (its own hit collection and feature code), for `hit` detections
    hit_source: MlAimbot,
    /// Recent hitscan `PlayerHurt` events waiting `HIT_FOLLOW_TICKS` before `shot_snap` evaluation
    pending_hits: Vec<PendingShotHit>,
    recorder: Option<u64>,
}

fn normalize_pitch(pitch: f32) -> f32 {
    if pitch > 180.0 {
        pitch - 360.0
    } else {
        pitch
    }
}

fn view_dir(yaw: f32, pitch: f32) -> (f32, f32, f32) {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    (p.cos() * y.cos(), p.cos() * y.sin(), -p.sin())
}

fn angle_between(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let la = (a.0 * a.0 + a.1 * a.1 + a.2 * a.2).sqrt();
    let lb = (b.0 * b.0 + b.1 * b.1 + b.2 * b.2).sqrt();
    if la < 1e-6 || lb < 1e-6 {
        return 180.0;
    }
    // atan2 keeps precision for tiny angles, unlike acos
    let cross = (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    );
    let c = (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt();
    let d = a.0 * b.0 + a.1 * b.1 + a.2 * b.2;
    c.atan2(d).to_degrees()
}

fn view_step(a: &Sample, b: &Sample) -> f32 {
    angle_between(a.dir, b.dir)
}

/// Normalize a neighbouring step `raw` over `sgap` sim ticks to an equivalent 2-tick step so choppy
/// demos (4–6 tick sim gaps) do not artificially inflate pre/post motion while keeping 1–2 tick gaps
/// identical to raw steps.
fn norm_step(raw: f32, sgap: i64) -> f32 {
    let g = sgap.clamp(1, 8) as f32;
    raw * (2.0 / g).min(1.0)
}

/// Eye height of a player class (standing), ducking players use 45 units.
fn eye_position(player: &Player) -> Vector {
    let height = if player.flags & FL_DUCKING != 0 {
        45.0
    } else {
        match player.class as u8 {
            1 => 65.0,
            2 | 5 | 6 | 8 => 75.0,
            _ => 68.0,
        }
    };
    Vector {
        x: player.position.x,
        y: player.position.y,
        z: player.position.z + height,
    }
}

fn head_height(class: u8, ducking: bool) -> f32 {
    if ducking {
        return 48.0;
    }
    match class {
        6 => 76.0, // Heavy
        2 => 74.0, // Sniper
        5 => 73.0, // Medic
        8 => 72.0, // Spy
        7 => 67.0, // Pyro
        9 => 66.0, // Engineer
        _ => 68.0,
    }
}

/// Perpendicular miss distance (HU) from the view ray to the target's approximate head point.
fn head_miss_hu(eye: Vector, dir: (f32, f32, f32), rec: &PositionRecord) -> f32 {
    let hx = rec.position.x - eye.x;
    let hy = rec.position.y - eye.y;
    let hz = rec.position.z + head_height(rec.class, rec.ducking) - eye.z;
    let proj = hx * dir.0 + hy * dir.1 + hz * dir.2;
    if proj <= 1.0 {
        return 999.0;
    }
    let cx = hx - proj * dir.0;
    let cy = hy - proj * dir.1;
    let cz = hz - proj * dir.2;
    (cx * cx + cy * cy + cz * cz).sqrt()
}

/// Angular error between the view ray and the closest point of the target's vertical column
/// `[target.z + bottom, target.z + top]`, plus the distance to the target and the unit vector to
/// that aimed point.
fn column_error_with_dir(
    eye: Vector,
    dir: (f32, f32, f32),
    target: Vector,
    bottom: f32,
    top: f32,
) -> Option<(f32, f32, (f32, f32, f32))> {
    let dx = target.x - eye.x;
    let dy = target.y - eye.y;
    let horizontal = dx.hypot(dy);
    let dir_h = dir.0.hypot(dir.1);
    if horizontal < 16.0 || dir_h < 1e-4 {
        return None;
    }
    let ray_z = eye.z + dir.2 / dir_h * horizontal;
    let z = ray_z.clamp(target.z + bottom, target.z + top);
    let to = (dx, dy, z - eye.z);
    let dist = (horizontal * horizontal + to.2 * to.2).sqrt();
    if dist < 1e-4 {
        return None;
    }
    let tdir = (to.0 / dist, to.1 / dist, to.2 / dist);
    Some((angle_between(dir, to), dist, tdir))
}

fn column_error(eye: Vector, dir: (f32, f32, f32), target: Vector, bottom: f32, top: f32) -> Option<(f32, f32)> {
    column_error_with_dir(eye, dir, target, bottom, top).map(|(e, d, _)| (e, d))
}

/// Position of a player at `tick`, linearly interpolated between records.
fn position_at(positions: &VecDeque<PositionRecord>, tick: i64) -> Option<PositionRecord> {
    // Records are strictly increasing in tick
    let newer_idx = positions.partition_point(|r| (r.tick as i64) < tick);
    if newer_idx == positions.len() {
        return None;
    }
    let newer = positions[newer_idx];
    if newer.tick as i64 == tick {
        return Some(newer);
    }
    if newer_idx == 0 {
        return None;
    }
    let older = positions[newer_idx - 1];
    if older.life_id != newer.life_id {
        return None;
    }
    let span = newer.tick.saturating_sub(older.tick);
    if span == 0 || span > MAX_INTERP_SPAN {
        return None;
    }
    let f = (tick - older.tick as i64) as f32 / span as f32;
    Some(PositionRecord {
        tick: tick as u32,
        position: Vector {
            x: older.position.x + (newer.position.x - older.position.x) * f,
            y: older.position.y + (newer.position.y - older.position.y) * f,
            z: older.position.z + (newer.position.z - older.position.z) * f,
        },
        team: newer.team,
        ducking: newer.ducking,
        class: newer.class,
        life_id: newer.life_id,
    })
}

/// Margin (degrees) of the `hitscan_error_exceeds` bound, far above the f32 rounding of the angles.
const HITSCAN_BOUND_MARGIN: f32 = 0.05;

/// Whether `model_error` of a hitscan sample is above `max_error` at every lag `0..=max_lag` (a
/// performance shortcut, it never skips a lag that could pass).
fn hitscan_error_exceeds(sample: &Sample, positions: &VecDeque<PositionRecord>, max_lag: u32, max_error: f32) -> bool {
    let first_tick = sample.tick as i64 - max_lag as i64 - MAX_INTERP_SPAN as i64;
    let last_tick = sample.tick as i64 + MAX_INTERP_SPAN as i64;
    let mut min = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for record in positions.iter().rev().take_while(|r| r.tick as i64 >= first_tick) {
        if record.tick as i64 > last_tick {
            continue;
        }
        let p = record.position;
        if !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite()) {
            return false;
        }
        min = (min.0.min(p.x), min.1.min(p.y), min.2.min(p.z));
        max = (max.0.max(p.x), max.1.max(p.y), max.2.max(p.z));
        any = true;
    }
    let eye = sample.eye;
    if !any || !(eye.x.is_finite() && eye.y.is_finite() && eye.z.is_finite()) {
        return false;
    }
    let (min_z, max_z) = (min.2 + BODY_BOTTOM, max.2 + BODY_TOP_STANDING);
    let half = ((max.0 - min.0) * 0.5, (max.1 - min.1) * 0.5, (max_z - min_z) * 0.5);
    let radius = (half.0 * half.0 + half.1 * half.1 + half.2 * half.2).sqrt();
    let to_centre = (min.0 + half.0 - eye.x, min.1 + half.1 - eye.y, min_z + half.2 - eye.z);
    let distance = (to_centre.0 * to_centre.0 + to_centre.1 * to_centre.1 + to_centre.2 * to_centre.2).sqrt();
    if !(distance > radius + 1.0) {
        return false;
    }
    let lower_bound = angle_between(sample.dir, to_centre) - (radius / distance).asin().to_degrees();
    lower_bound > max_error + HITSCAN_BOUND_MARGIN
}

/// Error of the view against the flat ballistic arcs (projectile `speed`, gravity `g`) that hit the
/// target column `[target.z + 5, target.z + top]`.
fn ballistic_error(sample: &Sample, target: Vector, speed: f32, g: f32, top: f32) -> Option<(f32, f32)> {
    let dx = target.x - sample.eye.x;
    let dy = target.y - sample.eye.y;
    let horizontal = dx.hypot(dy);
    if horizontal < 16.0 {
        return None;
    }
    let elevation = |h: f32| -> Option<f32> {
        let v2 = speed * speed;
        let disc = v2 * v2 - g * (g * horizontal * horizontal + 2.0 * h * v2);
        (disc >= 0.0).then(|| ((v2 - disc.sqrt()) / (g * horizontal)).atan().to_degrees())
    };
    let low = elevation(target.z + BODY_BOTTOM - sample.eye.z)?;
    let high = elevation(target.z + top - sample.eye.z).unwrap_or(low);
    let view_elevation = -sample.pitch;
    let dpitch = if view_elevation < low {
        low - view_elevation
    } else if view_elevation > high {
        view_elevation - high
    } else {
        0.0
    };
    let bearing = dy.atan2(dx).to_degrees();
    let dyaw = ((sample.yaw - bearing + 180.0).rem_euclid(360.0) - 180.0).abs() * sample.pitch.to_radians().cos();
    Some((dyaw.hypot(dpitch), horizontal))
}

/// Aim error of a sample against a target seen `lag` ticks ago, for the sample's weapon.
/// Returns `(error_deg, distance, target_unit_dir_opt, rec)`.
fn model_error_full(sample: &Sample, positions: &VecDeque<PositionRecord>, lag: i32) -> Option<(f32, f32, Option<(f32, f32, f32)>, PositionRecord)> {
    let at = sample.tick as i64 - lag as i64;
    let rec = position_at(positions, at)?;
    let top = if rec.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING };
    if lag < 0 {
        let bottom = match sample.model {
            AimModel::Projectile { splash: true, .. } => SPLASH_BOTTOM,
            _ => BODY_BOTTOM,
        };
        let (e, d, tdir) = column_error_with_dir(sample.eye, sample.dir, rec.position, bottom, top)?;
        return Some((e, d, Some(tdir), rec));
    }
    match sample.model {
        AimModel::Projectile { speed, gravity, splash, .. } => {
            let older = position_at(positions, at - VELOCITY_TICKS).unwrap_or(rec);
            let span = (rec.tick as i64 - older.tick as i64).max(1) as f32;
            let vel = rec.position - older.position;
            let vel = Vector { x: vel.x / span, y: vel.y / span, z: 0.0 };
            let mut predicted = rec.position;
            for _ in 0..2 {
                let d = predicted - sample.eye;
                let flight = ((d.x * d.x + d.y * d.y + d.z * d.z).sqrt() / speed * TICKS_PER_SECOND).min(MAX_FLIGHT_TICKS);
                predicted = Vector {
                    x: rec.position.x + vel.x * flight,
                    y: rec.position.y + vel.y * flight,
                    z: rec.position.z,
                };
            }
            if gravity > 0.0 {
                let (e, d) = ballistic_error(sample, predicted, speed, gravity * WORLD_GRAVITY, top)?;
                Some((e, d, None, rec))
            } else {
                let bottom = if splash { SPLASH_BOTTOM } else { BODY_BOTTOM };
                let (e, d, tdir) = column_error_with_dir(sample.eye, sample.dir, predicted, bottom, top)?;
                Some((e, d, Some(tdir), rec))
            }
        }
        _ => {
            let (e, d, tdir) = column_error_with_dir(sample.eye, sample.dir, rec.position, BODY_BOTTOM, top)?;
            Some((e, d, Some(tdir), rec))
        }
    }
}

fn model_error(sample: &Sample, target: &Target, lag: i32) -> Option<(f32, f32)> {
    model_error_full(sample, target.positions, lag).map(|(e, d, _, _)| (e, d))
}

fn hitscan_column_error(sample: &Sample, positions: &VecDeque<PositionRecord>, lag: i32) -> Option<(f32, f32, PositionRecord)> {
    let rec = position_at(positions, sample.tick as i64 - lag as i64)?;
    let top = if rec.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING };
    let (e, d) = column_error(sample.eye, sample.dir, rec.position, BODY_BOTTOM, top)?;
    Some((e, d, rec))
}

fn median_f32(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return f32::NAN;
    }
    vals.sort_by(f32::total_cmp);
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

fn player_sid(player: &Player) -> Option<u64> {
    let info = player.info.as_ref()?;
    if info.steam_id == "BOT" {
        return None;
    }
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

impl PlainAimbot {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("max_lag_ticks".to_string(), Parameter::Int(24)),
                ("projectile_model".to_string(), Parameter::Bool(true)),
                ("projectile_lead_ticks".to_string(), Parameter::Int(6)),
                ("lag_step".to_string(), Parameter::Int(2)),
                ("spawn_grace_ticks".to_string(), Parameter::Int(60)),
                ("max_sample_gap".to_string(), Parameter::Int(6)),
                ("max_simtime_gap".to_string(), Parameter::Int(5)),
                // snap
                ("check_snap".to_string(), Parameter::Bool(true)),
                ("snap_min_step".to_string(), Parameter::Float(10.0)),
                ("snap_max_step".to_string(), Parameter::Float(120.0)),
                ("snap_min_rate".to_string(), Parameter::Float(2.5)),
                ("snap_min_ratio".to_string(), Parameter::Float(12.0)),
                ("snap_max_error".to_string(), Parameter::Float(0.25)),
                ("snap_min_error_limit".to_string(), Parameter::Float(0.25)),
                ("snap_max_error_rel".to_string(), Parameter::Float(0.05)),
                ("snap_min_distance".to_string(), Parameter::Float(400.0)),
                ("snap_projectile_max_distance".to_string(), Parameter::Float(2200.0)),
                ("snap_follow_samples".to_string(), Parameter::Int(2)),
                ("snap_max_follow_error".to_string(), Parameter::Float(0.5)),
                ("snap_projectile_follow_samples".to_string(), Parameter::Int(1)),
                ("snap_projectile_max_follow_error".to_string(), Parameter::Float(1.0)),
                // shot_snap (hit-confirmed snap & snap-then-hold)
                ("check_shot_snap".to_string(), Parameter::Bool(true)),
                ("shot_snap_lookback_ticks".to_string(), Parameter::Int(150)),
                ("shot_snap_still_min_step".to_string(), Parameter::Float(7.0)),
                ("shot_snap_headshot_min_step".to_string(), Parameter::Float(3.5)),
                ("shot_snap_still_min_ratio".to_string(), Parameter::Float(10.0)),
                ("shot_snap_leave_min_step".to_string(), Parameter::Float(9.0)),
                ("shot_snap_leave_min_rate".to_string(), Parameter::Float(4.0)),
                ("shot_snap_leave_min_ratio".to_string(), Parameter::Float(14.0)),
                ("shot_snap_hold_min_step".to_string(), Parameter::Float(4.5)),
                ("shot_snap_hold_min_ratio".to_string(), Parameter::Float(12.0)),
                ("shot_snap_max_error".to_string(), Parameter::Float(0.25)),
                ("shot_snap_min_distance".to_string(), Parameter::Float(250.0)),
                // lock
                ("check_lock".to_string(), Parameter::Bool(true)),
                ("lock_max_gap".to_string(), Parameter::Int(8)),
                ("lock_max_error".to_string(), Parameter::Float(0.8)),
                ("lock_max_mean_error".to_string(), Parameter::Float(0.25)),
                ("lock_min_ticks".to_string(), Parameter::Int(20)),
                ("lock_min_view_path".to_string(), Parameter::Float(15.0)),
                ("lock_min_moving_fraction".to_string(), Parameter::Float(0.6)),
                ("lock_min_speed".to_string(), Parameter::Float(0.0)),
                ("lock_prefilter".to_string(), Parameter::Float(45.0)),
                // correlated tracking lock (tight pursuit or long sustained tracking while firing)
                ("check_lock_track".to_string(), Parameter::Bool(true)),
                ("lock_track_min_ticks".to_string(), Parameter::Int(28)),
                ("lock_track_min_target_path".to_string(), Parameter::Float(7.0)),
                ("lock_track_min_view_path".to_string(), Parameter::Float(12.0)),
                ("lock_track_max_mean_error".to_string(), Parameter::Float(0.20)),
                ("lock_track_max_resid_ratio".to_string(), Parameter::Float(0.35)),
                // corner
                ("check_corner".to_string(), Parameter::Bool(true)),
                ("corner_min_step".to_string(), Parameter::Float(10.0)),
                ("corner_max_step".to_string(), Parameter::Float(150.0)),
                ("corner_min_rate".to_string(), Parameter::Float(3.0)),
                ("corner_min_turn".to_string(), Parameter::Float(120.0)),
                ("corner_min_return".to_string(), Parameter::Float(0.3)),
                ("corner_max_target_error".to_string(), Parameter::Float(0.5)),
                ("corner_switch_max_error".to_string(), Parameter::Float(0.25)),
                // hit
                ("check_hit".to_string(), Parameter::Bool(true)),
                ("hit_min_score".to_string(), Parameter::Float(0.8)),
                ("hit_max_detections".to_string(), Parameter::Int(10)),
                ("min_detection_spacing".to_string(), Parameter::Int(66)),
            ]),
            ..Default::default()
        }
    }

    /// Melee weapons are swung around wildly, which produces sharp view corners.
    fn is_melee(name: &str) -> bool {
        let lw = name.to_lowercase();
        ["knife", "wrench", "bat", "bottle", "fist", "gloves", "shovel", "axe", "kukri", "bushwacka", "saw",
         "eyelander", "katana", "pan", "sword", "skullcutter", "equalizer", "escape plan", "jag", "gunslinger",
         "spy-cicle", "spycicle", "eternal reward", "kunai", "sharp dresser", "powerjack", "third degree",
         "candy cane", "sandman", "ubersaw", "vita-saw", "amputator", "persuader", "caber", "zatoichi",
         "claidheamh", "market gardener", "disciplinary", "holiday punch", "eviction notice", "fist of steel",
         "warrior", "big earner", "wrap assassin", "atomizer", "fan o'war", "unarmed combat", "homewrecker",
         "maul", "neon annihilator", "postal pummeler", "back scratcher", "lollichop", "ham shank",
         "conscientious", "freedom staff", "bat outta hell", "memory maker", "black rose", "bread bite",
         "necro smasher", "crossing guard", "prinny machete", "machete", "nessie", "golf club", "saxxy",
         "solemn vow", "southern hospitality", "boston basher", "sun-on-a-stick", "scotsman", "half-zatoichi",
         "tribalman", "shiv", "shahanshah", "melee", "eureka", "mackerel", "rune blade", "batsaber",
         "pain train", "axtinguisher", "volcano", "hot hand", "handshake", "headtaker", "wanga"]
            .iter()
            .any(|w| lw.contains(w))
    }

    fn aim_model(name: &str, projectile_model: bool) -> AimModel {
        let model = Self::projectile_aim_model(name);
        if projectile_model {
            return model;
        }
        let lw = name.to_lowercase();
        let grenade = ["grenade launcher", "iron bomber", "loch-n-load", "loose cannon", "stickybomb",
                       "scottish resistance", "quickiebomb"].iter().any(|w| lw.contains(w));
        match model {
            AimModel::Projectile { heals_team, .. } => AimModel::Projectile { speed: f32::INFINITY, gravity: 0.0, splash: false, heals_team },
            AimModel::Excluded if grenade => AimModel::Hitscan,
            other => other,
        }
    }

    fn projectile_aim_model(name: &str) -> AimModel {
        let lw = name.to_lowercase();
        let any = |words: &[&str]| words.iter().any(|w| lw.contains(w));
        let projectile = |speed: f32, gravity: f32, splash: bool| AimModel::Projectile { speed, gravity, splash, heals_team: false };
        if any(&["builder", "pda", "toolbox", "destruction", "disguise", "sapper", "red-tape", "invis watch",
                 "cloak and dagger", "dead ringer", "wrangler", "giger counter", "medi gun", "medigun", "kritzkrieg",
                 "quick-fix", "vaccinator", "jumper", "jarate", "mad milk", "gas passer", "mutated milk", "self-aware",
                 "grenade launcher", "iron bomber", "loch-n-load", "loose cannon", "stickybomb", "scottish resistance",
                 "quickiebomb", "flame", "degreaser", "backburner", "phlog", "rainblower", "nostromo", "dragon's fury"]) {
            AimModel::Excluded
        } else if any(&["crusader's crossbow", "crossbow"]) {
            AimModel::Projectile { speed: 2400.0, gravity: 0.2, splash: false, heals_team: true }
        } else if any(&["direct hit"]) {
            projectile(1980.0, 0.0, true)
        } else if any(&["liberty launcher"]) {
            projectile(1540.0, 0.0, true)
        } else if any(&["rocket launcher", "black box", "air strike", "beggar", "mangler", "original"]) {
            projectile(1100.0, 0.0, true)
        } else if any(&["righteous bison", "pomson"]) {
            projectile(1200.0, 0.0, false)
        } else if any(&["manmelter"]) {
            projectile(3000.0, 0.0, false)
        } else if any(&["flare gun", "detonator", "scorch shot"]) {
            projectile(2000.0, 0.3, false)
        } else if any(&["rescue ranger", "huntsman", "fortified compound"]) {
            projectile(2400.0, 0.2, false)
        } else if any(&["syringe gun", "blutsauger", "overdose"]) {
            projectile(1000.0, 0.3, false)
        } else {
            AimModel::Hitscan
        }
    }
}

struct Settings {
    max_lag: u32,
    projectile_model: bool,
    projectile_lead: u32,
    lag_step: u32,
    spawn_grace: u32,
    max_sample_gap: u32,
    max_simtime_gap: u16,
    check_snap: bool,
    snap_min_step: f32,
    snap_max_step: f32,
    snap_min_rate: f32,
    snap_min_ratio: f32,
    snap_max_error: f32,
    snap_min_error_limit: f32,
    snap_max_error_rel: f32,
    snap_min_distance: f32,
    snap_projectile_max_distance: f32,
    snap_follow_samples: usize,
    snap_max_follow_error: f32,
    snap_projectile_follow_samples: usize,
    snap_projectile_max_follow_error: f32,
    check_shot_snap: bool,
    shot_snap_lookback: u32,
    shot_snap_still_min_step: f32,
    shot_snap_headshot_min_step: f32,
    shot_snap_still_min_ratio: f32,
    shot_snap_leave_min_step: f32,
    shot_snap_leave_min_rate: f32,
    shot_snap_leave_min_ratio: f32,
    shot_snap_hold_min_step: f32,
    shot_snap_hold_min_ratio: f32,
    shot_snap_max_error: f32,
    shot_snap_min_distance: f32,
    check_corner: bool,
    corner_min_step: f32,
    corner_max_step: f32,
    corner_min_rate: f32,
    corner_min_turn: f32,
    corner_min_return: f32,
    corner_max_target_error: f32,
    corner_switch_max_error: f32,
    check_lock: bool,
    lock_max_gap: u32,
    lock_max_error: f32,
    lock_max_mean_error: f32,
    lock_min_ticks: u32,
    lock_min_view_path: f32,
    lock_min_moving_fraction: f32,
    lock_min_speed: f32,
    lock_prefilter: f32,
    check_lock_track: bool,
    lock_track_min_ticks: u32,
    lock_track_min_target_path: f32,
    lock_track_min_view_path: f32,
    lock_track_max_mean_error: f32,
    lock_track_max_resid_ratio: f32,
    check_hit: bool,
    hit_min_score: f32,
    hit_max_detections: u32,
    spacing: u32,
}

impl Settings {
    fn from(params: &Parameters) -> Self {
        let i = |n: &str| get_parameter_value::<i32>(params, n).max(0) as u32;
        let f = |n: &str| get_parameter_value::<f32>(params, n);
        let b = |n: &str| get_parameter_value::<bool>(params, n);
        Self {
            max_lag: i("max_lag_ticks"),
            projectile_model: b("projectile_model"),
            projectile_lead: i("projectile_lead_ticks"),
            lag_step: i("lag_step").max(1),
            spawn_grace: i("spawn_grace_ticks"),
            max_sample_gap: i("max_sample_gap").max(1),
            max_simtime_gap: i("max_simtime_gap").max(1) as u16,
            check_snap: b("check_snap"),
            snap_min_step: f("snap_min_step"),
            snap_max_step: f("snap_max_step"),
            snap_min_rate: f("snap_min_rate"),
            snap_min_ratio: f("snap_min_ratio"),
            snap_max_error: f("snap_max_error"),
            snap_min_error_limit: f("snap_min_error_limit"),
            snap_max_error_rel: f("snap_max_error_rel"),
            snap_min_distance: f("snap_min_distance"),
            snap_projectile_max_distance: f("snap_projectile_max_distance"),
            snap_follow_samples: (i("snap_follow_samples") as usize).min(FOLLOW_SAMPLES),
            snap_max_follow_error: f("snap_max_follow_error"),
            snap_projectile_follow_samples: (i("snap_projectile_follow_samples") as usize).min(FOLLOW_SAMPLES),
            snap_projectile_max_follow_error: f("snap_projectile_max_follow_error"),
            check_shot_snap: b("check_shot_snap"),
            shot_snap_lookback: i("shot_snap_lookback_ticks"),
            shot_snap_still_min_step: f("shot_snap_still_min_step"),
            shot_snap_headshot_min_step: f("shot_snap_headshot_min_step"),
            shot_snap_still_min_ratio: f("shot_snap_still_min_ratio"),
            shot_snap_leave_min_step: f("shot_snap_leave_min_step"),
            shot_snap_leave_min_rate: f("shot_snap_leave_min_rate"),
            shot_snap_leave_min_ratio: f("shot_snap_leave_min_ratio"),
            shot_snap_hold_min_step: f("shot_snap_hold_min_step"),
            shot_snap_hold_min_ratio: f("shot_snap_hold_min_ratio"),
            shot_snap_max_error: f("shot_snap_max_error"),
            shot_snap_min_distance: f("shot_snap_min_distance"),
            check_corner: b("check_corner"),
            corner_min_step: f("corner_min_step"),
            corner_max_step: f("corner_max_step"),
            corner_min_rate: f("corner_min_rate"),
            corner_min_turn: f("corner_min_turn"),
            corner_min_return: f("corner_min_return"),
            corner_max_target_error: f("corner_max_target_error"),
            corner_switch_max_error: f("corner_switch_max_error"),
            check_lock: b("check_lock"),
            lock_max_gap: i("lock_max_gap").max(2),
            lock_max_error: f("lock_max_error"),
            lock_max_mean_error: f("lock_max_mean_error"),
            lock_min_ticks: i("lock_min_ticks"),
            lock_min_view_path: f("lock_min_view_path"),
            lock_min_moving_fraction: f("lock_min_moving_fraction"),
            lock_min_speed: f("lock_min_speed"),
            lock_prefilter: f("lock_prefilter"),
            check_lock_track: b("check_lock_track"),
            lock_track_min_ticks: i("lock_track_min_ticks"),
            lock_track_min_target_path: f("lock_track_min_target_path"),
            lock_track_min_view_path: f("lock_track_min_view_path"),
            lock_track_max_mean_error: f("lock_track_max_mean_error"),
            lock_track_max_resid_ratio: f("lock_track_max_resid_ratio"),
            check_hit: b("check_hit"),
            hit_min_score: f("hit_min_score"),
            hit_max_detections: i("hit_max_detections"),
            spacing: i("min_detection_spacing"),
        }
    }
}

impl PlainAimbot {
    /// Lags to try for a weapon (coarse step for locks/projectiles).
    fn lags(cfg: &Settings, model: &AimModel) -> impl Iterator<Item = i32> {
        let lead = if matches!(model, AimModel::Projectile { .. }) { cfg.projectile_lead as i32 } else { 0 };
        (-lead..=cfg.max_lag as i32).step_by(cfg.lag_step as usize)
    }

    /// Fine lags (step 1 for hitscan, `lag_step` for projectiles) for rare snap/corner candidates.
    fn fine_lags(cfg: &Settings, model: &AimModel) -> impl Iterator<Item = i32> {
        let (lead, step) = if matches!(model, AimModel::Projectile { .. }) {
            (cfg.projectile_lead as i32, cfg.lag_step as usize)
        } else {
            (0, 1)
        };
        (-lead..=cfg.max_lag as i32).step_by(step)
    }

    /// Targets the sample's weapon can be aimed at.
    fn valid_target(sample: &Sample, target: &Target) -> bool {
        match sample.model {
            AimModel::Projectile { heals_team: true, .. } => true,
            _ => !target.ally,
        }
    }

    /// Cross-type spacing check so a single engagement cannot emit two detections (e.g. `snap` +
    /// `shot_snap`, or `snap` + `hit`) and artificially pass the 2-detection sliding window.
    fn can_flag_any(pdata: &mut PlayerData, kind: &'static str, tick: u32, spacing: u32) -> bool {
        if pdata.last_detection.values().any(|&t| t.abs_diff(tick) < spacing) {
            return false;
        }
        pdata.last_detection.insert(kind, tick);
        true
    }

    /// Jointly evaluates landing at `b` and follow-up samples at the SAME `(target, lag)`,
    /// preventing a quantized accidental lag at `b` from discarding the real lag's follow match.
    fn best_snap_match(
        cfg: &Settings,
        samples: &VecDeque<Sample>,
        j: usize,
        step: f32,
        max_error: f32,
        targets: &[Target],
    ) -> Option<SnapMatch> {
        let (a, b) = (&samples[j - 1], &samples[j]);
        let projectile = matches!(b.model, AimModel::Projectile { .. });
        let fired = (j - 1..=j + FOLLOW_SAMPLES).any(|idx| samples[idx].fired);
        if projectile && !fired {
            return None;
        }
        let (follow_samples, follow_error) = if projectile {
            (cfg.snap_projectile_follow_samples, cfg.snap_projectile_max_follow_error)
        } else {
            (cfg.snap_follow_samples, cfg.snap_max_follow_error)
        };
        let mut best: Option<SnapMatch> = None;

        for (ti, target) in targets.iter().enumerate() {
            if !Self::valid_target(b, target) {
                continue;
            }
            if !projectile && hitscan_error_exceeds(b, target.positions, cfg.max_lag, max_error) {
                continue;
            }
            for lag in Self::fine_lags(cfg, &b.model) {
                let Some((error, distance, _, rec)) = model_error_full(b, target.positions, lag) else { continue };
                if error > max_error || distance < cfg.snap_min_distance {
                    continue;
                }
                if projectile && distance > cfg.snap_projectile_max_distance {
                    continue;
                }
                let Some((error_before, _)) = model_error(a, target, lag) else { continue };
                if error_before < 0.5 * step {
                    continue;
                }
                let follow_limit = follow_error.max(error + 0.36);
                let mut follow = Vec::with_capacity(FOLLOW_SAMPLES);
                let mut ok = true;
                let mut follow_sum = 0.0f32;
                for k in 1..=FOLLOW_SAMPLES {
                    let fe = model_error(&samples[j + k], target, lag).map_or(99.0, |x| x.0);
                    if k <= follow_samples {
                        if fe > follow_limit {
                            ok = false;
                            break;
                        }
                        follow_sum += fe;
                    }
                    follow.push(fe);
                }
                if !ok {
                    continue;
                }
                let follow_mean = if follow_samples > 0 { follow_sum / follow_samples as f32 } else { 0.0 };
                let score = error + 0.5 * follow_mean;
                if best.as_ref().is_none_or(|x| score < x.score) {
                    best = Some(SnapMatch {
                        error,
                        error_before,
                        target: ti,
                        lag,
                        distance,
                        head_miss_hu: head_miss_hu(b.eye, b.dir, &rec),
                        follow,
                        score,
                    });
                }
            }
        }
        best
    }

    /// Impulsive jump into the sample `FOLLOW_SAMPLES` samples ago: gap-aware, rate-normalized
    /// stillness before and after, joint (target, lag) landing and follow check.
    fn check_snap(cfg: &Settings, samples: &VecDeque<Sample>, targets: &[Target]) -> Option<serde_json::Value> {
        let n = samples.len();
        if !cfg.check_snap || n < FOLLOW_SAMPLES + 4 {
            return None;
        }
        let j = n - 1 - FOLLOW_SAMPLES;
        let (before, a, b, after) = (&samples[j - 2], &samples[j - 1], &samples[j], &samples[j + 1]);
        let tgap = b.tick.saturating_sub(a.tick);
        let sgap = b.sim_tick - a.sim_tick;
        // Reject melee swings and choke/burst packets where simtime advances faster than demo ticks
        if b.melee
            || b.model == AimModel::Excluded
            || tgap > cfg.max_sample_gap
            || !(1..=cfg.max_simtime_gap as i64).contains(&sgap)
            || sgap > tgap as i64 + 1
            || (sgap > 4 && tgap < 4)
        {
            return None;
        }
        let step = view_step(a, b);
        let step_rate = step / sgap as f32;
        if step > cfg.snap_max_step || step < cfg.snap_min_step || step_rate < cfg.snap_min_rate {
            return None;
        }
        let pre_raw = view_step(before, a);
        let post_raw = view_step(b, after);
        let pre = norm_step(pre_raw, a.sim_tick - before.sim_tick);
        let post = norm_step(post_raw, after.sim_tick - b.sim_tick);
        if step < cfg.snap_min_ratio * pre.max(post).max(0.36) {
            return None;
        }
        let max_error = cfg.snap_max_error.min(cfg.snap_min_error_limit.max(cfg.snap_max_error_rel * step));
        let m = Self::best_snap_match(cfg, samples, j, step, max_error, targets)?;
        let target = &targets[m.target];
        let projectile = matches!(b.model, AimModel::Projectile { .. });
        Some(json!({
            "type": "snap",
            "tick": b.tick,
            "target": target.sid,
            "ally": target.ally,
            "snap_deg": step,
            "snap_rate_deg_per_sim": step_rate,
            "pre_step_deg": pre_raw,
            "post_step_deg": post_raw,
            "error_deg": m.error,
            "error_before_deg": m.error_before,
            "head_miss_hu": m.head_miss_hu,
            "lag_ticks": m.lag,
            "distance": m.distance,
            "sample_gap": tgap,
            "sim_gap": sgap,
            "follow_errors_deg": m.follow,
            "projectile": projectile,
            "fired": (j - 1..=j + FOLLOW_SAMPLES).any(|idx| samples[idx].fired),
        }))
    }

    /// Evaluates a hitscan `PlayerHurt` against the attacker's recent samples and that exact victim's
    /// position history to detect:
    /// - `on_shot_still`: dead-still low/medium-FOV snap right onto the victim at the shot tick,
    /// - `on_shot_leave`: impulsive mouse1-bound snap onto the victim on the shot tick (no post-stillness required),
    /// - `snap_hold`: impulsive snap onto the victim up to `shot_snap_lookback` ticks earlier, held tightly until the hit.
    fn check_shot_snap(
        cfg: &Settings,
        hit: &PendingShotHit,
        samples: &VecDeque<Sample>,
        victim_positions: &VecDeque<PositionRecord>,
    ) -> Option<ShotSnapMatch> {
        if !cfg.check_shot_snap || samples.len() < 3 || victim_positions.is_empty() {
            return None;
        }
        // Find the attacker sample at or right before the hurt tick
        let h_idx = samples.iter().rposition(|s| s.tick <= hit.tick)?;
        if hit.tick.saturating_sub(samples[h_idx].tick) > 6 {
            return None;
        }
        let h_sample = &samples[h_idx];
        if h_sample.melee || h_sample.model != AimModel::Hitscan {
            return None;
        }

        let mut best: Option<ShotSnapMatch> = None;
        let min_tick = hit.tick.saturating_sub(cfg.shot_snap_lookback);

        for k in (2..=h_idx).rev() {
            let b = &samples[k];
            if b.tick < min_tick {
                break;
            }
            if b.melee || b.model != AimModel::Hitscan {
                continue;
            }
            let (before, a) = (&samples[k - 2], &samples[k - 1]);
            let tgap = b.tick.saturating_sub(a.tick);
            let sgap = b.sim_tick - a.sim_tick;
            if tgap > cfg.max_sample_gap
                || !(1..=cfg.max_simtime_gap as i64).contains(&sgap)
                || sgap > tgap as i64 + 1
                || (sgap > 4 && tgap < 4)
            {
                continue;
            }
            let step = view_step(a, b);
            let min_any = cfg
                .shot_snap_still_min_step
                .min(cfg.shot_snap_headshot_min_step)
                .min(cfg.shot_snap_hold_min_step)
                .min(4.0);
            if step < min_any || step > cfg.snap_max_step {
                continue;
            }
            let step_rate = step / sgap as f32;
            let pre_raw = view_step(before, a);
            let pre = norm_step(pre_raw, a.sim_tick - before.sim_tick);
            let after = samples.get(k + 1);
            let post_raw = after.map_or(0.0, |af| view_step(b, af));
            let post = after.map_or(0.0, |af| norm_step(post_raw, af.sim_tick - b.sim_tick));
            let dt_to_hit = hit.tick.saturating_sub(b.tick);

            // Pre-check which tiers are even kinematically possible before scanning lags
            let still_min = if hit.headshot { cfg.shot_snap_headshot_min_step } else { cfg.shot_snap_still_min_step };
            let still_min_rate = if hit.headshot { 1.5 } else { 3.0 };
            let can_still = dt_to_hit <= 6
                && step >= still_min
                && step_rate >= still_min_rate
                && pre <= 0.40
                && post <= 0.40
                && pre_raw <= 0.72
                && post_raw <= 0.72
                && step >= cfg.shot_snap_still_min_ratio * pre.max(post).max(0.36);

            let can_leave = dt_to_hit <= 4
                && step >= cfg.shot_snap_leave_min_step
                && step_rate >= cfg.shot_snap_leave_min_rate
                && pre <= 1.2
                && pre_raw <= 1.5
                && step >= cfg.shot_snap_leave_min_ratio * pre.max(0.36);

            let hold_min = if hit.headshot { cfg.shot_snap_hold_min_step.min(4.0) } else { cfg.shot_snap_hold_min_step };
            let can_hold = (7..=cfg.shot_snap_lookback).contains(&dt_to_hit)
                && step >= hold_min
                && step_rate >= 2.0
                && pre_raw <= 1.1
                && post_raw <= 1.1
                && step >= cfg.shot_snap_hold_min_ratio * pre.max(post).max(0.36);

            if !(can_still || can_leave || can_hold) {
                continue;
            }
            if hitscan_error_exceeds(b, victim_positions, cfg.max_lag, cfg.shot_snap_max_error.max(0.35)) {
                continue;
            }

            for lag in 0..=cfg.max_lag as i32 {
                let Some((err_b, dist, rec_b)) = hitscan_column_error(b, victim_positions, lag) else { continue };
                if dist < 150.0 {
                    continue;
                }
                let ang_size = BODY_HALF_WIDTH.atan2(dist).to_degrees();
                let leave_err_lim = (0.20 * ang_size).clamp(0.18, 0.30);
                if err_b > cfg.shot_snap_max_error.max(leave_err_lim) {
                    continue;
                }
                let Some((err_a, _, _)) = hitscan_column_error(a, victim_positions, lag) else { continue };
                if err_a < 0.55 * step {
                    continue;
                }
                let Some((err_hit, _, rec_h)) = hitscan_column_error(h_sample, victim_positions, lag) else { continue };
                let head_hu = head_miss_hu(h_sample.eye, h_sample.dir, &rec_h);

                // Tier A: dead-still on-shot snap
                if can_still && dist >= 400.0 && err_b <= cfg.shot_snap_max_error && err_hit <= 0.35 {
                    let head_ok = !hit.headshot || step >= 5.0 || head_miss_hu(b.eye, b.dir, &rec_b) <= 24.0;
                    if head_ok {
                        let rank = step / pre.max(post).max(0.36) + 5.0;
                        if best.as_ref().is_none_or(|x| rank > x.rank) {
                            best = Some(ShotSnapMatch {
                                tier: "on_shot_still",
                                snap_tick: b.tick,
                                dt_to_hit,
                                step,
                                step_rate,
                                pre: pre_raw,
                                post: post_raw,
                                err_b,
                                err_a,
                                err_hit,
                                hold_med_err: err_b,
                                head_miss_hu: head_hu,
                                distance: dist,
                                lag,
                                tgap,
                                sgap,
                                rank,
                            });
                        }
                    }
                }

                // Tier B: impulsive snap-on-shot without post stillness (mouse1-bound)
                let leave_dist_ok = dist >= cfg.shot_snap_min_distance || (step >= 15.0 && step_rate >= 6.0);
                if can_leave && leave_dist_ok && err_b <= leave_err_lim && err_a >= 0.65 * step && err_hit <= leave_err_lim.max(0.35) {
                    let rank = step / pre.max(0.36);
                    if best.as_ref().is_none_or(|x| rank > x.rank) {
                        best = Some(ShotSnapMatch {
                            tier: "on_shot_leave",
                            snap_tick: b.tick,
                            dt_to_hit,
                            step,
                            step_rate,
                            pre: pre_raw,
                            post: post_raw,
                            err_b,
                            err_a,
                            err_hit,
                            hold_med_err: err_b,
                            head_miss_hu: head_hu,
                            distance: dist,
                            lag,
                            tgap,
                            sgap,
                            rank,
                        });
                    }
                }

                // Tier C: snap-then-hold-then-hit
                if can_hold && dist >= 400.0 && err_b <= cfg.shot_snap_max_error && err_a >= 0.60 * step && err_hit <= 0.35 {
                    let hold_cap = (ang_size * 1.25).max(1.0);
                    let mut hold_errs = Vec::with_capacity(h_idx - k + 1);
                    let mut hold_ok = true;
                    for m in k..=h_idx {
                        let Some((em, _, _)) = hitscan_column_error(&samples[m], victim_positions, lag) else {
                            hold_ok = false;
                            break;
                        };
                        if em > hold_cap {
                            hold_ok = false;
                            break;
                        }
                        hold_errs.push(em);
                    }
                    if hold_ok && !hold_errs.is_empty() {
                        let med = median_f32(&mut hold_errs);
                        if med <= 0.40 {
                            let rank = step / pre.max(post).max(0.36) + 2.0;
                            if best.as_ref().is_none_or(|x| rank > x.rank) {
                                best = Some(ShotSnapMatch {
                                    tier: "snap_hold",
                                    snap_tick: b.tick,
                                    dt_to_hit,
                                    step,
                                    step_rate,
                                    pre: pre_raw,
                                    post: post_raw,
                                    err_b,
                                    err_a,
                                    err_hit,
                                    hold_med_err: med,
                                    head_miss_hu: head_hu,
                                    distance: dist,
                                    lag,
                                    tgap,
                                    sgap,
                                    rank,
                                });
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// Smallest aim error of a sample against any valid target and lag, returning `(error, sid, pos, dist)`.
    fn nearest_target_info(cfg: &Settings, sample: &Sample, targets: &[Target]) -> (f32, Option<(u64, Vector, f32)>) {
        let mut best = 180.0f32;
        let mut info = None;
        for target in targets.iter().filter(|t| Self::valid_target(sample, t)) {
            for lag in Self::fine_lags(cfg, &sample.model) {
                if let Some((e, d, _, rec)) = model_error_full(sample, target.positions, lag) {
                    if e < best {
                        best = e;
                        info = Some((target.sid, rec.position, d));
                    }
                }
            }
        }
        (best, info)
    }

    /// Sharp turn at speed over the three newest samples A -> B -> C: both steps are large and point
    /// in very different directions. Normally C must not simply return to A (silent aim snapback),
    /// unless A/C and B land tightly on two distinct enemies (target-to-target ping-pong).
    fn check_corner(cfg: &Settings, samples: &VecDeque<Sample>, targets: &[Target]) -> Option<serde_json::Value> {
        let n = samples.len();
        if n < 3 {
            return None;
        }
        let (a, b, c) = (&samples[n - 3], &samples[n - 2], &samples[n - 1]);
        if b.melee || c.melee || matches!(b.model, AimModel::Excluded) || matches!(c.model, AimModel::Excluded) {
            return None;
        }
        let sgap1 = b.sim_tick - a.sim_tick;
        let sgap2 = c.sim_tick - b.sim_tick;
        for (x, y, sg) in [(a, b, sgap1), (b, c, sgap2)] {
            let tg = y.tick.saturating_sub(x.tick);
            if tg > cfg.max_sample_gap
                || !(1..=cfg.max_simtime_gap as i64).contains(&sg)
                || sg > tg as i64 + 1
                || (sg > 4 && tg < 4)
            {
                return None;
            }
        }
        let fired = a.fired || b.fired || c.fired;
        if matches!(b.model, AimModel::Projectile { .. }) && !fired {
            return None;
        }
        let step = |x: &Sample, y: &Sample| {
            let dyaw = (y.yaw - x.yaw + 180.0).rem_euclid(360.0) - 180.0;
            (dyaw * ((x.pitch + y.pitch) * 0.5).to_radians().cos(), y.pitch - x.pitch)
        };
        let (v1, v2) = (step(a, b), step(b, c));
        let (m1, m2) = (v1.0.hypot(v1.1), v2.0.hypot(v2.1));
        if m1.min(m2) < cfg.corner_min_step || m1.max(m2) > cfg.corner_max_step {
            return None;
        }
        if (m1 / sgap1 as f32).min(m2 / sgap2 as f32) < cfg.corner_min_rate {
            return None;
        }
        let turn = ((v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2)).clamp(-1.0, 1.0).acos().to_degrees();
        if turn < cfg.corner_min_turn {
            return None;
        }
        let returned = view_step(a, c);
        let (error_b, info_b) = Self::nearest_target_info(cfg, b, targets);
        let (error_c, info_c) = Self::nearest_target_info(cfg, c, targets);

        let is_target_switch = if let (Some((sid_b, pos_b, dist_b)), Some((sid_c, pos_c, dist_c))) = (info_b, info_c) {
            let sep = (pos_b.x - pos_c.x).hypot(pos_b.y - pos_c.y);
            sid_b != sid_c
                && sep >= 250.0
                && dist_b >= 400.0
                && dist_c >= 400.0
                && error_b <= cfg.corner_switch_max_error
                && error_c <= cfg.corner_switch_max_error
        } else {
            false
        };

        if !is_target_switch {
            if !fired {
                return None;
            }
            if returned < cfg.corner_min_return * m1.min(m2) {
                return None;
            }
            if error_b.min(error_c) > cfg.corner_max_target_error {
                return None;
            }
        }
        Some(json!({
            "type": "corner",
            "tick": b.tick,
            "step1_deg": m1,
            "step2_deg": m2,
            "turn_deg": turn,
            "return_deg": returned,
            "target_error_corner_deg": error_b,
            "target_error_end_deg": error_c,
            "target_switch": is_target_switch,
            "fired": fired,
        }))
    }

    /// Lock run stats are final once the run ends, so qualifying runs are evaluated when they end.
    fn lock_qualifies(cfg: &Settings, run: &LockRun) -> Option<&'static str> {
        if run.samples < 4 {
            return None;
        }
        let duration = run.last_tick.saturating_sub(run.start_tick);
        let mean_err = run.error_sum / run.samples as f32;
        let moving_frac = run.moving_samples as f32 / (run.samples - 1) as f32;

        // 1. Classic high-motion lock
        if duration >= cfg.lock_min_ticks
            && run.view_path >= cfg.lock_min_view_path
            && run.view_path >= cfg.lock_min_speed * duration as f32
            && moving_frac >= cfg.lock_min_moving_fraction
            && mean_err <= cfg.lock_max_mean_error
        {
            return Some("classic");
        }

        // 2. Correlated tracking lock (tight pursuit or long sustained tracking while firing)
        if cfg.check_lock_track && run.fired && run.target_path > 1e-3 && moving_frac >= 0.55 {
            let resid_ratio = run.resid_sum / run.target_path;
            let tight_track = run.samples >= 8
                && duration >= cfg.lock_track_min_ticks
                && run.target_path >= cfg.lock_track_min_target_path
                && run.view_path >= cfg.lock_track_min_view_path
                && mean_err <= cfg.lock_track_max_mean_error
                && resid_ratio <= cfg.lock_track_max_resid_ratio;
            let sustained_track = run.samples >= 14
                && duration >= 60
                && run.target_path >= 12.0
                && run.view_path >= 13.5
                && mean_err <= 0.27
                && resid_ratio <= 0.40;
            if tight_track || sustained_track {
                return Some("track");
            }
        }
        None
    }

    /// Extends lock runs with the newest sample, returns the qualifying locks that just ended.
    fn update_locks(cfg: &Settings, pdata: &mut PlayerData, targets: &[Target]) -> Vec<serde_json::Value> {
        let n = pdata.samples.len();
        if n == 0 {
            return Vec::new();
        }
        let cur = pdata.samples[n - 1];
        let prev = if n >= 2 { Some(pdata.samples[n - 2]) } else { None };
        let contiguous = prev.is_some_and(|p| {
            cur.tick.saturating_sub(p.tick) <= cfg.lock_max_gap
                && (1..=6).contains(&(cur.sim_tick - p.sim_tick))
        });
        let step = prev.map_or(0.0, |p| view_step(&p, &cur));
        let view_dv = prev.map(|p| (cur.dir.0 - p.dir.0, cur.dir.1 - p.dir.1, cur.dir.2 - p.dir.2));
        let mut alive: BTreeMap<(u64, i8), LockRun> = BTreeMap::new();

        for target in targets {
            if !Self::valid_target(&cur, target) {
                continue;
            }
            let Some(now) = target.positions.back() else { continue };
            match column_error(cur.eye, cur.dir, now.position, SPLASH_BOTTOM, BODY_TOP_STANDING) {
                Some((e, _)) if e <= cfg.lock_prefilter => {}
                _ => continue,
            }
            if cur.model == AimModel::Hitscan
                && hitscan_error_exceeds(&cur, target.positions, cfg.max_lag, cfg.lock_max_error)
            {
                continue;
            }
            for lag in Self::lags(cfg, &cur.model) {
                let Some((err, dist, tdir_opt, _)) = model_error_full(&cur, target.positions, lag) else { continue };
                if err > cfg.lock_max_error {
                    continue;
                }
                let key = (target.sid, lag as i8);
                let mut run = match (pdata.locks.get(&key), prev) {
                    (Some(r), Some(p)) if contiguous && r.last_tick == p.tick => *r,
                    _ => LockRun { start_tick: cur.tick, ..Default::default() },
                };
                if run.samples > 0 {
                    run.view_path += step;
                    if step > 0.0 {
                        run.moving_samples += 1;
                    }
                    if let (Some(prev_td), Some(cur_td), Some(dv)) = (run.last_tdir, tdir_opt, view_dv) {
                        run.target_path += angle_between(prev_td, cur_td);
                        let dt = (cur_td.0 - prev_td.0, cur_td.1 - prev_td.1, cur_td.2 - prev_td.2);
                        let r = (dv.0 - dt.0, dv.1 - dt.1, dv.2 - dt.2);
                        run.resid_sum += (r.0 * r.0 + r.1 * r.1 + r.2 * r.2).sqrt().to_degrees();
                    }
                }
                run.last_tdir = tdir_opt;
                run.samples += 1;
                run.error_sum += err;
                run.max_error = run.max_error.max(err);
                run.distance_sum += dist;
                run.fired |= cur.fired;
                run.last_tick = cur.tick;
                alive.insert(key, run);
            }
        }
        // Keep the longest qualifying run of each lock episode (runs at several lags overlap)
        for (key, run) in pdata.locks.iter() {
            if alive.contains_key(key) || Self::lock_qualifies(cfg, run).is_none() {
                continue;
            }
            let pending = pdata.pending_locks.entry(key.0).or_insert((key.1, *run));
            if run.last_tick.saturating_sub(run.start_tick) > pending.1.last_tick.saturating_sub(pending.1.start_tick) {
                *pending = (key.1, *run);
            }
        }
        pdata.locks = alive;
        let finished: Vec<u64> = pdata
            .pending_locks
            .keys()
            .filter(|target| !pdata.locks.keys().any(|k| k.0 == **target))
            .copied()
            .collect();
        let mut results = Vec::new();
        for target in finished {
            let (lag, run) = pdata.pending_locks.remove(&target).unwrap();
            let mode = Self::lock_qualifies(cfg, &run).unwrap_or("classic");
            let resid_ratio = if run.target_path > 1e-3 { run.resid_sum / run.target_path } else { 99.0 };
            results.push(json!({
                "type": "lock",
                "lock_mode": mode,
                "tick": run.start_tick,
                "target": target,
                "lock_end_tick": run.last_tick,
                "lock_ticks": run.last_tick.saturating_sub(run.start_tick),
                "samples": run.samples,
                "moving_samples": run.moving_samples,
                "view_path_deg": run.view_path,
                "target_path_deg": run.target_path,
                "resid_ratio": resid_ratio,
                "mean_error_deg": run.error_sum / run.samples as f32,
                "max_error_deg": run.max_error,
                "distance": run.distance_sum / run.samples as f32,
                "lag_ticks": lag,
                "fired": run.fired,
            }));
        }
        results
    }

    /// Evaluates pending hitscan hits whose follow-up window has elapsed (or all when `flush_all`).
    fn process_pending_hits(&mut self, cfg: &Settings, now: u32, flush_all: bool) -> Vec<Detection> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.pending_hits.len() {
            if !flush_all && self.pending_hits[i].tick + HIT_FOLLOW_TICKS > now {
                i += 1;
                continue;
            }
            let hit = self.pending_hits.swap_remove(i);
            let Some(m) = (|| {
                let att = self.players.get(&hit.attacker)?;
                let vic = self.players.get(&hit.victim)?;
                Self::check_shot_snap(cfg, &hit, &att.samples, &vic.positions)
            })() else {
                continue;
            };
            let Some(pdata) = self.players.get_mut(&hit.attacker) else { continue };
            // Cross-type spacing against BOTH the snap tick and the hit tick so a `snap` on
            // `snap_tick` and a `shot_snap` on `hit.tick` cannot double-count one engagement.
            let near_snap = pdata.last_detection.values().any(|&t| t.abs_diff(m.snap_tick) < cfg.spacing);
            if near_snap || !Self::can_flag_any(pdata, "shot_snap", hit.tick, cfg.spacing) {
                continue;
            }
            out.push(Detection {
                tick: hit.tick,
                algorithm: "fidoo/plain_aimbot".to_string(),
                player: hit.attacker,
                data: json!({
                    "type": "shot_snap",
                    "tier": m.tier,
                    "tick": hit.tick,
                    "snap_tick": m.snap_tick,
                    "acquired_ticks_before_hit": m.dt_to_hit,
                    "target": hit.victim,
                    "headshot": hit.headshot,
                    "snap_deg": m.step,
                    "snap_rate_deg_per_sim": m.step_rate,
                    "pre_step_deg": m.pre,
                    "post_step_deg": m.post,
                    "error_deg": m.err_b,
                    "error_before_deg": m.err_a,
                    "error_at_hit_deg": m.err_hit,
                    "hold_median_error_deg": m.hold_med_err,
                    "head_miss_hu": m.head_miss_hu,
                    "distance": m.distance,
                    "lag_ticks": m.lag,
                    "sample_gap": m.tgap,
                    "sim_gap": m.sgap,
                    "class": hit.class,
                    "weapon": hit.weapon,
                }),
            });
        }
        out
    }

    /// `hit` detections: finished hits of `hit_source` scored by the ml_aimbot hit model.
    fn flag_hits(&mut self, cfg: &Settings, hits: Vec<HitRecord>, players: &[Player]) -> Vec<Detection> {
        let mut out = Vec::new();
        for hit in hits {
            let Some(score) = ml_aimbot::hit_probability(&hit.features) else { break };
            if !(score >= cfg.hit_min_score) {
                continue;
            }
            let pdata = self.players.entry(hit.attacker).or_default();
            if pdata.hit_detections >= cfg.hit_max_detections {
                continue;
            }
            if !Self::can_flag_any(pdata, "hit", hit.tick, cfg.spacing) {
                continue;
            }
            pdata.hit_detections += 1;
            let class = players
                .iter()
                .find(|p| player_sid(p) == Some(hit.attacker))
                .map_or("unknown", |p| p.class_name());
            let f = &hit.features;
            out.push(Detection {
                tick: hit.tick,
                algorithm: "fidoo/plain_aimbot".to_string(),
                player: hit.attacker,
                data: json!({
                    "type": "hit",
                    "tick": hit.tick,
                    "target": hit.victim,
                    "hit_score": score,
                    "error_deg": f[0],
                    "error_rel": f[ml_aimbot::IDX_ERR0_REL],
                    "snap_deg_per_tick": f[ml_aimbot::IDX_STEP_HIT],
                    "stillness": f[ml_aimbot::IDX_STILLNESS],
                    "jerk_deg": f[ml_aimbot::IDX_JERK],
                    "acquire_deg_per_tick": f[ml_aimbot::IDX_ACQ_STEP],
                    "acquired_ticks_before_hit": f[ml_aimbot::IDX_ACQ_T],
                    "distance": f[ml_aimbot::IDX_DIST].round(),
                    "lag_ticks": f[ml_aimbot::IDX_LAG],
                    "class": class,
                    "weapon": hit.weapon,
                }),
            });
        }
        out
    }

    fn reset(pdata: &mut PlayerData) {
        pdata.samples.clear();
        pdata.locks.clear();
        pdata.pending_locks.clear();
        pdata.last_sim_tick = None;
        if pdata.positions.back().is_some_and(|r| r.life_id == pdata.life_id) {
            pdata.life_id = pdata.life_id.wrapping_add(1);
        }
    }
}

impl<'a> CheatAlgorithm<'a> for PlainAimbot {
    fn algorithm_name(&self) -> &str {
        "fidoo/plain_aimbot"
    }

    fn tooltip(&self) -> &'static str {
        "Detects visible aimbot behavior, including gap-aware single-update snaps, hit-confirmed acquisition snaps (on-shot and snap-then-hold), sharp corner/target switches, continuous and correlated tracking locks, and high-scoring ml_aimbot hits."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    // Hit features look back 198 ticks (acquisition window) plus 24 ticks of lag before the hit
    fn live_warmup_ticks(&self) -> u32 {
        250
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        let mut types = self.jg.handled_messages()?;
        for t in self.hit_source.handled_messages()? {
            if !types.contains(&t) {
                types.push(t);
            }
        }
        if !types.contains(&MessageType::GameEvent) {
            types.push(MessageType::GameEvent);
        }
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
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        if get_parameter_value::<bool>(&self.params, "check_hit") {
            self.hit_source.on_message(message, state, parser_state, tick)?;
        }
        if get_parameter_value::<bool>(&self.params, "check_shot_snap") {
            if let Message::GameEvent(GameEventMessage { event: GameEvent::PlayerHurt(h), .. }) = message {
                let tick_u32 = u32::from(tick);
                if h.custom != TF_DMG_CUSTOM_BURNING && h.custom != TF_DMG_CUSTOM_BLEEDING {
                    if let (Some(att_sid), Some(vic_sid)) = (
                        state.get_id64_from_userid(h.attacker.into()),
                        state.get_id64_from_userid(h.user_id.into()),
                    ) {
                        if att_sid != vic_sid && att_sid != 0 && Some(att_sid) != self.recorder {
                            let ap = state.players.iter().find(|p| player_sid(p) == Some(att_sid) && p.state == PlayerState::Alive);
                            let vp = state.players.iter().find(|p| player_sid(p) == Some(vic_sid));
                            if let (Some(ap), Some(vp)) = (ap, vp) {
                                if ap.team != vp.team && ap.in_pvs && !ap.is_taunting() {
                                    let weapon = state.get_player_weapon(ap);
                                    if ml_aimbot::is_hitscan(&weapon) {
                                        let spawn_grace = get_parameter_value::<i32>(&self.params, "spawn_grace_ticks").max(0) as u32;
                                        let since_spawn = self.jg.spawned(&att_sid, tick_u32).min(self.jg.teleported(&att_sid, tick_u32));
                                        if since_spawn >= spawn_grace {
                                            let pdata = self.players.entry(att_sid).or_default();
                                            let dup = pdata
                                                .last_pending_hit
                                                .get(&vic_sid)
                                                .is_some_and(|&t| tick_u32 < t + MIN_PENDING_HIT_SPACING);
                                            if !dup {
                                                pdata.last_pending_hit.insert(vic_sid, tick_u32);
                                                self.pending_hits.push(PendingShotHit {
                                                    tick: tick_u32,
                                                    attacker: att_sid,
                                                    victim: vic_sid,
                                                    headshot: h.custom == TF_DMG_CUSTOM_HEADSHOT || h.custom == TF_DMG_CUSTOM_HEADSHOT_DECAP,
                                                    weapon,
                                                    class: ap.class_name(),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, parser_state: &ParserState) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        if self.recorder.is_none() {
            self.recorder = resolve_recorder_sid(state);
        }
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let cfg = Settings::from(&self.params);
        if cfg.check_hit {
            self.hit_source.on_tick(state, parser_state)?;
        }
        let tick = u32::from(state.tick);

        // 1. Record positions and new eye angle samples
        let mut updated_players: Vec<(u64, &'static str, String)> = Vec::new();
        let mut by_sid: BTreeMap<u64, &Player> = BTreeMap::new();
        for player in &state.players {
            let Some(sid) = player_sid(player) else { continue };
            let live = |p: &Player| p.in_pvs && p.state == PlayerState::Alive;
            if by_sid.get(&sid).is_none_or(|old| !live(old) && live(player)) {
                by_sid.insert(sid, player);
            }
        }
        for (&sid, player) in &by_sid {
            let pdata = self.players.entry(sid).or_default();
            if pdata.positions.back().is_some_and(|r| r.tick > tick) {
                pdata.positions.clear();
                Self::reset(pdata);
            }
            if !player.in_pvs || player.state != PlayerState::Alive {
                Self::reset(pdata);
                continue;
            }
            if pdata.last_seen_tick > 0 && tick.saturating_sub(pdata.last_seen_tick) > 16 {
                Self::reset(pdata);
            }
            if let Some(last) = pdata.positions.back() {
                if last.life_id == pdata.life_id {
                    let d = player.position - last.position;
                    if d.x * d.x + d.y * d.y + d.z * d.z > TELEPORT_DIST_SQ {
                        Self::reset(pdata);
                    }
                }
            }
            pdata.last_seen_tick = tick;
            let fired = tick > 0 && self.jg.fired(&sid, tick) == 0;
            let sim_tick = state.sim_tick(player);
            if pdata.last_sim_tick == Some(sim_tick) {
                if fired {
                    if let Some(s) = pdata.samples.back_mut() {
                        s.fired = true;
                    }
                }
                continue;
            }
            pdata.last_sim_tick = Some(sim_tick);
            if pdata.positions.back().is_none_or(|r| r.tick < tick) {
                pdata.positions.push_back(PositionRecord {
                    tick,
                    position: player.position,
                    team: player.team as u8,
                    ducking: player.flags & FL_DUCKING != 0,
                    class: player.class as u8,
                    life_id: pdata.life_id,
                });
                if pdata.positions.len() > POSITION_HISTORY {
                    pdata.positions.pop_front();
                }
            }
            let weapon = state.get_player_weapon(player);
            let (model, melee) = match self.weapon_models.get(&weapon) {
                Some(&cached) => cached,
                None => {
                    let computed = (Self::aim_model(&weapon, cfg.projectile_model), Self::is_melee(&weapon));
                    self.weapon_models.insert(weapon.clone(), computed);
                    computed
                }
            };
            let since_spawn = self.jg.spawned(&sid, tick).min(self.jg.teleported(&sid, tick));
            if since_spawn < cfg.spawn_grace || player.is_taunting() || model == AimModel::Excluded {
                pdata.samples.clear();
                pdata.locks.clear();
                continue;
            }
            if pdata.samples.back().is_some_and(|s| s.tick == tick) {
                continue;
            }
            let pitch = normalize_pitch(player.pitch_angle);
            pdata.samples.push_back(Sample {
                tick,
                sim_tick,
                yaw: player.view_angle,
                pitch,
                dir: view_dir(player.view_angle, pitch),
                eye: eye_position(player),
                team: player.team as u8,
                fired,
                model,
                melee,
            });
            if pdata.samples.len() > SAMPLE_HISTORY {
                pdata.samples.pop_front();
            }
            updated_players.push((sid, player.class_name(), weapon));
        }

        // 2. Run geometric checks for every player with a new sample
        let mut out = Vec::new();
        for (sid, class, weapon) in updated_players {
            let Some(mut pdata) = self.players.remove(&sid) else { continue };
            let team = pdata.samples.back().map(|s| s.team).unwrap_or(0);
            let mut found: Vec<(&'static str, serde_json::Value)> = Vec::new();
            {
                let targets: Vec<Target> = self
                    .players
                    .iter()
                    .filter_map(|(s, d)| {
                        let r = d.positions.back()?;
                        (tick.saturating_sub(r.tick) <= 24).then_some(Target {
                            sid: *s,
                            ally: r.team == team,
                            positions: &d.positions,
                        })
                    })
                    .collect();
                if let Some(v) = Self::check_snap(&cfg, &pdata.samples, &targets) {
                    found.push(("snap", v));
                }
                if cfg.check_corner {
                    if let Some(v) = Self::check_corner(&cfg, &pdata.samples, &targets) {
                        found.push(("corner", v));
                    }
                }
                if cfg.check_lock {
                    for v in Self::update_locks(&cfg, &mut pdata, &targets) {
                        found.push(("lock", v));
                    }
                }
            }
            for (kind, mut data) in found {
                let det_tick = data["tick"].as_u64().unwrap_or(tick as u64) as u32;
                if !Self::can_flag_any(&mut pdata, kind, det_tick, cfg.spacing) {
                    continue;
                }
                data["class"] = json!(class);
                data["weapon"] = json!(weapon);
                out.push(Detection {
                    tick: det_tick,
                    algorithm: "fidoo/plain_aimbot".to_string(),
                    player: sid,
                    data,
                });
            }
            self.players.insert(sid, pdata);
        }

        // 3. Hit-confirmed acquisition snaps whose follow-up window has elapsed
        if cfg.check_shot_snap && !self.pending_hits.is_empty() {
            out.extend(self.process_pending_hits(&cfg, tick, false));
        }

        // 4. ML per-hit detections (after geometric checks so snaps/shot_snaps win the engagement)
        if cfg.check_hit && !self.hit_source.hits.is_empty() {
            let hits = std::mem::take(&mut self.hit_source.hits);
            out.extend(self.flag_hits(&cfg, hits, &state.players));
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let cfg = Settings::from(&self.params);
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        if cfg.check_shot_snap && !self.pending_hits.is_empty() {
            out.extend(self.process_pending_hits(&cfg, u32::MAX, true));
        }
        if cfg.check_hit {
            self.hit_source.flush_pending();
            let hits = std::mem::take(&mut self.hit_source.hits);
            out.extend(self.flag_hits(&cfg, hits, &[]));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_sample(tick: u32, sim_tick: i64, yaw: f32, pitch: f32) -> Sample {
        Sample {
            tick,
            sim_tick,
            yaw,
            pitch,
            dir: view_dir(yaw, pitch),
            eye: Vector { x: 0.0, y: 0.0, z: 75.0 },
            team: 2,
            fired: false,
            model: AimModel::Hitscan,
            melee: false,
        }
    }

    fn mk_pos(tick: u32, x: f32, y: f32, z: f32) -> PositionRecord {
        PositionRecord {
            tick,
            position: Vector { x, y, z },
            team: 3,
            ducking: false,
            class: 2,
            life_id: 0,
        }
    }

    #[test]
    fn snap_handles_choppy_demo_gap_and_joint_lag() {
        let algo = PlainAimbot::new();
        let cfg = Settings::from(&algo.params);
        // Target moves slowly along y; at lag=18, y=0 at tick 100
        let mut positions = VecDeque::new();
        for t in 60..=120u32 {
            let y = (t as f32 - 82.0) * 4.0;
            positions.push_back(mk_pos(t, 2000.0, y, 0.0));
        }
        let target = Target { sid: 42, ally: false, positions: &positions };
        // Attacker with 4-tick demo gap (sim gap 4): still at yaw=18°, snaps to yaw=0° (lag=18) and tracks
        let mut samples = VecDeque::new();
        samples.push_back(mk_sample(88, 88, 18.0, 0.0));
        samples.push_back(mk_sample(92, 92, 18.0, 0.0));
        samples.push_back(mk_sample(96, 96, 18.0, 0.0));
        samples.push_back(mk_sample(100, 100, 0.0, 0.0)); // 18 deg snap over 4-tick gap
        samples.push_back(mk_sample(104, 104, (16.0f32).atan2(2000.0).to_degrees(), 0.0));
        samples.push_back(mk_sample(108, 108, (32.0f32).atan2(2000.0).to_degrees(), 0.0));
        samples.push_back(mk_sample(112, 112, (48.0f32).atan2(2000.0).to_degrees(), 0.0));
        let det = PlainAimbot::check_snap(&cfg, &samples, &[target]);
        assert!(det.is_some(), "expected choppy 18° snap to be detected");
    }

    #[test]
    fn shot_snap_detects_dead_still_low_fov_headshot() {
        let algo = PlainAimbot::new();
        let cfg = Settings::from(&algo.params);
        let mut positions = VecDeque::new();
        for t in 70..=110u32 {
            positions.push_back(mk_pos(t, 1200.0, 0.0, 0.0));
        }
        let mut samples = VecDeque::new();
        samples.push_back(mk_sample(94, 94, 5.0, 0.0));
        samples.push_back(mk_sample(97, 97, 5.0, 0.0));
        samples.push_back(mk_sample(100, 100, 0.0, 0.0)); // 5° dead-still snap on shot tick
        samples.push_back(mk_sample(103, 103, 0.0, 0.0));
        let hit = PendingShotHit {
            tick: 100,
            attacker: 1,
            victim: 2,
            headshot: true,
            weapon: "Sniper Rifle".to_string(),
            class: "sniper",
        };
        let m = PlainAimbot::check_shot_snap(&cfg, &hit, &samples, &positions);
        assert!(m.is_some());
        assert_eq!(m.unwrap().tier, "on_shot_still");
    }

    #[test]
    fn shot_snap_detects_snap_and_hold_before_hit() {
        let algo = PlainAimbot::new();
        let cfg = Settings::from(&algo.params);
        let mut positions = VecDeque::new();
        for t in 50..=130u32 {
            positions.push_back(mk_pos(t, 2000.0, 0.0, 0.0));
        }
        let mut samples = VecDeque::new();
        samples.push_back(mk_sample(76, 76, 6.0, 0.0));
        samples.push_back(mk_sample(78, 78, 6.0, 0.0));
        for t in (80..=104u32).step_by(2) {
            samples.push_back(mk_sample(t, t as i64, 0.0, 0.0)); // 6° snap at tick 80, held to hit at 100
        }
        let hit = PendingShotHit {
            tick: 100,
            attacker: 1,
            victim: 2,
            headshot: true,
            weapon: "Bazaar Bargain".to_string(),
            class: "sniper",
        };
        let m = PlainAimbot::check_shot_snap(&cfg, &hit, &samples, &positions);
        assert!(m.is_some());
        assert_eq!(m.unwrap().tier, "snap_hold");
    }
}
