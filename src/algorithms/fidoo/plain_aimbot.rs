// Detection of plain (visible) aimbot in POV demos.
//
// Plain aimbot writes the aim angles into both the usercmd and the engine view angles, so the
// networked eye angles of the cheater visibly move onto the target and stay there (no snapback,
// unlike silent aim). Amalgam's "Plain" mode snaps on attack ticks, "Smooth" lerps the view towards
// the target by a constant fraction every tick, RijiN lerps by `1 - smoothing / 100`.
//
// Signatures (see docs/plain_aimbot.md):
// - snap:  an impulsive one-sample jump of the view (still before and after) that lands on a target
//          and stays there.
// - lock:  the aim error to a target stays tiny for a long time while the view moves a lot
//          (locking aimbots and the tracking phase of smoothed aimbots).
// - corner: two consecutive large view steps in sharply different directions (e.g. snapping from one
//          target to the next). A hand has to slow down to change direction, an aimbot does not.
//
// Projectile weapons are aimed at the target's predicted position (extrapolated by the projectile's
// flight time), splash weapons at the feet, arcing projectiles are only compared in yaw.
//
// Eye angles of other players are networked with ~0.35° precision and only change when the player's
// simulation time advances (every ~2 ticks), so all checks run on simtime samples.

use std::collections::{BTreeMap, HashMap, VecDeque};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::nocrex::jankguard::JankGuard;

const SAMPLE_HISTORY: usize = 48;
/// Samples after a jump used to check where the view went next
const FOLLOW_SAMPLES: usize = 3;
const POSITION_HISTORY: usize = 48;
const TELEPORT_DIST_SQ: f32 = 256.0 * 256.0;
// Vertical extent of the aimable body column above the origin
const BODY_BOTTOM: f32 = 5.0;
const BODY_TOP_STANDING: f32 = 80.0;
const BODY_TOP_DUCKING: f32 = 60.0;
/// Splash weapons are aimed at the feet / the ground below the target
const SPLASH_BOTTOM: f32 = -24.0;
const FL_DUCKING: u32 = 1 << 1;
const TICKS_PER_SECOND: f32 = 66.67;
/// Ticks over which a target's velocity is estimated for projectile prediction
const VELOCITY_TICKS: i64 = 6;
/// Longest projectile flight time that is extrapolated
const MAX_FLIGHT_TICKS: f32 = 100.0;
const WORLD_GRAVITY: f32 = 800.0;

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

#[derive(Clone, Copy)]
struct PositionRecord {
    tick: u32,
    position: Vector,
    team: u8,
    ducking: bool,
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
    last_detection: HashMap<&'static str, u32>,
    /// Longest qualifying ended run per target, emitted once no run on that target is alive
    pending_locks: BTreeMap<u64, (i8, LockRun)>,
}

#[derive(Clone, Copy, Default)]
struct LockRun {
    start_tick: u32,
    last_tick: u32,
    samples: u32,
    moving_samples: u32,
    view_path: f32,
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

/// Best landing of the view at one sample.
struct Landing {
    error: f32,
    error_before: f32,
    target: usize,
    lag: i32,
    distance: f32,
}

#[derive(Default)]
pub struct PlainAimbot {
    params: Parameters,
    jg: JankGuard,
    // Ordered by SteamID so targets are always tried in the same order (ties keep the first).
    players: BTreeMap<u64, PlayerData>,
    /// Aim model and melee flag per weapon name (both depend on the name only)
    weapon_models: HashMap<String, (AimModel, bool)>,
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

/// Angular error between the view ray and the closest point of the target's vertical column
/// `[target.z + bottom, target.z + top]`, plus the distance to the target. `dir` is the view
/// direction (`view_dir`).
fn column_error(eye: Vector, dir: (f32, f32, f32), target: Vector, bottom: f32, top: f32) -> Option<(f32, f32)> {
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
    Some((angle_between(dir, to), (horizontal * horizontal + to.2 * to.2).sqrt()))
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
    let span = newer.tick.saturating_sub(older.tick);
    if span == 0 || span > 6 {
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
    })
}

/// Margin (degrees) of the `hitscan_error_exceeds` bound, far above the f32 rounding of the angles.
const HITSCAN_BOUND_MARGIN: f32 = 0.05;

/// Whether `model_error` of a hitscan sample is above `max_error` at every lag `0..=max_lag` (a
/// performance shortcut, it never skips a lag that could pass). Hitscan errors are measured to a
/// point of the body column at a position interpolated between records at most 6 ticks apart
/// (`position_at`), so every such point lies in the bounding box of the records from
/// `max_lag + 6` ticks before the sample on, extended by the column height. The view error to any
/// point of the box's bounding sphere is at least the error to its centre minus its angular radius.
fn hitscan_error_exceeds(sample: &Sample, positions: &VecDeque<PositionRecord>, max_lag: u32, max_error: f32) -> bool {
    let first_tick = sample.tick as i64 - max_lag as i64 - 6;
    let mut min = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for record in positions.iter().rev().take_while(|r| r.tick as i64 >= first_tick) {
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
    // Body column: BODY_BOTTOM up to the standing (tallest) top
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
    // Launch elevation (degrees, up positive) of the flat arc to a point `h` above the eye
    let elevation = |h: f32| -> Option<f32> {
        let v2 = speed * speed;
        let disc = v2 * v2 - g * (g * horizontal * horizontal + 2.0 * h * v2);
        (disc >= 0.0).then(|| ((v2 - disc.sqrt()) / (g * horizontal)).atan().to_degrees())
    };
    let low = elevation(target.z + BODY_BOTTOM - sample.eye.z)?;
    let high = elevation(target.z + top - sample.eye.z).unwrap_or(low);
    // Source pitch is positive downwards
    let view_elevation = -sample.pitch;
    let dpitch = if view_elevation < low { low - view_elevation } else if view_elevation > high { view_elevation - high } else { 0.0 };
    let bearing = dy.atan2(dx).to_degrees();
    let dyaw = ((sample.yaw - bearing + 180.0).rem_euclid(360.0) - 180.0).abs() * sample.pitch.to_radians().cos();
    Some((dyaw.hypot(dpitch), horizontal))
}

/// Aim error of a sample against a target seen `lag` ticks ago, for the sample's weapon.
/// Negative lags (projectiles only) use the target's real future position as the lead.
fn model_error(sample: &Sample, target: &Target, lag: i32) -> Option<(f32, f32)> {
    let at = sample.tick as i64 - lag as i64;
    let rec = position_at(target.positions, at)?;
    let top = if rec.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING };
    if lag < 0 {
        let bottom = match sample.model {
            AimModel::Projectile { splash: true, .. } => SPLASH_BOTTOM,
            _ => BODY_BOTTOM,
        };
        return column_error(sample.eye, sample.dir, rec.position, bottom, top);
    }
    match sample.model {
        AimModel::Projectile { speed, gravity, splash, .. } => {
            // Extrapolate the target by the flight time, like a projectile aimbot does
            let older = position_at(target.positions, at - VELOCITY_TICKS).unwrap_or(rec);
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
                // Arcing projectiles are aimed above the target: compare with the launch pitch of the
                // flat ballistic arc that hits the target column
                ballistic_error(sample, predicted, speed, gravity * WORLD_GRAVITY, top)
            } else {
                let bottom = if splash { SPLASH_BOTTOM } else { BODY_BOTTOM };
                column_error(sample.eye, sample.dir, predicted, bottom, top)
            }
        }
        _ => column_error(sample.eye, sample.dir, rec.position, BODY_BOTTOM, top),
    }
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
                ("max_sample_gap".to_string(), Parameter::Int(3)),
                ("max_simtime_gap".to_string(), Parameter::Int(4)),
                // snap
                ("check_snap".to_string(), Parameter::Bool(true)),
                ("snap_min_step".to_string(), Parameter::Float(10.0)),
                ("snap_max_step".to_string(), Parameter::Float(120.0)),
                ("snap_min_ratio".to_string(), Parameter::Float(12.0)),
                ("snap_max_error".to_string(), Parameter::Float(0.25)),
                ("snap_min_error_limit".to_string(), Parameter::Float(0.25)),
                ("snap_max_error_rel".to_string(), Parameter::Float(0.05)),
                ("snap_min_distance".to_string(), Parameter::Float(400.0)),
                ("snap_follow_samples".to_string(), Parameter::Int(2)),
                ("snap_max_follow_error".to_string(), Parameter::Float(0.5)),
                ("snap_projectile_follow_samples".to_string(), Parameter::Int(1)),
                ("snap_projectile_max_follow_error".to_string(), Parameter::Float(1.0)),
                // lock
                ("check_lock".to_string(), Parameter::Bool(true)),
                ("lock_max_error".to_string(), Parameter::Float(0.8)),
                ("lock_max_mean_error".to_string(), Parameter::Float(0.25)),
                ("lock_min_ticks".to_string(), Parameter::Int(20)),
                ("lock_min_view_path".to_string(), Parameter::Float(15.0)),
                ("lock_min_moving_fraction".to_string(), Parameter::Float(0.6)),
                ("lock_min_speed".to_string(), Parameter::Float(0.0)),
                ("lock_prefilter".to_string(), Parameter::Float(45.0)),
                // corner
                ("check_corner".to_string(), Parameter::Bool(true)),
                ("corner_min_step".to_string(), Parameter::Float(10.0)),
                ("corner_max_step".to_string(), Parameter::Float(150.0)),
                ("corner_min_turn".to_string(), Parameter::Float(120.0)),
                ("corner_min_return".to_string(), Parameter::Float(0.3)),
                ("corner_max_target_error".to_string(), Parameter::Float(0.5)),
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
         "spy-cicle", "eternal reward", "kunai", "sharp dresser", "powerjack", "third degree", "candy cane",
         "sandman", "ubersaw", "vita-saw", "amputator", "persuader", "caber", "zatoichi", "claidheamh",
         "market gardener", "disciplinary", "holiday punch", "eviction notice", "fist of steel", "warrior",
         "big earner", "wrap assassin", "atomizer", "fan o'war", "unarmed combat", "homewrecker", "maul",
         "neon annihilator", "postal pummeler", "back scratcher", "lollichop", "ham shank", "conscientious",
         "freedom staff", "bat outta hell", "memory maker", "black rose", "bread bite", "necro smasher",
         "crossing guard", "prinny machete", "nessie", "golf club", "saxxy", "solemn vow", "southern hospitality",
         "boston basher", "sun-on-a-stick", "scotsman", "half-zatoichi", "tribalman", "shahanshah", "melee"]
            .iter()
            .any(|w| lw.contains(w))
    }

    /// Aim model of a weapon. Without `projectile_model`, every weapon is assumed to be aimed like a
    /// hitscan weapon (straight at the body, the crossbow also at teammates).
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
    snap_min_ratio: f32,
    snap_max_error: f32,
    snap_min_error_limit: f32,
    snap_max_error_rel: f32,
    snap_min_distance: f32,
    snap_follow_samples: usize,
    snap_max_follow_error: f32,
    snap_projectile_follow_samples: usize,
    snap_projectile_max_follow_error: f32,
    check_corner: bool,
    corner_min_step: f32,
    corner_max_step: f32,
    corner_min_turn: f32,
    corner_min_return: f32,
    corner_max_target_error: f32,
    check_lock: bool,
    lock_max_error: f32,
    lock_max_mean_error: f32,
    lock_min_ticks: u32,
    lock_min_view_path: f32,
    lock_min_moving_fraction: f32,
    lock_min_speed: f32,
    lock_prefilter: f32,
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
            snap_min_ratio: f("snap_min_ratio"),
            snap_max_error: f("snap_max_error"),
            snap_min_error_limit: f("snap_min_error_limit"),
            snap_max_error_rel: f("snap_max_error_rel"),
            snap_min_distance: f("snap_min_distance"),
            snap_follow_samples: (i("snap_follow_samples") as usize).min(FOLLOW_SAMPLES),
            snap_max_follow_error: f("snap_max_follow_error"),
            snap_projectile_follow_samples: (i("snap_projectile_follow_samples") as usize).min(FOLLOW_SAMPLES),
            snap_projectile_max_follow_error: f("snap_projectile_max_follow_error"),
            check_corner: b("check_corner"),
            corner_min_step: f("corner_min_step"),
            corner_max_step: f("corner_max_step"),
            corner_min_turn: f("corner_min_turn"),
            corner_min_return: f("corner_min_return"),
            corner_max_target_error: f("corner_max_target_error"),
            check_lock: b("check_lock"),
            lock_max_error: f("lock_max_error"),
            lock_max_mean_error: f("lock_max_mean_error"),
            lock_min_ticks: i("lock_min_ticks"),
            lock_min_view_path: f("lock_min_view_path"),
            lock_min_moving_fraction: f("lock_min_moving_fraction"),
            lock_min_speed: f("lock_min_speed"),
            lock_prefilter: f("lock_prefilter"),
            spacing: i("min_detection_spacing"),
        }
    }
}

impl PlainAimbot {
    /// Lags to try for a weapon. Projectiles also try short leads (future positions).
    fn lags(cfg: &Settings, model: &AimModel) -> impl Iterator<Item = i32> {
        let lead = if matches!(model, AimModel::Projectile { .. }) { cfg.projectile_lead as i32 } else { 0 };
        (-lead..=cfg.max_lag as i32).step_by(cfg.lag_step as usize)
    }

    /// Targets the sample's weapon can be aimed at.
    fn valid_target(sample: &Sample, target: &Target) -> bool {
        match sample.model {
            AimModel::Projectile { heals_team: true, .. } => true,
            _ => !target.ally,
        }
    }

    fn can_flag(pdata: &mut PlayerData, kind: &'static str, tick: u32, spacing: u32) -> bool {
        match pdata.last_detection.get(kind) {
            Some(&last) if tick.saturating_sub(last) < spacing => false,
            _ => {
                pdata.last_detection.insert(kind, tick);
                true
            }
        }
    }

    /// Target/lag with the smallest error at `b` (at most `max_error`) that the view was off before
    /// the jump from `a` (at least half the jump away).
    fn best_landing(cfg: &Settings, a: &Sample, b: &Sample, step: f32, max_error: f32, targets: &[Target]) -> Option<Landing> {
        let mut best: Option<Landing> = None;
        for (ti, target) in targets.iter().enumerate() {
            if !Self::valid_target(b, target) {
                continue;
            }
            for lag in Self::lags(cfg, &b.model) {
                let Some((error, distance)) = model_error(b, target, lag) else { continue };
                if error > max_error || best.as_ref().is_some_and(|x| x.error <= error) {
                    continue;
                }
                let Some((error_before, _)) = model_error(a, target, lag) else { continue };
                if error_before < 0.5 * step {
                    continue;
                }
                best = Some(Landing { error, error_before, target: ti, lag, distance });
            }
        }
        best
    }

    /// Impulsive jump into the sample `FOLLOW_SAMPLES` samples ago: the view is still before and
    /// after the jump, lands on a target and stays near it for the next sample(s). Projectiles are
    /// allowed to drift more, their predicted aim point moves with the target.
    fn check_snap(cfg: &Settings, samples: &VecDeque<Sample>, targets: &[Target]) -> Option<serde_json::Value> {
        let n = samples.len();
        if !cfg.check_snap || n < FOLLOW_SAMPLES + 4 {
            return None;
        }
        let j = n - 1 - FOLLOW_SAMPLES;
        let (before, a, b, after) = (&samples[j - 2], &samples[j - 1], &samples[j], &samples[j + 1]);
        if b.model == AimModel::Excluded
            || b.tick - a.tick > cfg.max_sample_gap
            || !(1..=cfg.max_simtime_gap as i64).contains(&(b.sim_tick - a.sim_tick))
        {
            return None;
        }
        let step = view_step(a, b);
        if step > cfg.snap_max_step || step < cfg.snap_min_step {
            return None;
        }
        let pre = view_step(before, a);
        let post = view_step(b, after);
        if step < cfg.snap_min_ratio * pre.max(post).max(0.36) {
            return None;
        }
        let max_error = cfg.snap_max_error.min(cfg.snap_min_error_limit.max(cfg.snap_max_error_rel * step));
        let landing = Self::best_landing(cfg, a, b, step, max_error, targets)?;
        if landing.distance < cfg.snap_min_distance {
            return None;
        }
        let target = &targets[landing.target];
        let follow: Vec<f32> = (1..=FOLLOW_SAMPLES)
            .map(|k| model_error(&samples[j + k], target, landing.lag).map_or(99.0, |x| x.0))
            .collect();
        let projectile = matches!(b.model, AimModel::Projectile { .. });
        let (follow_samples, follow_error) = if projectile {
            (cfg.snap_projectile_follow_samples, cfg.snap_projectile_max_follow_error)
        } else {
            (cfg.snap_follow_samples, cfg.snap_max_follow_error)
        };
        let follow_limit = follow_error.max(landing.error + 0.36);
        if follow.iter().take(follow_samples).any(|&e| e > follow_limit) {
            return None;
        }
        Some(json!({
            "type": "snap",
            "tick": b.tick,
            "target": target.sid,
            "ally": target.ally,
            "snap_deg": step,
            "pre_step_deg": pre,
            "post_step_deg": post,
            "error_deg": landing.error,
            "error_before_deg": landing.error_before,
            "lag_ticks": landing.lag,
            "distance": landing.distance,
            "follow_errors_deg": follow,
            "projectile": projectile,
            "fired": a.fired || b.fired || after.fired,
        }))
    }

    /// Smallest aim error of a sample against any valid target and lag.
    fn nearest_target_error(cfg: &Settings, sample: &Sample, targets: &[Target]) -> f32 {
        let mut best = 180.0f32;
        for target in targets.iter().filter(|t| Self::valid_target(sample, t)) {
            for lag in Self::lags(cfg, &sample.model) {
                if let Some((e, _)) = model_error(sample, target, lag) {
                    best = best.min(e);
                }
            }
        }
        best
    }

    /// Sharp turn at speed over the three newest samples A -> B -> C: both steps are large and point
    /// in very different directions, and C does not simply return to A (that is a snapback).
    fn check_corner(cfg: &Settings, samples: &VecDeque<Sample>, targets: &[Target]) -> Option<serde_json::Value> {
        let n = samples.len();
        if n < 3 {
            return None;
        }
        let (a, b, c) = (&samples[n - 3], &samples[n - 2], &samples[n - 1]);
        if b.melee || c.melee || matches!(b.model, AimModel::Excluded) || matches!(c.model, AimModel::Excluded) {
            return None;
        }
        for (x, y) in [(a, b), (b, c)] {
            if y.tick - x.tick > cfg.max_sample_gap || !(1..=cfg.max_simtime_gap as i64).contains(&(y.sim_tick - x.sim_tick)) {
                return None;
            }
        }
        // Steps in (yaw, pitch) space, yaw scaled to arc length
        let step = |x: &Sample, y: &Sample| {
            let dyaw = (y.yaw - x.yaw + 180.0).rem_euclid(360.0) - 180.0;
            (dyaw * ((x.pitch + y.pitch) * 0.5).to_radians().cos(), y.pitch - x.pitch)
        };
        let (v1, v2) = (step(a, b), step(b, c));
        let (m1, m2) = (v1.0.hypot(v1.1), v2.0.hypot(v2.1));
        if m1.min(m2) < cfg.corner_min_step || m1.max(m2) > cfg.corner_max_step {
            return None;
        }
        let turn = ((v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2)).clamp(-1.0, 1.0).acos().to_degrees();
        if turn < cfg.corner_min_turn {
            return None;
        }
        let returned = view_step(a, c);
        if returned < cfg.corner_min_return * m1.min(m2) {
            return None;
        }
        let error_b = Self::nearest_target_error(cfg, b, targets);
        let error_c = Self::nearest_target_error(cfg, c, targets);
        if error_b.min(error_c) > cfg.corner_max_target_error {
            return None;
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
            "fired": a.fired || b.fired || c.fired,
        }))
    }

    /// Lock run stats are final once the run ends, so qualifying runs are evaluated when they end.
    fn lock_qualifies(cfg: &Settings, run: &LockRun) -> bool {
        let duration = run.last_tick - run.start_tick;
        duration >= cfg.lock_min_ticks
            && run.view_path >= cfg.lock_min_view_path
            && run.view_path >= cfg.lock_min_speed * duration as f32
            && run.moving_samples as f32 >= cfg.lock_min_moving_fraction * (run.samples - 1) as f32
            && run.error_sum / run.samples as f32 <= cfg.lock_max_mean_error
    }

    /// Extends lock runs with the newest sample, returns the qualifying locks that just ended.
    fn update_locks(cfg: &Settings, pdata: &mut PlayerData, targets: &[Target]) -> Vec<serde_json::Value> {
        let n = pdata.samples.len();
        if n == 0 {
            return Vec::new();
        }
        let cur = pdata.samples[n - 1];
        let prev = if n >= 2 { Some(pdata.samples[n - 2]) } else { None };
        let contiguous = prev.is_some_and(|p| cur.tick - p.tick <= cfg.max_sample_gap + 1);
        let step = prev.map_or(0.0, |p| view_step(&p, &cur));
        let mut alive: BTreeMap<(u64, i8), LockRun> = BTreeMap::new();
        for target in targets {
            if !Self::valid_target(&cur, target) {
                continue;
            }
            // Cheap pre-filter on the current position before trying every lag
            let Some(now) = target.positions.back() else { continue };
            match column_error(cur.eye, cur.dir, now.position, SPLASH_BOTTOM, BODY_TOP_STANDING) {
                Some((e, _)) if e <= cfg.lock_prefilter => {}
                _ => continue,
            }
            // Skips the lags only when none of them can be within the lock error (same result)
            if cur.model == AimModel::Hitscan
                && hitscan_error_exceeds(&cur, target.positions, cfg.max_lag, cfg.lock_max_error)
            {
                continue;
            }
            for lag in Self::lags(cfg, &cur.model) {
                let Some((err, dist)) = model_error(&cur, target, lag) else { continue };
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
                }
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
            if alive.contains_key(key) || !Self::lock_qualifies(cfg, run) {
                continue;
            }
            let pending = pdata.pending_locks.entry(key.0).or_insert((key.1, *run));
            if run.last_tick - run.start_tick > pending.1.last_tick - pending.1.start_tick {
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
            results.push(json!({
                "type": "lock",
                "tick": run.start_tick,
                "target": target,
                "lock_end_tick": run.last_tick,
                "lock_ticks": run.last_tick - run.start_tick,
                "samples": run.samples,
                "moving_samples": run.moving_samples,
                "view_path_deg": run.view_path,
                "mean_error_deg": run.error_sum / run.samples as f32,
                "max_error_deg": run.max_error,
                "distance": run.distance_sum / run.samples as f32,
                "lag_ticks": lag,
                "fired": run.fired,
            }));
        }
        results
    }

    fn reset(pdata: &mut PlayerData) {
        pdata.samples.clear();
        pdata.positions.clear();
        pdata.locks.clear();
        pdata.pending_locks.clear();
        pdata.last_sim_tick = None;
    }
}

impl<'a> CheatAlgorithm<'a> for PlainAimbot {
    fn algorithm_name(&self) -> &str {
        "fidoo/plain_aimbot"
    }

    fn tooltip(&self) -> &'static str {
        "Detects visible aimbot behavior, including single-tick snaps onto targets, sharp corner transitions, and continuous tracking locks."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        self.jg.handled_messages()
    }

    fn on_message(
        &mut self,
        message: &tf_demo_parser::demo::message::Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: tf_demo_parser::demo::data::DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let cfg = Settings::from(&self.params);
        let tick = u32::from(state.tick);

        // 1. Record positions and new eye angle samples
        let mut updated_players: Vec<(u64, &'static str, String)> = Vec::new();
        // A SteamID can own several player entities (e.g. after a reconnect), use the live one
        let mut by_sid: BTreeMap<u64, &Player> = BTreeMap::new();
        for player in &state.players {
            let Some(info) = &player.info else { continue };
            if info.steam_id == "BOT" {
                continue;
            }
            let Ok(sid) = SteamID::from_steam3(&info.steam_id).map(u64::from) else { continue };
            let live = |p: &Player| p.in_pvs && p.state == PlayerState::Alive;
            if by_sid.get(&sid).is_none_or(|old| !live(old) && live(player)) {
                by_sid.insert(sid, player);
            }
        }
        for (&sid, player) in &by_sid {
            let pdata = self.players.entry(sid).or_default();
            if !player.in_pvs || player.state != PlayerState::Alive {
                Self::reset(pdata);
                continue;
            }
            if pdata.last_seen_tick > 0 && tick.saturating_sub(pdata.last_seen_tick) > 16 {
                Self::reset(pdata);
            }
            if let Some(last) = pdata.positions.back() {
                let d = player.position - last.position;
                if d.x * d.x + d.y * d.y + d.z * d.z > TELEPORT_DIST_SQ {
                    Self::reset(pdata);
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
            pdata.samples.push_back(Sample {
                tick,
                sim_tick,
                yaw: player.view_angle,
                pitch: player.pitch_angle,
                dir: view_dir(player.view_angle, player.pitch_angle),
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

        // 2. Run the checks for every player with a new sample
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
                        (tick.saturating_sub(r.tick) <= 8).then_some(Target { sid: *s, ally: r.team == team, positions: &d.positions })
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
                if !Self::can_flag(&mut pdata, kind, det_tick, cfg.spacing) {
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
        Ok(out)
    }
}
