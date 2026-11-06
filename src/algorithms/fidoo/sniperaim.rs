//! Low FOV sniper aimbot detection. See docs/sniperaim.md.
//!
//! A low FOV aimbot moves the crosshair a few degrees onto the head in the very usercmd that fires,
//! after the player's own (slow, imprecise) aim brought the target into its FOV. Spectated players'
//! eye angles are too coarse for this (0.35° yaw / 0.7° pitch), but a scoped sniper's `CSniperDot`
//! sits at the end of the server-side aim trace with 1/32 unit precision, so `dot - eye` gives the
//! exact aim of every server update.
//!
//! Usercmds often reach the server in pairs, so the aim is compared per *update* (a change of the
//! dot), not per tick. A rifle hit is flagged when the update on the shot tick moves the aim by
//! several head radii and by several times the largest of the player's preceding updates, and then
//! either
//! - stays there and the shot is a headshot (`snap`: plain / smooth aimbot), or
//! - returns to where the player was aiming right after the shot (`silent_snap`: silent aimbot,
//!   which no mouse movement can produce).
//! The coarse eye angles must agree with the dot, which rules out dot artifacts.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::packetentities::UpdateType;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::analyser::Class;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;

/// Ticks of aim history kept before a shot.
const PRE_TICKS: i32 = 24;
/// Ticks of aim history used after a shot.
const POST_TICKS: i32 = 6;
/// Hurt events this many ticks after the bullet still belong to it. Game events travel in the
/// reliable stream and often reach the demo 1–6 ticks after the bullet's temp entity.
const HURT_DELAY_TICKS: u32 = 10;
/// Shots are evaluated this many ticks after being fired, so the post-shot aim and hurt event exist.
const EVAL_DELAY: u32 = if HURT_DELAY_TICKS > POST_TICKS as u32 { HURT_DELAY_TICKS } else { POST_TICKS as u32 };
const HISTORY_TICKS: usize = PRE_TICKS as usize + EVAL_DELAY as usize + 4;
/// Preceding updates must lie within this many ticks before the snap.
const PRIOR_WINDOW: i32 = 16;
/// Number of preceding updates the snap is compared with.
const PRIOR_UPDATES: usize = 6;
/// Without preceding updates, the pre-snap aim must have rested this many ticks.
const MIN_REST_TICKS: i32 = 3;
/// A dot that has not moved while its owner's eye angles turned by more than this (degrees, two
/// quantization steps) or its owner walked `FROZEN_DOT_MOVE` units is frozen.
const FROZEN_DOT_YAW: f32 = 0.75;
const FROZEN_DOT_PITCH: f32 = 1.0;
const FROZEN_DOT_MOVE: f32 = 16.0;
/// Aim changes smaller than this (degrees) are the same update.
const SAME_AIM_EPSILON: f32 = 0.002;
/// Dot samples closer than this to the eye are unusable (eye height errors dominate).
const MIN_SAMPLE_DISTANCE: f32 = 64.0;
const MAX_HEAD_DISTANCE: f32 = 8192.0;
const HEAD_RADIUS: f32 = 6.0;
/// The eye angles may differ from the dot snap by this much plus `EYE_TOLERANCE_RATIO` of it (they
/// are quantized to 0.35° yaw / 0.7° pitch).
const EYE_TOLERANCE: f32 = 0.6;
const EYE_TOLERANCE_RATIO: f32 = 0.35;
/// Eye angle quantization steps (degrees) and the tolerance for "the same value".
const EYE_YAW_STEP: f32 = 0.36;
const EYE_PITCH_STEP: f32 = 0.72;
const EYE_SAME: f32 = 0.05;
/// Eye-only silent snaps: rest window before the shot, and the largest gap to the samples around it.
const EYE_REST_TICKS: i32 = 6;
const EYE_MAX_GAP: u32 = 3;
/// A snap that has neither returned nor moved on by this tick is treated as held.
const HOLD_TICKS: i32 = 2;
/// TF_DMG_CUSTOM_HEADSHOT, TF_DMG_CUSTOM_HEADSHOT_DECAPITATION
const HEADSHOT_CUSTOMS: [u16; 2] = [1, 51];
/// The Sydney Sleeper has no headshot damage type, but its scoped headshots always mini-crit
/// ("Nature's Call"). Conditions that mini-crit any hit: victim TF_COND_URINE,
/// TF_COND_MARKEDFORDEATH(_SILENT); attacker TF_COND_OFFENSEBUFF, TF_COND_ENERGY_BUFF,
/// TF_COND_MINICRITBOOSTED_ON_KILL.
const SYDNEY_SLEEPER: &str = "Sydney Sleeper";
const VICTIM_MINI_CRIT_CONDS: [u32; 3] = [24, 30, 48];
const ATTACKER_MINI_CRIT_CONDS: [u32; 3] = [16, 19, 78];

fn has_cond(p: &Player, cond: u32) -> bool {
    let bits = match cond / 32 {
        0 => p.cond,
        1 => p.cond_ex,
        2 => p.cond_ex2,
        3 => p.cond_ex3,
        _ => return false,
    };
    bits & (1 << (cond % 32)) != 0
}

const DOT_ORIGIN: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_vecOrigin");
const DOT_OWNER: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_hOwnerEntity");
const BULLETS_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
const BULLETS_WEAPON: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iWeaponID");
const BULLETS_ORIGIN: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_vecOrigin");
const BLOOD_X: SendPropIdentifier = SendPropIdentifier::new("DT_TETFBlood", "m_vecOrigin[0]");
const BLOOD_Y: SendPropIdentifier = SendPropIdentifier::new("DT_TETFBlood", "m_vecOrigin[1]");
const BLOOD_Z: SendPropIdentifier = SendPropIdentifier::new("DT_TETFBlood", "m_vecOrigin[2]");
const BLOOD_ENT: SendPropIdentifier = SendPropIdentifier::new("DT_TETFBlood", "entindex");

/// Head centre relative to the origin, fitted on 7000 server-confirmed headshots (see docs/sniperaim.md):
/// forward, forward per pitch / 90, left, up, up per pitch / 90, forward per forward speed / 300 HU/s,
/// left per left speed / 300 HU/s. Index: class, then standing / ducked. The head sits up to 10 HU in
/// front of the origin, and further when the player looks down.
const HEAD_MODEL: [[[f32; 7]; 2]; 9] = [
    [[10.1, -2.2, 0.6, 65.9, -5.8, 2.4, 3.2], [6.5, 18.9, 1.4, 56.7, -10.5, 4.6, -0.2]], // Scout
    [[9.5, -0.7, -0.7, 70.4, -7.4, 2.8, 0.6], [8.9, 8.9, -2.7, 66.3, 11.4, -2.2, 3.8]],  // Soldier
    [[8.7, 12.2, -2.4, 67.0, -7.8, 1.6, -0.1], [14.4, 19.3, -5.9, 57.5, 3.0, -14.6, -3.6]], // Pyro
    [[8.9, 7.8, -2.5, 70.7, -8.5, -2.0, 2.1], [12.7, -5.5, -4.9, 63.7, 13.2, 5.1, 3.5]], // Demoman
    [[3.6, 3.4, 1.6, 74.4, -2.6, 0.4, 4.4], [-0.9, -2.4, -1.2, 67.1, 16.7, 0.6, 0.3]],   // Heavy
    [[9.2, 12.1, -3.1, 65.6, -7.4, 0.0, 1.7], [11.7, 2.4, -2.5, 54.1, -7.3, -8.8, 12.1]], // Engineer
    [[4.9, -3.7, -1.0, 74.2, -4.1, 1.5, 2.2], [7.0, 0.0, -2.0, 60.0, 0.0, 0.0, 0.0]],    // Medic
    [[2.1, -4.2, -1.6, 74.3, -4.3, 6.0, 2.2], [3.0, -11.0, -3.9, 72.0, -3.4, 5.7, 9.8]], // Sniper
    [[7.4, 5.2, 0.1, 72.8, -3.3, 0.2, 2.7], [10.1, 2.8, -1.2, 68.1, 6.5, -6.0, 5.6]],    // Spy
];
/// Simulation records kept per player, enough for the largest rewind plus the evaluation delay.
const HEAD_RECORDS: usize = 90;
/// Records further apart than this (ticks) are not interpolated over: the player left the PVS or respawned.
const MAX_RECORD_GAP: f32 = 8.0;
/// A sniper dot within this many head radii (plus a small constant, degrees) of the fired aim shows the
/// fired aim. Otherwise it is stale: dots can stop updating in the demo.
const DOT_SHOWS_FIRED_RADII: f32 = 0.15;
const DOT_SHOWS_FIRED_DEGREES: f32 = 0.02;
/// Largest ticks from the shot to the update before / after it in a flick.
const FLICK_MAX_GAP: i32 = 3;
/// Rewinds tried for a head match on misses (whole ticks) and for confirmed headshots (quarter ticks).
const MISS_MAX_REWIND: i32 = 30;
const REWIND_STEPS: usize = 180;
const REWIND_STEPS_PER_TICK: f32 = 4.0;
/// Rewinds around the typical one that count as the typical rewind, ticks.
const TYPICAL_REWIND_WINDOW: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
/// A rewind explains a headshot when the fired ray passes this close to the head centre, HU.
const HEAD_MATCH: f32 = 8.0;
/// Lag offsets (rewind minus ping) tried for the typical rewind, quarter ticks.
const LAG_OFFSET_STEPS: usize = 120;

#[derive(Clone, Copy, Debug)]
struct Snap {
    tick: u32,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
}

/// A `CSniperDot`. A player can own several: a dot whose updates stopped reaching the demo stays
/// frozen where it was, so the owner's aim comes from the most recently moved / created one, and a
/// dot that stays put while its owner turns or walks is ignored.
#[derive(Clone, Copy, Debug, Default)]
struct Dot {
    owner: u32,
    pos: [f32; 3],
    entered: u32,
    moved: u32,
    /// Owner's eye angles and origin when the dot last moved.
    owner_view: Option<(f32, f32, [f32; 3])>,
    frozen: bool,
}

#[derive(Clone, Copy, Debug)]
struct Shot {
    tick: u32,
    ent: u32,
    weapon_id: i64,
    /// The shoot position sent with the bullet, whole units. The player's own origin can be stale.
    origin: Option<[f32; 3]>,
}

/// One simulation step of a player, like the server's lag compensation records.
#[derive(Clone, Copy, Debug)]
struct HeadRecord {
    sim: i64,
    tick: u32,
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
    ducking: bool,
    class: Class,
}

/// The aim around a shot relative to the fired aim (from the hit point when there is one), head radii.
#[derive(Clone, Copy, Debug)]
struct Flick {
    /// Distance of the fired aim from the segment between the updates before and after it.
    impulse: f32,
    /// Size of the jump into the fired aim and of the way back.
    jump: f32,
    back: f32,
    /// Cosine between the jump and the way back, -1 = straight back.
    turn: f32,
    /// Largest of the last 3 updates before the jump.
    prior: f32,
    prior_updates: usize,
    /// Ticks the aim stayed where it was before the jump.
    rest_ticks: i32,
    /// Jump in degrees and the distance of the fired aim.
    jump_degrees: f32,
    distance: f32,
}

/// A confirmed headshot with a hit point and how close the fired ray passes the head per rewind.
#[derive(Clone, Debug)]
struct HeadshotRewind {
    player: u64,
    tick: u32,
    target: Option<u64>,
    weapon: String,
    ping_ticks: f32,
    distance: f32,
    /// Ray to head centre, HU, per quarter tick of rewind (INFINITY = no record)
    errors: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Hurt {
    tick: u32,
    attacker_uid: u16,
    victim_uid: u16,
    custom: u16,
    /// A mini-crit with no condition on the victim or attacker that explains it.
    plain_mini_crit: bool,
}

/// Exact aim of one tick, from the sniper dot.
#[derive(Clone, Copy, Debug)]
struct AimSample {
    j: i32,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

/// Consecutive ticks with the same aim. `pos` is relative to the aim at the shot, in degrees.
#[derive(Clone, Copy, Debug)]
struct Update {
    start: i32,
    end: i32,
    pos: (f32, f32),
}

/// What is known about a rifle bullet when it is judged.
struct ShotInfo<'s> {
    shooter: &'s Player,
    sid: u64,
    hurt: Option<Hurt>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SnapKind {
    Held,
    Silent,
    /// Silent snap judged by its distance from the aim path, whatever the preceding motion.
    Outlier,
}

#[derive(Clone, Debug)]
struct SnapEvent {
    kind: SnapKind,
    /// Snap size in degrees and in head radii at the hit distance.
    angle: f32,
    head_radii: f32,
    /// Largest preceding update, degrees.
    prior_max: f32,
    /// Distance of the next update from the pre-snap aim, as a fraction of the snap.
    return_ratio: f32,
    /// Distance of the shot aim from the straight path between the updates around it, degrees.
    path_deviation: f32,
    eye_angle: f32,
    gap_ticks: i32,
    distance: f32,
}

/// Out-and-back jump of the update right after the shot.
#[derive(Clone, Debug)]
struct PostSpike {
    /// Yaw (scaled by cos pitch) and pitch of the jump, degrees.
    yaw: f32,
    pitch: f32,
    /// Distance of the update after the spike from the shot aim, as a fraction of the spike.
    return_ratio: f32,
    /// Ticks from the shot to the spike.
    delay: i32,
    distance: f32,
}

fn wrap180(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// (yaw, pitch) of the direction from `from` to `to`, Source conventions (pitch positive = down).
fn dir_angles(from: [f32; 3], to: [f32; 3]) -> (f32, f32) {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let dz = to[2] - from[2];
    (dy.atan2(dx).to_degrees(), -(dz.atan2((dx * dx + dy * dy).sqrt())).to_degrees())
}

fn sniper_eye(snap: &Snap) -> [f32; 3] {
    [snap.pos[0], snap.pos[1], snap.pos[2] + if snap.ducking { 45.0 } else { 75.0 }]
}

fn snap_at(hist: &VecDeque<Snap>, tick: u32) -> Option<&Snap> {
    let idx = hist.partition_point(|s| s.tick < tick);
    hist.get(idx).filter(|s| s.tick == tick)
}

fn len2(v: (f32, f32)) -> f32 {
    v.0.hypot(v.1)
}

fn sub2(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 - b.0, a.1 - b.1)
}

/// TF_WEAPON_SNIPERRIFLE, TF_WEAPON_SNIPERRIFLE_DECAP, TF_WEAPON_SNIPERRIFLE_CLASSIC.
fn is_rifle(weapon_id: i64) -> bool {
    matches!(weapon_id, 17 | 77 | 99)
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

fn head_model(class: Class, ducking: bool) -> Option<&'static [f32; 7]> {
    let i = match class {
        Class::Scout => 0,
        Class::Soldier => 1,
        Class::Pyro => 2,
        Class::Demoman => 3,
        Class::Heavy => 4,
        Class::Engineer => 5,
        Class::Medic => 6,
        Class::Sniper => 7,
        Class::Spy => 8,
        _ => return None,
    };
    Some(&HEAD_MODEL[i][usize::from(ducking)])
}

/// Head centre of record `i`, its velocity taken from the records two steps away.
fn head_of(records: &VecDeque<HeadRecord>, i: usize) -> Option<[f32; 3]> {
    let r = records.get(i)?;
    let m = head_model(r.class, r.ducking)?;
    let (a, b) = (&records[i.saturating_sub(2)], &records[(i + 2).min(records.len() - 1)]);
    let dt = (b.sim - a.sim).max(1) as f32;
    let v = [(b.pos[0] - a.pos[0]) / dt, (b.pos[1] - a.pos[1]) / dt];
    let (sin, cos) = r.yaw.to_radians().sin_cos();
    let f = [cos, sin];
    let l = [-sin, cos];
    let p = r.pitch / 90.0;
    let vf = (v[0] * f[0] + v[1] * f[1]) / 4.5;
    let vl = (v[0] * l[0] + v[1] * l[1]) / 4.5;
    let forward = m[0] + m[1] * p + m[5] * vf;
    let left = m[2] + m[6] * vl;
    Some([
        r.pos[0] + f[0] * forward + l[0] * left,
        r.pos[1] + f[1] * forward + l[1] * left,
        r.pos[2] + m[3] + m[4] * p,
    ])
}

/// Head centre at a fractional simulation tick, interpolated between records like lag compensation.
fn head_at(records: &VecDeque<HeadRecord>, sim: f32) -> Option<[f32; 3]> {
    let i = records.partition_point(|r| (r.sim as f32) < sim);
    let b = records.get(i)?;
    if b.sim as f32 == sim {
        return head_of(records, i);
    }
    let a = records.get(i.checked_sub(1)?)?;
    let gap = (b.sim - a.sim) as f32;
    if gap > MAX_RECORD_GAP {
        return None;
    }
    let t = (sim - a.sim as f32) / gap;
    let (ha, hb) = (head_of(records, i - 1)?, head_of(records, i)?);
    Some([ha[0] + (hb[0] - ha[0]) * t, ha[1] + (hb[1] - ha[1]) * t, ha[2] + (hb[2] - ha[2]) * t])
}

/// Distance of `point` from the ray, `f32::INFINITY` when it is behind the shooter.
fn ray_error(origin: [f32; 3], dir: [f32; 3], point: [f32; 3]) -> f32 {
    let q = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
    let t = q[0] * dir[0] + q[1] * dir[1] + q[2] * dir[2];
    if t <= 0.0 {
        return f32::INFINITY;
    }
    (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - t * t).max(0.0).sqrt()
}

fn unit_towards(from: [f32; 3], to: [f32; 3]) -> Option<[f32; 3]> {
    let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    (n > 0.0).then(|| [d[0] / n, d[1] / n, d[2] / n])
}

fn segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ab = sub2(b, a);
    let l = ab.0 * ab.0 + ab.1 * ab.1;
    if l < 1e-12 {
        return len2(sub2(p, a));
    }
    let ap = sub2(p, a);
    let t = ((ap.0 * ab.0 + ap.1 * ab.1) / l).clamp(0.0, 1.0);
    len2(sub2(p, (a.0 + ab.0 * t, a.1 + ab.1 * t)))
}

/// Lag offset (rewind minus ping, ticks) that lets the most of a player's headshots land.
fn typical_lag_offset(shots: &[&HeadshotRewind]) -> f32 {
    let mut best = (0, 0.0);
    for step in 0..LAG_OFFSET_STEPS {
        let offset = step as f32 / REWIND_STEPS_PER_TICK;
        let landed = shots.iter().filter(|s| error_at(s, offset + s.ping_ticks).is_some_and(|e| e <= HEAD_MATCH)).count();
        if landed > best.0 {
            best = (landed, offset);
        }
    }
    best.1
}

fn error_at(shot: &HeadshotRewind, rewind: f32) -> Option<f32> {
    let step = (rewind * REWIND_STEPS_PER_TICK).round();
    if step < 0.0 {
        return None;
    }
    shot.errors.get(step as usize).copied().filter(|e| e.is_finite())
}

/// Best rewind of a headshot and the ray to head distance there.
fn best_rewind(shot: &HeadshotRewind) -> Option<(f32, f32)> {
    let mut best = None;
    for (i, &e) in shot.errors.iter().enumerate() {
        if e.is_finite() && best.is_none_or(|(_, b)| e < b) {
            best = Some((i as f32 / REWIND_STEPS_PER_TICK, e));
        }
    }
    best
}

/// Distance of `cur` from the straight, time-interpolated path from `prev` to `next`, and the length of
/// that path.
fn path_deviation(prev: &Update, cur: &Update, next: &Update) -> (f32, f32) {
    let t = (cur.start - prev.end) as f32 / (next.start - prev.end) as f32;
    let path = sub2(next.pos, prev.pos);
    let expected = (prev.pos.0 + path.0 * t, prev.pos.1 + path.1 * t);
    (len2(sub2(cur.pos, expected)), len2(path))
}

/// Groups aim samples (sorted by tick) into updates, relative to the aim at `reference`.
fn group_updates(samples: &[AimSample], reference: &AimSample) -> Vec<Update> {
    let cos_pitch = reference.pitch.to_radians().cos();
    let mut updates: Vec<Update> = Vec::new();
    for s in samples {
        let pos = (wrap180(s.yaw - reference.yaw) * cos_pitch, s.pitch - reference.pitch);
        if let Some(last) = updates.last_mut() {
            if s.j - last.end <= 1 && len2(sub2(pos, last.pos)) < SAME_AIM_EPSILON {
                last.end = s.j;
                continue;
            }
        }
        updates.push(Update { start: s.j, end: s.j, pos });
    }
    updates
}

pub struct SniperAim {
    params: Parameters,
    history: FxHashMap<u32, VecDeque<Snap>>,
    /// CSniperDot entity -> its state
    dots: FxHashMap<u32, Dot>,
    dot_history: FxHashMap<u32, VecDeque<(u32, [f32; 3])>>,
    pending: Vec<Shot>,
    hurts: VecDeque<Hurt>,
    /// Simulation records of every player, by entity
    heads: FxHashMap<u32, VecDeque<HeadRecord>>,
    /// Hit points of bullets: tick, victim entity, position
    bloods: VecDeque<(u32, u32, [f32; 3])>,
    /// Newest simulation tick of any player per demo tick
    server_ticks: VecDeque<(u32, i64)>,
    rewinds: Vec<HeadshotRewind>,
    /// Shots that already have a detection
    reported: HashSet<(u64, u32)>,
    /// Silent flicks too small to flag anyone, reported when the player is flagged
    weak_flicks: Vec<Detection>,
    /// Plain flicks. Humans make these often enough that one kind pairing with another is no evidence,
    /// so they are reported from `gated_min_detections` per player or when the player is flagged.
    plain_flicks: Vec<Detection>,
}

impl Default for SniperAim {
    fn default() -> Self {
        Self::new()
    }
}

impl SniperAim {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_snap_head_radii".to_string(), Parameter::Float(3.0)),
                ("min_snap_degrees".to_string(), Parameter::Float(0.3)),
                ("max_snap_degrees".to_string(), Parameter::Float(10.0)),
                ("held_min_ratio".to_string(), Parameter::Float(3.5)),
                ("held_min_stay".to_string(), Parameter::Float(0.6)),
                ("check_silent".to_string(), Parameter::Bool(true)),
                ("silent_min_ratio".to_string(), Parameter::Float(3.0)),
                ("silent_max_return".to_string(), Parameter::Float(0.4)),
                ("max_snap_gap_ticks".to_string(), Parameter::Int(3)),
                ("min_dot_distance".to_string(), Parameter::Float(256.0)),
                ("check_eye_silent".to_string(), Parameter::Bool(true)),
                ("eye_silent_min_degrees".to_string(), Parameter::Float(1.0)),
                ("check_silent_outlier".to_string(), Parameter::Bool(true)),
                ("outlier_max_path_ratio".to_string(), Parameter::Float(0.4)),
                ("check_post_spike".to_string(), Parameter::Bool(true)),
                ("post_spike_degrees".to_string(), Parameter::Float(1.0)),
                ("post_spike_tolerance".to_string(), Parameter::Float(0.3)),
                ("post_spike_max_return".to_string(), Parameter::Float(0.3)),
                ("check_flick".to_string(), Parameter::Bool(true)),
                ("silent_flick_min_impulse".to_string(), Parameter::Float(1.5)),
                ("silent_flick_max_turn".to_string(), Parameter::Float(-0.85)),
                ("miss_flick_min_impulse".to_string(), Parameter::Float(3.0)),
                ("miss_flick_max_head_error".to_string(), Parameter::Float(8.0)),
                ("plain_flick_min_jump".to_string(), Parameter::Float(3.0)),
                ("plain_flick_min_ratio".to_string(), Parameter::Float(3.5)),
                ("plain_flick_large_jump".to_string(), Parameter::Float(6.0)),
                ("plain_flick_large_ratio".to_string(), Parameter::Float(1.5)),
                ("body_flick_min_jump".to_string(), Parameter::Float(5.0)),
                ("body_flick_min_ratio".to_string(), Parameter::Float(1.5)),
                ("body_flick_min_rest_ticks".to_string(), Parameter::Int(3)),
                ("check_backtrack".to_string(), Parameter::Bool(true)),
                ("backtrack_min_head_error".to_string(), Parameter::Float(16.0)),
                ("backtrack_min_rewind_offset".to_string(), Parameter::Float(4.0)),
                ("backtrack_min_headshots".to_string(), Parameter::Int(4)),
                ("gated_min_detections".to_string(), Parameter::Int(3)),
                ("flag_min_detections".to_string(), Parameter::Int(2)),
                ("flag_window_ticks".to_string(), Parameter::Int(20000)),
                ("silent_flick_weak_impulse".to_string(), Parameter::Float(1.0)),
                ("backtrack_weak_min_head_error".to_string(), Parameter::Float(9.0)),
                ("backtrack_weak_min_rewind_offset".to_string(), Parameter::Float(3.0)),
            ]),
            history: FxHashMap::default(),
            dots: FxHashMap::default(),
            dot_history: FxHashMap::default(),
            pending: Vec::new(),
            hurts: VecDeque::new(),
            heads: FxHashMap::default(),
            bloods: VecDeque::new(),
            server_ticks: VecDeque::new(),
            rewinds: Vec::new(),
            reported: HashSet::new(),
            weak_flicks: Vec::new(),
            plain_flicks: Vec::new(),
        }
    }

    /// Exact aim samples of the shot window (from the sniper dot), the sample at the shot, and their
    /// updates relative to the aim at the shot.
    fn aim_updates(shot_tick: u32, shooter: &VecDeque<Snap>, dots: &VecDeque<(u32, [f32; 3])>) -> Option<(Vec<AimSample>, AimSample, Vec<Update>)> {
        let mut samples = Vec::new();
        for &(tick, dot) in dots {
            let j = tick as i64 - shot_tick as i64;
            if j < -(PRE_TICKS as i64) || j > POST_TICKS as i64 {
                continue;
            }
            let Some(snap) = snap_at(shooter, tick) else { continue };
            let eye = sniper_eye(snap);
            let distance = dist(eye, dot);
            if distance < MIN_SAMPLE_DISTANCE {
                continue;
            }
            let (yaw, pitch) = dir_angles(eye, dot);
            samples.push(AimSample { j: j as i32, yaw, pitch, distance });
        }
        let shot = *samples.iter().find(|s| s.j == 0)?;
        let updates = group_updates(&samples, &shot);
        Some((samples, shot, updates))
    }

    /// Whether the coarse eye angles change from tick `j0` to `j1` (relative to the shot) by `angle`
    /// degrees, within their quantization.
    fn eye_confirms(shooter: &VecDeque<Snap>, shot_tick: u32, j0: i32, j1: i32, cos_pitch: f32, angle: f32) -> Option<f32> {
        let e0 = snap_at(shooter, (shot_tick as i64 + j0 as i64) as u32)?;
        let e1 = snap_at(shooter, (shot_tick as i64 + j1 as i64) as u32)?;
        let eye_angle = (wrap180(e1.yaw - e0.yaw) * cos_pitch).hypot(e1.pitch - e0.pitch);
        ((eye_angle - angle).abs() <= EYE_TOLERANCE + EYE_TOLERANCE_RATIO * angle).then_some(eye_angle)
    }

    /// Classifies the aim change on the shot tick, `None` when it is not an aimbot snap.
    fn classify(&self, shot_tick: u32, shooter: &VecDeque<Snap>, dots: &VecDeque<(u32, [f32; 3])>, headshot: bool) -> Option<SnapEvent> {
        let min_head_radii: f32 = get_parameter_value(&self.params, "min_snap_head_radii");
        let min_angle: f32 = get_parameter_value(&self.params, "min_snap_degrees");
        let max_angle: f32 = get_parameter_value(&self.params, "max_snap_degrees");
        let held_min_ratio: f32 = get_parameter_value(&self.params, "held_min_ratio");
        let held_min_stay: f32 = get_parameter_value(&self.params, "held_min_stay");
        let check_silent: bool = get_parameter_value(&self.params, "check_silent");
        let silent_min_ratio: f32 = get_parameter_value(&self.params, "silent_min_ratio");
        let silent_max_return: f32 = get_parameter_value(&self.params, "silent_max_return");
        let max_gap: i32 = get_parameter_value(&self.params, "max_snap_gap_ticks");
        let min_dot_distance: f32 = get_parameter_value(&self.params, "min_dot_distance");
        let check_outlier: bool = get_parameter_value(&self.params, "check_silent_outlier");
        let outlier_max_path_ratio: f32 = get_parameter_value(&self.params, "outlier_max_path_ratio");

        let (samples, shot, updates) = Self::aim_updates(shot_tick, shooter, dots)?;
        let i0 = updates.iter().position(|u| u.start <= 0 && 0 <= u.end)?;
        // The aimbot moves the aim in the usercmd that fires.
        if i0 == 0 || updates[i0].start != 0 {
            return None;
        }
        let cur = updates[i0];
        let prev = updates[i0 - 1];
        let gap = cur.start - prev.end;
        if gap > max_gap {
            return None;
        }

        let snap_vec = sub2(cur.pos, prev.pos);
        let angle = len2(snap_vec);
        let head_angle = (HEAD_RADIUS / shot.distance.min(MAX_HEAD_DISTANCE)).atan().to_degrees();
        let head_radii = angle / head_angle;
        let prev_sample = samples.iter().find(|s| s.j == prev.end)?;
        if shot.distance.min(prev_sample.distance) < min_dot_distance {
            return None;
        }

        let mut prior = Vec::new();
        for k in (1..i0).rev() {
            if updates[k].start < cur.start - PRIOR_WINDOW || prior.len() >= PRIOR_UPDATES {
                break;
            }
            prior.push(len2(sub2(updates[k].pos, updates[k - 1].pos)));
        }
        // With little preceding movement (e.g. right after scoping in) only a silent snap is judged, a
        // held snap from rest is a normal flick. With none at all the aim must have rested.
        let rested = prior.len() < 2;
        let snap_ok = head_radii >= min_head_radii
            && angle >= min_angle
            && angle <= max_angle
            && !(prior.is_empty() && prev.end - prev.start + 1 < MIN_REST_TICKS);
        let prior_max = prior.iter().copied().fold(0.0f32, f32::max);

        // A tick without a packet splits an unchanged aim into two updates; the next one that moved counts.
        let next = updates[i0 + 1..].iter().find(|u| len2(sub2(u.pos, cur.pos)) >= SAME_AIM_EPSILON);
        let return_ratio = match next {
            Some(next) => len2(sub2(next.pos, prev.pos)) / angle.max(f32::EPSILON),
            None if cur.end >= HOLD_TICKS => 1.0,
            None => return None,
        };

        // The coarse eye angles must show the same change.
        let cos_pitch = shot.pitch.to_radians().cos();
        let eye_angle = Self::eye_confirms(shooter, shot_tick, prev.end, cur.start, cos_pitch, angle)?;

        // Outlier: the shot aim lies off the straight path from the previous to the next update by several
        // head radii and the aim comes back, so the aim left the path for exactly one update. Unlike
        // `silent_snap` it does not compare the jump with the preceding motion, which hides silent snaps
        // fired while turning. The eye angles must confirm the way back as well.
        let path = next.filter(|n| n.start - cur.end <= max_gap).map(|n| (n, path_deviation(&prev, &cur, n)));
        let outlier = check_outlier
            && !(prior.is_empty() && prev.end - prev.start + 1 < MIN_REST_TICKS)
            && path.is_some_and(|(n, (deviation, span))| {
                deviation / head_angle >= min_head_radii
                    && deviation >= min_angle
                    && deviation <= max_angle
                    && span <= outlier_max_path_ratio * deviation
                    && samples.iter().find(|s| s.j == n.start).is_some_and(|s| s.distance >= min_dot_distance)
                    && Self::eye_confirms(shooter, shot_tick, cur.end, n.start, cos_pitch, len2(sub2(n.pos, cur.pos))).is_some()
            });

        let kind = if snap_ok && check_silent && return_ratio <= silent_max_return && angle >= silent_min_ratio * prior_max {
            SnapKind::Silent
        } else if snap_ok && !rested && headshot && return_ratio >= held_min_stay && angle >= held_min_ratio * prior_max {
            SnapKind::Held
        } else if outlier {
            SnapKind::Outlier
        } else {
            return None;
        };
        Some(SnapEvent {
            kind,
            angle,
            head_radii,
            prior_max,
            return_ratio,
            path_deviation: path.map_or(0.0, |(_, (deviation, _))| deviation),
            eye_angle,
            gap_ticks: gap,
            distance: shot.distance,
        })
    }

    /// Out-and-back jump of about `post_spike_degrees` in both yaw and pitch on the update right after the
    /// shot (seen after the reference cheater 76561198022035121's shots). The aim then returns to the
    /// shot aim. No mouse moves out diagonally by a fixed angle and back within one update.
    fn post_spike(&self, shot_tick: u32, shooter: &VecDeque<Snap>, dots: &VecDeque<(u32, [f32; 3])>) -> Option<PostSpike> {
        let degrees: f32 = get_parameter_value(&self.params, "post_spike_degrees");
        let tolerance: f32 = get_parameter_value(&self.params, "post_spike_tolerance");
        let max_return: f32 = get_parameter_value(&self.params, "post_spike_max_return");
        let max_gap: i32 = get_parameter_value(&self.params, "max_snap_gap_ticks");
        let min_dot_distance: f32 = get_parameter_value(&self.params, "min_dot_distance");

        let (samples, shot, updates) = Self::aim_updates(shot_tick, shooter, dots)?;
        let i0 = updates.iter().position(|u| u.start <= 0 && 0 <= u.end)?;
        let cur = updates[i0];
        let mut moved = updates[i0 + 1..].iter().filter(|u| len2(sub2(u.pos, cur.pos)) >= SAME_AIM_EPSILON);
        let spike = *moved.next()?;
        let back = *updates.iter().skip_while(|u| u.start <= spike.start).find(|u| len2(sub2(u.pos, spike.pos)) >= SAME_AIM_EPSILON)?;
        if spike.start > HOLD_TICKS || spike.start - cur.end > max_gap || back.start - spike.end > max_gap {
            return None;
        }
        let far = |j: i32| samples.iter().find(|s| s.j == j).is_some_and(|s| s.distance >= min_dot_distance);
        if shot.distance < min_dot_distance || !far(spike.start) || !far(back.start) {
            return None;
        }
        let (yaw, pitch) = sub2(spike.pos, cur.pos);
        let near = |v: f32| (v.abs() - degrees).abs() <= tolerance;
        if !near(yaw) || !near(pitch) {
            return None;
        }
        let size = len2((yaw, pitch));
        let return_ratio = len2(sub2(back.pos, cur.pos)) / size;
        if return_ratio > max_return {
            return None;
        }
        let cos_pitch = shot.pitch.to_radians().cos();
        Self::eye_confirms(shooter, shot_tick, cur.end, spike.start, cos_pitch, size)?;
        Self::eye_confirms(shooter, shot_tick, spike.end, back.start, cos_pitch, len2(sub2(back.pos, spike.pos)))?;
        Some(PostSpike { yaw, pitch, return_ratio, delay: spike.start, distance: shot.distance })
    }

    /// Silent snap seen in the coarse eye angles alone (for hits without a usable dot): the eye angles
    /// rest, jump on the shot tick and are back on the exact same quantized value by the next sample.
    /// Returns the jump in degrees.
    fn eye_spike(&self, shot_tick: u32, shooter: &VecDeque<Snap>) -> Option<f32> {
        let min_angle: f32 = get_parameter_value(&self.params, "eye_silent_min_degrees");
        let max_angle: f32 = get_parameter_value(&self.params, "max_snap_degrees");
        let at = snap_at(shooter, shot_tick)?;
        let before: Vec<&Snap> =
            shooter.iter().filter(|s| s.tick < shot_tick && s.tick + EYE_REST_TICKS as u32 >= shot_tick).collect();
        let last = *before.last()?;
        let after = shooter.iter().find(|s| s.tick > shot_tick)?;
        if last.tick + EYE_MAX_GAP < shot_tick || after.tick > shot_tick + EYE_MAX_GAP {
            return None;
        }
        let same = |a: &Snap, b: &Snap, yaw_tol: f32, pitch_tol: f32| {
            wrap180(a.yaw - b.yaw).abs() <= yaw_tol && (a.pitch - b.pitch).abs() <= pitch_tol
        };
        // Exact return, and at most one quantization step of movement before the shot.
        if !same(after, last, EYE_SAME, EYE_SAME) || before.len() < 2 || !before.iter().all(|s| same(s, last, EYE_YAW_STEP, EYE_PITCH_STEP)) {
            return None;
        }
        let angle = (wrap180(at.yaw - last.yaw) * last.pitch.to_radians().cos()).hypot(at.pitch - last.pitch);
        (angle >= min_angle && angle <= max_angle).then_some(angle)
    }

    /// The aim around a shot relative to the fired aim: the hit point when known, the dot on a miss. All
    /// aim samples use the bullet's shoot position, the player's own origin can be stale. `None` when
    /// the dot does not show the fired aim (it can stop updating) or there is no update before the shot.
    fn flick(shot: &Shot, shooter: &VecDeque<Snap>, dots: &VecDeque<(u32, [f32; 3])>, blood: Option<[f32; 3]>) -> Option<Flick> {
        let eye0 = sniper_eye(snap_at(shooter, shot.tick)?);
        let origin = shot.origin.unwrap_or(eye0);
        let corr = [origin[0] - eye0[0], origin[1] - eye0[1], origin[2] - eye0[2]];
        let dot_at_shot = dots.iter().find(|(t, _)| *t == shot.tick).map(|(_, d)| *d);
        let fired = blood.or(dot_at_shot)?;
        let distance = dist(origin, fired);
        if distance < MIN_SAMPLE_DISTANCE {
            return None;
        }
        let (fired_yaw, fired_pitch) = dir_angles(origin, fired);
        let cos_pitch = fired_pitch.to_radians().cos();
        let head = (HEAD_RADIUS / distance.min(MAX_HEAD_DISTANCE)).atan().to_degrees();

        let mut pre: Vec<(i32, (f32, f32))> = Vec::new();
        let mut post: Vec<(i32, (f32, f32))> = Vec::new();
        for &(tick, dot) in dots {
            let j = tick as i64 - shot.tick as i64;
            if j < -(PRE_TICKS as i64) || j > POST_TICKS as i64 {
                continue;
            }
            let Some(snap) = snap_at(shooter, tick) else { continue };
            let e = sniper_eye(snap);
            let eye = [e[0] + corr[0], e[1] + corr[1], e[2] + corr[2]];
            if dist(eye, dot) < MIN_SAMPLE_DISTANCE {
                continue;
            }
            let (yaw, pitch) = dir_angles(eye, dot);
            let pos = (wrap180(yaw - fired_yaw) * cos_pitch, pitch - fired_pitch);
            if j < 0 {
                pre.push((j as i32, pos));
            } else {
                post.push((j as i32, pos));
            }
        }
        let &(p1_j, p1) = pre.last()?;
        if -p1_j > FLICK_MAX_GAP {
            return None;
        }
        let &(q_j, q) = post.first()?;
        let p3 = if q_j == 0 {
            if len2(q) >= DOT_SHOWS_FIRED_RADII * head + DOT_SHOWS_FIRED_DEGREES {
                return None;
            }
            post.iter().find(|(_, pos)| len2(sub2(*pos, q)) >= SAME_AIM_EPSILON).copied()
        } else {
            Some((q_j, q))
        };

        let jump = len2(p1);
        let back = p3.map_or(0.0, |(_, pos)| len2(pos));
        let impulse = p3.filter(|(j, _)| *j <= FLICK_MAX_GAP).map_or(0.0, |(_, pos)| segment_distance((0.0, 0.0), p1, pos));
        let turn = p3.map_or(1.0, |(_, pos)| -(p1.0 * pos.0 + p1.1 * pos.1) / (jump * back).max(1e-9));

        let mut updates: Vec<(f32, f32)> = Vec::new();
        for &(_, pos) in &pre {
            if updates.last().is_none_or(|last| len2(sub2(pos, *last)) >= SAME_AIM_EPSILON) {
                updates.push(pos);
            }
        }
        let steps = updates.windows(2).rev().take(3).map(|w| len2(sub2(w[1], w[0])));
        let prior = steps.clone().fold(0.0f32, f32::max);
        Some(Flick {
            impulse: impulse / head,
            jump: jump / head,
            back: back / head,
            turn,
            prior: prior / head,
            prior_updates: steps.count(),
            rest_ticks: pre.iter().filter(|(_, pos)| len2(sub2(*pos, p1)) < SAME_AIM_EPSILON).count() as i32,
            jump_degrees: jump,
            distance,
        })
    }

    /// Kind of aimbot flick of a hit, `None` when the aim could be a human's.
    fn flick_kind(&self, flick: &Flick, headshot: bool) -> Option<&'static str> {
        let max_angle: f32 = get_parameter_value(&self.params, "max_snap_degrees");
        let min_dot_distance: f32 = get_parameter_value(&self.params, "min_dot_distance");
        if flick.jump_degrees > max_angle || flick.distance < min_dot_distance {
            return None;
        }
        let min_impulse: f32 = get_parameter_value(&self.params, "silent_flick_min_impulse");
        let max_turn: f32 = get_parameter_value(&self.params, "silent_flick_max_turn");
        if flick.impulse >= min_impulse && flick.turn <= max_turn {
            return Some("silent_flick");
        }
        if flick.prior_updates == 0 {
            return None;
        }
        let jump_over = |min_jump: &str, min_ratio: &str| {
            flick.jump >= get_parameter_value::<f32>(&self.params, min_jump)
                && flick.jump >= get_parameter_value::<f32>(&self.params, min_ratio) * flick.prior
        };
        if headshot && (jump_over("plain_flick_min_jump", "plain_flick_min_ratio") || jump_over("plain_flick_large_jump", "plain_flick_large_ratio")) {
            return Some("plain_flick");
        }
        let min_rest: i32 = get_parameter_value(&self.params, "body_flick_min_rest_ticks");
        if !headshot && flick.rest_ticks >= min_rest && jump_over("body_flick_min_jump", "body_flick_min_ratio") {
            return Some("plain_flick");
        }
        None
    }

    /// Closest enemy head to the fired ray over the past rewinds: (HU, rewind ticks, entity).
    fn nearest_head(&self, shot: &Shot, shooter: &Player, state: &CheatAnalyserState, fired: [f32; 3]) -> Option<(f32, i32, u32)> {
        let origin = shot.origin?;
        let dir = unit_towards(origin, fired)?;
        let now = self.server_tick_at(shot.tick)? as f32;
        let mut best = None;
        for enemy in state.players.iter().filter(|p| p.team != shooter.team && p.entity != shooter.entity) {
            let ent = u32::from(enemy.entity);
            let Some(records) = self.heads.get(&ent) else { continue };
            for k in 0..=MISS_MAX_REWIND {
                let Some(head) = head_at(records, now - k as f32) else { continue };
                let e = ray_error(origin, dir, head);
                if best.is_none_or(|(b, _, _)| e < b) {
                    best = Some((e, k, ent));
                }
            }
        }
        best
    }

    fn server_tick_at(&self, tick: u32) -> Option<i64> {
        self.server_ticks.iter().find(|(t, _)| *t == tick).map(|(_, sim)| *sim)
    }

    /// A rifle bullet that hurt nobody. Only a silent flick onto an enemy's past head is judged.
    fn evaluate_miss(&self, shot: Shot, shooter: &Player, sid: u64, state: &CheatAnalyserState) -> Option<Detection> {
        if !get_parameter_value::<bool>(&self.params, "check_flick") {
            return None;
        }
        let shooter_hist = self.history.get(&shot.ent)?;
        let dots = self.dot_history.get(&shot.ent)?;
        let flick = Self::flick(&shot, shooter_hist, dots, None)?;
        let min_impulse: f32 = get_parameter_value(&self.params, "miss_flick_min_impulse");
        let max_turn: f32 = get_parameter_value(&self.params, "silent_flick_max_turn");
        let max_angle: f32 = get_parameter_value(&self.params, "max_snap_degrees");
        let min_dot_distance: f32 = get_parameter_value(&self.params, "min_dot_distance");
        if flick.impulse < min_impulse || flick.turn > max_turn || flick.jump_degrees > max_angle || flick.distance < min_dot_distance {
            return None;
        }
        let fired = dots.iter().find(|(t, _)| *t == shot.tick).map(|(_, d)| *d)?;
        let (error, rewind, ent) = self.nearest_head(&shot, shooter, state, fired)?;
        if error > get_parameter_value::<f32>(&self.params, "miss_flick_max_head_error") {
            return None;
        }
        let target = state.players.iter().find(|p| u32::from(p.entity) == ent).and_then(|p| player_sid(p, state));
        let round2 = |v: f32| (v * 100.0).round() / 100.0;
        Some(Detection {
            tick: shot.tick,
            algorithm: "fidoo/sniperaim".to_string(),
            player: sid,
            data: json!({
                "type": "silent_flick_miss",
                "weapon": state.get_player_weapon(shooter),
                "headshot": false,
                "target": target,
                "impulse_head_radii": round2(flick.impulse),
                "jump_head_radii": round2(flick.jump),
                "back_head_radii": round2(flick.back),
                "head_error": round2(error),
                "rewind_ticks": rewind,
                "distance": flick.distance.round(),
            }),
        })
    }

    /// How the fired ray of a confirmed headshot passes the victim's head per rewind.
    fn headshot_rewind(&self, shot: &Shot, sid: u64, hit_point: [f32; 3], victim_ent: u32, target: Option<u64>, weapon: &str, ping: u16) -> Option<HeadshotRewind> {
        let origin = shot.origin?;
        let dir = unit_towards(origin, hit_point)?;
        let now = self.server_tick_at(shot.tick)?;
        let records = self.heads.get(&victim_ent)?;
        // The head model of ducked players is too coarse for this.
        let at_shot = records.partition_point(|r| r.sim <= now);
        if at_shot == 0 || records[at_shot - 1].ducking {
            return None;
        }
        let errors = (0..REWIND_STEPS)
            .map(|step| {
                head_at(records, now as f32 - step as f32 / REWIND_STEPS_PER_TICK).map_or(f32::INFINITY, |head| ray_error(origin, dir, head))
            })
            .collect();
        Some(HeadshotRewind {
            player: sid,
            tick: shot.tick,
            target,
            weapon: weapon.to_string(),
            ping_ticks: f32::from(ping) / 15.0,
            distance: dist(origin, hit_point),
            errors,
        })
    }

    fn evaluate_shot(&mut self, shot: Shot, state: &CheatAnalyserState) -> Option<Detection> {
        if !is_rifle(shot.weapon_id) {
            return None;
        }
        let (detection, rewind, weak) = self.judge_shot(shot, state);
        self.rewinds.extend(rewind);
        self.weak_flicks.extend(weak);
        if detection.as_ref().is_some_and(|d| d.data["type"] == "plain_flick") {
            self.plain_flicks.extend(detection);
            return None;
        }
        if let Some(d) = &detection {
            self.reported.insert((d.player, d.tick));
        }
        detection
    }

    /// The exact point a bullet hit, on the victim as the server rewound it.
    fn hit_point(&self, shot: &Shot, victim_ent: u32) -> Option<[f32; 3]> {
        self.bloods.iter().find(|(t, e, _)| *e == victim_ent && *t >= shot.tick && *t <= shot.tick + 1).map(|(_, _, p)| *p)
    }

    fn judge_shot(&self, shot: Shot, state: &CheatAnalyserState) -> (Option<Detection>, Option<HeadshotRewind>, Option<Detection>) {
        let Some(info) = self.shot_info(&shot, state) else { return (None, None, None) };
        let ShotInfo { shooter, sid, hurt } = info;
        let Some(hurt) = hurt else {
            return (self.evaluate_miss(shot, shooter, sid, state), None, None);
        };
        let victim = state.players.iter().find(|p| p.info.as_ref().is_some_and(|i| u16::from(i.user_id) == hurt.victim_uid));
        let victim_ent = victim.map(|p| u32::from(p.entity));
        let hit_point = victim_ent.and_then(|ent| self.hit_point(&shot, ent));
        let rewind = if get_parameter_value::<bool>(&self.params, "check_backtrack") && HEADSHOT_CUSTOMS.contains(&hurt.custom) {
            hit_point.zip(victim_ent).and_then(|(point, ent)| {
                let target = victim.and_then(|p| player_sid(p, state));
                self.headshot_rewind(&shot, sid, point, ent, target, &state.get_player_weapon(shooter), shooter.ping)
            })
        } else {
            None
        };
        let detection = self.evaluate_hit(shot, shooter, sid, &hurt, victim, hit_point, state);
        let weak = if detection.is_none() { self.weak_flick(&shot, shooter, sid, victim, hit_point, state) } else { None };
        (detection, rewind, weak)
    }

    /// An out-and-back too small to flag anyone on its own.
    fn weak_flick(&self, shot: &Shot, shooter: &Player, sid: u64, victim: Option<&Player>, hit_point: Option<[f32; 3]>, state: &CheatAnalyserState) -> Option<Detection> {
        if !get_parameter_value::<bool>(&self.params, "check_flick") {
            return None;
        }
        let f = Self::flick(shot, self.history.get(&shot.ent)?, self.dot_history.get(&shot.ent)?, hit_point)?;
        let max_angle: f32 = get_parameter_value(&self.params, "max_snap_degrees");
        let min_dot_distance: f32 = get_parameter_value(&self.params, "min_dot_distance");
        let min_impulse: f32 = get_parameter_value(&self.params, "silent_flick_weak_impulse");
        let max_turn: f32 = get_parameter_value(&self.params, "silent_flick_max_turn");
        if f.jump_degrees > max_angle || f.distance < min_dot_distance || f.impulse < min_impulse || f.turn > max_turn {
            return None;
        }
        let round2 = |v: f32| (v * 100.0).round() / 100.0;
        Some(Detection {
            tick: shot.tick,
            algorithm: "fidoo/sniperaim".to_string(),
            player: sid,
            data: json!({
                "type": "silent_flick",
                "weak": true,
                "weapon": state.get_player_weapon(shooter),
                "target": victim.and_then(|p| player_sid(p, state)),
                "impulse_head_radii": round2(f.impulse),
                "jump_head_radii": round2(f.jump),
                "back_head_radii": round2(f.back),
                "turn": round2(f.turn),
                "aim_from_hit_point": hit_point.is_some(),
                "distance": f.distance.round(),
            }),
        })
    }

    fn shot_info<'s>(&self, shot: &Shot, state: &'s CheatAnalyserState) -> Option<ShotInfo<'s>> {
        snap_at(self.history.get(&shot.ent)?, shot.tick)?;
        let shooter = state.players.iter().find(|p| u32::from(p.entity) == shot.ent)?;
        let sid = player_sid(shooter, state)?;
        let shooter_uid = u16::from(shooter.info.as_ref()?.user_id);
        let hurt = self
            .hurts
            .iter()
            .find(|h| h.attacker_uid == shooter_uid && h.victim_uid != shooter_uid && h.tick >= shot.tick && h.tick <= shot.tick + HURT_DELAY_TICKS)
            .copied();
        Some(ShotInfo { shooter, sid, hurt })
    }

    fn evaluate_hit(&self, shot: Shot, shooter: &Player, sid: u64, hurt: &Hurt, victim: Option<&Player>, hit_point: Option<[f32; 3]>, state: &CheatAnalyserState) -> Option<Detection> {
        let shooter_hist = self.history.get(&shot.ent)?;
        // Unscoped shots have no dot; only the eye angles can be checked.
        let no_dots = VecDeque::new();
        let dots = self.dot_history.get(&shot.ent).unwrap_or(&no_dots);
        let weapon = state.get_player_weapon(shooter);
        let headshot = HEADSHOT_CUSTOMS.contains(&hurt.custom) || (weapon == SYDNEY_SLEEPER && hurt.plain_mini_crit);

        let event = self.classify(shot.tick, shooter_hist, dots, headshot);
        let target = victim.and_then(|p| player_sid(p, state));
        let round2 = |v: f32| (v * 100.0).round() / 100.0;
        let post = if event.is_none() && get_parameter_value::<bool>(&self.params, "check_post_spike") {
            self.post_spike(shot.tick, shooter_hist, dots)
        } else {
            None
        };
        if let Some(post) = post {
            return Some(Detection {
                tick: shot.tick,
                algorithm: "fidoo/sniperaim".to_string(),
                player: sid,
                data: json!({
                    "type": "post_shot_spike",
                    "weapon": weapon,
                    "headshot": headshot,
                    "target": target,
                    "spike_yaw": round2(post.yaw),
                    "spike_pitch": round2(post.pitch),
                    "return_ratio": round2(post.return_ratio),
                    "delay_ticks": post.delay,
                    "distance": post.distance.round(),
                }),
            });
        }
        let Some(event) = event else {
            if get_parameter_value::<bool>(&self.params, "check_flick") {
                let flick = Self::flick(&shot, shooter_hist, dots, hit_point);
                if let Some((kind, f)) = flick.as_ref().and_then(|f| self.flick_kind(f, headshot).map(|kind| (kind, f))) {
                    return Some(Detection {
                        tick: shot.tick,
                        algorithm: "fidoo/sniperaim".to_string(),
                        player: sid,
                        data: json!({
                            "type": kind,
                            "weapon": weapon,
                            "headshot": headshot,
                            "target": target,
                            "impulse_head_radii": round2(f.impulse),
                            "jump_head_radii": round2(f.jump),
                            "back_head_radii": round2(f.back),
                            "turn": round2(f.turn),
                            "prior_max_update_head_radii": round2(f.prior),
                            "rest_ticks": f.rest_ticks,
                            "aim_from_hit_point": hit_point.is_some(),
                            "distance": f.distance.round(),
                        }),
                    });
                }
            }
            if !get_parameter_value::<bool>(&self.params, "check_eye_silent") {
                return None;
            }
            let angle = self.eye_spike(shot.tick, shooter_hist)?;
            return Some(Detection {
                tick: shot.tick,
                algorithm: "fidoo/sniperaim".to_string(),
                player: sid,
                data: json!({
                    "type": "eye_silent_snap",
                    "weapon": weapon,
                    "headshot": headshot,
                    "target": target,
                    "eye_angle_snap": round2(angle),
                }),
            });
        };
        Some(Detection {
            tick: shot.tick,
            algorithm: "fidoo/sniperaim".to_string(),
            player: sid,
            data: json!({
                "type": match event.kind {
                    SnapKind::Held => "snap",
                    SnapKind::Silent => "silent_snap",
                    SnapKind::Outlier => "silent_outlier",
                },
                "weapon": state.get_player_weapon(shooter),
                "headshot": headshot,
                "target": target,
                "snap_angle": round2(event.angle),
                "snap_head_radii": round2(event.head_radii),
                "prior_max_update": round2(event.prior_max),
                "return_ratio": round2(event.return_ratio),
                "path_deviation": round2(event.path_deviation),
                "eye_angle_snap": round2(event.eye_angle),
                "gap_ticks": event.gap_ticks,
                "distance": event.distance.round(),
            }),
        })
    }
}

impl<'a> CheatAlgorithm<'a> for SniperAim {
    fn default(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/sniperaim"
    }

    /// Plain flicks, off-lag headshots and weak flicks are judged per player over all their shots.
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn tooltip(&self) -> &'static str {
        "Detects low-FOV Sniper aimbots: silent and plain flicks measured against the exact fired aim, and headshots that only land on a past head position (backtrack)."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::TempEntities, MessageType::GameEvent, MessageType::PacketEntities])
    }

    fn init(&mut self) -> Result<(), Error> {
        self.history.clear();
        self.dots.clear();
        self.dot_history.clear();
        self.pending.clear();
        self.hurts.clear();
        self.heads.clear();
        self.bloods.clear();
        self.server_ticks.clear();
        self.rewinds.clear();
        self.reported.clear();
        self.weak_flicks.clear();
        self.plain_flicks.clear();
        Ok(())
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        let ticknum: u32 = tick.into();
        match message {
            Message::TempEntities(msg) => {
                for event in &msg.events {
                    let class = &parser_state.server_classes[usize::from(event.class_id)].name;
                    if class.as_str() == "CTETFBlood" {
                        let coord = |id: SendPropIdentifier| {
                            event.props.iter().find(|p| p.identifier == id).and_then(|p| f32::try_from(&p.value).ok())
                        };
                        let ent = event.props.iter().find(|p| p.identifier == BLOOD_ENT).and_then(|p| i64::try_from(&p.value).ok());
                        if let (Some(x), Some(y), Some(z), Some(ent)) = (coord(BLOOD_X), coord(BLOOD_Y), coord(BLOOD_Z), ent) {
                            self.bloods.push_back((ticknum, ent as u32, [x, y, z]));
                        }
                        continue;
                    }
                    if class.as_str() != "CTEFireBullets" {
                        continue;
                    }
                    let prop = |id: SendPropIdentifier| {
                        event.props.iter().find(|p| p.identifier == id).and_then(|p| i64::try_from(&p.value).ok())
                    };
                    if let (Some(player), Some(weapon_id)) = (prop(BULLETS_PLAYER), prop(BULLETS_WEAPON)) {
                        if is_rifle(weapon_id) {
                            let origin = event
                                .props
                                .iter()
                                .find(|p| p.identifier == BULLETS_ORIGIN)
                                .and_then(|p| Vector::try_from(&p.value).ok())
                                .map(|v| [v.x, v.y, v.z]);
                            // m_iPlayer is the entity index - 1
                            self.pending.push(Shot { tick: ticknum, ent: player as u32 + 1, weapon_id, origin });
                        }
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
                        // The index was reused by another entity.
                        if entering {
                            self.dots.remove(&id);
                        }
                        continue;
                    }
                    let dot = self.dots.entry(id).or_default();
                    if entering {
                        dot.entered = ticknum;
                    }
                    let mut apply = |identifier: SendPropIdentifier, value: &tf_demo_parser::demo::sendprop::SendPropValue| {
                        match identifier {
                            DOT_ORIGIN => {
                                if let Ok(v) = Vector::try_from(value) {
                                    dot.pos = [v.x, v.y, v.z];
                                    dot.moved = ticknum;
                                }
                            }
                            DOT_OWNER => {
                                if let Ok(v) = i64::try_from(value) {
                                    dot.owner = (v as u32) & 0x7FF;
                                }
                            }
                            _ => {}
                        }
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
                    let by_uid = |uid: u16| state.players.iter().find(|p| p.info.as_ref().is_some_and(|i| u16::from(i.user_id) == uid));
                    let other_mini_crit = by_uid(h.user_id).is_some_and(|p| VICTIM_MINI_CRIT_CONDS.iter().any(|&c| has_cond(p, c)))
                        || by_uid(h.attacker).is_some_and(|p| ATTACKER_MINI_CRIT_CONDS.iter().any(|&c| has_cond(p, c)));
                    self.hurts.push_back(Hurt {
                        tick: ticknum,
                        attacker_uid: h.attacker,
                        victim_uid: h.user_id,
                        custom: h.custom,
                        plain_mini_crit: h.mini_crit && !other_mini_crit,
                    });
                }
            }
            _ => {}
        }
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let ticknum = u32::from(state.tick);
        let mut newest_sim = None;
        for p in &state.players {
            if p.state != PlayerState::Alive || !p.in_pvs {
                continue;
            }
            let sim = state.sim_tick(p);
            newest_sim = newest_sim.max(Some(sim));
            let records = self.heads.entry(u32::from(p.entity)).or_default();
            if records.back().is_some_and(|r| r.sim > sim || r.tick > ticknum) {
                records.clear();
            }
            if records.back().is_none_or(|r| r.sim < sim) {
                records.push_back(HeadRecord {
                    sim,
                    tick: ticknum,
                    pos: [p.position.x, p.position.y, p.position.z],
                    yaw: p.view_angle,
                    pitch: p.pitch_angle,
                    ducking: p.is_ducking(),
                    class: p.class,
                });
                if records.len() > HEAD_RECORDS {
                    records.pop_front();
                }
            }
        }
        if let Some(sim) = newest_sim {
            if self.server_ticks.back().is_some_and(|(t, _)| *t > ticknum) {
                self.server_ticks.clear();
            }
            if self.server_ticks.back().is_none_or(|(t, _)| *t < ticknum) {
                self.server_ticks.push_back((ticknum, sim));
            }
            while self.server_ticks.len() > HISTORY_TICKS {
                self.server_ticks.pop_front();
            }
        }
        while self.bloods.front().is_some_and(|b| b.0 + 2 * EVAL_DELAY < ticknum) {
            self.bloods.pop_front();
        }
        for p in &state.players {
            if p.class != Class::Sniper || p.state != PlayerState::Alive {
                continue;
            }
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
            });
            while hist.len() > HISTORY_TICKS {
                hist.pop_front();
            }
        }
        for dot in self.dots.values_mut() {
            let Some(owner) = state.players.iter().find(|p| u32::from(p.entity) == dot.owner) else { continue };
            let view = (owner.view_angle, owner.pitch_angle, [owner.position.x, owner.position.y, owner.position.z]);
            match dot.owner_view {
                Some((yaw, pitch, pos)) if dot.moved != ticknum => {
                    if wrap180(view.0 - yaw).abs() > FROZEN_DOT_YAW
                        || (view.1 - pitch).abs() > FROZEN_DOT_PITCH
                        || dist(view.2, pos) > FROZEN_DOT_MOVE
                    {
                        dot.frozen = true;
                    }
                }
                _ => {
                    dot.owner_view = Some(view);
                    dot.frozen = false;
                }
            }
        }
        // Each owner's most recently moved / created dot first; the others are skipped below.
        let mut dots: Vec<(u32, &Dot)> = self.dots.iter().filter(|(_, d)| !d.frozen).map(|(&id, dot)| (id, dot)).collect();
        dots.sort_unstable_by_key(|&(id, d)| (d.owner, std::cmp::Reverse(d.moved.max(d.entered)), id));
        for (_, &Dot { owner, pos, .. }) in dots {
            let hist = self.dot_history.entry(owner).or_default();
            if hist.back().is_some_and(|(t, _)| *t > ticknum) {
                hist.clear();
            }
            if hist.back().is_some_and(|(t, _)| *t == ticknum) {
                continue;
            }
            hist.push_back((ticknum, pos));
            while hist.len() > HISTORY_TICKS {
                hist.pop_front();
            }
        }
        while self.hurts.front().is_some_and(|h| h.tick + 2 * EVAL_DELAY < ticknum) {
            self.hurts.pop_front();
        }
        if self.pending.is_empty() {
            return Ok(vec![]);
        }
        let (ready, rest): (Vec<Shot>, Vec<Shot>) = std::mem::take(&mut self.pending)
            .into_iter()
            .partition(|s| s.tick + EVAL_DELAY <= ticknum);
        self.pending = rest;
        Ok(ready.into_iter().filter_map(|shot| self.evaluate_shot(shot, state)).collect())
    }

    /// Everything that needs all of a player's shots: headshots that only land at a rewind far from the
    /// shooter's typical one, and the kinds that only count for an already flagged player.
    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let check_backtrack: bool = get_parameter_value(&self.params, "check_backtrack");
        let min_error: f32 = get_parameter_value(&self.params, "backtrack_min_head_error");
        let min_offset: f32 = get_parameter_value(&self.params, "backtrack_min_rewind_offset");
        let weak_error: f32 = get_parameter_value(&self.params, "backtrack_weak_min_head_error");
        let weak_offset: f32 = get_parameter_value(&self.params, "backtrack_weak_min_rewind_offset");
        let min_headshots = get_parameter_value::<i32>(&self.params, "backtrack_min_headshots").max(1) as usize;
        let gated_min = get_parameter_value::<i32>(&self.params, "gated_min_detections").max(1) as usize;
        let flag_min = get_parameter_value::<i32>(&self.params, "flag_min_detections").max(1) as usize;
        let flag_window = get_parameter_value::<i32>(&self.params, "flag_window_ticks").max(0) as u32;
        let mut players: Vec<u64> = self.rewinds.iter().map(|r| r.player).collect();
        players.sort_unstable();
        players.dedup();
        let mut backtrack = Vec::new();
        let mut weak_backtrack = Vec::new();
        let round2 = |v: f32| (v * 100.0).round() / 100.0;
        for player in players.iter().copied().filter(|_| check_backtrack) {
            let shots: Vec<&HeadshotRewind> = self.rewinds.iter().filter(|r| r.player == player).collect();
            if shots.len() < min_headshots {
                continue;
            }
            let lag_offset = typical_lag_offset(&shots);
            for shot in &shots {
                if self.reported.contains(&(player, shot.tick)) {
                    continue;
                }
                let Some((rewind, error)) = best_rewind(shot) else { continue };
                if error > HEAD_MATCH {
                    continue;
                }
                let typical = lag_offset + shot.ping_ticks;
                let typical_error = TYPICAL_REWIND_WINDOW.iter().filter_map(|d| error_at(shot, typical + d)).fold(f32::INFINITY, f32::min);
                let offset = (rewind - typical).abs();
                if !typical_error.is_finite() || typical_error < weak_error || offset < weak_offset {
                    continue;
                }
                let strong = typical_error >= min_error && offset >= min_offset;
                let detection = Detection {
                    tick: shot.tick,
                    algorithm: "fidoo/sniperaim".to_string(),
                    player,
                    data: json!({
                        "type": "backtrack_headshot",
                        "weak": !strong,
                        "weapon": shot.weapon,
                        "headshot": true,
                        "target": shot.target,
                        "rewind_ticks": round2(rewind),
                        "typical_rewind_ticks": round2(typical),
                        "head_error": round2(error),
                        "head_error_at_typical_rewind": round2(typical_error),
                        "distance": shot.distance.round(),
                    }),
                };
                if strong {
                    backtrack.push(detection);
                } else {
                    weak_backtrack.push(detection);
                }
            }
        }
        // Plain flicks and off-lag headshots count from `gated_min` per player, and never pair with another
        // kind. Weak flicks only tell a reviewer more about a player flagged by something else.
        let mut flagged: HashSet<u64> = HashSet::new();
        let mut ticks: HashMap<u64, Vec<u32>> = HashMap::new();
        for &(player, tick) in &self.reported {
            ticks.entry(player).or_default().push(tick);
        }
        for (&player, player_ticks) in ticks.iter_mut() {
            player_ticks.sort_unstable();
            if player_ticks.windows(flag_min).any(|w| w[flag_min - 1] - w[0] <= flag_window) {
                flagged.insert(player);
            }
        }
        let plain = std::mem::take(&mut self.plain_flicks);
        for group in [&plain, &backtrack] {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for d in group {
                *counts.entry(d.player).or_default() += 1;
            }
            flagged.extend(counts.into_iter().filter(|&(_, n)| n >= gated_min).map(|(player, _)| player));
        }
        let weak = std::mem::take(&mut self.weak_flicks);
        let mut detections = Vec::new();
        let mut seen = self.reported.clone();
        for d in plain.into_iter().chain(backtrack).chain(weak).chain(weak_backtrack) {
            if flagged.contains(&d.player) && seen.insert((d.player, d.tick)) {
                detections.push(d);
            }
        }
        Ok(detections)
    }
}

pub fn create() -> Box<dyn crate::lib::algorithm::CheatAlgorithm<'static> + Send> {
    Box::new(SniperAim::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(j: i32, yaw: f32) -> AimSample {
        AimSample { j, yaw, pitch: 0.0, distance: 2000.0 }
    }

    #[test]
    fn updates_merge_repeated_aim_on_adjacent_ticks_only() {
        let samples = [sample(-4, 1.0), sample(-3, 1.0), sample(-1, 1.0), sample(0, 0.0)];
        let updates = group_updates(&samples, &samples[3]);
        assert_eq!(updates.len(), 3);
        assert_eq!((updates[0].start, updates[0].end), (-4, -3));
        assert_eq!((updates[1].start, updates[1].end), (-1, -1));
        assert!((updates[1].pos.0 - 1.0).abs() < 1e-6);
        assert_eq!((updates[2].start, updates[2].end), (0, 0));
    }

    #[test]
    fn updates_unwrap_yaw_around_180() {
        let samples = [sample(-1, 179.5), sample(0, -179.5)];
        let updates = group_updates(&samples, &samples[1]);
        assert!((updates[0].pos.0 + 1.0).abs() < 1e-4);
    }

    fn dot_track(aim: &[(i32, f32)], shot_tick: u32) -> (VecDeque<Snap>, VecDeque<(u32, [f32; 3])>) {
        let mut shooter = VecDeque::new();
        let mut dots = VecDeque::new();
        for &(j, yaw) in aim {
            let tick = (shot_tick as i32 + j) as u32;
            shooter.push_back(Snap { tick, pos: [0.0, 0.0, 0.0], yaw, pitch: 0.0, ducking: false });
            let r = yaw.to_radians();
            dots.push_back((tick, [2000.0 * r.cos(), 2000.0 * r.sin(), 75.0]));
        }
        (shooter, dots)
    }

    fn enabled() -> SniperAim {
        let mut algo = SniperAim::new();
        algo.params.insert("enabled".to_string(), Parameter::Bool(true));
        algo
    }

    #[test]
    fn held_snap_on_shot_tick_is_flagged() {
        // Slow drift of 0.02°/tick, then a 1.5° jump on the shot tick that is kept.
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, if j < 0 { 1.5 + j as f32 * 0.02 } else { 0.0 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        let algo = enabled();
        let event = algo.classify(1000, &shooter, &dots, true).expect("snap");
        assert_eq!(event.kind, SnapKind::Held);
        assert!((event.angle - 1.48).abs() < 0.01);
        // Not a headshot: a held snap needs a headshot.
        assert!(algo.classify(1000, &shooter, &dots, false).is_none());
    }

    #[test]
    fn silent_snap_returning_after_shot_is_flagged_without_headshot() {
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, if j == 0 { 0.0 } else { 1.5 + j as f32 * 0.02 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        let event = enabled().classify(1000, &shooter, &dots, false).expect("silent snap");
        assert_eq!(event.kind, SnapKind::Silent);
    }

    #[test]
    fn silent_snap_after_resting_aim_is_flagged() {
        // Scoped in at -7 and held still: no preceding movement to compare with.
        let aim: Vec<(i32, f32)> = (-7..=4).filter(|&j| j != 2).map(|j| (j, if j == 0 || j == 1 { 0.0 } else { 0.9 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        let event = enabled().classify(1000, &shooter, &dots, false).expect("silent snap");
        assert_eq!(event.kind, SnapKind::Silent);
    }

    #[test]
    fn held_snap_from_resting_aim_is_not_flagged() {
        // A flick out of a still aim that stays: human.
        let aim: Vec<(i32, f32)> = (-7..=4).map(|j| (j, if j < 0 { 0.9 } else { 0.0 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, true).is_none());
        // Resting too briefly.
        let aim: Vec<(i32, f32)> = (-2..=4).map(|j| (j, if j == 0 || j == 1 { 0.0 } else { 0.9 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, false).is_none());
    }

    #[test]
    fn silent_return_after_a_tick_without_packet_is_seen() {
        // j=2 has no packet, so the snapped aim is split into two updates before the return at j=4.
        let aim: Vec<(i32, f32)> =
            [(-5, 2.0), (-3, 2.0), (-2, 1.7), (0, 0.0), (1, 0.0), (3, 0.0), (4, 1.8)].into_iter().collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        let event = enabled().classify(1000, &shooter, &dots, false).expect("silent snap");
        assert_eq!(event.kind, SnapKind::Silent);
    }

    fn eye_track(yaw: &[(i32, f32)]) -> VecDeque<Snap> {
        yaw.iter()
            .map(|&(j, yaw)| Snap { tick: (1000 + j) as u32, pos: [0.0; 3], yaw, pitch: 0.35, ducking: false })
            .collect()
    }

    #[test]
    fn eye_spike_out_and_back_is_flagged() {
        let shooter = eye_track(&[(-8, 39.41), (-6, 39.41), (-3, 39.41), (0, 38.01), (1, 39.41), (3, 39.41)]);
        let angle = enabled().eye_spike(1000, &shooter).expect("spike");
        assert!((angle - 1.4).abs() < 0.01);
    }

    #[test]
    fn eye_spike_needs_rest_and_exact_return() {
        // Moves on afterwards.
        let shooter = eye_track(&[(-6, 39.41), (-3, 39.41), (0, 38.01), (1, 38.71)]);
        assert!(enabled().eye_spike(1000, &shooter).is_none());
        // Was turning before the shot.
        let shooter = eye_track(&[(-6, 40.81), (-3, 39.41), (0, 38.01), (1, 39.41)]);
        assert!(enabled().eye_spike(1000, &shooter).is_none());
        // One quantization step: jitter.
        let shooter = eye_track(&[(-6, 39.41), (-3, 39.41), (0, 39.06), (1, 39.41)]);
        assert!(enabled().eye_spike(1000, &shooter).is_none());
    }

    #[test]
    fn fast_continuous_flick_is_not_flagged() {
        // Constant 0.5°/tick sweep: the shot update is no larger than the preceding ones.
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, -0.5 * j as f32)).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, true).is_none());
    }

    #[test]
    fn snap_before_shot_tick_is_not_flagged() {
        // The jump happened two ticks before the shot, the aim then rested.
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, if j < -2 { 1.5 } else { 0.0 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, true).is_none());
    }

    #[test]
    fn silent_outlier_while_turning_is_flagged() {
        // Turning at 0.05°/tick (0.35°/tick before), the shot update jumps 0.6° off the path and the aim
        // continues on it. The jump is less than 3x the preceding updates, so it is no `silent_snap`.
        let aim: Vec<(i32, f32)> =
            (-12..=4).map(|j| (j, if j < -3 { 0.35 * j as f32 + 1.0 } else { 0.05 * j as f32 } + if j == 0 { -0.6 } else { 0.0 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        let event = enabled().classify(1000, &shooter, &dots, false).expect("outlier");
        assert_eq!(event.kind, SnapKind::Outlier);
        assert!((event.path_deviation - 0.6).abs() < 0.01);
        let mut algo = enabled();
        algo.params.insert("check_silent_outlier".to_string(), Parameter::Bool(false));
        assert!(algo.classify(1000, &shooter, &dots, false).is_none());
    }

    #[test]
    fn curved_turn_is_not_an_outlier() {
        // Accelerating turn: the shot update is off the straight path, but the aim does not come back.
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, if j <= 0 { 0.02 * j as f32 } else { 0.5 * j as f32 })).collect();
        let (shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, true).is_none());
    }

    fn diagonal_track(aim: &[(i32, f32, f32)], shot_tick: u32) -> (VecDeque<Snap>, VecDeque<(u32, [f32; 3])>) {
        let mut shooter = VecDeque::new();
        let mut dots = VecDeque::new();
        for &(j, yaw, pitch) in aim {
            let tick = (shot_tick as i32 + j) as u32;
            shooter.push_back(Snap { tick, pos: [0.0, 0.0, 0.0], yaw, pitch, ducking: false });
            let (y, p) = (yaw.to_radians(), pitch.to_radians());
            dots.push_back((tick, [2000.0 * p.cos() * y.cos(), 2000.0 * p.cos() * y.sin(), 75.0 - 2000.0 * p.sin()]));
        }
        (shooter, dots)
    }

    #[test]
    fn post_shot_diagonal_spike_is_flagged() {
        let aim: Vec<(i32, f32, f32)> = (-6..=4).map(|j| if j == 1 { (j, 21.0, 6.0) } else { (j, 20.0, 5.0) }).collect();
        let (shooter, dots) = diagonal_track(&aim, 1000);
        let spike = enabled().post_spike(1000, &shooter, &dots).expect("spike");
        assert_eq!(spike.delay, 1);
        assert!(spike.return_ratio < 0.01);
        // A one-axis jump of the same size is not the artifact.
        let aim: Vec<(i32, f32, f32)> = (-6..=4).map(|j| if j == 1 { (j, 21.4, 5.0) } else { (j, 20.0, 5.0) }).collect();
        let (shooter, dots) = diagonal_track(&aim, 1000);
        assert!(enabled().post_spike(1000, &shooter, &dots).is_none());
        // Moving on instead of coming back.
        let aim: Vec<(i32, f32, f32)> = (-6..=4).map(|j| if j >= 1 { (j, 21.0, 6.0) } else { (j, 20.0, 5.0) }).collect();
        let (shooter, dots) = diagonal_track(&aim, 1000);
        assert!(enabled().post_spike(1000, &shooter, &dots).is_none());
    }

    fn record(sim: i64, pos: [f32; 3], yaw: f32) -> HeadRecord {
        HeadRecord { sim, tick: sim as u32, pos, yaw, pitch: 0.0, ducking: false, class: Class::Soldier }
    }

    #[test]
    fn head_sits_in_front_of_the_origin_and_interpolates_between_records() {
        // Facing +x and standing still: forward 9.5, left -0.7, up 70.4.
        let records: VecDeque<HeadRecord> = (0..5).map(|i| record(100 + i, [0.0, 0.0, 0.0], 0.0)).collect();
        let head = head_at(&records, 102.0).unwrap();
        assert!((head[0] - 9.5).abs() < 0.01 && (head[1] + 0.7).abs() < 0.01 && (head[2] - 70.4).abs() < 0.01);
        // A missing simulation tick is interpolated; a long gap is not.
        let records: VecDeque<HeadRecord> = [record(100, [0.0, 0.0, 0.0], 0.0), record(104, [40.0, 0.0, 0.0], 0.0)].into_iter().collect();
        assert!(head_at(&records, 102.0).is_some());
        let records: VecDeque<HeadRecord> = [record(100, [0.0, 0.0, 0.0], 0.0), record(120, [40.0, 0.0, 0.0], 0.0)].into_iter().collect();
        assert!(head_at(&records, 110.0).is_none());
    }

    fn flick_of(aim: &[(i32, f32)], blood_yaw: Option<f32>) -> Option<Flick> {
        let (shooter, dots) = dot_track(aim, 1000);
        let shot = Shot { tick: 1000, ent: 1, weapon_id: 17, origin: Some([0.0, 0.0, 75.0]) };
        let blood = blood_yaw.map(|y: f32| [2000.0 * y.to_radians().cos(), 2000.0 * y.to_radians().sin(), 75.0]);
        SniperAim::flick(&shot, &shooter, &dots, blood)
    }

    #[test]
    fn small_out_and_back_is_a_silent_flick() {
        // 2.5 head radii (0.43°) out and straight back.
        let aim: Vec<(i32, f32)> = (-6..=4).map(|j| (j, if j == 0 { 0.0 } else { 0.43 })).collect();
        let f = flick_of(&aim, Some(0.0)).expect("flick");
        assert!((f.impulse - 2.5).abs() < 0.1 && f.turn < -0.99);
        assert_eq!(enabled().flick_kind(&f, false), Some("silent_flick"));
    }

    #[test]
    fn sweep_through_the_head_is_no_flick() {
        let aim: Vec<(i32, f32)> = (-6..=4).map(|j| (j, 0.2 * j as f32)).collect();
        let f = flick_of(&aim, Some(0.0)).expect("flick");
        assert!(f.impulse < 0.1 && f.turn > 0.99);
        assert_eq!(enabled().flick_kind(&f, true), None);
    }

    #[test]
    fn jump_from_a_slow_aim_that_stays_is_a_plain_flick_on_a_headshot() {
        let aim: Vec<(i32, f32)> = (-8..=4).map(|j| (j, if j < 0 { 1.0 + j as f32 * 0.01 } else { 0.0 })).collect();
        let f = flick_of(&aim, Some(0.0)).expect("flick");
        assert_eq!(enabled().flick_kind(&f, true), Some("plain_flick"));
        // A body hit also has to come from a resting aim.
        assert_eq!(enabled().flick_kind(&f, false), None);
    }

    #[test]
    fn dot_away_from_the_hit_point_is_stale() {
        let aim: Vec<(i32, f32)> = (-6..=4).map(|j| (j, 1.0)).collect();
        assert!(flick_of(&aim, Some(0.0)).is_none());
    }

    fn gated(kind: &str, player: u64, tick: u32) -> Detection {
        Detection { tick, algorithm: "fidoo/sniperaim".to_string(), player, data: json!({ "type": kind }) }
    }

    #[test]
    fn plain_flicks_count_from_three_and_do_not_pair_with_other_kinds() {
        let mut algo = enabled();
        algo.plain_flicks = vec![gated("plain_flick", 1, 100), gated("plain_flick", 1, 200), gated("plain_flick", 2, 100), gated("plain_flick", 2, 200), gated("plain_flick", 2, 300), gated("plain_flick", 3, 100)];
        // Player 3 has one strong detection and one plain flick: not enough. Player 4 is flagged and so
        // also gets a weak flick.
        algo.reported.extend([(3, 500), (4, 500), (4, 600)]);
        algo.weak_flicks = vec![gated("silent_flick", 1, 900), gated("silent_flick", 3, 900), gated("silent_flick", 4, 900)];
        let out = algo.finish().unwrap();
        let of = |p: u64| out.iter().filter(|d| d.player == p).count();
        assert_eq!((of(1), of(2), of(3), of(4)), (0, 3, 0, 1));
    }

    #[test]
    fn typical_rewind_comes_from_most_headshots() {
        let shot = |k: usize| HeadshotRewind {
            player: 1,
            tick: 1,
            target: None,
            weapon: String::new(),
            ping_ticks: 2.0,
            distance: 1000.0,
            errors: (0..REWIND_STEPS).map(|s| if s == k * 4 { 1.0 } else { 40.0 }).collect(),
        };
        let shots = [shot(12), shot(12), shot(12), shot(25)];
        let refs: Vec<&HeadshotRewind> = shots.iter().collect();
        assert!((typical_lag_offset(&refs) - 10.0).abs() < 0.01);
        assert_eq!(best_rewind(&shots[3]), Some((25.0, 1.0)));
    }

    #[test]
    fn eye_angles_must_confirm_the_dot() {
        let aim: Vec<(i32, f32)> = (-12..=4).map(|j| (j, if j < 0 { 1.5 + j as f32 * 0.02 } else { 0.0 })).collect();
        let (mut shooter, dots) = dot_track(&aim, 1000);
        assert!(enabled().classify(1000, &shooter, &dots, true).is_some());
        for s in shooter.iter_mut() {
            s.yaw = 1.5; // eye angles never moved
        }
        assert!(enabled().classify(1000, &shooter, &dots, true).is_none());
    }
}
