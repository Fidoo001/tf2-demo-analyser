//! Learned plain aimbot detection on projectile classes and pyros. See docs/ml_plain_projectile.md.
//!
//! Plain aimbot on projectile weapons and flamethrowers locks or snaps onto the enemy player's
//! predicted/extrapolated position to lead the projectile, elevated along ballistic arcs (pipes, stickies,
//! arrows, crossbow) and instantly snapping when targets counterstrafe.
//!
//! This algorithm tracks projectile launches (from PacketEntities) and flamethrower damage bursts,
//! computes kinematic and ballistic lead features for each event, aggregates them per player over the demo,
//! and scores players with an embedded LightGBM model (`assets/ml_plain_projectile_model.json`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::message::packetentities::UpdateType;
use tf_demo_parser::demo::message::{GameEventMessage, Message};
use tf_demo_parser::demo::sendprop::{SendProp, SendPropIdentifier};
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{handle_to_entid, resolve_recorder_sid};
use crate::util::nocrex::jankguard::JankGuard;

const TICKS_PER_SECOND: f32 = 66.67;
const WORLD_GRAVITY: f32 = 800.0;
const MAX_FLIGHT_TICKS: f32 = 100.0;
const FL_DUCKING: u32 = 1 << 1;
const SAMPLE_HISTORY: usize = 64;
const POSITION_HISTORY: usize = 64;
const POST_EVENT_SAMPLES: usize = 6;
const VELOCITY_TICKS: i64 = 6;
const BODY_BOTTOM: f32 = 5.0;
const BODY_TOP_STANDING: f32 = 80.0;
const BODY_TOP_DUCKING: f32 = 60.0;
const SPLASH_BOTTOM: f32 = -24.0;
const WEAPON_ID_FLAMETHROWER: u16 = 25;
const FLAME_SPEED: f32 = 1000.0;
const FLAME_START: f32 = 40.0;
const MIN_STEP: f32 = 0.18;
const MIN_DISTANCE: f32 = 60.0;
const MAX_DISTANCE: f32 = 2500.0;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

pub const EVENT_FEATURES: [&str; 24] = [
    "err_lead",
    "err_direct",
    "err_lead_rel",
    "err_lead_hu",
    "lead_adv",
    "err_lead_m1",
    "err_lead_m2",
    "err_lead_m4",
    "err_lead_p1",
    "err_lead_p2",
    "ballistic_pitch_diff",
    "step_event",
    "step_pre_med",
    "stillness",
    "jerk",
    "max_step_pre",
    "post_step",
    "t_on_lead",
    "tgt_accel_lat",
    "counterstrafe_detected",
    "reaction_ticks",
    "trk_lead_resid",
    "dist",
    "tgt_speed",
];

const AGGREGATED: usize = 22;
const IDX_ERR_LEAD_REL: usize = 2;
const IDX_LEAD_ADV: usize = 4;
const IDX_STEP_EVENT: usize = 11;
const IDX_STILLNESS: usize = 13;
const IDX_JERK: usize = 14;
const IDX_CS_DETECTED: usize = 19;
const IDX_REACTION_TICKS: usize = 20;

pub fn unit_feature_names() -> Vec<String> {
    let mut names = Vec::new();
    for suffix in ["med", "p10", "p90"] {
        names.extend(EVENT_FEATURES[..AGGREGATED].iter().map(|f| format!("{f}_{suffix}")));
    }
    names.push("snap_lead_share".to_string());
    names.push("precise_lead_share".to_string());
    names.push("lead_dominance_share".to_string());
    names.push("instant_reaction_share".to_string());
    names.push("corner_oscillation_share".to_string());
    names
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeaponKind {
    Rocket,
    DirectHit,
    LibertyLauncher,
    Pipe,
    LochNLoad,
    LooseCannon,
    Sticky,
    Flare,
    Crossbow,
    Huntsman,
    Syringe,
    Flame,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectilePhysics {
    pub speed: f32,
    pub gravity: f32,
    pub splash: bool,
    pub heals_team: bool,
}

impl WeaponKind {
    pub fn physics(self) -> Option<ProjectilePhysics> {
        match self {
            WeaponKind::Rocket => Some(ProjectilePhysics { speed: 1100.0, gravity: 0.0, splash: true, heals_team: false }),
            WeaponKind::DirectHit => Some(ProjectilePhysics { speed: 1980.0, gravity: 0.0, splash: true, heals_team: false }),
            WeaponKind::LibertyLauncher => Some(ProjectilePhysics { speed: 1540.0, gravity: 0.0, splash: true, heals_team: false }),
            WeaponKind::Pipe => Some(ProjectilePhysics { speed: 1200.0, gravity: WORLD_GRAVITY, splash: true, heals_team: false }),
            WeaponKind::LochNLoad => Some(ProjectilePhysics { speed: 1500.0, gravity: WORLD_GRAVITY, splash: true, heals_team: false }),
            WeaponKind::LooseCannon => Some(ProjectilePhysics { speed: 1440.0, gravity: WORLD_GRAVITY, splash: true, heals_team: false }),
            WeaponKind::Sticky => Some(ProjectilePhysics { speed: 900.0, gravity: WORLD_GRAVITY, splash: true, heals_team: false }),
            WeaponKind::Flare => Some(ProjectilePhysics { speed: 2000.0, gravity: 0.3 * WORLD_GRAVITY, splash: false, heals_team: false }),
            WeaponKind::Crossbow => Some(ProjectilePhysics { speed: 2400.0, gravity: 0.2 * WORLD_GRAVITY, splash: false, heals_team: true }),
            WeaponKind::Huntsman => Some(ProjectilePhysics { speed: 2400.0, gravity: 0.2 * WORLD_GRAVITY, splash: false, heals_team: false }),
            WeaponKind::Syringe => Some(ProjectilePhysics { speed: 1000.0, gravity: 0.3 * WORLD_GRAVITY, splash: false, heals_team: false }),
            WeaponKind::Flame => Some(ProjectilePhysics { speed: FLAME_SPEED, gravity: 0.0, splash: false, heals_team: false }),
            WeaponKind::Other => None,
        }
    }
}

pub fn classify_weapon(name: &str) -> WeaponKind {
    let lw = name.to_lowercase();
    let any = |words: &[&str]| words.iter().any(|w| lw.contains(w));
    if any(&["direct hit"]) {
        WeaponKind::DirectHit
    } else if any(&["liberty launcher"]) {
        WeaponKind::LibertyLauncher
    } else if any(&["rocket launcher", "black box", "air strike", "beggar", "mangler", "original"]) {
        WeaponKind::Rocket
    } else if any(&["loch-n-load"]) {
        WeaponKind::LochNLoad
    } else if any(&["loose cannon"]) {
        WeaponKind::LooseCannon
    } else if any(&["grenade launcher", "iron bomber"]) {
        WeaponKind::Pipe
    } else if any(&["stickybomb", "scottish resistance", "quickiebomb"]) {
        WeaponKind::Sticky
    } else if any(&["flare gun", "detonator", "scorch shot", "manmelter"]) {
        WeaponKind::Flare
    } else if any(&["crusader's crossbow", "crossbow"]) {
        WeaponKind::Crossbow
    } else if any(&["huntsman", "fortified compound", "compound bow"]) {
        WeaponKind::Huntsman
    } else if any(&["syringe gun", "blutsauger", "overdose"]) {
        WeaponKind::Syringe
    } else if any(&["flame", "degreaser", "backburner", "phlog", "rainblower", "nostromo", "dragon's fury"]) {
        WeaponKind::Flame
    } else {
        WeaponKind::Other
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub tick: u32,
    pub sim_tick: i64,
    pub yaw: f32,
    pub pitch: f32,
    pub eye: Vector,
    pub team: u8,
    pub ducking: bool,
    pub weapon_kind: WeaponKind,
}

#[derive(Clone, Copy, Debug)]
pub struct PositionRecord {
    pub tick: u32,
    pub sim_tick: i64,
    pub position: Vector,
    pub team: u8,
    pub ducking: bool,
}

#[derive(Clone, Debug)]
pub struct EventRecord {
    pub tick: u32,
    pub attacker: u64,
    pub target: u64,
    pub weapon: String,
    pub kind: WeaponKind,
    pub features: [f32; EVENT_FEATURES.len()],
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_events: usize,
    pub features: Vec<f32>,
    pub score: f32,
}

struct PendingEvent {
    tick: u32,
    attacker: u64,
    weapon: String,
    kind: WeaponKind,
    target_hint: Option<u64>,
}

#[derive(Default)]
struct PlayerHistory {
    samples: VecDeque<Sample>,
    positions: VecDeque<PositionRecord>,
    last_sim_tick: Option<i64>,
    last_seen_tick: u32,
}

#[derive(Default)]
pub struct MlPlainProjectile {
    params: Parameters,
    jg: JankGuard,
    players: HashMap<u64, PlayerHistory>,
    pending: Vec<PendingEvent>,
    flame_hits: HashMap<u64, u32>,
    recorder: Option<u64>,
    pub events: Vec<EventRecord>,
    pub units: Vec<UnitRecord>,
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
    let cross = (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0);
    let c = (cross.0 * cross.0 + cross.1 * cross.1 + cross.2 * cross.2).sqrt();
    let d = a.0 * b.0 + a.1 * b.1 + a.2 * b.2;
    c.atan2(d).to_degrees()
}

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

fn position_at(positions: &VecDeque<PositionRecord>, tick: i64) -> Option<PositionRecord> {
    let newer_idx = positions.iter().position(|r| r.tick as i64 >= tick)?;
    let newer = positions[newer_idx];
    if newer.tick as i64 == tick {
        return Some(newer);
    }
    if newer_idx == 0 {
        return None;
    }
    let older = positions[newer_idx - 1];
    let span = newer.tick.saturating_sub(older.tick);
    if span == 0 || span > 8 {
        return None;
    }
    let f = (tick - older.tick as i64) as f32 / span as f32;
    Some(PositionRecord {
        tick: tick as u32,
        sim_tick: newer.sim_tick,
        position: Vector {
            x: older.position.x + (newer.position.x - older.position.x) * f,
            y: older.position.y + (newer.position.y - older.position.y) * f,
            z: older.position.z + (newer.position.z - older.position.z) * f,
        },
        team: newer.team,
        ducking: newer.ducking,
    })
}

fn solve_ballistic_arc(eye: Vector, target: Vector, speed: f32, gravity: f32) -> Option<(f32, f32, f32)> {
    let dx = target.x - eye.x;
    let dy = target.y - eye.y;
    let x = dx.hypot(dy);
    if x < 16.0 {
        return None;
    }
    let yaw = dy.atan2(dx).to_degrees();
    let y = target.z - eye.z;
    if gravity <= 1e-4 {
        let dist = (x * x + y * y).sqrt();
        let pitch = (-y).atan2(x).to_degrees();
        let flight_time = (dist / speed * TICKS_PER_SECOND).min(MAX_FLIGHT_TICKS);
        return Some((pitch, yaw, flight_time));
    }
    let v2 = speed * speed;
    let disc = v2 * v2 - gravity * (gravity * x * x + 2.0 * y * v2);
    if disc < 0.0 {
        return None;
    }
    let theta = ((v2 - disc.sqrt()) / (gravity * x)).atan();
    let pitch = -theta.to_degrees();
    let v_h = speed * theta.cos();
    let flight_time = if v_h > 1.0 { (x / v_h * TICKS_PER_SECOND).min(MAX_FLIGHT_TICKS) } else { MAX_FLIGHT_TICKS };
    Some((pitch, yaw, flight_time))
}

fn column_error(eye: Vector, yaw: f32, pitch: f32, target: Vector, bottom: f32, top: f32) -> Option<(f32, f32)> {
    let dir = view_dir(yaw, pitch);
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

fn quantile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = q * (v.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let f = idx - lo as f32;
        v[lo] * (1.0 - f) + v[hi] * f
    }
}

fn median(v: &mut [f32]) -> f32 {
    quantile(v, 0.5)
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

fn prop_int(props: &[SendProp], ids: &[(&str, &str)]) -> Option<i64> {
    ids.iter().find_map(|(table, name)| {
        let id = SendPropIdentifier::new(table, name);
        props.iter().find(|p| p.identifier == id).and_then(|p| i64::try_from(&p.value).ok())
    })
}

fn owner_of(props: &[SendProp], state: &CheatAnalyserState) -> Option<u32> {
    let thrower = prop_int(props, &[("DT_BaseGrenade", "m_hThrower")]).map(|h| u32::from(handle_to_entid(h as u32)));
    let owner = prop_int(props, &[("DT_BaseEntity", "m_hOwnerEntity")]).map(|h| u32::from(handle_to_entid(h as u32)));
    let launcher = prop_int(props, &[("DT_TFProjectile_Pipebomb", "m_hLauncher"), ("DT_TFBaseRocket", "m_hLauncher")]).map(|h| u32::from(handle_to_entid(h as u32)));
    [thrower, owner, launcher].into_iter().flatten().find(|&e| state.players.iter().any(|p| u32::from(p.entity) == e))
}

impl MlPlainProjectile {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_score".to_string(), Parameter::Float(0.65)),
                ("min_events".to_string(), Parameter::Int(5)),
                ("max_detections".to_string(), Parameter::Int(5)),
            ]),
            ..Default::default()
        }
    }

    fn evaluate_event(
        &self,
        event: &PendingEvent,
        shooter_samples: &VecDeque<Sample>,
        target_positions: &VecDeque<PositionRecord>,
    ) -> Option<[f32; EVENT_FEATURES.len()]> {
        let physics = event.kind.physics()?;
        let at_idx = shooter_samples.iter().position(|s| s.tick >= event.tick)?;
        if at_idx < 3 {
            return None;
        }
        let s0 = shooter_samples[at_idx];
        let p0 = position_at(target_positions, s0.tick as i64)?;
        let top = if p0.ducking { BODY_TOP_DUCKING } else { BODY_TOP_STANDING };
        let bottom = if physics.splash { SPLASH_BOTTOM } else { BODY_BOTTOM };
        let target_center = Vector { x: p0.position.x, y: p0.position.y, z: p0.position.z + (top + bottom) * 0.5 };
        let dist = ((target_center.x - s0.eye.x).powi(2) + (target_center.y - s0.eye.y).powi(2) + (target_center.z - s0.eye.z).powi(2)).sqrt();
        if dist < MIN_DISTANCE || dist > MAX_DISTANCE {
            return None;
        }

        // Target velocity estimation
        let p_old = position_at(target_positions, s0.tick as i64 - VELOCITY_TICKS).unwrap_or(p0);
        let dt = (p0.tick.saturating_sub(p_old.tick)).max(1) as f32;
        let vel = Vector {
            x: (p0.position.x - p_old.position.x) / dt * TICKS_PER_SECOND,
            y: (p0.position.y - p_old.position.y) / dt * TICKS_PER_SECOND,
            z: (p0.position.z - p_old.position.z) / dt * TICKS_PER_SECOND,
        };
        let tgt_speed = (vel.x * vel.x + vel.y * vel.y).sqrt();

        // Lead prediction with 2 iterations
        let mut predicted = target_center;
        let mut ballistic_opt = None;
        for _ in 0..2 {
            let flight_ticks;
            if event.kind == WeaponKind::Flame {
                let d = (predicted.x - s0.eye.x).hypot(predicted.y - s0.eye.y);
                flight_ticks = ((d - FLAME_START).max(0.0) / FLAME_SPEED * TICKS_PER_SECOND).min(MAX_FLIGHT_TICKS);
            } else if let Some((p, y, ft)) = solve_ballistic_arc(s0.eye, predicted, physics.speed, physics.gravity) {
                flight_ticks = ft;
                ballistic_opt = Some((p, y));
            } else {
                return None;
            }
            let flight_sec = flight_ticks / TICKS_PER_SECOND;
            predicted = Vector {
                x: target_center.x + vel.x * flight_sec,
                y: target_center.y + vel.y * flight_sec,
                z: target_center.z + (if physics.gravity > 0.0 { 0.0 } else { vel.z * flight_sec }),
            };
        }

        let dir_view = view_dir(s0.yaw, s0.pitch);
        let to_lead = (predicted.x - s0.eye.x, predicted.y - s0.eye.y, predicted.z - s0.eye.z);
        let err_lead = angle_between(dir_view, to_lead);
        let err_direct = column_error(s0.eye, s0.yaw, s0.pitch, p0.position, bottom, top)?.0;
        let target_angular_size = (24.0f32 / dist).atan().to_degrees().max(0.2);
        let err_lead_rel = err_lead / target_angular_size;
        let err_lead_hu = (err_lead.to_radians().sin() * dist).abs();
        let lead_adv = err_direct - err_lead;

        let sample_lead_err = |s: &Sample| -> f32 {
            let to = (predicted.x - s.eye.x, predicted.y - s.eye.y, predicted.z - s.eye.z);
            angle_between(view_dir(s.yaw, s.pitch), to)
        };

        let err_lead_m1 = at_idx.checked_sub(1).map_or(err_lead, |i| sample_lead_err(&shooter_samples[i]));
        let err_lead_m2 = at_idx.checked_sub(2).map_or(err_lead_m1, |i| sample_lead_err(&shooter_samples[i]));
        let err_lead_m4 = at_idx.checked_sub(4).map_or(err_lead_m2, |i| sample_lead_err(&shooter_samples[i]));
        let err_lead_p1 = shooter_samples.get(at_idx + 1).map_or(err_lead, sample_lead_err);
        let err_lead_p2 = shooter_samples.get(at_idx + 2).map_or(err_lead_p1, sample_lead_err);

        let ballistic_pitch_diff = if let Some((bp, _)) = ballistic_opt {
            (s0.pitch - bp).abs()
        } else {
            0.0
        };

        // Kinematics around event
        let s_prev = shooter_samples[at_idx - 1];
        let sim_gap = (s0.sim_tick - s_prev.sim_tick).max(1) as f32;
        let step_event = angle_between(view_dir(s_prev.yaw, s_prev.pitch), view_dir(s0.yaw, s0.pitch)) / sim_gap;

        let mut pre_steps = Vec::new();
        let pre_start = at_idx.saturating_sub(10);
        for w in shooter_samples.range(pre_start..at_idx).collect::<Vec<_>>().windows(2) {
            let gap = (w[1].sim_tick - w[0].sim_tick).max(1) as f32;
            pre_steps.push(angle_between(view_dir(w[0].yaw, w[0].pitch), view_dir(w[1].yaw, w[1].pitch)) / gap);
        }
        let step_pre_med = if pre_steps.is_empty() { step_event } else { median(&mut pre_steps) };
        let stillness = step_event / step_pre_med.max(MIN_STEP);
        let max_step_pre = pre_steps.iter().copied().fold(0.0f32, f32::max);

        let mut post_steps = Vec::new();
        let post_end = (at_idx + 3).min(shooter_samples.len());
        for w in shooter_samples.range(at_idx..post_end).collect::<Vec<_>>().windows(2) {
            let gap = (w[1].sim_tick - w[0].sim_tick).max(1) as f32;
            post_steps.push(angle_between(view_dir(w[0].yaw, w[0].pitch), view_dir(w[1].yaw, w[1].pitch)) / gap);
        }
        let post_step = post_steps.iter().copied().fold(0.0f32, f32::max);

        // Jerk: max 2nd difference of view direction
        let mut jerk = 0.0f32;
        let jerk_start = at_idx.saturating_sub(6);
        let jerk_end = (at_idx + 3).min(shooter_samples.len());
        let j_samples: Vec<_> = shooter_samples.range(jerk_start..jerk_end).copied().collect();
        if j_samples.len() >= 3 {
            for w in j_samples.windows(3) {
                let d1 = angle_between(view_dir(w[0].yaw, w[0].pitch), view_dir(w[1].yaw, w[1].pitch));
                let d2 = angle_between(view_dir(w[1].yaw, w[1].pitch), view_dir(w[2].yaw, w[2].pitch));
                jerk = jerk.max((d2 - d1).abs());
            }
        }

        // Ticks view stayed locked on lead
        let mut t_on_lead = 0.0f32;
        for i in (0..=at_idx).rev() {
            let s = &shooter_samples[i];
            if sample_lead_err(s) <= target_angular_size * 1.5 {
                t_on_lead += (s0.tick.saturating_sub(s.tick)).min(4) as f32;
            } else {
                break;
            }
            if at_idx - i > 12 {
                break;
            }
        }

        // Counterstrafe & reaction detection
        let mut counterstrafe_detected = 0.0f32;
        let mut reaction_ticks = f32::NAN;
        let mut tgt_accel_lat = 0.0f32;
        let cs_window_start = s0.tick as i64 - 16;
        if let (Some(p_cs_old), Some(p_cs_mid)) = (position_at(target_positions, cs_window_start), position_at(target_positions, s0.tick as i64 - 8)) {
            let v_old = Vector {
                x: p_cs_mid.position.x - p_cs_old.position.x,
                y: p_cs_mid.position.y - p_cs_old.position.y,
                z: 0.0,
            };
            let v_curr = Vector {
                x: p0.position.x - p_cs_mid.position.x,
                y: p0.position.y - p_cs_mid.position.y,
                z: 0.0,
            };
            let l_old = (v_old.x * v_old.x + v_old.y * v_old.y).sqrt();
            let l_curr = (v_curr.x * v_curr.x + v_curr.y * v_curr.y).sqrt();
            if l_old > 40.0 && l_curr > 40.0 {
                let dot = (v_old.x * v_curr.x + v_old.y * v_curr.y) / (l_old * l_curr);
                if dot < -0.2 {
                    counterstrafe_detected = 1.0;
                    tgt_accel_lat = (l_curr + l_old) / 8.0;
                    // Find how many ticks before event the shooter view stepped sharply towards new lead
                    let mut r_ticks = 16.0f32;
                    for i in (at_idx.saturating_sub(8)..=at_idx).rev() {
                        let s = &shooter_samples[i];
                        if sample_lead_err(s) < target_angular_size * 2.0 {
                            r_ticks = (s0.tick.saturating_sub(s.tick)) as f32;
                        }
                    }
                    reaction_ticks = r_ticks;
                }
            }
        }

        // Tracking residual vs predicted motion
        let trk_lead_resid = (err_lead - err_lead_m2).abs();

        Some([
            err_lead,
            err_direct,
            err_lead_rel,
            err_lead_hu,
            lead_adv,
            err_lead_m1,
            err_lead_m2,
            err_lead_m4,
            err_lead_p1,
            err_lead_p2,
            ballistic_pitch_diff,
            step_event,
            step_pre_med,
            stillness,
            jerk,
            max_step_pre,
            post_step,
            t_on_lead,
            tgt_accel_lat,
            counterstrafe_detected,
            reaction_ticks,
            trk_lead_resid,
            dist,
            tgt_speed,
        ])
    }

    fn flush_pending(&mut self, state: &CheatAnalyserState) {
        let current_tick = u32::from(state.tick);
        let mut remaining = Vec::new();

        for pending in std::mem::take(&mut self.pending) {
            let Some(pdata) = self.players.get(&pending.attacker) else { continue };
            let has_post = pdata.samples.back().is_some_and(|s| s.tick >= pending.tick + POST_EVENT_SAMPLES as u32);
            if !has_post && current_tick.saturating_sub(pending.tick) < 30 {
                remaining.push(pending);
                continue;
            }

            let s0 = match pdata.samples.iter().find(|s| s.tick >= pending.tick) {
                Some(s) => *s,
                None => continue,
            };

            // Find best target. Exact angle ties are common (players stay in `players` at their last
            // position, and two players can share one), so they are broken by the fresher position
            // record, then the lower SteamID, not by the random hash map order.
            let mut best_target = None;
            let mut best_angle = 45.0f32;
            let mut best_tie_key = (0u32, 0u64);

            for (&target_sid, target_hist) in &self.players {
                if target_sid == pending.attacker {
                    continue;
                }
                if let Some(hint) = pending.target_hint {
                    if target_sid == hint {
                        best_target = Some(target_sid);
                        break;
                    }
                }
                let Some(p) = target_hist.positions.back() else { continue };
                if p.team == s0.team && pending.kind != WeaponKind::Crossbow {
                    continue;
                }
                let dx = p.position.x - s0.eye.x;
                let dy = p.position.y - s0.eye.y;
                let dz = p.position.z - s0.eye.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist > MAX_DISTANCE || dist < MIN_DISTANCE {
                    continue;
                }
                let ang = angle_between(view_dir(s0.yaw, s0.pitch), (dx, dy, dz));
                let tie_key = (p.tick, target_sid);
                let wins_tie = best_target.is_some()
                    && ang == best_angle
                    && (tie_key.0 > best_tie_key.0 || (tie_key.0 == best_tie_key.0 && tie_key.1 < best_tie_key.1));
                if ang < best_angle || wins_tie {
                    best_angle = ang;
                    best_target = Some(target_sid);
                    best_tie_key = tie_key;
                }
            }

            let Some(target_sid) = best_target else { continue };
            let Some(target_hist) = self.players.get(&target_sid) else { continue };

            if let Some(features) = self.evaluate_event(&pending, &pdata.samples, &target_hist.positions) {
                self.events.push(EventRecord {
                    tick: pending.tick,
                    attacker: pending.attacker,
                    target: target_sid,
                    weapon: pending.weapon,
                    kind: pending.kind,
                    features,
                    score: f32::NAN,
                });
            }
        }

        self.pending = remaining;
    }
}

impl<'a> CheatAlgorithm<'a> for MlPlainProjectile {
    fn algorithm_name(&self) -> &str {
        "fidoo/ml_plain_projectile"
    }

    // Never hidden by the global sliding window: aggregates its evidence per player over the whole demo
    fn excluded_from_sliding_window(&self) -> bool {
        true
    }

    fn tooltip(&self) -> &'static str {
        "Learned detection of plain/smoothed aimbot on projectile classes and pyros (target lead extrapolation and counterstrafe snaps)."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![
            MessageType::PacketEntities,
            MessageType::GameEvent,
            MessageType::TempEntities,
        ])
    }

    fn init(&mut self) -> Result<(), Error> {
        let params = std::mem::take(&mut self.params);
        *self = Self::new();
        self.params = params;
        Ok(())
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);
        let tick_u32 = u32::from(tick);

        if self.recorder.is_none() {
            self.recorder = resolve_recorder_sid(state);
        }

        match message {
            Message::PacketEntities(packet) => {
                for entity in &packet.entities {
                    if entity.update_type != UpdateType::Enter {
                        continue;
                    }
                    let Some(class) = parser_state.server_classes.get(usize::from(entity.server_class)).map(|c| c.name.as_str()) else {
                        continue;
                    };
                    let kind = match class {
                        "CTFBaseRocket" | "CTFProjectile_Rocket" | "CTFProjectile_EnergyBall" => WeaponKind::Rocket,
                        "CTFGrenadePipebombProjectile" | "CTFWeaponBaseGrenadeProj" | "CTFProjectile_Pipebomb" => {
                            match prop_int(&entity.props, &[("DT_TFProjectile_Pipebomb", "m_iType")]) {
                                Some(1) | Some(2) => WeaponKind::Sticky,
                                _ => WeaponKind::Pipe,
                            }
                        }
                        "CTFProjectile_Flare" => WeaponKind::Flare,
                        "CTFProjectile_Arrow" => WeaponKind::Huntsman,
                        "CTFProjectile_HealingBolt" => WeaponKind::Crossbow,
                        "CTFProjectile_Syringe" => WeaponKind::Syringe,
                        _ => continue,
                    };

                    let Some(owner_ent) = owner_of(&entity.props, state) else {
                        continue;
                    };

                    let Some(player) = state.players.iter().find(|p| u32::from(p.entity) == owner_ent) else {
                        continue;
                    };
                    let Some(sid) = player_sid(player, state) else {
                        continue;
                    };
                    if self.recorder == Some(sid) || !player.in_pvs || player.state != PlayerState::Alive {
                        continue;
                    }

                    let weapon = state.get_player_weapon(player);
                    let actual_kind = match classify_weapon(&weapon) {
                        WeaponKind::Other => kind,
                        k => k,
                    };

                    self.pending.push(PendingEvent {
                        tick: tick_u32,
                        attacker: sid,
                        weapon,
                        kind: actual_kind,
                        target_hint: None,
                    });
                }
            }
            Message::GameEvent(GameEventMessage { event: GameEvent::PlayerHurt(hurt), .. }) => {
                if hurt.weapon_id == WEAPON_ID_FLAMETHROWER && hurt.attacker != hurt.user_id {
                    if let (Some(attacker), Some(victim)) = (
                        state.get_id64_from_userid(hurt.attacker.into()),
                        state.get_id64_from_userid(hurt.user_id.into()),
                    ) {
                        if self.recorder != Some(attacker) {
                            let last = self.flame_hits.entry(attacker).or_insert(0);
                            if tick_u32.saturating_sub(*last) >= 8 {
                                *last = tick_u32;
                                let weapon = state.players.iter().find(|p| player_sid(p, state) == Some(attacker)).map_or_else(|| "Flame Thrower".to_string(), |p| state.get_player_weapon(p));
                                self.pending.push(PendingEvent {
                                    tick: tick_u32,
                                    attacker,
                                    weapon,
                                    kind: WeaponKind::Flame,
                                    target_hint: Some(victim),
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        let tick = u32::from(state.tick);

        for player in &state.players {
            let Some(sid) = player_sid(player, state) else {
                continue;
            };
            if !player.in_pvs {
                continue;
            }

            let pdata = self.players.entry(sid).or_default();
            pdata.last_seen_tick = tick;

            pdata.positions.push_back(PositionRecord {
                tick,
                sim_tick: state.sim_tick(player),
                position: player.position,
                team: player.team as u8,
                ducking: player.flags & FL_DUCKING != 0,
            });
            while pdata.positions.len() > POSITION_HISTORY {
                pdata.positions.pop_front();
            }

            let sim_tick = state.sim_tick(player);
            if pdata.last_sim_tick == Some(sim_tick) {
                continue;
            }
            pdata.last_sim_tick = Some(sim_tick);

            let weapon = state.get_player_weapon(player);
            let weapon_kind = classify_weapon(&weapon);

            pdata.samples.push_back(Sample {
                tick,
                sim_tick,
                yaw: player.view_angle,
                pitch: player.pitch_angle,
                eye: eye_position(player),
                team: player.team as u8,
                ducking: player.flags & FL_DUCKING != 0,
                weapon_kind,
            });
            while pdata.samples.len() > SAMPLE_HISTORY {
                pdata.samples.pop_front();
            }
        }

        self.flush_pending(state);
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let enabled: bool = get_parameter_value(&self.params, "enabled");
        let min_score: f32 = get_parameter_value(&self.params, "min_score");
        let min_events: i32 = get_parameter_value(&self.params, "min_events");
        let max_detections: i32 = get_parameter_value(&self.params, "max_detections");

        // Group events by attacker (ordered, so `units` come out in the same order on every run)
        let mut by_player: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (idx, ev) in self.events.iter().enumerate() {
            by_player.entry(ev.attacker).or_default().push(idx);
        }

        let model = MODEL.get_or_init(load_model);

        for (&player, event_indices) in &by_player {
            if event_indices.len() < min_events as usize {
                continue;
            }

            let mut cols: [Vec<f32>; AGGREGATED] = Default::default();
            let mut snap_count = 0;
            let mut precise_count = 0;
            let mut lead_dom_count = 0;
            let mut instant_reaction_count = 0;
            let mut cs_count = 0;
            let mut corner_osc_count = 0;

            for &idx in event_indices {
                let ev = &self.events[idx];
                for i in 0..AGGREGATED {
                    let v = ev.features[i];
                    if !v.is_nan() {
                        cols[i].push(v);
                    }
                }

                if ev.features[IDX_STILLNESS] >= 4.0 && ev.features[IDX_STEP_EVENT] >= 2.0 && ev.features[IDX_ERR_LEAD_REL] <= 1.0 {
                    snap_count += 1;
                }
                if ev.features[IDX_ERR_LEAD_REL] <= 0.4 {
                    precise_count += 1;
                }
                if ev.features[IDX_LEAD_ADV] >= 1.0 {
                    lead_dom_count += 1;
                }
                if ev.features[IDX_CS_DETECTED] > 0.5 {
                    cs_count += 1;
                    if ev.features[IDX_REACTION_TICKS] <= 3.0 {
                        instant_reaction_count += 1;
                    }
                }
                if ev.features[IDX_JERK] >= 8.0 && ev.features[0] <= 3.0 {
                    corner_osc_count += 1;
                }
            }

            let n = event_indices.len() as f32;
            let mut u_features = Vec::with_capacity(71);

            for col in &mut cols {
                u_features.push(median(col));
            }
            for col in &mut cols {
                u_features.push(quantile(col, 0.10));
            }
            for col in &mut cols {
                u_features.push(quantile(col, 0.90));
            }

            u_features.push(snap_count as f32 / n);
            u_features.push(precise_count as f32 / n);
            u_features.push(lead_dom_count as f32 / n);
            u_features.push(if cs_count > 0 { instant_reaction_count as f32 / cs_count as f32 } else { 0.0 });
            u_features.push(corner_osc_count as f32 / n);

            let score = model.as_ref().map_or(f32::NAN, |m| m.predict_unit(&u_features));

            self.units.push(UnitRecord {
                player,
                n_events: event_indices.len(),
                features: u_features,
                score,
            });
        }

        if !enabled {
            return Ok(vec![]);
        }

        let mut detections = Vec::new();
        for u in &self.units {
            if u.score >= min_score {
                // Find top scoring or most suspicious events
                let mut p_events: Vec<&EventRecord> = self.events.iter().filter(|e| e.attacker == u.player).collect();
                p_events.sort_by(|a, b| {
                    let lead_a = a.features[IDX_LEAD_ADV];
                    let lead_b = b.features[IDX_LEAD_ADV];
                    lead_b.partial_cmp(&lead_a).unwrap_or(std::cmp::Ordering::Equal)
                });

                for ev in p_events.into_iter().take(max_detections as usize) {
                    detections.push(Detection {
                        tick: ev.tick.into(),
                        algorithm: self.algorithm_name().to_string(),
                        player: u.player,
                        data: json!({
                            "score": u.score,
                            "events": u.n_events,
                            "weapon": ev.weapon,
                            "target": ev.target,
                            "err_lead_deg": ev.features[0],
                            "err_direct_deg": ev.features[1],
                            "lead_adv_deg": ev.features[4],
                            "stillness": ev.features[13],
                            "jerk": ev.features[14],
                            "reaction_ticks": ev.features[20],
                            "dist": ev.features[22],
                        }),
                    });
                }
            }
        }

        Ok(detections)
    }
}

// Model evaluator
#[derive(Deserialize)]
struct FlatTree {
    feature: Vec<usize>,
    threshold: Vec<f32>,
    left: Vec<i32>,
    right: Vec<i32>,
    default_left: Vec<bool>,
    missing: Vec<u8>,
    leaf: Vec<f32>,
}

impl FlatTree {
    fn eval(&self, x: &[f32]) -> f32 {
        if self.feature.is_empty() {
            return 0.0;
        }
        let mut idx = 0i32;
        while idx >= 0 {
            let i = idx as usize;
            let val = x[self.feature[i]];
            let is_missing = val.is_nan();
            let go_left = if is_missing {
                match self.missing[i] {
                    1 => 0.0 <= self.threshold[i],
                    2 => self.default_left[i],
                    _ => false,
                }
            } else {
                val <= self.threshold[i]
            };
            idx = if go_left { self.left[i] } else { self.right[i] };
        }
        self.leaf[(!idx) as usize]
    }
}

#[derive(Deserialize)]
struct FlatBooster {
    features: Vec<String>,
    trees: Vec<FlatTree>,
}

impl FlatBooster {
    fn predict(&self, x: &[f32]) -> f32 {
        let sum: f32 = self.trees.iter().map(|t| t.eval(x)).sum();
        1.0 / (1.0 + (-sum).exp())
    }
}

#[derive(Deserialize)]
struct ModelJson {
    unit: Option<FlatBooster>,
}

struct Model {
    unit: FlatBooster,
}

impl Model {
    fn predict_unit(&self, x: &[f32]) -> f32 {
        self.unit.predict(x)
    }
}

fn load_model() -> Option<Model> {
    let raw = include_str!("../../../assets/ml_plain_projectile_model.json");
    let parsed: ModelJson = serde_json::from_str(raw).ok()?;
    let unit = parsed.unit?;
    if unit.features != unit_feature_names() {
        return None;
    }
    Some(Model { unit })
}
