//! Learned TF2 autostrafe detection. See docs/ml_autostrafe.md.
//!
//! Evaluates airborne flights (between grounded snapshots) using simulation-tick timing.
//! Captures directional autostrafe (circle strafing with a still view, zigzag kicks) through the speed gain and rotation
//! a perpendicular wish direction produces, and through how well the movement keys explain the velocity.
//! Per-player flight metrics are aggregated across the demo and scored by an embedded LightGBM model
//! (`assets/ml_autostrafe_model.json`). Detections report the most suspicious flights for human review.
//!
//! `examples/dump_ml_autostrafe_features.rs` dumps the features for model training and extraction.

use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;
use std::sync::OnceLock;

use anyhow::Error;
use serde::Deserialize;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState, Team};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;
use crate::util::nocrex::jankguard::JankGuard;

const ALGORITHM_NAME: &str = "fidoo/ml_autostrafe";
const MIN_AIRBORNE_TICKS: usize = 10;
const TICKRATE: f32 = 66.0;
const MIN_SPEED: f32 = 250.0;
const OPTIMAL_STRAFE_ANGLE: f32 = 90.0;
const STRAFE_ANGLE_TOLERANCE: f32 = 20.0;
const MAX_AIR_ACCEL_PER_TICK: f32 = 100.0;
const MIN_ROTATION_SPEED_KEPT: f32 = 0.9;
const FROZEN_VIEW_DELTA: f32 = 0.5;
const MIN_SPEED_GAIN_SAMPLES: usize = 12;
const EXTERNAL_MOVEMENT_CONDS: [u32; 5] = [80, 122, 82, 98, 115];
const ZIGZAG_MIN_KICK: f32 = 15.0;
const ZIGZAG_MAX_KICK: f32 = 45.0;
const ZIGZAG_MIN_SPEED: f32 = 150.0;
/// AirAccelerate: the velocity's projection on the wish direction is raised to this cap.
const AIR_SPEED_CAP: f32 = 30.0;
const GAIN_MIN_SPEED: f32 = 200.0;
const GAIN_MIN_SAMPLES: usize = 6;
const GAIN_MIN_TICKS: f32 = 6.0;
/// Stretches at least this long are compared with the perpendicular wish direction's gain.
const GAIN_THEORY_TICKS: f32 = 8.0;
const KEY_NOISE: f32 = 6.0;
const KEY_ERROR_CAP: f32 = 40.0;
const KEY_BAD_ERROR: f32 = 15.0;
const KEY_SWITCH_COST: f32 = 6.0;
const KEY_BLEND: f32 = 0.3;
const KEY_MIN_SEGMENT: usize = 6;
/// A key change this many ticks after the previous one is too fast for a finger.
const KEY_FAST_DWELL: u32 = 2;

static MODEL: OnceLock<Option<Model>> = OnceLock::new();

#[derive(Deserialize, Clone)]
struct Tree {
    feature: Vec<usize>,
    threshold: Vec<f32>,
    left: Vec<i32>,
    right: Vec<i32>,
    default_left: Vec<bool>,
    missing: Vec<u8>,
    leaf: Vec<f64>,
}

impl Tree {
    fn predict(&self, x: &[f32]) -> f64 {
        if self.feature.is_empty() {
            return self.leaf.first().copied().unwrap_or(0.0);
        }
        let mut node: i32 = 0;
        while node >= 0 {
            let n = node as usize;
            let mut v = x[self.feature[n]];
            let m = self.missing[n];
            let is_nan = v.is_nan();
            let go_default = match m {
                2 => is_nan,
                1 => is_nan || v.abs() <= 1e-35,
                _ => false,
            };
            if m != 2 && is_nan {
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
    flight: Booster,
}

fn model() -> Option<&'static Model> {
    MODEL
        .get_or_init(|| {
            let model: Model = serde_json::from_str(include_str!("../../../assets/ml_autostrafe_model.json")).ok()?;
            let flight_names: Vec<String> = FLIGHT_FEATURES.iter().map(|s| s.to_string()).collect();
            (model.unit.features == unit_feature_names() && model.flight.features == flight_names).then_some(model)
        })
        .as_ref()
}

pub const FLIGHT_FEATURES: [&str; 37] = [
    "airborne_ticks",
    "mean_speed",
    "max_speed",
    "speed_gain",
    "static_view_rotation",
    "frozen_view_speed_gain",
    "zigzag_flips",
    "zigzag_samples",
    "zigzag_rate",
    "max_consecutive_flips",
    "optimal_ratio",
    "perfect_ratio",
    "desync_ratio",
    "sync_ratio",
    "view_static_ratio",
    "speed_maintained_ratio",
    "max_turn_streak",
    "accel_angle_consistency",
    "mean_turn_rate",
    "ideal_turn_diff",
    "blast_jumping",
    "gain_theory_ticks",
    "gain_match_ticks",
    "gain_over_ticks",
    "gain_ticks",
    "gain_sum",
    "rot_match_ticks",
    "any_flips",
    "any_samples",
    "key_ticks",
    "key_error_sum",
    "key_bad_ticks",
    "key_switches",
    "key_fast_switches",
    "key_stuck_ticks",
    "single_steps",
    "steps",
];

pub fn flight_feature_names() -> Vec<String> {
    FLIGHT_FEATURES.iter().map(|s| s.to_string()).collect()
}

pub const UNIT_FEATURES: [&str; 56] = [
    "static_rot_med", "static_rot_p10", "static_rot_p90", "static_rot_max",
    "frozen_gain_med", "frozen_gain_p10", "frozen_gain_p90", "frozen_gain_max",
    "zigzag_rate_med", "zigzag_rate_p90",
    "optimal_ratio_med", "optimal_ratio_p90",
    "perfect_ratio_med", "perfect_ratio_p90",
    "desync_ratio_med", "desync_ratio_p90",
    "sync_ratio_med", "sync_ratio_p90",
    "view_static_med", "view_static_p90",
    "turn_diff_med", "turn_diff_p10",
    "speed_maint_med", "speed_maint_p90",
    "mean_speed_med", "max_speed_p90",
    "total_zigzag_flips", "total_zigzag_samples", "overall_zigzag_rate",
    "max_flight_flips", "max_consecutive_flips",
    "circle_flight_share", "zigzag_flight_share", "speed_gain_flight_share",
    "high_desync_share", "high_sync_share",
    "zigzag_binom_score",
    "n_flights",
    "gain_rate", "gain_match_share", "gain_match_shrunk", "gain_match_ticks", "gain_over_share", "gain_theory_ticks",
    "rot_match_share", "rot_match_shrunk",
    "any_flip_rate", "any_binom", "moving_flip_rate",
    "key_rms", "key_bad_share", "key_switch_rate", "key_fast_rate", "key_fast_binom", "key_ticks",
    "single_step_share",
];

pub fn unit_feature_names() -> Vec<String> {
    UNIT_FEATURES.iter().map(|s| s.to_string()).collect()
}

#[derive(Clone, Debug)]
pub struct FlightRecord {
    pub start_tick: u32,
    pub end_tick: u32,
    pub player: u64,
    pub features: [f32; FLIGHT_FEATURES.len()],
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct UnitRecord {
    pub player: u64,
    pub n_flights: u32,
    pub total_flips: u32,
    pub total_samples: u32,
    pub score: f32,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug)]
struct AirborneFrame {
    tick: DemoTick,
    sim_tick: i64,
    dt: f32,
    position: Vector,
    velocity: Vector,
    speed: f32,
    view_yaw: f32,
    velocity_yaw: f32,
    acceleration_angle: Option<f32>,
    blast_jumping: bool,
    external_movement: bool,
}

impl AirborneFrame {
    fn velocity_dt(&self, prev: &AirborneFrame) -> f32 {
        (self.dt + prev.dt) * 0.5
    }
}

#[derive(Clone, Copy, Default)]
struct GainStats {
    ticks: f32,
    sum: f32,
    theory_ticks: f32,
    match_ticks: f32,
    over_ticks: f32,
    rot_match_ticks: f32,
}

#[derive(Clone, Copy, Default)]
struct KeyFit {
    ticks: u32,
    error: f32,
    bad: u32,
    switches: u32,
    fast: u32,
    stuck: u32,
}

#[derive(Default)]
struct PlayerTracker {
    airborne_frames: Vec<AirborneFrame>,
    last_position: Option<Vector>,
    last_velocity: Option<Vector>,
    last_sim_tick: Option<i64>,
    usable_flights: Vec<FlightRecord>,
    total_flips: u32,
    total_samples: u32,
}

pub struct MlAutoStrafe {
    params: Parameters,
    player_trackers: FxHashMap<u64, PlayerTracker>,
    jankguard: JankGuard,
    host_entity: Option<u32>,
    is_stv: bool,
    pub flights: Vec<FlightRecord>,
    pub units: Vec<UnitRecord>,
    /// Training only: keep every n-th sim tick of a player, which is what a low cl_cmdrate looks like.
    pub sample_every: i64,
}

impl MlAutoStrafe {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_score".to_string(), Parameter::Float(0.8)),
                ("min_flights".to_string(), Parameter::Int(3)),
                ("max_detections".to_string(), Parameter::Int(10)),
            ]),
            player_trackers: FxHashMap::default(),
            jankguard: JankGuard::default(),
            host_entity: None,
            is_stv: false,
            flights: Vec::new(),
            units: Vec::new(),
            sample_every: 1,
        }
    }

    fn get_speed_2d(velocity: &Vector) -> f32 {
        (velocity.x * velocity.x + velocity.y * velocity.y).sqrt()
    }

    fn get_angle_2d(vec: &Vector) -> f32 {
        vec.y.atan2(vec.x) * 180.0 / PI
    }

    fn normalize_angle(angle: f32) -> f32 {
        let mut a = angle;
        while a > 180.0 {
            a -= 360.0;
        }
        while a < -180.0 {
            a += 360.0;
        }
        a
    }

    fn in_cond(player: &Player, cond: u32) -> bool {
        let bits = match cond / 32 {
            0 => player.cond,
            1 => player.cond_ex,
            2 => player.cond_ex2,
            3 => player.cond_ex3,
            _ => return false,
        };
        bits & (1 << (cond % 32)) != 0
    }

    fn is_grounded(player: &Player) -> bool {
        (player.flags & 1) == 1
    }

    fn calculate_velocity(prev_pos: &Vector, curr_pos: &Vector, tick_delta: f32) -> Vector {
        let time_delta = tick_delta / TICKRATE;
        if time_delta <= 0.0 {
            return Vector { x: 0.0, y: 0.0, z: 0.0 };
        }
        Vector {
            x: (curr_pos.x - prev_pos.x) / time_delta,
            y: (curr_pos.y - prev_pos.y) / time_delta,
            z: (curr_pos.z - prev_pos.z) / time_delta,
        }
    }

    fn calculate_acceleration_angle(prev_vel: &Vector, curr_vel: &Vector) -> Option<f32> {
        let accel = Vector {
            x: curr_vel.x - prev_vel.x,
            y: curr_vel.y - prev_vel.y,
            z: 0.0,
        };
        let accel_magnitude = (accel.x * accel.x + accel.y * accel.y).sqrt();
        if accel_magnitude < 1.0 {
            return None;
        }
        let vel_angle = Self::get_angle_2d(curr_vel);
        let accel_angle = Self::get_angle_2d(&accel);
        Some(Self::normalize_angle(accel_angle - vel_angle).abs())
    }

    fn is_standing_on_lift_or_platform(frames: &[AirborneFrame]) -> bool {
        if frames.len() < 25 {
            return false;
        }
        let first = &frames[0];
        let last = frames.last().unwrap();
        let dx = last.position.x - first.position.x;
        let dy = last.position.y - first.position.y;
        let net_horiz_dist = (dx * dx + dy * dy).sqrt();

        if frames.len() >= 40 && net_horiz_dist < 120.0 {
            return true;
        }

        let max_radius = frames
            .iter()
            .map(|f| {
                let x = f.position.x - first.position.x;
                let y = f.position.y - first.position.y;
                (x * x + y * y).sqrt()
            })
            .fold(0.0f32, f32::max);

        if frames.len() >= 50 && max_radius < 100.0 {
            return true;
        }
        false
    }

    fn is_ramp_or_stair_motion(frames: &[AirborneFrame], is_blast_jumping: bool) -> bool {
        if frames.len() < 15 {
            return false;
        }
        let first = &frames[0];
        let last = frames.last().unwrap();
        let total_dz = last.position.z - first.position.z;

        if !is_blast_jumping && frames.len() >= 35 && total_dz > 15.0 {
            return true;
        }

        let mut total_dv_z = 0.0;
        let mut total_dt = 0.0;
        let mut sample_count = 0;
        let mut supported_frames = 0;

        for i in 1..frames.len() {
            let dt = frames[i].velocity_dt(&frames[i - 1]);
            let dv_z = frames[i].velocity.z - frames[i - 1].velocity.z;
            total_dv_z += dv_z;
            total_dt += dt;
            sample_count += 1;

            if dv_z / dt >= -2.0 {
                supported_frames += 1;
            }
        }

        let supported_ratio = supported_frames as f32 / sample_count.max(1) as f32;

        if sample_count >= 15 && is_blast_jumping && supported_ratio > 0.65 {
            return true;
        }

        if sample_count >= 15 && !is_blast_jumping {
            let avg_accel_z = (total_dv_z / total_dt) * TICKRATE;
            if avg_accel_z > -250.0 || supported_ratio > 0.65 {
                return true;
            }

            let dx = last.position.x - first.position.x;
            let dy = last.position.y - first.position.y;
            let horiz_dist = (dx * dx + dy * dy).sqrt();
            let vert_drop = total_dz.abs();

            if horiz_dist > 180.0 && vert_drop > 8.0 {
                let slope_ratio = horiz_dist / vert_drop.max(1.0);
                if slope_ratio > 2.5 && supported_ratio > 0.55 {
                    return true;
                }
            }
        }
        false
    }

    fn is_free_air_step(prev: &AirborneFrame, curr: &AirborneFrame) -> bool {
        let dv_x = curr.velocity.x - prev.velocity.x;
        let dv_y = curr.velocity.y - prev.velocity.y;
        let accel_per_tick = (dv_x * dv_x + dv_y * dv_y).sqrt() / curr.velocity_dt(prev);
        curr.speed >= MIN_SPEED && prev.speed >= MIN_SPEED && accel_per_tick <= MAX_AIR_ACCEL_PER_TICK
    }

    fn max_static_view_rotation(frames: &[AirborneFrame]) -> f32 {
        fn stretch_rotation(stretch: &[(f32, f32)]) -> f32 {
            let mut best = 0.0f32;
            let speeds: Vec<f32> = (0..stretch.len())
                .map(|i| {
                    let window = &stretch[i.saturating_sub(1)..(i + 2).min(stretch.len())];
                    window.iter().map(|s| s.1).sum::<f32>() / window.len() as f32
                })
                .collect();
            for i in 0..stretch.len() {
                for j in i + 1..stretch.len() {
                    if speeds[j] >= speeds[i] * MIN_ROTATION_SPEED_KEPT {
                        best = best.max((stretch[j].0 - stretch[i].0).abs());
                    }
                }
            }
            best
        }

        let mut best = 0.0f32;
        let mut stretch: Vec<(f32, f32)> = Vec::new();

        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];
            let view_delta = Self::normalize_angle(curr.view_yaw - prev.view_yaw);

            if !Self::is_free_air_step(prev, curr) || view_delta.abs() / curr.dt >= 2.0 {
                best = best.max(stretch_rotation(&stretch));
                stretch.clear();
                continue;
            }

            if stretch.is_empty() {
                stretch.push((0.0, prev.speed));
            }
            let relative = stretch.last().unwrap().0
                + Self::normalize_angle(curr.velocity_yaw - prev.velocity_yaw)
                - view_delta;
            stretch.push((relative, curr.speed));
        }

        best.max(stretch_rotation(&stretch))
    }

    fn max_frozen_view_speed_gain(frames: &[AirborneFrame]) -> f32 {
        fn stretch_gain(samples: &[(f32, f32)]) -> f32 {
            if samples.len() < MIN_SPEED_GAIN_SAMPLES {
                return 0.0;
            }
            let n = samples.len() as f32;
            let mean_t = samples.iter().map(|s| s.0).sum::<f32>() / n;
            let mean_v = samples.iter().map(|s| s.1).sum::<f32>() / n;
            let cov: f32 = samples.iter().map(|s| (s.0 - mean_t) * (s.1 - mean_v)).sum();
            let var: f32 = samples.iter().map(|s| (s.0 - mean_t).powi(2)).sum();
            if var <= 0.0 {
                return 0.0;
            }
            cov / var * (samples.last().unwrap().0 - samples[0].0)
        }

        let mut best = 0.0f32;
        let mut samples: Vec<(f32, f32)> = Vec::new();
        let mut time = 0.0f32;

        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];
            let view_delta = Self::normalize_angle(curr.view_yaw - prev.view_yaw).abs() / curr.dt;
            time += curr.dt;

            if !Self::is_free_air_step(prev, curr) || view_delta >= FROZEN_VIEW_DELTA {
                best = best.max(stretch_gain(&samples));
                samples.clear();
                continue;
            }
            if samples.is_empty() {
                samples.push((time - curr.dt, prev.speed));
            }
            samples.push((time, curr.speed));
        }

        best.max(stretch_gain(&samples))
    }

    fn zigzag_stats(frames: &[AirborneFrame]) -> (u32, u32, u32) {
        Self::zigzag_stats_view(frames, true)
    }

    fn zigzag_stats_view(frames: &[AirborneFrame], frozen_view: bool) -> (u32, u32, u32) {
        fn kick(a: &AirborneFrame, b: &AirborneFrame) -> f32 {
            (a.velocity.x * b.velocity.y - a.velocity.y * b.velocity.x) / a.speed
        }

        let mut flips = 0;
        let mut samples = 0;
        let mut current_consecutive = 0;
        let mut max_consecutive = 0;

        for i in 2..frames.len() {
            let (a, b, c) = (&frames[i - 2], &frames[i - 1], &frames[i]);
            if a.dt != 1.0
                || b.sim_tick - a.sim_tick != 1
                || c.sim_tick - b.sim_tick != 1
                || a.speed.min(b.speed).min(c.speed) < ZIGZAG_MIN_SPEED
                || !Self::is_free_air_step(a, b)
                || !Self::is_free_air_step(b, c)
                || (frozen_view
                    && (Self::normalize_angle(b.view_yaw - a.view_yaw).abs() >= FROZEN_VIEW_DELTA
                        || Self::normalize_angle(c.view_yaw - b.view_yaw).abs() >= FROZEN_VIEW_DELTA))
            {
                current_consecutive = 0;
                continue;
            }
            samples += 1;
            let (k1, k2) = (kick(a, b), kick(b, c));
            let in_range = |k: f32| (ZIGZAG_MIN_KICK..=ZIGZAG_MAX_KICK).contains(&k.abs());
            if k1 * k2 < 0.0 && in_range(k1) && in_range(k2) {
                flips += 1;
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                current_consecutive = 0;
            }
        }
        (flips, samples, max_consecutive)
    }

    /// Speed gain and rotation of the velocity per tick on stretches under a still view, against what a wish
    /// direction perpendicular to the velocity produces: the projection on it is raised from 0 to 30 every tick, so
    /// the speed grows by sqrt(v^2 + 30^2) - v and the velocity turns by atan(30 / v). A still view only offers 8
    /// fixed wish directions, and each gives one such kick before it saturates. Measured on the speed and on the
    /// telescoping yaw, so it needs no single-tick samples and works at every cl_cmdrate.
    fn gain_stats(frames: &[AirborneFrame]) -> GainStats {
        let mut stats = GainStats::default();
        let mut stretch: Vec<(f32, f32, f32)> = Vec::new();
        let mut close = |stretch: &mut Vec<(f32, f32, f32)>| {
            if stretch.len() >= GAIN_MIN_SAMPLES {
                let ticks = stretch.last().unwrap().0 - stretch[0].0;
                let n = stretch.len() as f32;
                let mean_t = stretch.iter().map(|s| s.0).sum::<f32>() / n;
                let mean_v = stretch.iter().map(|s| s.1).sum::<f32>() / n;
                let cov: f32 = stretch.iter().map(|s| (s.0 - mean_t) * (s.1 - mean_v)).sum();
                let var: f32 = stretch.iter().map(|s| (s.0 - mean_t).powi(2)).sum();
                if ticks >= GAIN_MIN_TICKS && var > 0.0 {
                    let slope = cov / var;
                    stats.ticks += ticks;
                    stats.sum += slope * ticks;
                    if ticks >= GAIN_THEORY_TICKS {
                        let gain = (mean_v * mean_v + AIR_SPEED_CAP * AIR_SPEED_CAP).sqrt() - mean_v;
                        let ratio = slope / gain;
                        stats.theory_ticks += ticks;
                        if (0.6..=1.4).contains(&ratio) {
                            stats.match_ticks += ticks;
                        } else if ratio > 1.6 {
                            stats.over_ticks += ticks;
                        }
                        let turn = (AIR_SPEED_CAP / mean_v).atan().to_degrees();
                        let rot = (stretch.last().unwrap().2 - stretch[0].2).abs() / ticks / turn;
                        if (0.5..=1.3).contains(&rot) {
                            stats.rot_match_ticks += ticks;
                        }
                    }
                }
            }
            stretch.clear();
        };

        for i in 1..frames.len() {
            let (prev, curr) = (&frames[i - 1], &frames[i]);
            let view_delta = Self::normalize_angle(curr.view_yaw - prev.view_yaw);
            let dv = ((curr.velocity.x - prev.velocity.x).powi(2) + (curr.velocity.y - prev.velocity.y).powi(2)).sqrt();
            if prev.speed < GAIN_MIN_SPEED
                || curr.speed < GAIN_MIN_SPEED
                || dv / curr.velocity_dt(prev) > MAX_AIR_ACCEL_PER_TICK
                || view_delta.abs() / curr.dt >= FROZEN_VIEW_DELTA
            {
                close(&mut stretch);
                continue;
            }
            if stretch.is_empty() {
                stretch.push((prev.sim_tick as f32, prev.speed, 0.0));
            }
            let rel = stretch.last().unwrap().2 + Self::normalize_angle(curr.velocity_yaw - prev.velocity_yaw) - view_delta;
            stretch.push((curr.sim_tick as f32, curr.speed, rel));
        }
        close(&mut stretch);
        stats
    }

    /// Explains the velocity of single-tick stretches with the movement keys: every tick a key state (none or one of
    /// the 8 WASD directions relative to the eye yaw) goes through AirAccelerate. A Viterbi search with a cost per key
    /// change finds the cheapest explanation; every state keeps its own velocity estimate, blended with the 1/8 unit
    /// quantized observation. A keyboard is explained by few changes and small errors; a wish direction written by a
    /// cheat needs a change on every tick, or stays unexplained, also while the view moves.
    fn key_fit(frames: &[AirborneFrame]) -> KeyFit {
        let mut fit = KeyFit::default();
        let mut start = 0;
        for i in 1..=frames.len() {
            let joined = i < frames.len() && {
                let (a, b) = (&frames[i - 1], &frames[i]);
                let dv = ((b.velocity.x - a.velocity.x).powi(2) + (b.velocity.y - a.velocity.y).powi(2)).sqrt();
                b.sim_tick - a.sim_tick == 1 && a.speed >= ZIGZAG_MIN_SPEED && b.speed >= ZIGZAG_MIN_SPEED && dv <= MAX_AIR_ACCEL_PER_TICK
            };
            if !joined {
                if i - start >= KEY_MIN_SEGMENT {
                    Self::key_fit_segment(&frames[start..i], &mut fit);
                }
                start = i;
            }
        }
        fit
    }

    fn key_fit_segment(frames: &[AirborneFrame], fit: &mut KeyFit) {
        #[derive(Clone, Copy, Default)]
        struct Survivor {
            vx: f32,
            vy: f32,
            cost: f32,
            error: f32,
            bad: u32,
            switches: u32,
            fast: u32,
            dwell: u32,
            stuck: u32,
        }
        const KEYS: usize = 9;
        let first = &frames[0];
        let mut states = [Survivor { vx: first.velocity.x, vy: first.velocity.y, dwell: KEY_FAST_DWELL + 1, ..Default::default() }; KEYS];
        for f in &frames[1..] {
            let yaw = f.view_yaw.to_radians();
            let best_prev = (0..KEYS).min_by(|&a, &b| states[a].cost.total_cmp(&states[b].cost)).unwrap();
            let mut next = states;
            for k in 0..KEYS {
                // key 8 is no key; the others are 45 degree steps from the eye yaw
                let wish = (k < 8).then(|| {
                    let a = yaw + k as f32 * std::f32::consts::FRAC_PI_4;
                    (a.cos(), a.sin())
                });
                let mut best: Option<(f32, Survivor)> = None;
                for &j in &[k, best_prev] {
                    let from = &states[j];
                    let (mut vx, mut vy) = (from.vx, from.vy);
                    if let Some((wx, wy)) = wish {
                        let add = AIR_SPEED_CAP - (vx * wx + vy * wy);
                        if add > 0.0 {
                            vx += add * wx;
                            vy += add * wy;
                        }
                    }
                    let (ex, ey) = (f.velocity.x - vx, f.velocity.y - vy);
                    let err2 = (ex * ex + ey * ey).min(KEY_ERROR_CAP * KEY_ERROR_CAP);
                    let cost = from.cost + err2 / (KEY_NOISE * KEY_NOISE) + if j == k { 0.0 } else { KEY_SWITCH_COST };
                    if best.as_ref().is_none_or(|b| cost < b.0) {
                        let mut s = *from;
                        s.cost = cost;
                        s.vx = vx + KEY_BLEND * ex;
                        s.vy = vy + KEY_BLEND * ey;
                        s.error += err2;
                        if err2 >= KEY_BAD_ERROR * KEY_BAD_ERROR {
                            s.bad += 1;
                        }
                        if j == k {
                            s.dwell += 1;
                        } else {
                            s.switches += 1;
                            if s.dwell <= KEY_FAST_DWELL {
                                s.fast += 1;
                            }
                            s.dwell = 1;
                        }
                        if err2 >= KEY_ERROR_CAP * KEY_ERROR_CAP {
                            s.stuck += 1;
                        }
                        best = Some((cost, s));
                    }
                }
                next[k] = best.unwrap().1;
            }
            states = next;
        }
        let best = states.iter().min_by(|a, b| a.cost.total_cmp(&b.cost)).unwrap();
        fit.ticks += (frames.len() - 1) as u32;
        fit.error += best.error;
        fit.bad += best.bad;
        fit.switches += best.switches;
        fit.fast += best.fast;
        fit.stuck += best.stuck;
    }

    fn compute_flight_features(frames: &[AirborneFrame], is_blast_jumping: bool) -> Option<[f32; FLIGHT_FEATURES.len()]> {
        if frames.len() < MIN_AIRBORNE_TICKS {
            return None;
        }

        let airborne_ticks = frames.len() as f32;
        let speeds: Vec<f32> = frames.iter().map(|f| f.speed).collect();
        let mean_speed = speeds.iter().sum::<f32>() / airborne_ticks;
        let max_speed = speeds.iter().copied().fold(0.0f32, f32::max);
        let speed_gain = frames.last().unwrap().speed - frames[0].speed;

        let static_view_rotation = Self::max_static_view_rotation(frames);
        let frozen_view_speed_gain = Self::max_frozen_view_speed_gain(frames);
        let (zigzag_flips, zigzag_samples, max_consecutive_flips) = Self::zigzag_stats(frames);
        // Blast jumps can have other air speed caps (and knockback), so the air control physics below skip them.
        let (gain, key, any_flips, any_samples) = if is_blast_jumping {
            (GainStats::default(), KeyFit::default(), 0, 0)
        } else {
            let (flips, samples, _) = Self::zigzag_stats_view(frames, false);
            (Self::gain_stats(frames), Self::key_fit(frames), flips, samples)
        };
        let single_steps = frames.windows(2).filter(|w| w[1].sim_tick - w[0].sim_tick == 1).count();
        let zigzag_rate = if zigzag_samples > 0 {
            zigzag_flips as f32 / zigzag_samples as f32
        } else {
            0.0
        };

        let mut analyzed_frames = 0;
        let mut optimal_strafes = 0;
        let mut perfect_strafes = 0;
        let mut desync_strafes = 0;
        let mut synced_strafes = 0;
        let mut view_static_frames = 0;
        let mut velocity_changing_frames = 0;

        let mut accel_angle_variance_sum = 0.0f32;
        let mut accel_angle_count = 0;
        let mut last_accel_angle: Option<f32> = None;

        let mut turn_deltas = Vec::new();
        let mut ideal_diffs = Vec::new();
        let mut current_turn_streak = 0;
        let mut max_turn_streak = 0;

        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];
            let dt = curr.velocity_dt(prev);
            if dt <= 0.0 {
                continue;
            }

            let view_yaw_delta = Self::normalize_angle(curr.view_yaw - prev.view_yaw).abs() / dt;
            let vel_yaw_delta = Self::normalize_angle(curr.velocity_yaw - prev.velocity_yaw).abs() / dt;

            if vel_yaw_delta > 1.0 {
                velocity_changing_frames += 1;
                if view_yaw_delta < 0.5 {
                    view_static_frames += 1;
                }
            }

            if vel_yaw_delta >= 2.0 {
                current_turn_streak += 1;
                max_turn_streak = max_turn_streak.max(current_turn_streak);
                turn_deltas.push(vel_yaw_delta);

                let ideal_turn = if curr.speed > 0.0 {
                    (30.0 / curr.speed).asin().to_degrees()
                } else {
                    0.0
                };
                ideal_diffs.push((vel_yaw_delta - ideal_turn).abs());
            } else {
                current_turn_streak = 0;
            }

            if let Some(accel_angle) = curr.acceleration_angle {
                analyzed_frames += 1;
                let is_optimal = (accel_angle - OPTIMAL_STRAFE_ANGLE).abs() < STRAFE_ANGLE_TOLERANCE;
                let is_perfect = (accel_angle - OPTIMAL_STRAFE_ANGLE).abs() < 10.0;

                if is_optimal {
                    optimal_strafes += 1;
                    if is_perfect {
                        perfect_strafes += 1;
                    }

                    if let Some(last) = last_accel_angle {
                        accel_angle_variance_sum += (accel_angle - last).abs();
                        accel_angle_count += 1;
                    }
                    last_accel_angle = Some(accel_angle);

                    let ratio = if vel_yaw_delta > 0.1 {
                        view_yaw_delta / vel_yaw_delta
                    } else {
                        0.0
                    };

                    if ratio < 0.3 {
                        desync_strafes += 1;
                    } else if (0.7..=1.3).contains(&ratio) {
                        synced_strafes += 1;
                    }
                }
            }
        }

        let optimal_ratio = if analyzed_frames > 0 {
            optimal_strafes as f32 / analyzed_frames as f32
        } else {
            0.0
        };
        let perfect_ratio = if optimal_strafes > 0 {
            perfect_strafes as f32 / optimal_strafes as f32
        } else {
            0.0
        };
        let desync_ratio = if optimal_strafes > 0 {
            desync_strafes as f32 / optimal_strafes as f32
        } else {
            0.0
        };
        let sync_ratio = if optimal_strafes > 0 {
            synced_strafes as f32 / optimal_strafes as f32
        } else {
            0.0
        };
        let view_static_ratio = if velocity_changing_frames > 0 {
            view_static_frames as f32 / velocity_changing_frames as f32
        } else {
            0.0
        };

        let accel_angle_consistency = if accel_angle_count > 0 {
            1.0 - (accel_angle_variance_sum / (accel_angle_count as f32 * 20.0)).min(1.0)
        } else {
            f32::NAN
        };

        let mut speed_checks = 0;
        let mut speed_maintained = 0;
        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];
            if prev.speed > MIN_SPEED && curr.speed > MIN_SPEED {
                speed_checks += 1;
                let loss = (prev.speed - curr.speed) / curr.velocity_dt(prev);
                if loss < 5.0 {
                    speed_maintained += 1;
                }
            }
        }
        let speed_maintained_ratio = if speed_checks > 0 {
            speed_maintained as f32 / speed_checks as f32
        } else {
            f32::NAN
        };

        let mean_turn_rate = if !turn_deltas.is_empty() {
            turn_deltas.iter().sum::<f32>() / turn_deltas.len() as f32
        } else {
            f32::NAN
        };

        let ideal_turn_diff = if !ideal_diffs.is_empty() {
            ideal_diffs.iter().sum::<f32>() / ideal_diffs.len() as f32
        } else {
            f32::NAN
        };

        Some([
            airborne_ticks,
            mean_speed,
            max_speed,
            speed_gain,
            static_view_rotation,
            frozen_view_speed_gain,
            zigzag_flips as f32,
            zigzag_samples as f32,
            zigzag_rate,
            max_consecutive_flips as f32,
            optimal_ratio,
            perfect_ratio,
            desync_ratio,
            sync_ratio,
            view_static_ratio,
            speed_maintained_ratio,
            max_turn_streak as f32,
            accel_angle_consistency,
            mean_turn_rate,
            ideal_turn_diff,
            if is_blast_jumping { 1.0 } else { 0.0 },
            gain.theory_ticks,
            gain.match_ticks,
            gain.over_ticks,
            gain.ticks,
            gain.sum,
            gain.rot_match_ticks,
            any_flips as f32,
            any_samples as f32,
            key.ticks as f32,
            key.error,
            key.bad as f32,
            key.switches as f32,
            key.fast as f32,
            key.stuck as f32,
            single_steps as f32,
            (frames.len() - 1) as f32,
        ])
    }
}

impl Default for MlAutoStrafe {
    fn default() -> Self {
        Self::new()
    }
}

fn quantile(values: &mut [f32], q: f32) -> f32 {
    let mut clean: Vec<f32> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if clean.is_empty() {
        return f32::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let n = clean.len();
    if n == 1 {
        return clean[0];
    }
    let pos = q * (n - 1) as f32;
    let base = pos.floor() as usize;
    let rest = pos - base as f32;
    if base + 1 < n {
        clean[base] + rest * (clean[base + 1] - clean[base])
    } else {
        clean[base]
    }
}

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
    (-ln_tail / std::f64::consts::LN_10).clamp(0.0, 50.0)
}

pub fn unit_features(flights: &[FlightRecord], total_flips: u32, total_samples: u32) -> Vec<f32> {
    let n_flights = flights.len() as f32;
    let col = |idx: usize| flights.iter().map(|f| f.features[idx]).collect::<Vec<f32>>();

    let mut rot = col(4);
    let mut gain = col(5);
    let mut z_rate = col(8);
    let mut opt = col(10);
    let mut perf = col(11);
    let mut desync = col(12);
    let mut sync = col(13);
    let mut v_stat = col(14);
    let mut spd_maint = col(15);
    let mut diff = col(19);
    let mut m_spd = col(1);
    let mut max_spd = col(2);

    let overall_zigzag_rate = if total_samples > 0 {
        total_flips as f32 / total_samples as f32
    } else {
        0.0
    };

    let max_flight_flips = flights.iter().map(|f| f.features[6]).fold(0.0f32, f32::max);
    let max_consecutive_flips = flights.iter().map(|f| f.features[9]).fold(0.0f32, f32::max);

    let circle_flight_share = flights.iter().filter(|f| f.features[4] >= 45.0).count() as f32 / n_flights;
    let zigzag_flight_share = flights.iter().filter(|f| f.features[6] >= 3.0 && f.features[8] >= 0.25).count() as f32 / n_flights;
    let speed_gain_flight_share = flights.iter().filter(|f| f.features[5] >= 25.0).count() as f32 / n_flights;
    let high_desync_share = flights.iter().filter(|f| f.features[12] >= 0.70).count() as f32 / n_flights;
    let high_sync_share = flights.iter().filter(|f| f.features[13] >= 0.40).count() as f32 / n_flights;

    let zigzag_binom = binom_score(total_flips, total_samples, 0.02) as f32;

    let total = |idx: usize| flights.iter().map(|f| f.features[idx]).sum::<f32>();
    let share = |a: f32, b: f32| if b > 0.0 { a / b } else { f32::NAN };
    let (theory_ticks, match_ticks, over_ticks, gain_ticks, gain_sum, rot_match) =
        (total(21), total(22), total(23), total(24), total(25), total(26));
    let (any_flips, any_samples, key_ticks, key_error, key_bad, key_switches, key_fast) =
        (total(27), total(28), total(29), total(30), total(31), total(32), total(33));
    let frozen_flips = flights.iter().filter(|f| f.features[20] == 0.0).map(|f| f.features[6]).sum::<f32>();
    let frozen_samples = flights.iter().filter(|f| f.features[20] == 0.0).map(|f| f.features[7]).sum::<f32>();

    vec![
        quantile(&mut rot, 0.5),
        quantile(&mut rot, 0.1),
        quantile(&mut rot, 0.9),
        rot.iter().copied().fold(0.0f32, f32::max),
        quantile(&mut gain, 0.5),
        quantile(&mut gain, 0.1),
        quantile(&mut gain, 0.9),
        gain.iter().copied().fold(0.0f32, f32::max),
        quantile(&mut z_rate, 0.5),
        quantile(&mut z_rate, 0.9),
        quantile(&mut opt, 0.5),
        quantile(&mut opt, 0.9),
        quantile(&mut perf, 0.5),
        quantile(&mut perf, 0.9),
        quantile(&mut desync, 0.5),
        quantile(&mut desync, 0.9),
        quantile(&mut sync, 0.5),
        quantile(&mut sync, 0.9),
        quantile(&mut v_stat, 0.5),
        quantile(&mut v_stat, 0.9),
        quantile(&mut diff, 0.5),
        quantile(&mut diff, 0.1),
        quantile(&mut spd_maint, 0.5),
        quantile(&mut spd_maint, 0.9),
        quantile(&mut m_spd, 0.5),
        quantile(&mut max_spd, 0.9),
        total_flips as f32,
        total_samples as f32,
        overall_zigzag_rate,
        max_flight_flips,
        max_consecutive_flips,
        circle_flight_share,
        zigzag_flight_share,
        speed_gain_flight_share,
        high_desync_share,
        high_sync_share,
        zigzag_binom,
        n_flights,
        share(gain_sum, gain_ticks),
        share(match_ticks, theory_ticks),
        match_ticks / (theory_ticks + 60.0),
        match_ticks,
        share(over_ticks, theory_ticks),
        theory_ticks,
        share(rot_match, theory_ticks),
        rot_match / (theory_ticks + 60.0),
        share(any_flips, any_samples),
        binom_score(any_flips as u32, any_samples as u32, 0.02) as f32,
        share((any_flips - frozen_flips).max(0.0), (any_samples - frozen_samples).max(0.0)),
        share(key_error, key_ticks).sqrt(),
        share(key_bad, key_ticks),
        share(key_switches, key_ticks),
        share(key_fast, key_ticks),
        binom_score(key_fast as u32, key_ticks as u32, 0.02) as f32,
        key_ticks,
        share(total(35), total(36)),
    ]
}

impl<'a> CheatAlgorithm<'a> for MlAutoStrafe {
    fn default(&self) -> bool {
        true
    }

    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        ALGORITHM_NAME
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::ServerInfo])
    }

    fn init(&mut self) -> Result<(), Error> {
        self.player_trackers.clear();
        self.jankguard = JankGuard::default();
        self.host_entity = None;
        self.is_stv = false;
        self.flights.clear();
        self.units.clear();
        Ok(())
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jankguard.on_message(message, state, parser_state, tick);

        if let Message::ServerInfo(info) = message {
            self.is_stv = info.stv;
            self.host_entity = Some(info.player_slot as u32 + 1);
        }

        Ok(vec![])
    }

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _parser_state: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        let current_tick: u32 = state.tick.into();

        for player in &state.players {
            let player_info = match player.info.as_ref() {
                Some(info) => info,
                None => continue,
            };

            let steam_id = match SteamID::from_steam3(&player_info.steam_id) {
                Ok(sid) => u64::from(sid),
                Err(_) => continue,
            };

            if matches!(player.team, Team::Other | Team::Spectator) {
                continue;
            }

            if !self.is_stv && Some(u32::from(player.entity)) == self.host_entity {
                if player.state != PlayerState::Alive {
                    continue;
                }
            }

            let is_grounded = Self::is_grounded(player);
            let sim_tick = state.sim_tick(player);
            if !is_grounded && sim_tick.rem_euclid(self.sample_every) != 0 {
                continue;
            }

            let tracker = self.player_trackers.entry(steam_id).or_default();

            if is_grounded {
                if tracker.airborne_frames.len() >= MIN_AIRBORNE_TICKS {
                    let spawned_ticks = self.jankguard.spawned(&steam_id, current_tick);
                    let teleported_ticks = self.jankguard.teleported(&steam_id, current_tick);
                    let is_blast_jumping = tracker.airborne_frames.iter().any(|f| f.blast_jumping);

                    if spawned_ticks > 30 && teleported_ticks > 30 {
                        let on_lift = Self::is_standing_on_lift_or_platform(&tracker.airborne_frames);
                        let on_ramp = Self::is_ramp_or_stair_motion(&tracker.airborne_frames, is_blast_jumping);
                        let external_movement = tracker.airborne_frames.iter().any(|f| f.external_movement);

                        if !on_lift && !on_ramp && !external_movement {
                            // the first sample of a flight has no velocity yet
                            let flown = &tracker.airborne_frames[1..];
                            let (flips, samples, _) = Self::zigzag_stats(flown);
                            tracker.total_flips += flips;
                            tracker.total_samples += samples;

                            if let Some(features) = Self::compute_flight_features(flown, is_blast_jumping) {
                                let start_tick = u32::from(tracker.airborne_frames[0].tick);
                                let end_tick = u32::from(tracker.airborne_frames.last().unwrap().tick);

                                // Flight score: model if loaded, or physical heuristic proxy
                                let score = if let Some(m) = model() {
                                    m.flight.probability(&features)
                                } else {
                                    (features[4] / 100.0).max(features[5] / 50.0).max(features[8])
                                };

                                tracker.usable_flights.push(FlightRecord {
                                    start_tick,
                                    end_tick,
                                    player: steam_id,
                                    features,
                                    score,
                                });
                            }
                        }
                    }
                }
                tracker.airborne_frames.clear();
                tracker.last_position = None;
                tracker.last_velocity = None;
                tracker.last_sim_tick = None;
            } else {
                let dt = match tracker.last_sim_tick {
                    Some(last_sim) => {
                        let delta = sim_tick - last_sim;
                        if delta <= 0 {
                            continue;
                        }
                        if delta > 24 {
                            tracker.airborne_frames.clear();
                            tracker.last_position = None;
                            tracker.last_velocity = None;
                            tracker.last_sim_tick = Some(sim_tick);
                            continue;
                        }
                        delta as f32
                    }
                    None => 1.0,
                };

                let velocity = match &tracker.last_position {
                    Some(prev_pos) => Self::calculate_velocity(prev_pos, &player.position, dt),
                    None => Vector { x: 0.0, y: 0.0, z: 0.0 },
                };

                let acceleration_angle = tracker
                    .last_velocity
                    .as_ref()
                    .and_then(|prev_vel| Self::calculate_acceleration_angle(prev_vel, &velocity));

                let frame = AirborneFrame {
                    tick: state.tick,
                    sim_tick,
                    dt,
                    position: player.position,
                    velocity,
                    speed: Self::get_speed_2d(&velocity),
                    view_yaw: player.view_angle,
                    velocity_yaw: Self::get_angle_2d(&velocity),
                    acceleration_angle,
                    blast_jumping: Self::in_cond(player, 81),
                    external_movement: EXTERNAL_MOVEMENT_CONDS.iter().any(|&c| Self::in_cond(player, c)),
                };

                tracker.last_position = Some(frame.position);
                tracker.last_velocity = Some(frame.velocity);
                tracker.last_sim_tick = Some(sim_tick);
                tracker.airborne_frames.push(frame);

                if tracker.airborne_frames.len() > 300 {
                    tracker.airborne_frames.drain(0..100);
                }
            }
        }

        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let enabled: bool = get_parameter_value(&self.params, "enabled");
        let min_flights: usize = get_parameter_value::<i32>(&self.params, "min_flights") as usize;
        let max_detections: usize = get_parameter_value::<i32>(&self.params, "max_detections") as usize;
        let min_score: f32 = get_parameter_value(&self.params, "min_score");

        let m = model();

        let mut detections = Vec::new();

        for (&steam_id, tracker) in &self.player_trackers {
            if tracker.usable_flights.len() < min_flights {
                continue;
            }

            let u_feats = unit_features(&tracker.usable_flights, tracker.total_flips, tracker.total_samples);
            let u_score = if let Some(m) = m {
                m.unit.probability(&u_feats)
            } else {
                f32::NAN
            };

            let unit_rec = UnitRecord {
                player: steam_id,
                n_flights: tracker.usable_flights.len() as u32,
                total_flips: tracker.total_flips,
                total_samples: tracker.total_samples,
                score: u_score,
                features: u_feats,
            };

            for f in &tracker.usable_flights {
                self.flights.push(f.clone());
            }
            self.units.push(unit_rec.clone());

            if !enabled || u_score.is_nan() || u_score < min_score {
                continue;
            }

            let mut sorted_flights = tracker.usable_flights.clone();
            sorted_flights.sort_by(|a, b| b.score.total_cmp(&a.score));

            for flight in sorted_flights.into_iter().take(max_detections) {
                detections.push(Detection {
                    tick: flight.start_tick,
                    algorithm: ALGORITHM_NAME.to_string(),
                    player: steam_id,
                    data: json!({
                        "player_score": u_score,
                        "flight_score": flight.score,
                        "start_tick": flight.start_tick,
                        "end_tick": flight.end_tick,
                        "static_view_rotation": flight.features[4],
                        "frozen_view_speed_gain": flight.features[5],
                        "zigzag_flips": flight.features[6] as u32,
                        "zigzag_rate": flight.features[8],
                        "desync_ratio": flight.features[12],
                        "sync_ratio": flight.features[13],
                        "perfect_ratio": flight.features[11],
                        "gain_match_ticks": flight.features[22],
                        "key_bad_ticks": flight.features[31],
                        "key_fast_switches": flight.features[33],
                        "total_flips": tracker.total_flips,
                        "total_samples": tracker.total_samples,
                    }),
                });
            }
        }

        detections.sort_by_key(|d| d.tick);
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_counts() {
        assert_eq!(flight_feature_names().len(), FLIGHT_FEATURES.len());
        assert_eq!(unit_feature_names().len(), UNIT_FEATURES.len());
    }

    #[test]
    fn test_model_loaded_and_valid() {
        let m = model().expect("ml_autostrafe model should be embedded and parseable");
        assert_eq!(m.unit.features, unit_feature_names());
        assert_eq!(m.flight.features, flight_feature_names());
    }

    /// A flight with the wish direction chosen per tick by `wish`, simulated with AirAccelerate.
    fn simulate(ticks: i64, wish: impl Fn(i64, (f32, f32)) -> Option<f32>) -> Vec<AirborneFrame> {
        let (mut vx, mut vy) = (320.0f32, 0.0f32);
        let (mut x, mut y) = (0.0f32, 0.0f32);
        (0..ticks)
            .map(|t| {
                if let Some(a) = wish(t, (vx, vy)) {
                    let (wx, wy) = (a.to_radians().cos(), a.to_radians().sin());
                    let add = AIR_SPEED_CAP - (vx * wx + vy * wy);
                    if add > 0.0 {
                        vx += add * wx;
                        vy += add * wy;
                    }
                }
                x += vx / TICKRATE;
                y += vy / TICKRATE;
                let velocity = Vector { x: vx, y: vy, z: 0.0 };
                AirborneFrame {
                    tick: (t as u32).into(),
                    sim_tick: t,
                    dt: 1.0,
                    position: Vector { x, y, z: 0.0 },
                    velocity,
                    speed: MlAutoStrafe::get_speed_2d(&velocity),
                    view_yaw: 0.0,
                    velocity_yaw: MlAutoStrafe::get_angle_2d(&velocity),
                    acceleration_angle: None,
                    blast_jumping: false,
                    external_movement: false,
                }
            })
            .collect()
    }

    fn cheat_wish(t: i64, v: (f32, f32)) -> Option<f32> {
        let side = if t % 2 == 0 { 90.0 } else { -90.0 };
        Some(v.1.atan2(v.0).to_degrees() + side)
    }

    #[test]
    fn test_gain_matches_perpendicular_wish_only() {
        let cheat = MlAutoStrafe::gain_stats(&simulate(60, cheat_wish));
        assert!(cheat.theory_ticks >= 50.0);
        assert_eq!(cheat.match_ticks, cheat.theory_ticks);

        // a still view: one strafe key saturates after a single kick
        let keys = MlAutoStrafe::gain_stats(&simulate(60, |_, _| Some(90.0)));
        assert!(keys.theory_ticks >= 50.0);
        assert_eq!(keys.match_ticks, 0.0);
    }

    #[test]
    fn test_key_fit_explains_keys_not_cheat() {
        let mut keys = KeyFit::default();
        MlAutoStrafe::key_fit_segment(&simulate(60, |t, _| Some(if t < 30 { 90.0 } else { -90.0 })), &mut keys);
        assert!(keys.switches <= 1 && keys.fast == 0, "{} {}", keys.switches, keys.fast);
        assert_eq!(keys.bad, 0);

        let mut cheat = KeyFit::default();
        MlAutoStrafe::key_fit_segment(&simulate(60, cheat_wish), &mut cheat);
        assert!(cheat.fast + cheat.bad >= 30, "{} {}", cheat.fast, cheat.bad);
    }

    #[test]
    fn test_zigzag_flips_detected() {
        // Construct 4 frames with alternating kicks under frozen view
        let f0 = AirborneFrame {
            tick: 100.into(),
            sim_tick: 1,
            dt: 1.0,
            position: Vector { x: 0.0, y: 0.0, z: 0.0 },
            velocity: Vector { x: 300.0, y: 0.0, z: 0.0 },
            speed: 300.0,
            view_yaw: 0.0,
            velocity_yaw: 0.0,
            acceleration_angle: Some(90.0),
            blast_jumping: false,
            external_movement: false,
        };
        let f1 = AirborneFrame {
            tick: 101.into(),
            sim_tick: 2,
            dt: 1.0,
            position: Vector { x: 300.0, y: 0.0, z: 0.0 },
            // Kicked left +15 HU/s
            velocity: Vector { x: 300.0, y: 15.0, z: 0.0 },
            speed: 300.4,
            view_yaw: 0.0,
            velocity_yaw: 2.86,
            acceleration_angle: Some(90.0),
            blast_jumping: false,
            external_movement: false,
        };
        let f2 = AirborneFrame {
            tick: 102.into(),
            sim_tick: 3,
            dt: 1.0,
            position: Vector { x: 600.0, y: 15.0, z: 0.0 },
            // Kicked right -15 HU/s (delta -30 HU/s)
            velocity: Vector { x: 300.0, y: -15.0, z: 0.0 },
            speed: 300.4,
            view_yaw: 0.0,
            velocity_yaw: -2.86,
            acceleration_angle: Some(90.0),
            blast_jumping: false,
            external_movement: false,
        };

        let frames = vec![f0, f1, f2];
        let (flips, samples, max_consec) = MlAutoStrafe::zigzag_stats(&frames);
        assert_eq!(samples, 1);
        assert_eq!(flips, 1);
        assert_eq!(max_consec, 1);
    }
}
