use crate::base::cheat_analyser_base::{CheatAnalyserState, Player};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::nocrex::jankguard::JankGuard;
use anyhow::Error;
use serde_json::json;
use std::collections::HashMap;
use std::f32::consts::PI;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::{MessageType, ParserState};

const ALGORITHM_NAME: &str = "autostrafe";
const MIN_AIRBORNE_TICKS: usize = 10;
const TICKRATE: f32 = 66.0;
const MIN_SPEED: f32 = 250.0;
const OPTIMAL_STRAFE_ANGLE: f32 = 90.0;
const STRAFE_ANGLE_TOLERANCE: f32 = 20.0;
/// Largest simulation-tick step still used as one velocity sample (command batches, light choke).
const MAX_SAMPLE_GAP: i64 = 4;
/// Beyond this step (PVS exit, long lag) a landing may have been missed, so the sequence restarts.
const MAX_SEQUENCE_GAP: i64 = 24;
const REPORT_COOLDOWN_TICKS: u32 = 200;
/// Per-tick velocity change above anything air acceleration can produce: knockback, airblast,
/// explosions. Such a tick ends a static-view stretch.
const MAX_AIR_ACCEL_PER_TICK: f32 = 100.0;
/// Static-view rotations only count while this share of the speed is kept (see
/// `max_static_view_rotation`).
const MIN_ROTATION_SPEED_KEPT: f32 = 0.9;
/// Yaw change per tick below the 10-bit eye angle step (0.35 degrees): the view did not move.
const FROZEN_VIEW_DELTA: f32 = 0.5;
const MIN_SPEED_GAIN_SAMPLES: usize = 12;
/// cond 80/122 B.A.S.E. Jumper parachute (active / deployed this flight), cond 82 kart, cond 98
/// grappling hook, cond 115 knocked into air (airblast): movement that air acceleration physics
/// does not describe. A parachute glide keeps speed and extends air control, so any flight that
/// had one is skipped entirely.
const EXTERNAL_MOVEMENT_CONDS: [u32; 5] = [80, 122, 82, 98, 115];
/// Air acceleration adds at most 30 HU/s (the air wish speed cap) per tick. A zigzag kick must be
/// close to it; 1/8 HU origin quantization noise stays well below the lower bound.
const ZIGZAG_MIN_KICK: f32 = 15.0;
const ZIGZAG_MAX_KICK: f32 = 45.0;
const ZIGZAG_MIN_SPEED: f32 = 150.0;
/// Flights reported per zigzag detection, the ones with the most flips.
const ZIGZAG_MAX_REPORTED_FLIGHTS: usize = 6;

#[derive(Clone, Debug)]
struct AirborneFrame {
    tick: DemoTick,
    sim_tick: i64,
    /// Simulation ticks since the previous sample; position/velocity are averaged over this span.
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
    /// Ticks between this frame's average velocity and the previous frame's.
    fn velocity_dt(&self, prev: &AirborneFrame) -> f32 {
        (self.dt + prev.dt) * 0.5
    }
}

struct PlayerTracker {
    airborne_frames: Vec<AirborneFrame>,
    last_position: Option<Vector>,
    last_velocity: Option<Vector>,
    last_sim_tick: Option<i64>,
    total_detections: u32,
    last_reported_tick: Option<u32>,
    zigzag_flips: u32,
    zigzag_samples: u32,
    /// (flight start tick, flight end tick, flips, samples) of flights with any flip
    zigzag_flights: Vec<(u32, u32, u32, u32)>,
}

pub struct AutoStrafe {
    params: Parameters,
    player_trackers: HashMap<u64, PlayerTracker>,
    mantreads_entities: HashMap<EntityId, EntityId>,
    jankguard: JankGuard,
    host_entity: Option<u32>,
    is_stv: bool,
}

impl AutoStrafe {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_confidence".to_string(), Parameter::Float(80.0)),
                ("min_static_view_rotation".to_string(), Parameter::Float(50.0)),
                ("min_frozen_view_speed_gain".to_string(), Parameter::Float(30.0)),
                ("zigzag_enabled".to_string(), Parameter::Bool(true)),
                ("zigzag_min_flips".to_string(), Parameter::Int(20)),
                ("zigzag_min_rate".to_string(), Parameter::Float(0.3)),
                ("zigzag_min_flight_flips".to_string(), Parameter::Int(2)),
            ]),
            player_trackers: HashMap::new(),
            mantreads_entities: HashMap::new(),
            jankguard: JankGuard::default(),
            host_entity: None,
            is_stv: false,
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
        // First bit is Grounded flag
        (player.flags & 1) == 1
    }

    fn calculate_velocity_from_positions(
        prev_pos: &Vector,
        curr_pos: &Vector,
        tick_delta: f32,
    ) -> Vector {
        let time_delta = tick_delta / TICKRATE;
        if time_delta <= 0.0 {
            return Vector {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
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

        // 1. Long airborne sequence with tiny horizontal movement (e.g. riding cart lift)
        if frames.len() >= 40 && net_horiz_dist < 120.0 {
            return true;
        }

        // 2. Maximum radial distance from starting position across all frames
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

        // 1. Sustained height gain over 35+ frames without blast jumping:
        // A standard TF2 jump peaks within ~30 ticks (0.45s) and descends.
        // Gaining altitude over 35+ ticks without an explosive boost is ramp/slope climbing.
        if !is_blast_jumping && frames.len() >= 35 && total_dz > 15.0 {
            return true;
        }

        // 2. Vertical kinematic acceleration analysis:
        // Ballistic motion in Source has gravity acceleration dv_z/dt ≈ -800 HU/s^2 (-12 HU/s/tick).
        // On stairs and ramps, the player is supported by collisions, so dv_z is near zero or positive.
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

        // Blast impulses only last a tick or two, so a flight held up by geometry still shows
        // mostly non-falling frames even while the blast jump condition is set.
        if sample_count >= 15 && is_blast_jumping && supported_ratio > 0.65 {
            return true;
        }

        if sample_count >= 15 && !is_blast_jumping {
            let avg_accel_z = (total_dv_z / total_dt) * TICKRATE;

            // In freefall, avg_accel_z is strongly negative (< -400) and supported_ratio is low (< 0.4).
            // On stairs/ramps, supported_ratio is high (> 0.60) or avg_accel_z is near zero (> -250).
            if avg_accel_z > -250.0 || supported_ratio > 0.65 {
                return true;
            }

            // Slope ratio check for gradual stair descent:
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

    /// Largest rotation of the velocity relative to the view within one stretch of near-static view.
    /// With a fixed view the wish direction is fixed too, and air acceleration stops once the
    /// velocity is within acos(30 / speed) (~84 degrees) of it. A human holding a strafe key turns
    /// the velocity at most ~25 degrees this way. Directional autostrafe rotates the wish direction
    /// with the velocity, so the velocity keeps turning while the view stays still.
    /// A fixed wish direction pointing behind the velocity (holding S) turns it further, but only by
    /// braking: the perpendicular component is conserved, so every degree past ~30 costs over 10% of
    /// speed. Only rotations that keep 90% of the speed count.
    fn max_static_view_rotation(frames: &[AirborneFrame]) -> f32 {
        fn stretch_rotation(stretch: &[(f32, f32)]) -> f32 {
            let mut best = 0.0f32;
            // Smooth speeds over 3 samples, single samples carry origin quantization noise
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
        // (velocity yaw relative to the view, unwrapped from the stretch start; speed)
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
            // Relative angle deltas telescope, so per-sample noise does not accumulate
            let relative = stretch.last().unwrap().0
                + Self::normalize_angle(curr.velocity_yaw - prev.velocity_yaw)
                - view_delta;
            stretch.push((relative, curr.speed));
        }

        best.max(stretch_rotation(&stretch))
    }

    /// Fast enough to analyse, and no velocity kick that air acceleration cannot produce.
    fn is_free_air_step(prev: &AirborneFrame, curr: &AirborneFrame) -> bool {
        let dv_x = curr.velocity.x - prev.velocity.x;
        let dv_y = curr.velocity.y - prev.velocity.y;
        let accel_per_tick = (dv_x * dv_x + dv_y * dv_y).sqrt() / curr.velocity_dt(prev);
        curr.speed >= MIN_SPEED && prev.speed >= MIN_SPEED && accel_per_tick <= MAX_AIR_ACCEL_PER_TICK
    }

    /// Largest horizontal speed gain within one stretch of frozen view (the networked yaw does not
    /// move). With the view frozen only 8 fixed wish directions exist, and air acceleration adds
    /// speed only while the velocity is within ~6 degrees of perpendicular to one of them, about
    /// 1 HU/s in total. Directional autostrafe holding W zigzags the velocity around the view
    /// instead, keeping the wish direction perpendicular and gaining ~1-2 HU/s every tick.
    /// The gain is a least-squares slope, as single samples carry +-10 HU/s of origin quantization.
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

    /// Per-tick velocity zigzag under a frozen view: (flips, samples). Directional autostrafe turns
    /// the wish direction 90 degrees to whichever side of the velocity the key direction is, so once
    /// the velocity reaches the key direction the side swaps every tick and the velocity is kicked
    /// ~30 HU/s left, right, left... A human cannot alternate strafe keys every tick, and a held key
    /// gives a single kick before the wish direction saturates.
    /// A sample is three consecutive single-tick frames; it flips when the two perpendicular kicks
    /// have opposite signs and both are near the air acceleration cap.
    fn zigzag_stats(frames: &[AirborneFrame]) -> (u32, u32) {
        fn kick(a: &AirborneFrame, b: &AirborneFrame) -> f32 {
            // Velocity change perpendicular to the previous velocity, signed by side
            (a.velocity.x * b.velocity.y - a.velocity.y * b.velocity.x) / a.speed
        }

        let mut flips = 0;
        let mut samples = 0;
        for i in 2..frames.len() {
            let (a, b, c) = (&frames[i - 2], &frames[i - 1], &frames[i]);
            if a.dt != 1.0
                || b.sim_tick - a.sim_tick != 1
                || c.sim_tick - b.sim_tick != 1
                || a.speed.min(b.speed).min(c.speed) < ZIGZAG_MIN_SPEED
                || !Self::is_free_air_step(a, b)
                || !Self::is_free_air_step(b, c)
                || Self::normalize_angle(b.view_yaw - a.view_yaw).abs() >= FROZEN_VIEW_DELTA
                || Self::normalize_angle(c.view_yaw - b.view_yaw).abs() >= FROZEN_VIEW_DELTA
            {
                continue;
            }
            samples += 1;
            let (k1, k2) = (kick(a, b), kick(b, c));
            let in_range = |k: f32| (ZIGZAG_MIN_KICK..=ZIGZAG_MAX_KICK).contains(&k.abs());
            if k1 * k2 < 0.0 && in_range(k1) && in_range(k2) {
                flips += 1;
            }
        }
        (flips, samples)
    }

    fn detect_directional_autostrafe(
        frames: &[AirborneFrame],
        has_mantreads: bool,
        is_blast_jumping: bool,
        min_static_view_rotation: f32,
        min_frozen_view_speed_gain: f32,
    ) -> Option<(f32, String)> {
        if frames.len() < MIN_AIRBORNE_TICKS {
            return None;
        }

        // Physical evidence that the wish direction moved without the view
        let static_view_rotation = Self::max_static_view_rotation(frames);
        let frozen_view_speed_gain = Self::max_frozen_view_speed_gain(frames);
        if static_view_rotation < min_static_view_rotation
            && frozen_view_speed_gain < min_frozen_view_speed_gain
        {
            return None;
        }

        let mut desync_optimal_strafes = 0;
        let mut consecutive_desync_strafes = 0;
        let mut max_consecutive_desync = 0;
        let mut synced_optimal_strafes = 0;
        let mut total_analyzed_frames = 0;
        let mut view_static_frames = 0;
        let mut velocity_changing_frames = 0;
        let mut perfect_strafes = 0;
        let mut accel_angle_variance_sum = 0.0;
        let mut accel_angle_count = 0;
        let mut last_accel_angle: Option<f32> = None;

        for i in 1..frames.len() - 1 {
            let prev = &frames[i - 1];
            let curr = &frames[i];

            if curr.speed < MIN_SPEED {
                consecutive_desync_strafes = 0;
                continue;
            }

            // Per-tick rates, so command batches (dt > 1) compare like single-tick samples
            let view_yaw_delta =
                Self::normalize_angle(curr.view_yaw - prev.view_yaw).abs() / curr.dt;
            let velocity_yaw_delta = Self::normalize_angle(curr.velocity_yaw - prev.velocity_yaw)
                .abs()
                / curr.velocity_dt(prev);

            if velocity_yaw_delta < 2.0 {
                continue;
            }

            velocity_changing_frames += 1;

            if view_yaw_delta < 2.0 {
                view_static_frames += 1;
            }

            if let Some(accel_angle) = curr.acceleration_angle {
                let is_optimal =
                    (accel_angle - OPTIMAL_STRAFE_ANGLE).abs() < STRAFE_ANGLE_TOLERANCE;
                let is_perfect = (accel_angle - OPTIMAL_STRAFE_ANGLE).abs() < 10.0;

                if is_optimal {
                    total_analyzed_frames += 1;

                    if is_perfect {
                        perfect_strafes += 1;
                    }

                    if let Some(last_angle) = last_accel_angle {
                        let angle_diff = (accel_angle - last_angle).abs();
                        accel_angle_variance_sum += angle_diff;
                        accel_angle_count += 1;
                    }
                    last_accel_angle = Some(accel_angle);

                    let view_velocity_ratio = if velocity_yaw_delta > 0.1 {
                        view_yaw_delta / velocity_yaw_delta
                    } else {
                        0.0
                    };

                    if view_velocity_ratio < 0.3 {
                        desync_optimal_strafes += 1;
                        consecutive_desync_strafes += 1;
                        max_consecutive_desync =
                            max_consecutive_desync.max(consecutive_desync_strafes);
                    } else {
                        consecutive_desync_strafes = 0;
                        if view_velocity_ratio > 0.7 && view_velocity_ratio < 1.3 {
                            synced_optimal_strafes += 1;
                        }
                    }
                } else {
                    consecutive_desync_strafes = 0;
                }
            }
        }

        if total_analyzed_frames < 5 {
            return None;
        }

        let desync_ratio = desync_optimal_strafes as f32 / total_analyzed_frames as f32;
        let sync_ratio = synced_optimal_strafes as f32 / total_analyzed_frames as f32;
        let perfect_ratio = perfect_strafes as f32 / total_analyzed_frames as f32;
        let static_view_ratio = if velocity_changing_frames > 0 {
            view_static_frames as f32 / velocity_changing_frames as f32
        } else {
            0.0
        };

        let angle_consistency = if accel_angle_count > 0 {
            let avg_variance = accel_angle_variance_sum / accel_angle_count as f32;
            1.0 - (avg_variance / 20.0).min(1.0)
        } else {
            0.0
        };

        if sync_ratio > 0.4 {
            return None;
        }

        let mut speed_consistency = 0.0;
        let mut speed_checks = 0;

        for i in 1..frames.len() {
            let prev = &frames[i - 1];
            let curr = &frames[i];

            if prev.speed > MIN_SPEED && curr.speed > MIN_SPEED {
                speed_checks += 1;
                let speed_loss = (prev.speed - curr.speed) / curr.velocity_dt(prev);
                if speed_loss < 5.0 {
                    speed_consistency += 1.0;
                }
            }
        }

        let speed_maintain_ratio = if speed_checks > 0 {
            speed_consistency / speed_checks as f32
        } else {
            0.0
        };

        let mut turning_sequences = Vec::new();
        let mut current_sequence_length = 0;

        for i in 1..frames.len() {
            let velocity_yaw_delta =
                Self::normalize_angle(frames[i].velocity_yaw - frames[i - 1].velocity_yaw).abs()
                    / frames[i].velocity_dt(&frames[i - 1]);

            if velocity_yaw_delta > 5.0 {
                current_sequence_length += 1;
            } else {
                if current_sequence_length > 3 {
                    turning_sequences.push(current_sequence_length);
                }
                current_sequence_length = 0;
            }
        }
        // A turn that lasts until landing is still a turn
        if current_sequence_length > 3 {
            turning_sequences.push(current_sequence_length);
        }

        let has_sustained_turns = turning_sequences.iter().any(|&len| len > 15);
        let has_multiple_long_turns =
            turning_sequences.iter().filter(|&&len| len > 10).count() >= 2;

        let is_directional_autostrafe = if has_mantreads && is_blast_jumping {
            desync_ratio > 0.80
                && static_view_ratio > 0.75
                && speed_maintain_ratio > 0.85
                && max_consecutive_desync >= 8
                && perfect_ratio > 0.60
                && (has_sustained_turns || has_multiple_long_turns || desync_optimal_strafes >= 14)
                && angle_consistency > 0.50
        } else {
            desync_ratio > 0.70
                && static_view_ratio > 0.60
                && speed_maintain_ratio > 0.80
                && (max_consecutive_desync > 5 || perfect_ratio > 0.50)
                && (has_sustained_turns || has_multiple_long_turns || desync_optimal_strafes > 10)
        };

        if is_directional_autostrafe {
            let confidence = desync_ratio * 30.0
                + static_view_ratio * 20.0
                + speed_maintain_ratio * 15.0
                + perfect_ratio * 15.0
                + angle_consistency * 10.0
                + (max_consecutive_desync as f32 * 2.0).min(10.0);

            let details = format!(
                "desync={}/{}, perfect={}, consecutive={}, view_static={:.0}%, speed_maintained={:.0}%, consistency={:.0}%, static_view_rotation={:.0}, frozen_view_speed_gain={:.0}",
                desync_optimal_strafes, total_analyzed_frames,
                perfect_strafes,
                max_consecutive_desync,
                static_view_ratio * 100.0,
                speed_maintain_ratio * 100.0,
                angle_consistency * 100.0,
                static_view_rotation,
                frozen_view_speed_gain
            );

            Some((confidence, details))
        } else {
            None
        }
    }
}

impl<'a> CheatAlgorithm<'a> for AutoStrafe {
    fn algorithm_name(&self) -> &str {
        ALGORITHM_NAME
    }

    fn needs_whole_demo(&self) -> bool {
        // The directional check only needs the current flight, but the zigzag flips are added up
        // over the whole demo (and Mantreads are only seen when the wearable is created)
        true
    }

    fn default(&self) -> bool {
        true
    }

    fn tooltip(&self) -> &'static str {
        "Detects directional auto-strafe used to gain speed or better mid-air control"
    }

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _parser_state: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        self.jankguard.on_tick(state);
        let mut detections = Vec::new();

        for player in state.players.iter().filter(|p| p.in_pvs) {
            if matches!(
                player.team,
                crate::base::cheat_analyser_base::Team::Other
                    | crate::base::cheat_analyser_base::Team::Spectator
            ) {
                continue;
            }

            // Ignore the host (POV recorder) if they are in spectator mode (dead)
            if !self.is_stv && Some(u32::from(player.entity)) == self.host_entity {
                if player.state != crate::base::cheat_analyser_base::PlayerState::Alive {
                    continue;
                }
            }

            // ==========================================

            let player_info = match player.info.as_ref() {
                Some(info) => info,
                None => continue,
            };

            let steam_id = match SteamID::from_steam3(&player_info.steam_id) {
                Ok(sid) => u64::from(sid),
                Err(_) => continue,
            };

            let tracker = self
                .player_trackers
                .entry(steam_id)
                .or_insert_with(|| PlayerTracker {
                    airborne_frames: Vec::new(),
                    last_position: None,
                    last_velocity: None,
                    last_sim_tick: None,
                    total_detections: 0,
                    last_reported_tick: None,
                    zigzag_flips: 0,
                    zigzag_samples: 0,
                    zigzag_flights: Vec::new(),
                });

            let is_grounded = Self::is_grounded(player);
            let current_tick = state.tick;
            // Demo ticks repeat and skip between snapshots, so movement is timed by the player's
            // own simulation tick, which advances once per processed usercmd.
            let sim_tick = state.sim_tick(player);
            let has_mantreads = self
                .mantreads_entities
                .values()
                .any(|&owner| owner == player.entity);

            if is_grounded {
                if tracker.airborne_frames.len() >= MIN_AIRBORNE_TICKS {
                    let tick_u32: u32 = current_tick.into();
                    let spawned_ticks = self.jankguard.spawned(&steam_id, tick_u32);
                    let teleported_ticks = self.jankguard.teleported(&steam_id, tick_u32);
                    // The blast jump condition is removed on landing, so look at the whole flight
                    let is_blast_jumping =
                        tracker.airborne_frames.iter().any(|f| f.blast_jumping);

                    if spawned_ticks > 30 && teleported_ticks > 30 {
                        let on_lift = Self::is_standing_on_lift_or_platform(&tracker.airborne_frames);
                        let on_ramp = Self::is_ramp_or_stair_motion(&tracker.airborne_frames, is_blast_jumping);

                        let external_movement =
                            tracker.airborne_frames.iter().any(|f| f.external_movement);

                        if !on_lift && !on_ramp && !external_movement {
                            let (flips, samples) = Self::zigzag_stats(&tracker.airborne_frames);
                            tracker.zigzag_flips += flips;
                            tracker.zigzag_samples += samples;
                            if flips > 0 {
                                tracker.zigzag_flights.push((
                                    u32::from(tracker.airborne_frames[0].tick),
                                    u32::from(tracker.airborne_frames.last().unwrap().tick),
                                    flips,
                                    samples,
                                ));
                            }

                            let min_rotation: f32 =
                                get_parameter_value(&self.params, "min_static_view_rotation");
                            let min_speed_gain: f32 =
                                get_parameter_value(&self.params, "min_frozen_view_speed_gain");
                            if let Some((confidence, details)) = Self::detect_directional_autostrafe(
                                &tracker.airborne_frames,
                                has_mantreads,
                                is_blast_jumping,
                                min_rotation,
                                min_speed_gain,
                            ) {
                                let min_conf: f32 = get_parameter_value(&self.params, "min_confidence");
                                let cooldown_ok = tracker.last_reported_tick.map_or(true, |last| {
                                    tick_u32.saturating_sub(last) >= REPORT_COOLDOWN_TICKS
                                });

                                if confidence >= min_conf && cooldown_ok {
                                    tracker.total_detections += 1;
                                    tracker.last_reported_tick = Some(tick_u32);

                                    let first_frame = &tracker.airborne_frames[0];
                                    let last_frame = tracker.airborne_frames.last().unwrap();
                                    let start_tick = u32::from(first_frame.tick);

                                    detections.push(Detection {
                                        player: steam_id,
                                        tick: start_tick,
                                        algorithm: ALGORITHM_NAME.to_string(),
                                        data: json!({
                                            "type": "directional_autostrafe",
                                            "confidence": confidence as i32,
                                            "airborne_ticks": tracker.airborne_frames.len(),
                                            "details": details,
                                            "start_tick": start_tick,
                                            "end_tick": u32::from(last_frame.tick),
                                            "total_detections": tracker.total_detections,
                                            "has_mantreads": has_mantreads,
                                            "blast_jumping": is_blast_jumping,
                                        }),
                                    });
                                }
                            }
                        }
                    }
                }

                tracker.airborne_frames.clear();
                tracker.last_position = Some(player.position);
                tracker.last_velocity = None;
                tracker.last_sim_tick = Some(sim_tick);
            } else {
                let gap = tracker.last_sim_tick.map(|last| sim_tick - last);
                match gap {
                    // Same snapshot again, or no new usercmd simulated: nothing moved
                    Some(0) => continue,
                    Some(gap) if gap > 0 && gap <= MAX_SAMPLE_GAP => {}
                    Some(gap) if gap > 0 && gap <= MAX_SEQUENCE_GAP => {
                        // Too coarse to sample, but still the same flight: only break the chain
                        tracker.last_position = Some(player.position);
                        tracker.last_velocity = None;
                        tracker.last_sim_tick = Some(sim_tick);
                        continue;
                    }
                    _ => {
                        // First sighting, PVS re-entry, long lag or a simtime discontinuity
                        tracker.airborne_frames.clear();
                        tracker.last_position = Some(player.position);
                        tracker.last_velocity = None;
                        tracker.last_sim_tick = Some(sim_tick);
                        continue;
                    }
                }
                let dt = gap.unwrap() as f32;
                let last_pos = tracker.last_position.unwrap_or(player.position);
                let velocity = Self::calculate_velocity_from_positions(&last_pos, &player.position, dt);
                let acceleration_angle = tracker
                    .last_velocity
                    .and_then(|last_vel| Self::calculate_acceleration_angle(&last_vel, &velocity));

                let frame = AirborneFrame {
                    tick: current_tick,
                    sim_tick,
                    dt,
                    position: player.position,
                    velocity,
                    speed: Self::get_speed_2d(&velocity),
                    view_yaw: player.view_angle,
                    velocity_yaw: Self::get_angle_2d(&velocity),
                    acceleration_angle,
                    blast_jumping: Self::in_cond(player, 81),
                    external_movement: EXTERNAL_MOVEMENT_CONDS
                        .iter()
                        .any(|&cond| Self::in_cond(player, cond)),
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

        Ok(detections)
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jankguard
            .on_message(message, state, parser_state, tick);

        if let Message::ServerInfo(info) = message {
            self.is_stv = info.stv;
            self.host_entity = Some(info.player_slot as u32 + 1);
        }

        if let Message::PacketEntities(msg) = message {
            for removed in &msg.removed_entities {
                self.mantreads_entities.remove(removed);
            }
            for entity in &msg.entities {
                if entity.update_type == UpdateType::Delete {
                    self.mantreads_entities.remove(&entity.entity_index);
                    continue;
                }
                let class_name = parser_state
                    .server_classes
                    .get(usize::from(entity.server_class))
                    .filter(|c| c.id == entity.server_class)
                    .or_else(|| parser_state.server_classes.iter().find(|c| c.id == entity.server_class))
                    .map(|c| c.name.as_str())
                    .unwrap_or("");

                if class_name.contains("Wearable") || class_name.contains("Weapon") {
                    let mut owner_ent = None;
                    let mut item_def = None;
                    for prop in entity.props(parser_state) {
                        if let Some((_table, name)) = prop.identifier.names() {
                            match name.as_str() {
                                "m_hOwnerEntity" | "m_hOwner" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        let handle = val as u32;
                                        let ent = EntityId::from(handle & 0x7FF);
                                        if u32::from(ent) != 0x7FF && u32::from(ent) != 0 {
                                            owner_ent = Some(ent);
                                        }
                                    }
                                }
                                "m_iItemDefinitionIndex" => {
                                    if let Ok(val) = i64::try_from(&prop.value) {
                                        item_def = Some(val as u16);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    if item_def == Some(444) {
                        if let Some(owner) = owner_ent {
                            self.mantreads_entities.insert(entity.entity_index, owner);
                        }
                    } else if item_def.is_some() {
                        self.mantreads_entities.remove(&entity.entity_index);
                    }
                }
            }
        }

        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let mut detections = Vec::new();
        if !get_parameter_value::<bool>(&self.params, "zigzag_enabled") {
            return Ok(detections);
        }
        let min_flips: i32 = get_parameter_value(&self.params, "zigzag_min_flips");
        let min_rate: f32 = get_parameter_value(&self.params, "zigzag_min_rate");
        let min_flight_flips: i32 = get_parameter_value(&self.params, "zigzag_min_flight_flips");

        // Judged over the whole demo: a single flight holds too few single-tick samples
        for (&steam_id, tracker) in &self.player_trackers {
            let rate = tracker.zigzag_flips as f32 / tracker.zigzag_samples.max(1) as f32;
            if (tracker.zigzag_flips as i32) < min_flips || rate < min_rate {
                continue;
            }
            let mut flights: Vec<_> = tracker
                .zigzag_flights
                .iter()
                .filter(|f| f.2 as i32 >= min_flight_flips)
                .collect();
            flights.sort_by(|a, b| b.2.cmp(&a.2));
            flights.truncate(ZIGZAG_MAX_REPORTED_FLIGHTS);
            flights.sort_by_key(|f| f.0);
            for &&(start_tick, end_tick, flips, samples) in &flights {
                detections.push(Detection {
                    player: steam_id,
                    tick: start_tick,
                    algorithm: ALGORITHM_NAME.to_string(),
                    data: json!({
                        "type": "zigzag_autostrafe",
                        "start_tick": start_tick,
                        "end_tick": end_tick,
                        "flight_flips": flips,
                        "flight_samples": samples,
                        "total_flips": tracker.zigzag_flips,
                        "total_samples": tracker.zigzag_samples,
                        "flip_rate": (rate as f64 * 100.0).round() / 100.0,
                    }),
                });
            }
        }
        Ok(detections)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        match self.jankguard.handled_messages() {
            Ok(mut msgs) => {
                msgs.push(MessageType::ServerInfo);
                if !msgs.contains(&MessageType::PacketEntities) {
                    msgs.push(MessageType::PacketEntities);
                }
                Ok(msgs)
            }
            Err(e) => Err(e),
        }
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }
}
