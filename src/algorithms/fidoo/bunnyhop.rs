use std::collections::HashMap;

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::ParserState;

use crate::base::cheat_analyser_base::{CheatAnalyserState, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::nocrex::jankguard::JankGuard;

const TICK_INTERVAL: f32 = 0.015; // 66.67 Hz standard Source engine tick interval

#[derive(Default, Clone)]
struct PlayerBhopState {
    ground_ticks: u32,
    air_ticks: u32,
    air_ticks_at_landing: u32,
    current_streak: u32,
    hop_ground_ticks: Vec<u32>,
    hop_speeds: Vec<f32>,
    last_reported_streak: u32,
    prev_pos: Option<Vector>,
    prev_on_ground: bool,
    last_movement_dz: f32,
    launch_z: f32,
    apex_z: f32,
    last_pos_sample: Option<Vector>,
    last_sample_sim: Option<i64>,
    last_sample_server_tick: Option<u32>,
    current_speed_hu_s: f32,
    hop_start_pos: Option<Vector>,
    hop_start_sim: Option<i64>,
    hop_start_server_tick: Option<u32>,
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

impl PlayerBhopState {
    fn reset_streak(&mut self) {
        self.current_streak = 0;
        self.hop_ground_ticks.clear();
        self.hop_speeds.clear();
        self.last_reported_streak = 0;
        self.min_x = 0.0;
        self.max_x = 0.0;
        self.min_y = 0.0;
        self.max_y = 0.0;
    }

    fn set_hop_start(&mut self, pos: Vector, sim: Option<i64>, server_tick: u32) {
        self.hop_start_pos = Some(pos);
        self.hop_start_sim = sim;
        self.hop_start_server_tick = Some(server_tick);
    }

    fn record_hop(&mut self, pos: Vector) {
        if self.current_streak <= 1 {
            self.min_x = pos.x;
            self.max_x = pos.x;
            self.min_y = pos.y;
            self.max_y = pos.y;
        } else {
            self.min_x = self.min_x.min(pos.x);
            self.max_x = self.max_x.max(pos.x);
            self.min_y = self.min_y.min(pos.y);
            self.max_y = self.max_y.max(pos.y);
        }
    }

    fn bbox_diagonal(&self) -> f32 {
        let dx = self.max_x - self.min_x;
        let dy = self.max_y - self.min_y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Computes elapsed simulation ticks (preferring networked simtime, falling back to server ticks).
/// Returns `None` if neither clock advanced (e.g. duplicate NetTick).
fn elapsed_sim_or_server_ticks(
    curr_sim: Option<i64>,
    prev_sim: Option<i64>,
    curr_server_tick: u32,
    prev_server_tick: Option<u32>,
) -> Option<u32> {
    if let (Some(s), Some(prev_s)) = (curr_sim, prev_sim) {
        if s > prev_s {
            return Some((s - prev_s) as u32);
        }
    }
    if let Some(prev_st) = prev_server_tick {
        let delta = curr_server_tick.saturating_sub(prev_st);
        if delta > 0 {
            return Some(delta);
        }
    }
    None
}

pub struct BunnyHop {
    player_states: HashMap<u64, PlayerBhopState>,
    jg: JankGuard,
    params: Parameters,
}

impl Default for BunnyHop {
    fn default() -> Self {
        Self::new()
    }
}

impl BunnyHop {
    pub fn new() -> Self {
        Self {
            player_states: HashMap::new(),
            jg: JankGuard::default(),
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_streak".to_string(), Parameter::Int(5)),
                ("min_air_ticks".to_string(), Parameter::Int(6)),
                ("max_ground_ticks_bhop".to_string(), Parameter::Int(1)),
                ("min_speed".to_string(), Parameter::Float(150.0)),
                ("min_streak_bbox_diagonal".to_string(), Parameter::Float(100.0)),
                ("min_hop_distance".to_string(), Parameter::Float(20.0)),
            ]),
        }
    }
}

impl<'a> CheatAlgorithm<'a> for BunnyHop {
    fn algorithm_name(&self) -> &str {
        "fidoo/bunnyhop"
    }

    fn live_warmup_ticks(&self) -> u32 {
        // Measured on the reference demos: 300 ticks of history are not enough, 350 are
        400
    }

    fn tooltip(&self) -> &'static str {
        "Detects auto-bhop by tracking consecutive perfect 1-tick ground jumps"
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn handled_messages(&self) -> Result<Vec<tf_demo_parser::MessageType>, bool> {
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

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        let ticknum = u32::from(state.tick);
        let server_tick = if state.server_tick != 0 {
            state.server_tick
        } else {
            ticknum
        };
        let players = &state.players;

        let min_streak: i32 = get_parameter_value(&self.params, "min_streak");
        let min_air_ticks: i32 = get_parameter_value(&self.params, "min_air_ticks");
        let max_ground_ticks: i32 =
            get_parameter_value(&self.params, "max_ground_ticks_bhop");
        let min_speed: f32 = get_parameter_value(&self.params, "min_speed");
        let min_streak_bbox_diagonal: f32 =
            get_parameter_value(&self.params, "min_streak_bbox_diagonal");
        let min_hop_distance: f32 =
            get_parameter_value(&self.params, "min_hop_distance");

        let mut tick_detections = Vec::new();
        let algo_name = self.algorithm_name().to_string();

        for player in players {
            let info = match &player.info {
                Some(info) if info.steam_id != "BOT" => info,
                _ => continue,
            };

            let steam_id: u64 = match SteamID::from_steam3(&info.steam_id) {
                Ok(sid) => u64::from(sid),
                Err(_) => continue,
            };

            if !player.in_pvs || player.state != PlayerState::Alive {
                self.player_states.remove(&steam_id);
                continue;
            }

            let ticks_since_event = self
                .jg
                .teleported(&steam_id, ticknum)
                .min(self.jg.spawned(&steam_id, ticknum));

            // Ignore detections within 60 ticks of spawn or teleport
            if ticks_since_event < 60 {
                self.player_states.remove(&steam_id);
                continue;
            }

            // Swimming in water alters ground/air flag behavior
            if player.is_in_water() {
                self.player_states.remove(&steam_id);
                continue;
            }

            // Stunned, taunting, or special movement conditions (Halloween spells: tiny, ghost, kart, bomb head, etc.)
            if player.is_taunting() || player.is_stunned() || player.has_special_movement_cond() {
                self.player_states.remove(&steam_id);
                continue;
            }

            let pstate = self.player_states.entry(steam_id).or_default();

            let sim = state.sim_ticks.get(&player.entity).copied();
            let on_ground = player.is_on_ground();
            let prev_z = pstate.prev_pos.map_or(player.position.z, |p| p.z);
            let dz = player.position.z - prev_z;

            let (dx, dy) = if let Some(prev_pos) = pstate.prev_pos {
                (player.position.x - prev_pos.x, player.position.y - prev_pos.y)
            } else {
                (0.0, 0.0)
            };
            let pos_delta_sq = dx * dx + dy * dy;

            // Time is measured in the player's networked sim ticks (server ticks as fallback),
            // never demo ticks: recorder stalls or packet jitter can pack multiple server ticks
            // into one demo tick (inflating speed 2x-3x) or stretch one sim tick across multiple
            // demo ticks (halving speed).
            let pos_updated = dz.abs() > 0.001 || pos_delta_sq > 0.001;

            if pstate.last_pos_sample.is_none() {
                pstate.last_pos_sample = Some(player.position);
                pstate.last_sample_sim = sim;
                pstate.last_sample_server_tick = Some(server_tick);
            } else if pos_updated {
                if let Some(last_sample) = pstate.last_pos_sample {
                    if let Some(tick_delta) = elapsed_sim_or_server_ticks(
                        sim,
                        pstate.last_sample_sim,
                        server_tick,
                        pstate.last_sample_server_tick,
                    ) {
                        let sample_dx = player.position.x - last_sample.x;
                        let sample_dy = player.position.y - last_sample.y;
                        let dist = (sample_dx * sample_dx + sample_dy * sample_dy).sqrt();
                        pstate.current_speed_hu_s = dist / (tick_delta as f32 * TICK_INTERVAL);
                    }
                }

                pstate.last_pos_sample = Some(player.position);
                pstate.last_sample_sim = sim;
                pstate.last_sample_server_tick = Some(server_tick);
            } else {
                let sim_advanced = matches!((sim, pstate.last_sample_sim), (Some(s), Some(prev_s)) if s > prev_s);
                let stale_server = pstate
                    .last_sample_server_tick
                    .is_some_and(|prev_st| server_tick.saturating_sub(prev_st) > 4);
                if sim_advanced {
                    pstate.current_speed_hu_s = 0.0;
                    pstate.last_pos_sample = Some(player.position);
                    pstate.last_sample_sim = sim;
                    pstate.last_sample_server_tick = Some(server_tick);
                } else if stale_server {
                    pstate.current_speed_hu_s = 0.0;
                }
            }

            let current_speed = pstate.current_speed_hu_s;

            if on_ground {
                pstate.launch_z = player.position.z;
                pstate.apex_z = player.position.z;

                if !pstate.prev_on_ground {
                    pstate.air_ticks_at_landing = pstate.air_ticks;
                    pstate.ground_ticks = 1;
                    pstate.air_ticks = 0;
                } else {
                    pstate.ground_ticks += 1;
                    if pstate.ground_ticks > (max_ground_ticks.max(1) + 3) as u32 {
                        // Walking on ground breaks the bunnyhop streak
                        pstate.reset_streak();
                    }
                }

                if pstate.hop_start_pos.is_none() {
                    pstate.set_hop_start(player.position, sim, server_tick);
                }
            } else {
                pstate.apex_z = pstate.apex_z.max(player.position.z);

                // Compute horizontal displacement and average speed across the hop using actual
                // elapsed sim/server ticks (not snapshot counts, which undercount at low updaterates).
                let hop_dist = if let Some(start_pos) = pstate.hop_start_pos {
                    let hdx = player.position.x - start_pos.x;
                    let hdy = player.position.y - start_pos.y;
                    (hdx * hdx + hdy * hdy).sqrt()
                } else {
                    0.0
                };

                let hop_elapsed_ticks = elapsed_sim_or_server_ticks(
                    sim,
                    pstate.hop_start_sim,
                    server_tick,
                    pstate.hop_start_server_tick,
                )
                .unwrap_or(1);
                let hop_speed_from_dist = hop_dist / (hop_elapsed_ticks as f32 * TICK_INTERVAL);

                // Check for zero-tick ground bounce (instant jump impulse without FL_ONGROUND toggle)
                // Filter out Scout mid-air double/triple jumps, blast jumping trajectories, and stationary collision oscillations:
                let fallen_from_apex = pstate.apex_z - prev_z;
                let near_ground_level = prev_z <= pstate.launch_z + 25.0;

                let is_ground_bounce = pos_updated
                    && !pstate.prev_on_ground
                    && !player.is_blastjumping()
                    && pstate.last_movement_dz < -1.0
                    && dz > 5.0
                    && pstate.air_ticks >= min_air_ticks as u32
                    && fallen_from_apex >= 20.0
                    && near_ground_level
                    && hop_dist >= min_hop_distance
                    && hop_speed_from_dist >= 80.0;

                let mut hop_accepted = false;

                if is_ground_bounce {
                    pstate.air_ticks_at_landing = pstate.air_ticks;
                    pstate.air_ticks = 1;
                    pstate.ground_ticks = 0;
                    pstate.launch_z = player.position.z;
                    pstate.apex_z = player.position.z;

                    // Check if 0 ground ticks satisfies the configured max_ground_ticks limit
                    if 0 <= max_ground_ticks && current_speed >= min_speed {
                        pstate.current_streak += 1;
                        pstate.hop_ground_ticks.push(0);
                        pstate.hop_speeds.push(current_speed);
                        pstate.record_hop(player.position);
                        hop_accepted = true;
                    } else {
                        pstate.reset_streak();
                    }
                    pstate.set_hop_start(player.position, sim, server_tick);
                } else if pstate.prev_on_ground {
                    let hopped_ground_ticks = pstate.ground_ticks;
                    let air_at_landing = pstate.air_ticks_at_landing;
                    pstate.ground_ticks = 0;
                    pstate.air_ticks = 1;
                    pstate.launch_z = player.position.z;
                    pstate.apex_z = player.position.z;

                    if air_at_landing >= min_air_ticks as u32
                        && current_speed >= min_speed
                        && hop_dist >= (min_hop_distance * 0.75)
                        && hop_speed_from_dist >= 80.0
                    {
                        // Strictly enforce that this hop does not exceed max_ground_ticks
                        if (hopped_ground_ticks as i32) <= max_ground_ticks {
                            pstate.current_streak += 1;
                            pstate.hop_ground_ticks.push(hopped_ground_ticks);
                            pstate.hop_speeds.push(current_speed);
                            pstate.record_hop(player.position);
                            hop_accepted = true;
                        } else {
                            // Ground contact exceeded max_ground_ticks -> break streak
                            pstate.reset_streak();
                        }
                    } else if air_at_landing >= min_air_ticks as u32 {
                        pstate.reset_streak();
                    }
                    pstate.set_hop_start(player.position, sim, server_tick);
                } else {
                    pstate.air_ticks += 1;
                }

                if hop_accepted
                    && pstate.current_streak >= min_streak as u32
                    && pstate.current_streak > pstate.last_reported_streak
                    && pstate.hop_ground_ticks.iter().all(|&g| g <= max_ground_ticks as u32)
                    && pstate.bbox_diagonal() >= min_streak_bbox_diagonal
                {
                    pstate.last_reported_streak = pstate.current_streak;
                    let avg_speed = pstate.hop_speeds.iter().sum::<f32>()
                        / pstate.hop_speeds.len() as f32;
                    let max_speed = pstate
                        .hop_speeds
                        .iter()
                        .cloned()
                        .fold(0.0_f32, f32::max);
                    tick_detections.push(Detection {
                        tick: ticknum,
                        algorithm: algo_name.clone(),
                        player: steam_id,
                        data: json!({
                            "class": player.class_name(),
                            "weapon": state.get_player_weapon(player),
                            "streak": pstate.current_streak,
                            "ground_ticks": pstate.hop_ground_ticks.clone(),
                            "avg_speed": avg_speed,
                            "max_speed": max_speed,
                            "speed": current_speed,
                        }),
                    });
                }
            }

            if pos_updated {
                pstate.last_movement_dz = dz;
            }
            pstate.prev_pos = Some(player.position);
            pstate.prev_on_ground = on_ground;
        }

        Ok(tick_detections)
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        Ok(vec![])
    }
}
