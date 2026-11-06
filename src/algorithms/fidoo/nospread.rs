use std::collections::HashMap;

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::message::packetentities::EntityId;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::{handle_to_entid, viewangle_delta};
use crate::util::nocrex::jankguard::JankGuard;

#[derive(Clone, Copy, Debug)]
struct WeaponSpreadInfo {
    category_id: u32,
    name: &'static str,
    spread_factor: f32,
    max_cone_degrees: f32,
}       

const MINIGUN: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 18, name: "Minigun", spread_factor: 0.030, max_cone_degrees: 1.72 };
const SMG: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 19, name: "SMG", spread_factor: 0.025, max_cone_degrees: 1.43 };
const PISTOL: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 41, name: "Pistol", spread_factor: 0.025, max_cone_degrees: 1.43 };
const REVOLVER: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 43, name: "Revolver", spread_factor: 0.020, max_cone_degrees: 1.15 };
const SCATTERGUN: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 16, name: "Scattergun", spread_factor: 0.040, max_cone_degrees: 2.29 };
const SHOTGUN: WeaponSpreadInfo =
    WeaponSpreadInfo { category_id: 12, name: "Shotgun", spread_factor: 0.040, max_cone_degrees: 2.29 };

/// Maps a bullet to its spread family. `weapon_id` is the TF_WEAPON_* id carried by
/// CTEFireBullets.m_iWeaponID; when it is present it alone decides, so weapons without
/// spread (sniper rifles, sentries, ...) return None. The name is only a fallback for a
/// missing id.
fn get_weapon_spread_info(weapon_id: Option<u32>, weapon_name: &str) -> Option<WeaponSpreadInfo> {
    if let Some(id) = weapon_id {
        return match id {
            // TF_WEAPON_MINIGUN
            18 => Some(MINIGUN),
            // TF_WEAPON_SMG, TF_WEAPON_CHARGED_SMG
            19 | 103 => Some(SMG),
            // TF_WEAPON_PISTOL, TF_WEAPON_PISTOL_SCOUT, TF_WEAPON_HANDGUN_SCOUT_SECONDARY
            41 | 42 | 75 => Some(PISTOL),
            // TF_WEAPON_REVOLVER
            43 => Some(REVOLVER),
            // TF_WEAPON_SCATTERGUN, TF_WEAPON_HANDGUN_SCOUT_PRIMARY, TF_WEAPON_SODA_POPPER, TF_WEAPON_PEP_BRAWLER_BLASTER
            16 | 71 | 76 | 85 => Some(SCATTERGUN),
            // TF_WEAPON_SHOTGUN_PRIMARY/SOLDIER/HWG/PYRO
            12..=15 => Some(SHOTGUN),
            _ => None,
        };
    }

    if ["Minigun", "Tomislav", "Brass Beast", "Huo-Long", "Deflector"].iter().any(|n| weapon_name.contains(n)) {
        return Some(MINIGUN);
    }
    if ["SMG", "Cleaner's Carbine"].iter().any(|n| weapon_name.contains(n)) {
        return Some(SMG);
    }
    if ["Pistol", "Winger", "Lugermorph"].iter().any(|n| weapon_name.contains(n)) {
        return Some(PISTOL);
    }
    if ["Revolver", "Ambassador", "Big Kill", "L'Etranger", "Enforcer", "Diamondback"]
        .iter()
        .any(|n| weapon_name.contains(n))
    {
        return Some(REVOLVER);
    }
    if ["Scattergun", "Shortstop", "Force-A-Nature", "Soda Popper", "Baby Face", "Back Scatter"]
        .iter()
        .any(|n| weapon_name.contains(n))
    {
        return Some(SCATTERGUN);
    }
    if weapon_name.contains("Shotgun") {
        return Some(SHOTGUN);
    }

    None
}

// Standard CUniformRandomStream PRNG
pub struct UniformRandomStream {
    m_idum: i32,
    m_iy: i32,
    m_iv: [i32; 32],
}

impl UniformRandomStream {
    pub fn new() -> Self {
        Self {
            m_idum: 0,
            m_iy: 0,
            m_iv: [0; 32],
        }
    }

    pub fn set_seed(&mut self, i_seed: i32) {
        self.m_idum = if i_seed < 0 { i_seed } else { -i_seed };
        self.m_iy = 0;
    }

    pub fn generate_random_number(&mut self) -> i32 {
        const IA: i32 = 16807;
        const IM: i32 = 2147483647;
        const IQ: i32 = 127773;
        const IR: i32 = 2836;
        const NTAB: usize = 32;
        const NDIV: i32 = 1 + (IM - 1) / NTAB as i32;

        if self.m_idum <= 0 || self.m_iy == 0 {
            if -self.m_idum < 1 {
                self.m_idum = 1;
            } else {
                self.m_idum = -self.m_idum;
            }

            for j in (0..NTAB + 8).rev() {
                let k = self.m_idum / IQ;
                self.m_idum = IA.wrapping_mul(self.m_idum - k * IQ) - IR * k;
                if self.m_idum < 0 {
                    self.m_idum += IM;
                }
                if j < NTAB {
                    self.m_iv[j] = self.m_idum;
                }
            }
            self.m_iy = self.m_iv[0];
        }

        let k = self.m_idum / IQ;
        self.m_idum = IA.wrapping_mul(self.m_idum - k * IQ) - IR * k;
        if self.m_idum < 0 {
            self.m_idum += IM;
        }
        let j = (self.m_iy / NDIV) as usize;
        self.m_iy = self.m_iv[j];
        self.m_iv[j] = self.m_idum;

        self.m_iy
    }

    pub fn random_float(&mut self, fl_low: f32, fl_high: f32) -> f32 {
        const AM: f32 = 1.0 / 2147483647.0;
        let mut fl = AM * self.generate_random_number() as f32;
        if fl > 0.99999 {
            fl = 0.99999;
        }
        (fl * (fl_high - fl_low)) + fl_low
    }
}

fn compute_predicted_spread(seed: i32, spread_factor: f32) -> (f32, f32) {
    let mut rng = UniformRandomStream::new();
    rng.set_seed(seed & 255);
    let x = rng.random_float(-0.5, 0.5) + rng.random_float(-0.5, 0.5);
    let y = rng.random_float(-0.5, 0.5) + rng.random_float(-0.5, 0.5);

    let rad_to_deg = 180.0 / std::f32::consts::PI;
    let dx = x * spread_factor * rad_to_deg;
    let dy = y * spread_factor * rad_to_deg;
    (dx, dy)
}

fn p_value_from_t(t: f64) -> f64 {
    let x = t.abs();
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;
    let k = 1.0 / (1.0 + p * x);
    let z = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_tail = z * (b1 * k + b2 * k.powi(2) + b3 * k.powi(3) + b4 * k.powi(4) + b5 * k.powi(5));
    (2.0 * cdf_tail).clamp(0.0, 1.0)
}

fn resolve_player_sid(player_raw: u32, state: &CheatAnalyserState) -> Option<u64> {
    let ent_ids = [
        EntityId::from(player_raw + 1),
        EntityId::from(player_raw),
        handle_to_entid(player_raw),
        handle_to_entid(player_raw + 1),
    ];
    for ent_id in ent_ids {
        if let Some(uid) = state.entid_to_userid.get(&ent_id) {
            if let Some(sid) = state.userid_to_id64.get(uid) {
                return Some(*sid);
            }
        }
        if let Some(player) = state.players.iter().find(|p| p.entity == ent_id) {
            if let Some(info) = &player.info {
                if let Ok(sid) = SteamID::from_steam3(&info.steam_id) {
                    return Some(u64::from(sid));
                }
            }
        }
    }
    None
}

/// Firing-tick reversals are counted over this many ticks (10 s).
const REVERSAL_WINDOW_TICKS: u32 = 600;

#[derive(Clone, Debug)]
struct ReversalEvent {
    tick: u32,
    magnitude: f32,
}

#[derive(Default, Clone)]
struct PlayerTracker {
    last_flag_tick: u32,
    reversals: Vec<ReversalEvent>,
    shot_samples: Vec<(u32, (f32, f32), Option<i32>, Option<f32>)>, // tick, delta, seed, spread
    current_weapon: Option<WeaponSpreadInfo>,
}

#[derive(Default)]
pub struct NoSpread {
    params: Parameters,
    jg: JankGuard,
    prev_players: HashMap<u64, (f32, f32, u32)>, // pitch, yaw, tick
    player_tick_deltas: HashMap<u64, HashMap<u32, (f32, f32)>>, // sid -> tick -> (dp, dy)
    player_trackers: HashMap<u64, PlayerTracker>,
    fire_events: HashMap<u64, Vec<(u32, Option<i32>, Option<f32>, Option<u32>)>>, // (tick, seed, spread, weapon_id)
    detections: Vec<Detection>,
}

impl PlayerTracker {
    fn check_seed_correlation(
        &mut self,
        player_sid: u64,
        tick: u32,
        min_shots: usize,
        min_seed_correlation: f32,
    ) -> Option<Detection> {
        if self.shot_samples.len() < min_shots {
            return None;
        }

        let weapon_info = self.current_weapon?;

        let valid_seeds: Vec<_> = self
            .shot_samples
            .iter()
            .rev()
            .take(16)
            .filter_map(|s| s.2.map(|seed| (s.1, seed, s.3)))
            .collect();

        if valid_seeds.len() >= min_shots {
            let mut deltas = Vec::new();
            let mut preds = Vec::new();
            for (delta, seed, spread) in &valid_seeds {
                let spread_factor = spread.unwrap_or(weapon_info.spread_factor);
                let pred = compute_predicted_spread(*seed, spread_factor);
                deltas.push(*delta);
                preds.push(pred);
            }

            let count = deltas.len();
            let mean_d_yaw: f32 = deltas.iter().map(|(y, _)| *y).sum::<f32>() / count as f32;
            let mean_d_pitch: f32 = deltas.iter().map(|(_, p)| *p).sum::<f32>() / count as f32;
            let mean_p_x: f32 = preds.iter().map(|(x, _)| *x).sum::<f32>() / count as f32;
            let mean_p_y: f32 = preds.iter().map(|(_, y)| *y).sum::<f32>() / count as f32;

            let mut cov = 0.0f32;
            let mut var_d = 0.0f32;
            let mut var_p = 0.0f32;
            let mut total_residual = 0.0f32;

            for i in 0..count {
                let dy = deltas[i].0 - mean_d_yaw;
                let dp = deltas[i].1 - mean_d_pitch;
                let px = preds[i].0 - mean_p_x;
                let py = preds[i].1 - mean_p_y;

                cov += dy * px + dp * py;
                var_d += dy * dy + dp * dp;
                var_p += px * px + py * py;

                let res_x = deltas[i].0 + preds[i].0;
                let res_y = deltas[i].1 + preds[i].1;
                total_residual += (res_x * res_x + res_y * res_y).sqrt();
            }

            let denom = (var_d * var_p).sqrt();
            let r = if denom > 1e-7 { cov / denom } else { 0.0 };
            let mean_residual = total_residual / count as f32;

            let df = (count - 2) as f64;
            let r_f64 = r as f64;
            let t_stat = r_f64 * (df / (1.0 - r_f64 * r_f64).max(1e-7)).sqrt();
            let p_val = p_value_from_t(t_stat);

            if r <= min_seed_correlation && p_val < 0.001 {
                self.last_flag_tick = tick;
                return Some(Detection {
                    tick,
                    algorithm: "fidoo/nospread".to_string(),
                    player: player_sid,
                    data: json!({
                        "type": "SeedCorrelation",
                        "weapon_id": weapon_info.category_id,
                        "weapon_name": weapon_info.name,
                        "shot_count": count,
                        "correlation_coefficient": (r * 1000.0).round() / 1000.0,
                        "p_value": (p_val * 100000.0).round() / 100000.0,
                        "mean_residual_degrees": (mean_residual * 1000.0).round() / 1000.0,
                    }),
                });
            }
        }
        None
    }
}

impl NoSpread {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_shots_evaluated".to_string(), Parameter::Int(8)),
                ("min_seed_correlation".to_string(), Parameter::Float(-0.85)),
                ("min_variance_ratio".to_string(), Parameter::Float(4.5)),
                ("max_tracking_variance".to_string(), Parameter::Float(0.5)),
                ("spread_tolerance_margin".to_string(), Parameter::Float(0.5)),
                ("min_reversals".to_string(), Parameter::Int(4)),
            ]),
            jg: JankGuard::default(),
            prev_players: HashMap::new(),
            player_tick_deltas: HashMap::new(),
            player_trackers: HashMap::new(),
            fire_events: HashMap::new(),
            detections: Vec::new(),
        }
    }
}

impl<'a> CheatAlgorithm<'a> for NoSpread {
    fn default(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/nospread"
    }

    fn tooltip(&self) -> &'static str {
        "Detects No-Spread cheats on hitscan weapons."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn live_warmup_ticks(&self) -> u32 {
        // The reversals are counted over a window, and JankGuard ignores the 60 ticks after a spawn
        REVERSAL_WINDOW_TICKS + 60
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        let mut types = self.jg.handled_messages().unwrap_or_default();
        if !types.contains(&MessageType::TempEntities) {
            types.push(MessageType::TempEntities);
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
        tick: tf_demo_parser::demo::data::DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);

        if let Message::TempEntities(msg) = message {
            for event in &msg.events {
                let class = &parser_state.server_classes[usize::from(event.class_id)].name;
                if class.as_str() == "CTEFireBullets" {
                    const BULLETS_PLAYER: SendPropIdentifier =
                        SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
                    const BULLETS_SEED: SendPropIdentifier =
                        SendPropIdentifier::new("DT_TEFireBullets", "m_iSeed");
                    const BULLETS_SPREAD: SendPropIdentifier =
                        SendPropIdentifier::new("DT_TEFireBullets", "m_flSpread");
                    const BULLETS_WEAPON: SendPropIdentifier =
                        SendPropIdentifier::new("DT_TEFireBullets", "m_iWeaponID");

                    let mut player_opt = None;
                    let mut seed_opt = None;
                    let mut spread_opt = None;
                    let mut weapon_opt = None;

                    for prop in &event.props {
                        if prop.identifier == BULLETS_PLAYER {
                            if let Ok(id) = i64::try_from(&prop.value) {
                                player_opt = Some(id as u32);
                            }
                        } else if prop.identifier == BULLETS_SEED {
                            if let Ok(seed) = i64::try_from(&prop.value) {
                                seed_opt = Some(seed as i32);
                            }
                        } else if prop.identifier == BULLETS_SPREAD {
                            if let Ok(spread) = f32::try_from(&prop.value) {
                                spread_opt = Some(spread);
                            }
                        } else if prop.identifier == BULLETS_WEAPON {
                            if let Ok(wep) = i64::try_from(&prop.value) {
                                weapon_opt = Some(wep as u32);
                            }
                        }
                    }

                    if let Some(player_raw) = player_opt {
                        if let Some(sid) = resolve_player_sid(player_raw, state) {
                            let current_tick: u32 = tick.into();
                            self.fire_events
                                .entry(sid)
                                .or_default()
                                .push((current_tick, seed_opt, spread_opt, weapon_opt));
                        }
                    }
                }
            }
        }

        Ok(vec![])
    }

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        let ticknum = u32::from(state.tick);

        let min_shots_evaluated: i32 = get_parameter_value(&self.params, "min_shots_evaluated");
        let min_seed_correlation: f32 = get_parameter_value(&self.params, "min_seed_correlation");
        let spread_tolerance_margin: f32 =
            get_parameter_value(&self.params, "spread_tolerance_margin");
        let min_reversals = get_parameter_value::<i32>(&self.params, "min_reversals").max(1) as usize;

        let min_shots = (min_shots_evaluated.max(4)) as usize;

        for player in state.players.iter().filter(|p| {
            p.in_pvs
                && p.state == PlayerState::Alive
                && p.info.as_ref().is_some_and(|info| info.steam_id != "BOT")
        }) {
            let info = match &player.info {
                Some(info) => info,
                None => continue,
            };

            let steam_id: u64 = match SteamID::from_steam3(&info.steam_id) {
                Ok(sid) => u64::from(sid),
                Err(_) => continue,
            };

            let ticks_since_event = self
                .jg
                .teleported(&steam_id, ticknum)
                .min(self.jg.spawned(&steam_id, ticknum));

            if ticks_since_event < 60 {
                if ticks_since_event == 0 {
                    self.detections
                        .retain(|det| det.player != steam_id || ticknum.saturating_sub(det.tick) > 60);
                }
                self.prev_players.insert(steam_id, (player.pitch_angle, player.view_angle, ticknum));
                continue;
            }

            let prev_angles = match self.prev_players.get(&steam_id) {
                Some(&p) => p,
                None => {
                    self.prev_players.insert(steam_id, (player.pitch_angle, player.view_angle, ticknum));
                    continue;
                }
            };

            if ticknum == prev_angles.2 + 1 {
                let (va_delta, pa_delta) = viewangle_delta(
                    player.view_angle,
                    player.pitch_angle,
                    prev_angles.1,
                    prev_angles.0,
                    1,
                );
                let deltas = self.player_tick_deltas.entry(steam_id).or_default();
                // Only the deltas around the latest shot (at most 2 ticks old) are looked up.
                if deltas.len() > 64 {
                    deltas.retain(|&t, _| ticknum.saturating_sub(t) <= 8);
                }
                deltas.insert(ticknum, (pa_delta, va_delta));
            }

            self.prev_players.insert(steam_id, (player.pitch_angle, player.view_angle, ticknum));

            // Process bullet fire events for this player
            if let Some(events) = self.fire_events.get(&steam_id) {
                if let Some(&(f_tick, seed, spread, weapon_id)) = events.last() {
                    // Check if bullet was fired recently (within last 2 ticks)
                    if ticknum >= f_tick && ticknum <= f_tick + 2 {
                        let weapon_name = state.get_player_weapon(player);
                        // Shots from weapons without spread (e.g. sniper rifles) are not evaluated.
                        let Some(weapon_info) = get_weapon_spread_info(weapon_id, &weapon_name) else {
                            continue;
                        };

                        let tracker = self.player_trackers.entry(steam_id).or_default();
                        tracker.current_weapon = Some(weapon_info);

                        if let Some(deltas) = self.player_tick_deltas.get(&steam_id) {
                            // Check single-tick flick and return reversal at f_tick or f_tick+1
                            for t in f_tick..=f_tick + 1 {
                                if let (Some(&(dp1, dy1)), Some(&(dp2, dy2))) = (deltas.get(&t), deltas.get(&(t + 1))) {
                                    let mag1 = (dp1 * dp1 + dy1 * dy1).sqrt();
                                    let mag2 = (dp2 * dp2 + dy2 * dy2).sqrt();

                                    let dot = dp1 * dp2 + dy1 * dy2;
                                    let sum_mag = ((dp1 + dp2).powi(2) + (dy1 + dy2).powi(2)).sqrt();

                                    let max_cone = weapon_info.max_cone_degrees + spread_tolerance_margin;

                                    if mag1 >= 0.70 && mag2 >= 0.70 && mag1 <= max_cone * 2.0 && dot < 0.0 {
                                        let cos_angle = dot / (mag1 * mag2);
                                        let cancel_ratio = sum_mag / mag1.max(mag2);

                                        if cos_angle <= -0.65 && cancel_ratio <= 0.65 {
                                            // Confirmed micro-reversal
                                            tracker
                                                .reversals
                                                .retain(|r| ticknum.saturating_sub(r.tick) <= REVERSAL_WINDOW_TICKS);
                                            tracker.reversals.push(ReversalEvent {
                                                tick: t,
                                                magnitude: mag1,
                                            });

                                            tracker.shot_samples.push((t, (dy1, dp1), seed, spread));

                                            let recent_count = tracker
                                                .reversals
                                                .iter()
                                                .filter(|r| ticknum.saturating_sub(r.tick) <= REVERSAL_WINDOW_TICKS)
                                                .count();

                                            let mean_mag: f32 = tracker
                                                .reversals
                                                .iter()
                                                .filter(|r| ticknum.saturating_sub(r.tick) <= REVERSAL_WINDOW_TICKS)
                                                .map(|r| r.magnitude)
                                                .sum::<f32>()
                                                / recent_count.max(1) as f32;

                                            // Each detection stands on its own (no per-demo count), so the
                                            // algorithm also works on a part of a demo (live scanning).
                                            let should_flag = recent_count >= min_reversals
                                                && ticknum >= tracker.last_flag_tick + 60;
                                            if should_flag {
                                                tracker.last_flag_tick = ticknum;
                                                self.detections.push(Detection {
                                                    tick: ticknum,
                                                    algorithm: "fidoo/nospread".to_string(),
                                                    player: steam_id,
                                                    data: json!({
                                                        "type": "MicroJitterReversal",
                                                        "weapon_id": weapon_info.category_id,
                                                        "weapon_name": weapon_info.name,
                                                        "reversals_count": recent_count,
                                                        "mean_flick_magnitude": (mean_mag * 100.0).round() / 100.0,
                                                        "max_allowed_cone": (weapon_info.max_cone_degrees * 100.0).round() / 100.0,
                                                    }),
                                                });
                                            }

                                            // Check Method A deterministic seed correlation
                                            if let Some(det) = tracker.check_seed_correlation(steam_id, ticknum, min_shots, min_seed_correlation) {
                                                self.detections.push(det);
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

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        // Detections are held until the end because a respawn/teleport retracts the ones just before it.
        Ok(std::mem::take(&mut self.detections))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_finish_returns_every_detection() {
        // No minimum detection count per player: a single detection is reported.
        let mut algo = NoSpread::new();
        algo.detections.push(Detection {
            tick: 100,
            algorithm: "fidoo/nospread".to_string(),
            player: 11111,
            data: json!({"type": "MicroJitterReversal"}),
        });

        let finished = algo.finish().unwrap();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].player, 11111);
    }

    #[test]
    fn test_weapon_id_decides_spread_family() {
        // TF_WEAPON_SCATTERGUN (16) is not an SMG; TF_WEAPON_SMG is 19.
        assert_eq!(get_weapon_spread_info(Some(16), "SMG").map(|w| w.name), Some("Scattergun"));
        assert_eq!(get_weapon_spread_info(Some(19), "").map(|w| w.name), Some("SMG"));
        // Sniper rifles have no spread, whatever the held weapon name says.
        assert!(get_weapon_spread_info(Some(17), "SMG").is_none());
        assert!(get_weapon_spread_info(Some(22), "").is_none());
        assert_eq!(get_weapon_spread_info(None, "The Brass Beast").map(|w| w.name), Some("Minigun"));
    }

    #[test]
    fn test_predicted_spread_uses_low_seed_byte() {
        assert_eq!(compute_predicted_spread(42, 0.03), compute_predicted_spread(42 + 256, 0.03));
    }
}
