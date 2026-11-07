// Written by Nocrex
// ignore grappling hook, halloween kart conds, and delay after respawn by SkiesRed

use std::{collections::{HashMap, HashSet}};

use crate::{
    base::cheat_analyser_base::{CheatAnalyserState, PlayerState, Player}
};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::{ParserState, demo::message::Message};

use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{Parameter, Parameters, get_parameter_value};

pub struct OOBPitch {
    last_detections: HashSet<String>,
    server_name: String,
    params: Parameters,
    exclude_until: HashMap<String, u32>,
    prev_alive: HashSet<String>,
}

impl OOBPitch {
    pub fn new() -> Self {
        let analyser: OOBPitch = OOBPitch {
            last_detections: HashSet::new(),
            server_name: "".to_string(),
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_pitch".to_string(), Parameter::Float(-89.999)),
                ("max_pitch".to_string(), Parameter::Float(89.999)),
                ("exclude_respawn_ticks".to_string(), Parameter::Int(67)),
            ]),
            exclude_until: HashMap::new(),
            prev_alive: HashSet::new(),
        };
        analyser
    }

    fn badconds(player: &Player) -> bool {
        let kart = player.cond_ex2 & (1 << 18) != 0;
        let zipl = player.cond_ex3 & (1 << 2) != 0;
        kart || zipl
    }
}

impl<'a> CheatAlgorithm<'a> for OOBPitch {
    fn default(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        "skiesred/oob_pitch"
    }

    fn tooltip(&self) -> &'static str {
        "Detects out-of-bounds pitch angles outside [-89°, +89°] (anti-aim / anti-backstab) that distort player hitboxes."
    }

    fn handled_messages(&self) -> Result<Vec<tf_demo_parser::MessageType>, bool> {
        Ok(vec![tf_demo_parser::MessageType::ServerInfo, tf_demo_parser::MessageType::NetTick])
    }

    fn on_message(&mut self,
        message: &Message,
        state: &CheatAnalyserState,
        _: &ParserState,
        _: tf_demo_parser::demo::data::DemoTick) -> Result<Vec<Detection>, Error> {
        let mut submitted_detections = Vec::new();

        if let Message::ServerInfo(event) = message {
            if self.server_name == "" {
                self.server_name = event.server_name.trim().to_string();
            }
        }
        
        if let Message::NetTick(_) = message {
            let ticknum = u32::from(state.tick);
            let players = &state.players;

            let mut detections = HashSet::new();
            let mut still_alive = HashSet::new(); //haha portal reference

            let min_pitch: f32 = get_parameter_value(&self.params, "min_pitch");
            let max_pitch: f32 = get_parameter_value(&self.params, "max_pitch");
            let exclude_respawn_ticks: u32 = get_parameter_value::<u64>(&self.params, "exclude_respawn_ticks").min(u32::MAX as u64) as u32;

            let is_valve_server = self.server_name.starts_with("Valve Matchmaking Server");

            for player in players.iter().filter(|p| {
                p.in_pvs
                    && p.state == PlayerState::Alive
                    && p.info.as_ref().is_some_and(|info| info.steam_id != "BOT")
            }) {
                let info = match &player.info {
                    Some(info) => info,
                    None => continue,
                };

                let steam_id = &info.steam_id;

                still_alive.insert(steam_id.clone());
                if !self.prev_alive.contains(steam_id) {
                    self.exclude_until
                        .insert(steam_id.clone(), ticknum + exclude_respawn_ticks);
                }
                if let Some(&until) = self.exclude_until.get(steam_id) {
                    if ticknum <= until {
                        continue;
                    }
                    else {
                        self.exclude_until.remove(steam_id);
                    }
                } //respawn checks
                if Self::badconds(player) {
                    continue;
                } //cond checks

                if !(min_pitch..=max_pitch).contains(&player.pitch_angle) {
                    detections.insert(steam_id.clone());
                    if !self.last_detections.contains(steam_id){
                        let class_name = player.class_name();
                        let weapon_name = state.get_player_weapon(player);
                        submitted_detections.push(Detection {
                            tick: ticknum,
                            algorithm: self.algorithm_name().to_string(),
                            player: u64::from(SteamID::from_steam3(&steam_id).unwrap()),
                            data: json!({
                                "class": class_name,
                                "weapon": weapon_name,
                                "pitch": player.pitch_angle,
                                "valve_server": is_valve_server
                            }),
                        });
                    }
                }
            } //og

            self.last_detections = detections;
            self.prev_alive = still_alive;

        }

        Ok(submitted_detections)

    }
    
    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }
}
