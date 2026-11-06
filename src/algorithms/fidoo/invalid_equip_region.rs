use std::collections::{HashMap, HashSet};
use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gameevent_gen::GameEvent;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::parser::analyser::Class;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::schema_equip_regions::{
    check_cosmetic_conflict, get_cosmetic_info, is_action_item, is_dummy_or_invalid,
    is_weapon_wearable, CosmeticInfo,
};

#[derive(Debug, Clone)]
struct WearableEntityState {
    owner: EntityId,
    item_def_index: u16,
    account_id: u32,
    is_disguise: bool,
    nodraw: bool,
    last_seen_tick: u32,
    spawn_generation: u32,
}

#[derive(Debug, Clone, Default)]
struct PlayerGenerationState {
    generation: u32,
    current_class: Class,
    conflict_ticks: u32,
    post_reset_conflict_ticks: u32,
    witnessed_in_pvs_reset: bool,
    reported_loadouts: HashSet<Vec<u16>>,
}

pub struct InvalidEquipRegion {
    pub params: Parameters,
    wearables: HashMap<EntityId, WearableEntityState>,
    players: HashMap<EntityId, PlayerGenerationState>,
    current_tick: u32,
    pub server_name: String,
}

impl Default for InvalidEquipRegion {
    fn default() -> Self {
        Self::new()
    }
}

impl InvalidEquipRegion {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_persistence_ticks".to_string(), Parameter::Int(600)),
                ("check_cosmetic_limit".to_string(), Parameter::Bool(true)),
                ("ignore_quickswitch".to_string(), Parameter::Bool(true)),
                ("valve_servers_only".to_string(), Parameter::Bool(true)),
                ("ignore_preset_swap".to_string(), Parameter::Bool(true)),
            ]),
            wearables: HashMap::new(),
            players: HashMap::new(),
            current_tick: 0,
            server_name: String::new(),
        }
    }

    fn advance_generation(&mut self, player_ent: EntityId, in_pvs_reset: bool) {
        let p = self.players.entry(player_ent).or_default();
        p.generation += 1;
        p.conflict_ticks = 0;
        p.post_reset_conflict_ticks = 0;
        p.witnessed_in_pvs_reset = in_pvs_reset;
    }
}

impl<'a> CheatAlgorithm<'a> for InvalidEquipRegion {
    fn default(&self) -> bool {
        true
    }

    fn algorithm_name(&self) -> &str {
        "fidoo/invalid_equip_region"
    }

    fn tooltip(&self) -> &'static str {
        "Detects impossible cosmetic loadouts with conflicting equip regions."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    // Tracks wearable entities from their creation.
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![
            MessageType::PacketEntities,
            MessageType::NetTick,
            MessageType::GameEvent,
            MessageType::ServerInfo,
            MessageType::SetConVar,
        ])
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        pstate: &ParserState,
        tick: DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        self.current_tick = u32::from(tick);

        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }

        match message {
            Message::ServerInfo(si) => {
                if self.server_name.is_empty() {
                    self.server_name = si.server_name.trim().to_string();
                }
            }
            Message::SetConVar(cvar) => {
                for cv in &cvar.vars {
                    if cv.key == "hostname" && self.server_name.is_empty() {
                        self.server_name = cv.value.trim().to_string();
                    }
                }
            }
            Message::GameEvent(event_msg) => match &event_msg.event {
                GameEvent::RoundStart(_) | GameEvent::TeamPlayRoundStart(_) => {
                    self.wearables.clear();
                    self.players.clear();
                }
                GameEvent::PlayerDeath(death) => {
                    let victim_uid = u32::from(death.user_id);
                    if let Some(player) = state.players.iter().find(|p| {
                        p.info
                            .as_ref()
                            .is_some_and(|info| u32::from(info.user_id) == victim_uid)
                    }) {
                        self.advance_generation(player.entity, false);
                    }
                }
                GameEvent::PlayerSpawn(spawn) => {
                    let spawn_uid = u32::from(spawn.user_id);
                    if let Some(player) = state.players.iter().find(|p| {
                        p.info
                            .as_ref()
                            .is_some_and(|info| u32::from(info.user_id) == spawn_uid)
                    }) {
                        let in_pvs = player.in_pvs;
                        self.advance_generation(player.entity, in_pvs);
                    }
                }
                GameEvent::PostInventoryApplication(inv) => {
                    let inv_uid = u32::from(inv.user_id);
                    if let Some(player) = state.players.iter().find(|p| {
                        p.info
                            .as_ref()
                            .is_some_and(|info| u32::from(info.user_id) == inv_uid)
                    }) {
                        let in_pvs = player.in_pvs;
                        let p = self.players.entry(player.entity).or_default();
                        p.conflict_ticks = 0;
                        p.post_reset_conflict_ticks = 0;
                        p.witnessed_in_pvs_reset = in_pvs;
                    }
                }
                _ => {}
            },
            Message::PacketEntities(msg) => {
                for removed in &msg.removed_entities {
                    self.wearables.remove(removed);
                }

                for entity in &msg.entities {
                    let class_name = pstate
                        .server_classes
                        .get(usize::from(entity.server_class))
                        .filter(|c| c.id == entity.server_class)
                        .or_else(|| pstate.server_classes.iter().find(|c| c.id == entity.server_class))
                        .map(|c| c.name.as_str())
                        .unwrap_or("");

                    if entity.update_type == UpdateType::Delete {
                        self.wearables.remove(&entity.entity_index);
                        continue;
                    }

                    if class_name.contains("Wearable") {
                        let mut owner_ent = None;
                        let mut item_def = None;
                        let mut is_disguise_prop = None;
                        let mut nodraw_prop = None;
                        let mut account_id_prop = None;

                        for prop in entity.props(pstate) {
                            if let Some((_table, name)) = prop.identifier.names() {
                                match name.as_str() {
                                    "m_hOwnerEntity" | "m_hOwner" => {
                                        if let Ok(val) = i64::try_from(&prop.value) {
                                            let handle = val as u32;
                                            let ent = EntityId::from(handle & 0x7FF);
                                            if u32::from(ent) != 0x7FF && u32::from(ent) != 0 {
                                                owner_ent = Some(ent);
                                            } else {
                                                owner_ent = Some(EntityId::from(0u32));
                                            }
                                        }
                                    }
                                    "m_iItemDefinitionIndex" => {
                                        if let Ok(val) = i64::try_from(&prop.value) {
                                            item_def = Some(val as u16);
                                        }
                                    }
                                    "m_iAccountID" => {
                                        if let Ok(val) = i64::try_from(&prop.value) {
                                            account_id_prop = Some(val as u32);
                                        }
                                    }
                                    "m_bDisguiseWearable" => {
                                        if let Ok(val) = i64::try_from(&prop.value) {
                                            is_disguise_prop = Some(val > 0);
                                        }
                                    }
                                    "m_fEffects" => {
                                        if let Ok(val) = i64::try_from(&prop.value) {
                                            nodraw_prop = Some((val & 32) != 0); // EF_NODRAW
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }

                        let tick_u32 = self.current_tick;
                        let w = self.wearables.entry(entity.entity_index).or_insert_with(|| {
                            WearableEntityState {
                                owner: EntityId::from(0u32),
                                item_def_index: 0,
                                account_id: 0,
                                is_disguise: false,
                                nodraw: false,
                                last_seen_tick: tick_u32,
                                spawn_generation: 0,
                            }
                        });

                        if let Some(owner) = owner_ent {
                            if u32::from(owner) == 0 {
                                w.owner = EntityId::from(0u32);
                            } else {
                                let p_state = self.players.entry(owner).or_default();
                                // A PVS re-enter of an entity we already track with the same owner and
                                // item is the same (possibly stale) wearable resent after the player was
                                // out of PVS, not a freshly equipped one: keep its original generation so
                                // leftovers from a previous class/loadout are not revived.
                                let is_pvs_reenter = entity.update_type == UpdateType::Enter
                                    && owner == w.owner
                                    && item_def == Some(w.item_def_index);
                                if owner != w.owner
                                    || (entity.update_type == UpdateType::Enter && !is_pvs_reenter)
                                {
                                    w.owner = owner;
                                    w.spawn_generation = p_state.generation;
                                }
                            }
                        }
                        if let Some(def) = item_def {
                            w.item_def_index = def;
                        }
                        if let Some(acc) = account_id_prop {
                            w.account_id = acc;
                        }
                        if let Some(disguise) = is_disguise_prop {
                            w.is_disguise = disguise;
                        }
                        if let Some(nd) = nodraw_prop {
                            w.nodraw = nd;
                        }
                        w.last_seen_tick = tick_u32;
                    }
                }
            }
            Message::NetTick(_) => {
                let min_persistence_ticks =
                    get_parameter_value::<i32>(&self.params, "min_persistence_ticks").max(1) as u32;
                let check_cosmetic_limit =
                    get_parameter_value::<bool>(&self.params, "check_cosmetic_limit");
                let ignore_quickswitch =
                    get_parameter_value::<bool>(&self.params, "ignore_quickswitch");
                let valve_servers_only =
                    get_parameter_value::<bool>(&self.params, "valve_servers_only");
                let ignore_preset_swap =
                    get_parameter_value::<bool>(&self.params, "ignore_preset_swap");
                let is_valve_server = self.server_name.contains("Valve Matchmaking Server");

                if valve_servers_only && !is_valve_server {
                    return Ok(vec![]);
                }

                let mut detections = Vec::new();

                for player in &state.players {
                    if player.state != PlayerState::Alive {
                        continue;
                    }

                    let steam_id64 = match &player.info {
                        Some(info) => {
                            if info.steam_id == "BOT" {
                                continue;
                            }
                            SteamID::from_steam3(&info.steam_id)
                                .map(u64::from)
                                .unwrap_or(0)
                        }
                        None => continue,
                    };

                    if steam_id64 == 0 {
                        continue;
                    }

                    let p_state = self.players.entry(player.entity).or_default();

                    // If player changed class, advance generation to invalidate previous class wearables
                    if p_state.current_class != Class::Other && p_state.current_class != player.class {
                        p_state.generation += 1;
                        p_state.conflict_ticks = 0;
                        p_state.post_reset_conflict_ticks = 0;
                        p_state.witnessed_in_pvs_reset = player.in_pvs;
                    }
                    p_state.current_class = player.class;

                    let target_gen = p_state.generation;

                    let player_account_id = (steam_id64 & 0xFFFFFFFF) as u32;

                    // Collect active cosmetic wearables for current spawn generation
                    let mut active_cosmetic_map: HashMap<u16, CosmeticInfo> = HashMap::new();
                    for w in self.wearables.values() {
                        if w.owner == player.entity
                            && w.spawn_generation == target_gen
                            && !w.nodraw
                            && !w.is_disguise
                            && (w.account_id == 0 || w.account_id == player_account_id)
                            && !is_dummy_or_invalid(w.item_def_index)
                            && !is_weapon_wearable(w.item_def_index)
                            && !is_action_item(w.item_def_index)
                        {
                            if let Some(info) = get_cosmetic_info(w.item_def_index) {
                                active_cosmetic_map.insert(w.item_def_index, info);
                            }
                        }
                    }

                    let mut active_items: Vec<(u16, CosmeticInfo)> =
                        active_cosmetic_map.into_iter().collect();
                    active_items.sort_by_key(|(id, _)| *id);
                    let loadout_key: Vec<u16> = active_items.iter().map(|(id, _)| *id).collect();

                    let mut has_conflict = false;
                    let mut conflicting_region_names: HashSet<&'static str> = HashSet::new();

                    // Check pairwise region conflicts
                    for i in 0..active_items.len() {
                        for j in (i + 1)..active_items.len() {
                            let (_, ref info_a) = active_items[i];
                            let (_, ref info_b) = active_items[j];

                            if let Some(conflicts) = check_cosmetic_conflict(info_a, info_b, ignore_quickswitch) {
                                has_conflict = true;
                                for rname in conflicts {
                                    conflicting_region_names.insert(rname);
                                }
                            }
                        }
                    }

                    let exceeds_limit = check_cosmetic_limit && active_items.len() > 3;
                    let is_violating = has_conflict || exceeds_limit;

                    let hat_cosmetics_count = active_items
                        .iter()
                        .filter(|(_, info)| info.regions.iter().any(|r| *r == "hat"))
                        .count();

                    let is_preset_swap_candidate = has_conflict
                        && active_items.len() <= 3
                        && conflicting_region_names.len() == 1
                        && conflicting_region_names.contains("hat")
                        && hat_cosmetics_count == 2;

                    if is_violating {
                        p_state.conflict_ticks += 1;
                        if player.in_pvs && p_state.witnessed_in_pvs_reset {
                            p_state.post_reset_conflict_ticks += 1;
                        }

                        let can_flag = if ignore_preset_swap && is_preset_swap_candidate {
                            p_state.witnessed_in_pvs_reset
                                && p_state.post_reset_conflict_ticks >= min_persistence_ticks
                        } else {
                            p_state.conflict_ticks >= min_persistence_ticks
                        };

                        if can_flag && !p_state.reported_loadouts.contains(&loadout_key) {
                            p_state.reported_loadouts.insert(loadout_key);

                            let duration_ticks = if ignore_preset_swap && is_preset_swap_candidate {
                                p_state.post_reset_conflict_ticks
                            } else {
                                p_state.conflict_ticks
                            };

                            let violation_type = if has_conflict && exceeds_limit {
                                "equip_region_conflict_and_excess_cosmetics"
                            } else if has_conflict {
                                "equip_region_conflict"
                            } else {
                                "excess_cosmetic_count"
                            };

                            let items_data: Vec<_> = active_items
                                .iter()
                                .map(|(id, info)| {
                                    json!({
                                        "id": id,
                                        "name": info.name,
                                        "regions": info.regions,
                                        "region_mask": format!("0x{:X}", info.region_mask),
                                    })
                                })
                                .collect();

                            let mut regions_vec: Vec<_> = conflicting_region_names.into_iter().collect();
                            regions_vec.sort();

                            detections.push(Detection {
                                tick: self.current_tick,
                                algorithm: "fidoo/invalid_equip_region".to_string(),
                                player: steam_id64,
                                data: json!({
                                    "class": player.class_name(),
                                    "violation_type": violation_type,
                                    "conflicting_regions": regions_vec,
                                    "total_cosmetics": active_items.len(),
                                    "cosmetics": items_data,
                                    "duration_ticks": duration_ticks,
                                    "valve_server": is_valve_server,
                                    "server_name": self.server_name.clone(),
                                }),
                            });
                        }
                    } else {
                        p_state.conflict_ticks = 0;
                        if player.in_pvs {
                            p_state.post_reset_conflict_ticks = 0;
                            p_state.witnessed_in_pvs_reset = false;
                        }
                    }
                }

                return Ok(detections);
            }
            _ => {}
        }

        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::cheat_analyser_base::Player;
    use tf_demo_parser::demo::message::NetTickMessage;

    #[test]
    fn test_valve_servers_only_filter_enabled_community() {
        let mut algo = InvalidEquipRegion::new();
        algo.server_name = "UGC.TF | Trade #11 | FREE ITEMS!".to_string();
        algo.params.insert("valve_servers_only".to_string(), Parameter::Bool(true));

        let state = CheatAnalyserState::default();
        let pstate = ParserState::new(24, |_| false, false);
        let msg = Message::NetTick(NetTickMessage {
            tick: 100.into(),
            frame_time: 15,
            std_dev: 0,
        });

        let res = algo.on_message(&msg, &state, &pstate, DemoTick::from(100)).unwrap();
        assert!(res.is_empty(), "Community server should be skipped when valve_servers_only is true");
    }

    #[test]
    fn test_valve_servers_only_filter_disabled_community() {
        let mut algo = InvalidEquipRegion::new();
        algo.server_name = "UGC.TF | Trade #11 | FREE ITEMS!".to_string();
        algo.params.insert("valve_servers_only".to_string(), Parameter::Bool(false));
        algo.params.insert("min_persistence_ticks".to_string(), Parameter::Int(1));
        algo.params.insert("ignore_preset_swap".to_string(), Parameter::Bool(false));

        let player_ent = EntityId::from(10u32);
        let mut state = CheatAnalyserState::default();
        let mut p = Player::default();
        p.entity = player_ent;
        p.state = PlayerState::Alive;
        p.class = Class::Heavy;
        let mut uinfo = tf_demo_parser::demo::data::UserInfo::default();
        uinfo.player_info.steam_id = "[U:1:1]".to_string();
        p.info = Some(uinfo.into());
        state.players.push(p);

        algo.wearables.insert(EntityId::from(101u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 378, // Team Captain (hat)
            account_id: 1,
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });
        algo.wearables.insert(EntityId::from(102u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 538, // Killer Exclusive (hat)
            account_id: 1,
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });

        let pstate = ParserState::new(24, |_| false, false);
        let msg = Message::NetTick(NetTickMessage {
            tick: 1.into(),
            frame_time: 15,
            std_dev: 0,
        });

        let res = algo.on_message(&msg, &state, &pstate, DemoTick::from(1)).unwrap();
        assert_eq!(res.len(), 1, "Community server should flag when valve_servers_only is false");
        assert_eq!(res[0].data["valve_server"], false);
        assert_eq!(res[0].data["server_name"], "UGC.TF | Trade #11 | FREE ITEMS!");
    }

    #[test]
    fn test_valve_servers_only_valve_server() {
        let mut algo = InvalidEquipRegion::new();
        algo.server_name = "Valve Matchmaking Server (Stockholm srcds2015-sto1 #69)".to_string();
        algo.params.insert("valve_servers_only".to_string(), Parameter::Bool(true));
        algo.params.insert("min_persistence_ticks".to_string(), Parameter::Int(1));
        algo.params.insert("ignore_preset_swap".to_string(), Parameter::Bool(false));

        let player_ent = EntityId::from(10u32);
        let mut state = CheatAnalyserState::default();
        let mut p = Player::default();
        p.entity = player_ent;
        p.state = PlayerState::Alive;
        p.class = Class::Heavy;
        let mut uinfo = tf_demo_parser::demo::data::UserInfo::default();
        uinfo.player_info.steam_id = "[U:1:1]".to_string();
        p.info = Some(uinfo.into());
        state.players.push(p);

        algo.wearables.insert(EntityId::from(101u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 378, // Team Captain (hat)
            account_id: 1,
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });
        algo.wearables.insert(EntityId::from(102u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 538, // Killer Exclusive (hat)
            account_id: 1,
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });

        let pstate = ParserState::new(24, |_| false, false);
        let msg = Message::NetTick(NetTickMessage {
            tick: 1.into(),
            frame_time: 15,
            std_dev: 0,
        });

        let res = algo.on_message(&msg, &state, &pstate, DemoTick::from(1)).unwrap();
        assert_eq!(res.len(), 1, "Valve server should flag when valve_servers_only is true");
        assert_eq!(res[0].data["valve_server"], true);
        assert_eq!(res[0].data["server_name"], "Valve Matchmaking Server (Stockholm srcds2015-sto1 #69)");
    }

    #[test]
    fn test_account_id_filtering() {
        let mut algo = InvalidEquipRegion::new();
        algo.server_name = "Valve Matchmaking Server (test)".to_string();
        algo.params.insert("min_persistence_ticks".to_string(), Parameter::Int(1));

        let player_ent = EntityId::from(13u32);
        let steam_id64 = 76561199232172054u64; // account_id = 1271906326
        let player_account_id = (steam_id64 & 0xFFFFFFFF) as u32;

        let mut state = CheatAnalyserState::default();
        let mut p = Player::default();
        p.entity = player_ent;
        p.state = PlayerState::Alive;
        p.class = Class::Pyro;
        let mut uinfo = tf_demo_parser::demo::data::UserInfo::default();
        uinfo.player_info.steam_id = "[U:1:1271906326]".to_string();
        p.info = Some(uinfo.into());
        state.players.push(p);

        // Wearable 1: owned by player, def 940 (Ghostly Gibus, hat)
        algo.wearables.insert(EntityId::from(101u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 940,
            account_id: player_account_id,
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });

        // Wearable 2: orphaned entity with different account_id (e.g. 345727541 from disconnected player), def 471 (hat)
        algo.wearables.insert(EntityId::from(102u32), WearableEntityState {
            owner: player_ent,
            item_def_index: 471,
            account_id: 345727541, // Mismatch!
            is_disguise: false,
            nodraw: false,
            last_seen_tick: 1,
            spawn_generation: 0,
        });

        let pstate = ParserState::new(24, |_| false, false);
        let msg = Message::NetTick(NetTickMessage {
            tick: 1.into(),
            frame_time: 15,
            std_dev: 0,
        });

        let res = algo.on_message(&msg, &state, &pstate, DemoTick::from(1)).unwrap();
        assert!(res.is_empty(), "Orphaned wearable with mismatching account_id must be ignored and not cause conflict");
    }
}

