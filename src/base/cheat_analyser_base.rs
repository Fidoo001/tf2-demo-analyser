// This file is a modified version of GameStateAnalyser.
// https://github.com/demostf/parser/blob/master/src/demo/parser/gamestateanalyser.rs
// TODO: This version will add support for sub analysers that can be used to extend functionality as needed
// without creating an entirely separate analyser.
// Additional functionality that has broad utility can be merged into this base analyser.

use anyhow::Error;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::hash::{BuildHasherDefault, Hasher};
use std::str::FromStr;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gameevent_gen::ObjectDestroyedEvent;
use tf_demo_parser::demo::gamevent::GameEvent;
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::gameevent::GameEventMessage;
use tf_demo_parser::demo::message::packetentities::{EntityId, PacketEntity, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::datatable::{ParseSendTable, ServerClass, ServerClassName};
use tf_demo_parser::demo::packet::message::MessagePacketMeta;
use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
use tf_demo_parser::demo::parser::analyser::UserInfo;
pub use tf_demo_parser::demo::parser::analyser::{Class, Team, UserId};
use tf_demo_parser::demo::parser::handler::BorrowMessageHandler;
use tf_demo_parser::demo::parser::MessageHandler;
use tf_demo_parser::demo::sendprop::{SendProp, SendPropIdentifier, SendPropValue};
use tf_demo_parser::demo::vector::{Vector, VectorXY};
use tf_demo_parser::{MessageType, ParserState, ReadResult, Stream};
use web_time::Instant;

use crate::lib::algorithm::{CheatAlgorithm, Detection, PacketInfo};
use crate::lib::algorithm_costs;
use crate::util::helpers::reconstruct_simtick;
use crate::dev_print;

/// Stores new algorithm detections, first forwarding the flagged players to the live
/// detection listener (set by the GUI only).
fn record_detections(all: &mut Vec<Detection>, new: Vec<Detection>) {
    if !new.is_empty() {
        crate::notify_detections(&new);
    }
    all.extend(new);
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum PlayerState {
    Alive = 0,
    Dying = 1,
    Death = 2,
    #[default]
    Respawnable = 3,
}

impl PlayerState {
    pub fn new(number: i64) -> Self {
        match number {
            0 => PlayerState::Alive,
            1 => PlayerState::Dying,
            2 => PlayerState::Death,
            3 => PlayerState::Respawnable,
            _ => PlayerState::Respawnable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Player {
    pub entity: EntityId,
    pub position: Vector,
    pub health: u16,
    pub max_health: u16,
    pub class: Class,
    pub team: Team,
    pub view_angle: f32,
    pub pitch_angle: f32,
    pub state: PlayerState,
    pub info: Option<UserInfo>,
    pub charge: u8,
    /// Raw 8-bit network value of m_flSimulationTime, not a tick counter. Decode it with
    /// `CheatAnalyserState::sim_tick`.
    pub simtime: u16,
    pub ping: u16,
    pub in_pvs: bool,
    pub active_weapon: Option<EntityId>,
    pub cond: u32,
    pub cond_ex: u32,
    pub cond_ex2: u32,
    pub cond_ex3: u32,
    pub invis_change_complete_time: f32,
    pub flags: u32,
    // pub shot_fired: u32,
}

impl Player {
    pub fn is_on_ground(&self) -> bool {
        (self.flags & 1) != 0
    }

    pub fn is_ducking(&self) -> bool {
        (self.flags & (1 << 1)) != 0
    }

    pub fn is_in_water(&self) -> bool {
        (self.flags & ((1 << 9) | (1 << 10))) != 0
    }

    pub fn class_name(&self) -> &'static str {
        match self.class {
            Class::Scout => "scout",
            Class::Sniper => "sniper",
            Class::Soldier => "soldier",
            Class::Demoman => "demoman",
            Class::Medic => "medic",
            Class::Heavy => "heavy",
            Class::Pyro => "pyro",
            Class::Spy => "spy",
            Class::Engineer => "engineer",
            Class::Other => "unknown",
        }
    }

    pub fn is_cloaked(&self) -> bool {
        (self.cond & (1 << 4)) != 0
    }

    pub fn is_burning(&self) -> bool {
        (self.cond & (1 << 22)) != 0
    }

    pub fn is_jarated(&self) -> bool {
        (self.cond & (1 << 24)) != 0
    }

    pub fn is_bleeding(&self) -> bool {
        (self.cond & (1 << 25)) != 0
    }

    pub fn is_milked(&self) -> bool {
        (self.cond & (1 << 27)) != 0
    }

    pub fn is_flickering(&self) -> bool {
        (self.cond_ex & (1 << (61 - 32))) != 0
    }

    pub fn is_gas_coated(&self) -> bool {
        (self.cond_ex2 & (1 << (73 - 64))) != 0
    }

    pub fn is_stealthed_blink(&self) -> bool {
        (self.cond_ex2 & (1 << (70 - 64))) != 0
    }

    pub fn is_taunting(&self) -> bool {
        (self.cond & (1 << 14)) != 0
    }

    pub fn is_stunned(&self) -> bool {
        (self.cond & (1 << 15)) != 0
    }

    pub fn is_marked_for_death(&self) -> bool {
        (self.cond & (1 << 30)) != 0
    }

    pub fn has_visible_effect(&self) -> bool {
        self.is_burning()
            || self.is_jarated()
            || self.is_bleeding()
            || self.is_milked()
            || self.is_flickering()
            || self.is_gas_coated()
            || self.is_stealthed_blink()
            || self.is_taunting()
            || self.is_stunned()
            || self.is_marked_for_death()
    }

    pub fn is_completely_invisible(&self) -> bool {
        self.is_cloaked() && !self.has_visible_effect()
    }

    pub fn is_blastjumping(&self) -> bool {
        (self.cond_ex2 & (1 << (81 - 64))) != 0
    }

    pub fn is_knocked_into_air(&self) -> bool {
        (self.cond_ex3 & (1 << (115 - 96))) != 0
    }

    pub fn has_special_movement_cond(&self) -> bool {
        // Halloween & movement override conditions:
        // Bomb head (53), Thriller (54)
        let bomb_head = (self.cond_ex & (1 << (53 - 32))) != 0;
        let thriller = (self.cond_ex & (1 << (54 - 32))) != 0;
        // Tiny (75 - infinite jumps), Ghost (77 - flight), Parachute (80),
        // Kart (82, 83, 88)
        let tiny = (self.cond_ex2 & (1 << (75 - 64))) != 0;
        let ghost = (self.cond_ex2 & (1 << (77 - 64))) != 0;
        let parachute = (self.cond_ex2 & (1 << (80 - 64))) != 0
            || (self.cond_ex3 & (1 << (122 - 96))) != 0;
        let kart = (self.cond_ex2 & ((1 << (82 - 64)) | (1 << (83 - 64)) | (1 << (88 - 64)))) != 0;

        bomb_head || thriller || tiny || ghost || parachute || kart || self.is_knocked_into_air()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Sentry {
    pub entity: EntityId,
    pub builder: UserId,
    pub position: Vector,
    pub level: u8,
    pub max_health: u16,
    pub health: u16,
    pub building: bool,
    pub sapped: bool,
    pub team: Team,
    pub angle: f32,
    pub player_controlled: bool,
    pub auto_aim_target: UserId,
    pub shells: u16,
    pub rockets: u16,
    pub is_mini: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Dispenser {
    pub entity: EntityId,
    pub builder: UserId,
    pub position: Vector,
    pub level: u8,
    pub max_health: u16,
    pub health: u16,
    pub building: bool,
    pub sapped: bool,
    pub team: Team,
    pub angle: f32,
    pub healing: Vec<UserId>,
    pub metal: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Teleporter {
    pub entity: EntityId,
    pub builder: UserId,
    pub position: Vector,
    pub level: u8,
    pub max_health: u16,
    pub health: u16,
    pub building: bool,
    pub sapped: bool,
    pub team: Team,
    pub angle: f32,
    pub is_entrance: bool,
    pub other_end: EntityId,
    pub recharge_time: f32,
    pub recharge_duration: f32,
    pub times_used: u16,
    pub yaw_to_exit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Building {
    Sentry(Sentry),
    Dispenser(Dispenser),
    Teleporter(Teleporter),
}

impl Building {
    pub fn new(entity_id: EntityId, class: BuildingClass) -> Building {
        match class {
            BuildingClass::Sentry => Building::Sentry(Sentry {
                entity: entity_id,
                ..Sentry::default()
            }),
            BuildingClass::Dispenser => Building::Dispenser(Dispenser {
                entity: entity_id,
                ..Dispenser::default()
            }),
            BuildingClass::Teleporter => Building::Teleporter(Teleporter {
                entity: entity_id,
                ..Teleporter::default()
            }),
        }
    }

    pub fn entity_id(&self) -> EntityId {
        match self {
            Building::Sentry(Sentry { entity, .. })
            | Building::Dispenser(Dispenser { entity, .. })
            | Building::Teleporter(Teleporter { entity, .. }) => *entity,
        }
    }

    pub fn level(&self) -> u8 {
        match self {
            Building::Sentry(Sentry { level, .. })
            | Building::Dispenser(Dispenser { level, .. })
            | Building::Teleporter(Teleporter { level, .. }) => *level,
        }
    }

    pub fn position(&self) -> Vector {
        match self {
            Building::Sentry(Sentry { position, .. })
            | Building::Dispenser(Dispenser { position, .. })
            | Building::Teleporter(Teleporter { position, .. }) => *position,
        }
    }

    pub fn builder(&self) -> UserId {
        match self {
            Building::Sentry(Sentry { builder, .. })
            | Building::Dispenser(Dispenser { builder, .. })
            | Building::Teleporter(Teleporter { builder, .. }) => *builder,
        }
    }

    pub fn angle(&self) -> f32 {
        match self {
            Building::Sentry(Sentry { angle, .. })
            | Building::Dispenser(Dispenser { angle, .. })
            | Building::Teleporter(Teleporter { angle, .. }) => *angle,
        }
    }

    pub fn max_health(&self) -> u16 {
        match self {
            Building::Sentry(Sentry { max_health, .. })
            | Building::Dispenser(Dispenser { max_health, .. })
            | Building::Teleporter(Teleporter { max_health, .. }) => *max_health,
        }
    }

    pub fn health(&self) -> u16 {
        match self {
            Building::Sentry(Sentry { health, .. })
            | Building::Dispenser(Dispenser { health, .. })
            | Building::Teleporter(Teleporter { health, .. }) => *health,
        }
    }

    pub fn sapped(&self) -> bool {
        match self {
            Building::Sentry(Sentry { sapped, .. })
            | Building::Dispenser(Dispenser { sapped, .. })
            | Building::Teleporter(Teleporter { sapped, .. }) => *sapped,
        }
    }

    pub fn team(&self) -> Team {
        match self {
            Building::Sentry(Sentry { team, .. })
            | Building::Dispenser(Dispenser { team, .. })
            | Building::Teleporter(Teleporter { team, .. }) => *team,
        }
    }

    pub fn class(&self) -> BuildingClass {
        match self {
            Building::Sentry(_) => BuildingClass::Sentry,
            Building::Dispenser(_) => BuildingClass::Sentry,
            Building::Teleporter(_) => BuildingClass::Teleporter,
        }
    }
}

pub enum BuildingClass {
    Sentry,
    Dispenser,
    Teleporter,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct World {
    pub boundary_min: Vector,
    pub boundary_max: Vector,
}

// #[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
// pub struct Kill {
//     pub attacker_id: u16,
//     pub assister_id: u16,
//     pub victim_id: u16,
//     pub weapon: String,
//     pub tick: DemoTick,
// }

// impl Kill {
//     fn new(tick: DemoTick, death: &PlayerDeathEvent) -> Self {
//         Kill {
//             attacker_id: death.attacker,
//             assister_id: death.assister,
//             victim_id: death.user_id,
//             weapon: death.weapon.to_string(),
//             tick,
//         }
//     }
// }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WeaponEntity {
    pub entity: EntityId,
    pub class_name: String,
    pub item_def_index: Option<u16>,
}

impl WeaponEntity {
    pub fn name(&self, player_class: Class) -> String {
        crate::util::helpers::weapon_name_from_id_or_class(self.item_def_index, &self.class_name, player_class)
    }
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct CheatAnalyserState {
    pub players: Vec<Player>,
    pub player_names: HashMap<u64, String>,
    pub user_info_history: HashMap<u64, UserInfo>,
    pub entid_to_userid: HashMap<EntityId, UserId>,
    pub userid_to_id64: HashMap<UserId, u64>,
    pub buildings: BTreeMap<EntityId, Building>,
    pub weapons: HashMap<EntityId, WeaponEntity>,
    pub world: Option<World>,
    pub friends_ids: HashMap<EntityId, u32>,
    pub custom_files: HashMap<EntityId, [u32; 4]>,
    // pub kills: Vec<Kill>,
    pub tick: DemoTick,
    /// Server tick of the latest NetTick, the snapshot the current entity state belongs to.
    #[serde(default)]
    pub server_tick: u32,
    /// Decoded simulation tick per player entity, see `sim_tick`.
    #[serde(skip)]
    pub sim_ticks: HashMap<EntityId, i64>,
    /// Player entities whose simtime was received since the last NetTick.
    #[serde(skip)]
    pub simtime_received: Vec<EntityId>,
    pub header: Option<Header>,
    pub user_cmds: Vec<tf_demo_parser::demo::packet::usercmd::UserCmdPacket>,
    pub console_cmds: Vec<tf_demo_parser::demo::packet::consolecmd::ConsoleCmdPacket>,
}

/// A simulation tick can't run this far ahead of the server tick (the tickbase leads it by a few
/// ticks). Some servers encode simtimes against a tick slightly older than the NetTick, so right
/// after a network base step the value decodes ~100 ticks ahead; it belongs to the previous base.
const MAX_SIM_TICK_LEAD: i64 = 50;

fn decode_simtick(server_tick: u32, player: &Player) -> i64 {
    let server_tick = server_tick as i64;
    let sim = reconstruct_simtick(server_tick, u32::from(player.entity) as i64, player.simtime as i64);
    if sim - server_tick > MAX_SIM_TICK_LEAD {
        sim - 100
    } else {
        sim
    }
}

impl CheatAnalyserState {
    /// Full simulation tick of a player, decoded from the 8-bit networked simtime when it was received.
    pub fn sim_tick(&self, player: &Player) -> i64 {
        match self.sim_ticks.get(&player.entity) {
            Some(&sim) => sim,
            None => decode_simtick(self.server_tick, player),
        }
    }

    /// Decodes the simtimes received in this packet. Like the engine, a value is decoded once on
    /// receipt: re-decoding an unchanged value later would shift it by 100 at every base step.
    fn decode_received_simtimes(&mut self) {
        for entity in std::mem::take(&mut self.simtime_received) {
            if let Some(player) = self.players.iter().find(|p| p.entity == entity) {
                self.sim_ticks.insert(entity, decode_simtick(self.server_tick, player));
            }
        }
    }

    pub fn get_player_name(&self, sid: u64) -> Option<&str> {
        if let Some(name) = self.player_names.get(&sid) {
            return Some(name.as_str());
        }
        for player in &self.players {
            if let Some(info) = &player.info {
                if let Ok(steam_id) = SteamID::from_steam3(&info.steam_id) {
                    if u64::from(steam_id) == sid {
                        return Some(info.name.as_str());
                    }
                }
            }
        }
        None
    }

    pub fn get_player_slot(&self, sid: u64) -> Option<u32> {
        if let Some(user_info) = self.user_info_history.get(&sid) {
            let ent = u32::from(user_info.entity_id);
            if ent > 0 {
                return Some(ent - 1);
            }
        }
        for player in &self.players {
            if let Some(info) = &player.info {
                if let Ok(steam_id) = SteamID::from_steam3(&info.steam_id) {
                    if u64::from(steam_id) == sid {
                        let ent = u32::from(player.entity);
                        if ent > 0 {
                            return Some(ent - 1);
                        }
                    }
                }
            }
        }
        for (ent, uid) in &self.entid_to_userid {
            if let Some(&id64) = self.userid_to_id64.get(uid) {
                if id64 == sid {
                    let ent_num = u32::from(*ent);
                    if ent_num > 0 {
                        return Some(ent_num - 1);
                    }
                }
            }
        }
        None
    }

    pub fn get_player_weapon(&self, player: &Player) -> String {
        if let Some(weapon_ent) = player.active_weapon {
            if let Some(weapon) = self.weapons.get(&weapon_ent) {
                return weapon.name(player.class);
            }
        }
        "unknown".to_string()
    }

    pub fn get_player_class_and_weapon_by_sid(&self, sid: u64) -> (&'static str, String) {
        if let Some(player) = self.players.iter().find(|p| {
            p.info.as_ref().is_some_and(|info| {
                SteamID::from_steam3(&info.steam_id).map(u64::from).ok() == Some(sid)
            })
        }) {
            (player.class_name(), self.get_player_weapon(player))
        } else {
            ("unknown", "unknown".to_string())
        }
    }

    pub fn get_or_create_player(&mut self, entity_id: EntityId) -> &mut Player {
        let index = match self
            .players
            .iter()
            .enumerate()
            .find(|(_index, player)| player.entity == entity_id)
            .map(|(index, _)| index)
        {
            Some(index) => index,
            None => {
                let index = self.players.len();
                self.players.push(Player {
                    entity: entity_id,
                    ..Player::default()
                });
                index
            }
        };

        #[allow(clippy::indexing_slicing)]
        &mut self.players[index]
    }
    pub fn get_userid_from_entid(&self, entid: EntityId) -> Option<UserId> {
        self.entid_to_userid.get(&entid).copied()
    }

    pub fn get_id64_from_userid(&self, userid: UserId) -> Option<u64> {
        self.userid_to_id64.get(&userid).copied()
    }

    pub fn set_entid_to_userid(&mut self, entid: EntityId, userid: UserId) {
        self.entid_to_userid.insert(entid, userid);
    }

    pub fn set_userid_to_id64(&mut self, userid: UserId, id64: u64) {
        self.userid_to_id64.insert(userid, id64);
    }

    pub fn get_or_create_building(
        &mut self,
        entity_id: EntityId,
        class: BuildingClass,
    ) -> &mut Building {
        self.buildings
            .entry(entity_id)
            .or_insert_with(|| Building::new(entity_id, class))
    }

    pub fn remove_building(&mut self, entity_id: EntityId) {
        self.buildings.remove(&entity_id);
    }
}

// ParserState requires a non-self impl of does_handle, so the message types handled by the
// analyser are kept per thread (each parallel worker parses the demo on its own thread).
// Bit N is set when MessageType N is handled. Threads that never created an analyser parse all.
thread_local! {
    static HANDLED_MESSAGE_MASK: Cell<u64> = const { Cell::new(u64::MAX) };
}

// Message types the analyser itself needs to build CheatAnalyserState.
const BASE_MESSAGE_TYPES: [MessageType; 3] = [
    MessageType::PacketEntities,
    MessageType::GameEvent,
    MessageType::NetTick,
];

/// With a start tick set, algorithm-only message types are parsed again this many ticks early,
/// so the packets around the start tick are never parsed with the reduced filter.
const FILTER_LOOKAHEAD_TICKS: u32 = 4;

fn message_mask(types: &[MessageType]) -> u64 {
    types.iter().fold(0, |mask, message_type| mask | (1u64 << (*message_type as u8)))
}

fn mask_contains(mask: u64, message_type: MessageType) -> bool {
    (mask >> (message_type as u8)) & 1 != 0
}


// SendPropIdentifiers are already hashes, so they are used as hash map keys directly.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(*byte);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

/// Which class-specific handler an entity's server class uses.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum EntityHandler {
    #[default]
    None,
    Player,
    PlayerResource,
    World,
    Sentry,
    Dispenser,
    Teleporter,
}

/// Per-ServerClass information, computed once when the data tables are received.
#[derive(Debug, Clone, Copy, Default)]
struct ClassKind {
    is_weapon: bool,
    handler: EntityHandler,
}

impl ClassKind {
    fn new(class_name: &str) -> Self {
        let handler = match class_name {
            "CTFPlayer" => EntityHandler::Player,
            "CTFPlayerResource" => EntityHandler::PlayerResource,
            "CWorld" => EntityHandler::World,
            "CObjectSentrygun" => EntityHandler::Sentry,
            "CObjectDispenser" => EntityHandler::Dispenser,
            "CObjectTeleporter" => EntityHandler::Teleporter,
            _ => EntityHandler::None,
        };
        ClassKind {
            is_weapon: is_weapon_class(class_name),
            handler,
        }
    }
}

fn is_weapon_class(class_name: &str) -> bool {
    const WEAPON_PREFIXES: [&str; 27] = [
        "CTFWeapon",
        "CTFShotgun",
        "CTFRocketLauncher",
        "CTFPipebombLauncher",
        "CTFGrenadeLauncher",
        "CTFCompoundBow",
        "CTFMinigun",
        "CTFScatterGun",
        "CTFKnife",
        "CTFRevolver",
        "CTFMedigun",
        "CTFBonesaw",
        "CTFClub",
        "CTFSword",
        "CTFWrench",
        "CTFFlameThrower",
        "CTFPistol",
        "CTFJar",
        "CTFBuffItem",
        "CTFLunchBox",
        "CTFCrossbow",
        "CTFStickBomb",
        "CTFParticleCannon",
        "CTFRaygun",
        "CTFDRG_Pomson",
        "CTFSniperRifle",
        "CWeapon",
    ];
    WEAPON_PREFIXES.iter().any(|prefix| class_name.starts_with(prefix))
        || (class_name.starts_with("CTF") && class_name.contains("Weapon"))
}

/// CTFPlayer fields matched by prop name, regardless of the prop's table.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum PlayerPropField {
    #[default]
    None,
    ActiveWeapon,
    Cond,
    CondEx,
    CondEx2,
    CondEx3,
    InvisChangeCompleteTime,
    Flags,
    EyeAnglesYaw,
    EyeAnglesPitch,
}

/// CTFPlayerResource array fields (the table name is the array, the prop name the player index).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResourceField {
    Team,
    MaxHealth,
    Class,
    Charge,
    Ping,
    Other,
}

/// What the base analyser does with a send prop, derived from its compile-time names
/// (`SendPropIdentifier::names`) once per identifier instead of on every prop update.
#[derive(Debug, Clone, Copy, Default)]
struct PropInfo {
    item_definition_index: bool,
    player_field: PlayerPropField,
    resource: Option<(u32, ResourceField)>,
}

impl PropInfo {
    fn new(identifier: SendPropIdentifier) -> Self {
        let Some((table_name, prop_name)) = identifier.names() else {
            return PropInfo::default();
        };
        let player_field = match prop_name.as_str() {
            "m_hActiveWeapon" => PlayerPropField::ActiveWeapon,
            "m_nPlayerCond" | "_condition_bits" => PlayerPropField::Cond,
            "m_nPlayerCondEx" => PlayerPropField::CondEx,
            "m_nPlayerCondEx2" => PlayerPropField::CondEx2,
            "m_nPlayerCondEx3" => PlayerPropField::CondEx3,
            "m_flInvisChangeCompleteTime" => PlayerPropField::InvisChangeCompleteTime,
            "m_fFlags" => PlayerPropField::Flags,
            "m_angEyeAngles[1]" => PlayerPropField::EyeAnglesYaw,
            "m_angEyeAngles[0]" => PlayerPropField::EyeAnglesPitch,
            _ => PlayerPropField::None,
        };
        let resource = u32::from_str(prop_name.as_str()).ok().map(|player_id| {
            let field = match table_name.as_str() {
                "m_iTeam" => ResourceField::Team,
                "m_iMaxHealth" => ResourceField::MaxHealth,
                "m_iPlayerClass" => ResourceField::Class,
                "m_iChargeLevel" => ResourceField::Charge,
                "m_iPing" => ResourceField::Ping,
                _ => ResourceField::Other,
            };
            (player_id, field)
        });
        PropInfo {
            item_definition_index: prop_name == "m_iItemDefinitionIndex",
            player_field,
            resource,
        }
    }
}

type PropInfoCache = HashMap<SendPropIdentifier, PropInfo, BuildHasherDefault<IdentityHasher>>;

fn prop_info(cache: &mut PropInfoCache, identifier: SendPropIdentifier) -> PropInfo {
    *cache
        .entry(identifier)
        .or_insert_with(|| PropInfo::new(identifier))
}

pub struct CheatAnalyser<'a> {
    pub state: CheatAnalyserState,
    pub algorithms: Vec<Box<dyn CheatAlgorithm<'a> + 'a + Send>>,
    pub detections: Vec<Detection>,
    pub header: Option<Header>,
    pub tick: DemoTick,
    pub worker_id: usize,
    /// Set when the demo could only be analysed up to a point: the packet stream hit a parse error, or
    /// the file ends unexpectedly (a recording cut short by a crash). Everything before it was analysed.
    pub parse_error: Option<String>,
    last_progress_update_time: Instant,
    progress: Vec<u32>,
    class_names: Vec<ServerClassName>, // indexed by ClassId
    class_kinds: Vec<ClassKind>,       // indexed by ClassId
    prop_infos: PropInfoCache,
    handled_message_mask: u64,
    algorithm_message_masks: Vec<u64>, // indexed like `algorithms`
    // Cost learning: algorithms are timed on one out of every SAMPLE_INTERVAL ticks.
    cost_sampling: bool,
    cost_sample_ticks: u32,
    algorithm_sampled_nanos: Vec<u64>, // indexed like `algorithms`
    algorithm_finish_nanos: Vec<u64>,  // indexed like `algorithms`
    // The live scanner only wants the detections from `analyse_from_tick` on. Each algorithm
    // receives ticks, packets and messages (plus the signon data) from its own start tick: its
    // `live_warmup_ticks()` before `analyse_from_tick`, so that it has the history it needs; the
    // algorithms that need the whole demo (`whole_demo`) start at 0. The game state is still built
    // from the whole demo.
    analyse_from_tick: u32,
    in_signon: bool,
    whole_demo: Vec<bool>,  // indexed like `algorithms`
    start_ticks: Vec<u32>,  // indexed like `algorithms`
    running: Vec<bool>,     // indexed like `algorithms`: the start tick is reached
    next_start_tick: u32,   // earliest start tick of the algorithms that are not running yet
    base_message_mask: u64,
}

impl<'a> Default for CheatAnalyser<'a> {
    fn default() -> Self {
        Self {
            state: Default::default(),
            algorithms: Default::default(),
            detections: Default::default(),
            header: Default::default(),
            tick: Default::default(),
            worker_id: 0,
            parse_error: None,
            last_progress_update_time: Instant::now(),
            progress: Default::default(),
            class_names: Default::default(),
            class_kinds: Default::default(),
            prop_infos: Default::default(),
            handled_message_mask: message_mask(&BASE_MESSAGE_TYPES),
            algorithm_message_masks: Default::default(),
            cost_sampling: false,
            cost_sample_ticks: 0,
            algorithm_sampled_nanos: Default::default(),
            algorithm_finish_nanos: Default::default(),
            analyse_from_tick: 0,
            in_signon: true,
            whole_demo: Default::default(),
            start_ticks: Default::default(),
            running: Default::default(),
            next_start_tick: u32::MAX,
            base_message_mask: message_mask(&BASE_MESSAGE_TYPES),
        }
    }
}

impl MessageHandler for CheatAnalyser<'_> {
    type Output = CheatAnalyserState;

    fn does_handle(message_type: MessageType) -> bool {
        HANDLED_MESSAGE_MASK.with(|mask| mask_contains(mask.get(), message_type))
    }

    fn handle_header(&mut self, _header: &tf_demo_parser::demo::header::Header) {
        self.header = Some(_header.clone());
        self.state.header = Some(_header.clone());
        if _header.ticks > 0 {
            crate::PROGRESS_TOTAL.store(_header.ticks, std::sync::atomic::Ordering::Relaxed);
        }
        if self.worker_id == 0 {
            self.print_metadata();
        }
    }

    fn handle_message(&mut self, message: &Message, _tick: DemoTick, parser_state: &ParserState) {
        match message {
            Message::PacketEntities(message) => {
                for removed in &message.removed_entities {
                    if let Some(player) = self.state.players.iter_mut().find(|p| p.entity == *removed) {
                        player.in_pvs = false;
                    }
                    self.state.weapons.remove(removed);
                    self.state.remove_building(*removed);
                }
                for entity in &message.entities {
                    self.handle_entity(entity, parser_state);
                }
            }
            Message::NetTick(net_tick) => {
                self.state.server_tick = u32::from(net_tick.tick);
                self.state.decode_received_simtimes();
                self.check_progress();
                self.update_running();
                for (index, algorithm) in self.algorithms.iter_mut().enumerate() {
                    if !self.in_signon && !self.running[index] {
                        continue;
                    }
                    let start = self.cost_sampling.then(Instant::now);
                    match algorithm.on_tick(&self.state, parser_state) {
                        Ok(detections) => record_detections(&mut self.detections, detections),
                        Err(_) => {}
                    }
                    if let Some(start) = start {
                        self.algorithm_sampled_nanos[index] += start.elapsed().as_nanos() as u64;
                    }
                }
                self.state.user_cmds.clear();
                self.state.console_cmds.clear();
                if algorithm_costs::COST_LEARNING {
                    self.cost_sample_ticks = self.cost_sample_ticks.wrapping_add(1);
                    self.cost_sampling = self.cost_sample_ticks % algorithm_costs::SAMPLE_INTERVAL == 0;
                }
            }
            Message::TempEntities(_) => {
                // println!("{}: {:#?}", _tick, message);
            }
            Message::GameEvent(GameEventMessage { event, .. }) => match event {
                // GameEvent::PlayerDeath(death) => {
                //     self.state.kills.push(Kill::new(self.tick, death.as_ref()))
                // }
                // TODO: Wait for https://github.com/demostf/parser/issues/25 to be resolved
                // GameEvent::PlayerShoot(_) => {
                //     println!("player shoot event");
                //     // let player = self.state.players.iter_mut().find(|p|{
                //     //     p.info.as_ref().is_some_and(|info| {
                //     //         println!("{} == {}", info.user_id, user_id);
                //     //         info.user_id == *user_id
                //     //     })
                //     // });
                //     // if let Some(player) = player {
                //     //     player.shot_fired = u32::from(_tick);
                //     // }
                // }
                GameEvent::RoundStart(_) => {
                    self.state.buildings.clear();
                }
                GameEvent::TeamPlayRoundStart(_) => {
                    self.state.buildings.clear();
                }
                GameEvent::ObjectDestroyed(ObjectDestroyedEvent { index, .. }) => {
                    self.state.remove_building((*index as u32).into());
                }
                GameEvent::PlayerConnectClient(event) => {
                    self.state.set_entid_to_userid(
                        EntityId::from(event.index as u32),
                        UserId::from(event.user_id),
                    );
                    if event.network_id != "BOT".into() {
                        let steamid = SteamID::from_steam3(event.network_id.to_string().as_str());
                        let steamid64 = u64::from(steamid.unwrap_or(0.into()));
                        self.state
                            .set_userid_to_id64(event.user_id.into(), steamid64);
                    }
                }
                _ => {}
            },
            _ => {}
        }
        self.update_running();
        let message_type = message.get_message_type();
        for (index, algorithm) in self.algorithms.iter_mut().enumerate() {
            if !self.in_signon && !self.running[index] {
                continue;
            }
            if !mask_contains(self.algorithm_message_masks[index], message_type) {
                continue;
            }
            let start = self.cost_sampling.then(Instant::now);
            match algorithm.on_message(message, &self.state, &parser_state, _tick) {
                Ok(detections) => record_detections(&mut self.detections, detections),
                Err(_) => {}
            }
            if let Some(start) = start {
                self.algorithm_sampled_nanos[index] += start.elapsed().as_nanos() as u64;
            }
        }
    }

    fn handle_string_entry(
        &mut self,
        table: &str,
        index: usize,
        entry: &StringTableEntry,
        _parser_state: &ParserState,
    ) {
        if table == "userinfo" {
            let _ = self.parse_user_info(
                index,
                entry.text.as_ref().map(|s| s.as_ref()),
                entry.extra_data.as_ref().map(|data| data.data.clone()),
            );
        }
        for algorithm in self.algorithms.iter_mut() {
            if let Ok(detections) = algorithm.on_string_entry(table, index, entry, &self.state) {
                record_detections(&mut self.detections, detections);
            }
        }
    }

    fn handle_data_tables(
        &mut self,
        _parse_tables: &[ParseSendTable],
        server_classes: &[ServerClass],
        _parser_state: &ParserState,
    ) {
        self.class_names = server_classes
            .iter()
            .map(|class| &class.name)
            .cloned()
            .collect();
        self.class_kinds = self
            .class_names
            .iter()
            .map(|class_name| ClassKind::new(class_name.as_str()))
            .collect();
    }

    fn handle_packet_meta(
        &mut self,
        tick: DemoTick,
        _meta: &MessagePacketMeta,
        _parser_state: &ParserState,
    ) {
        self.state.tick = tick;
        self.tick = tick;
    }

    fn into_output(self, _state: &ParserState) -> Self::Output {
        self.state
    }
}

impl BorrowMessageHandler for CheatAnalyser<'_> {
    fn borrow_output(&self, _state: &ParserState) -> &Self::Output {
        &self.state
    }
}

impl crate::base::demo_handler_base::DemoHandlerAnalyser for CheatAnalyser<'_> {
    fn handle_user_cmd(&mut self, packet: tf_demo_parser::demo::packet::usercmd::UserCmdPacket) {
        self.state.user_cmds.push(packet);
    }
    fn handle_console_cmd(&mut self, packet: tf_demo_parser::demo::packet::consolecmd::ConsoleCmdPacket) {
        self.state.console_cmds.push(packet);
    }
    fn handle_packet_info(&mut self, packet: &PacketInfo) {
        self.update_running();
        // The first packet after the signon data still counts as signon for the algorithms
        let in_signon = self.in_signon;
        if !packet.signon {
            self.in_signon = false;
        }
        for (index, algorithm) in self.algorithms.iter_mut().enumerate() {
            if !in_signon && !self.running[index] {
                continue;
            }
            if let Ok(detections) = algorithm.on_packet(packet) {
                record_detections(&mut self.detections, detections);
            }
        }
    }
}

impl<'a> CheatAnalyser<'a> {
    /// Creates the analyser and sets the message types parsed on the current thread, so it must
    /// be created on the thread that parses the demo (or call `activate_message_filter` there).
    pub fn new(algorithms: Vec<Box<dyn CheatAlgorithm<'a> + 'a + Send>>) -> Self {
        // Figure out what message types we're going to be using.
        // Err(true) means an algorithm wants every message, Err(false) means none.
        let algorithm_message_masks: Vec<u64> = algorithms
            .iter()
            .map(|algorithm| match algorithm.handled_messages() {
                Ok(types) => message_mask(&types),
                Err(true) => u64::MAX,
                Err(false) => 0,
            })
            .collect();
        let handled_message_mask = algorithm_message_masks
            .iter()
            .fold(message_mask(&BASE_MESSAGE_TYPES), |mask, algorithm_mask| mask | algorithm_mask);

        let algorithm_count = algorithms.len();
        let whole_demo: Vec<bool> = algorithms.iter().map(|algorithm| algorithm.needs_whole_demo()).collect();
        let analyser = Self {
            state: Default::default(),
            algorithms,
            detections: Vec::new(),
            header: None,
            tick: DemoTick::default(),
            worker_id: 0,
            parse_error: None,
            last_progress_update_time: Instant::now(),
            progress: vec![],
            class_names: Vec::new(),
            class_kinds: Vec::new(),
            prop_infos: Default::default(),
            handled_message_mask,
            algorithm_message_masks,
            cost_sampling: algorithm_costs::COST_LEARNING,
            cost_sample_ticks: 0,
            algorithm_sampled_nanos: vec![0; algorithm_count],
            algorithm_finish_nanos: vec![0; algorithm_count],
            analyse_from_tick: 0,
            in_signon: true,
            whole_demo,
            start_ticks: vec![0; algorithm_count],
            running: vec![true; algorithm_count],
            next_start_tick: u32::MAX,
            base_message_mask: message_mask(&BASE_MESSAGE_TYPES),
        };
        analyser.activate_message_filter();
        analyser
    }

    /// Makes the current thread parse only the message types this analyser needs.
    pub fn activate_message_filter(&self) {
        HANDLED_MESSAGE_MASK.with(|mask| mask.set(self.handled_message_mask));
    }

    /// Makes the detections from `tick` on the ones that count (0 = whole demo). Each algorithm
    /// runs from its `live_warmup_ticks()` before `tick`, the ones that need the whole demo from
    /// the start. Everything before an algorithm's start (except the signon data, string tables
    /// and the raw demo bytes) only updates the game state.
    pub fn set_analyse_from_tick(&mut self, tick: u32) {
        self.analyse_from_tick = tick;
        self.start_ticks = self
            .algorithms
            .iter()
            .zip(&self.whole_demo)
            .map(|(algorithm, &whole)| if whole { 0 } else { tick.saturating_sub(algorithm.live_warmup_ticks()) })
            .collect();
        self.running = self.start_ticks.iter().map(|&start| start == 0).collect();
        self.next_start_tick = self
            .start_ticks
            .iter()
            .zip(&self.running)
            .filter(|(_, &running)| !running)
            .map(|(&start, _)| start)
            .min()
            .unwrap_or(u32::MAX);
    }

    /// Whether the demo got past its signon data. Until then `tick` is the server tick of the
    /// signon frames, not a demo tick (those start at 0 with the first game packets).
    pub fn past_signon(&self) -> bool {
        !self.in_signon
    }

    /// Starts the algorithms whose start tick is reached. Once started they keep running, even if
    /// a broken demo jumps back to lower ticks.
    fn update_running(&mut self) {
        let tick = u32::from(self.tick);
        if tick < self.next_start_tick {
            return;
        }
        let mut next_start_tick = u32::MAX;
        for (running, &start) in self.running.iter_mut().zip(&self.start_ticks) {
            if *running {
                continue;
            }
            if tick >= start {
                *running = true;
            } else {
                next_start_tick = next_start_tick.min(start);
            }
        }
        self.next_start_tick = next_start_tick;
    }

    /// Call between packets: before an algorithm's start tick the current thread doesn't parse
    /// the message types only that algorithm uses (temp entities, user messages, sounds...).
    pub fn update_message_filter(&self) {
        if self.analyse_from_tick == 0 {
            return;
        }
        let wanted = if self.in_signon {
            self.handled_message_mask
        } else {
            let tick = u32::from(self.tick).saturating_add(FILTER_LOOKAHEAD_TICKS);
            self.algorithm_message_masks
                .iter()
                .zip(self.start_ticks.iter().zip(&self.running))
                .filter(|(_, (&start, &running))| running || tick >= start)
                .fold(self.base_message_mask, |mask, (algorithm_mask, _)| mask | algorithm_mask)
        };
        HANDLED_MESSAGE_MASK.with(|mask| mask.set(wanted));
    }

    /// Estimated CPU time (seconds) spent in each algorithm, extrapolated from the sampled ticks.
    /// Only measured in builds that learn algorithm costs.
    pub fn measured_algorithm_costs(&self) -> Vec<(String, f64)> {
        self.algorithms
            .iter()
            .enumerate()
            .map(|(index, algorithm)| {
                let nanos = self.algorithm_sampled_nanos[index] as f64
                    * f64::from(algorithm_costs::SAMPLE_INTERVAL)
                    + self.algorithm_finish_nanos[index] as f64;
                (algorithm.algorithm_name().to_string(), nanos / 1e9)
            })
            .collect()
    }

    pub fn handle_user_cmd(&mut self, packet: tf_demo_parser::demo::packet::usercmd::UserCmdPacket) {
        self.state.user_cmds.push(packet);
    }

    pub fn handle_console_cmd(&mut self, packet: tf_demo_parser::demo::packet::consolecmd::ConsoleCmdPacket) {
        self.state.console_cmds.push(packet);
    }

    pub fn init(&mut self) -> Result<(), Error> {
        for algorithm in &mut self.algorithms {
            match algorithm.init() {
                Ok(_) => {}
                Err(_) => continue,
            }
        }
        Ok(())
    }

    pub fn handle_demo_bytes(&mut self, bytes: &[u8]) {
        for algorithm in self.algorithms.iter_mut() {
            if let Ok(detections) = algorithm.on_demo_bytes(bytes) {
                record_detections(&mut self.detections, detections);
            }
        }
    }

    pub fn finish(&mut self) -> Result<(), Error> {
        for (index, algorithm) in self.algorithms.iter_mut().enumerate() {
            let start = algorithm_costs::COST_LEARNING.then(Instant::now);
            let result = algorithm.finish();
            if let Some(start) = start {
                self.algorithm_finish_nanos[index] += start.elapsed().as_nanos() as u64;
            }
            match result {
                Ok(detections) => record_detections(&mut self.detections, detections),
                Err(_) => continue,
            }
        }
        self.detections.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.player.cmp(&b.player))
                .then_with(|| a.algorithm.cmp(&b.algorithm))
        });
        Ok(())
    }

    pub fn apply_sliding_window(&mut self, settings: &crate::lib::general_settings::GeneralSettings) {
        self.detections = settings.filter_detections(&self.detections);
    }

    pub fn print_metadata(&self) {
        if self.header.is_none() {
            return;
        }
        let header = self.header.as_ref().unwrap();
        let ticks = self.get_tick_count_u32();

        dev_print!("Map: {}", header.map);
        let hours = (header.duration / 3600.0).floor();
        let minutes = ((header.duration % 3600.0) / 60.0).floor();
        let seconds = (header.duration % 60.0).floor();
        let milliseconds = ((header.duration % 1.0) * 100.0).floor();
        dev_print!(
            "Duration: {:02}:{:02}:{:02}.{:03} ({} ticks)",
            hours,
            minutes,
            seconds,
            milliseconds,
            ticks
        );
        dev_print!("User: {}", header.nick);
        dev_print!("Server: {}", header.server);
    }

    pub fn print_detection_json(&self, pretty: bool) {
        let analysis = serde_json::json!({
            "server_ip": self.header.as_ref().map_or("unknown".to_string(), |h| h.server.clone()),
            "duration": self.tick,
            "author": self.header.as_ref().map_or("unknown".to_string(), |h| h.nick.clone()),
            "map": self.header.as_ref().map_or("unknown".to_string(), |h| h.map.clone()),
            "detections": self.detections
        });
        let json = if pretty {
            serde_json::to_string_pretty(&analysis).unwrap()
        } else {
            serde_json::to_string(&analysis).unwrap()
        };
        println!("{}", json);
    }

    pub fn print_detection_summary(&self) {
        let mut algorithm_counts: HashMap<String, HashMap<u64, usize>> = HashMap::new();
        for detection in &self.detections {
            let algorithm = detection.algorithm.clone();
            let steamid = detection.player;
            *algorithm_counts
                .entry(algorithm)
                .or_insert(HashMap::new())
                .entry(steamid)
                .or_insert(0) += 1;
        }

        dev_print!("Total detections: {}", self.detections.len());
        if self.detections.is_empty() {
            return;
        }
        dev_print!("Detections by Algorithm:");
        for (algorithm, steamid_counts) in algorithm_counts {
            dev_print!(
                "  {}: {} players, {} detections",
                algorithm,
                steamid_counts.len(),
                steamid_counts.values().sum::<usize>()
            );
            let mut steamid_counts_vec: Vec<_> = steamid_counts.into_iter().collect();
            steamid_counts_vec.sort_by(|a, b| b.1.cmp(&a.1));
            for (steamid, count) in steamid_counts_vec {
                dev_print!("    {}: {}", steamid, count);
            }
        }
    }
    // This code doesn't include the very first interval in any averages.
    // I didn't intend for that but it makes sense to exclude the intitial interval since
    // there tends to be a lot of boiler plate stuff which throws off the average anyway.
    fn check_progress(&mut self) {
        const PROGRESS_UPDATE_INTERVAL_MS: u128 = 1000;
        const TPS_ROLLING_AVERAGE_WINDOW: u32 = 10;
        let tick: u32 = self.tick.into();
        let total_ticks = self.get_header_tick_count_u32();

        if self.worker_id < crate::MAX_WORKERS {
            crate::WORKER_TICKS[self.worker_id].store(tick, std::sync::atomic::Ordering::Relaxed);
        }
        if total_ticks > 0 {
            crate::PROGRESS_TOTAL.store(total_ticks, std::sync::atomic::Ordering::Relaxed);
        }
        crate::PROGRESS_CURRENT.store(tick, std::sync::atomic::Ordering::Relaxed);

        if self.last_progress_update_time.elapsed().as_millis() < PROGRESS_UPDATE_INTERVAL_MS {
            return;
        }

        self.last_progress_update_time = Instant::now();
        self.progress.push(tick);
        while self.progress.len() > TPS_ROLLING_AVERAGE_WINDOW.try_into().unwrap() {
            self.progress.remove(0);
        }

        let tps = if self.progress.len() >= 2 {
            let tps = (self.progress.last().unwrap() - self.progress.first().unwrap()) as f64
                / (self.progress.len() as f64 - 1.0);
            tps * PROGRESS_UPDATE_INTERVAL_MS as f64 / 1000.0
        } else {
            tick.into()
        };

        if self.worker_id == 0 {
            dev_print!(
                "Processing tick {} ({} remaining, {:.0} tps)",
                tick,
                if total_ticks > tick { total_ticks - tick } else { 0 },
                tps
            );
        }
    }

    pub fn get_header_tick_count_u32(&self) -> u32 {
        self.header.as_ref().map(|h| h.ticks).unwrap_or(0)
    }

    pub fn get_tick_count_u32(&self) -> u32 {
        if self.header.is_none() {
            return self.tick.into();
        }
        let header = self.header.as_ref().unwrap();
        if self.tick > header.ticks {
            self.tick.into()
        } else {
            header.ticks
        }
    }

    pub fn handle_entity(&mut self, entity: &PacketEntity, parser_state: &ParserState) {
        if entity.update_type == UpdateType::Delete {
            self.state.weapons.remove(&entity.entity_index);
            self.state.remove_building(entity.entity_index);
            if let Some(player) = self
                .state
                .players
                .iter_mut()
                .find(|player| player.entity == entity.entity_index)
            {
                player.in_pvs = false;
            }
            return;
        }

        if entity.update_type == UpdateType::Leave {
            if let Some(player) = self
                .state
                .players
                .iter_mut()
                .find(|player| player.entity == entity.entity_index)
            {
                player.in_pvs = false;
            }
            return;
        }

        let class_kind = self
            .class_kinds
            .get(usize::from(entity.server_class))
            .copied()
            .unwrap_or_default();

        // Entering entities are merged with their baseline; other updates only carry the changed
        // props, which can be borrowed instead of cloned.
        let props: Cow<[SendProp]> = if entity.update_type == UpdateType::Enter {
            Cow::Owned(entity.props(parser_state).collect())
        } else {
            Cow::Borrowed(&entity.props)
        };

        let mut item_def = None;
        for prop in props.iter() {
            if prop_info(&mut self.prop_infos, prop.identifier).item_definition_index {
                if let Ok(val) = i64::try_from(&prop.value) {
                    item_def = Some(val as u16);
                }
            }
        }
        if item_def.is_some() || class_kind.is_weapon {
            let class_name: &str = self
                .class_names
                .get(usize::from(entity.server_class))
                .map(|class_name| class_name.as_str())
                .unwrap_or("");
            let entry = self.state.weapons.entry(entity.entity_index).or_default();
            entry.entity = entity.entity_index;
            if entry.class_name != class_name {
                entry.class_name = class_name.to_string();
            }
            if item_def.is_some() {
                entry.item_def_index = item_def;
            }
        }

        match class_kind.handler {
            EntityHandler::Player => self.handle_player_entity(entity, &props),
            EntityHandler::PlayerResource => self.handle_player_resource(&props),
            EntityHandler::World => self.handle_world_entity(&props),
            EntityHandler::Sentry => self.handle_sentry_entity(entity, &props),
            EntityHandler::Dispenser => self.handle_dispenser_entity(entity, &props),
            EntityHandler::Teleporter => self.handle_teleporter_entity(entity, &props),
            EntityHandler::None => {}
        }
    }

    pub fn handle_player_resource(&mut self, props: &[SendProp]) {
        for prop in props {
            let Some((player_id, field)) = prop_info(&mut self.prop_infos, prop.identifier).resource else {
                continue;
            };
            let entity_id = EntityId::from(player_id);
            let Some(player) = self
                .state
                .players
                .iter_mut()
                .find(|player| player.entity == entity_id)
            else {
                continue;
            };
            let user_id = player.info.as_ref().map(|info| info.user_id);
            match field {
                ResourceField::Team => {
                    player.team = Team::new(i64::try_from(&prop.value).unwrap_or_default())
                }
                ResourceField::MaxHealth => {
                    player.max_health = i64::try_from(&prop.value).unwrap_or_default() as u16
                }
                ResourceField::Class => {
                    player.class = Class::new(i64::try_from(&prop.value).unwrap_or_default())
                }
                ResourceField::Charge => {
                    player.charge = i64::try_from(&prop.value).unwrap_or_default() as u8
                }
                ResourceField::Ping => {
                    player.ping = i64::try_from(&prop.value).unwrap_or_default() as u16
                }
                ResourceField::Other => {}
            }
            if let Some(user_id) = user_id {
                if self.state.entid_to_userid.get(&entity_id) != Some(&user_id) {
                    self.state.set_entid_to_userid(entity_id, user_id);
                }
            }
        }
    }

    pub fn handle_player_entity(&mut self, entity: &PacketEntity, props: &[SendProp]) {
        let player = self.state.get_or_create_player(entity.entity_index);

        const HEALTH_PROP: SendPropIdentifier =
            SendPropIdentifier::new("DT_BasePlayer", "m_iHealth");
        const MAX_HEALTH_PROP: SendPropIdentifier =
            SendPropIdentifier::new("DT_BasePlayer", "m_iMaxHealth");
        const LIFE_STATE_PROP: SendPropIdentifier =
            SendPropIdentifier::new("DT_BasePlayer", "m_lifeState");

        const LOCAL_ORIGIN: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFLocalPlayerExclusive", "m_vecOrigin");
        const NON_LOCAL_ORIGIN: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFNonLocalPlayerExclusive", "m_vecOrigin");
        const LOCAL_ORIGIN_Z: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFLocalPlayerExclusive", "m_vecOrigin[2]");
        const NON_LOCAL_ORIGIN_Z: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFNonLocalPlayerExclusive", "m_vecOrigin[2]");
        const LOCAL_EYE_ANGLES: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFLocalPlayerExclusive", "m_angEyeAngles[1]");
        const NON_LOCAL_EYE_ANGLES: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFNonLocalPlayerExclusive", "m_angEyeAngles[1]");
        const LOCAL_PITCH_ANGLES: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFLocalPlayerExclusive", "m_angEyeAngles[0]");
        const NON_LOCAL_PITCH_ANGLES: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFNonLocalPlayerExclusive", "m_angEyeAngles[0]");

        const SIMTIME_PROP: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseEntity", "m_flSimulationTime");

        player.in_pvs = entity.in_pvs;
        let mut simtime_received = false;

        for prop in props {
            match prop_info(&mut self.prop_infos, prop.identifier).player_field {
                PlayerPropField::ActiveWeapon => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        let handle = val as u32;
                        let ent_id = crate::util::helpers::handle_to_entid(handle);
                        player.active_weapon = if u32::from(ent_id) != 0x7FF && u32::from(ent_id) != 0 {
                            Some(ent_id)
                        } else {
                            None
                        };
                    }
                }
                PlayerPropField::Cond => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        player.cond = val as u32;
                    }
                }
                PlayerPropField::CondEx => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        player.cond_ex = val as u32;
                    }
                }
                PlayerPropField::CondEx2 => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        player.cond_ex2 = val as u32;
                    }
                }
                PlayerPropField::CondEx3 => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        player.cond_ex3 = val as u32;
                    }
                }
                PlayerPropField::InvisChangeCompleteTime => {
                    if let Ok(val) = f32::try_from(&prop.value) {
                        player.invis_change_complete_time = val;
                    }
                }
                PlayerPropField::Flags => {
                    if let Ok(val) = i64::try_from(&prop.value) {
                        player.flags = val as u32;
                    }
                }
                PlayerPropField::EyeAnglesYaw => {
                    if let Ok(val) = f32::try_from(&prop.value) {
                        player.view_angle = val;
                    }
                }
                PlayerPropField::EyeAnglesPitch => {
                    if let Ok(val) = f32::try_from(&prop.value) {
                        player.pitch_angle = val;
                    }
                }
                PlayerPropField::None => {}
            }

            match prop.identifier {
                HEALTH_PROP => {
                    player.health = i64::try_from(&prop.value).unwrap_or_default() as u16
                }
                MAX_HEALTH_PROP => {
                    player.max_health = i64::try_from(&prop.value).unwrap_or_default() as u16
                }
                LIFE_STATE_PROP => {
                    player.state = PlayerState::new(i64::try_from(&prop.value).unwrap_or_default())
                }
                LOCAL_ORIGIN | NON_LOCAL_ORIGIN => {
                    let pos_xy = VectorXY::try_from(&prop.value).unwrap_or_default();
                    player.position.x = pos_xy.x;
                    player.position.y = pos_xy.y;
                }
                LOCAL_ORIGIN_Z | NON_LOCAL_ORIGIN_Z => {
                    player.position.z = f32::try_from(&prop.value).unwrap_or_default()
                }
                LOCAL_EYE_ANGLES | NON_LOCAL_EYE_ANGLES => {
                    player.view_angle = f32::try_from(&prop.value).unwrap_or_default()
                }
                LOCAL_PITCH_ANGLES | NON_LOCAL_PITCH_ANGLES => {
                    player.pitch_angle = f32::try_from(&prop.value).unwrap_or_default()
                }
                SIMTIME_PROP => {
                    player.simtime = i64::try_from(&prop.value).unwrap_or_default() as u16;
                    simtime_received = true;
                }
                _ => {}
            }
        }
        if simtime_received {
            self.state.simtime_received.push(entity.entity_index);
        }
    }

    pub fn handle_world_entity(&mut self, props: &[SendProp]) {
        const WORLD_MINS: SendPropIdentifier = SendPropIdentifier::new("DT_WORLD", "m_WorldMins");
        const WORLD_MAXS: SendPropIdentifier = SendPropIdentifier::new("DT_WORLD", "m_WorldMaxs");

        if let (
            Some(SendProp {
                value: SendPropValue::Vector(boundary_min),
                ..
            }),
            Some(SendProp {
                value: SendPropValue::Vector(boundary_max),
                ..
            }),
        ) = (
            props.iter().find(|prop| prop.identifier == WORLD_MINS).cloned(),
            props.iter().find(|prop| prop.identifier == WORLD_MAXS).cloned(),
        ) {
            self.state.world = Some(World {
                boundary_min,
                boundary_max,
            })
        }
    }

    pub fn handle_sentry_entity(&mut self, entity: &PacketEntity, props: &[SendProp]) {
        const ANGLE: SendPropIdentifier =
            SendPropIdentifier::new("DT_TFNonLocalPlayerExclusive", "m_angEyeAngles[1]");
        const MINI: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseObject", "m_bMiniBuilding");
        const CONTROLLED: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectSentrygun", "m_bPlayerControlled");
        const TARGET: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectSentrygun", "m_hAutoAimTarget");
        const SHELLS: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectSentrygun", "m_iAmmoShells");
        const ROCKETS: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectSentrygun", "m_iAmmoRockets");

        if entity.update_type == UpdateType::Delete {
            self.state.remove_building(entity.entity_index);
            return;
        }

        self.handle_building(entity, props, BuildingClass::Sentry);

        let building = self
            .state
            .get_or_create_building(entity.entity_index, BuildingClass::Sentry);

        if let Building::Sentry(sentry) = building {
            for prop in props {
                match prop.identifier {
                    ANGLE => sentry.angle = f32::try_from(&prop.value).unwrap_or_default(),
                    MINI => sentry.is_mini = i64::try_from(&prop.value).unwrap_or_default() > 0,
                    CONTROLLED => {
                        sentry.player_controlled =
                            i64::try_from(&prop.value).unwrap_or_default() > 0
                    }
                    TARGET => {
                        sentry.auto_aim_target =
                            UserId::from(i64::try_from(&prop.value).unwrap_or_default() as u16)
                    }
                    SHELLS => sentry.shells = i64::try_from(&prop.value).unwrap_or_default() as u16,
                    ROCKETS => {
                        sentry.rockets = i64::try_from(&prop.value).unwrap_or_default() as u16
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn handle_teleporter_entity(&mut self, entity: &PacketEntity, props: &[SendProp]) {
        const RECHARGE_TIME: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectTeleporter", "m_flRechargeTime");
        const RECHARGE_DURATION: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectTeleporter", "m_flCurrentRechargeDuration");
        const TIMES_USED: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectTeleporter", "m_iTimesUsed");
        const OTHER_END: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectTeleporter", "m_bMatchBuilding");
        const YAW_TO_EXIT: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectTeleporter", "m_flYawToExit");
        const IS_ENTRANCE: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseObject", "m_iObjectMode");

        if entity.update_type == UpdateType::Delete {
            self.state.remove_building(entity.entity_index);
            return;
        }

        self.handle_building(entity, props, BuildingClass::Teleporter);

        let building = self
            .state
            .get_or_create_building(entity.entity_index, BuildingClass::Teleporter);

        if let Building::Teleporter(teleporter) = building {
            for prop in props {
                match prop.identifier {
                    RECHARGE_TIME => {
                        teleporter.recharge_time = f32::try_from(&prop.value).unwrap_or_default()
                    }
                    RECHARGE_DURATION => {
                        teleporter.recharge_duration =
                            f32::try_from(&prop.value).unwrap_or_default()
                    }
                    TIMES_USED => {
                        teleporter.times_used =
                            i64::try_from(&prop.value).unwrap_or_default() as u16
                    }
                    OTHER_END => {
                        teleporter.other_end =
                            EntityId::from(i64::try_from(&prop.value).unwrap_or_default() as u32)
                    }
                    YAW_TO_EXIT => {
                        teleporter.yaw_to_exit = f32::try_from(&prop.value).unwrap_or_default()
                    }
                    IS_ENTRANCE => {
                        teleporter.is_entrance = i64::try_from(&prop.value).unwrap_or_default() == 0
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn handle_dispenser_entity(&mut self, entity: &PacketEntity, props: &[SendProp]) {
        const AMMO: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectDispenser", "m_iAmmoMetal");
        const HEALING: SendPropIdentifier =
            SendPropIdentifier::new("DT_ObjectDispenser", "healing_array");

        if entity.update_type == UpdateType::Delete {
            self.state.remove_building(entity.entity_index);
            return;
        }

        self.handle_building(entity, props, BuildingClass::Dispenser);

        let building = self
            .state
            .get_or_create_building(entity.entity_index, BuildingClass::Dispenser);

        if let Building::Dispenser(dispenser) = building {
            for prop in props {
                match prop.identifier {
                    AMMO => dispenser.metal = i64::try_from(&prop.value).unwrap_or_default() as u16,
                    HEALING => {
                        let values = match &prop.value {
                            SendPropValue::Array(vec) => vec.as_slice(),
                            _ => Default::default(),
                        };

                        dispenser.healing = values
                            .iter()
                            .map(|val| UserId::from(i64::try_from(val).unwrap_or_default() as u16))
                            .collect()
                    }
                    _ => {}
                }
            }
        }
    }

    fn handle_building(
        &mut self,
        entity: &PacketEntity,
        props: &[SendProp],
        class: BuildingClass,
    ) {
        let building = self
            .state
            .get_or_create_building(entity.entity_index, class);

        const LOCAL_ORIGIN: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseEntity", "m_vecOrigin");
        const TEAM: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_iTeamNum");
        const ANGLE: SendPropIdentifier = SendPropIdentifier::new("DT_BaseEntity", "m_angRotation");
        const SAPPED: SendPropIdentifier = SendPropIdentifier::new("DT_BaseObject", "m_bHasSapper");
        const BUILDING: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseObject", "m_bBuilding");
        const LEVEL: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseObject", "m_iUpgradeLevel");
        const BUILDER: SendPropIdentifier = SendPropIdentifier::new("DT_BaseObject", "m_hBuilder");
        const MAX_HEALTH: SendPropIdentifier =
            SendPropIdentifier::new("DT_BaseObject", "m_iMaxHealth");
        const HEALTH: SendPropIdentifier = SendPropIdentifier::new("DT_BaseObject", "m_iHealth");

        match building {
            Building::Sentry(Sentry {
                position,
                team,
                angle,
                sapped,
                builder,
                level,
                building,
                max_health,
                health,
                ..
            })
            | Building::Dispenser(Dispenser {
                position,
                team,
                angle,
                sapped,
                builder,
                level,
                building,
                max_health,
                health,
                ..
            })
            | Building::Teleporter(Teleporter {
                position,
                team,
                angle,
                sapped,
                builder,
                level,
                building,
                max_health,
                health,
                ..
            }) => {
                for prop in props {
                    match prop.identifier {
                        LOCAL_ORIGIN => {
                            *position = Vector::try_from(&prop.value).unwrap_or_default()
                        }
                        TEAM => *team = Team::new(i64::try_from(&prop.value).unwrap_or_default()),
                        ANGLE => *angle = f32::try_from(&prop.value).unwrap_or_default(),
                        SAPPED => *sapped = i64::try_from(&prop.value).unwrap_or_default() > 0,
                        BUILDING => *building = i64::try_from(&prop.value).unwrap_or_default() > 0,
                        LEVEL => *level = i64::try_from(&prop.value).unwrap_or_default() as u8,
                        BUILDER => {
                            *builder =
                                UserId::from(i64::try_from(&prop.value).unwrap_or_default() as u16)
                        }
                        MAX_HEALTH => {
                            *max_health = i64::try_from(&prop.value).unwrap_or_default() as u16
                        }
                        HEALTH => *health = i64::try_from(&prop.value).unwrap_or_default() as u16,
                        _ => {}
                    }
                }
            }
        }
    }

    fn parse_user_info(
        &mut self,
        index: usize,
        text: Option<&str>,
        data: Option<Stream>,
    ) -> ReadResult<()> {
        if let Some(user_info) =
            tf_demo_parser::demo::data::UserInfo::parse_from_string_table(index as u16, text, data)?
        {
            let ent_id = user_info.entity_id;
            self.state
                .friends_ids
                .insert(ent_id, user_info.player_info.friends_id);
            self.state
                .custom_files
                .insert(ent_id, user_info.player_info.custom_file);
            self.state
                .set_entid_to_userid(ent_id, user_info.player_info.user_id.clone());
            if let Ok(steam_id) = SteamID::from_steam3(&user_info.player_info.steam_id) {
                let id64: u64 = steam_id.into();
                self.state
                    .set_userid_to_id64(user_info.player_info.user_id.clone(), id64);
                if !user_info.player_info.name.is_empty() {
                    self.state
                        .player_names
                        .insert(id64, user_info.player_info.name.clone());
                }
                self.state
                    .user_info_history
                    .insert(id64, user_info.clone().into());
            }
            self.state.get_or_create_player(ent_id).info = Some(user_info.into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_player_slot() {
        let mut state = CheatAnalyserState::default();
        let sid = 76561198000000001u64;

        assert_eq!(state.get_player_slot(sid), None);

        // Add user info with entity_id = 3 (client slot 2)
        let user_info = tf_demo_parser::demo::parser::analyser::UserInfo {
            classes: Default::default(),
            name: "Suspect".to_string(),
            user_id: 10u32.into(),
            steam_id: "[U:1:1]".to_string(),
            entity_id: 3u32.into(),
            team: Team::default(),
        };
        state.user_info_history.insert(sid, user_info);

        assert_eq!(state.get_player_slot(sid), Some(2));
    }
}
