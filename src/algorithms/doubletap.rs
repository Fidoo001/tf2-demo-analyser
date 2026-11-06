// Originally written by Tellta, rewritten to be weapon agnostic and tick-shift aware.
//
// Doubletap (tick shifting) lets a cheater fire a weapon again long before its refire delay has
// elapsed, because the shifted ticks make the server believe enough time has passed. It works on
// every weapon, so this algorithm works on the *shot timeline* instead of on damage events:
//
//  * every attack animation event (CTEPlayerAnimEvent, event 0) is a shot, whether it hits or not
//  * the demo only shows snapshots, so shots are grouped per snapshot ("frame")
//  * each weapon class has a minimum legitimate refire interval (measured from real demos and from
//    the weapon definitions). Two shots closer together than that are impossible.
//  * two shots inside ONE snapshot are impossible for any weapon whose refire delay is longer than
//    the snapshot interval. This is the "same tick" doubletap that damage-event based detection
//    cannot see, because the second shot is delta-compressed away from the raw event list.
//
// Continuous fire weapons (minigun, flamethrower, syringe gun, SMG) fire faster than the snapshot
// interval by design and cannot be judged by interval. For the minigun the old damage based check
// (`min_tick_heavy`) is kept.
use std::collections::{HashMap, VecDeque};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player};

use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::handle_to_entid;

use anyhow::Error;
use serde_json::{Map, Value};
use steamid_ng::SteamID;
use tf_demo_parser::demo::gameevent_gen::GameEvent;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

/// One CTEPlayerAnimEvent: the player that triggered it and the animation event id.
struct TeShot {
    sid: u64,
    event: i64,
}

impl TeShot {
    /// PLAYERANIMEVENT_ATTACK_PRIMARY
    fn is_primary_attack(&self) -> bool {
        self.event == 0
    }
}

/// Collects the player animation events of a TempEntities message.
/// Every event of a message is delta encoded against the previous event of the same class (zero for the
/// first one), so a prop that is missing from an event keeps the value of the previous event. Reading a
/// missing `m_iEvent` as 0 turns every repeated reload event into an "attack".
fn extract_te_shots(message: &Message, parser_state: &ParserState, state: &CheatAnalyserState) -> Vec<TeShot> {
    const ANIM_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_hPlayer");
    const ANIM_EVENT: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_iEvent");
    let Message::TempEntities(msg) = message else {
        return vec![];
    };
    let mut out = Vec::new();
    let mut prev_class = None;
    let mut handle = 0u32;
    let mut anim_event = 0i64;
    for event in &msg.events {
        if prev_class != Some(event.class_id) {
            handle = 0;
            anim_event = 0;
            prev_class = Some(event.class_id);
        }
        let Some(class) = parser_state.server_classes.get(usize::from(event.class_id)) else {
            continue;
        };
        if class.name.as_str() != "CTEPlayerAnimEvent" {
            continue;
        }
        for prop in &event.props {
            if prop.identifier == ANIM_PLAYER {
                if let Some(h) = i64::try_from(&prop.value).ok().and_then(|h| u32::try_from(h).ok()) {
                    handle = h;
                }
            } else if prop.identifier == ANIM_EVENT {
                if let Ok(e) = i64::try_from(&prop.value) {
                    anim_event = e;
                }
            }
        }
        let Some(sid) = state
            .entid_to_userid
            .get(&handle_to_entid(handle))
            .and_then(|uid| state.userid_to_id64.get(uid))
        else {
            continue;
        };
        out.push(TeShot { sid: *sid, event: anim_event });
    }
    out
}

#[derive(Clone, Copy)]
struct Profile {
    /// fastest legitimate interval between two shots, in ticks
    floor: u32,
    /// how many attack animation events one shot produces (grenade/sticky launchers, bows and
    /// the Dragon's Fury send every shot twice)
    events_per_shot: u32,
    /// judge the interval between shots (very fast weapons only get the same-snapshot check)
    interval: bool,
}

/// Weapons with real doubletap potential: very fast firing weapons (scatterguns, miniguns, shotguns, pistols, SMGs).
/// The Rescue Ranger is not a typical shotgun and is not included; rocket launchers and other slow weapons are not either.
fn dt_weapon(class_name: &str) -> bool {
    matches!(
        class_name,
        "CTFScatterGun" | "CTFSodaPopper" | "CTFPEPBrawlerBlaster" | "CTFPistol_ScoutPrimary" | "CTFPistol_Scout" | "CTFPistol"
            | "CTFShotgun" | "CTFShotgun_Soldier" | "CTFShotgun_Pyro" | "CTFShotgun_HWG" | "CTFShotgun_Revenge"
            | "CTFMinigun" | "CTFSMG"
    )
}

/// Fastest legitimate refire interval per weapon class, in ticks (1 tick = 15ms).
/// Melee weapons (wrench pickup exploit, variable swing speed), the Air Strike and Eviction Notice
/// (variable fire rate) are deliberately NOT judged.
/// Values are the smallest intervals seen between real shots in recorded demos, rounded down,
/// cross checked against the weapon definitions. Unlisted classes are not judged.
fn weapon_profile(class_name: &str, item: Option<u16>) -> Option<Profile> {
    let p = |floor, events_per_shot| Some(Profile { floor, events_per_shot, interval: true });
    match class_name {
        // Scout
        "CTFScatterGun" => match item {
            Some(45) | Some(1078) => p(19, 1), // Force-A-Nature
            _ => p(37, 1),
        },
        "CTFSodaPopper" => p(18, 1),
        "CTFPEPBrawlerBlaster" => p(33, 1),
        "CTFPistol_ScoutPrimary" => p(14, 1), // Shortstop
        // pistols
        // pistols fire so fast that consecutive snapshots legitimately show intervals of 0-3 ticks, so only two
        // shots inside one snapshot count
        "CTFPistol_Scout" | "CTFPistol" => Some(Profile { floor: 8, events_per_shot: 1, interval: false }),
        // shotguns (the Rescue Ranger is not a typical shotgun and is deliberately not judged)
        "CTFShotgun_Soldier" | "CTFShotgun_Pyro" | "CTFShotgun" | "CTFShotgun_Revenge" => p(36, 1),
        "CTFShotgun_HWG" => p(31, 1),
        // The minigun is judged separately (see `min_tick_heavy`).
        _ => None,
    }
}

struct PendingShot {
    ping: u32,
    sid: u64,
    weapon_ent: usize,
    class_name: String,
    item: Option<u16>,
    weapon_name: String,
    player_class: &'static str,
}

#[derive(Clone, Copy)]
struct LastShot {
    /// server tick of the shot
    tick: u32,
    /// demo tick of the shot (what a reviewer can seek to)
    demo_tick: u32,
    gap: u32,
}

#[derive(Default)]
pub struct DoubleTap {
    params: Parameters,

    /// shot events collected for the snapshot currently being parsed
    pending: Vec<PendingShot>,
    /// last judged shot per (player, weapon entity)
    last: HashMap<(u64, usize), LastShot>,
    prev_frame_tick: u32,
    /// decoded simulation ticks per player (lag spike detection)
    sim_hist: HashMap<u64, VecDeque<(u32, i64)>>,
    /// active weapon entity per player and the tick it became active (weapon switch guard)
    active: HashMap<u64, (usize, u32)>,
    /// detections held back until the player has enough evidence, with the weight of each
    buffered: Vec<(Detection, u32)>,
    weights: HashMap<u64, u32>,

    /// minigun: attack event ticks inside the sliding window, per (player, weapon entity)
    mg_events: HashMap<(u64, usize), VecDeque<u32>>,
    /// minigun: highest event count seen in one window per player, and where (for tuning)
    mg_max: HashMap<u64, (u32, u32)>,
    mg_cool: HashMap<u64, u32>,

    /// legacy damage event path (minigun only): (tick, weapon, victim, damage)
    heavy_shots: HashMap<u64, Vec<u32>>,
}

impl DoubleTap {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("assert_pvs".to_string(), Parameter::Bool(true)),
                // slack subtracted from every weapon's refire floor to absorb snapshot/ping jitter
                ("tolerance_ticks".to_string(), Parameter::Int(0)),
                // skip shots that follow a simulation time jump covering the missing refire time (lag burst)
                ("lag_excuse".to_string(), Parameter::Bool(true)),
                ("lag_excuse_ratio".to_string(), Parameter::Float(1.0)),
                // MvM fire rate upgrades make fast refires legal, so those demos are not judged
                ("ignore_mvm".to_string(), Parameter::Bool(true)),
                // a shot pair is flagged when its interval is below this fraction of the weapon's fastest legal
                // interval (lag can squeeze a legit pair, so only clear violations are reported)
                ("max_floor_ratio".to_string(), Parameter::Float(0.8)),
                // detect several shots inside one snapshot
                ("same_frame".to_string(), Parameter::Bool(true)),
                // fraction of the refire floor that must fit between shots seen in one snapshot
                ("same_frame_ratio".to_string(), Parameter::Float(0.5)),
                // shots right after a weapon switch are attributed to the wrong weapon, ignore them
                ("switch_guard_ticks".to_string(), Parameter::Int(3)),
                // evidence needed before a player is reported at all (same_frame counts 2, short_interval 1).
                // Default 1 = report everything; incident grouping / suspicion scoring downstream decides.
                ("min_events".to_string(), Parameter::Int(1)),
                // Minigun fires faster than the snapshot interval, so it is judged by RATE: attack events inside a
                // sliding window must not exceed what the weapon can legitimately fire.
                ("minigun_rate".to_string(), Parameter::Bool(false)),
                ("minigun_window_ticks".to_string(), Parameter::Int(66)),
                ("minigun_max_events".to_string(), Parameter::Int(30)),
                // diagnostic: at the end report each minigun user's highest event count per window
                ("minigun_stats".to_string(), Parameter::Bool(false)),
                // minigun damage events closer than this many ticks (same-tick pairs are NOT counted: they occur in clean demos),
                // INCLUDING two in the same tick. The event-rate check above is off: attack animation events are
                // not one per shot for the minigun (63-106 per 66 ticks in clean demos), so it cannot discriminate.
                // Off by default (0): on 1000 database demos only 3 of 10 players it flagged were marked cheaters.
                ("min_tick_heavy".to_string(), Parameter::Int(0)),
            ]),
            ..Default::default()
        }
    }
}

fn is_mvm(state: &CheatAnalyserState) -> bool {
    state.header.as_ref().is_some_and(|h| h.map.starts_with("mvm_"))
}

fn player_sid(p: &Player) -> Option<u64> {
    let info = p.info.as_ref()?;
    SteamID::from_steam3(&info.steam_id).ok().map(u64::from)
}

fn build_data(items: Vec<(&str, Value)>) -> Value {
    let u200b = "\u{200B}";
    let mut m = Map::new();
    for (i, (k, v)) in items.into_iter().enumerate() {
        m.insert(format!("{}{}", u200b.repeat(i), k), v);
    }
    Value::Object(m)
}

impl<'a> CheatAlgorithm<'a> for DoubleTap {
    fn default(&self) -> bool {
        false
    }

    fn algorithm_name(&self) -> &str {
        "doubletap"
    }

    fn on_tick(
        &mut self,
        state: &CheatAnalyserState,
        _: &ParserState,
    ) -> Result<Vec<Detection>, Error> {
        // Shots are timed in SERVER ticks. The demo tick is the recorder's client tick: when the client stalls
        // (alt-tab, lag burst) it freezes while dozens of server snapshots are processed, so shots that are a
        // whole refire delay apart can all carry the same demo tick.
        let demo_tick = u32::from(state.tick);
        let tick = state.server_tick;
        let gap = tick.saturating_sub(self.prev_frame_tick).max(1);
        self.prev_frame_tick = tick;

        for p in &state.players {
            if let Some(sid) = player_sid(p) {
                let sim = state.sim_tick(p);
                let h = self.sim_hist.entry(sid).or_default();
                if h.back().map_or(true, |(_, s)| *s != sim) {
                    h.push_back((tick, sim));
                }
                while h.front().is_some_and(|(t, _)| tick.saturating_sub(*t) > 45) {
                    h.pop_front();
                }
            }
            if let (Some(sid), Some(ent)) = (player_sid(p), p.active_weapon) {
                let ent = usize::from(ent);
                match self.active.get(&sid) {
                    Some((e, _)) if *e == ent => {}
                    _ => {
                        self.active.insert(sid, (ent, tick));
                    }
                }
            }
        }

        if self.pending.is_empty() {
            return Ok(vec![]);
        }
        if get_parameter_value::<bool>(&self.params, "ignore_mvm") && is_mvm(state) {
            self.pending.clear();
            return Ok(vec![]);
        }

        let switch_guard = get_parameter_value::<i32>(&self.params, "switch_guard_ticks").max(0) as u32;
        let tolerance = get_parameter_value::<i32>(&self.params, "tolerance_ticks").max(0) as u32;
        let same_frame_on: bool = get_parameter_value(&self.params, "same_frame");
        let same_frame_ratio: f32 = get_parameter_value(&self.params, "same_frame_ratio");
        let max_floor_ratio: f32 = get_parameter_value(&self.params, "max_floor_ratio");
        let lag_excuse: bool = get_parameter_value(&self.params, "lag_excuse");
        let lag_excuse_ratio: f32 = get_parameter_value(&self.params, "lag_excuse_ratio");

        let mg_rate: bool = get_parameter_value(&self.params, "minigun_rate");
        let mg_window = get_parameter_value::<i32>(&self.params, "minigun_window_ticks").max(1) as u32;
        let mg_max_events = get_parameter_value::<i32>(&self.params, "minigun_max_events").max(1) as u32;
        let pending = std::mem::take(&mut self.pending);
        let mut counts: HashMap<(u64, usize), (usize, u32)> = HashMap::new(); // -> (index of first, events)
        for (i, shot) in pending.iter().enumerate() {
            let e = counts.entry((shot.sid, shot.weapon_ent)).or_insert((i, 0));
            e.1 += 1;
        }

        let detections = Vec::new();
        for ((sid, weapon_ent), (first_idx, events)) in counts {
            let shot = &pending[first_idx];
            if shot.class_name == "CTFMinigun" {
                if mg_rate {
                    let dq = self.mg_events.entry((sid, weapon_ent)).or_default();
                    for _ in 0..events {
                        dq.push_back(tick);
                    }
                    while dq.front().is_some_and(|t| tick.saturating_sub(*t) > mg_window) {
                        dq.pop_front();
                    }
                    let n = dq.len() as u32;
                    let e = self.mg_max.entry(sid).or_insert((0, demo_tick));
                    if n > e.0 {
                        *e = (n, demo_tick);
                    }
                    if n > mg_max_events && self.mg_cool.get(&sid).map_or(true, |t| tick >= *t) {
                        self.mg_cool.insert(sid, tick + mg_window);
                        *self.weights.entry(sid).or_default() += 2;
                        self.buffered.push((Detection {
                            tick: demo_tick,
                            algorithm: self.algorithm_name().to_string(),
                            player: sid,
                            data: build_data(vec![
                                ("class", Value::from(shot.player_class)),
                                ("weapon", Value::from(shot.weapon_name.clone())),
                                ("type", Value::from("rate")),
                                ("events_in_window", Value::from(n)),
                                ("window_ticks", Value::from(mg_window)),
                                ("max_events", Value::from(mg_max_events)),
                                ("weapon_class", Value::from(shot.class_name.clone())),
                                ("dt_weapon", Value::from(true)),
                            ]),
                        }, 2));
                    }
                }
                continue;
            }
            let Some(profile) = weapon_profile(&shot.class_name, shot.item) else {
                continue;
            };
            // events right after a weapon switch may belong to the previous weapon
            if let Some((ent, since)) = self.active.get(&sid) {
                if *ent != weapon_ent || tick.saturating_sub(*since) < switch_guard {
                    continue;
                }
            }

            // number of shots represented by this snapshot's events
            let shots_in_frame = if profile.events_per_shot > 1 {
                if events < profile.events_per_shot {
                    // half of a pair that straddles two snapshots, cannot be timed reliably
                    continue;
                }
                events / profile.events_per_shot
            } else {
                events
            };

            let mut kind: Option<&str> = None;
            let mut interval = 0u32;
            let mut threshold = 0i64;

            if same_frame_on
                && shots_in_frame >= 2
                && (gap as f32) < (shots_in_frame - 1) as f32 * profile.floor as f32 * same_frame_ratio
            {
                kind = Some("same_frame");
                interval = 0;
                threshold = (profile.floor as f32 * same_frame_ratio) as i64;
            }

            let key = (sid, weapon_ent);
            if kind.is_none() && profile.interval {
                if let Some(last) = self.last.get(&key) {
                    interval = tick.saturating_sub(last.tick);
                    let slack = gap.saturating_sub(1) + last.gap.saturating_sub(1);
                    threshold = (profile.floor as f32 * max_floor_ratio).floor() as i64 - tolerance as i64 - slack as i64;
                    // very fast weapons leave no usable margin, an interval there proves nothing
                    if threshold >= 6 && (interval as i64) < threshold {
                        kind = Some("short_interval");
                    }
                }
            }

            // for a same-snapshot flag both shots are in this snapshot, so tick_1 == tick_2 and tick_diff is 0
            let prev_tick = if kind == Some("same_frame") {
                demo_tick
            } else {
                self.last.get(&key).map(|l| l.demo_tick).unwrap_or(demo_tick)
            };
            self.last.insert(key, LastShot { tick, demo_tick, gap });

            if let Some(kind) = kind {
                // largest simulation tick step within the 30 ticks before this shot (lag spike / choke)
                let sim_gap = self
                    .sim_hist
                    .get(&sid)
                    .map(|h| {
                        let mut best = 0i64;
                        let mut prev: Option<i64> = None;
                        for &(t, sim) in h.iter().filter(|(t, _)| tick.saturating_sub(*t) <= 30) {
                            let _ = t;
                            if let Some(p) = prev {
                                // steps over 100 ticks are decode glitches (spawns), not lag
                                if sim - p < 100 {
                                    best = best.max(sim - p);
                                }
                            }
                            prev = Some(sim);
                        }
                        best
                    })
                    .unwrap_or(0);
                // A lag burst (choke / packet loss) makes the server run the queued commands of several ticks in one
                // frame, so legit shots can land close together. If the shooter's simulation time advanced by
                // at least the missing refire time right before the shot, the shots were legitimately spaced.
                let missing = profile.floor.saturating_sub(interval) as f32 * (shots_in_frame.max(2) - 1) as f32;
                if lag_excuse && sim_gap >= ((missing * lag_excuse_ratio) as i64).max(4) {
                    continue;
                }
                let weight = if kind == "same_frame" { 2 } else { 1 };
                *self.weights.entry(sid).or_default() += weight;
                self.buffered.push((Detection {
                    tick: demo_tick,
                    algorithm: self.algorithm_name().to_string(),
                    player: sid,
                    data: build_data(vec![
                        ("class", Value::from(shot.player_class)),
                        ("weapon", Value::from(shot.weapon_name.clone())),
                        ("type", Value::from(kind)),
                        ("tick_1", Value::from(prev_tick)),
                        ("tick_2", Value::from(demo_tick)),
                        ("tick_diff", Value::from(interval)), // in server ticks
                        ("min_interval", Value::from(profile.floor)),
                        ("threshold", Value::from(threshold)),
                        ("shots_in_snapshot", Value::from(shots_in_frame)),
                        ("snapshot_gap", Value::from(gap)),
                        ("interval_ratio", Value::from(if profile.floor > 0 { interval as f32 / profile.floor as f32 } else { 0.0 })),
                        ("ping_ms", Value::from(shot.ping)),
                        ("sim_gap_ticks", Value::from(sim_gap)),
                        ("weapon_class", Value::from(shot.class_name.clone())),
                        ("dt_weapon", Value::from(dt_weapon(&shot.class_name))),
                    ]),
                }, weight));
            }
        }
        Ok(detections)
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::GameEvent, MessageType::TempEntities])
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        tick: tf_demo_parser::demo::data::DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        match message {
            Message::TempEntities(_) => {
                for te in extract_te_shots(message, parser_state, state) {
                    if !te.is_primary_attack() {
                        continue;
                    }
                    let Some(player) = state.players.iter().find(|p| player_sid(p) == Some(te.sid))
                    else {
                        continue;
                    };
                    let Some(ent) = player.active_weapon else {
                        continue;
                    };
                    let Some(weapon) = state.weapons.get(&ent) else {
                        continue;
                    };
                    self.pending.push(PendingShot {
                        ping: player.ping as u32,
                        sid: te.sid,
                        weapon_ent: usize::from(ent),
                        class_name: weapon.class_name.clone(),
                        item: weapon.item_def_index,
                        weapon_name: weapon.name(player.class),
                        player_class: player.class_name(),
                    });
                }
                Ok(vec![])
            }
            Message::GameEvent(event_msg) => {
                let GameEvent::PlayerHurt(hurt) = &event_msg.event else {
                    return Ok(vec![]);
                };
                // legacy damage based path, minigun only
                let min_tick_heavy = get_parameter_value::<i32>(&self.params, "min_tick_heavy");
                if min_tick_heavy <= 0 || hurt.weapon_id != 18 {
                    return Ok(vec![]);
                }
                if get_parameter_value::<bool>(&self.params, "ignore_mvm") && is_mvm(state) {
                    return Ok(vec![]);
                }
                let assert_pvs: bool = get_parameter_value(&self.params, "assert_pvs");
                // timed in server ticks, see on_tick
                let ticknum = state.server_tick;
                let demo_tick = u32::from(tick);
                let dmg = hurt.damage_amount as u32;
                let weapon = hurt.weapon_id as u32;
                let attacker_uid = u32::from(hurt.attacker);
                let victim_uid = u32::from(hurt.user_id);

                let (Some(attacker_sid), Some(_)) = (
                    state.get_id64_from_userid(attacker_uid.into()),
                    state.get_id64_from_userid(victim_uid.into()),
                ) else {
                    return Ok(vec![]);
                };
                let Some(attacker) = state
                    .players
                    .iter()
                    .find(|p| player_sid(p) == Some(attacker_sid))
                else {
                    return Ok(vec![]);
                };
                if assert_pvs && !attacker.in_pvs {
                    return Ok(vec![]);
                }

                let shots = self.heavy_shots.entry(attacker_sid).or_default();
                let past = shots.clone();
                shots.clear();
                shots.extend([ticknum, weapon, victim_uid, dmg]);

                let Ok([past_tick, past_weapon, past_victim, past_dmg]): Result<[u32; 4], _> =
                    past.try_into()
                else {
                    return Ok(vec![]);
                };
                if past_weapon != weapon || past_victim != victim_uid {
                    return Ok(vec![]);
                }
                let diff = ticknum.saturating_sub(past_tick);
                if diff < min_tick_heavy as u32 && diff > 0 {
                    return Ok(vec![Detection {
                        tick: demo_tick,
                        algorithm: self.algorithm_name().to_string(),
                        player: attacker_sid,
                        data: build_data(vec![
                            ("class", Value::from(attacker.class_name())),
                            ("weapon", Value::from(state.get_player_weapon(attacker))),
                            ("type", Value::from("damage_events")),
                            ("tick_1", Value::from(past_tick)),
                            ("tick_2", Value::from(ticknum)),
                            ("tick_diff", Value::from(diff)),
                            ("victim", Value::from(victim_uid)),
                            ("weapon_id", Value::from(weapon)),
                            ("dt_weapon", Value::from(true)),
                            ("damage_1", Value::from(past_dmg)),
                            ("damage_2", Value::from(dmg)),
                        ]),
                    }]);
                }
                Ok(vec![])
            }
            _ => Ok(vec![]),
        }
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let min_events = get_parameter_value::<i32>(&self.params, "min_events").max(1) as u32;
        let weights = &self.weights;
        let mut out: Vec<Detection> = std::mem::take(&mut self.buffered)
            .into_iter()
            .filter(|(d, _)| weights.get(&d.player).copied().unwrap_or(0) >= min_events)
            .map(|(d, _)| d)
            .collect();
        if get_parameter_value::<bool>(&self.params, "minigun_stats") {
            for (sid, (n, at)) in &self.mg_max {
                out.push(Detection {
                    tick: *at,
                    algorithm: self.algorithm_name().to_string(),
                    player: *sid,
                    data: build_data(vec![
                        ("type", Value::from("minigun_stats")),
                        ("max_events_in_window", Value::from(*n)),
                    ]),
                });
            }
        }
        out.sort_by_key(|d| d.tick);
        Ok(out)
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }
}
