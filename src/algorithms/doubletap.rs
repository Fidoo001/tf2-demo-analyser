// Originally written by Tellta, rewritten to be weapon agnostic and tick-shift aware.
//
// Doubletap (tick shifting) fires a weapon again long before its refire delay has elapsed, because the
// shifted ticks make the server believe enough time has passed. This works on the *shot timeline*:
//
//  * every attack animation event (CTEPlayerAnimEvent, event 0) is a shot, whether it hits or not
//  * the demo only shows snapshots, so shots are grouped per snapshot ("frame")
//  * every weapon is judged against the FASTEST refire it can legally reach (see `weapon_profile`)
//
// How a weapon is judged depends on how its attack events relate to its shots (measured on real demos):
//  * Semi (rockets, scatterguns, snipers, ...): one event per shot, slower than a snapshot. Two shots closer
//    together than the refire interval are impossible, and so are two shots inside ONE snapshot.
//  * Auto (SMG, pistols, syringe gun): one event per shot, faster than a snapshot, so intervals prove
//    nothing. Shots inside a rolling window may not exceed what the weapon can fire in that time.
//  * Minigun: about ONE EVENT PER TICK while firing (1.14 on average), not per shot, so a snapshot may not hold many
//    more events than the ticks that elapsed since the previous snapshot. A lag catch-up (simulation time jumping)
//    explains that many extra events, so only bursts the simulation clock cannot account for are flagged.
use std::collections::{HashMap, VecDeque};

use anyhow::Error;
use serde_json::{Map, Value};
use steamid_ng::SteamID;
use tf_demo_parser::demo::message::packetentities::EntityId;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::handle_to_entid;

const TICKS_PER_SECOND: f32 = 66.67;

/// Attack events a firing minigun emits per tick of simulated time (measured on real demos: median 1.14, 99th percentile
/// 1.21 over 33-tick windows). It emits about one event per tick, not one per shot, and slightly more than one.
const MINIGUN_EVENTS_PER_TICK: f32 = 1.2;

/// Collects the shooters of the primary attack events (PLAYERANIMEVENT_ATTACK_PRIMARY = 0) of a TempEntities message.
/// Every event of a message is delta encoded against the previous event of the same class (zero for the
/// first one), so a prop that is missing from an event keeps the value of the previous event. Reading a
/// missing `m_iEvent` as 0 turns every repeated reload event into an "attack".
fn te_attackers(message: &Message, parser_state: &ParserState, state: &CheatAnalyserState) -> Vec<(u64, EntityId)> {
    const ANIM_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_hPlayer");
    const ANIM_EVENT: SendPropIdentifier = SendPropIdentifier::new("DT_TEPlayerAnimEvent", "m_iEvent");
    let Message::TempEntities(msg) = message else {
        return vec![];
    };
    let mut out = Vec::new();
    let mut prev_class = None;
    let (mut handle, mut anim_event) = (0u32, 0i64);
    for event in &msg.events {
        if prev_class != Some(event.class_id) {
            (handle, anim_event, prev_class) = (0, 0, Some(event.class_id));
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
        let entid = handle_to_entid(handle);
        if let Some(sid) = state.entid_to_userid.get(&entid).and_then(|uid| state.userid_to_id64.get(uid)) {
            if anim_event == 0 {
                out.push((*sid, entid));
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Semi,
    Auto,
    Minigun,
}

#[derive(Clone, Copy)]
struct Profile {
    kind: Kind,
    /// fastest legitimate time between two shots, in ticks (1 tick = 15ms)
    ticks: f32,
    /// attack events one shot produces (launchers, bows, the cannon and the Dragon's Fury send every shot twice)
    events_per_shot: u32,
}

/// Fastest legitimate refire per weapon class. Base intervals are the wiki's attack intervals (and match the
/// smallest intervals seen in real demos); the best fire rate a weapon can reach is applied on top, using
/// the `fire rate bonus` delay multipliers from the item definitions (items_game).
/// Not judged: melee (swing speed has too many legitimate variations and exploits), cooldown weapons (jars, cleaver),
/// the flamethrower and builders. Unlisted classes are not judged.
fn weapon_profile(class_name: &str, item: Option<u16>) -> Option<Profile> {
    let (secs, kind, events_per_shot) = match class_name {
        "CTFScatterGun"
        | "CTFSodaPopper"
        | "CTFPEPBrawlerBlaster"
        | "CTFShotgun"
        | "CTFShotgun_Soldier"
        | "CTFShotgun_Pyro"
        | "CTFShotgun_HWG"
        | "CTFShotgun_Revenge"
        | "CTFShotgunBuildingRescue" => (0.625, Kind::Semi, 1),
        "CTFPistol_ScoutPrimary" => (0.36, Kind::Semi, 1), // Shortstop
        "CTFRocketLauncher" | "CTFRocketLauncher_DirectHit" | "CTFRocketLauncher_AirStrike" | "CTFParticleCannon" | "CTFRaygun" | "CTFDRGPomson" => {
            (0.8, Kind::Semi, 1)
        }
        "CTFGrenadeLauncher" | "CTFPipebombLauncher" => (0.6, Kind::Semi, 2),
        "CTFCannon" => (0.4, Kind::Semi, 2),          // Loose Cannon, lowered: its events come in pairs
        "CTFWeaponFlameBall" => (0.5, Kind::Semi, 2), // Dragon's Fury
        "CTFCompoundBow" => (1.0, Kind::Semi, 2),     // Huntsman, lowered: real shots were 2.1s apart or more
        "CTFCrossbow" => (1.2, Kind::Semi, 1),        // lowered: real shots were 1.7s apart or more
        "CTFFlareGun" | "CTFFlareGun_Revenge" => (2.0, Kind::Semi, 1),
        "CTFRevolver" => (0.5, Kind::Semi, 1),
        "CTFSniperRifle" | "CTFSniperRifleClassic" | "CTFSniperRifleDecap" => (1.5, Kind::Semi, 1),
        "CTFPistol" | "CTFPistol_Scout" | "CTFPistol_ScoutSecondary" => (0.15, Kind::Auto, 1),
        "CTFSMG" | "CTFChargedSMG" | "CTFSyringeGun" => (0.1, Kind::Auto, 1),
        "CTFMinigun" => (0.1, Kind::Minigun, 1),
        _ => return None,
    };
    let scale = match (class_name, item) {
        ("CTFRocketLauncher_AirStrike", _) => 0.35, // while blast jumping (0.28s)
        (_, Some(45 | 1078 | 448)) => 0.5,          // Force-a-Nature (+ Festive), Soda Popper
        (_, Some(130)) => 0.75,                     // Scottish Resistance
        (_, Some(730)) => 0.3,                      // Beggar's Bazooka (releasing a charge)
        (_, Some(425 | 773)) => 0.85,               // Family Business, Pretty Boy's Pocket Pistol
        _ => 1.0,
    };
    Some(Profile { kind, ticks: secs * scale * TICKS_PER_SECOND, events_per_shot })
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
    /// auto weapons: event ticks inside the rolling window, and the tick the next rate flag is allowed
    auto_events: HashMap<(u64, usize), (VecDeque<u32>, u32)>,
}

impl DoubleTap {
    pub fn new() -> Self {
        let (i, f, b) = (
            |n: &str, v: i32| (n.to_string(), Parameter::Int(v)),
            |n: &str, v: f32| (n.to_string(), Parameter::Float(v)),
            |n: &str, v: bool| (n.to_string(), Parameter::Bool(v)),
        );
        Self {
            params: HashMap::from([
                b("enabled", true),
                // slack subtracted from every weapon's refire floor to absorb snapshot/ping jitter
                i("tolerance_ticks", 0),
                // skip shots that follow a simulation time jump covering the missing refire time (lag burst)
                b("lag_excuse", true),
                f("lag_excuse_ratio", 1.0),
                // MvM fire rate upgrades make fast refires legal, so those demos are not judged
                b("ignore_mvm", true),
                // a shot pair is flagged when its interval is below this fraction of the weapon's fastest legal
                // interval (lag can squeeze a legit pair, so only clear violations are reported)
                f("max_floor_ratio", 0.8),
                // detect several shots inside one snapshot, and the fraction of the refire floor that must fit between them
                b("same_frame", true),
                f("same_frame_ratio", 0.5),
                // shots right after a weapon switch are attributed to the wrong weapon, ignore them
                i("switch_guard_ticks", 3),
                // evidence needed before a player is reported at all (same_frame counts 2, short_interval 1).
                // Default 1 = report everything; incident grouping / suspicion scoring downstream decides.
                i("min_events", 1),
                // SMG, pistols, syringe gun: shots inside the window may exceed the weapon's maximum by at most `auto_slack`
                // (real play peaks at the maximum +1)
                b("auto_rate", true),
                i("auto_window_ticks", 66),
                i("auto_slack", 2),
                // minigun: events in a snapshot may exceed the ticks since the previous snapshot by at most this much minus one
                // (real play: 99.9% of snapshots are within 4, and every confirmed doubletap burst was 5 or more)
                b("minigun_burst", true),
                i("minigun_excess", 5),
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

/// Keys are padded with zero-width spaces so they keep their order in the output.
fn build_data(items: Vec<(&str, Value)>) -> Value {
    let mut m = Map::new();
    for (i, (k, v)) in items.into_iter().enumerate() {
        m.insert(format!("{}{}", "\u{200B}".repeat(i), k), v);
    }
    Value::Object(m)
}

/// Largest simulation tick step in the 30 ticks before `tick` (lag spike / choke).
fn max_sim_step(hist: &VecDeque<(u32, i64)>, tick: u32) -> i64 {
    let mut best = 0;
    let mut prev: Option<i64> = None;
    for &(_, sim) in hist.iter().filter(|(t, _)| tick.saturating_sub(*t) <= 30) {
        // steps over 100 ticks are decode glitches (spawns), not lag
        if let Some(p) = prev.filter(|p| sim - p < 100) {
            best = best.max(sim - p);
        }
        prev = Some(sim);
    }
    best
}

impl<'a> CheatAlgorithm<'a> for DoubleTap {
    fn algorithm_name(&self) -> &str {
        "doubletap"
    }

    fn tooltip(&self) -> &'static str {
        "Detects tick shifting (doubletap): a weapon fired again long before its refire delay has elapsed, \
         several shots inside one snapshot, or more shots than the weapon can fire in a time window. \
         Bursts explained by lag catch-ups are ignored."
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        // Shots are timed in SERVER ticks. The demo tick is the recorder's client tick: when the client stalls
        // (alt-tab, lag burst) it freezes while dozens of server snapshots are processed, so shots that are a
        // whole refire delay apart can all carry the same demo tick.
        let demo_tick = u32::from(state.tick);
        let tick = state.server_tick;
        let gap = tick.saturating_sub(self.prev_frame_tick).max(1);
        self.prev_frame_tick = tick;

        for p in &state.players {
            let Some(sid) = player_sid(p) else { continue };
            let sim = state.sim_tick(p);
            let h = self.sim_hist.entry(sid).or_default();
            if h.back().map_or(true, |(_, s)| *s != sim) {
                h.push_back((tick, sim));
            }
            while h.front().is_some_and(|(t, _)| tick.saturating_sub(*t) > 45) {
                h.pop_front();
            }
            if let Some(ent) = p.active_weapon {
                let ent = usize::from(ent);
                if self.active.get(&sid).map_or(true, |(e, _)| *e != ent) {
                    self.active.insert(sid, (ent, tick));
                }
            }
        }

        if self.pending.is_empty() {
            return Ok(vec![]);
        }
        let params = &self.params;
        if get_parameter_value::<bool>(params, "ignore_mvm") && is_mvm(state) {
            self.pending.clear();
            return Ok(vec![]);
        }
        let int = |n: &str| get_parameter_value::<i32>(params, n).max(0) as u32;
        let flt = |n: &str| get_parameter_value::<f32>(params, n);
        let flag = |n: &str| get_parameter_value::<bool>(params, n);
        let (switch_guard, tolerance, auto_window, auto_slack, minigun_excess) =
            (int("switch_guard_ticks"), int("tolerance_ticks"), int("auto_window_ticks").max(1), int("auto_slack"), int("minigun_excess").max(1));
        let (same_frame_on, auto_rate, minigun_burst, lag_excuse) =
            (flag("same_frame"), flag("auto_rate"), flag("minigun_burst"), flag("lag_excuse"));
        let (same_frame_ratio, max_floor_ratio, lag_excuse_ratio) = (flt("same_frame_ratio"), flt("max_floor_ratio"), flt("lag_excuse_ratio"));

        let pending = std::mem::take(&mut self.pending);
        let mut counts: HashMap<(u64, usize), (usize, u32)> = HashMap::new(); // -> (index of first, events)
        for (i, shot) in pending.iter().enumerate() {
            counts.entry((shot.sid, shot.weapon_ent)).or_insert((i, 0)).1 += 1;
        }

        for (key @ (sid, weapon_ent), (first_idx, events)) in counts {
            let shot = &pending[first_idx];
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
            let shots = if profile.events_per_shot > 1 {
                if events < profile.events_per_shot {
                    // half of a pair that straddles two snapshots, cannot be timed reliably
                    continue;
                }
                events / profile.events_per_shot
            } else {
                events
            };
            let floor = profile.ticks as u32;

            // (type, interval, threshold, shots reported, refire time missing in ticks)
            let mut hit: Option<(&str, u32, i64, u32, f32)> = None;
            if profile.kind == Kind::Minigun {
                if minigun_burst && events >= gap + minigun_excess {
                    hit = Some(("minigun_burst", gap, (gap + minigun_excess) as i64, events, (events - gap) as f32));
                }
            } else {
                if same_frame_on && shots >= 2 && (gap as f32) < (shots - 1) as f32 * floor as f32 * same_frame_ratio {
                    hit = Some(("same_frame", 0, (floor as f32 * same_frame_ratio) as i64, shots, floor as f32 * (shots - 1) as f32));
                }
                if profile.kind == Kind::Semi && hit.is_none() {
                    if let Some(last) = self.last.get(&key) {
                        let interval = tick.saturating_sub(last.tick);
                        let slack = gap.saturating_sub(1) + last.gap.saturating_sub(1);
                        let threshold = (floor as f32 * max_floor_ratio).floor() as i64 - tolerance as i64 - slack as i64;
                        // very fast weapons leave no usable margin, an interval there proves nothing
                        if threshold >= 6 && (interval as i64) < threshold {
                            hit = Some((
                                "short_interval",
                                interval,
                                threshold,
                                shots,
                                floor.saturating_sub(interval) as f32 * (shots.max(2) - 1) as f32,
                            ));
                        }
                    }
                }
                if profile.kind == Kind::Auto && auto_rate {
                    let (dq, cooldown) = self.auto_events.entry(key).or_default();
                    dq.extend(std::iter::repeat(tick).take(events as usize));
                    while dq.front().is_some_and(|t| tick.saturating_sub(*t) > auto_window) {
                        dq.pop_front();
                    }
                    let (n, legal) = (dq.len() as u32, (auto_window as f32 / profile.ticks).ceil() as u32);
                    if hit.is_none() && n > legal + auto_slack && tick >= *cooldown {
                        *cooldown = tick + auto_window;
                        hit = Some(("rate", auto_window, (legal + auto_slack) as i64, n, (n - legal) as f32 * profile.ticks));
                    }
                }
            }

            // for a same-snapshot flag both shots are in this snapshot, so tick_1 == tick_2 and tick_diff is 0
            let prev_tick = match hit {
                Some(("same_frame", ..)) => demo_tick,
                _ => self.last.get(&key).map_or(demo_tick, |l| l.demo_tick),
            };
            self.last.insert(key, LastShot { tick, demo_tick, gap });

            let Some((kind, interval, threshold, shots_reported, missing)) = hit else {
                continue;
            };
            // A lag burst (choke / packet loss) makes the server run the queued commands of several ticks in one
            // frame, so legit shots can land close together. If the shooter's simulation time advanced by
            // at least the missing refire time right before the shot, the shots were legitimately spaced.
            let sim_gap = self.sim_hist.get(&sid).map_or(0, |h| max_sim_step(h, tick));
            let excused = if kind == "minigun_burst" {
                // the minigun emits events per tick of simulated time, so a catch-up of `sim_gap` ticks legitimately holds
                // that many events (at the measured rate), however few ticks passed between snapshots
                let legit = (MINIGUN_EVENTS_PER_TICK * (gap as f32).max(sim_gap as f32 / lag_excuse_ratio)).ceil() as u32;
                sim_gap >= 4 && shots_reported <= legit
            } else {
                sim_gap >= ((missing * lag_excuse_ratio) as i64).max(4)
            };
            if lag_excuse && excused {
                continue;
            }
            let weight = if kind == "same_frame" || kind == "minigun_burst" || kind == "rate" { 2 } else { 1 };
            *self.weights.entry(sid).or_default() += weight;
            self.buffered.push((
                Detection {
                    tick: demo_tick,
                    algorithm: self.algorithm_name().to_string(),
                    player: sid,
                    data: build_data(vec![
                        ("class", Value::from(shot.player_class)),
                        ("weapon", Value::from(shot.weapon_name.clone())),
                        ("type", Value::from(kind)),
                        ("tick_1", Value::from(prev_tick)),
                        ("tick_2", Value::from(demo_tick)),
                        ("tick_diff", Value::from(interval)), // in server ticks (the window for rate, the snapshot gap for minigun_burst)
                        ("min_interval", Value::from(floor)),
                        ("threshold", Value::from(threshold)),
                        ("shots_in_snapshot", Value::from(shots_reported)), // events in the window for rate, in the snapshot for minigun_burst
                        ("snapshot_gap", Value::from(gap)),
                        ("interval_ratio", Value::from(if floor > 0 { interval as f32 / floor as f32 } else { 0.0 })),
                        ("ping_ms", Value::from(shot.ping)),
                        ("sim_gap_ticks", Value::from(sim_gap)),
                        ("weapon_class", Value::from(shot.class_name.clone())),
                    ]),
                },
                weight,
            ));
        }
        Ok(vec![])
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::TempEntities])
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        _: tf_demo_parser::demo::data::DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        for (sid, entid) in te_attackers(message, parser_state, state) {
            let Some(player) = state.players.iter().find(|p| p.entity == entid) else {
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
                sid,
                weapon_ent: usize::from(ent),
                class_name: weapon.class_name.clone(),
                item: weapon.item_def_index,
                weapon_name: weapon.name(player.class),
                player_class: player.class_name(),
            });
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let min_events = get_parameter_value::<i32>(&self.params, "min_events").max(1) as u32;
        let weights = &self.weights;
        let mut out: Vec<Detection> = std::mem::take(&mut self.buffered)
            .into_iter()
            .filter(|(d, _)| weights.get(&d.player).copied().unwrap_or(0) >= min_events)
            .map(|(d, _)| d)
            .collect();
        out.sort_by_key(|d| d.tick);
        Ok(out)
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }
}
