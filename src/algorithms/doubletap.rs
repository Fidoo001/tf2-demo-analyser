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
//  * Minigun: about ONE EVENT PER TICK while firing (1.14 on average), not per shot. Events may reach the recorder
//    late and in batches (one batch can hold 10+ ticks worth of a player's events, sometimes split over two
//    messages), so a delivery is compared with the time since an EARLIER DELIVERY of that player's events, not
//    with the time since the previous snapshot. A delivery is flagged only when no trailing window (back to an
//    earlier delivery, at most `minigun_window_ticks`, never before the minigun started firing) explains its
//    events. A lag catch-up (simulation time jumping) also explains extra events.
use std::collections::{HashMap, VecDeque};

use anyhow::Error;
use serde_json::{json, Map, Value};
use steamid_ng::SteamID;
use tf_demo_parser::demo::message::packetentities::{EntityId, UpdateType};
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::sendprop::SendPropIdentifier;
use tf_demo_parser::{MessageType, ParserState};

use crate::base::cheat_analyser_base::{CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::handle_to_entid;

const TICKS_PER_SECOND: f32 = 66.67;

/// Attack events a firing minigun emits per tick of simulated time (measured on real demos: median 1.14, 99th percentile
/// 1.21 over 33-tick windows). It emits about one event per tick, not one per shot, and slightly more than one.
const MINIGUN_EVENTS_PER_TICK: f32 = 1.2;
/// `m_iWeaponState` of a minigun that is firing (AC_STATE_FIRING). Attack events are only emitted in this state.
const MINIGUN_FIRING: i64 = 2;
const MINIGUN_STATE: SendPropIdentifier = SendPropIdentifier::new("DT_WeaponMinigun", "m_iWeaponState");
/// Conditions that change fire rate or movement (Mannpower runes, Halloween spells): never judged.
const IGNORED_CONDS: [u32; 7] = [34, 72, 81, 91, 96, 109, 113];
/// Player confidence is a logistic curve over the evidence score, centered at `CONF_MID`.
const CONF_MID: f32 = 16.0;
const CONF_SLOPE: f32 = 3.0;

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
    /// tick since which the player has been alive and in PVS (their weapon and state are known)
    ok_since: HashMap<u64, u32>,
    /// shots up to this tick are not judged because of an ignored condition
    cond_until: HashMap<u64, u32>,
    /// auto weapons: event ticks inside the rolling window, and the tick the next rate flag is allowed
    auto_events: HashMap<(u64, usize), (VecDeque<u32>, u32)>,
    /// recent deliveries (tick, attack events) per (player, weapon entity), newest last
    deliveries: HashMap<(u64, usize), VecDeque<(u32, u32)>>,
    /// latest networked `m_iWeaponState` per minigun entity
    minigun_state: HashMap<usize, i64>,
    /// per minigun entity: (tick, server ticks covered) of the recent snapshots in which it was firing
    firing_hist: HashMap<usize, VecDeque<(u32, u32)>>,
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
                // a player is reported only when their flags add up to this confidence (percent): many flags across several
                // separate episodes score high, one burst that never repeats scores low
                f("min_confidence", 70.0),
                // flags closer together than this many ticks belong to the same episode
                i("episode_gap_ticks", 1000),
                // when this many players reach min_confidence in one demo the cause is the server or map (a fire rate effect no condition
                // shows), not cheaters, so nothing is reported. 0 = off (needed for demos of servers full of cheaters)
                i("max_flagged_players", 5),
                // shots are ignored until this many ticks after the player is alive and in PVS again (their weapon is unknown before)
                i("pvs_grace_ticks", 7),
                // shots are ignored for this many ticks after an ignored condition ends
                i("cond_guard_ticks", 33),
                // SMG, pistols, syringe gun: shots inside the window may exceed the weapon's maximum by at most `auto_slack`
                // (real play peaks at the maximum +1)
                b("auto_rate", true),
                i("auto_window_ticks", 66),
                i("auto_slack", 2),
                // minigun: events in a snapshot may exceed the ticks since the previous snapshot by at most this much minus one
                // (real play: 99.9% of snapshots are within 4, and every confirmed doubletap burst was 5 or more)
                b("minigun_burst", true),
                i("minigun_excess", 5),
                // minigun: longest trailing window that may explain a delivery's events (covers late, batched delivery)
                i("minigun_window_ticks", 20),
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
fn push_key(data: &mut Value, k: &str, v: Value) {
    if let Value::Object(m) = data {
        m.insert(format!("{}{}", "\u{200B}".repeat(m.len()), k), v);
    }
}

fn has_ignored_cond(p: &Player) -> bool {
    let bits = [p.cond, p.cond_ex, p.cond_ex2, p.cond_ex3];
    IGNORED_CONDS.iter().any(|&c| bits[(c / 32) as usize] >> (c % 32) & 1 != 0)
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

/// Most minigun attack events a window of `ticks` server ticks can legitimately hold (about one per tick, at most
/// `MINIGUN_EVENTS_PER_TICK`), before the `minigun_excess` slack. For 1-4 ticks this is the tick count itself.
fn minigun_allowed(ticks: u32) -> u32 {
    ticks + (ticks as f32 * (MINIGUN_EVENTS_PER_TICK - 1.0)).floor() as u32
}

/// Server ticks inside the last `ticks` ticks before `tick` during which the minigun was networked as firing.
/// `None` (state never seen) does not limit anything.
fn firing_ticks(firing: Option<&VecDeque<(u32, u32)>>, tick: u32, ticks: u32) -> u32 {
    let Some(hist) = firing else { return ticks };
    let covered: u32 = hist.iter().rev().take_while(|(t, _)| tick.saturating_sub(*t) < ticks).map(|(_, n)| *n).sum();
    covered.min(ticks)
}

/// The trailing window that best explains a minigun delivery of `events` at `tick`: for every earlier delivery
/// inside `window` ticks, the events delivered after it against the ticks since it, plus the whole window. A window
/// is only credited with the ticks the minigun spent FIRING inside it (at least the snapshot gap, which is what the
/// check credited before windows existed): attack events are only emitted while firing, and late delivery only ever
/// makes events arrive after the state. Tick shifting fires the whole burst while the networked state is still
/// spinning or idle.
/// Returns (excess over the allowed events, credited ticks, events in that window) of the best window.
fn minigun_best_window(prev: &VecDeque<(u32, u32)>, tick: u32, events: u32, gap: u32, window: u32, firing: Option<&VecDeque<(u32, u32)>>) -> (i64, u32, u32) {
    let mut total = events;
    let mut best: Option<(i64, u32, u32)> = None;
    let mut consider = |ticks: u32, total: u32| {
        let ticks = ticks.min(firing_ticks(firing, tick, ticks).max(gap)).max(1);
        let excess = total as i64 - minigun_allowed(ticks) as i64;
        if best.map_or(true, |(e, ..)| excess < e) {
            best = Some((excess, ticks, total));
        }
    };
    for &(t, n) in prev.iter().rev() {
        let since = tick.saturating_sub(t);
        if since == 0 {
            continue;
        }
        if since > window {
            break;
        }
        // events delivered after this earlier delivery, against the time since it
        consider(since, total);
        total += n;
    }
    // everything delivered inside the window (or since the previous delivery if that is older)
    consider(window, total);
    best.unwrap_or((events as i64 - minigun_allowed(gap) as i64, gap, events))
}

impl<'a> CheatAlgorithm<'a> for DoubleTap {
    fn algorithm_name(&self) -> &str {
        "doubletap"
    }

    fn tooltip(&self) -> &'static str {
        "Detects tick shifting (doubletap): a weapon fired again long before its refire delay has elapsed,          several shots inside one snapshot, or more shots than the weapon can fire in a time window.          Bursts explained by lag catch-ups are ignored, and a player is only reported once their flags add up to enough confidence."
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _: &ParserState) -> Result<Vec<Detection>, Error> {
        // Shots are timed in SERVER ticks. The demo tick is the recorder's client tick: when the client stalls
        // (alt-tab, lag burst) it freezes while dozens of server snapshots are processed, so shots that are a
        // whole refire delay apart can all carry the same demo tick.
        let demo_tick = u32::from(state.tick);
        let tick = state.server_tick;
        let gap = tick.saturating_sub(self.prev_frame_tick).max(1);
        self.prev_frame_tick = tick;
        let cond_guard = get_parameter_value::<i32>(&self.params, "cond_guard_ticks").max(0) as u32;

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
            if p.in_pvs && p.state == PlayerState::Alive {
                self.ok_since.entry(sid).or_insert(tick);
            } else {
                self.ok_since.remove(&sid);
            }
            if has_ignored_cond(p) {
                self.cond_until.insert(sid, tick + cond_guard);
            }
        }

        let history = get_parameter_value::<i32>(&self.params, "minigun_window_ticks").max(1) as u32 + 2;
        self.firing_hist.retain(|ent, _| self.minigun_state.contains_key(ent));
        for (&ent, &st) in &self.minigun_state {
            let hist = self.firing_hist.entry(ent).or_default();
            if st == MINIGUN_FIRING {
                hist.push_back((tick, gap));
            }
            while hist.front().is_some_and(|(t, _)| tick.saturating_sub(*t) > history) {
                hist.pop_front();
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
        let (switch_guard, tolerance, auto_window, auto_slack, minigun_excess, pvs_grace, minigun_window) = (
            int("switch_guard_ticks"),
            int("tolerance_ticks"),
            int("auto_window_ticks").max(1),
            int("auto_slack"),
            int("minigun_excess").max(1),
            int("pvs_grace_ticks"),
            int("minigun_window_ticks").max(1),
        );
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
            // every delivery is remembered, also the ones not judged below: a later batch is timed against it
            let prev_deliveries = {
                let dq = self.deliveries.entry(key).or_default();
                while dq.front().is_some_and(|(t, _)| tick.saturating_sub(*t) > minigun_window) {
                    dq.pop_front();
                }
                let prev = dq.clone();
                dq.push_back((tick, events));
                prev
            };
            // players out of sight or dead have a stale weapon and position, and ignored conditions change fire rates
            if self.ok_since.get(&sid).map_or(true, |s| tick.saturating_sub(*s) < pvs_grace) || self.cond_until.get(&sid).is_some_and(|u| tick <= *u) {
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

            // (type, interval, shots reported, refire time missing in ticks)
            let mut hit: Option<(&str, u32, u32, f32)> = None;
            // minigun: (window ticks, events in the window) of the window that best explains the delivery
            let mut minigun_window_info = None;
            if profile.kind == Kind::Minigun {
                if minigun_burst {
                    let firing = self.firing_hist.get(&weapon_ent);
                    let (excess, ticks, total) = minigun_best_window(&prev_deliveries, tick, events, gap, minigun_window, firing);
                    if excess >= minigun_excess as i64 {
                        hit = Some(("minigun_burst", ticks, events, (total - minigun_allowed(ticks)) as f32));
                        minigun_window_info = Some((ticks, total));
                    }
                }
            } else {
                if same_frame_on && shots >= 2 && (gap as f32) < (shots - 1) as f32 * floor as f32 * same_frame_ratio {
                    hit = Some(("same_frame", 0, shots, floor as f32 * (shots - 1) as f32));
                }
                if profile.kind == Kind::Semi && hit.is_none() {
                    if let Some(last) = self.last.get(&key) {
                        let interval = tick.saturating_sub(last.tick);
                        let slack = gap.saturating_sub(1) + last.gap.saturating_sub(1);
                        let threshold = (floor as f32 * max_floor_ratio).floor() as i64 - tolerance as i64 - slack as i64;
                        // very fast weapons leave no usable margin, an interval there proves nothing
                        if threshold >= 6 && (interval as i64) < threshold {
                            hit = Some(("short_interval", interval, shots, floor.saturating_sub(interval) as f32 * (shots.max(2) - 1) as f32));
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
                        hit = Some(("rate", auto_window, n, (n - legal) as f32 * profile.ticks));
                    }
                }
            }

            // for a same-snapshot flag both shots are in this snapshot, so tick_1 == tick_2 and tick_diff is 0
            let prev_tick = match hit {
                Some(("same_frame", ..)) => demo_tick,
                _ => self.last.get(&key).map_or(demo_tick, |l| l.demo_tick),
            };
            self.last.insert(key, LastShot { tick, demo_tick, gap });

            let Some((kind, interval, shots_reported, missing)) = hit else {
                continue;
            };
            // A lag burst (choke / packet loss) makes the server run the queued commands of several ticks in one
            // frame, so legit shots can land close together. If the shooter's simulation time advanced by
            // at least the missing refire time right before the shot, the shots were legitimately spaced.
            let sim_gap = self.sim_hist.get(&sid).map_or(0, |h| max_sim_step(h, tick));
            let excused = if kind == "minigun_burst" {
                // the minigun emits events per tick of simulated time, so a catch-up of `sim_gap` ticks legitimately holds
                // that many events (at the measured rate), however few ticks passed between snapshots
                // (a catch-up explains the delivery's own events; the earlier events of its window are excused with it)
                let legit = |ticks: u32| (MINIGUN_EVENTS_PER_TICK * (ticks as f32).max(sim_gap as f32 / lag_excuse_ratio)).ceil() as u32;
                let (ticks, total) = minigun_window_info.unwrap_or((gap, shots_reported));
                sim_gap >= 4 && (shots_reported <= legit(gap) || total <= legit(ticks))
            } else {
                sim_gap >= ((missing * lag_excuse_ratio) as i64).max(4)
            };
            if lag_excuse && excused {
                continue;
            }
            let weight = if kind == "same_frame" || kind == "minigun_burst" || kind == "rate" { 2 } else { 1 };
            let mut data = Value::Object(Map::new());
            push_key(&mut data, "weapon", json!(shot.weapon_name));
            push_key(&mut data, "type", json!(kind));
            push_key(&mut data, "tick_1", json!(prev_tick));
            push_key(&mut data, "tick_2", json!(demo_tick));
            // server ticks, unlike tick_1/tick_2 (the window for rate, the snapshot gap for minigun_burst)
            push_key(&mut data, "server_ticks", json!(interval));
            push_key(&mut data, "min_server_ticks", json!(floor));
            if kind != "short_interval" {
                // events in the window for rate, in the delivery for minigun_burst
                push_key(&mut data, "shots", json!(shots_reported));
            }
            if let Some((_, total)) = minigun_window_info {
                // events delivered inside the window that explains the delivery best (server_ticks long)
                push_key(&mut data, "window_shots", json!(total));
            }
            push_key(&mut data, "ping_ms", json!(shot.ping));
            self.buffered.push((Detection { tick: demo_tick, algorithm: self.algorithm_name().to_string(), player: sid, data }, weight));
        }
        Ok(vec![])
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        Ok(vec![MessageType::TempEntities, MessageType::PacketEntities])
    }

    fn on_message(
        &mut self,
        message: &Message,
        state: &CheatAnalyserState,
        parser_state: &ParserState,
        _: tf_demo_parser::demo::data::DemoTick,
    ) -> Result<Vec<Detection>, Error> {
        if let Message::PacketEntities(msg) = message {
            // minigun firing state: attack events cannot be older than the tick the minigun started firing
            for entity in &msg.entities {
                let ent = usize::from(entity.entity_index);
                match entity.update_type {
                    UpdateType::Delete | UpdateType::Leave => {
                        self.minigun_state.remove(&ent);
                    }
                    UpdateType::Enter => {
                        let is_minigun = parser_state
                            .server_classes
                            .get(usize::from(entity.server_class))
                            .is_some_and(|c| c.name.as_str() == "CTFMinigun");
                        if is_minigun {
                            // entering entities are merged with their baseline, so the state is always known
                            let v = entity.props(parser_state).find(|p| p.identifier == MINIGUN_STATE).and_then(|p| i64::try_from(&p.value).ok());
                            self.minigun_state.insert(ent, v.unwrap_or(0));
                        }
                    }
                    // only miniguns (registered when they entered) are scanned, the state is only sent when it changes
                    _ if self.minigun_state.contains_key(&ent) => {
                        if let Some(v) = entity.props.iter().find(|p| p.identifier == MINIGUN_STATE).and_then(|p| i64::try_from(&p.value).ok()) {
                            self.minigun_state.insert(ent, v);
                        }
                    }
                    _ => {}
                }
            }
            return Ok(vec![]);
        }
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
            });
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let episode_gap = get_parameter_value::<i32>(&self.params, "episode_gap_ticks").max(1) as u32;
        let min_confidence = get_parameter_value::<f32>(&self.params, "min_confidence");
        let max_players = get_parameter_value::<i32>(&self.params, "max_flagged_players").max(0) as usize;
        let mut by_player: HashMap<u64, Vec<(Detection, u32)>> = HashMap::new();
        for (d, weight) in std::mem::take(&mut self.buffered) {
            by_player.entry(d.player).or_default().push((d, weight));
        }
        let mut players = Vec::new();
        for (_, mut flags) in by_player {
            flags.sort_by_key(|(d, _)| d.tick);
            let weight: u32 = flags.iter().map(|(_, w)| w).sum();
            let episodes = 1 + flags.windows(2).filter(|w| w[1].0.tick - w[0].0.tick > episode_gap).count() as u32;
            // repeated flags in separate episodes are what a real doubletapper produces, a single burst barely counts
            let evidence = ((episodes - 1) * weight) as f32 + (weight * weight) as f32 / 64.0;
            let confidence = (100.0 / (1.0 + (-(evidence - CONF_MID) / CONF_SLOPE).exp())).min(99.99);
            if confidence >= min_confidence {
                players.push((flags, confidence));
            }
        }
        if max_players > 0 && players.len() >= max_players {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        for (flags, confidence) in players {
            for (mut d, _) in flags {
                push_key(&mut d.data, "confidence", json!((confidence as f64 * 100.0).round() / 100.0));
                out.push(d);
            }
        }
        out.sort_by_key(|d| d.tick);
        Ok(out)
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }
}
