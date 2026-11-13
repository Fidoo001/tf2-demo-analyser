//! Aimbot on buildings: sentries, dispensers and teleporters of either team (auto sapper, hitscan
//! building aimbot, auto repair with the wrench). See docs/building_aimbot.md.
//!
//! Cheats aim at the centre of the building's collision box (Amalgam hitscan / melee / RijiN auto
//! sapper) or keep the pitch and only set the yaw towards it (Amalgam auto sapper). Two signatures:
//! - **lock**: the view yaw stays within `lock_yaw_tolerance` of the bearing to the building centre
//!   for several samples while that bearing sweeps because the player walks past or around the
//!   building. Standing still and staring at a building sweeps nothing, so engineers working on
//!   their buildings are not flagged however long they look at them.
//! - **silent**: a single eye angle sample lands on the building centre (yaw and pitch) while the
//!   samples before and after point elsewhere and agree with each other (silent aim out-and-back).

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::{MessageType, ParserState};

use crate::algorithms::fidoo::ml_aimbot::is_hitscan;
use crate::base::cheat_analyser_base::{Building, CheatAnalyserState, Player, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::FxHashMap;
use crate::util::nocrex::jankguard::JankGuard;

const ALGORITHM_NAME: &str = "fidoo/building_aimbot";
const FL_DUCKING: u32 = 1 << 1;
/// Largest tick gap between two locked samples of one run
const LOCK_GAP: u32 = 4;
/// Largest tick gap between the three samples of an out-and-back
const SILENT_GAP: u32 = 3;
/// Buildings closer than this horizontally (standing on / in them) have no meaningful bearing
const MIN_HORIZONTAL_DIST: f32 = 32.0;
const MAX_DIST: f32 = 2500.0;
/// Steeper than this the yaw to a building is ill defined
const MAX_TARGET_PITCH: f32 = 60.0;
/// A bearing step larger than this between two samples is a teleport / respawn, not movement
const MAX_BEARING_STEP: f32 = 25.0;
const MAX_EYE_STEP: f32 = 120.0;
/// Silent events closer together than this are one flick
const SILENT_DEDUPE: u32 = 16;
/// Beyond this a silent event needs a hitscan weapon
const MELEE_SILENT_DIST: f32 = 200.0;

fn is_sapper(weapon: &str) -> bool {
    let w = weapon.to_lowercase();
    ["sapper", "red-tape", "ap-sap", "snack attack"].iter().any(|s| w.contains(s))
}

#[derive(Clone, Copy)]
struct Sample {
    tick: u32,
    eye: [f32; 3],
    yaw: f32,
    pitch: f32,
}

/// An open lock on one building
struct Run {
    start: u32,
    last: u32,
    samples: u32,
    sweep: f32,
    bearing: f32,
    eye: [f32; 3],
    max_yaw_err: f32,
    pitch_err_sum: f32,
    pitch_abs_sum: f32,
    min_dist: f32,
    max_dist: f32,
    weapon: String,
    building: &'static str,
    own: bool,
    enemy: bool,
}

#[derive(Clone)]
struct Event {
    tick: u32,
    kind: &'static str,
    /// Bearing sweep for locks, flick size for silent events
    size: f32,
    data: serde_json::Value,
}

#[derive(Default)]
struct PlayerData {
    last_sim: Option<i64>,
    prev: [Option<Sample>; 2],
    runs: FxHashMap<u32, Run>,
    events: Vec<Event>,
}

pub struct BuildingAimbot {
    params: Parameters,
    jg: JankGuard,
    players: FxHashMap<u64, PlayerData>,
}

impl Default for BuildingAimbot {
    fn default() -> Self {
        Self::new()
    }
}

fn wrap(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

fn view_height(player: &Player, ducking: bool) -> f32 {
    if ducking {
        return 45.0;
    }
    match player.class_name() {
        "scout" => 65.0,
        "sniper" | "medic" | "heavy" | "spy" => 75.0,
        _ => 68.0,
    }
}

/// Centre of the collision box above the origin, and the box half width / half height
fn building_box(b: &Building) -> (&'static str, f32, f32, f32) {
    match b {
        Building::Sentry(s) if s.is_mini => ("sentry", 24.75, 15.0, 24.75),
        Building::Sentry(_) => ("sentry", 33.0, 20.0, 33.0),
        Building::Dispenser(_) => ("dispenser", 27.5, 20.0, 27.5),
        Building::Teleporter(_) => ("teleporter", 6.0, 24.0, 6.0),
    }
}

fn dir(yaw: f32, pitch: f32) -> [f32; 3] {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    [p.cos() * y.cos(), p.cos() * y.sin(), -p.sin()]
}

fn angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(d).to_degrees()
}

impl BuildingAimbot {
    pub fn new() -> Self {
        Self {
            params: Parameters::from([
                ("enabled".to_string(), Parameter::Bool(false)),
                ("lock_yaw_tolerance".to_string(), Parameter::Float(0.3)),
                ("min_lock_samples".to_string(), Parameter::Int(3)),
                ("min_lock_sweep".to_string(), Parameter::Float(20.0)),
                ("min_yaw_lock_sweep".to_string(), Parameter::Float(30.0)),
                ("lock_pitch_tolerance".to_string(), Parameter::Float(1.0)),
                ("silent_tolerance".to_string(), Parameter::Float(0.35)),
                ("min_silent_flick".to_string(), Parameter::Float(1.5)),
                ("min_silent".to_string(), Parameter::Int(2)),
                ("min_silent_yaw".to_string(), Parameter::Int(2)),
                ("max_detections".to_string(), Parameter::Int(5)),
            ]),
            jg: JankGuard::default(),
            players: FxHashMap::default(),
        }
    }

    fn close_run(run: Run, ent: u32, cfg: (u32, f32, f32, f32), events: &mut Vec<Event>) {
        let (min_samples, min_sweep, min_yaw_sweep, pitch_tol) = cfg;
        if run.samples < min_samples || run.sweep < min_sweep {
            return;
        }
        // Yaw only locks are what circle strafing around a building looks like; only the auto
        // sapper aims the yaw alone, everything else aims at the centre with the pitch too.
        // A legit spy running around a sentry and jumping can line up the yaw by luck (25° seen),
        // so yaw only locks need a larger sweep.
        let yaw_only = run.pitch_abs_sum / run.samples as f32 > pitch_tol;
        if yaw_only && (!is_sapper(&run.weapon) || run.sweep < min_yaw_sweep) {
            return;
        }
        events.push(Event {
            tick: run.start,
            kind: "lock",
            size: run.sweep,
            data: json!({
                "kind": "lock",
                "building": run.building,
                "entity": ent,
                "own_building": run.own,
                "enemy_building": run.enemy,
                "weapon": run.weapon,
                "sweep_deg": (run.sweep * 10.0).round() / 10.0,
                "samples": run.samples,
                "ticks": run.last - run.start,
                "max_yaw_error_deg": (run.max_yaw_err * 100.0).round() / 100.0,
                "mean_pitch_error_deg": (run.pitch_err_sum / run.samples as f32 * 100.0).round() / 100.0,
                "distance": [run.min_dist.round(), run.max_dist.round()],
            }),
        });
    }
}

impl<'a> CheatAlgorithm<'a> for BuildingAimbot {
    fn algorithm_name(&self) -> &str {
        ALGORITHM_NAME
    }

    fn tooltip(&self) -> &'static str {
        "Aimbot on sentries, dispensers and teleporters (auto sapper, building aimbot, auto repair): the view yaw stays locked on the building centre while the player moves past it, or single eye angle samples snap onto the centre and back (silent aim)."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    // Never hidden by the global sliding window: detections are chosen per player over the whole demo
    fn excluded_from_sliding_window(&self) -> bool {
        true
    }

    // Silent events are counted per player over the whole demo before they are reported
    fn needs_whole_demo(&self) -> bool {
        true
    }

    fn handled_messages(&self) -> Result<Vec<MessageType>, bool> {
        self.jg.handled_messages()
    }

    fn init(&mut self) -> Result<(), Error> {
        let params = std::mem::take(&mut self.params);
        *self = Self::new();
        self.params = params;
        Ok(())
    }

    fn on_message(&mut self, message: &Message, state: &CheatAnalyserState, parser_state: &ParserState, tick: DemoTick) -> Result<Vec<Detection>, Error> {
        self.jg.on_message(message, state, parser_state, tick);
        Ok(vec![])
    }

    fn on_tick(&mut self, state: &CheatAnalyserState, _parser_state: &ParserState) -> Result<Vec<Detection>, Error> {
        self.jg.on_tick(state);
        if !get_parameter_value::<bool>(&self.params, "enabled") {
            return Ok(vec![]);
        }
        let yaw_tol = get_parameter_value::<f32>(&self.params, "lock_yaw_tolerance");
        let min_samples = get_parameter_value::<i32>(&self.params, "min_lock_samples").max(1) as u32;
        let min_sweep = get_parameter_value::<f32>(&self.params, "min_lock_sweep");
        let silent_tol = get_parameter_value::<f32>(&self.params, "silent_tolerance");
        let min_flick = get_parameter_value::<f32>(&self.params, "min_silent_flick");
        let lock_cfg = (min_samples, min_sweep, get_parameter_value::<f32>(&self.params, "min_yaw_lock_sweep"), get_parameter_value::<f32>(&self.params, "lock_pitch_tolerance"));
        let tick = u32::from(state.tick);
        if state.buildings.is_empty() {
            return Ok(vec![]);
        }

        for p in &state.players {
            if !p.in_pvs || p.state != PlayerState::Alive {
                continue;
            }
            let Some(info) = p.info.as_ref() else { continue };
            if info.steam_id == "BOT" {
                continue;
            }
            let Ok(sid) = SteamID::from_steam3(&info.steam_id).map(u64::from) else { continue };
            let pd = self.players.entry(sid).or_default();
            let sim = state.sim_tick(p);
            if pd.last_sim.replace(sim) == Some(sim) {
                continue;
            }
            let ducking = p.flags & FL_DUCKING != 0;
            let eye = [p.position.x, p.position.y, p.position.z + view_height(p, ducking)];
            let cur = Sample { tick, eye, yaw: p.view_angle, pitch: p.pitch_angle };
            let cur_dir = dir(cur.yaw, cur.pitch);
            let mut weapon: Option<String> = None;

            // silent: the previous sample on a building centre, its neighbours elsewhere and together
            let silent_ctx = match pd.prev {
                [Some(b), Some(a)] if cur.tick - b.tick <= SILENT_GAP && b.tick - a.tick <= SILENT_GAP => {
                    let (da, db) = (dir(a.yaw, a.pitch), dir(b.yaw, b.pitch));
                    let (ab, bc, ac) = (angle(da, db), angle(db, cur_dir), angle(da, cur_dir));
                    (ab >= min_flick && bc >= min_flick && ac < 0.5 * ab.min(bc)).then_some((b, db, ab.min(bc), ac))
                }
                _ => None,
            };

            for b in state.buildings.values() {
                let ent = u32::from(b.entity_id());
                // blueprints and carried buildings move with their engineer's view
                if b.placing_or_carried() || (b.max_health() > 0 && b.health() == 0) {
                    pd.runs.remove(&ent);
                    continue;
                }
                let own = b.builder_entity_index() == u32::from(p.entity);
                let (ty, cz, _half_w, _half_h) = building_box(b);
                let bp = b.position();
                let c = [bp.x, bp.y, bp.z + cz];
                let (dx, dy, dz) = (c[0] - eye[0], c[1] - eye[1], c[2] - eye[2]);
                let hd = dx.hypot(dy);
                if hd < MIN_HORIZONTAL_DIST || hd > MAX_DIST {
                    pd.runs.remove(&ent);
                    continue;
                }
                let target_pitch = -dz.atan2(hd).to_degrees();
                if target_pitch.abs() > MAX_TARGET_PITCH {
                    continue;
                }
                let bearing = dy.atan2(dx).to_degrees();
                let yaw_err = wrap(cur.yaw - bearing);

                if let Some((bs, db, flick, ret)) = silent_ctx {
                    let to_c = [c[0] - bs.eye[0], c[1] - bs.eye[1], c[2] - bs.eye[2]];
                    let err = angle(db, to_c);
                    let a = pd.prev[1].unwrap();
                    let a_err = angle(dir(a.yaw, a.pitch), to_c);
                    // yaw only: Amalgam's auto sapper does not aim the pitch
                    let b_yaw_err = wrap(bs.yaw - to_c[1].atan2(to_c[0]).to_degrees()).abs();
                    let a_yaw_err = wrap(a.yaw - to_c[1].atan2(to_c[0]).to_degrees()).abs();
                    let enemy = b.team() != p.team;
                    let dist_b = norm3(to_c);
                    let kind = if err <= silent_tol && a_err > flick * 0.5 && enemy {
                        Some("silent")
                    } else if b_yaw_err <= yaw_tol && a_yaw_err > flick * 0.5 {
                        Some("silent_yaw")
                    } else {
                        None
                    };
                    let w = kind.map(|_| weapon.get_or_insert_with(|| state.get_player_weapon(p)).clone());
                    // silent: guns at any range, anything else (melee) only up close; projectile aimbots
                    // aim at the feet / with gravity, not the centre. silent_yaw: the auto sapper.
                    let kind = match (kind, w.as_deref()) {
                        (Some("silent"), Some(w)) if dist_b <= MELEE_SILENT_DIST || is_hitscan(w) => Some("silent"),
                        (Some("silent_yaw"), Some(w)) if is_sapper(w) => Some("silent_yaw"),
                        _ => None,
                    };
                    // one flick across several samples counts once
                    let repeat = kind.is_some_and(|k| {
                        pd.events.iter().rev().take(8).any(|e| e.kind == k && bs.tick - e.tick <= SILENT_DEDUPE)
                    });
                    if let (Some(kind), Some(w), false) = (kind, w, repeat) {
                        pd.events.push(Event {
                            tick: bs.tick,
                            kind,
                            size: flick,
                            data: json!({
                                "kind": kind,
                                "building": ty,
                                "entity": ent,
                                "own_building": own,
                                "enemy_building": enemy,
                                "weapon": w,
                                "error_deg": (err * 100.0).round() / 100.0,
                                "yaw_error_deg": (b_yaw_err * 100.0).round() / 100.0,
                                "flick_deg": (flick * 100.0).round() / 100.0,
                                "return_deg": (ret * 100.0).round() / 100.0,
                                "distance": dist_b.round(),
                            }),
                        });
                    }
                }

                if yaw_err.abs() >= yaw_tol {
                    continue;
                }
                let pitch_err = cur.pitch - target_pitch;
                let dist = (hd * hd + dz * dz).sqrt();
                let extend = pd.runs.get(&ent).is_some_and(|r| {
                    let step = wrap(bearing - r.bearing).abs();
                    let moved = ((eye[0] - r.eye[0]).powi(2) + (eye[1] - r.eye[1]).powi(2) + (eye[2] - r.eye[2]).powi(2)).sqrt();
                    tick - r.last <= LOCK_GAP && step <= MAX_BEARING_STEP && moved <= MAX_EYE_STEP
                });
                if extend {
                    let r = pd.runs.get_mut(&ent).unwrap();
                    r.sweep += wrap(bearing - r.bearing).abs();
                    r.bearing = bearing;
                    r.eye = eye;
                    r.last = tick;
                    r.samples += 1;
                    r.max_yaw_err = r.max_yaw_err.max(yaw_err.abs());
                    r.pitch_err_sum += pitch_err;
                    r.pitch_abs_sum += pitch_err.abs();
                    r.min_dist = r.min_dist.min(dist);
                    r.max_dist = r.max_dist.max(dist);
                    continue;
                }
                let w = weapon.get_or_insert_with(|| state.get_player_weapon(p)).clone();
                let run = Run {
                    start: tick,
                    last: tick,
                    samples: 1,
                    sweep: 0.0,
                    bearing,
                    eye,
                    max_yaw_err: yaw_err.abs(),
                    pitch_err_sum: pitch_err,
                    pitch_abs_sum: pitch_err.abs(),
                    min_dist: dist,
                    max_dist: dist,
                    weapon: w,
                    building: ty,
                    own,
                    enemy: b.team() != p.team,
                };
                if let Some(old) = pd.runs.insert(ent, run) {
                    Self::close_run(old, ent, lock_cfg, &mut pd.events);
                }
            }

            // close runs that can no longer be extended
            let stale: Vec<u32> = pd.runs.iter().filter(|(_, r)| tick - r.last > LOCK_GAP).map(|(&e, _)| e).collect();
            for ent in stale {
                let run = pd.runs.remove(&ent).unwrap();
                Self::close_run(run, ent, lock_cfg, &mut pd.events);
            }
            pd.prev = [Some(cur), pd.prev[0]];
        }
        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let min_samples = get_parameter_value::<i32>(&self.params, "min_lock_samples").max(1) as u32;
        let min_sweep = get_parameter_value::<f32>(&self.params, "min_lock_sweep");
        let lock_cfg = (min_samples, min_sweep, get_parameter_value::<f32>(&self.params, "min_yaw_lock_sweep"), get_parameter_value::<f32>(&self.params, "lock_pitch_tolerance"));
        let min_silent =get_parameter_value::<i32>(&self.params, "min_silent").max(1) as usize;
        let min_silent_yaw = get_parameter_value::<i32>(&self.params, "min_silent_yaw").max(1) as usize;
        let max_detections = get_parameter_value::<i32>(&self.params, "max_detections").max(1) as usize;
        let mut detections = Vec::new();
        let mut sids: Vec<u64> = self.players.keys().copied().collect();
        sids.sort_unstable();
        for sid in sids {
            let pd = self.players.get_mut(&sid).unwrap();
            for (ent, run) in std::mem::take(&mut pd.runs) {
                Self::close_run(run, ent, lock_cfg, &mut pd.events);
            }
            let locks: Vec<&Event> = pd.events.iter().filter(|e| e.kind == "lock").collect();
            let silents: Vec<&Event> = pd.events.iter().filter(|e| e.kind == "silent").collect();
            let silents_yaw: Vec<&Event> = pd.events.iter().filter(|e| e.kind == "silent_yaw").collect();
            let total_sweep: f32 = locks.iter().map(|e| e.size).sum();
            let mut report: Vec<&Event> = Vec::new();
            for (events, min) in [(locks.clone(), 1), (silents.clone(), min_silent), (silents_yaw.clone(), min_silent_yaw)] {
                if events.len() >= min {
                    let mut e = events;
                    e.sort_by(|a, b| b.size.total_cmp(&a.size));
                    report.extend(e.into_iter().take(max_detections));
                }
            }
            for e in report {
                let mut data = e.data.clone();
                data["player_lock_sweep_deg"] = json!((total_sweep * 10.0).round() / 10.0);
                data["player_locks"] = json!(locks.len());
                data["player_silent"] = json!(silents.len());
                data["player_silent_yaw"] = json!(silents_yaw.len());
                detections.push(Detection { tick: e.tick, algorithm: ALGORITHM_NAME.to_string(), player: sid, data });
            }
        }
        detections.sort_by_key(|d| d.tick);
        Ok(detections)
    }
}

fn norm3(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
