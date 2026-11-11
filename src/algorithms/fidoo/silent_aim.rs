use std::collections::{HashMap, VecDeque};

use anyhow::Error;
use serde_json::json;
use steamid_ng::SteamID;
use tf_demo_parser::demo::vector::Vector;
use tf_demo_parser::ParserState;

use crate::base::cheat_analyser_base::{CheatAnalyserState, PlayerState};
use crate::lib::algorithm::{CheatAlgorithm, Detection};
use crate::lib::parameters::{get_parameter_value, Parameter, Parameters};
use crate::util::helpers::angle_delta;
use crate::util::nocrex::jankguard::JankGuard;

const TELEPORT_DIST_SQ: f32 = 256.0 * 256.0;
// Frames deflected by at least this fraction of the peak belong to the excursion. Kept above 0.5 so
// interpolated halfway frames on the way back are not counted as a second excursion angle.
const EXCURSION_PEAK_FRACTION: f32 = 0.6;

#[derive(Clone, Debug)]
struct PlayerSnapshot {
    tick: u32,
    /// Decoded simulation tick
    sim_tick: i64,
    position: Vector,
    view_angle: f32,
    pitch_angle: f32,
}

#[derive(Default)]
struct PlayerHistoryState {
    pvs_ticks: u32,
    history: VecDeque<PlayerSnapshot>,
    last_update_tick: u32,
    /// Recent attack ticks (JankGuard only keeps the latest one)
    fire_ticks: VecDeque<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeaponCategory {
    Hitscan,
    Melee,
    Sapper,
    Projectile,
    Excluded,
}

#[derive(Default)]
pub struct SilentAim {
    player_states: HashMap<u64, PlayerHistoryState>,
    jg: JankGuard,
    params: Parameters,
    detections: Vec<Detection>,
    // classify_weapon results by weapon name (it is called for every player on every tick)
    weapon_categories: HashMap<String, WeaponCategory>,
}

impl SilentAim {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("tick_window".to_string(), Parameter::Int(4)),
                ("max_delta_first_third".to_string(), Parameter::Float(2.2)),
                ("min_delta_second_third".to_string(), Parameter::Float(8.0)),
                ("min_ratio".to_string(), Parameter::Float(4.8)),
                ("exact_match_epsilon".to_string(), Parameter::Float(0.0001)),
                // f32 acos in angle_delta bottoms out at ~0.0198°, so snapbacks below this are exact returns
                ("near_exact_epsilon".to_string(), Parameter::Float(0.05)),
                ("min_delta_near_exact".to_string(), Parameter::Float(4.0)),
                ("max_excursion_angles".to_string(), Parameter::Int(1)),
                ("max_excursion_hold_ticks".to_string(), Parameter::Int(1)),
                ("min_pvs_ticks".to_string(), Parameter::Int(3)),
                ("require_fire".to_string(), Parameter::Bool(true)),
                // Attack event must land within [excursion - before, excursion + after] ticks
                ("fire_sync_ticks_before".to_string(), Parameter::Int(1)),
                ("fire_sync_ticks_after".to_string(), Parameter::Int(0)),
                ("melee_require_near_exact".to_string(), Parameter::Bool(true)),
            ]),
            ..Default::default()
        }
    }

    /// Classifies weapons into Hitscan, Melee, Sapper, Projectile, or Excluded.
    fn classify_weapon(weapon_name: &str) -> WeaponCategory {
        let lw = weapon_name.to_lowercase();

        // Sapper (Spy sappers, including builder_spy)
        if lw.contains("sapper") || lw.contains("ap-sap") || lw.contains("red-tape") || lw.contains("builder_spy") {
            return WeaponCategory::Sapper;
        }

        // Excluded: Non-combat mobility tools, non-damaging healing tools, engineer tools,
        // and continuous-fire weapons where silent aim is physically impossible / non-existent (Flamethrowers & Miniguns)
        if lw.contains("medigun")
            || lw.contains("medi gun")
            || lw.contains("kritzkrieg")
            || lw.contains("quick-fix")
            || lw.contains("vaccinator")
            || lw.contains("jumper")
            || lw.contains("parachute")
            || lw.contains("builder")
            || lw.contains("toolbox")
            || lw.contains("pda")
            || lw.contains("destruction")
            || lw.contains("destructor")
            // Continuous stream flame weapons (damage over time / flame particles; silent aim does not exist)
            || lw.contains("flamethrower")
            || lw.contains("flame thrower")
            || lw.contains("degreaser")
            || lw.contains("backburner")
            || lw.contains("phlog")
            || lw.contains("rainblower")
            || lw.contains("nostromo")
            || lw.contains("napalm")
            // Rapid-fire continuous spin-up miniguns (40 rounds/sec; silent aim does not exist)
            || lw.contains("minigun")
            || lw.contains("iron curtain")
            || lw.contains("natascha")
            || lw.contains("brass beast")
            || lw.contains("tomislav")
            || lw.contains("huo-long")
            || lw.contains("huo long")
        {
            return WeaponCategory::Excluded;
        }

        // Melee weapons (all classes)
        if lw.contains("bat")
            || lw.contains("knife")
            || lw.contains("wrench")
            || lw.contains("shovel")
            || lw.contains("bottle")
            || lw.contains("fire axe")
            || lw.contains("fireaxe")
            || lw.contains("fists")
            || lw.contains("bonesaw")
            || lw.contains("ubersaw")
            || lw.contains("amputator")
            || lw.contains("solemn vow")
            || lw.contains("vita-saw")
            || lw.contains("shahanshah")
            || lw.contains("bushwacka")
            || lw.contains("kukri")
            || lw.contains("tribalman")
            || lw.contains("shiv")
            || lw.contains("eyelander")
            || lw.contains("scotsman")
            || lw.contains("claidheamh")
            || lw.contains("half-zatoichi")
            || lw.contains("persian persuader")
            || lw.contains("nessie")
            || lw.contains("paintrain")
            || lw.contains("pain train")
            || lw.contains("market gardener")
            || lw.contains("disciplinary")
            || lw.contains("escape plan")
            || lw.contains("equalizer")
            || lw.contains("frying pan")
            || lw.contains("golden frying pan")
            || lw.contains("saxxy")
            || lw.contains("conscientious objector")
            || lw.contains("freedom staff")
            || lw.contains("ham shank")
            || lw.contains("necro smasher")
            || lw.contains("crossing guard")
            || lw.contains("prinny machete")
            || lw.contains("gunslinger")
            || lw.contains("southern hospitality")
            || lw.contains("jag")
            || lw.contains("eureka effect")
            || lw.contains("powerjack")
            || lw.contains("backscratcher")
            || lw.contains("sharpened volcano")
            || lw.contains("third degree")
            || lw.contains("maul")
            || lw.contains("gloves")
            || lw.contains("k.g.b.")
            || lw.contains("warrior's spirit")
            || lw.contains("fist of steel")
            || lw.contains("eviction notice")
            || lw.contains("holiday punch")
            || lw.contains("sandman")
            || lw.contains("candy cane")
            || lw.contains("boston basher")
            || lw.contains("sun-on-a-stick")
            || lw.contains("fan o'war")
            || lw.contains("atomizer")
            || lw.contains("wrap assassin")
            || lw.contains("spy_cicle")
            || lw.contains("big earner")
            || lw.contains("kunai")
            || lw.contains("eternal reward")
            || lw.contains("melee")
        {
            return WeaponCategory::Melee;
        }

        // Projectile weapons (rockets, pipes, stickies, flares, syringes, arrows, energy balls, fireballs)
        if lw.contains("rocket")
            || lw.contains("grenade")
            || lw.contains("pipe")
            || lw.contains("stickybomb")
            || lw.contains("sticky")
            || lw.contains("flare")
            || lw.contains("detonator")
            || lw.contains("scorch")
            || lw.contains("manmelter")
            || lw.contains("syringe")
            || lw.contains("crossbow")
            || lw.contains("huntsman")
            || lw.contains("compound bow")
            || lw.contains("rescue ranger")
            || lw.contains("pomson")
            || lw.contains("bison")
            || lw.contains("cow mangler")
            || lw.contains("loch")
            || lw.contains("loose cannon")
            || lw.contains("iron bomber")
            || lw.contains("scottish resistance")
            || lw.contains("direct hit")
            || lw.contains("black box")
            || lw.contains("air strike")
            || lw.contains("liberty launcher")
            || lw.contains("beggar")
            || lw.contains("dragon's fury")
        {
            return WeaponCategory::Projectile;
        }

        // Default: hitscan bullet weapons (Scattergun, Sniper Rifle, Pistol, SMG, Shotgun, Revolver, etc.)
        WeaponCategory::Hitscan
    }

    /// Checks if the pitch matches the Source Engine 1-byte quantized baseline glitch (~ +/-0.352943°)
    #[inline]
    fn is_quantized_pitch_glitch(pitch: f32) -> bool {
        (pitch.abs() - 0.35294342).abs() < 0.01
    }

    /// Checks if a pitch angle is at or beyond the extreme engine clamp boundary (~ +/-88.0° to 89.3°).
    /// Target coordinates for aimbot never lie at the extreme vertical sky/floor boundaries.
    #[inline]
    fn is_extreme_pitch_clamp(pitch: f32) -> bool {
        pitch.abs() >= 88.0
    }

    /// Detects console command pitch flipping (+lookup; +lookdown / pitch bounds toggling).
    /// In Source engine, +lookup drives pitch to -89.29° and +lookdown drives pitch to +89.29°.
    /// Rapid toggling between opposite vertical extremes is impossible for legitimate aimbot targeting.
    #[inline]
    fn is_pitch_command_flip(pitch_a: f32, pitch_b: f32) -> bool {
        let pitch_diff = (pitch_a - pitch_b).abs();
        pitch_diff > 120.0
            && pitch_a.abs() > 75.0
            && pitch_b.abs() > 75.0
            && (pitch_a * pitch_b < 0.0)
    }

    /// Verifies if fire checking can be bypassed for non-bullet weapons with strong geometric evidence.
    /// In TF2, Hitscan (CTEFireBullets / CTEPlayerAnimEvent), Melee (CTEPlayerAnimEvent / PlayerHurt),
    /// and Projectile weapons (CTEPlayerAnimEvent on launch) emit firing messages.
    /// Only Sapper placement lacks attack animation and bullet events.
    fn should_skip_fire_check(
        weapon_category: WeaponCategory,
        is_exact: bool,
        ratio: f32,
        _peak_delta: f32,
        _delta_0_i: f32,
    ) -> bool {
        // Only Sapper placement does not emit attack animation or bullet events
        if weapon_category != WeaponCategory::Sapper {
            return false;
        }

        // Sapper: require exact-angle snapback (is_exact) and extreme ratio (>= 25.0)
        is_exact && ratio >= 25.0
    }

    /// Silent aim overrides the view angles of a single usercmd, so the off-target frames between
    /// the anchor (history[i]) and the snapback (history[0]) show one distinct angle held for at
    /// most a tick. Human flicks sweep through several angles or hold the deflection longer.
    /// Returns (distinct excursion angles, tick span of the excursion frames).
    fn excursion_shape(
        history: &VecDeque<PlayerSnapshot>,
        i: usize,
        current_angle: (f32, f32),
        anchor_angle: (f32, f32),
    ) -> (usize, u32) {
        let peak = (1..i)
            .map(|m| {
                let a = (history[m].view_angle, history[m].pitch_angle);
                angle_delta(current_angle, a).max(angle_delta(anchor_angle, a))
            })
            .fold(0.0f32, f32::max);

        let mut angles: Vec<(f32, f32)> = Vec::new();
        let (mut first_tick, mut last_tick) = (u32::MAX, 0u32);
        for m in 1..i {
            let snap = &history[m];
            let a = (snap.view_angle, snap.pitch_angle);
            // Frames still near either endpoint are not part of the excursion
            if angle_delta(current_angle, a).min(angle_delta(anchor_angle, a)) < EXCURSION_PEAK_FRACTION * peak {
                continue;
            }
            if !angles.contains(&a) {
                angles.push(a);
            }
            first_tick = first_tick.min(snap.tick);
            last_tick = last_tick.max(snap.tick);
        }
        (angles.len(), last_tick.saturating_sub(first_tick))
    }

    /// Specifically identifies downward rocket jumps (Soldier + Rocket Launcher + downward pitch floor > 20°).
    /// Prevents false exclusions for Scout/Sniper/Spy high-ground combat.
    fn is_rocket_jump_context(
        class_name: &str,
        weapon_name: &str,
        curr_pitch: f32,
        anchor_pitch: f32,
    ) -> bool {
        if class_name == "soldier" {
            let lw = weapon_name.to_lowercase();
            if lw.contains("rocket")
                || lw.contains("direct hit")
                || lw.contains("black box")
                || lw.contains("air strike")
                || lw.contains("liberty launcher")
                || lw.contains("beggar")
            {
                // Rocket jumping flick specifically aims at the feet/floor/ramp/wall (> 20.0°)
                if curr_pitch > 20.0 && anchor_pitch > 20.0 {
                    return true;
                }
            }
        }
        false
    }
}

impl<'a> CheatAlgorithm<'a> for SilentAim {
    fn algorithm_name(&self) -> &str {
        "fidoo/silent_aim"
    }

    fn tooltip(&self) -> &'static str {
        "Detects silent aim snapbacks where view angles snap to a target on the attack tick and instantly restore."
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

        // Record PlayerHurt damage events as confirmed attack hits (for melee and bullets)
        if let tf_demo_parser::demo::message::Message::GameEvent(event_msg) = message {
            if let tf_demo_parser::demo::gamevent::GameEvent::PlayerHurt(hurt) = &event_msg.event {
                if let Some(attacker_id64) = state.get_id64_from_userid(hurt.attacker.into()) {
                    self.jg.set_last_fire(attacker_id64, u32::from(tick));
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
        let algo_name = self.algorithm_name().to_string();

        let enabled = get_parameter_value::<bool>(&self.params, "enabled");
        if !enabled {
            return Ok(vec![]);
        }

        let tick_window = get_parameter_value::<i32>(&self.params, "tick_window").max(2) as usize;
        let max_delta_first_third =
            get_parameter_value::<f32>(&self.params, "max_delta_first_third");
        let min_delta_second_third =
            get_parameter_value::<f32>(&self.params, "min_delta_second_third");
        let min_ratio = get_parameter_value::<f32>(&self.params, "min_ratio");
        let exact_match_epsilon =
            get_parameter_value::<f32>(&self.params, "exact_match_epsilon");
        let near_exact_epsilon = get_parameter_value::<f32>(&self.params, "near_exact_epsilon");
        let min_delta_near_exact =
            get_parameter_value::<f32>(&self.params, "min_delta_near_exact");
        let max_excursion_angles =
            get_parameter_value::<i32>(&self.params, "max_excursion_angles").max(1) as usize;
        let max_excursion_hold_ticks =
            get_parameter_value::<i32>(&self.params, "max_excursion_hold_ticks").max(0) as u32;
        let min_pvs_ticks =
            get_parameter_value::<i32>(&self.params, "min_pvs_ticks").max(1) as u32;
        let require_fire = get_parameter_value::<bool>(&self.params, "require_fire");
        let fire_sync_ticks_before =
            get_parameter_value::<i32>(&self.params, "fire_sync_ticks_before").max(0) as u32;
        let fire_sync_ticks_after =
            get_parameter_value::<i32>(&self.params, "fire_sync_ticks_after").max(0) as u32;
        let melee_require_near_exact =
            get_parameter_value::<bool>(&self.params, "melee_require_near_exact");

        let mut current_active_sids = Vec::new();
        let mut tick_detections = Vec::new();

        for player in &state.players {
            let Some(info) = &player.info else {
                continue;
            };
            if info.steam_id == "BOT" {
                continue;
            }
            let Ok(steam_id) = SteamID::from_steam3(&info.steam_id).map(u64::from) else {
                continue;
            };

            current_active_sids.push(steam_id);
            let pstate = self.player_states.entry(steam_id).or_default();

            let last_fire = ticknum.saturating_sub(self.jg.fired(&steam_id, ticknum));
            if last_fire > 0 && pstate.fire_ticks.back() != Some(&last_fire) {
                pstate.fire_ticks.push_back(last_fire);
            }
            while pstate.fire_ticks.front().is_some_and(|&t| t + 64 < ticknum) {
                pstate.fire_ticks.pop_front();
            }

            // PVS and Life state validation: player must be alive and in PVS
            if !player.in_pvs || player.state != PlayerState::Alive {
                pstate.pvs_ticks = 0;
                pstate.history.clear();
                continue;
            }

            // Check network continuity: reset if a large packet drop occurs (> 16 ticks / 250ms)
            if pstate.last_update_tick > 0 && ticknum.saturating_sub(pstate.last_update_tick) > 16 {
                pstate.pvs_ticks = 0;
                pstate.history.clear();
            }

            pstate.pvs_ticks += 1;
            pstate.last_update_tick = ticknum;

            // JankGuard spawn/teleport immunity check (60 ticks)
            let ticks_since_event = self
                .jg
                .teleported(&steam_id, ticknum)
                .min(self.jg.spawned(&steam_id, ticknum));

            if ticks_since_event < 60 {
                if ticks_since_event == 0 {
                    self.detections
                        .retain(|det| det.player != steam_id || (ticknum - det.tick) > 60);
                }
                pstate.history.clear();
                continue;
            }

            // Must satisfy PVS grace period before recording snapshots
            if pstate.pvs_ticks < min_pvs_ticks {
                pstate.history.clear();
                continue;
            }

            let snapshot = PlayerSnapshot {
                tick: ticknum,
                sim_tick: state.sim_tick(player),
                position: player.position,
                view_angle: player.view_angle,
                pitch_angle: player.pitch_angle,
            };

            pstate.history.push_front(snapshot);
            while pstate.history.len() > tick_window + 2 {
                pstate.history.pop_back();
            }

            let history_len = pstate.history.len();
            if history_len < 3 {
                continue;
            }

            let curr_snap = &pstate.history[0];
            let current_angle = (curr_snap.view_angle, curr_snap.pitch_angle);

            // Classify weapon
            let weapon_name = state.get_player_weapon(player);
            let weapon_category = match self.weapon_categories.get(weapon_name.as_str()) {
                Some(category) => *category,
                None => {
                    let category = Self::classify_weapon(&weapon_name);
                    self.weapon_categories.insert(weapon_name.clone(), category);
                    category
                }
            };

            // Skip excluded non-damaging utility/mobility tools (healing beams, jumper training tools, parachutes)
            if weapon_category == WeaponCategory::Excluded {
                continue;
            }

            let class_name = player.class_name();

            // Search for an anchor snapshot in history [2..=max_lookback]
            let max_lookback = tick_window.min(history_len - 1);
            let mut detected = false;
            let mut flagged = false;

            for i in 2..=max_lookback {
                let anchor_snap = &pstate.history[i];
                let anchor_angle = (anchor_snap.view_angle, anchor_snap.pitch_angle);

                // Verify the total window duration is within silent aimbot flick time (<= 8 ticks)
                if curr_snap.tick.saturating_sub(anchor_snap.tick) > 8 {
                    continue;
                }

                // Verify packet and spatial continuity across each adjacent frame in window
                let mut contiguous = true;
                for step in 0..i {
                    let s_newer = &pstate.history[step];
                    let s_older = &pstate.history[step + 1];

                    // Packet continuity: max 2 ticks gap between consecutive snapshots
                    if s_newer.tick.saturating_sub(s_older.tick) > 2 {
                        contiguous = false;
                        break;
                    }

                    // Spatial continuity: no teleportation between frames
                    let dx = s_newer.position.x - s_older.position.x;
                    let dy = s_newer.position.y - s_older.position.y;
                    let dz = s_newer.position.z - s_older.position.z;
                    if (dx * dx + dy * dy + dz * dz) > TELEPORT_DIST_SQ {
                        contiguous = false;
                        break;
                    }
                }

                if !contiguous {
                    continue;
                }

                // Exclude quantized pitch baseline glitches
                if Self::is_quantized_pitch_glitch(curr_snap.pitch_angle)
                    || Self::is_quantized_pitch_glitch(anchor_snap.pitch_angle)
                {
                    continue;
                }

                // Exclude console command pitch flipping (+lookup; +lookdown) on baseline
                if Self::is_pitch_command_flip(curr_snap.pitch_angle, anchor_snap.pitch_angle) {
                    continue;
                }

                // Specifically filter downward rocket jumping (Soldier + Rocket Launcher + pitch > 65°)
                if Self::is_rocket_jump_context(class_name, &weapon_name, curr_snap.pitch_angle, anchor_snap.pitch_angle) {
                    continue;
                }

                // Return delta between current angle (t) and anchor angle (t - i)
                let delta_0_i = angle_delta(current_angle, anchor_angle);

                // Check exact angle match (zero drift or within float quantization epsilon)
                let is_exact = delta_0_i <= exact_match_epsilon
                    || (curr_snap.view_angle == anchor_snap.view_angle && curr_snap.pitch_angle == anchor_snap.pitch_angle);

                if !is_exact && delta_0_i > max_delta_first_third {
                    continue;
                }

                // Snapbacks within f32 precision of the anchor are allowed a smaller excursion
                let is_near_exact = is_exact || delta_0_i <= near_exact_epsilon;

                // Legit melee swings often flick through the target and back with some drift
                if melee_require_near_exact && weapon_category == WeaponCategory::Melee && !is_near_exact {
                    continue;
                }
                let min_peak_delta = if is_near_exact {
                    min_delta_near_exact.min(min_delta_second_third)
                } else {
                    min_delta_second_third
                };

                let (excursion_angles, excursion_hold) =
                    Self::excursion_shape(&pstate.history, i, current_angle, anchor_angle);
                if excursion_angles > max_excursion_angles || excursion_hold > max_excursion_hold_ticks {
                    continue;
                }

                // Intermediate spike check: at least one tick strictly between 0 and i must have spiked
                for m in 1..i {
                    let mid_snap = &pstate.history[m];
                    let mid_angle = (mid_snap.view_angle, mid_snap.pitch_angle);

                    // Skip quantized pitch glitch on intermediate snapshot
                    if Self::is_quantized_pitch_glitch(mid_snap.pitch_angle) {
                        continue;
                    }

                    // Skip extreme engine pitch clamp or console command pitch flips (+lookup; +lookdown)
                    if Self::is_extreme_pitch_clamp(mid_snap.pitch_angle)
                        || Self::is_pitch_command_flip(curr_snap.pitch_angle, mid_snap.pitch_angle)
                        || Self::is_pitch_command_flip(anchor_snap.pitch_angle, mid_snap.pitch_angle)
                    {
                        continue;
                    }

                    let mid_delta_curr = angle_delta(current_angle, mid_angle);
                    let mid_delta_anchor = angle_delta(anchor_angle, mid_angle);
                    let peak_delta = mid_delta_curr.max(mid_delta_anchor);

                    if peak_delta < min_peak_delta {
                        continue;
                    }

                    // Ratio check: excursion must be significantly larger than baseline drift
                    let ratio = if delta_0_i > 0.0001 {
                        peak_delta / delta_0_i
                    } else {
                        peak_delta / 0.0001
                    };

                    let effective_min_ratio = match weapon_category {
                        WeaponCategory::Projectile => min_ratio.max(12.0),
                        _ => min_ratio,
                    };

                    if !is_exact && ratio < effective_min_ratio {
                        continue;
                    }

                    // Weapon fire check
                    if require_fire {
                        // Small near-exact excursions always need a matching attack event
                        let skip_fire = peak_delta >= min_delta_second_third
                            && Self::should_skip_fire_check(
                                weapon_category,
                                is_exact,
                                ratio,
                                peak_delta,
                                delta_0_i,
                            );

                        if !skip_fire {
                            // The override and its attack share one usercmd, so the attack event
                            // lands on the excursion tick (or one tick earlier for anim events)
                            let fired_in_sync = pstate.fire_ticks.iter().any(|&t| {
                                t + fire_sync_ticks_before >= mid_snap.tick
                                    && t <= mid_snap.tick + fire_sync_ticks_after
                            });
                            if !fired_in_sync {
                                continue;
                            }
                        }
                    }

                    let detection = Detection {
                        tick: ticknum,
                        algorithm: algo_name.clone(),
                        player: steam_id,
                        data: json!({
                            "class": class_name,
                            "weapon": weapon_name,
                            "weapon_category": format!("{:?}", weapon_category),
                            "simtime": curr_snap.sim_tick,
                            "angle_current": current_angle,
                            "angle_middle": mid_angle,
                            "angle_trigger": anchor_angle,
                            "delta_1_3": delta_0_i,
                            "delta_2_3": peak_delta,
                            "ratio": ratio,
                            "match_index": i,
                            "middle_index": m,
                            "pvs_ticks": pstate.pvs_ticks,
                            "is_exact_match": is_exact,
                            "is_near_exact_match": is_near_exact,
                            "excursion_angles": excursion_angles,
                            "excursion_hold_ticks": excursion_hold,
                        }),
                    };

                    tick_detections.push(detection.clone());
                    self.detections.push(detection);

                    flagged = true;
                    detected = true;
                    break;
                }

                if detected {
                    break;
                }
            }

            if flagged {
                // Clear history buffer to debounce cleanly
                pstate.history.clear();
            }
        }

        // Clean up disconnected players
        self.player_states
            .retain(|sid, _| current_active_sids.contains(sid));

        Ok(tick_detections)
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::cheat_analyser_base::Player;
    use tf_demo_parser::demo::message::packetentities::EntityId;
    use tf_demo_parser::demo::parser::analyser::{Class, Team, UserId, UserInfo};

    fn test_parser_state() -> ParserState {
        ParserState::new(24, |_| false, false)
    }

    fn create_test_player(
        entity_id: u32,
        user_id: u32,
        steam_id: &str,
        pos: Vector,
        yaw: f32,
        pitch: f32,
        in_pvs: bool,
    ) -> Player {
        Player {
            entity: EntityId::from(entity_id),
            position: pos,
            health: 125,
            max_health: 125,
            class: Class::Scout,
            team: Team::Red,
            view_angle: yaw,
            pitch_angle: pitch,
            state: PlayerState::Alive,
            info: Some(UserInfo {
                classes: Default::default(),
                name: "TestSuspect".to_string(),
                user_id: UserId::from(user_id as u16),
                steam_id: steam_id.to_string(),
                entity_id: EntityId::from(entity_id),
                team: Team::Red,
            }),
            charge: 0,
            simtime: 100,
            ping: 25,
            in_pvs,
            active_weapon: None,
            cond: 0,
            cond_ex: 0,
            cond_ex2: 0,
            cond_ex3: 0,
            invis_change_complete_time: 0.0,
            flags: 1,
        }
    }

    #[test]
    fn test_exact_match_instant_flag() {
        let mut algo = SilentAim::new();
        algo.params
            .insert("require_fire".to_string(), Parameter::Bool(false));
        let mut state = CheatAnalyserState::default();
        let parser_state = test_parser_state();

        let steam_id_str = "[U:1:12345678]";
        let sid = u64::from(SteamID::from_steam3(steam_id_str).unwrap());
        state.set_entid_to_userid(EntityId::from(1u32), UserId::from(1u16));
        state.set_userid_to_id64(UserId::from(1u16), sid);

        // Pre-fill PVS grace ticks at base angle (45.0, 0.0)
        for t in 100..105 {
            state.tick = t.into();
            state.players = vec![create_test_player(
                1,
                1,
                steam_id_str,
                Vector { x: 0.0, y: 0.0, z: 0.0 },
                45.0,
                0.0,
                true,
            )];
            let _ = algo.on_tick(&state, &parser_state);
        }

        // Tick 105: Flick to (75.0, 15.0) -> peak delta > 30°
        state.tick = 105.into();
        state.players = vec![create_test_player(
            1,
            1,
            steam_id_str,
            Vector { x: 0.0, y: 0.0, z: 0.0 },
            75.0,
            15.0,
            true,
        )];
        let _ = algo.on_tick(&state, &parser_state);

        // Tick 106: Snap back to EXACT SAME (45.0, 0.0)
        state.tick = 106.into();
        state.players = vec![create_test_player(
            1,
            1,
            steam_id_str,
            Vector { x: 0.0, y: 0.0, z: 0.0 },
            45.0,
            0.0,
            true,
        )];
        let detections = algo.on_tick(&state, &parser_state).unwrap();

        // Exact match should flag immediately on first flick!
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].algorithm, "fidoo/silent_aim");
        assert_eq!(detections[0].player, sid);
        assert_eq!(detections[0].data["is_exact_match"], true);
    }

    #[test]
    fn test_drift_flick_detection() {
        let mut algo = SilentAim::new();
        algo.params
            .insert("require_fire".to_string(), Parameter::Bool(false));
        let mut state = CheatAnalyserState::default();
        let parser_state = test_parser_state();

        let steam_id_str = "[U:1:12345678]";
        let sid = u64::from(SteamID::from_steam3(steam_id_str).unwrap());
        state.set_entid_to_userid(EntityId::from(1u32), UserId::from(1u16));
        state.set_userid_to_id64(UserId::from(1u16), sid);

        // Baseline
        for t in 100..105 {
            state.tick = t.into();
            state.players = vec![create_test_player(
                1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 0.0 }, 45.0, 0.0, true,
            )];
            let _ = algo.on_tick(&state, &parser_state);
        }

        // Flick 1: Peak to 75.0, then snapback to 45.5 (drift = 0.5°)
        state.tick = 105.into();
        state.players = vec![create_test_player(
            1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 0.0 }, 75.0, 15.0, true,
        )];
        let _ = algo.on_tick(&state, &parser_state);

        state.tick = 106.into();
        state.players = vec![create_test_player(
            1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 0.0 }, 45.5, 0.0, true,
        )];
        let det1 = algo.on_tick(&state, &parser_state).unwrap();
        // Drift flick should flag directly without internal sliding window threshold
        assert_eq!(det1.len(), 1);
        assert_eq!(det1[0].data["is_exact_match"], false);
    }

    /// Feeds (tick, yaw, pitch) frames after a stable baseline and returns detections from the last frame.
    fn run_sequence(frames: &[(u32, f32, f32)]) -> Vec<Detection> {
        let mut algo = SilentAim::new();
        algo.params
            .insert("require_fire".to_string(), Parameter::Bool(false));
        let mut state = CheatAnalyserState::default();
        let parser_state = test_parser_state();

        let steam_id_str = "[U:1:12345678]";
        let sid = u64::from(SteamID::from_steam3(steam_id_str).unwrap());
        state.set_entid_to_userid(EntityId::from(1u32), UserId::from(1u16));
        state.set_userid_to_id64(UserId::from(1u16), sid);

        let (base_yaw, base_pitch) = (frames[0].1, frames[0].2);
        let baseline = (95..frames[0].0).map(|t| (t, base_yaw, base_pitch));
        let mut detections = vec![];
        for (t, yaw, pitch) in baseline.chain(frames.iter().copied()) {
            state.tick = t.into();
            state.players = vec![create_test_player(
                1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 0.0 }, yaw, pitch, true,
            )];
            detections = algo.on_tick(&state, &parser_state).unwrap();
        }
        detections
    }

    #[test]
    fn test_small_near_exact_snapback_flagged() {
        // 4.3° single-frame excursion returning within f32 acos precision of the anchor
        let dets = run_sequence(&[(104, 177.36, 2.47), (105, 177.71, -1.76), (106, 177.36, 2.4701)]);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].data["is_near_exact_match"], true);
    }

    #[test]
    fn test_small_drifting_snapback_not_flagged() {
        // Same 4.3° excursion with 0.7° drift is below the 7° non-exact threshold
        let dets = run_sequence(&[(104, 177.36, 2.47), (105, 177.71, -1.76), (106, 178.06, 2.47)]);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_human_sweep_excursion_not_flagged() {
        // Flick out and back through several distinct off-target angles (human sweep)
        let dets = run_sequence(&[
            (104, 270.6, 7.4),
            (105, 279.4, 8.1),
            (106, 281.9, 8.8),
            (107, 271.0, 7.4),
        ]);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_held_excursion_not_flagged() {
        // Single off-target angle held for 3 ticks is not a one-usercmd override
        let dets = run_sequence(&[
            (104, 261.1, 3.2),
            (105, 250.2, 3.2),
            (106, 250.2, 3.2),
            (107, 250.2, 3.2),
            (108, 261.1, 3.2),
        ]);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_high_pitch_scout_aim_not_filtered() {
        let mut algo = SilentAim::new();
        algo.params
            .insert("require_fire".to_string(), Parameter::Bool(false));
        let mut state = CheatAnalyserState::default();
        let parser_state = test_parser_state();

        let steam_id_str = "[U:1:12345678]";
        let sid = u64::from(SteamID::from_steam3(steam_id_str).unwrap());
        state.set_entid_to_userid(EntityId::from(1u32), UserId::from(1u16));
        state.set_userid_to_id64(UserId::from(1u16), sid);

        // Pre-fill at steep downward pitch (58.0°) - Scout shooting downward from high ground
        for t in 100..105 {
            state.tick = t.into();
            state.players = vec![create_test_player(
                1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 500.0 }, 45.0, 58.0, true,
            )];
            let _ = algo.on_tick(&state, &parser_state);
        }

        // Flick to (65.0, 60.0) -> peak delta ~ 20°
        state.tick = 105.into();
        state.players = vec![create_test_player(
            1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 500.0 }, 65.0, 60.0, true,
        )];
        let _ = algo.on_tick(&state, &parser_state);

        // Snapback to exact (45.0, 58.0)
        state.tick = 106.into();
        state.players = vec![create_test_player(
            1, 1, steam_id_str, Vector { x: 0.0, y: 0.0, z: 500.0 }, 45.0, 58.0, true,
        )];
        let detections = algo.on_tick(&state, &parser_state).unwrap();

        // Scout aiming downward from high ground MUST NOT be filtered!
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].player, sid);
    }

    #[test]
    fn test_damaging_weapons_classification() {
        // Continuous flame stream weapons and miniguns must be Excluded (silent aim impossible / non-existent)
        assert_eq!(SilentAim::classify_weapon("Flame Thrower"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Degreaser"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Backburner"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The Phlogistinator"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Rainblower"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Nostromo Napalmer"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Minigun"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Iron Curtain"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Tomislav"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Natascha"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The Brass Beast"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Huo-Long Heater"), WeaponCategory::Excluded);

        // Projectile weapons that fire discrete projectiles
        assert_eq!(SilentAim::classify_weapon("Dragon's Fury"), WeaponCategory::Projectile);
        assert_eq!(SilentAim::classify_weapon("Rocket Launcher"), WeaponCategory::Projectile);
        assert_eq!(SilentAim::classify_weapon("Grenade Launcher"), WeaponCategory::Projectile);
        assert_eq!(SilentAim::classify_weapon("Stickybomb Launcher"), WeaponCategory::Projectile);

        // Non-damaging mobility and healing items must be Excluded
        assert_eq!(SilentAim::classify_weapon("Medi Gun"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The Kritzkrieg"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The Quick-Fix"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The Vaccinator"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Rocket Jumper"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Sticky Jumper"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("The B.A.S.E. Jumper"), WeaponCategory::Excluded);

        // Engineer building tools and PDAs must be Excluded
        assert_eq!(SilentAim::classify_weapon("Builder"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Toolbox"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Construction PDA"), WeaponCategory::Excluded);
        assert_eq!(SilentAim::classify_weapon("Destruction PDA"), WeaponCategory::Excluded);
    }

    #[test]
    fn test_rocket_jump_pitch_floor() {
        // Soldier with Rocket Launcher aiming downward (> 20.0°) is filtered
        assert!(SilentAim::is_rocket_jump_context("soldier", "Rocket Launcher", 25.0, 25.0));
        assert!(SilentAim::is_rocket_jump_context("soldier", "Air Strike", 59.6, 59.6));
        assert!(SilentAim::is_rocket_jump_context("soldier", "Direct Hit", 29.3, 29.3));
        // Soldier aiming horizontal (<= 20.0°) is NOT filtered
        assert!(!SilentAim::is_rocket_jump_context("soldier", "Rocket Launcher", 5.0, 5.0));
        // Non-soldier or non-rocket weapon aiming downward is NOT filtered
        assert!(!SilentAim::is_rocket_jump_context("scout", "Pistol", 58.0, 58.0));
    }

    #[test]
    fn test_pitch_command_flipping_filtered() {
        // +lookdown (+89.29°) and +lookup (-89.29°) toggling must be detected as command flip
        assert!(SilentAim::is_pitch_command_flip(89.29411, -89.29412));
        assert!(SilentAim::is_pitch_command_flip(-89.29412, 89.29411));
        assert!(SilentAim::is_extreme_pitch_clamp(89.29411));
        assert!(SilentAim::is_extreme_pitch_clamp(-89.29412));

        // Normal combat pitch changes must NOT be detected as command flip
        assert!(!SilentAim::is_pitch_command_flip(58.24, 59.65));
        assert!(!SilentAim::is_pitch_command_flip(0.0, 15.0));
        assert!(!SilentAim::is_pitch_command_flip(-20.0, 20.0));
        assert!(!SilentAim::is_extreme_pitch_clamp(58.24));
        assert!(!SilentAim::is_extreme_pitch_clamp(0.0));
    }

    #[test]
    fn test_fire_check_requirements() {
        // Hitscan, Melee, and Projectiles must all verify attack/fire events
        assert_eq!(
            SilentAim::should_skip_fire_check(WeaponCategory::Melee, true, 100.0, 15.0, 0.0),
            false
        );
        assert_eq!(
            SilentAim::should_skip_fire_check(WeaponCategory::Hitscan, true, 100.0, 15.0, 0.0),
            false
        );
        assert_eq!(
            SilentAim::should_skip_fire_check(WeaponCategory::Projectile, true, 100.0, 15.0, 0.0),
            false
        );

        // Only Sapper with exact angle match and high ratio may skip fire check
        assert_eq!(
            SilentAim::should_skip_fire_check(WeaponCategory::Sapper, true, 50.0, 15.0, 0.0),
            true
        );
        assert_eq!(
            SilentAim::should_skip_fire_check(WeaponCategory::Sapper, false, 50.0, 15.0, 0.0),
            false
        );
    }
}
