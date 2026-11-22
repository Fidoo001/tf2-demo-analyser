use std::collections::{HashMap, VecDeque};

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
use crate::util::helpers::handle_to_entid;
use crate::util::nocrex::jankguard::JankGuard;

#[derive(Clone, Copy, Debug, PartialEq)]
struct WeaponSpreadInfo {
    name: &'static str,
    /// Bullets fired per shot (weapon script `BulletsPerShot`, attributes ignored).
    bullets: u32,
    /// Weapon script spread, only used when a bullet carries no `m_flSpread`.
    default_spread: f32,
    /// Rapid-fire weapons (refire well under 0.25 s), whose shot-tick jitter ShotReversal counts.
    /// Slow weapons' out-and-backs are mostly silent-aim flicks, and shotguns and scatterguns
    /// are not compensated at all outside double-tap (their first pellet is perfect).
    rapid: bool,
}

const MINIGUN: WeaponSpreadInfo = WeaponSpreadInfo { name: "Minigun", bullets: 4, default_spread: 0.08, rapid: true };
const SMG: WeaponSpreadInfo = WeaponSpreadInfo { name: "SMG", bullets: 1, default_spread: 0.025, rapid: true };
const PISTOL: WeaponSpreadInfo = WeaponSpreadInfo { name: "Pistol", bullets: 1, default_spread: 0.04, rapid: true };
const REVOLVER: WeaponSpreadInfo = WeaponSpreadInfo { name: "Revolver", bullets: 1, default_spread: 0.025, rapid: false };
const SCATTERGUN: WeaponSpreadInfo =
    WeaponSpreadInfo { name: "Scattergun", bullets: 10, default_spread: 0.0675, rapid: false };
const SHORTSTOP: WeaponSpreadInfo = WeaponSpreadInfo { name: "Shortstop", bullets: 4, default_spread: 0.04, rapid: false };
const SHOTGUN: WeaponSpreadInfo = WeaponSpreadInfo { name: "Shotgun", bullets: 10, default_spread: 0.0675, rapid: false };

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
            // TF_WEAPON_HANDGUN_SCOUT_PRIMARY
            71 => Some(SHORTSTOP),
            // TF_WEAPON_SCATTERGUN, TF_WEAPON_SODA_POPPER, TF_WEAPON_PEP_BRAWLER_BLASTER
            16 | 76 | 85 => Some(SCATTERGUN),
            // TF_WEAPON_SHOTGUN_PRIMARY/SOLDIER/HWG/PYRO, TF_WEAPON_SENTRY_REVENGE
            12..=15 | 69 => Some(SHOTGUN),
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
    if weapon_name.contains("Shortstop") {
        return Some(SHORTSTOP);
    }
    if ["Scattergun", "Force-A-Nature", "Soda Popper", "Baby Face", "Back Scatter"]
        .iter()
        .any(|n| weapon_name.contains(n))
    {
        return Some(SCATTERGUN);
    }
    if weapon_name.contains("Shotgun") || weapon_name.contains("Frontier Justice") {
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

/// The (x, y) spread multipliers FX_FireBullets draws for one bullet from `seed`.
fn bullet_spread(seed: i32) -> (f32, f32) {
    let mut rng = UniformRandomStream::new();
    rng.set_seed(seed & 255);
    let x = rng.random_float(-0.5, 0.5) + rng.random_float(-0.5, 0.5);
    let y = rng.random_float(-0.5, 0.5) + rng.random_float(-0.5, 0.5);
    (x, y)
}

/// The bullet Amalgam's NoSpreadHitscan corrects: of all `bullets` pellets (seeds seed+i),
/// the one closest to their average. Single-bullet weapons give the first bullet.
fn amalgam_corrected_spread(seed: i32, bullets: u32) -> (f32, f32) {
    let pellets: Vec<(f32, f32)> = (0..bullets.max(1) as i32).map(|i| bullet_spread(seed + i)).collect();
    let n = pellets.len() as f32;
    let ax = pellets.iter().map(|p| p.0).sum::<f32>() / n;
    let ay = pellets.iter().map(|p| p.1).sum::<f32>() / n;
    pellets
        .into_iter()
        .min_by(|a, b| {
            let da = (a.0 - ax).powi(2) + (a.1 - ay).powi(2);
            let db = (b.0 - ax).powi(2) + (b.1 - ay).powi(2);
            da.total_cmp(&db)
        })
        .unwrap_or((0.0, 0.0))
}

/// P(X >= k) for X ~ Binomial(n, p).
fn binomial_sf(k: usize, n: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // ln C(n, i) built up incrementally.
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let mut ln_c = 0.0f64;
    let mut sum = 0.0f64;
    for i in 0..=n {
        if i > 0 {
            ln_c += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        if i >= k {
            sum += (ln_c + i as f64 * lp + (n - i) as f64 * lq).exp();
        }
    }
    sum.min(1.0)
}

/// Running sums for a Pearson correlation that samples can be added to and removed from.
#[derive(Default, Clone, Copy, Debug)]
struct CorrSums {
    n: f64,
    a: f64,
    b: f64,
    aa: f64,
    bb: f64,
    ab: f64,
}

impl CorrSums {
    fn add(&mut self, a: f32, b: f32, sign: f64) {
        let (a, b) = (a as f64, b as f64);
        self.n += sign;
        self.a += sign * a;
        self.b += sign * b;
        self.aa += sign * a * a;
        self.bb += sign * b * b;
        self.ab += sign * a * b;
    }

    /// Pearson correlation and its Fisher z (one-sided, positive = nospread direction).
    fn correlation_z(&self) -> (f64, f64) {
        let n = self.n.round();
        if n < 4.0 {
            return (0.0, 0.0);
        }
        let cov = self.ab - self.a * self.b / n;
        let va = self.aa - self.a * self.a / n;
        let vb = self.bb - self.b * self.b / n;
        if va <= 1e-9 || vb <= 1e-9 {
            return (0.0, 0.0);
        }
        let r = (cov / (va * vb).sqrt()).clamp(-0.999, 0.999);
        (r, r.atanh() * (n - 3.0).sqrt())
    }
}

/// The four correlations (yaw/pitch excursion vs both pellet models) of the seed samples.
#[derive(Default, Clone, Copy, Debug)]
struct SeedSums {
    amalgam: [CorrSums; 2],
    first: [CorrSums; 2],
}

impl SeedSums {
    fn add(&mut self, s: &SeedSample, sign: f64) {
        self.amalgam[0].add(s.exc_yaw, s.amalgam.0, sign);
        self.amalgam[1].add(s.exc_pitch, s.amalgam.1, sign);
        self.first[0].add(s.exc_yaw, s.first.0, sign);
        self.first[1].add(s.exc_pitch, s.first.1, sign);
    }
}

fn wrap_degrees(d: f32) -> f32 {
    let d = d.rem_euclid(360.0);
    if d > 180.0 {
        d - 360.0
    } else {
        d
    }
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

/// Entity updates further apart than this (in sim ticks) are not consecutive.
const MAX_UPDATE_GAP: i64 = 4;
/// Ticks of entity updates kept per player.
const HISTORY_TICKS: u32 = 48;
/// A pulse is searched in the updates within this many ticks of its anchor (shot or idle time).
const PULSE_WINDOW: u32 = 8;
/// The jump away from the mouse path starts this many ticks before to `PULSE_OUT_AFTER` after the bullet.
const PULSE_OUT_BEFORE: u32 = 3;
const PULSE_OUT_AFTER: u32 = 1;
/// The jump back must follow within this many ticks.
const PULSE_MAX_TICKS: u32 = 6;
/// Smallest jump (degrees, after removing the mouse motion) that starts or ends a pulse.
const PULSE_STEP: f32 = 0.6;
/// The view is back on the mouse path when the offset shrinks below this share of the pulse.
const PULSE_RETURN: f32 = 0.35;
/// Faster mouse motion (degrees per sim tick) is not evaluated.
const MAX_MOUSE_SPEED: f32 = 1.0;
/// The pulse must be this many times larger than the other jerks in its window.
const PULSE_BACKGROUND_MULT: f32 = 3.0;
/// Shots further apart than this start a new series.
const SERIES_GAP_TICKS: u32 = 20;
/// Idle anchors: no bullet within this many ticks, spaced like minigun shots.
const IDLE_CLEARANCE_TICKS: u32 = 15;
const IDLE_SPACING_TICKS: u32 = 7;
/// Bullets are paired with updates once this many ticks have passed.
const SEED_EVAL_DELAY: u32 = 4;
/// Shots after a longer pause fire a perfectly accurate first bullet, which no cheat corrects
/// (FX_FireBullets: 0.25 s for multi-pellet weapons, 1.25 s for single bullets).
const PERFECT_GAP_MULTI: u32 = 17;
const PERFECT_GAP_SINGLE: u32 = 83;

#[derive(Clone, Copy, Debug)]
struct Shot {
    tick: u32,
    seed: Option<i32>,
    spread: Option<f32>,
    weapon: WeaponSpreadInfo,
    weapon_id: Option<u32>,
}

impl Shot {
    fn spread_degrees(&self) -> f32 {
        self.spread.filter(|s| *s > 0.0).unwrap_or(self.weapon.default_spread).to_degrees()
    }
}

#[derive(Clone, Copy, Debug)]
struct SeedSample {
    exc_yaw: f32,
    exc_pitch: f32,
    // Predicted compensation (degrees) for Amalgam's pellet and for the first pellet (RijiN).
    amalgam: (f32, f32),
    first: (f32, f32),
}

/// One networked state of the player's eye angles.
#[derive(Clone, Copy, Debug)]
struct AngleUpdate {
    tick: u32,
    sim: i64,
    pitch: f32,
    yaw: f32,
}

/// A temporary offset of the view from the mouse path: jump away, hold, jump back.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pulse {
    tick: u32,
    /// Largest offset (yaw, pitch) in degrees.
    offset: (f32, f32),
    height: f32,
    /// Updates between the jump away and the jump back (1 = a one-update spike).
    updates: usize,
}

/// Searches a pulse that starts near `anchor`. None when the window cannot be evaluated (too
/// few updates or fast mouse motion), Some(None) when it can but holds no pulse.
fn find_pulse(updates: &VecDeque<AngleUpdate>, anchor: u32, min_height: f32) -> Option<Option<Pulse>> {
    let window: Vec<&AngleUpdate> = updates
        .iter()
        .filter(|u| u.tick + PULSE_WINDOW >= anchor && u.tick <= anchor + PULSE_WINDOW)
        .collect();
    if window.len() < 4 {
        return None;
    }
    // Per-update deltas (tick of the later update, sim gap, yaw, pitch).
    let deltas: Vec<(u32, f32, f32, f32)> = window
        .windows(2)
        .map(|w| {
            let gap = (w[1].sim - w[0].sim).clamp(1, MAX_UPDATE_GAP) as f32;
            (w[1].tick, gap, wrap_degrees(w[1].yaw - w[0].yaw), w[1].pitch - w[0].pitch)
        })
        .collect();
    let median = |mut v: Vec<f32>| {
        v.sort_by(f32::total_cmp);
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    };
    // The mouse velocity per sim tick, robust to the pulse itself.
    let vy = median(deltas.iter().map(|d| d.2 / d.1).collect());
    let vp = median(deltas.iter().map(|d| d.3 / d.1).collect());
    if vy.hypot(vp) > MAX_MOUSE_SPEED {
        return None;
    }
    let jerk: Vec<(u32, f32, f32)> = deltas.iter().map(|d| (d.0, d.2 - vy * d.1, d.3 - vp * d.1)).collect();

    let mut best: Option<Pulse> = None;
    for (i, &(ti, jy, jp)) in jerk.iter().enumerate() {
        if ti + PULSE_OUT_BEFORE < anchor || ti > anchor + PULSE_OUT_AFTER || jy.hypot(jp) < PULSE_STEP {
            continue;
        }
        let (mut sy, mut sp) = (jy, jp);
        let (mut peak, mut peak_h) = ((jy, jp), jy.hypot(jp));
        for (k, &(tk, ky, kp)) in jerk.iter().enumerate().skip(i + 1) {
            if tk > ti + PULSE_MAX_TICKS {
                break;
            }
            sy += ky;
            sp += kp;
            let h = sy.hypot(sp);
            if h <= PULSE_RETURN * peak_h && ky.hypot(kp) >= PULSE_STEP {
                let rest: Vec<f32> = jerk
                    .iter()
                    .enumerate()
                    .filter(|(m, _)| *m < i || *m > k)
                    .map(|(_, x)| x.1.hypot(x.2))
                    .collect();
                let background = median(rest);
                if peak_h >= min_height
                    && peak_h >= PULSE_BACKGROUND_MULT * background
                    && best.is_none_or(|b| peak_h > b.height)
                {
                    best = Some(Pulse { tick: ti, offset: peak, height: peak_h, updates: k - i });
                }
                break;
            }
            if h > peak_h {
                peak_h = h;
                peak = (sy, sp);
            }
        }
    }
    Some(best)
}

/// One evaluated shot of a series.
#[derive(Clone, Copy, Debug)]
struct ShotEval {
    tick: u32,
    pulse: Option<Pulse>,
    spread_deg: f32,
    weapon: WeaponSpreadInfo,
    weapon_id: Option<u32>,
}

#[derive(Default)]
struct PlayerTracker {
    /// Entity updates of the last ticks while the player was alive, visible and not just spawned.
    updates: VecDeque<AngleUpdate>,
    /// Ticks of recent bullets of any spread weapon (idle anchors keep clear of them).
    recent_shot_ticks: VecDeque<u32>,
    /// Rapid-fire bullets waiting for the updates after them.
    pending_pulse: VecDeque<Shot>,
    /// Bullets waiting to be paired with their update for the seed correlation.
    pending_seed: VecDeque<Shot>,
    /// The evaluated shots of the current series.
    series: Vec<ShotEval>,
    next_idle_anchor: u32,
    idle_anchors: usize,
    idle_pulses: usize,
    last_rapid_spread_deg: Option<f32>,
    last_seed_shot_tick: Option<u32>,
    seed_shots: usize,
    seed_sums: SeedSums,
}

impl PlayerTracker {
    fn seed_z(&self) -> (f64, &'static str) {
        let mut best = (f64::NEG_INFINITY, "closest_to_average");
        for (model, sums) in [("closest_to_average", self.seed_sums.amalgam), ("first_bullet", self.seed_sums.first)] {
            let z = (sums[0].correlation_z().1 + sums[1].correlation_z().1) / std::f64::consts::SQRT_2;
            if z > best.0 {
                best = (z, model);
            }
        }
        best
    }
}

/// The deltas into and out of the middle of three consecutive updates, as excursion (yaw, pitch).
fn excursion(a: &AngleUpdate, b: &AngleUpdate, c: &AngleUpdate) -> Option<(f32, f32)> {
    if b.sim - a.sim > MAX_UPDATE_GAP || c.sim - b.sim > MAX_UPDATE_GAP {
        return None;
    }
    let d1 = (wrap_degrees(b.yaw - a.yaw), b.pitch - a.pitch);
    let d2 = (wrap_degrees(c.yaw - b.yaw), c.pitch - b.pitch);
    Some(((d1.0 - d2.0) / 2.0, (d1.1 - d2.1) / 2.0))
}

/// A series that passed every series test, waiting for the player-level confirmation.
struct Candidate {
    player: u64,
    log10_p: f64,
    seeded: bool,
    detection: Detection,
}

#[derive(Default)]
pub struct NoSpread {
    params: Parameters,
    jg: JankGuard,
    trackers: HashMap<u64, PlayerTracker>,
    candidates: Vec<Candidate>,
}

impl NoSpread {
    pub fn new() -> Self {
        Self {
            params: HashMap::from([
                ("enabled".to_string(), Parameter::Bool(true)),
                ("min_pulses".to_string(), Parameter::Int(3)),
                ("min_pulse_rate".to_string(), Parameter::Float(0.1)),
                ("min_pulse_height".to_string(), Parameter::Float(0.7)),
                // Silent-aim shape: flicks beyond the spread cone, large offsets, one direction.
                ("max_beyond_cone_fraction".to_string(), Parameter::Float(0.2)),
                ("max_pulse_to_spread".to_string(), Parameter::Float(0.9)),
                ("max_direction_consistency".to_string(), Parameter::Float(0.7)),
                ("max_series_p_value".to_string(), Parameter::Float(1e-5)),
                // The product of a player's series p-values (single weak series are legit noise).
                ("max_player_p_value".to_string(), Parameter::Float(1e-7)),
                ("min_idle_rate".to_string(), Parameter::Float(0.005)),
                ("min_seed_shots".to_string(), Parameter::Int(20)),
                ("min_seed_z".to_string(), Parameter::Float(6.0)),
            ]),
            jg: JankGuard::default(),
            trackers: HashMap::new(),
            candidates: Vec::new(),
        }
    }

    /// Evaluates the finished series of a player: one detection per nospread series.
    fn close_series(&mut self, sid: u64) {
        let min_pulses = get_parameter_value::<i32>(&self.params, "min_pulses").max(1) as usize;
        let min_rate: f32 = get_parameter_value(&self.params, "min_pulse_rate");
        let max_beyond: f32 = get_parameter_value(&self.params, "max_beyond_cone_fraction");
        let max_p: f32 = get_parameter_value(&self.params, "max_series_p_value");
        let min_idle_rate: f32 = get_parameter_value(&self.params, "min_idle_rate");
        let min_seed_shots = get_parameter_value::<i32>(&self.params, "min_seed_shots").max(4) as usize;
        let min_seed_z: f32 = get_parameter_value(&self.params, "min_seed_z");

        let Some(tracker) = self.trackers.get_mut(&sid) else { return };
        let series = std::mem::take(&mut tracker.series);
        let pulses: Vec<(&ShotEval, Pulse)> = series.iter().filter_map(|s| s.pulse.map(|p| (s, p))).collect();
        let (n, k) = (series.len(), pulses.len());
        if k < min_pulses || (k as f32) < min_rate * n as f32 {
            return;
        }
        // Nospread offsets stay inside the spread cone; larger flicks are silent aim.
        let beyond = pulses
            .iter()
            .filter(|(s, p)| p.offset.0.abs().max(p.offset.1.abs()) > s.spread_deg + 0.5)
            .count();
        if beyond as f32 > max_beyond * k as f32 {
            return;
        }
        let idle_rate = if tracker.idle_anchors > 0 {
            tracker.idle_pulses as f64 / tracker.idle_anchors as f64
        } else {
            0.0
        };
        let p = binomial_sf(k, n, idle_rate.max(min_idle_rate as f64));
        let (seed_z, model) = tracker.seed_z();
        let seeded = tracker.seed_shots >= min_seed_shots && seed_z >= min_seed_z as f64;
        if p > max_p as f64 && !seeded {
            return;
        }
        // Shape: silent-aim flicks are one-update spikes in one direction (towards the target),
        // nospread offsets scatter around the crosshair.
        let spikes = pulses.iter().filter(|(_, p)| p.updates == 1).count();
        let (ux, uy) = pulses.iter().fold((0.0f32, 0.0f32), |acc, (_, p)| {
            (acc.0 + p.offset.0 / p.height, acc.1 + p.offset.1 / p.height)
        });
        let resultant = ux.hypot(uy) / k as f32;
        let mut ratios: Vec<f32> = pulses.iter().map(|(s, p)| p.height / s.spread_deg.max(0.1)).collect();
        ratios.sort_by(f32::total_cmp);
        let max_ratio: f32 = get_parameter_value(&self.params, "max_pulse_to_spread");
        let max_consistency: f32 = get_parameter_value(&self.params, "max_direction_consistency");
        if ratios[k / 2] > max_ratio || resultant > max_consistency {
            return;
        }
        let mut heights: Vec<f32> = pulses.iter().map(|(_, p)| p.height).collect();
        heights.sort_by(f32::total_cmp);
        let mut weapons: Vec<(&'static str, Option<u32>)> = series.iter().map(|s| (s.weapon.name, s.weapon_id)).collect();
        weapons.sort();
        let weapon = weapons
            .chunk_by(|a, b| a == b)
            .max_by_key(|c| c.len())
            .map(|c| c[0])
            .unwrap_or(("", None));
        let log10_p = p.max(1e-300).log10();
        let detection = Detection {
            tick: pulses[0].1.tick,
            algorithm: "fidoo/nospread".to_string(),
            player: sid,
            data: json!({
                "type": "NospreadSeries",
                "weapon_id": weapon.1,
                "weapon_name": weapon.0,
                "series_start": series[0].tick,
                "series_end": series[n - 1].tick,
                "shots": n,
                "pulses": k,
                "beyond_cone": beyond,
                "one_update_pulses": spikes,
                "direction_consistency": (resultant as f64 * 100.0).round() / 100.0,
                "median_pulse_to_spread": (ratios[k / 2] as f64 * 100.0).round() / 100.0,
                "median_pulse_degrees": (heights[k / 2] as f64 * 100.0).round() / 100.0,
                "idle_pulses": tracker.idle_pulses,
                "idle_anchors": tracker.idle_anchors,
                "log10_p_value": (log10_p * 10.0).round() / 10.0,
                "seed_z_score": if tracker.seed_shots >= 4 { Some((seed_z * 10.0).round() / 10.0) } else { None },
                "seed_model": model,
            }),
        };
        self.candidates.push(Candidate { player: sid, log10_p, seeded, detection });
    }

    /// Evaluates the shots and idle anchors whose windows are complete.
    fn evaluate(&mut self, sid: u64, ticknum: u32) {
        let min_height: f32 = get_parameter_value(&self.params, "min_pulse_height");
        let mut finished_series = false;
        {
            let Some(tracker) = self.trackers.get_mut(&sid) else { return };
            while let Some(shot) = tracker.pending_pulse.front().copied() {
                if ticknum < shot.tick + PULSE_WINDOW {
                    break;
                }
                if tracker.series.last().is_some_and(|l| shot.tick > l.tick + SERIES_GAP_TICKS) {
                    // This shot starts the next series; it stays queued until the close below.
                    finished_series = true;
                    break;
                }
                tracker.pending_pulse.pop_front();
                let spread_deg = shot.spread_degrees();
                tracker.last_rapid_spread_deg = Some(spread_deg);
                if let Some(pulse) = find_pulse(&tracker.updates, shot.tick, min_height) {
                    tracker.series.push(ShotEval {
                        tick: shot.tick,
                        pulse,
                        spread_deg,
                        weapon: shot.weapon,
                        weapon_id: shot.weapon_id,
                    });
                }
            }
            if !finished_series && tracker.series.last().is_some_and(|l| {
                ticknum > l.tick + SERIES_GAP_TICKS + PULSE_WINDOW && tracker.pending_pulse.is_empty()
            }) {
                finished_series = true;
            }

            // Idle anchors: the same pulse search where no bullet was fired.
            if tracker.next_idle_anchor == 0 {
                tracker.next_idle_anchor = ticknum;
            }
            while ticknum >= tracker.next_idle_anchor + IDLE_CLEARANCE_TICKS.max(PULSE_WINDOW) {
                let anchor = tracker.next_idle_anchor;
                tracker.next_idle_anchor += IDLE_SPACING_TICKS;
                if tracker.recent_shot_ticks.iter().any(|&t| t.abs_diff(anchor) <= IDLE_CLEARANCE_TICKS) {
                    continue;
                }
                if let Some(pulse) = find_pulse(&tracker.updates, anchor, min_height) {
                    tracker.idle_anchors += 1;
                    tracker.idle_pulses += pulse.is_some() as usize;
                }
            }
        }
        if finished_series {
            self.close_series(sid);
            // Continue with the shots of the next series.
            self.evaluate(sid, ticknum);
        }
    }

    /// Pairs bullets old enough with the update of their tick for the seed correlation.
    fn evaluate_seeds(&mut self, sid: u64, ticknum: u32) {
        let Some(tracker) = self.trackers.get_mut(&sid) else { return };
        while tracker.pending_seed.front().is_some_and(|s| ticknum >= s.tick + SEED_EVAL_DELAY) {
            let shot = tracker.pending_seed.pop_front().unwrap();
            let gap = tracker.last_seed_shot_tick.map_or(u32::MAX, |t| shot.tick.saturating_sub(t));
            tracker.last_seed_shot_tick = Some(shot.tick);
            let perfect_gap = if shot.weapon.bullets > 1 { PERFECT_GAP_MULTI } else { PERFECT_GAP_SINGLE };
            let (Some(seed), Some(_)) = (shot.seed, shot.spread.filter(|s| *s > 0.0)) else { continue };
            if gap > perfect_gap {
                continue;
            }
            // The update of the bullet's tick with the largest excursion (several can share a tick).
            let u = &tracker.updates;
            let best = (1..u.len().saturating_sub(1))
                .filter(|&i| u[i].tick == shot.tick)
                .filter_map(|i| excursion(&u[i - 1], &u[i], &u[i + 1]))
                .max_by(|x, y| (x.0.hypot(x.1)).total_cmp(&y.0.hypot(y.1)));
            let Some((exc_yaw, exc_pitch)) = best else { continue };
            let spread_deg = shot.spread_degrees();
            let first = bullet_spread(seed);
            let amalgam = amalgam_corrected_spread(seed, shot.weapon.bullets);
            tracker.seed_sums.add(
                &SeedSample {
                    exc_yaw,
                    exc_pitch,
                    amalgam: (amalgam.0 * spread_deg, amalgam.1 * spread_deg),
                    first: (first.0 * spread_deg, first.1 * spread_deg),
                },
                1.0,
            );
            tracker.seed_shots += 1;
        }
    }
}

impl<'a> CheatAlgorithm<'a> for NoSpread {
    fn algorithm_name(&self) -> &str {
        "fidoo/nospread"
    }

    fn tooltip(&self) -> &'static str {
        "Detects No-Spread cheats on rapid-fire hitscan weapons (one detection per series of jitters)."
    }

    fn params(&mut self) -> Option<&mut Parameters> {
        Some(&mut self.params)
    }

    fn excluded_from_sliding_window(&self) -> bool {
        // One detection per series of nospread jitters, already aggregated over the series.
        true
    }

    fn needs_whole_demo(&self) -> bool {
        // The idle pulse baseline and the seed correlation aggregate the player's whole demo.
        true
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
                if class.as_str() != "CTEFireBullets" {
                    continue;
                }
                const BULLETS_PLAYER: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iPlayer");
                const BULLETS_SEED: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iSeed");
                const BULLETS_SPREAD: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_flSpread");
                const BULLETS_WEAPON: SendPropIdentifier = SendPropIdentifier::new("DT_TEFireBullets", "m_iWeaponID");

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

                let Some(player_raw) = player_opt else { continue };
                let Some(sid) = resolve_player_sid(player_raw, state) else { continue };
                let weapon_name = if weapon_opt.is_none() {
                    state
                        .players
                        .iter()
                        .find(|p| p.entity == EntityId::from(player_raw + 1))
                        .map(|p| state.get_player_weapon(p))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                // Shots from weapons without spread (e.g. sniper rifles) are not evaluated.
                let Some(weapon) = get_weapon_spread_info(weapon_opt, &weapon_name) else { continue };
                let shot = Shot { tick: tick.into(), seed: seed_opt, spread: spread_opt, weapon, weapon_id: weapon_opt };
                let tracker = self.trackers.entry(sid).or_default();
                tracker.recent_shot_ticks.push_back(shot.tick);
                while tracker.recent_shot_ticks.len() > 32 {
                    tracker.recent_shot_ticks.pop_front();
                }
                if weapon.rapid {
                    tracker.pending_pulse.push_back(shot);
                }
                tracker.pending_seed.push_back(shot);
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

        for player in state.players.iter().filter(|p| {
            p.in_pvs
                && p.state == PlayerState::Alive
                && p.info.as_ref().is_some_and(|info| info.steam_id != "BOT")
        }) {
            let Some(info) = &player.info else { continue };
            let steam_id: u64 = match SteamID::from_steam3(&info.steam_id) {
                Ok(sid) => u64::from(sid),
                Err(_) => continue,
            };

            let ticks_since_event = self
                .jg
                .teleported(&steam_id, ticknum)
                .min(self.jg.spawned(&steam_id, ticknum));

            let tracker = self.trackers.entry(steam_id).or_default();
            if ticks_since_event < 60 {
                if ticks_since_event == 0 {
                    self.candidates.retain(|c| {
                        c.player != steam_id || ticknum.saturating_sub(c.detection.tick) > 60
                    });
                }
                // Angles around a spawn or teleport are never evaluated.
                tracker.updates.clear();
                continue;
            }

            // A new entity update: the sim tick or the angles changed (demo ticks can repeat and
            // several updates can arrive in one demo tick).
            let update = AngleUpdate {
                tick: ticknum,
                sim: state.sim_tick(player),
                pitch: player.pitch_angle,
                yaw: player.view_angle,
            };
            let changed = tracker.updates.back().is_none_or(|u| {
                u.sim != update.sim || u.pitch != update.pitch || u.yaw != update.yaw
            });
            if !changed {
                continue;
            }
            // Signon frames carry the server tick; demo ticks restart after them.
            if tracker.updates.back().is_some_and(|u| u.tick > ticknum || u.sim > update.sim) {
                tracker.updates.clear();
            }
            tracker.updates.push_back(update);
            while tracker.updates.front().is_some_and(|u| u.tick + HISTORY_TICKS < ticknum) {
                tracker.updates.pop_front();
            }
        }

        let sids: Vec<u64> = self.trackers.keys().copied().collect();
        for sid in sids {
            self.evaluate(sid, ticknum);
            self.evaluate_seeds(sid, ticknum);
        }

        Ok(vec![])
    }

    fn finish(&mut self) -> Result<Vec<Detection>, Error> {
        let sids: Vec<u64> = self.trackers.keys().copied().collect();
        for sid in sids {
            self.close_series(sid);
        }
        // Detections are held until the end because a respawn/teleport retracts the ones just
        // before it, and a player's series are only reported when they confirm each other.
        let max_player_p: f32 = get_parameter_value(&self.params, "max_player_p_value");
        let mut pooled: HashMap<u64, (f64, bool)> = HashMap::new();
        for c in &self.candidates {
            let e = pooled.entry(c.player).or_insert((0.0, false));
            e.0 += c.log10_p;
            e.1 |= c.seeded;
        }
        let threshold = (max_player_p as f64).max(1e-300).log10();
        let mut detections: Vec<Detection> = std::mem::take(&mut self.candidates)
            .into_iter()
            .filter_map(|mut c| {
                let (log10_p, seeded) = pooled[&c.player];
                if log10_p > threshold && !seeded {
                    return None;
                }
                c.detection.data["player_log10_p_value"] = json!((log10_p * 10.0).round() / 10.0);
                Some(c.detection)
            })
            .collect();
        detections.sort_by_key(|d| (d.tick, d.player));
        Ok(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_finish_pools_a_players_series() {
        let candidate = |player, log10_p, tick| Candidate {
            player,
            log10_p,
            seeded: false,
            detection: Detection {
                tick,
                algorithm: "fidoo/nospread".to_string(),
                player,
                data: json!({"type": "NospreadSeries"}),
            },
        };
        let mut algo = NoSpread::new();
        // Player 1: two weak series confirm each other (1e-5 * 1e-4 <= 1e-7).
        algo.candidates.push(candidate(1, -5.0, 100));
        algo.candidates.push(candidate(1, -4.0, 900));
        // Player 2: one weak series alone is not reported.
        algo.candidates.push(candidate(2, -5.5, 200));
        // Player 3: one strong series.
        algo.candidates.push(candidate(3, -12.0, 300));

        let finished = algo.finish().unwrap();
        let players: Vec<u64> = finished.iter().map(|d| d.player).collect();
        assert_eq!(players, vec![1, 3, 1]);
        assert_eq!(finished[0].data["player_log10_p_value"], json!(-9.0));
    }

    #[test]
    fn test_weapon_id_decides_spread_family() {
        // TF_WEAPON_SCATTERGUN (16) is not an SMG; TF_WEAPON_SMG is 19.
        assert_eq!(get_weapon_spread_info(Some(16), "SMG").map(|w| w.name), Some("Scattergun"));
        assert_eq!(get_weapon_spread_info(Some(19), "").map(|w| w.name), Some("SMG"));
        // Frontier Justice (TF_WEAPON_SENTRY_REVENGE) is a shotgun.
        assert_eq!(get_weapon_spread_info(Some(69), "").map(|w| w.name), Some("Shotgun"));
        assert_eq!(get_weapon_spread_info(Some(18), "").map(|w| w.bullets), Some(4));
        // Sniper rifles have no spread, whatever the held weapon name says.
        assert!(get_weapon_spread_info(Some(17), "SMG").is_none());
        assert!(get_weapon_spread_info(Some(22), "").is_none());
        assert_eq!(get_weapon_spread_info(None, "The Brass Beast").map(|w| w.name), Some("Minigun"));
    }

    #[test]
    fn test_spread_uses_low_seed_byte() {
        assert_eq!(bullet_spread(42), bullet_spread(42 + 256));
        let (x, y) = bullet_spread(42);
        assert!(x.abs() <= 1.0 && y.abs() <= 1.0);
    }

    #[test]
    fn test_amalgam_pellet_is_closest_to_average() {
        // One bullet: the first pellet. Several: one of the pellets, nearest their mean.
        assert_eq!(amalgam_corrected_spread(7, 1), bullet_spread(7));
        let chosen = amalgam_corrected_spread(7, 4);
        assert!((0..4).any(|i| bullet_spread(7 + i) == chosen));
    }

    #[test]
    fn test_binomial_sf() {
        assert_eq!(binomial_sf(0, 10, 0.1), 1.0);
        assert!((binomial_sf(1, 1, 0.3) - 0.3).abs() < 1e-12);
        // P(X >= 2), n = 3, p = 0.5: (3 + 1) / 8
        assert!((binomial_sf(2, 3, 0.5) - 0.5).abs() < 1e-12);
        assert!(binomial_sf(20, 40, 0.01) < 1e-20);
    }

    #[test]
    fn test_only_rapid_fire_weapons_count_reversals() {
        for id in [18, 19, 103, 41, 42, 75] {
            assert!(get_weapon_spread_info(Some(id), "").unwrap().rapid, "{id}");
        }
        for id in [16, 12, 43, 71, 69] {
            assert!(!get_weapon_spread_info(Some(id), "").unwrap().rapid, "{id}");
        }
    }

    fn updates(path: &[(u32, f32, f32)]) -> VecDeque<AngleUpdate> {
        path.iter().map(|&(tick, pitch, yaw)| AngleUpdate { tick, sim: tick as i64, pitch, yaw }).collect()
    }

    #[test]
    fn test_pulse_is_a_held_offset_off_the_mouse_path() {
        // Mouse turning 0.35 deg/tick in yaw; at the shot (tick 10) pitch jumps +1.41 for two
        // updates and comes back: a nospread pulse.
        let mut path: Vec<(u32, f32, f32)> = (0..20).map(|t| (t, 5.0, t as f32 * 0.35)).collect();
        path[9].1 += 1.41;
        path[10].1 += 1.41;
        let pulse = find_pulse(&updates(&path), 10, 0.7).unwrap().unwrap();
        assert_eq!(pulse.tick, 9);
        assert!((pulse.height - 1.41).abs() < 0.05);
        assert_eq!(pulse.updates, 2);
        // No offset: evaluable, no pulse.
        let calm: Vec<(u32, f32, f32)> = (0..20).map(|t| (t, 5.0, t as f32 * 0.35)).collect();
        assert_eq!(find_pulse(&updates(&calm), 10, 0.7), Some(None));
        // A single pitch quantum that never returns is not a pulse.
        let mut step = calm.clone();
        for p in step.iter_mut().skip(9) {
            p.1 += 0.71;
        }
        assert_eq!(find_pulse(&updates(&step), 10, 0.7), Some(None));
    }

    #[test]
    fn test_fast_turns_are_not_evaluated() {
        let fast: Vec<(u32, f32, f32)> = (0..20).map(|t| (t, 5.0, t as f32 * 4.0)).collect();
        assert_eq!(find_pulse(&updates(&fast), 10, 0.7), None);
    }

    #[test]
    fn test_seed_correlation_is_positive_for_compensation() {
        // Excursion = predicted compensation (+x yaw, +y pitch) plus noise: positive z.
        let mut sums = CorrSums::default();
        for (i, p) in (0..60).map(|s| bullet_spread(s * 3)).enumerate() {
            sums.add(p.0 + 0.1 * ((i % 5) as f32 - 2.0), p.0, 1.0);
        }
        let (r, z) = sums.correlation_z();
        assert!(r > 0.8 && z > 6.0);
        // Removing the samples again leaves an empty correlation.
        let mut removed = sums;
        for (i, p) in (0..60).map(|s| bullet_spread(s * 3)).enumerate() {
            removed.add(p.0 + 0.1 * ((i % 5) as f32 - 2.0), p.0, -1.0);
        }
        assert_eq!(removed.correlation_z(), (0.0, 0.0));
    }
}
