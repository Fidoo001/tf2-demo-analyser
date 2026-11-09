//! The player screen: loading, playback, cameras and everything on top of the 3D view.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use eframe::{egui_glow, glow};
use egui::{Align2, Color32, FontId, Key, PointerButton, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use glam::{Mat4, Vec3, Vec4};

use super::characters::Characters;
use super::map::{load_map, rotation_from_angles, DemoContent, LightProbe, MapData, MapProp, ModelLighting, SkyCamera};
use super::audio::ear_gains;
use super::demo_sounds::{CHAN_WEAPON, SND_CHANGE_PITCH, SND_CHANGE_VOL, SND_STOP, SND_STOP_LOOPING};
use super::recording::{
    read_demo, Kill, ObjectFrame, ObjectKind, PlayerFrame, Recording, ANIM_ATTACK_PRIMARY, ANIM_ATTACK_SECONDARY,
    ANIM_DOUBLE_JUMP, ANIM_JUMP, ANIM_RELOAD, ANIM_RELOAD_END, ANIM_RELOAD_LOOP, CLASS_HEAVY, CLASS_MEDIC, TEAM_BLUE, TEAM_RED,
};

/// How far back to look when working out which way someone is moving.
const VELOCITY_WINDOW_SECONDS: f64 = 0.1;
/// Slower than this counts as standing still.
const MOVING_SPEED: f32 = 40.0;
/// Speeds this fast play the run animation at full stride.
const FULL_RUN_SPEED: f32 = 240.0;
/// How far back to look for a shooting or reloading animation still playing.
const GESTURE_LOOKBACK_SECONDS: f64 = 2.5;
/// TF2 plays the start of a jump for this long, then the floating part.
const JUMP_START_SECONDS: f32 = 0.5;
/// How long a body takes to blend into a new movement animation, like TF2 does
/// when someone stops running, lands or crouches. In match time.
const STANCE_BLEND_SECONDS: f64 = 0.2;
/// Leaving the ground faster than this, upwards, is a jump rather than a step off a ledge.
const JUMP_SPEED: f32 = 120.0;
/// Rising faster than this is being flung, as by a rocket, which TF2 shows as an air walk.
const FLUNG_SPEED: f32 = 300.0;
/// Positions of other players arrive every couple of ticks, so a single frame can
/// hold two ticks of movement. Rising speed is measured across this many frames.
const RISE_WINDOW_FRAMES: usize = 4;
/// How far back to look for where a player left the ground.
const MAX_AIR_FRAMES: usize = 600;
use super::ragdoll::Ragdoll;
use super::renderer::{Actor, Frame, Renderer, Shape, ShapeInstance, TRACE_FLOATS};
use super::skeleton::{DeathKind, Gesture, Pose, PoseInput, Rig, Stance, WeaponPose, HITGROUP_HEAD};

const SPEEDS: [f32; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
const SKIP_SECONDS: f64 = 5.0;
const LONG_SKIP_SECONDS: f64 = 30.0;
/// Jumping to a detection starts a little early so the moment can be seen building up.
const DETECTION_LEAD_SECONDS: f64 = 3.0;

/// Review mode starts each detection this long before it, in match time.
const REVIEW_LEAD_SECONDS: f64 = 2.0;
/// and plays on this long after it.
const REVIEW_FOLLOW_SECONDS: f64 = 1.5;
/// How long it pauses on the detection itself, in real time.
const REVIEW_HOLD_SECONDS: f64 = 1.5;
/// How long it waits before going on to the next detection, in real time.
const REVIEW_NEXT_SECONDS: f64 = 2.5;
/// Detections closer together than this are reviewed as one moment.
const REVIEW_GROUP_SECONDS: f64 = 1.0;
/// Review plays at the slowest speed, 0.25x.
const REVIEW_SPEED: usize = 0;
/// How many measured values the review bar lists before tucking the rest into a tooltip.
const REVIEW_MAX_FIELDS: usize = 9;
const DEFAULT_FOV: f32 = 90.0;
/// TF2's default `viewmodel_fov`.
const VIEW_MODEL_FOV: f32 = 54.0;
/// The field of view through a sniper scope.
const SCOPED_FOV: f32 = 20.0;
/// TF2's crosshair is 32 pixels on a 480 pixel tall screen.
const CROSSHAIR_SCALE: f32 = 32.0 / 480.0;
/// A first-person weapon bone further than this from the hand isn't where the gun is held.
const GRIP_DISTANCE: f32 = 12.0;
/// Where a gun with a broken first-person grip is held, in view space (forward, left, up).
const HELD_GUN_OFFSET: Vec3 = Vec3::new(16.0, -17.0, -19.0);
/// How long a bullet trace stays on screen.
const TRACE_SECONDS: f64 = 0.6;
/// How far a hitscan bullet can travel in TF2.
const BULLET_RANGE: f32 = 8192.0;
/// How much of a projectile's path stays visible behind it.
const TRAIL_SECONDS: f64 = 1.0;
/// Trails aren't drawn closer than this to the camera.
const TRAIL_CLEAR_OF_EYE: f32 = 48.0;
/// Where the third-person camera sits from the player's eyes: behind, to the
/// right, and up, in units.
const CHASE_BEHIND: f32 = 110.0;
const CHASE_RIGHT: f32 = 30.0;
const CHASE_UP: f32 = 10.0;
/// How far the third-person camera stays in front of a wall behind the player.
const CHASE_WALL_GAP: f32 = 10.0;
/// How close and how far the mouse wheel can bring the third-person camera.
const CHASE_NEAREST: f32 = 50.0;
const CHASE_FURTHEST: f32 = 500.0;
/// When swung around a player, the camera turns about a point this far below their eyes, about their chest.
const ORBIT_DROP: f32 = 18.0;
/// Steepest the swung camera looks down or up, in degrees.
const ORBIT_PITCH_LIMIT: f32 = 85.0;
/// Faster than this, the demo plays without sound.
const MAX_SOUND_SPEED: f32 = 2.0;
/// A jump in time bigger than this is a seek, not playing, so the sounds in
/// between are skipped instead of all playing at once.
const MAX_SOUND_STEP_SECONDS: f64 = 0.5;
/// Voice chat recordings of one player closer together than this are joined.
const VOICE_JOIN_SECONDS: f32 = 1.0;
const DEFAULT_VOLUME: f32 = 0.7;
/// TF2's explosion sounds (`BaseExplosionEffect.Sound`), at 95 dB.
const EXPLOSION_SOUNDS: [&str; 3] = ["sound/weapons/explode1.wav", "sound/weapons/explode2.wav", "sound/weapons/explode3.wav"];
const EXPLOSION_LEVEL: f32 = 95.0;
/// How far the recorder's own shots carry, like TF2's guns.
const OWN_SHOT_LEVEL: f32 = 94.0;
const HIT_SOUND: &str = "sound/ui/hitsound.wav";
const KILL_SOUND: &str = "sound/ui/killsound.wav";
const DING_VOLUME: f32 = 0.6;
/// Tags keep the kinds of sounds apart in the mixer, so one kind can be stopped.
const DEMO_SOUND_TAG: u64 = 1 << 56;
const VOICE_TAG: u64 = 2 << 56;
const TAG_KIND: u64 = 0xff << 56;
/// What the recorder's weapon sounds like when nobody else in the demo fired the
/// same one: words in its model's name, and TF2's usual sound for it. More
/// particular names come first.
const USUAL_FIRE_SOUNDS: [(&str, &str); 30] = [
    ("scatter_gun_double", "sound/weapons/scatter_gun_double_shoot.wav"),
    ("scattergun", "sound/weapons/scatter_gun_shoot.wav"),
    ("shotgun", "sound/weapons/shotgun_shoot.wav"),
    ("pistol", "sound/weapons/pistol_shoot.wav"),
    ("smg", "sound/weapons/smg_shoot.wav"),
    ("minigun", "sound/weapons/minigun_shoot.wav"),
    ("sniperrifle", "sound/weapons/sniper_shoot.wav"),
    ("ambassador", "sound/weapons/revolver_shoot.wav"),
    ("revolver", "sound/weapons/revolver_shoot.wav"),
    ("directhit", "sound/weapons/rocket_directhit_shoot.wav"),
    ("blackbox", "sound/weapons/rocket_blackbox_shoot.wav"),
    ("liberty", "sound/weapons/rocket_ll_shoot.wav"),
    ("rocketjumper", "sound/weapons/rocket_jumper_shoot.wav"),
    ("quake", "sound/weapons/quake_rpg_fire_remastered.wav"),
    ("rocketlauncher", "sound/weapons/rocket_shoot.wav"),
    ("stickybomb", "sound/weapons/stickybomblauncher_shoot.wav"),
    ("grenadelauncher", "sound/weapons/grenade_launcher_shoot.wav"),
    ("flaregun", "sound/weapons/flaregun_shoot.wav"),
    ("syringegun", "sound/weapons/syringegun_shoot.wav"),
    ("crossbow", "sound/weapons/crusaders_crossbow_shoot.wav"),
    ("bow", "sound/weapons/bow_shoot.wav"),
    ("dragons_fury", "sound/weapons/dragons_fury_shoot.wav"),
    ("pomson", "sound/weapons/pomson_fire_01.wav"),
    ("wrench", "sound/weapons/wrench_swing.wav"),
    ("shovel", "sound/weapons/shovel_swing.wav"),
    ("knife", "sound/weapons/knife_swing.wav"),
    ("machete", "sound/weapons/machete_swing.wav"),
    ("sword", "sound/weapons/demo_sword_swing1.wav"),
    ("boxing", "sound/weapons/boxing_gloves_swing1.wav"),
    ("bat", "sound/weapons/cbar_miss1.wav"),
];
/// Any other melee weapon.
const USUAL_SWING_SOUND: &str = "sound/weapons/cbar_miss1.wav";
/// Sizes of the glowing projectiles TF2 draws as particles, in world units.
const ENERGY_RING_RADIUS: f32 = 11.0;
const ENERGY_BALL_RADIUS: f32 = 9.0;
const ORB_RADIUS: f32 = 16.0;
const FIREBALL_RADIUS: f32 = 18.0;
const FIRE_COLOR: [f32; 3] = [1.0, 0.5, 0.12];
/// Segments in a glow's circle.
const GLOW_STEPS: usize = 18;
/// Faster than anything TF2 fires, in units a second. A projectile that seems to
/// move faster is really a new one that took over an old entity number.
const MAX_PROJECTILE_SPEED: f32 = 4000.0;
/// Trace width as a share of the distance to the camera, which keeps lines
/// about the same thickness on screen near and far.
const TRACE_WIDTH: f32 = 0.0025;
const TRACE_MIN_WIDTH: f32 = 0.15;
/// Projectile trails are drawn wider than bullet lines, like the smoke they leave.
const TRAIL_WIDEN: f32 = 2.2;
/// How long a body stays where someone died.
const BODY_SECONDS: f64 = 6.0;
/// Map objects closer than this to one the demo shows are the same object.
const SAME_PROP_DISTANCE: f32 = 96.0;
const NEAR_PLANE: f32 = 4.0;
const FAR_PLANE: f32 = 40000.0;
const MAX_STEP_PER_FRAME: f32 = 300.0;
const FREE_CAMERA_SPEED: f32 = 900.0;
const LOOK_SENSITIVITY: f32 = 0.18;
const KILL_FEED_SECONDS: f64 = 7.0;
const KILL_FEED_LINES: usize = 6;
const CHAT_SECONDS: f64 = 12.0;
const CHAT_LINES: usize = 6;
const TIMELINE_HEIGHT: f32 = 46.0;
const SEEK_LANE: f32 = 12.0;
/// The bar along the bottom of the timeline showing when the watched player is in the demo.
const PRESENCE_LANE: f32 = 4.0;
const PRESENCE_ALIVE: Color32 = Color32::from_rgb(96, 190, 120);
const PRESENCE_DEAD: Color32 = Color32::from_rgb(92, 98, 106);
/// How close the pointer has to be to a kill on the timeline to name it, in points.
const KILL_HOVER_REACH: f32 = 4.0;

const RED: Color32 = Color32::from_rgb(232, 86, 79);
const BLUE: Color32 = Color32::from_rgb(88, 162, 240);
const AMBER: Color32 = Color32::from_rgb(240, 190, 80);
/// Check names in the review bar.
const REVIEW_CHECK_COLOR: Color32 = Color32::from_rgb(235, 205, 140);
const MUTED: Color32 = Color32::from_rgb(160, 168, 176);
const PING_GOOD: Color32 = Color32::from_rgb(130, 200, 120);
const PING_HIGH: Color32 = Color32::from_rgb(235, 105, 90);
/// Width of the ping column in the player list.
const PING_COLUMN: f32 = 34.0;
/// Width of the health (or uber, or "dead") column in the player list.
const HP_COLUMN: f32 = 38.0;
const PING_TOOLTIP: &str = "Ping as the server measured it, the same number TF2's scoreboard shows. It updates about once a second. TF2 subtracts about 30 ms of tick and command time, so the full round trip is a little higher.";

/// The ping column's text and colour for one player.
fn ping_label(ping: u16, bot: bool) -> (String, Color32) {
    if bot {
        return ("BOT".to_string(), MUTED);
    }
    match ping {
        0 => ("-".to_string(), MUTED),
        1..=80 => (ping.to_string(), PING_GOOD),
        81..=150 => (ping.to_string(), AMBER),
        _ => (ping.to_string(), PING_HIGH),
    }
}

fn team_color(team: u8) -> Color32 {
    match team {
        TEAM_RED => RED,
        TEAM_BLUE => BLUE,
        _ => MUTED,
    }
}

fn team_tint(team: u8) -> [f32; 3] {
    match team {
        TEAM_RED => [0.75, 0.16, 0.13],
        TEAM_BLUE => [0.16, 0.34, 0.7],
        _ => [0.6, 0.6, 0.6],
    }
}

fn class_name(class: u8) -> &'static str {
    match class {
        1 => "Scout",
        2 => "Sniper",
        3 => "Soldier",
        4 => "Demoman",
        5 => "Medic",
        6 => "Heavy",
        7 => "Pyro",
        8 => "Spy",
        9 => "Engineer",
        _ => "",
    }
}

fn format_clock(seconds: f64) -> String {
    let whole = seconds.max(0.0) as u64;
    format!("{}:{:02}", whole / 60, whole % 60)
}

fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
    let delta = (to - from + 540.0).rem_euclid(360.0) - 180.0;
    from + delta * t
}

fn normalize_pitch(pitch: f32) -> f32 {
    if pitch > 180.0 {
        pitch - 360.0
    } else {
        pitch
    }
}

/// TF2's default interpolation delay (cl_interp 0.1), assumed for backtrack ghosts.
const DEFAULT_INTERP_MS: f32 = 100.0;
/// TF2's scoreboard ping leaves out about this much tick and command time.
const SCOREBOARD_PING_OFFSET_MS: f32 = 30.0;
/// The aim graph shows this many seconds either side of the current moment.
const AIM_GRAPH_SECONDS: f64 = 3.0;
const AIM_GRAPH_HEIGHT: f32 = 58.0;
/// The aim graph's scale never goes below this many degrees per tick.
const AIM_GRAPH_MIN_SCALE: f32 = 6.0;

/// How fast a view turned between its updates: (tick, degrees per tick).
fn aim_speeds(ticks: &[u32], angles: &[(f32, f32)]) -> Vec<(u32, f32)> {
    let mut speeds = Vec::new();
    let mut last: Option<(u32, (f32, f32))> = None;
    for (&tick, &angle) in ticks.iter().zip(angles) {
        match last {
            Some((last_tick, last_angle)) if last_angle != angle => {
                let gap = tick.saturating_sub(last_tick).max(1) as f32;
                let (a, b) = (forward_from(last_angle.0, last_angle.1), forward_from(angle.0, angle.1));
                let degrees = a.angle_between(b).to_degrees();
                speeds.push((tick, degrees / gap));
                last = Some((tick, angle));
            }
            Some(_) => {}
            None => last = Some((tick, angle)),
        }
    }
    speeds
}

/// How long damage numbers float over a player.
const DAMAGE_SECONDS: f64 = 1.2;
/// How long a bullet impact mark stays.
const IMPACT_SECONDS: f64 = 1.5;
/// How far aim lines reach when they don't hit the map first.
const AIM_LINE_RANGE: f32 = 4096.0;
/// Hitbox edges are drawn thinner than bullet traces.
const HITBOX_WIDEN: f32 = 0.55;

/// A player's hitboxes for this frame, placed in the world.
struct PlayerBoxes {
    slot: usize,
    team: u8,
    /// Each hitbox's bone matrix, corners in bone space, and whether it's the head.
    boxes: Vec<(Mat4, Vec3, Vec3, bool)>,
}

/// Everything the 3D scene shows this frame.
struct Scene {
    shapes: Vec<ShapeInstance>,
    actors: Vec<Actor>,
    boxes: Vec<PlayerBoxes>,
}

/// A bone by name, ignoring case: models spell the hands both bip_hand_r and bip_hand_R.
fn bone_named(skeleton: &super::skeleton::Skeleton, name: &str) -> Option<usize> {
    skeleton.bones.iter().position(|bone| bone.name.eq_ignore_ascii_case(name))
}

/// The twelve edges of a box in a bone's space, as world-space line segments.
fn box_edges(bone: &Mat4, min: Vec3, max: Vec3) -> [(Vec3, Vec3); 12] {
    let corner = |x: bool, y: bool, z: bool| {
        bone.transform_point3(Vec3::new(
            if x { max.x } else { min.x },
            if y { max.y } else { min.y },
            if z { max.z } else { min.z },
        ))
    };
    let c = [
        corner(false, false, false),
        corner(true, false, false),
        corner(true, true, false),
        corner(false, true, false),
        corner(false, false, true),
        corner(true, false, true),
        corner(true, true, true),
        corner(false, true, true),
    ];
    [
        (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]),
        (c[4], c[5]), (c[5], c[6]), (c[6], c[7]), (c[7], c[4]),
        (c[0], c[4]), (c[1], c[5]), (c[2], c[6]), (c[3], c[7]),
    ]
}

fn team_rgb(team: u8) -> [f32; 3] {
    match team {
        TEAM_RED => [1.0, 0.36, 0.3],
        TEAM_BLUE => [0.4, 0.66, 1.0],
        _ => [0.85, 0.85, 0.85],
    }
}

fn forward_from(pitch: f32, yaw: f32) -> Vec3 {
    let (pitch, yaw) = (pitch.to_radians(), yaw.to_radians());
    Vec3::new(pitch.cos() * yaw.cos(), pitch.cos() * yaw.sin(), -pitch.sin())
}

/// Where the third-person camera goes for a player whose eyes are at `eye`, looking
/// along `pitch` and `yaw`: behind their right shoulder, pulled in front of any wall.
fn chase_eye(eye: Vec3, pitch: f32, yaw: f32, distance: f32, first_hit: impl Fn(Vec3, Vec3) -> Option<Vec3>) -> Vec3 {
    let forward = forward_from(pitch, yaw);
    let right = forward.cross(Vec3::Z).normalize_or_zero();
    let wanted = eye - forward * distance + right * CHASE_RIGHT + Vec3::Z * CHASE_UP;
    clear_of_walls(eye, wanted, first_hit)
}

/// Where the camera goes when swung around a player: `distance` back from `target`
/// along the camera's own look, pulled in front of any wall.
fn orbit_eye(target: Vec3, pitch: f32, yaw: f32, distance: f32, first_hit: impl Fn(Vec3, Vec3) -> Option<Vec3>) -> Vec3 {
    clear_of_walls(target, target - forward_from(pitch, yaw) * distance, first_hit)
}

/// `wanted`, or the point just in front of the first wall between it and `from`.
fn clear_of_walls(from: Vec3, wanted: Vec3, first_hit: impl Fn(Vec3, Vec3) -> Option<Vec3>) -> Vec3 {
    match first_hit(from, wanted) {
        Some(wall) => {
            let away = wall - from;
            let length = away.length();
            if length <= CHASE_WALL_GAP {
                from
            } else {
                from + away * ((length - CHASE_WALL_GAP) / length)
            }
        }
        None => wanted,
    }
}

/// TF2's sound files, decoded the first time each one plays.
struct SoundBank {
    files: tf_asset_loader::Loader,
    clips: HashMap<String, Option<Arc<super::audio::Clip>>>,
}

impl SoundBank {
    fn clip(&mut self, path: &str) -> Option<Arc<super::audio::Clip>> {
        if let Some(known) = self.clips.get(path) {
            return known.clone();
        }
        let clip = self
            .files
            .load(path)
            .ok()
            .flatten()
            .and_then(|bytes| super::audio::decode(path, &bytes))
            .map(Arc::new);
        self.clips.insert(path.to_string(), clip.clone());
        clip
    }
}

/// One stretch of someone talking on voice chat.
struct VoiceLine {
    start: f64,
    end: f64,
    slot: Option<usize>,
    clip: Arc<super::audio::Clip>,
}

/// The demo's voice chat as clips on the timeline, matched to the players talking.
fn voice_lines(recording: &Recording, voice: std::collections::BTreeMap<u64, Vec<crate::util::voice::VoiceClip>>) -> Vec<VoiceLine> {
    let interval = f64::from(recording.tick_interval.max(1e-4));
    let mut lines: Vec<VoiceLine> = voice
        .into_iter()
        .flat_map(|(steam_id, clips)| {
            clips.into_iter().map(move |clip| {
                let start = clip.start_secs / interval;
                let end = start + clip.duration_secs() / interval;
                let samples = clip.samples.iter().map(|&sample| f32::from(sample) / 32768.0).collect();
                (steam_id, start, end, super::audio::Clip { samples, rate: clip.sample_rate, loop_start: None })
            })
        })
        .map(|(steam_id, start, end, clip)| VoiceLine {
            start,
            end,
            slot: (0..recording.slot_count).find(|&slot| recording.steam_id_at(slot, start) == steam_id),
            clip: Arc::new(clip),
        })
        .collect();
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    lines
}

/// Whether a first-person animation is an attack: a shot or a swing.
fn is_attack_activity(activity: &str) -> bool {
    let activity = activity.to_ascii_uppercase();
    ["PRIMARYATTACK", "SWINGHARD", "HITCENTER", "HITLEFT", "HITRIGHT", "SWING", "THROW"].iter().any(|word| activity.contains(word))
}

/// Whether a sound the server sent on the weapon channel is a shot or a swing,
/// rather than a reload or a weapon being drawn.
fn is_fire_sound(name: &str) -> bool {
    let firing = ["shoot", "fire", "swing", "single", "shot", "miss", "throw"].iter().any(|word| name.contains(word));
    let other = ["reload", "draw", "deploy", "empty", "crit", "spin", "wind", "loop", "_off", "_on."].iter().any(|word| name.contains(word));
    firing && !other
}

/// The sound each weapon made when the server played it for its shooter, by
/// weapon model: the most common shot or swing heard from players holding it.
fn learned_fire_sounds(recording: &Recording) -> HashMap<u16, String> {
    let mut counts: HashMap<(u16, u16), usize> = HashMap::new();
    for sound in &recording.sounds {
        let (Some(slot), true) = (sound.slot, sound.channel == CHAN_WEAPON) else { continue };
        let Some(name) = recording.sound_names.get(usize::from(sound.sound)) else { continue };
        if !is_fire_sound(name) {
            continue;
        }
        let weapon = recording.player(recording.frame_at(f64::from(sound.tick)), slot).weapon_model;
        *counts.entry((weapon, sound.sound)).or_default() += 1;
    }
    let mut best: HashMap<u16, (usize, u16)> = HashMap::new();
    for ((weapon, sound), count) in counts {
        let entry = best.entry(weapon).or_insert((0, sound));
        if count > entry.0 || (count == entry.0 && sound < entry.1) {
            *entry = (count, sound);
        }
    }
    best.into_iter()
        .filter_map(|(weapon, (_, sound))| Some((weapon, recording.sound_names.get(usize::from(sound))?.clone())))
        .collect()
}

/// TF2's usual sound for a weapon, from its model's name.
fn usual_fire_sound(model: &str, melee: bool) -> Option<&'static str> {
    let name = model.rsplit('/').next().unwrap_or(model);
    USUAL_FIRE_SOUNDS
        .iter()
        .find(|(word, _)| name.contains(word))
        .map(|(_, sound)| *sound)
        .or(melee.then_some(USUAL_SWING_SOUND))
}

/// The recorder's own shots and swings. Their game played those sounds itself, so
/// the demo doesn't have them: they're found from the first-person animation, and
/// sound like what the server played when someone else fired the same weapon, or
/// like TF2's usual sound for it.
fn own_shot_sounds(recording: &Recording, characters: &Characters) -> Vec<(u32, String)> {
    let Some(recorder) = recording.recorder_slot else { return Vec::new() };
    let learned = learned_fire_sounds(recording);
    let mut shots = Vec::new();
    for frame in 1..recording.views.len().min(recording.ticks.len()) {
        let (before, now) = (&recording.views[frame - 1], &recording.views[frame]);
        if now.sequence_start == before.sequence_start || now.arms == 0 {
            continue;
        }
        let Some(mesh) = characters.by_model.get(usize::from(now.arms)).copied().flatten() else { continue };
        let Some(activity) = characters.animations[mesh].as_ref().and_then(|animations| animations.activity_of(usize::from(now.sequence))) else {
            continue;
        };
        if !is_attack_activity(&activity) {
            continue;
        }
        let weapon = recording.player(frame, recorder).weapon_model;
        let model = recording.models.get(usize::from(weapon)).map_or("", String::as_str);
        let melee = activity.to_ascii_uppercase().contains("MELEE") || activity.to_ascii_uppercase().contains("SWING");
        let sound = learned.get(&weapon).cloned().or_else(|| usual_fire_sound(model, melee).map(str::to_string));
        if let Some(sound) = sound {
            shots.push((now.sequence_start, sound));
        }
    }
    shots.sort_by_key(|(tick, _)| *tick);
    shots
}

/// Whether the demo has a player's data at a moment. A demo recorded by a player
/// only gets other players while they're near enough to matter to the recorder's
/// game (its PVS); the rest of the time they're frozen or missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Presence {
    Missing,
    Alive,
    Dead,
}

/// A stretch of the demo with the same presence, from one tick to another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresenceRun {
    from: u32,
    to: u32,
    presence: Presence,
}

/// Each player's stretches in the demo, alive or dead. Stretches without them are left out.
fn presence_runs(recording: &Recording) -> Vec<Vec<PresenceRun>> {
    (0..recording.slot_count)
        .map(|slot| {
            let mut runs: Vec<PresenceRun> = Vec::new();
            for frame in 0..recording.ticks.len() {
                let tick = recording.ticks[frame];
                let player = recording.player(frame, slot);
                let presence = if !player.is_on_a_team() || player.is_out_of_view() {
                    Presence::Missing
                } else if player.is_alive() {
                    Presence::Alive
                } else {
                    Presence::Dead
                };
                match runs.last_mut() {
                    Some(run) if run.presence == presence => run.to = tick,
                    Some(run) => {
                        run.to = tick;
                        runs.push(PresenceRun { from: tick, to: tick, presence });
                    }
                    None => runs.push(PresenceRun { from: tick, to: tick, presence }),
                }
            }
            runs.retain(|run| run.presence != Presence::Missing);
            runs
        })
        .collect()
}

/// A kill in plain words, for the timeline's tooltip.
fn describe_kill(kill: &Kill) -> String {
    let weapon = if kill.weapon.is_empty() { String::new() } else { format!(" ({})", kill.weapon) };
    match kill.attacker {
        Some(attacker) if Some(attacker) != kill.victim => format!("{} killed {}{weapon}", kill.attacker_name, kill.victim_name),
        _ => format!("{} died{weapon}", kill.victim_name),
    }
}

/// Where one rocket, pipe or other projectile was on each frame it existed.
#[derive(Clone, Debug, Default)]
struct Trail {
    kind: ObjectKind,
    team: u8,
    crit: bool,
    ticks: Vec<u32>,
    points: Vec<Vec3>,
}

impl Trail {
    fn first_tick(&self) -> u32 {
        self.ticks.first().copied().unwrap_or(0)
    }

    fn last_tick(&self) -> u32 {
        self.ticks.last().copied().unwrap_or(0)
    }
}

/// Follows each projectile from frame to frame. An id that disappears and comes
/// back later is a new projectile, since the game reuses entity numbers.
fn build_trails(recording: &Recording) -> Vec<Trail> {
    let interval = recording.tick_interval;
    let mut trails: Vec<Trail> = Vec::new();
    let mut open: HashMap<u16, (usize, usize, ObjectKind)> = HashMap::new();
    for frame in 0..recording.ticks.len() {
        let tick = recording.ticks[frame];
        for object in recording.objects_at(frame) {
            if object.kind.is_building() || object.kind == ObjectKind::Cart {
                continue;
            }
            let position = Vec3::from(object.position);
            let continuing = open
                .get(&object.id)
                .filter(|&&(index, last_frame, kind)| {
                    let trail = &trails[index];
                    let (Some(&last_tick), Some(&last_point)) = (trail.ticks.last(), trail.points.last()) else {
                        return false;
                    };
                    let seconds = tick.saturating_sub(last_tick) as f32 * interval;
                    last_frame + 1 == frame
                        && kind == object.kind
                        && trail.team == object.team
                        && last_point.distance(position) <= MAX_PROJECTILE_SPEED * seconds + 32.0
                })
                .map(|&(index, last_frame, _)| (index, last_frame));
            let index = match continuing {
                Some((index, _)) => index,
                None => {
                    trails.push(Trail {
                        kind: object.kind,
                        team: object.team,
                        crit: object.crit,
                        ..Trail::default()
                    });
                    trails.len() - 1
                }
            };
            let trail = &mut trails[index];
            trail.crit |= object.crit;
            trail.ticks.push(tick);
            trail.points.push(position);
            open.insert(object.id, (index, frame, object.kind));
        }
    }
    // A single sighting has no path to draw.
    trails.retain(|trail| trail.points.len() > 1);
    trails.sort_by_key(Trail::first_tick);
    trails
}

/// Which players get an outline that shows through walls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlowMode {
    Off,
    #[default]
    Everyone,
    /// The other team from whoever is being watched, the way a wallhack shows them.
    Enemies,
    Teammates,
}

impl GlowMode {
    const ALL: [GlowMode; 4] = [GlowMode::Off, GlowMode::Everyone, GlowMode::Enemies, GlowMode::Teammates];

    fn label(self) -> &'static str {
        match self {
            GlowMode::Off => "Off",
            GlowMode::Everyone => "Everyone",
            GlowMode::Enemies => "Enemies",
            GlowMode::Teammates => "Teammates",
        }
    }

    fn next(self) -> GlowMode {
        let at = Self::ALL.iter().position(|&mode| mode == self).unwrap_or(0);
        Self::ALL[(at + 1) % Self::ALL.len()]
    }
}

/// Outline colours, like TF2's team glows.
const GLOW_RED: [f32; 3] = [0.95, 0.32, 0.28];
const GLOW_BLUE: [f32; 3] = [0.36, 0.62, 1.0];

/// The view settings people pick in the top bar. They carry over to the next
/// demo opened while the tool is running.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewOptions {
    fov: f32,
    names: bool,
    bullets: bool,
    projectiles: bool,
    glow: GlowMode,
    aim_lines: bool,
    hitboxes: bool,
    ghosts: bool,
    ghost_interp_ms: f32,
    damage: bool,
    impacts: bool,
    aim_graph: bool,
    fov_circle: bool,
    volume: f32,
    muted: bool,
    hit_sounds: bool,
    voice_chat: bool,
    third_person: bool,
    chase_distance: f32,
}

impl Default for ViewOptions {
    fn default() -> Self {
        Self {
            fov: DEFAULT_FOV,
            names: true,
            bullets: true,
            projectiles: true,
            glow: GlowMode::default(),
            aim_lines: false,
            hitboxes: false,
            ghosts: false,
            ghost_interp_ms: DEFAULT_INTERP_MS,
            damage: true,
            impacts: true,
            aim_graph: true,
            fov_circle: true,
            volume: DEFAULT_VOLUME,
            muted: false,
            hit_sounds: true,
            voice_chat: true,
            third_person: false,
            chase_distance: CHASE_BEHIND,
        }
    }
}

fn view_options_id() -> egui::Id {
    egui::Id::new("demo_player_view_options")
}

/// How a player came to be off the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Airborne {
    /// Jumped, this many seconds ago.
    Jump(f32),
    /// Thrown upwards, like a rocket jump.
    Flung,
    /// Walked off an edge.
    Fell,
}

/// A moment worth marking on the timeline, such as a detection.
#[derive(Clone, Debug, Default)]
pub struct Marker {
    pub tick: u32,
    pub label: String,
    pub steam_id: u64,
    /// The check that made the detection, such as "fidoo/ml_silent_aim".
    pub check: String,
    /// What the check measured, as it reported it.
    pub data: serde_json::Value,
}

/// A clip being recorded: one screenshot of the view per frame.
struct ClipRecording {
    next: f64,
    end: f64,
    step: f64,
    waiting: bool,
    frames: usize,
    total: usize,
    writer: super::clip::ClipWriter,
}

/// Clip lengths offered, in seconds.
const CLIP_LENGTHS: [f64; 3] = [5.0, 10.0, 20.0];

/// How far back a body's speed at death is measured.
const RAGDOLL_SPEED_WINDOW_SECONDS: f64 = 0.1;
/// The fastest a body can be moving when it starts to fall.
const RAGDOLL_MAX_SPEED: f32 = 1200.0;

/// How hard a kill shoves the body: sideways and upward speed. Explosions throw it,
/// bullets and melee knock it back a little.
fn blow_strength(weapon: &str) -> (f32, f32) {
    const EXPLOSIVE: [&str; 10] = ["rocket", "pipe", "sticky", "quake", "airstrike", "cannon", "bomb", "mangler", "blackbox", "liberty"];
    let weapon = weapon.to_ascii_lowercase();
    if EXPLOSIVE.iter().any(|word| weapon.contains(word)) {
        (300.0, 260.0)
    } else {
        (140.0, 60.0)
    }
}

/// Size of the squares that floors players stood on are kept in, in units.
const FLOOR_CELL: f32 = 32.0;
/// How far around a death, and how long before and after it, standing spots count.
const FLOOR_RADIUS: f32 = 400.0;
const FLOOR_SECONDS_BEFORE: f64 = 10.0;
const FLOOR_SECONDS_AFTER: f64 = 6.0;

/// Where a line going down first crosses one of these floors (square cells
/// centred on each point, at its height).
fn floor_hit(floors: &[Vec3], from: Vec3, to: Vec3) -> Option<Vec3> {
    if to.z >= from.z {
        return None;
    }
    let half = FLOOR_CELL * 0.5;
    let mut best: Option<(f32, Vec3)> = None;
    for floor in floors {
        if from.z < floor.z || to.z >= floor.z {
            continue;
        }
        let t = (from.z - floor.z) / (from.z - to.z);
        let at = from.lerp(to, t);
        let inside = (at.x - floor.x).abs() <= half && (at.y - floor.y).abs() <= half;
        if inside && best.map_or(true, |(nearest, _)| t < nearest) {
            best = Some((t, at));
        }
    }
    best.map(|(_, at)| at)
}

/// How long after taking out a weapon its draw animation can still be playing.
const DRAW_LOOKBACK_SECONDS: f64 = 1.5;
/// How long before a detection the view is taken as where the player was looking
/// before the flick, for checks that don't report the flick's size.
const FLICK_LOOKBACK_SECONDS: f64 = 0.06;
/// How much of the view's past is replayed to work out first-person weapon sway.
const SWAY_HISTORY_SECONDS: f64 = 1.0;
/// TF2's limit on how far behind a turn the first-person weapon can fall.
const MAX_SWAY: f32 = 1.5;
/// How long a sentry keeps playing its firing animation after a shot.
const SENTRY_FIRING_SECONDS: f64 = 0.3;
/// How long a muzzle flash shows.
const MUZZLE_FLASH_SECONDS: f64 = 0.06;

/// A sentry's shot, worked out from its bullets running down while it has a target.
struct SentryShot {
    tick: u32,
    id: u16,
    team: u8,
    from: Vec3,
    to: Vec3,
}

/// Every sentry shot in the demo, in time order.
fn sentry_shots(recording: &Recording, lighting: &LightProbe) -> Vec<SentryShot> {
    let mut shells: HashMap<u16, i16> = HashMap::new();
    let mut shots = Vec::new();
    for frame in 0..recording.ticks.len() {
        for object in recording.objects_at(frame) {
            let now = object.animation.shells;
            if object.kind != ObjectKind::Sentry || now < 0 {
                continue;
            }
            let Some(before) = shells.insert(object.id, now) else { continue };
            // Refilled, or a jump from being out of view for a while.
            if now >= before || before - now > 4 {
                continue;
            }
            let Some(target) = object.animation.target else { continue };
            let victim = recording.player(frame, usize::from(target));
            if !victim.is_alive() || victim.is_out_of_view() {
                continue;
            }
            let height = if object.mini {
                24.0
            } else {
                match object.level {
                    0 | 1 => 32.0,
                    2 => 40.0,
                    _ => 44.0,
                }
            };
            let from = Vec3::from(object.position) + Vec3::Z * height;
            let aim = Vec3::from(victim.position) + Vec3::Z * if victim.is_ducking() { 28.0 } else { 42.0 };
            let direction = (aim - from).normalize_or_zero();
            // Past the target, in case it was hit at the edge, and stopped by the map.
            let far = aim + direction * 32.0;
            let to = lighting.first_hit(from + direction * 16.0, far).unwrap_or(aim);
            shots.push(SentryShot { tick: recording.ticks[frame], id: object.id, team: object.team, from, to });
        }
    }
    shots
}

/// How close a hit player has to be to a bullet's line for it to stop there.
const HIT_REACH: f32 = 48.0;
/// A muzzle further than this from where the demo says a shot came from belongs to
/// something else, like a weapon switched to since.
const MUZZLE_REACH: f32 = 96.0;

/// Where each shot ended: at the player it hurt, if it hurt someone on its line,
/// otherwise where it hit the map.
fn shot_ends(recording: &Recording, first_hit: impl Fn(Vec3, Vec3) -> Option<Vec3>) -> Vec<Vec3> {
    recording
        .shots
        .iter()
        .map(|shot| {
            let from = Vec3::from(shot.origin);
            let aim = forward_from(shot.pitch, shot.yaw);
            let wall = first_hit(from, from + aim * BULLET_RANGE).unwrap_or(from + aim * BULLET_RANGE);
            let reach = from.distance(wall);
            // Damage from this shooter in the same tick or the next.
            let first = recording.hurts.partition_point(|hurt| hurt.tick < shot.tick);
            let nearest = recording.hurts[first..]
                .iter()
                .take_while(|hurt| hurt.tick <= shot.tick + 1)
                .filter(|hurt| hurt.attacker == Some(shot.slot))
                .filter_map(|hurt| {
                    let victim = recording.player(recording.frame_at(f64::from(shot.tick)), hurt.victim?);
                    let center = Vec3::from(victim.position) + Vec3::Z * victim.eye_height() * 0.6;
                    let along = (center - from).dot(aim);
                    let off = (from + aim * along).distance(center);
                    (along > 0.0 && along < reach && off < HIT_REACH).then_some(along)
                })
                .reduce(f32::min);
            nearest.map_or(wall, |along| from + aim * along)
        })
        .collect()
}

/// The weapon sets to look for first-person animations in: the weapon's own, then
/// the one for its slot (a Demoman's sword is held like his other melee weapons, not
/// like his sticky launcher), then the general ones.
fn draw_names(sets: &[WeaponPose]) -> Vec<String> {
    sets.iter().map(|set| format!("ACT_{}_VM_DRAW", set.suffix())).collect()
}

fn first_person_sets(pose: WeaponPose, slot: u8) -> Vec<WeaponPose> {
    let by_slot = match slot {
        0 => Some(WeaponPose::Primary),
        1 => Some(WeaponPose::Secondary),
        2 => Some(WeaponPose::Melee),
        _ => None,
    };
    let mut sets = vec![pose];
    for set in by_slot.into_iter().chain(pose.chain()) {
        if !sets.contains(&set) {
            sets.push(set);
        }
    }
    sets
}

/// TF2's first-person animation names for a shot, swing or reload with one weapon
/// set, best first.
fn first_person_actions(gesture: Gesture, set: WeaponPose) -> Vec<String> {
    let s = set.suffix();
    let attack = vec![
        format!("ACT_{s}_VM_PRIMARYATTACK"),
        format!("ACT_{s}_VM_HITCENTER"),
        format!("ACT_{s}_VM_FIRE"),
        // The Heavy's fists punch left, then right on the alternate attack.
        if set == WeaponPose::Melee { "ACT_FISTS_VM_HITLEFT".to_string() } else { String::new() },
        format!("ACT_{s}_VM_SWINGHARD"),
    ];
    let mut names = match gesture {
        Gesture::Attack => attack,
        Gesture::AttackSecondary => {
            let mut names = vec![
                format!("ACT_{s}_VM_SECONDARYATTACK"),
                format!("ACT_{s}_VM_SWINGHARD"),
                if set == WeaponPose::Melee { "ACT_FISTS_VM_HITRIGHT".to_string() } else { String::new() },
                format!("ACT_{s}_VM_ALTATTACK"),
            ];
            names.extend(attack);
            names
        }
        Gesture::Reload => vec![format!("ACT_{s}_RELOAD_START"), format!("ACT_{s}_VM_RELOAD")],
        Gesture::ReloadLoop => vec![format!("ACT_{s}_VM_RELOAD")],
        Gesture::ReloadEnd => vec![format!("ACT_{s}_RELOAD_FINISH")],
    };
    names.retain(|name| !name.is_empty());
    names
}

/// What a body was doing, so a change of animation can blend instead of snapping.
#[derive(Clone, Copy)]
struct PoseMemory {
    /// What it was doing when last drawn.
    current: PoseInput,
    /// What it was doing before its last change, and the tick of that change.
    previous: Option<(PoseInput, f64)>,
    /// The tick it was last drawn at.
    seen_at: f64,
}

/// Whether two moments use different movement animations, which is what needs blending.
fn animation_changed(a: &PoseInput, b: &PoseInput) -> bool {
    a.stance != b.stance || a.weapon != b.weapon || a.deployed != b.deployed
}

/// Where review mode is in playing one detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReviewPhase {
    /// Slow motion up to the detection.
    LeadIn,
    /// Paused on it.
    Hold,
    /// Slow motion a little after it.
    Follow,
    /// Finished, waiting to go on.
    Done,
}

/// What review mode changes, to put back when it ends.
struct ReviewRestore {
    speed_index: usize,
    show_aim_lines: bool,
    show_aim_graph: bool,
    show_ghosts: bool,
}

/// Review mode: one player's detections, played one after another from their eyes.
struct Review {
    steam_id: u64,
    name: String,
    moments: Vec<super::review::Moment>,
    index: usize,
    phase: ReviewPhase,
    /// When the pause on the detection, or before the next one, ends (egui time).
    wait_until: f64,
    auto_advance: bool,
    before: ReviewRestore,
}

enum ReviewAction {
    Go(usize),
    Exit,
}

/// Something the player asks the rest of the tool to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlayerRequest {
    /// Open the TF2BD "Mark player" dialog for this player.
    Mark { steam_id: u64, name: String },
}

/// What to open and where to start.
#[derive(Clone, Debug, Default)]
pub struct OpenRequest {
    pub demo: PathBuf,
    /// Tick to start from.
    pub tick: Option<u32>,
    /// Player whose eyes to start from.
    pub steam_id: Option<u64>,
    pub markers: Vec<Marker>,
}

struct Loaded {
    recording: Recording,
    map: MapData,
    /// Voice chat recordings by speaker SteamID64.
    voice: std::collections::BTreeMap<u64, Vec<crate::util::voice::VoiceClip>>,
    /// For reading sound files as they're needed.
    sound_files: tf_asset_loader::Loader,
}

#[derive(Default)]
struct Progress {
    label: String,
    share: f32,
}

struct Loading {
    progress: Arc<Mutex<Progress>>,
    result: Receiver<Result<Loaded, String>>,
}

enum State {
    Loading(Loading),
    Failed(String),
    Ready(Box<Session>),
}

pub struct DemoPlayer {
    state: State,
    request: OpenRequest,
}

fn load(path: &PathBuf, progress: &Mutex<Progress>) -> Result<Loaded, String> {
    let report = |label: &str, share: f32| {
        if let Ok(mut progress) = progress.lock() {
            progress.label = label.to_string();
            progress.share = share;
        }
    };

    report("Reading demo", 0.0);
    let bytes = std::fs::read(path).map_err(|error| format!("Couldn't read this demo: {error}"))?;
    let recording = read_demo(&bytes, |share| report("Reading demo", share * 0.5))?;
    let voice = crate::util::voice::extract_voice_clips(&bytes, VOICE_JOIN_SECONDS)
        .map(|(clips, _)| clips)
        .unwrap_or_default();
    drop(bytes);

    report("Finding TF2", 0.5);
    let loader = tf_asset_loader::Loader::new().map_err(|_| {
        "Team Fortress 2 wasn't found. It needs to be installed through Steam, but it doesn't have to be running."
            .to_string()
    })?;

    let mut classes: Vec<u8> = recording
        .players
        .iter()
        .filter(|player| player.is_on_a_team())
        .map(|player| player.class)
        .collect();
    classes.sort_unstable();
    classes.dedup();
    let content = DemoContent {
        sky_name: recording.sky_name.clone(),
        classes,
        models: recording.models.clone(),
    };
    let map = load_map(&loader, &recording.map, &content, |share| {
        report("Loading map", 0.5 + share * 0.5);
    })?;
    Ok(Loaded { recording, map, voice, sound_files: loader })
}

impl DemoPlayer {
    pub fn open(request: OpenRequest) -> Self {
        let progress = Arc::new(Mutex::new(Progress::default()));
        let (sender, result) = mpsc::channel();
        let path = request.demo.clone();
        let shared = Arc::clone(&progress);
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| load(&path, &shared)))
                .unwrap_or_else(|_| Err("This demo or its map is damaged and couldn't be loaded.".into()));
            let _ = sender.send(outcome);
        });
        Self {
            state: State::Loading(Loading { progress, result }),
            request,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(&self.state, State::Ready(session) if session.renderer_ready())
    }

    pub fn failure(&self) -> Option<&str> {
        match &self.state {
            State::Failed(message) => Some(message),
            _ => None,
        }
    }

    /// Requests from the player for the rest of the tool, such as marking someone.
    pub fn take_requests(&mut self) -> Vec<PlayerRequest> {
        match &mut self.state {
            State::Ready(session) => std::mem::take(&mut session.requests),
            _ => Vec::new(),
        }
    }

    /// TF2BD tags to show in the player list, and whether marking is possible.
    pub fn set_marks(&mut self, marks: HashMap<u64, (String, Color32)>, can_mark: bool) {
        if let State::Ready(session) = &mut self.state {
            session.marks = marks;
            session.can_mark = can_mark;
        }
    }

    pub fn session_mut(&mut self) -> Option<&mut Session> {
        match &mut self.state {
            State::Ready(session) => Some(session),
            _ => None,
        }
    }

    /// Shows the player. Returns false once the user has closed it.
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        if let State::Loading(loading) = &self.state {
            match loading.result.try_recv() {
                Ok(Ok(loaded)) => {
                    self.state = State::Ready(Box::new(Session::new(loaded, &self.request)));
                }
                Ok(Err(message)) => self.state = State::Failed(message),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.state = State::Failed("Loading stopped unexpectedly.".into());
                }
            }
        }

        match &mut self.state {
            State::Loading(loading) => {
                let (label, share) = loading
                    .progress
                    .lock()
                    .map_or((String::new(), 0.0), |progress| (progress.label.clone(), progress.share));
                let mut keep_open = true;
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() * 0.4);
                        ui.heading(label);
                        ui.add_space(8.0);
                        ui.add(egui::ProgressBar::new(share).desired_width(320.0));
                        ui.add_space(8.0);
                        if ui.button("Cancel").clicked() {
                            keep_open = false;
                        }
                    });
                });
                ctx.request_repaint();
                keep_open
            }
            State::Failed(message) => {
                let mut keep_open = true;
                let message = message.clone();
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(ui.available_height() * 0.4);
                        ui.heading("Couldn't open this demo");
                        ui.add_space(8.0);
                        ui.label(message);
                        ui.add_space(12.0);
                        if ui.button("Back").clicked() {
                            keep_open = false;
                        }
                    });
                });
                keep_open
            }
            State::Ready(session) => session.show(ctx),
        }
    }

    /// Frees the GPU copy of the map. Call with the app's GL context.
    pub fn destroy(&mut self, gl: &glow::Context) {
        if let State::Ready(session) = &mut self.state {
            if let Ok(mut renderer) = session.renderer.lock() {
                if let Some(renderer) = renderer.as_mut() {
                    renderer.destroy(gl);
                }
                *renderer = None;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraMode {
    /// Looking through a player's eyes.
    Eyes(usize),
    Free,
}

#[derive(Clone, Copy, Debug)]
pub struct FreeCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Clone, Copy, Debug)]
struct Camera {
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
}

pub struct Session {
    demo: PathBuf,
    review: Option<Review>,
    /// Each body's last animation, by slot, for blending between them.
    pose_memory: RefCell<HashMap<usize, PoseMemory>>,
    /// Dead bodies' falls, by kill, worked out the first time each is shown.
    ragdolls: RefCell<HashMap<usize, Option<Arc<Ragdoll>>>>,
    sentry_shots: Vec<SentryShot>,
    /// Where each player's gun muzzle is this frame, by slot, for bullet lines.
    muzzles: RefCell<HashMap<usize, Vec3>>,
    requests: Vec<PlayerRequest>,
    /// TF2BD tags to show next to names: (label, colour).
    marks: HashMap<u64, (String, Color32)>,
    can_mark: bool,
    /// The last report written, or why it failed, and when.
    report_status: Option<(String, bool, f64)>,
    /// Where the 3D view was drawn last frame, for cropping clips.
    view_rect: Rect,
    clip: Option<ClipRecording>,
    /// A finished clip still being encoded.
    encoding: Option<super::clip::ClipWriter>,
    clip_seconds: f64,
    clip_status: Option<(String, bool)>,
    recording: Recording,
    lighting: Arc<LightProbe>,
    sky_camera: Option<SkyCamera>,
    characters: Characters,
    /// Objects from the map file that the demo never showed, with their lighting.
    map_props: Vec<(MapProp, ModelLighting)>,
    scope_image: Option<super::assets::Image>,
    scope_texture: Option<egui::TextureHandle>,
    crosshair_image: Option<super::assets::Image>,
    crosshair_texture: Option<egui::TextureHandle>,
    pending_map: Arc<Mutex<Option<MapData>>>,
    renderer: Arc<Mutex<Option<Renderer>>>,
    render_error: Arc<Mutex<Option<String>>>,
    markers: Vec<Marker>,
    tick: f64,
    playing: bool,
    resume_after_scrub: bool,
    speed_index: usize,
    fov: f32,
    show_names: bool,
    /// Lines for where hitscan bullets went.
    pub show_bullets: bool,
    /// Trails behind rockets, pipes and other projectiles.
    pub show_projectiles: bool,
    /// Outlines through walls.
    pub glow: GlowMode,
    /// A line from every player's eyes to where they look.
    pub show_aim_lines: bool,
    /// Every player's hitboxes as wire boxes.
    pub show_hitboxes: bool,
    /// Where lag compensation puts enemies for the watched player's shots.
    pub show_ghosts: bool,
    /// The interpolation delay assumed for backtrack ghosts, in milliseconds.
    pub ghost_interp_ms: f32,
    /// Damage numbers over players who get hurt.
    pub show_damage: bool,
    /// Marks where bullets hit.
    pub show_impacts: bool,
    /// The watched player's aim speed under the timeline.
    pub show_aim_graph: bool,
    /// A circle around the crosshair as big as the watched player's estimated aimbot FOV.
    pub show_fov_circle: bool,
    /// Estimated aimbot FOV by player: degrees, and how many flicks it's based on.
    aimbot_fovs: HashMap<u64, (f32, usize)>,
    /// When the timeline shows only the watched player's kills and deaths.
    pub only_their_kills: bool,
    /// Watching a player from behind instead of through their eyes.
    pub third_person: bool,
    /// The third-person camera's own angles (yaw, pitch) once swung around the
    /// player; `None` keeps it behind them, looking where they look.
    pub orbit: Option<(f32, f32)>,
    /// How far the third-person camera is from the player.
    pub chase_distance: f32,
    /// The sound output; `None` on a computer without a sound device.
    audio: Option<super::audio::Audio>,
    sound_bank: SoundBank,
    /// How far sounds have been played, in ticks.
    sound_tick: f64,
    /// Whether sound was playing last frame, so voice chat can pick up partway after a pause.
    sound_was_on: bool,
    /// 0 to 1.
    pub volume: f32,
    pub muted: bool,
    /// TF2's hit and kill sounds for the watched player.
    pub hit_sounds: bool,
    pub voice_chat: bool,
    voice_lines: Vec<VoiceLine>,
    /// The recorder's own shots and swings, which their game played and the demo doesn't have.
    own_shots: Vec<(u32, String)>,
    /// Each player's stretches in the demo, for the bar under the timeline.
    presence: Vec<Vec<PresenceRun>>,
    /// Whether the settings from the last demo have been brought over yet.
    options_restored: bool,
    /// Where each of `Recording::shots` hit the map.
    shot_ends: Vec<Vec3>,
    trails: Vec<Trail>,
    weapon_roles: super::items::WeaponRoles,
    pub camera_mode: CameraMode,
    pub free_camera: FreeCamera,
}

impl Session {
    fn new(loaded: Loaded, request: &OpenRequest) -> Self {
        let Loaded { recording, mut map, voice, sound_files } = loaded;
        let demo = request.demo.clone();
        let characters = std::mem::take(&mut map.characters);
        let voice_lines = voice_lines(&recording, voice);
        let own_shots = own_shot_sounds(&recording, &characters);
        let weapon_roles = std::mem::take(&mut map.weapon_roles);
        let scope_image = map.scope.take();
        let crosshair_image = map.crosshair.take();
        // Skip map file objects the demo shows itself, so nothing appears twice.
        let map_props = std::mem::take(&mut map.map_props)
            .into_iter()
            .filter(|prop| {
                !recording.props.iter().any(|track| {
                    usize::from(track.model) == prop.model
                        && (prop.attached
                            || track.keys.first().is_some_and(|key| {
                                Vec3::from(key.position).distance(prop.origin) < SAME_PROP_DISTANCE
                            }))
                })
            })
            .map(|prop| {
                let lighting = map.lighting.at(prop.origin + Vec3::Z * 16.0);
                (prop, lighting)
            })
            .collect();
        let first = f64::from(recording.first_tick());
        let last = f64::from(recording.last_tick());
        let lead = DETECTION_LEAD_SECONDS / f64::from(recording.tick_interval);
        let tick = request
            .tick
            .map_or(first, |tick| (f64::from(tick) - lead).clamp(first, last));

        let wanted = request.steam_id.and_then(|id| recording.slot_of_steam_id(id, tick));
        let slot = wanted.or(recording.recorder_slot).or_else(|| first_player(&recording));
        let camera_mode = slot.map_or(CameraMode::Free, CameraMode::Eyes);
        let (low, high) = map.bounds;
        let center = (low + high) * 0.5;

        let markers = request
            .markers
            .iter()
            .map(|marker| {
                let at = f64::from(marker.tick);
                let name = recording
                    .slot_of_steam_id(marker.steam_id, at)
                    .map_or("Unknown player", |slot| recording.name_at(slot, at));
                Marker {
                    label: format!("{name}: {}", marker.label),
                    ..marker.clone()
                }
            })
            .collect();

        let mut session = Self {
            demo,
            review: None,
            pose_memory: RefCell::new(HashMap::new()),
            ragdolls: RefCell::new(HashMap::new()),
            sentry_shots: sentry_shots(&recording, &map.lighting),
            presence: presence_runs(&recording),
            muzzles: RefCell::new(HashMap::new()),
            requests: Vec::new(),
            marks: HashMap::new(),
            can_mark: false,
            report_status: None,
            view_rect: Rect::NOTHING,
            clip: None,
            encoding: None,
            clip_seconds: CLIP_LENGTHS[1],
            clip_status: None,
            shot_ends: shot_ends(&recording, |from, to| map.lighting.first_hit(from, to)),
            trails: build_trails(&recording),
            weapon_roles,
            lighting: Arc::clone(&map.lighting),
            sky_camera: map.sky_camera,
            characters,
            map_props,
            scope_image,
            scope_texture: None,
            crosshair_image,
            crosshair_texture: None,
            recording,
            pending_map: Arc::new(Mutex::new(Some(map))),
            renderer: Arc::new(Mutex::new(None)),
            render_error: Arc::new(Mutex::new(None)),
            markers,
            tick,
            playing: false,
            resume_after_scrub: false,
            speed_index: 2,
            fov: DEFAULT_FOV,
            show_names: true,
            show_bullets: true,
            show_projectiles: true,
            glow: GlowMode::default(),
            show_aim_lines: false,
            show_hitboxes: false,
            show_ghosts: false,
            ghost_interp_ms: DEFAULT_INTERP_MS,
            show_damage: true,
            show_impacts: true,
            show_aim_graph: true,
            show_fov_circle: true,
            aimbot_fovs: HashMap::new(),
            only_their_kills: true,
            third_person: false,
            orbit: None,
            chase_distance: CHASE_BEHIND,
            audio: super::audio::Audio::start(),
            sound_bank: SoundBank { files: sound_files, clips: HashMap::new() },
            sound_tick: tick,
            sound_was_on: false,
            volume: DEFAULT_VOLUME,
            muted: false,
            hit_sounds: true,
            voice_chat: true,
            voice_lines,
            own_shots,
            options_restored: false,
            camera_mode,
            free_camera: FreeCamera {
                position: center,
                yaw: 0.0,
                pitch: 20.0,
            },
        };
        let flagged: Vec<u64> = super::review::flagged_players(&session.markers).into_iter().map(|(steam_id, _)| steam_id).collect();
        for steam_id in flagged {
            let measure = |marker: &Marker| session.measured_flick(marker);
            if let Some(estimate) = super::review::aimbot_fov(&session.markers, steam_id, &measure) {
                session.aimbot_fovs.insert(steam_id, estimate);
            }
        }
        session
    }

    fn renderer_ready(&self) -> bool {
        self.renderer.lock().is_ok_and(|renderer| renderer.is_some())
    }

    pub fn recording(&self) -> &Recording {
        &self.recording
    }

    pub fn seek(&mut self, tick: f64) {
        let first = f64::from(self.recording.first_tick());
        let last = f64::from(self.recording.last_tick());
        self.tick = tick.clamp(first, last);
    }

    /// What there is to hear in this demo, for checking.
    pub fn sound_summary(&self) -> String {
        let talkers: std::collections::BTreeSet<_> = self.voice_lines.iter().filter_map(|line| line.slot).collect();
        let starts: Vec<String> = self.voice_lines.iter().take(5).map(|line| format!("{:.0}", line.start)).collect();
        format!(
            "{} demo sounds, {} explosions, {} own shots, {} voice clips from {} players ({:.0} s, first at ticks {}), sound device {}",
            self.recording.sounds.len(),
            self.recording.explosions.len(),
            self.own_shots.len(),
            self.voice_lines.len(),
            talkers.len(),
            self.voice_lines.iter().fold(0.0, |total, line| total + (line.end - line.start) * f64::from(self.recording.tick_interval)),
            starts.join(" "),
            self.audio.is_some(),
        )
    }

    /// Starts keeping a copy of the sound played, for checking it. False without a sound device.
    pub fn start_sound_recording(&mut self) -> bool {
        self.audio.as_ref().map(|audio| audio.start_recording(true)).is_some()
    }

    /// The sound played since `start_sound_recording`: its sample rate and the left
    /// and right samples interleaved.
    pub fn take_sound_recording(&mut self) -> Option<(u32, Vec<f32>)> {
        self.audio.as_ref().map(|audio| (audio.rate(), audio.take_recording()))
    }

    pub fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
    }

    pub fn tick(&self) -> f64 {
        self.tick
    }

    /// The two frames around the current tick and how far between them it is.
    fn frames_around(&self) -> (usize, usize, f32) {
        let recording = &self.recording;
        let frame = recording.frame_at(self.tick);
        let next = (frame + 1).min(recording.ticks.len() - 1);
        let span = f64::from(recording.ticks[next]) - f64::from(recording.ticks[frame]);
        let t = if span > 0.0 {
            ((self.tick - f64::from(recording.ticks[frame])) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (frame, next, t as f32)
    }

    fn blended_player(&self, slot: usize) -> PlayerFrame {
        let (frame, next, t) = self.frames_around();
        let a = *self.recording.player(frame, slot);
        let b = *self.recording.player(next, slot);
        let (pa, pb) = (Vec3::from(a.position), Vec3::from(b.position));
        if !a.is_alive() || !b.is_alive() || pa.distance(pb) > MAX_STEP_PER_FRAME {
            return a;
        }
        PlayerFrame {
            position: pa.lerp(pb, t).to_array(),
            yaw: lerp_angle(a.yaw, b.yaw, t),
            pitch: lerp_angle(normalize_pitch(a.pitch), normalize_pitch(b.pitch), t),
            ..a
        }
    }

    fn camera(&self) -> Camera {
        let camera = self.eyes_or_free_camera();
        let first_hit = |from, to| self.lighting.first_hit(from, to);
        match (self.camera_mode, self.orbit) {
            (CameraMode::Eyes(_), None) if self.third_person => Camera {
                eye: chase_eye(camera.eye, camera.pitch, camera.yaw, self.chase_distance, first_hit),
                roll: 0.0,
                ..camera
            },
            (CameraMode::Eyes(_), Some((yaw, pitch))) if self.third_person => Camera {
                eye: orbit_eye(camera.eye - Vec3::Z * ORBIT_DROP, pitch, yaw, self.chase_distance, first_hit),
                yaw,
                pitch,
                roll: 0.0,
            },
            _ => camera,
        }
    }

    /// The player watched through their eyes in first person, whose hands, scope and
    /// crosshair show. `None` in the free camera and in third person.
    fn first_person_slot(&self) -> Option<usize> {
        self.watched_slot().filter(|_| !self.third_person)
    }

    fn eyes_or_free_camera(&self) -> Camera {
        match self.camera_mode {
            CameraMode::Free => Camera {
                eye: self.free_camera.position,
                yaw: self.free_camera.yaw,
                pitch: self.free_camera.pitch,
                roll: 0.0,
            },
            CameraMode::Eyes(slot) if Some(slot) == self.recording.recorder_slot => {
                let (frame, next, t) = self.frames_around();
                let a = self.recording.views[frame];
                let b = self.recording.views[next];
                let (pa, pb) = (Vec3::from(a.origin), Vec3::from(b.origin));
                let mut eye = if pa.distance(pb) > MAX_STEP_PER_FRAME {
                    pa
                } else {
                    pa.lerp(pb, t)
                };
                // While playing, the demo records where the player's feet are.
                // When spectating it records the camera itself.
                let body = self.recording.player(frame, slot);
                if body.is_alive() && Vec3::from(body.position).distance(pa) < 4.0 {
                    eye.z += body.eye_height();
                }
                Camera {
                    eye,
                    yaw: lerp_angle(a.yaw, b.yaw, t),
                    pitch: lerp_angle(a.pitch, b.pitch, t),
                    roll: lerp_angle(a.roll, b.roll, t),
                }
            }
            CameraMode::Eyes(slot) => {
                let player = self.blended_player(slot);
                Camera {
                    eye: Vec3::from(player.position) + Vec3::Z * player.eye_height(),
                    yaw: player.yaw,
                    pitch: normalize_pitch(player.pitch),
                    roll: 0.0,
                }
            }
        }
    }

    /// Whether the player being watched is looking through a sniper scope.
    fn scoped(&self) -> bool {
        self.first_person_slot().is_some_and(|slot| {
            let frame = self.recording.frame_at(self.tick);
            let player = self.recording.player(frame, slot);
            player.is_alive() && player.is_zoomed()
        })
    }

    fn watched_slot(&self) -> Option<usize> {
        match self.camera_mode {
            CameraMode::Eyes(slot) => Some(slot),
            CameraMode::Free => None,
        }
    }

    fn matrices(&self, camera: &Camera, aspect: f32) -> (Mat4, Mat4) {
        let forward = forward_from(camera.pitch, camera.yaw);
        let mut view = Mat4::look_to_rh(camera.eye, forward, Vec3::Z);
        if camera.roll != 0.0 {
            view = Mat4::from_rotation_z(camera.roll.to_radians()) * view;
        }
        // `fov` is measured on a 4:3 screen, the way TF2 measures it.
        let fov = if self.scoped() { SCOPED_FOV } else { self.fov };
        let vertical = 2.0 * ((fov.to_radians() * 0.5).tan() * 0.75).atan();
        (view, Mat4::perspective_rh_gl(vertical, aspect, NEAR_PLANE, FAR_PLANE))
    }

    /// Which way and how fast a player is moving, from where they were a moment ago.
    fn velocity(&self, slot: usize) -> Vec3 {
        let window = VELOCITY_WINDOW_SECONDS / f64::from(self.recording.tick_interval);
        let now = self.recording.frame_at(self.tick);
        let before = self.recording.frame_at(self.tick - window);
        if now == before {
            return Vec3::ZERO;
        }
        let a = self.recording.player(before, slot);
        let b = self.recording.player(now, slot);
        if !a.is_alive() || !b.is_alive() {
            return Vec3::ZERO;
        }
        let seconds = self
            .recording
            .seconds_between(f64::from(self.recording.ticks[before]), f64::from(self.recording.ticks[now]))
            as f32;
        let moved = Vec3::from(b.position) - Vec3::from(a.position);
        if seconds <= 0.0 || moved.length() > MAX_STEP_PER_FRAME * 4.0 {
            return Vec3::ZERO;
        }
        moved / seconds
    }

    /// Works out how a player's body should be posed from how they're moving.
    fn pose_input(&self, slot: usize, player: &PlayerFrame) -> PoseInput {
        let velocity = self.velocity(slot);
        let flat = velocity.truncate();
        let speed = flat.length();
        let moving = speed > MOVING_SPEED;

        let on_foot = match (player.is_ducking(), moving) {
            (true, true) => Stance::CrouchWalk,
            (true, false) => Stance::Crouch,
            (false, true) => Stance::Run,
            (false, false) => Stance::Stand,
        };
        let mut air_seconds = 0.0;
        let stance = if player.is_in_water() {
            Stance::Swim
        } else if !player.is_on_a_team() || player.is_on_ground() {
            on_foot
        } else {
            match self.airborne(slot) {
                Airborne::Jump(seconds) if seconds < JUMP_START_SECONDS => {
                    air_seconds = seconds;
                    Stance::JumpStart
                }
                Airborne::Jump(seconds) => {
                    air_seconds = seconds - JUMP_START_SECONDS;
                    Stance::JumpFloat
                }
                Airborne::Flung => Stance::Air,
                // Like TF2, stepping off a ledge keeps the legs going.
                Airborne::Fell => on_foot,
            }
        };

        let (move_x, move_y) = if moving {
            let heading = flat.y.atan2(flat.x).to_degrees();
            let relative = (heading - player.yaw).to_radians();
            let stride = (speed / FULL_RUN_SPEED).min(1.0);
            (relative.cos() * stride, -relative.sin() * stride)
        } else {
            (0.0, 0.0)
        };

        let weapon = self.weapon_pose(player);
        let deployed = player.is_aiming() && matches!(weapon, WeaponPose::Primary | WeaponPose::Item2);
        let seconds = self
            .recording
            .seconds_between(f64::from(self.recording.first_tick()), self.tick) as f32;
        // Walking animations advance with distance so feet don't slide.
        let walked = {
            let (frame, next, t) = self.frames_around();
            let a = self.recording.player(frame, slot).travelled;
            let b = self.recording.player(next, slot).travelled;
            a + (b - a) * t
        };
        let speed = run_speed(player.class);
        let time = match stance {
            Stance::Run => walked / speed,
            Stance::CrouchWalk => walked / (speed / 3.0),
            Stance::Swim => walked / (speed * 0.8),
            Stance::Stand | Stance::Crouch | Stance::Air => seconds,
            Stance::JumpStart | Stance::JumpFloat => air_seconds,
        };
        // Offsetting by slot keeps everyone from stepping in time. A jump plays from its start.
        let offset = if matches!(stance, Stance::JumpStart | Stance::JumpFloat) { 0.0 } else { slot as f32 * 0.37 };

        PoseInput {
            stance,
            weapon,
            gesture: self.gesture_at(slot),
            deployed,
            move_x,
            move_y,
            body_pitch: -normalize_pitch(player.pitch),
            body_yaw: 0.0,
            time: time + offset,
        }
    }

    /// A body's pose, blended from its last animation when it just changed, so
    /// stopping, landing or crouching doesn't snap. Playing forward blends; seeking
    /// or stepping back starts clean.
    fn blended_pose(&self, slot: usize, player: &PlayerFrame, rig: &Rig) -> Pose {
        let input = self.pose_input(slot, player);
        let blend_ticks = STANCE_BLEND_SECONDS / f64::from(self.recording.tick_interval);
        let mut memory = self.pose_memory.borrow_mut();
        let fresh = PoseMemory { current: input, previous: None, seen_at: self.tick };
        let entry = memory.entry(slot).or_insert(fresh);
        let step = self.tick - entry.seen_at;
        if !(0.0..=blend_ticks).contains(&step) {
            *entry = fresh;
        } else if animation_changed(&entry.current, &input) {
            entry.previous = Some((entry.current, entry.seen_at));
        }
        entry.current = input;
        entry.seen_at = self.tick;
        let pose = rig.pose(&input);
        let Some((mut before, changed_at)) = entry.previous else { return pose };
        let progress = ((self.tick - changed_at) / blend_ticks) as f32;
        if progress >= 1.0 {
            entry.previous = None;
            return pose;
        }
        // The old animation keeps moving while it fades out.
        before.time += ((self.tick - changed_at) * f64::from(self.recording.tick_interval)) as f32;
        before.gesture = input.gesture;
        let eased = progress * progress * (3.0 - 2.0 * progress);
        rig.pose(&before).blend(&pose, eased)
    }

    /// How a player's body is posed right now, for checking animations by hand:
    /// a short key (class, stance, animation set, gesture) and the full details.
    pub fn pose_summary(&self, slot: usize) -> Option<(String, String)> {
        let player = self.blended_player(slot);
        if !player.is_alive() || !player.is_on_a_team() || player.is_out_of_view() {
            return None;
        }
        let input = self.pose_input(slot, &player);
        let mut key = format!("{} {:?} {:?}", class_name(player.class), input.stance, input.weapon);
        if input.deployed {
            key.push_str(" deployed");
        }
        if let Some((gesture, _)) = input.gesture {
            key.push_str(&format!(" {gesture:?}"));
        }
        let weapon = self.recording.models.get(usize::from(player.weapon_model)).map_or("", String::as_str);
        let details = format!(
            "{key} | move ({:.2}, {:.2}) pitch {:.0} time {:.2} gesture {:?} | {weapon}",
            input.move_x, input.move_y, input.body_pitch, input.time, input.gesture
        );
        Some((key, details))
    }

    /// Where the 3D view was drawn last frame.
    pub fn view_rect(&self) -> Rect {
        self.view_rect
    }

    /// Whether nothing solid in the map is between two points.
    pub fn line_is_clear(&self, from: Vec3, to: Vec3) -> bool {
        self.lighting.first_hit(from, to).is_none()
    }

    /// The outline colour for a player on `team`, if the glow setting shows them.
    fn glow_color(&self, team: u8) -> Option<[f32; 3]> {
        let color = match team {
            TEAM_RED => GLOW_RED,
            TEAM_BLUE => GLOW_BLUE,
            _ => return None,
        };
        // Teams are judged from whoever is watched, or the recording player.
        let viewer = self
            .watched_slot()
            .or(self.recording.recorder_slot)
            .map(|slot| self.recording.player(self.recording.frame_at(self.tick), slot).team)
            .filter(|&team| team == TEAM_RED || team == TEAM_BLUE);
        let shown = match (self.glow, viewer) {
            (GlowMode::Off, _) => false,
            (GlowMode::Everyone, _) | (_, None) => true,
            (GlowMode::Enemies, Some(viewer)) => team != viewer,
            (GlowMode::Teammates, Some(viewer)) => team == viewer,
        };
        shown.then_some(color)
    }

    /// The shooting or reloading animation a player is in the middle of, from the demo's
    /// animation events.
    fn gesture_at(&self, slot: usize) -> Option<(Gesture, f32)> {
        let anims = &self.recording.anims;
        let interval = f64::from(self.recording.tick_interval);
        let end = anims.partition_point(|anim| f64::from(anim.tick) <= self.tick);
        let oldest = self.tick - GESTURE_LOOKBACK_SECONDS / interval;
        for anim in anims[..end].iter().rev() {
            if f64::from(anim.tick) < oldest {
                break;
            }
            if anim.slot != slot {
                continue;
            }
            let gesture = match anim.event {
                ANIM_ATTACK_PRIMARY => Gesture::Attack,
                ANIM_ATTACK_SECONDARY => Gesture::AttackSecondary,
                ANIM_RELOAD => Gesture::Reload,
                ANIM_RELOAD_LOOP => Gesture::ReloadLoop,
                ANIM_RELOAD_END => Gesture::ReloadEnd,
                _ => continue,
            };
            // Switching weapons ends whatever the last one was doing.
            let recording = &self.recording;
            let now = recording.player(recording.frame_at(self.tick), slot).weapon_model;
            let then = recording.frame_at(f64::from(anim.tick));
            let same_weapon = [then, (then + 1).min(recording.ticks.len() - 1)]
                .iter()
                .any(|&frame| recording.player(frame, slot).weapon_model == now);
            if !same_weapon {
                return None;
            }
            return Some((gesture, ((self.tick - f64::from(anim.tick)) * interval) as f32));
        }
        None
    }

    /// When a player last pressed jump, from the demo's animation events.
    fn jump_event(&self, slot: usize, since: f64) -> Option<f64> {
        let anims = &self.recording.anims;
        let end = anims.partition_point(|anim| f64::from(anim.tick) <= self.tick);
        anims[..end]
            .iter()
            .rev()
            .take_while(|anim| f64::from(anim.tick) >= since)
            .find(|anim| anim.slot == slot && matches!(anim.event, ANIM_JUMP | ANIM_DOUBLE_JUMP))
            .map(|anim| f64::from(anim.tick))
    }

    /// Which animation set goes with what the player holds: from TF2's item
    /// schema when the weapon is known, otherwise from its slot.
    fn weapon_pose(&self, player: &PlayerFrame) -> WeaponPose {
        self.weapon_roles
            .get(usize::from(player.weapon_model))
            .and_then(|roles| roles.get(usize::from(player.class)).copied().flatten())
            .unwrap_or(match player.weapon_slot {
                1 => WeaponPose::Secondary,
                2 => WeaponPose::Melee,
                _ => WeaponPose::Primary,
            })
    }

    /// How a player off the ground got there, worked out from how fast they
    /// rose after leaving it.
    fn airborne(&self, slot: usize) -> Airborne {
        let recording = &self.recording;
        let now = recording.frame_at(self.tick);
        let mut first = now;
        while first > 0 && now - first < MAX_AIR_FRAMES {
            let before = recording.player(first - 1, slot);
            if !before.is_alive() || before.is_on_ground() || before.is_in_water() {
                break;
            }
            first -= 1;
        }
        if first == 0 || now - first >= MAX_AIR_FRAMES {
            return Airborne::Flung;
        }
        let rise = |from: usize, to: usize| -> Option<f32> {
            let (a, b) = (recording.player(from, slot), recording.player(to, slot));
            // Crouching in the air lifts the body 20 units at once, which isn't speed.
            if a.is_ducking() != b.is_ducking() {
                return None;
            }
            let seconds = recording.seconds_between(f64::from(recording.ticks[from]), f64::from(recording.ticks[to])) as f32;
            let moved = Vec3::from(b.position) - Vec3::from(a.position);
            let frames = (to - from).max(1) as f32;
            (seconds > 0.0 && moved.length() < MAX_STEP_PER_FRAME * frames).then(|| moved.z / seconds)
        };
        let flung = (first..=now).any(|frame| {
            let from = frame.saturating_sub(RISE_WINDOW_FRAMES).max(first - 1);
            rise(from, frame).is_some_and(|speed| speed > FLUNG_SPEED)
        });
        if flung {
            return Airborne::Flung;
        }
        // The demo says when the jump key was pressed, which beats guessing from speed.
        let left = f64::from(recording.ticks[first - 1]);
        if let Some(jumped) = self.jump_event(slot, left - 4.0) {
            let interval = f64::from(recording.tick_interval);
            return Airborne::Jump(((self.tick - jumped) * interval).max(0.0) as f32);
        }
        let launch = rise(first - 1, (first + 1).min(now.max(first))).unwrap_or(0.0);
        if launch < JUMP_SPEED {
            return Airborne::Fell;
        }
        Airborne::Jump(recording.seconds_between(left, self.tick).max(0.0) as f32)
    }

    fn scene(&self) -> Scene {
        self.muzzles.borrow_mut().clear();
        let (frame, next, t) = self.frames_around();
        let recording = &self.recording;
        let characters = &self.characters;
        let mut shapes = Vec::new();
        let mut actors = Vec::new();
        let mut boxes = Vec::new();
        let want_boxes = self.show_hitboxes || self.show_ghosts;

        for slot in 0..recording.slot_count {
            if Some(slot) == self.first_person_slot() {
                continue;
            }
            let player = self.blended_player(slot);
            // Without a class they haven't spawned yet, whatever the alive flag says.
            if !player.is_alive() || !player.is_on_a_team() || player.is_out_of_view() || player.class == 0 {
                continue;
            }
            let position = Vec3::from(player.position);
            let lighting = self.lighting.at(position + Vec3::Z * 40.0);
            let root = Mat4::from_translation(position) * Mat4::from_rotation_z(player.yaw.to_radians());
            let skin = usize::from(player.team == TEAM_BLUE);

            let body = characters
                .classes
                .get(usize::from(player.class))
                .and_then(Option::as_ref);
            let Some(body) = body else {
                let height = if player.is_ducking() { 0.74 } else { 1.0 };
                shapes.push(ShapeInstance {
                    shape: Shape::Figure,
                    transform: root * Mat4::from_scale(Vec3::new(1.0, 1.0, height)),
                    tint: team_tint(player.team),
                    cube: lighting.cube,
                    sun: lighting.sun,
                });
                continue;
            };

            let pose = self.blended_pose(slot, &player, &body.rig);
            let skeleton = &body.rig.skeleton;
            let world = skeleton.world_matrices(&pose, root);
            if want_boxes {
                boxes.push(PlayerBoxes {
                    slot,
                    team: player.team,
                    boxes: skeleton
                        .hitboxes
                        .iter()
                        .map(|hitbox| (world[hitbox.bone], hitbox.min, hitbox.max, hitbox.group == HITGROUP_HEAD))
                        .collect(),
                });
            }
            let glow = self.glow_color(player.team);
            actors.push(Actor {
                mesh: body.mesh,
                skin,
                body: 0,
                bones: skeleton.skinning_matrices(&world),
                cube: lighting.cube,
                sun: lighting.sun,
                glow,
            });

            let weapon_mesh = characters
                .by_model
                .get(usize::from(player.weapon_model))
                .copied()
                .flatten();
            // Worn items with nothing to hang on would float at the player's feet.
            let weapon_mesh = weapon_mesh.filter(|&mesh| characters.skeletons[mesh].attaches_to(skeleton));
            if let Some(mesh) = weapon_mesh {
                let weapon = &characters.skeletons[mesh];
                let weapon_world = weapon.attached_world(skeleton, &world, root);
                if let Some(muzzle) = weapon.attachment_point("muzzle", &weapon_world) {
                    self.muzzles.borrow_mut().insert(slot, muzzle);
                }
                actors.push(Actor {
                    mesh,
                    skin,
                    body: 0,
                    bones: weapon.skinning_matrices(&weapon_world),
                    cube: lighting.cube,
                    sun: lighting.sun,
                    glow,
                });
            }

        }

        actors.extend(self.bodies());
        for (prop, lighting) in &self.map_props {
            let Some(mesh) = characters.by_model.get(prop.model).copied().flatten() else { continue };
            let skeleton = &characters.skeletons[mesh];
            let world = skeleton.world_matrices(&characters.idle_poses[mesh], prop.transform);
            actors.push(Actor {
                mesh,
                skin: prop.skin,
                body: 0,
                bones: skeleton.skinning_matrices(&world),
                cube: lighting.cube,
                sun: lighting.sun,
                glow: None,
            });
        }

        // Map objects such as carts, doors and pickups, playing their own animations.
        for track in &recording.props {
            let Some((key, next)) = track.keys_around(self.tick) else { continue };
            let Some(mesh) = characters.by_model.get(usize::from(track.model)).copied().flatten() else {
                continue;
            };
            let mut position = Vec3::from(key.position);
            let mut rotation = glam::Quat::from_array(key.rotation);
            if let Some(next) = next {
                let span = f64::from(next.tick) - f64::from(key.tick);
                let target = Vec3::from(next.position);
                if span > 0.0 && span < 20.0 && position.distance(target) < MAX_STEP_PER_FRAME * 2.0 {
                    let t = ((self.tick - f64::from(key.tick)) / span).clamp(0.0, 1.0) as f32;
                    position = position.lerp(target, t);
                    rotation = rotation.slerp(glam::Quat::from_array(next.rotation), t);
                }
            }
            let skeleton = &characters.skeletons[mesh];
            let seconds = recording.seconds_between(f64::from(key.sequence_start), self.tick) as f32;
            let pose = characters.animations[mesh]
                .as_ref()
                .and_then(|animations| animations.pose(skeleton, usize::from(key.sequence), seconds))
                .unwrap_or_else(|| characters.idle_poses[mesh].clone());
            let root = Mat4::from_rotation_translation(rotation, position);
            let lighting = self.lighting.at(position + Vec3::Z * 16.0);
            actors.push(Actor {
                mesh,
                skin: usize::from(key.skin),
                body: u32::from(key.body),
                bones: skeleton.skinning_matrices(&skeleton.world_matrices(&pose, root)),
                cube: lighting.cube,
                sun: lighting.sun,
                glow: None,
            });
        }

        let upcoming: HashMap<u16, &ObjectFrame> = recording
            .objects_at(next)
            .iter()
            .map(|object| (object.id, object))
            .collect();

        for object in recording.objects_at(frame) {
            // Drawn as a glow with the traces, like TF2's particles.
            if object.kind.is_energy() {
                continue;
            }
            let mut position = Vec3::from(object.position);
            if let Some(after) = upcoming.get(&object.id) {
                let target = Vec3::from(after.position);
                if position.distance(target) < MAX_STEP_PER_FRAME * 2.0 {
                    position = position.lerp(target, t);
                }
            }
            let lighting = self.lighting.at(position + Vec3::Z * 12.0);
            let rotation = rotation_from_angles(object.pitch, object.yaw, 0.0);

            let mesh = characters.by_model.get(usize::from(object.model)).copied().flatten();
            if let Some(mesh) = mesh {
                let skeleton = &characters.skeletons[mesh];
                let root = Mat4::from_translation(position) * rotation;
                let pose = object
                    .kind
                    .is_building()
                    .then(|| self.building_pose(object, upcoming.get(&object.id).copied(), t, mesh))
                    .flatten()
                    .unwrap_or_else(|| characters.idle_poses[mesh].clone());
                let world = skeleton.world_matrices(&pose, root);
                actors.push(Actor {
                    mesh,
                    skin: usize::from(object.team == TEAM_BLUE),
                    body: 0,
                    bones: skeleton.skinning_matrices(&world),
                    cube: lighting.cube,
                    sun: lighting.sun,
                    glow: None,
                });
                continue;
            }

            if object.kind == ObjectKind::Cart {
                continue;
            }
            let (shape, scale) = match object.kind {
                ObjectKind::Rocket | ObjectKind::Arrow | ObjectKind::Flare => (Shape::Rocket, Vec3::ONE),
                ObjectKind::Sticky
                | ObjectKind::Pipe
                | ObjectKind::LooseCannon
                | ObjectKind::Thrown
                | ObjectKind::OtherProjectile
                | ObjectKind::EnergyRing
                | ObjectKind::EnergyBall
                | ObjectKind::Orb
                | ObjectKind::Fireball => (Shape::Ball, Vec3::ONE),
                ObjectKind::Sentry => (Shape::Box, Vec3::new(20.0, 20.0, if object.mini { 18.0 } else { 26.0 })),
                ObjectKind::Dispenser => (Shape::Box, Vec3::new(18.0, 18.0, 30.0)),
                ObjectKind::TeleEntrance | ObjectKind::TeleExit => (Shape::Box, Vec3::new(24.0, 24.0, 5.0)),
                ObjectKind::Cart => (Shape::Box, Vec3::new(32.0, 22.0, 24.0)),
            };
            let lift = if matches!(shape, Shape::Box) { scale.z } else { 0.0 };
            shapes.push(ShapeInstance {
                shape,
                transform: Mat4::from_translation(position + Vec3::Z * lift) * rotation * Mat4::from_scale(scale),
                tint: team_tint(object.team),
                cube: lighting.cube,
                sun: lighting.sun,
            });
        }
        Scene { shapes, actors, boxes }
    }

    /// A dead body's fall, worked out the first time it's shown: from a standing
    /// pose where they died, moving as they were, shoved by the killing blow.
    fn ragdoll(&self, index: usize, kill: &Kill, slot: usize, body: &PlayerFrame, rig: &Rig) -> Option<Arc<Ragdoll>> {
        if let Some(found) = self.ragdolls.borrow().get(&index) {
            return found.clone();
        }
        let recording = &self.recording;
        let interval = f64::from(recording.tick_interval);
        let tick = f64::from(kill.tick);
        let frame = recording.frame_at(tick).saturating_sub(1);
        let position = Vec3::from(body.position);
        // How it was moving just before.
        let earlier = recording.frame_at(tick - RAGDOLL_SPEED_WINDOW_SECONDS / interval).min(frame);
        let before = recording.player(earlier, slot);
        let seconds = recording.seconds_between(f64::from(recording.ticks[earlier]), f64::from(recording.ticks[frame])) as f32;
        let moved = position - Vec3::from(before.position);
        let velocity = if seconds > 0.0 && before.is_alive() && moved.length() < MAX_STEP_PER_FRAME {
            (moved / seconds).clamp_length_max(RAGDOLL_MAX_SPEED)
        } else {
            Vec3::ZERO
        };
        // The killing blow shoves the body away from whoever made it.
        let push = kill
            .attacker
            .filter(|&attacker| attacker != slot)
            .map(|attacker| {
                let from = Vec3::from(recording.player(frame, attacker).position);
                let away = (position - from).truncate().normalize_or_zero().extend(0.0);
                let (sideways, upward) = blow_strength(&kill.weapon);
                away * sideways + Vec3::Z * upward
            })
            .unwrap_or(Vec3::ZERO);
        let floors = self.standing_floors(position, tick);
        let stance = if body.is_ducking() { Stance::Crouch } else { Stance::Stand };
        let pose = rig.pose(&PoseInput { stance, weapon: self.weapon_pose(body), ..PoseInput::default() });
        let root = Mat4::from_translation(position) * Mat4::from_rotation_z(body.yaw.to_radians());
        let start = rig.skeleton.world_matrices(&pose, root);
        let lighting = Arc::clone(&self.lighting);
        let ragdoll = Ragdoll::simulate(
            &rig.skeleton,
            start,
            velocity,
            push,
            BODY_SECONDS as f32,
            recording.tick_interval,
            move |from, to| {
                // Whichever comes first: the map, or a floor players were seen standing on.
                match (lighting.first_hit(from, to), floor_hit(&floors, from, to)) {
                    (Some(wall), Some(floor)) => Some(if from.distance_squared(wall) <= from.distance_squared(floor) { wall } else { floor }),
                    (wall, floor) => wall.or(floor),
                }
            },
        )
        .map(Arc::new);
        self.ragdolls.borrow_mut().insert(index, ragdoll.clone());
        ragdoll
    }

    /// Floors players stood on around a death, a little before and after it, that the
    /// map's own solid shape leaves out: bridges, platforms, crates and props are
    /// separate objects, so a body would otherwise fall straight through them.
    fn standing_floors(&self, around: Vec3, tick: f64) -> Vec<Vec3> {
        let recording = &self.recording;
        let interval = f64::from(recording.tick_interval);
        let first = recording.frame_at(tick - FLOOR_SECONDS_BEFORE / interval);
        let last = recording.frame_at(tick + FLOOR_SECONDS_AFTER / interval);
        let cell = |value: f32| (value / FLOOR_CELL).floor() as i32;
        let mut cells: HashMap<(i32, i32), Vec<f32>> = HashMap::new();
        for frame in (first..=last).step_by(2) {
            for slot in 0..recording.slot_count {
                let player = recording.player(frame, slot);
                if !player.is_alive() || !player.is_on_ground() || player.is_out_of_view() {
                    continue;
                }
                let position = Vec3::from(player.position);
                if position.distance(around) > FLOOR_RADIUS {
                    continue;
                }
                let heights = cells.entry((cell(position.x), cell(position.y))).or_default();
                if !heights.iter().any(|height| (height - position.z).abs() < 4.0) {
                    heights.push(position.z);
                }
            }
        }
        let mut floors = Vec::new();
        for ((x, y), heights) in cells {
            for height in heights {
                let center = Vec3::new((x as f32 + 0.5) * FLOOR_CELL, (y as f32 + 0.5) * FLOOR_CELL, height);
                // Where the map already has the floor, its own shape is better.
                if self.lighting.first_hit(center + Vec3::Z * 3.0, center - Vec3::Z * 6.0).is_none() {
                    floors.push(center);
                }
            }
        }
        floors
    }

    /// A building's pose: its sequence at the cycle the server sent, or by time for
    /// ones that animate on their own like a spinning teleporter, turned by its pose
    /// parameters, which is how a sentry aims.
    fn building_pose(&self, object: &ObjectFrame, after: Option<&ObjectFrame>, t: f32, mesh: usize) -> Option<Pose> {
        let animations = self.characters.animations[mesh].as_ref()?;
        let skeleton = &self.characters.skeletons[mesh];
        let now = object.animation;
        let later = after.map(|after| after.animation).filter(|later| later.sequence == now.sequence);
        let sequence = usize::from(now.sequence);
        let cycle = match now.cycle {
            Some(cycle) => match later.and_then(|later| later.cycle) {
                Some(next) if next >= cycle => cycle + (next - cycle) * t,
                _ => cycle,
            },
            None => {
                let length = animations.sequence_seconds(skeleton, sequence)?.max(0.001);
                let seconds = self.recording.seconds_between(f64::from(self.recording.first_tick()), self.tick) as f32;
                (seconds * now.rate / length).rem_euclid(1.0)
            }
        };
        let mut pose = now.pose;
        if let Some(later) = later {
            for (value, next) in pose.iter_mut().zip(later.pose) {
                // A turn across the end of the range wraps around, so it isn't blended.
                if (next - *value).abs() < 0.5 {
                    *value += (next - *value) * t;
                }
            }
        }
        // While firing, the server switches a sentry to its firing sequence, which only
        // adds to the aiming idle: the idle stays underneath and the firing plays on top.
        if animations.is_overlay(sequence) {
            let base = animations.find_activity("ACT_OBJ_RUNNING").unwrap_or(0);
            let mut posed = animations.pose_at(skeleton, base, 0.0, &pose)?;
            animations.add_layer(skeleton, &mut posed, sequence, cycle, &pose);
            return Some(posed);
        }
        let mut posed = animations.pose_at(skeleton, sequence, cycle, &pose)?;
        // A sentry that fired a moment ago keeps its firing animation going.
        if object.kind == ObjectKind::Sentry && self.sentry_firing(object.id) {
            if let Some(fire) = animations.find_activity("ACT_RANGE_ATTACK1") {
                let length = animations.sequence_seconds(skeleton, fire).unwrap_or(0.25).max(0.01);
                let seconds = self.recording.seconds_between(f64::from(self.recording.first_tick()), self.tick) as f32;
                animations.add_layer(skeleton, &mut posed, fire, (seconds / length).rem_euclid(1.0), &pose);
            }
        }
        Some(posed)
    }

    /// A small cross where each recent bullet hit, fading out.
    fn impact_marks(&self, camera: &Camera, floats: &mut Vec<f32>) {
        let recording = &self.recording;
        let window = IMPACT_SECONDS / f64::from(recording.tick_interval);
        let first = recording.shots.partition_point(|shot| f64::from(shot.tick) < self.tick - window);
        for (shot, end) in recording.shots[first..].iter().zip(&self.shot_ends[first..]) {
            let tick = f64::from(shot.tick);
            if tick > self.tick {
                break;
            }
            let fade = (1.0 - (self.tick - tick) / window) as f32;
            // Size grows with distance so far marks stay visible.
            let size = (end.distance(camera.eye) * 0.006).clamp(2.0, 14.0);
            let color = if shot.crit { [1.0, 0.85, 0.35] } else { [1.0, 0.95, 0.8] };
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                push_strip(floats, *end - axis * size, *end + axis * size, camera.eye, color, [0.9 * fade, 0.9 * fade], 1.6);
            }
        }
    }

    /// Lines that show through walls: aim lines, hitboxes and backtrack ghosts.
    fn overlay(&self, camera: &Camera, boxes: &[PlayerBoxes]) -> Vec<f32> {
        let mut floats = Vec::new();
        let watched = self.first_person_slot();

        if self.show_aim_lines {
            for slot in 0..self.recording.slot_count {
                if Some(slot) == watched {
                    continue;
                }
                let player = self.blended_player(slot);
                if !player.is_alive() || !player.is_on_a_team() || player.is_out_of_view() {
                    continue;
                }
                let eye = Vec3::from(player.position) + Vec3::Z * player.eye_height();
                let far = eye + forward_from(normalize_pitch(player.pitch), player.yaw) * AIM_LINE_RANGE;
                let end = self.lighting.first_hit(eye, far).unwrap_or(far);
                push_strip(&mut floats, eye, end, camera.eye, team_rgb(player.team), [0.85, 0.35], 1.0);
            }
        }

        if self.show_hitboxes {
            for player in boxes {
                for (bone, min, max, head) in &player.boxes {
                    let color = if *head { [1.0, 0.9, 0.3] } else { team_rgb(player.team) };
                    for (from, to) in box_edges(bone, *min, *max) {
                        push_strip(&mut floats, from, to, camera.eye, color, [0.9, 0.9], HITBOX_WIDEN);
                    }
                }
            }
        }

        // Where the server rewinds enemies to judge the watched player's shots: their
        // positions one round trip plus the interpolation delay ago.
        if self.show_ghosts {
            if let Some(shooter) = watched {
                let frame = self.recording.frame_at(self.tick);
                let shooter_frame = *self.recording.player(frame, shooter);
                let rewind_ms = f32::from(shooter_frame.ping) + SCOREBOARD_PING_OFFSET_MS + self.ghost_interp_ms;
                let rewind_ticks = f64::from(rewind_ms / 1000.0) / f64::from(self.recording.tick_interval);
                let past_frame = self.recording.frame_at(self.tick - rewind_ticks);
                for player in boxes.iter().filter(|p| p.team != shooter_frame.team) {
                    let now = Vec3::from(self.blended_player(player.slot).position);
                    let then = *self.recording.player(past_frame, player.slot);
                    if !then.is_alive() {
                        continue;
                    }
                    let shift = Mat4::from_translation(Vec3::from(then.position) - now);
                    for (bone, min, max, _) in &player.boxes {
                        for (from, to) in box_edges(&(shift * *bone), *min, *max) {
                            push_strip(&mut floats, from, to, camera.eye, [0.95, 0.95, 0.95], [0.75, 0.75], HITBOX_WIDEN);
                        }
                    }
                }
            }
        }
        floats
    }

    /// Recent hitscan shots as thin glowing strips that face the camera.
    fn traces(&self, camera: &Camera) -> Vec<f32> {
        let recording = &self.recording;
        let window = TRACE_SECONDS / f64::from(recording.tick_interval);
        let first = recording.shots.partition_point(|shot| f64::from(shot.tick) < self.tick - window);
        let mut floats = Vec::new();
        self.energy_glows(camera, &mut floats);
        if self.show_projectiles {
            self.trail_strips(camera, &mut floats);
        }
        if self.show_impacts {
            self.impact_marks(camera, &mut floats);
        }
        if !self.show_bullets {
            return floats;
        }
        for (shot, end) in recording.shots[first..].iter().zip(&self.shot_ends[first..]) {
            let tick = f64::from(shot.tick);
            if tick > self.tick {
                break;
            }
            let fade = (1.0 - (self.tick - tick) / window) as f32;
            let origin = Vec3::from(shot.origin);
            let aim = forward_from(shot.pitch, shot.yaw);
            // TF2 draws the tracer from the gun's muzzle; the demo only says where the
            // bullet itself came from, the player's eyes.
            let muzzle = self.muzzles.borrow().get(&shot.slot).copied().filter(|muzzle| muzzle.distance(origin) < MUZZLE_REACH);
            let start = muzzle.unwrap_or_else(|| {
                if self.first_person_slot() == Some(shot.slot) {
                    // Through the shooter's own eyes, roughly where the gun is.
                    let right = aim.cross(Vec3::Z).normalize_or_zero();
                    origin + aim * 24.0 + right * 6.0 - Vec3::Z * 6.0
                } else {
                    origin
                }
            });
            let color = match (shot.team, shot.crit) {
                (_, true) => [1.0, 0.85, 0.35],
                (TEAM_RED, false) => [1.0, 0.45, 0.35],
                (TEAM_BLUE, false) => [0.45, 0.75, 1.0],
                _ => [1.0, 1.0, 0.85],
            };
            push_strip(&mut floats, start, *end, camera.eye, color, [0.9 * fade, 0.3 * fade], 1.0);
        }
        // Sentry fire: a line to what it hit, and a flash at the gun.
        let flash = MUZZLE_FLASH_SECONDS / f64::from(recording.tick_interval);
        let first = self.sentry_shots.partition_point(|shot| f64::from(shot.tick) < self.tick - window);
        for shot in &self.sentry_shots[first..] {
            let tick = f64::from(shot.tick);
            if tick > self.tick {
                break;
            }
            let fade = (1.0 - (self.tick - tick) / window) as f32;
            let color = match shot.team {
                TEAM_RED => [1.0, 0.45, 0.35],
                TEAM_BLUE => [0.45, 0.75, 1.0],
                _ => [1.0, 1.0, 0.85],
            };
            let direction = (shot.to - shot.from).normalize_or_zero();
            let muzzle = shot.from + direction * 18.0;
            push_strip(&mut floats, muzzle, shot.to, camera.eye, color, [0.8 * fade, 0.25 * fade], 1.0);
            if self.tick - tick < flash {
                let side = direction.cross(Vec3::Z).normalize_or_zero();
                let up = side.cross(direction);
                for axis in [direction, side, up] {
                    push_strip(&mut floats, muzzle - axis * 5.0, muzzle + axis * 5.0, camera.eye, [1.0, 0.9, 0.5], [1.0, 1.0], 3.0);
                }
            }
        }
        floats
    }

    /// The volume slider and mute switch in the top bar (right to left).
    fn volume_controls(&mut self, ui: &mut egui::Ui) {
        if self.audio.is_none() {
            ui.label(RichText::new("No sound device").color(MUTED))
                .on_hover_text("No sound output was found on this computer, so the demo plays without sound.");
            return;
        }
        let slider = ui
            .add_sized([80.0, 18.0], egui::Slider::new(&mut self.volume, 0.0..=1.0).show_value(false))
            .on_hover_text(format!("Volume {:.0}%", self.volume * 100.0));
        if slider.changed() {
            self.muted = false;
        }
        let label = if self.muted || self.volume <= 0.0 { "Muted" } else { "Sound" };
        if ui
            .selectable_label(self.muted, label)
            .on_hover_text("TF2's sounds and voice chat from the demo. Click or press M to mute. Sound plays at up to 2x speed.")
            .clicked()
        {
            self.muted = !self.muted;
        }
    }

    /// Plays the sounds of the moments just played: what the server sent, explosions,
    /// the recorder's own shots, hit and kill sounds, and voice chat. A pause, a jump
    /// or fast playback stops everything instead.
    fn play_sounds(&mut self) {
        let Some(audio) = &self.audio else { return };
        audio.set_volume(if self.muted { 0.0 } else { self.volume });
        let interval = f64::from(self.recording.tick_interval.max(1e-4));
        let (from, to) = (std::mem::replace(&mut self.sound_tick, self.tick), self.tick);
        let speed = SPEEDS[self.speed_index];
        let jumped = to < from || to - from > MAX_SOUND_STEP_SECONDS / interval;
        let on = self.playing && self.clip.is_none() && speed <= MAX_SOUND_SPEED && !jumped;
        if !on {
            if self.sound_was_on || jumped {
                audio.stop_all();
            }
            self.sound_was_on = false;
            return;
        }
        let resumed = !self.sound_was_on;
        self.sound_was_on = true;

        let camera = self.camera();
        let eye = camera.eye;
        let right = forward_from(camera.pitch, camera.yaw).cross(Vec3::Z).normalize_or_zero();
        let watched = self.watched_slot();
        let first_person = self.first_person_slot();
        let recording = &self.recording;
        let bank = &mut self.sound_bank;
        let between = |tick: u32| f64::from(tick) > from && f64::from(tick) <= to;

        // What the server sent.
        let first = recording.sounds.partition_point(|sound| f64::from(sound.tick) <= from);
        for sound in recording.sounds[first..].iter().take_while(|sound| between(sound.tick)) {
            let Some(name) = recording.sound_names.get(usize::from(sound.sound)) else { continue };
            if name.ends_with("null.wav") || name == "sound/" {
                continue;
            }
            let tag = DEMO_SOUND_TAG | (u64::from(sound.entity) << 16) | u64::from(sound.sound);
            let flags = u32::from(sound.flags);
            if flags & (SND_STOP | SND_STOP_LOOPING) != 0 {
                audio.stop(tag);
                continue;
            }
            let gains = ear_gains(eye, right, Vec3::from(sound.origin), f32::from(sound.level), sound.volume);
            let pitch = if sound.pitch == 0 { 1.0 } else { f32::from(sound.pitch) / 100.0 };
            // A change to a sound already playing, like a cart's engine speeding up.
            if flags & (SND_CHANGE_VOL | SND_CHANGE_PITCH) != 0 && audio.update(tag, gains, pitch) {
                continue;
            }
            if let Some(clip) = bank.clip(name) {
                let looping = clip.loop_start.is_some();
                audio.play(clip, gains, pitch, tag, looping, 0.0);
            }
        }

        let first = recording.explosions.partition_point(|explosion| f64::from(explosion.tick) <= from);
        for explosion in recording.explosions[first..].iter().take_while(|explosion| between(explosion.tick)) {
            let path = EXPLOSION_SOUNDS[explosion.tick as usize % EXPLOSION_SOUNDS.len()];
            if let Some(clip) = bank.clip(path) {
                audio.play(clip, ear_gains(eye, right, Vec3::from(explosion.origin), EXPLOSION_LEVEL, 1.0), 1.0, 0, false, 0.0);
            }
        }

        if let Some(recorder) = recording.recorder_slot {
            let first = self.own_shots.partition_point(|(tick, _)| f64::from(*tick) <= from);
            for (tick, path) in self.own_shots[first..].iter().take_while(|(tick, _)| between(*tick)) {
                let frame = recording.frame_at(f64::from(*tick));
                let player = recording.player(frame, recorder);
                let at = Vec3::from(player.position) + Vec3::Z * player.eye_height();
                let gains = if first_person == Some(recorder) {
                    [std::f32::consts::FRAC_1_SQRT_2; 2]
                } else {
                    ear_gains(eye, right, at, OWN_SHOT_LEVEL, 1.0)
                };
                if let Some(clip) = bank.clip(path) {
                    audio.play(clip, gains, 1.0, 0, false, 0.0);
                }
            }
        }

        if let (Some(slot), true) = (watched, self.hit_sounds) {
            let first = recording.hurts.partition_point(|hurt| f64::from(hurt.tick) <= from);
            let hit = recording.hurts[first..]
                .iter()
                .take_while(|hurt| between(hurt.tick))
                .any(|hurt| hurt.attacker == Some(slot) && hurt.victim != Some(slot));
            let killed = recording
                .kills
                .iter()
                .any(|kill| between(kill.tick) && !kill.feigned && kill.attacker == Some(slot) && kill.victim != Some(slot));
            for (heard, path) in [(hit && !killed, HIT_SOUND), (killed, KILL_SOUND)] {
                if let (true, Some(clip)) = (heard, bank.clip(path)) {
                    audio.play(clip, [DING_VOLUME; 2], 1.0, 0, false, 0.0);
                }
            }
        }

        // Voice chat only keeps in step at normal speed. After a pause or a jump it
        // picks up in the middle of whatever was being said.
        if self.voice_chat && speed == 1.0 {
            for (index, line) in self.voice_lines.iter().enumerate() {
                if line.start > to {
                    break;
                }
                let starting = line.start > from;
                let midway = resumed && line.start <= to && to < line.end;
                if starting || midway {
                    let skip = if starting { 0.0 } else { (to - line.start) * interval };
                    audio.play(Arc::clone(&line.clip), [std::f32::consts::FRAC_1_SQRT_2; 2], 1.0, VOICE_TAG | index as u64, false, skip);
                }
            }
        } else {
            audio.stop_matching(VOICE_TAG, TAG_KIND);
        }
    }

    /// Projectiles TF2 draws as particles, with no model: the Righteous Bison's and
    /// Pomson's ring, the Cow Mangler's and Short Circuit's balls of energy, and the
    /// Dragon's Fury's fireball.
    fn energy_glows(&self, camera: &Camera, floats: &mut Vec<f32>) {
        let recording = &self.recording;
        let (frame, next, t) = self.frames_around();
        let upcoming: HashMap<u16, &ObjectFrame> = recording
            .objects_at(next)
            .iter()
            .filter(|object| object.kind.is_energy())
            .map(|object| (object.id, object))
            .collect();
        for object in recording.objects_at(frame) {
            if !object.kind.is_energy() {
                continue;
            }
            let mut position = Vec3::from(object.position);
            let mut heading = forward_from(object.pitch, object.yaw);
            if let Some(after) = upcoming.get(&object.id) {
                let target = Vec3::from(after.position);
                let step = position.distance(target);
                if step < MAX_STEP_PER_FRAME * 2.0 {
                    if step > 1.0 {
                        heading = (target - position) / step;
                    }
                    position = position.lerp(target, t);
                }
            }
            match object.kind {
                ObjectKind::EnergyRing => {
                    let color = energy_color(object.team);
                    push_ring(floats, position, heading, ENERGY_RING_RADIUS, color, 0.9);
                    push_glow(floats, position, ENERGY_RING_RADIUS * 0.5, camera.eye, color, 0.35);
                }
                ObjectKind::Fireball => {
                    push_glow(floats, position, FIREBALL_RADIUS, camera.eye, FIRE_COLOR, 0.55);
                    push_glow(floats, position, FIREBALL_RADIUS * 0.45, camera.eye, [1.0, 0.85, 0.5], 0.8);
                }
                kind => {
                    let radius = if kind == ObjectKind::Orb { ORB_RADIUS } else { ENERGY_BALL_RADIUS };
                    push_glow(floats, position, radius, camera.eye, energy_color(object.team), 0.6);
                    push_glow(floats, position, radius * 0.4, camera.eye, [1.0, 1.0, 1.0], 0.7);
                }
            }
        }
    }

    /// The last second of each projectile's flight, fading towards its oldest part.
    /// Once it explodes or lands, what's left fades away too.
    fn trail_strips(&self, camera: &Camera, floats: &mut Vec<f32>) {
        let interval = f64::from(self.recording.tick_interval);
        let window = TRAIL_SECONDS / interval;
        let last_started = self.trails.partition_point(|trail| f64::from(trail.first_tick()) <= self.tick);
        for trail in &self.trails[..last_started] {
            let ended = f64::from(trail.last_tick());
            if ended + window < self.tick {
                continue;
            }
            let color = match (trail.kind, trail.team, trail.crit) {
                (ObjectKind::Fireball, _, _) => FIRE_COLOR,
                (kind, team, _) if kind.is_energy() => energy_color(team),
                (_, _, true) => [1.0, 0.85, 0.35],
                (_, TEAM_RED, false) => [1.0, 0.5, 0.3],
                (_, TEAM_BLUE, false) => [0.45, 0.75, 1.0],
                _ => [1.0, 1.0, 0.85],
            };
            let strength = |tick: f64| {
                let age = (self.tick - tick) / window;
                (0.75 * (1.0 - age)).clamp(0.0, 0.75) as f32
            };

            // Points up to now, plus the projectile's in-between position this frame.
            let shown = trail.ticks.partition_point(|&tick| f64::from(tick) <= self.tick);
            let mut path: Vec<(f64, Vec3)> = trail.ticks[..shown]
                .iter()
                .zip(&trail.points[..shown])
                .map(|(&tick, &point)| (f64::from(tick), point))
                .filter(|(tick, _)| *tick >= self.tick - window)
                .collect();
            if let (Some(&(before_tick, before)), Some(&after_tick)) = (path.last(), trail.ticks.get(shown)) {
                let after = trail.points[shown];
                let t = ((self.tick - before_tick) / (f64::from(after_tick) - before_tick)).clamp(0.0, 1.0) as f32;
                path.push((self.tick, before.lerp(after, t)));
            }
            for pair in path.windows(2) {
                let ((older_tick, older), (newer_tick, newer)) = (pair[0], pair[1]);
                if older.distance(newer) > MAX_STEP_PER_FRAME * 4.0 {
                    continue;
                }
                // Right next to the camera a trail would cross the whole screen, like the
                // start of a shot just fired by the player we're watching.
                for (from, to) in outside_sphere(older, newer, camera.eye, TRAIL_CLEAR_OF_EYE).into_iter().flatten() {
                    push_strip(floats, from, to, camera.eye, color, [strength(older_tick), strength(newer_tick)], TRAIL_WIDEN);
                }
            }
        }
    }

    /// Players who died in the last few seconds: TF2's own death animation for
    /// headshots, backstabs and burning, otherwise a ragdoll falling from where they died.
    fn bodies(&self) -> Vec<Actor> {
        let recording = &self.recording;
        let window = BODY_SECONDS / f64::from(recording.tick_interval);
        let mut actors = Vec::new();
        for (index, kill) in recording.kills.iter().enumerate() {
            let tick = f64::from(kill.tick);
            if kill.feigned || tick > self.tick || tick < self.tick - window {
                continue;
            }
            let Some(slot) = kill.victim.filter(|slot| Some(*slot) != self.first_person_slot()) else { continue };
            // Where they were standing just before the kill.
            let frame = recording.frame_at(tick).saturating_sub(1);
            let body = *recording.player(frame, slot);
            if body.is_out_of_view() || !body.is_on_a_team() || recording.player(recording.frame_at(self.tick), slot).is_alive() {
                continue;
            }
            let Some(class) = self.characters.classes.get(usize::from(body.class)).and_then(Option::as_ref) else {
                continue;
            };
            let special = if kill.backstab {
                Some(DeathKind::Backstab)
            } else if kill.headshot {
                Some(DeathKind::Headshot)
            } else if kill.weapon.contains("flame") || kill.weapon.contains("burn") {
                Some(DeathKind::Burning)
            } else {
                None
            };
            let seconds = recording.seconds_between(tick, self.tick) as f32;
            let position = Vec3::from(body.position);
            let root = Mat4::from_translation(position) * Mat4::from_rotation_z(body.yaw.to_radians());
            let skeleton = &class.rig.skeleton;
            let ragdoll = if special.is_none() { self.ragdoll(index, kill, slot, &body, &class.rig) } else { None };
            let world = match ragdoll {
                Some(ragdoll) => ragdoll.world_matrices(skeleton, seconds),
                None => {
                    let kind = special.unwrap_or(DeathKind::Headshot);
                    let Some(pose) = class.rig.death_pose(kind, seconds) else { continue };
                    skeleton.world_matrices(&pose, root)
                }
            };
            // Light the body where it is now, which may be far from where it died.
            let pelvis = skeleton.bone_index("bip_pelvis").unwrap_or(0);
            let resting = world.get(pelvis).map_or(position, |matrix| matrix.w_axis.truncate());
            let lighting = self.lighting.at(resting + Vec3::Z * 12.0);
            actors.push(Actor {
                mesh: class.mesh,
                skin: usize::from(body.team == TEAM_BLUE),
                body: 0,
                bones: skeleton.skinning_matrices(&world),
                cube: lighting.cube,
                sun: lighting.sun,
                glow: None,
            });
        }
        actors
    }

    /// First-person arms and weapon for whoever is being watched. The recording
    /// player's come straight from the demo. Everyone else's are rebuilt from their
    /// class's arms, playing the idle for the weapon slot they have out.
    fn view_model(&self, camera: &Camera) -> Vec<Actor> {
        let Some(slot) = self.first_person_slot() else { return Vec::new() };
        let frame = self.recording.frame_at(self.tick);
        let player = self.recording.player(frame, slot);
        if !player.is_alive() || player.is_zoomed() || player.is_out_of_view() {
            return Vec::new();
        }
        let characters = &self.characters;
        let view = self.recording.views[frame];

        let recorded = (Some(slot) == self.recording.recorder_slot)
            .then(|| characters.by_model.get(usize::from(view.arms)).copied().flatten())
            .flatten()
            .map(|arms| {
                let seconds = self.recording.seconds_between(f64::from(view.sequence_start), self.tick) as f32;
                (arms, usize::from(view.sequence), seconds)
            });
        let rebuilt = || {
            let class = characters.classes.get(usize::from(player.class))?.as_ref()?;
            let arms = class.arms?;
            let animations = characters.animations[arms].as_ref()?;
            let sets = first_person_sets(self.weapon_pose(player), player.weapon_slot);
            // Drawing, firing, swinging or reloading, from the demo's events.
            if let Some((sequence, seconds)) = self.first_person_action(slot, player, animations, &characters.skeletons[arms], &sets) {
                return Some((arms, sequence, seconds));
            }
            let mut activities: Vec<String> = Vec::new();
            // A revved minigun spins.
            if player.is_aiming() && player.class == CLASS_HEAVY && sets[0] == WeaponPose::Primary {
                activities.push("ACT_PRIMARY_VM_SECONDARYATTACK".to_string());
            }
            for (index, set) in sets.iter().enumerate() {
                activities.push(format!("ACT_{}_VM_IDLE", set.suffix()));
                // Sappers and PDAs use the plain idle when they have no named one.
                if index == 0 && matches!(set, WeaponPose::Building | WeaponPose::Pda) {
                    activities.push("ACT_VM_IDLE".to_string());
                }
            }
            activities.push("ACT_MELEE_ALLCLASS_VM_IDLE".to_string());
            let sequence = activities.iter().find_map(|activity| animations.find_activity(activity))?;
            let seconds = self
                .recording
                .seconds_between(f64::from(self.recording.first_tick()), self.tick) as f32;
            Some((arms, sequence, seconds))
        };
        let Some((arms, sequence, seconds)) = recorded.or_else(rebuilt) else {
            return Vec::new();
        };

        let skeleton = &characters.skeletons[arms];
        let pose = characters.animations[arms]
            .as_ref()
            .and_then(|animations| animations.pose(skeleton, sequence, seconds))
            .unwrap_or_else(|| characters.idle_poses[arms].clone());
        let sway = self.view_model_sway(slot, camera);
        let root = Mat4::from_translation(camera.eye + sway) * rotation_from_angles(camera.pitch, camera.yaw, camera.roll);
        let mut world = skeleton.world_matrices(&pose, root);
        self.fix_grip(skeleton, root, &mut world);
        let lighting = self.lighting.at(camera.eye);
        let skin = usize::from(player.team == TEAM_BLUE);

        let mut actors = vec![Actor {
            mesh: arms,
            skin,
            body: 0,
            bones: skeleton.skinning_matrices(&world),
            cube: lighting.cube,
            sun: lighting.sun,
            glow: None,
        }];
        // A weapon the demo doesn't place in a slot, like the Spy's disguise kit,
        // has no first-person pose to sit in, so only the hands are shown.
        let known_slot = player.weapon_slot != super::recording::WEAPON_UNKNOWN;
        let weapon = characters.by_model.get(usize::from(player.weapon_model)).copied().flatten();
        // Only things held in the hands belong in first person. Worn items like the
        // Pyro's jetpack have no weapon bone and would sit right over the camera.
        let held = weapon
            .filter(|_| known_slot || recorded.is_some())
            .filter(|&weapon| characters.skeletons[weapon].bone_index("weapon_bone").is_some());
        if let Some(weapon) = held {
            let weapon_skeleton = &characters.skeletons[weapon];
            let weapon_world = weapon_skeleton.attached_world(skeleton, &world, root);
            if let Some(muzzle) = weapon_skeleton.attachment_point("muzzle", &weapon_world) {
                self.muzzles.borrow_mut().insert(slot, muzzle);
            }
            actors.push(Actor {
                mesh: weapon,
                skin,
                body: 0,
                bones: weapon_skeleton.skinning_matrices(&weapon_world),
                cube: lighting.cube,
                sun: lighting.sun,
                glow: None,
            });
        }
        actors
    }

    /// What the watched player's hands are doing, from the demo's events: drawing the
    /// weapon they just took out, firing, swinging or reloading. None when idle.
    fn first_person_action(
        &self,
        slot: usize,
        player: &PlayerFrame,
        animations: &super::skeleton::Animations,
        skeleton: &super::skeleton::Skeleton,
        sets: &[WeaponPose],
    ) -> Option<(usize, f32)> {
        let gesture = self.gesture_at(slot);
        let drawn = self.drawn_seconds(slot, player);
        let (names, seconds): (Vec<String>, f32) = match (gesture, drawn) {
            (Some((_, since)), Some(drawn)) if drawn < since => (draw_names(sets), drawn),
            (Some((gesture, since)), _) => (sets.iter().flat_map(|&set| first_person_actions(gesture, set)).collect(), since),
            (None, Some(drawn)) => (draw_names(sets), drawn),
            (None, None) => return None,
        };
        let sequence = names.iter().find_map(|name| animations.find_activity(name))?;
        let length = animations.sequence_seconds(skeleton, sequence)?;
        (seconds < length).then_some((sequence, seconds))
    }

    /// Seconds since the watched player took out the weapon they're holding, if
    /// recently enough for its draw animation to still be playing.
    fn drawn_seconds(&self, slot: usize, player: &PlayerFrame) -> Option<f32> {
        let recording = &self.recording;
        let interval = f64::from(recording.tick_interval);
        let oldest = self.tick - DRAW_LOOKBACK_SECONDS / interval;
        let mut frame = recording.frame_at(self.tick);
        while frame > 0 && f64::from(recording.ticks[frame]) >= oldest {
            let before = recording.player(frame - 1, slot);
            if !before.is_alive() || before.weapon_model != player.weapon_model {
                return Some(((self.tick - f64::from(recording.ticks[frame])) * interval).max(0.0) as f32);
            }
            frame -= 1;
        }
        None
    }

    /// Where a player was looking in a frame: the recorder's own view, or their angles.
    fn view_angles(&self, frame: usize, slot: usize) -> (f32, f32) {
        if Some(slot) == self.recording.recorder_slot {
            let view = self.recording.views[frame];
            (view.pitch, view.yaw)
        } else {
            let player = self.recording.player(frame, slot);
            (normalize_pitch(player.pitch), player.yaw)
        }
    }

    /// How far TF2 shifts the first-person weapon: it trails behind quick turns and
    /// tucks in when looking up or down. TF2 works this out every frame from where the
    /// view pointed, so it's replayed here from the demo's view angles.
    fn view_model_sway(&self, slot: usize, camera: &Camera) -> Vec3 {
        let recording = &self.recording;
        let interval = f64::from(recording.tick_interval);
        let now = recording.frame_at(self.tick);
        let first = recording.frame_at(self.tick - SWAY_HISTORY_SECONDS / interval);
        let facing = |frame: usize| {
            let (pitch, yaw) = self.view_angles(frame, slot);
            forward_from(pitch, yaw)
        };
        // TF2 eases where the weapon faces toward where the view faces, faster when
        // it has fallen far behind.
        let follow = |last: Vec3, forward: Vec3, seconds: f32| {
            let difference = forward - last;
            let behind = difference.length();
            let speed = if behind > MAX_SWAY { 5.0 * behind / MAX_SWAY } else { 5.0 };
            (last + difference * (speed * seconds).min(1.0)).normalize_or_zero()
        };
        let mut last = facing(first);
        let mut time = f64::from(recording.ticks[first]);
        for frame in first + 1..=now {
            let tick = f64::from(recording.ticks[frame]);
            last = follow(last, facing(frame), ((tick - time) * interval) as f32);
            time = tick;
        }
        let forward = forward_from(camera.pitch, camera.yaw);
        let behind = forward - last;
        let yaw = camera.yaw.to_radians();
        let right = Vec3::new(yaw.sin(), -yaw.cos(), 0.0);
        let up = right.cross(forward);
        let pitch = normalize_pitch(camera.pitch);
        -behind * 5.0 - forward * (pitch * 0.035) - right * (pitch * 0.03) - up * (pitch * 0.02)
    }

    /// Every sentry shot as (tick, building id, gun position), for checking by hand.
    pub fn sentry_shot_list(&self) -> Vec<(u32, u16, Vec3)> {
        self.sentry_shots.iter().map(|shot| (shot.tick, shot.id, shot.from)).collect()
    }

    /// How far a detection's aim flicked, measured from the demo: the angle between
    /// where the player looked a moment before and the target's head.
    fn measured_flick(&self, marker: &Marker) -> Option<f32> {
        let target = marker.data.get("target")?.as_u64()?;
        let recording = &self.recording;
        let tick = f64::from(marker.tick);
        let shooter = recording.slot_of_steam_id(marker.steam_id, tick)?;
        let victim = recording.slot_of_steam_id(target, tick)?;
        let before = recording.frame_at(tick - FLICK_LOOKBACK_SECONDS / f64::from(recording.tick_interval));
        let shooter_body = recording.player(before, shooter);
        let victim_body = recording.player(recording.frame_at(tick), victim);
        if !shooter_body.is_alive() || !victim_body.is_alive() {
            return None;
        }
        let (pitch, yaw) = self.view_angles(before, shooter);
        let eye = Vec3::from(shooter_body.position) + Vec3::Z * shooter_body.eye_height();
        let head = Vec3::from(victim_body.position) + Vec3::Z * victim_body.eye_height();
        let toward = (head - eye).try_normalize()?;
        Some(forward_from(pitch, yaw).dot(toward).clamp(-1.0, 1.0).acos().to_degrees())
    }

    /// A player's estimated aimbot FOV: degrees, and how many flicks it's based on.
    pub fn aimbot_fov(&self, steam_id: u64) -> Option<(f32, usize)> {
        self.aimbot_fovs.get(&steam_id).copied()
    }

    /// The watched player's estimated aimbot FOV as a circle around the crosshair:
    /// everything inside is within reach of their aimbot.
    fn draw_fov_circle(&self, ui: &egui::Ui, rect: Rect) {
        let Some(slot) = self.first_person_slot() else { return };
        let player = self.recording.player(self.recording.frame_at(self.tick), slot);
        if !player.is_alive() || player.is_out_of_view() {
            return;
        }
        let steam_id = self.recording.steam_id_at(slot, self.tick);
        let Some((degrees, _)) = self.aimbot_fov(steam_id) else { return };
        // The same projection the view uses: `fov` is measured across a 4:3 screen.
        let fov = if self.scoped() { SCOPED_FOV } else { self.fov };
        let half_height = (fov.to_radians() * 0.5).tan() * 0.75;
        let radius = degrees.to_radians().tan() / half_height * rect.height() * 0.5;
        let painter = ui.painter_at(rect);
        let center = rect.center();
        painter.circle(
            center,
            radius,
            Color32::from_rgba_unmultiplied(255, 200, 60, 14),
            Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(255, 200, 60, 190)),
        );
        let corner = center + Vec2::new(radius, -radius) * std::f32::consts::FRAC_1_SQRT_2 + Vec2::new(6.0, -6.0);
        painter.text(
            corner,
            Align2::LEFT_BOTTOM,
            format!("aimbot FOV \u{2265} {degrees:.1}\u{b0}"),
            FontId::proportional(13.0),
            Color32::from_rgb(255, 210, 90),
        );
    }

    /// Whether a sentry fired in the last moment.
    fn sentry_firing(&self, id: u16) -> bool {
        let window = SENTRY_FIRING_SECONDS / f64::from(self.recording.tick_interval);
        let end = self.sentry_shots.partition_point(|shot| f64::from(shot.tick) <= self.tick);
        self.sentry_shots[..end]
            .iter()
            .rev()
            .take_while(|shot| f64::from(shot.tick) >= self.tick - window)
            .any(|shot| shot.id == id)
    }

    /// The Heavy's minigun idle leaves the weapon bone far from the hand and pointing
    /// up, so the gun would stand on end in the middle of the view. When the weapon
    /// bone is that far from the hand, the gun is put where TF2 shows the minigun:
    /// low on the right, barrel forward.
    fn fix_grip(&self, arms: &super::skeleton::Skeleton, view_root: Mat4, world: &mut [Mat4]) {
        let (Some(hand), Some(weapon)) = (bone_named(arms, "bip_hand_r"), arms.bone_index("weapon_bone")) else {
            return;
        };
        let held = world[hand].w_axis.truncate().distance(world[weapon].w_axis.truncate());
        if held <= GRIP_DISTANCE {
            return;
        }
        // In view space: x forward, y left, z up. The gun's barrel runs along its z axis
        // and points a little towards the middle of the view.
        let barrel = Vec3::new(1.0, 0.16, 0.06).normalize();
        let side = Vec3::Z.cross(barrel).normalize();
        let up = barrel.cross(side);
        let placement = Mat4::from_cols(side.extend(0.0), up.extend(0.0), barrel.extend(0.0), HELD_GUN_OFFSET.extend(1.0));
        world[weapon] = view_root * placement;
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let tick_seconds = f64::from(self.recording.tick_interval);
        let (space, left, right, shift, comma, period, slower, faster, traces, glow, aim, hitboxes) = ctx.input(|input| {
            (
                input.key_pressed(Key::Space),
                input.key_pressed(Key::ArrowLeft),
                input.key_pressed(Key::ArrowRight),
                input.modifiers.shift,
                input.key_pressed(Key::Comma),
                input.key_pressed(Key::Period),
                input.key_pressed(Key::OpenBracket),
                input.key_pressed(Key::CloseBracket),
                input.key_pressed(Key::T),
                input.key_pressed(Key::G),
                input.key_pressed(Key::A),
                input.key_pressed(Key::H),
            )
        });
        if ctx.input(|input| input.key_pressed(Key::Escape)) {
            self.stop_review();
        }
        if ctx.input(|input| input.key_pressed(Key::M)) {
            self.muted = !self.muted;
        }
        if ctx.input(|input| input.key_pressed(Key::C)) {
            self.third_person = !self.third_person;
            self.orbit = None;
        }
        // A is also strafe left in the free camera, so it only toggles aim lines when watching someone.
        if aim && self.camera_mode != CameraMode::Free {
            self.show_aim_lines = !self.show_aim_lines;
        }
        if hitboxes {
            self.show_hitboxes = !self.show_hitboxes;
        }
        if glow {
            self.glow = self.glow.next();
        }
        if space {
            self.playing = !self.playing;
        }
        if traces {
            let on = !(self.show_bullets || self.show_projectiles);
            self.show_bullets = on;
            self.show_projectiles = on;
        }
        let skip = if shift { LONG_SKIP_SECONDS } else { SKIP_SECONDS } / tick_seconds;
        if left {
            self.seek(self.tick - skip);
        }
        if right {
            self.seek(self.tick + skip);
        }
        if comma || period {
            self.playing = false;
            let frame = self.recording.frame_at(self.tick);
            let target = if period {
                (frame + 1).min(self.recording.ticks.len() - 1)
            } else {
                frame.saturating_sub(1)
            };
            self.tick = f64::from(self.recording.ticks[target]);
        }
        if slower {
            self.speed_index = self.speed_index.saturating_sub(1);
        }
        if faster {
            self.speed_index = (self.speed_index + 1).min(SPEEDS.len() - 1);
        }
    }

    fn fly(&mut self, ctx: &egui::Context, seconds: f32) {
        if self.camera_mode != CameraMode::Free || ctx.wants_keyboard_input() {
            return;
        }
        let (forward, side, up, fast) = ctx.input(|input| {
            let axis = |plus: Key, minus: Key| {
                f32::from(u8::from(input.key_down(plus))) - f32::from(u8::from(input.key_down(minus)))
            };
            (
                axis(Key::W, Key::S),
                axis(Key::D, Key::A),
                axis(Key::E, Key::Q),
                input.modifiers.shift,
            )
        });
        if forward == 0.0 && side == 0.0 && up == 0.0 {
            return;
        }
        let look = forward_from(self.free_camera.pitch, self.free_camera.yaw);
        let right = look.cross(Vec3::Z).normalize_or_zero();
        let speed = FREE_CAMERA_SPEED * if fast { 3.0 } else { 1.0 } * seconds;
        self.free_camera.position += (look * forward + right * side + Vec3::Z * up) * speed;
        ctx.request_repaint();
    }

    fn advance(&mut self, ctx: &egui::Context) {
        let seconds = ctx.input(|input| input.stable_dt).min(0.1);
        self.handle_keys(ctx);
        self.fly(ctx, seconds);
        if self.playing {
            let ticks = f64::from(seconds) / f64::from(self.recording.tick_interval);
            self.tick += ticks * f64::from(SPEEDS[self.speed_index]);
            let last = f64::from(self.recording.last_tick());
            if self.tick >= last {
                self.tick = last;
                self.playing = false;
            }
            ctx.request_repaint();
        }
    }

    /// The right-click menu on a player in the list.
    fn player_menu(&mut self, ui: &mut egui::Ui, slot: usize, steam_id: u64, name: &str) {
        let human = steam_id != 0;
        if ui.button("Watch through their eyes").clicked() {
            self.camera_mode = CameraMode::Eyes(slot);
            ui.close_menu();
        }
        let detections = if human { self.markers.iter().filter(|marker| marker.steam_id == steam_id).count() } else { 0 };
        let review = ui
            .add_enabled(detections > 0, egui::Button::new(format!("Review their detections ({detections})")))
            .on_hover_text("Plays each detection in slow motion from their eyes, with what the check measured")
            .on_disabled_hover_text("No detections for this player");
        if review.clicked() {
            self.start_review(steam_id);
            ui.close_menu();
        }
        let mark = ui
            .add_enabled(human && self.can_mark, egui::Button::new("Mark in TF2BD…"))
            .on_disabled_hover_text("Turn on TF2BD in the main window to mark players");
        if mark.clicked() {
            self.requests.push(PlayerRequest::Mark { steam_id, name: name.to_string() });
            ui.close_menu();
        }
        if ui.add_enabled(human, egui::Button::new("Export report")).clicked() {
            self.export_report(slot, steam_id, name);
            ui.close_menu();
        }
        ui.separator();
        if ui.add_enabled(human, egui::Button::new("Open Steam profile")).clicked() {
            super::report::open_externally(&format!("https://steamcommunity.com/profiles/{steam_id}"));
            ui.close_menu();
        }
        if ui.add_enabled(human, egui::Button::new("Copy SteamID64")).clicked() {
            ui.ctx().copy_text(steam_id.to_string());
            ui.close_menu();
        }
    }

    /// Writes a report for a Steam account without opening it. Returns where it went.
    pub fn save_report(&mut self, steam_id: u64) -> Option<String> {
        let slot = self.recording.slot_of_steam_id(steam_id, self.tick)?;
        let name = self.recording.name_at(slot, self.tick).to_string();
        self.export_report_to(slot, steam_id, &name, false);
        self.report_status.as_ref().filter(|(_, ok, _)| *ok).map(|(message, _, _)| message.clone())
    }

    /// Writes this player's report next to the program and opens it in the browser.
    fn export_report(&mut self, slot: usize, steam_id: u64, name: &str) {
        self.export_report_to(slot, steam_id, name, true);
    }

    fn export_report_to(&mut self, slot: usize, steam_id: u64, name: &str, open: bool) {
        let fov = self.aimbot_fov(steam_id);
        let html = super::report::render(&self.recording, &self.markers, slot, steam_id, &self.demo, self.tick, fov);
        let folder = super::report::reports_folder();
        let demo_stem = self.demo.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let file = folder.join(format!(
            "{}_{}_{}.html",
            super::report::safe_file_name(&demo_stem),
            super::report::safe_file_name(name),
            steam_id
        ));
        let result = std::fs::create_dir_all(&folder).and_then(|_| std::fs::write(&file, html));
        let now = self.tick;
        self.report_status = Some(match result {
            Ok(()) => {
                if open {
                    super::report::open_externally(&file.to_string_lossy());
                }
                (format!("Report saved to {}", file.display()), true, now)
            }
            Err(error) => (format!("Couldn't save the report: {error}"), false, now),
        });
    }

    /// Everything the view can draw on top of the match, with its shortcut.
    fn show_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(230.0);
        ui.checkbox(&mut self.show_names, "Names");
        ui.separator();
        ui.label(RichText::new("Traces (T)").small().color(MUTED));
        ui.checkbox(&mut self.show_bullets, "Bullets").on_hover_text("Lines showing where every hitscan shot went");
        ui.checkbox(&mut self.show_projectiles, "Projectile trails")
            .on_hover_text("Trails behind rockets, pipes, stickies, flares, arrows and other projectiles");
        ui.checkbox(&mut self.show_impacts, "Bullet impacts").on_hover_text("A mark where each bullet hit");
        ui.checkbox(&mut self.show_damage, "Damage numbers").on_hover_text(
            "How much damage each hit did, over the player who took it. When watching someone, only their own hits.",
        );
        ui.separator();
        ui.label(RichText::new("Aim").small().color(MUTED));
        ui.checkbox(&mut self.show_aim_lines, "Aim lines (A)")
            .on_hover_text("A line from every player's eyes to where they look, through walls too");
        ui.checkbox(&mut self.show_aim_graph, "Aim graph")
            .on_hover_text("How fast the watched player's view turns, under the timeline. Snaps show as spikes.");
        ui.checkbox(&mut self.show_fov_circle, "Aimbot FOV circle").on_hover_text(
            "When watching a player with aim detections: a circle around the crosshair as big as the biggest \
             flick onto a target among them. Their aimbot reaches at least that far from the crosshair.",
        );
        ui.separator();
        ui.label(RichText::new("Hitboxes").small().color(MUTED));
        ui.checkbox(&mut self.show_hitboxes, "Hitboxes (H)")
            .on_hover_text("The boxes bullets are tested against. The head is yellow.");
        ui.checkbox(&mut self.show_ghosts, "Backtrack ghosts").on_hover_text(
            "When watching someone: where the server puts enemies to judge their shots, one round trip \
             (scoreboard ping + 30 ms) plus the interpolation delay ago. Hits on a ghost far from the real \
             player are what backtrack cheats abuse.",
        );
        ui.add_enabled(
            self.show_ghosts,
            egui::Slider::new(&mut self.ghost_interp_ms, 0.0..=200.0).text("interp ms").integer(),
        )
        .on_hover_text("cl_interp of the watched player: 100 ms is TF2's default, 15 ms is common in competitive configs");
        ui.separator();
        ui.label(RichText::new("Sound").small().color(MUTED));
        ui.checkbox(&mut self.hit_sounds, "Hit and kill sounds")
            .on_hover_text("TF2's hit and kill sounds when the player you're watching hits or kills someone");
        ui.checkbox(&mut self.voice_chat, "Voice chat")
            .on_hover_text("Players talking on voice chat, at normal speed. Who's talking shows at the bottom right.");
    }

    fn view_options(&self) -> ViewOptions {
        // Review mode turns some things on for itself; remember the viewer's own choices.
        let (aim_lines, aim_graph, ghosts) = match &self.review {
            Some(review) => (review.before.show_aim_lines, review.before.show_aim_graph, review.before.show_ghosts),
            None => (self.show_aim_lines, self.show_aim_graph, self.show_ghosts),
        };
        ViewOptions {
            fov: self.fov,
            names: self.show_names,
            bullets: self.show_bullets,
            projectiles: self.show_projectiles,
            glow: self.glow,
            aim_lines,
            hitboxes: self.show_hitboxes,
            ghosts,
            ghost_interp_ms: self.ghost_interp_ms,
            damage: self.show_damage,
            impacts: self.show_impacts,
            aim_graph,
            fov_circle: self.show_fov_circle,
            volume: self.volume,
            muted: self.muted,
            hit_sounds: self.hit_sounds,
            voice_chat: self.voice_chat,
            third_person: self.third_person,
            chase_distance: self.chase_distance,
        }
    }

    fn set_view_options(&mut self, options: ViewOptions) {
        self.fov = options.fov;
        self.show_names = options.names;
        self.show_bullets = options.bullets;
        self.show_projectiles = options.projectiles;
        self.glow = options.glow;
        self.show_aim_lines = options.aim_lines;
        self.show_hitboxes = options.hitboxes;
        self.show_ghosts = options.ghosts;
        self.ghost_interp_ms = options.ghost_interp_ms;
        self.show_damage = options.damage;
        self.show_impacts = options.impacts;
        self.show_aim_graph = options.aim_graph;
        self.show_fov_circle = options.fov_circle;
        self.volume = options.volume;
        self.muted = options.muted;
        self.hit_sounds = options.hit_sounds;
        self.voice_chat = options.voice_chat;
        self.third_person = options.third_person;
        self.chase_distance = options.chase_distance;
    }

    /// Steps a clip recording forward: shows the next moment, then keeps the
    /// screenshot of it when it arrives a frame later.
    fn drive_clip(&mut self, ctx: &egui::Context) {
        if let Some(writer) = &mut self.encoding {
            match writer.result() {
                Some(Ok(path)) => {
                    self.clip_status = Some((format!("Clip saved: {}", path.display()), true));
                    self.encoding = None;
                }
                Some(Err(error)) => {
                    self.clip_status = Some((format!("Couldn't save the clip: {error}"), false));
                    self.encoding = None;
                }
                None => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
            }
        }
        let Some(mut clip) = self.clip.take() else { return };
        self.playing = false;
        ctx.request_repaint();
        if clip.waiting {
            let shot = ctx.input(|input| {
                input.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            let Some(image) = shot else {
                self.clip = Some(clip);
                return;
            };
            let scale = ctx.pixels_per_point();
            let [image_width, image_height] = image.size;
            let left = ((self.view_rect.left() * scale).round().max(0.0) as usize).min(image_width);
            let top = ((self.view_rect.top() * scale).round().max(0.0) as usize).min(image_height);
            let right = ((self.view_rect.right() * scale).round() as usize).clamp(left, image_width);
            let bottom = ((self.view_rect.bottom() * scale).round() as usize).clamp(top, image_height);
            let (width, height) = (right - left, bottom - top);
            let mut rgba = Vec::with_capacity(width * height * 4);
            for y in top..bottom {
                for pixel in &image.pixels[y * image_width + left..y * image_width + right] {
                    rgba.extend_from_slice(&pixel.to_array());
                }
            }
            clip.writer.push(super::clip::ClipFrame { width: width as u32, height: height as u32, rgba });
            clip.frames += 1;
            clip.waiting = false;
            clip.next += clip.step;
            if clip.next > clip.end {
                clip.writer.close();
                self.encoding = Some(clip.writer);
                self.clip_status = Some(("Saving clip…".to_string(), true));
                return;
            }
        }
        self.tick = clip.next;
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        clip.waiting = true;
        self.clip = Some(clip);
    }

    /// Starts a clip from outside the player (the preview tool uses this).
    pub fn record_clip(&mut self, seconds: f64, mp4: bool) {
        self.clip_seconds = seconds;
        let format = if mp4 { super::clip::ClipFormat::Mp4 } else { super::clip::ClipFormat::Gif };
        self.start_clip(format);
    }

    /// The outcome of the last clip, once it's fully saved.
    pub fn clip_outcome(&self) -> Option<(String, bool)> {
        if self.clip.is_some() || self.encoding.is_some() {
            return None;
        }
        self.clip_status.clone()
    }

    /// Starts recording the chosen length of match around the current moment.
    fn start_clip(&mut self, format: super::clip::ClipFormat) {
        let interval = f64::from(self.recording.tick_interval);
        let half = self.clip_seconds / 2.0 / interval;
        let first = f64::from(self.recording.first_tick());
        let last = f64::from(self.recording.last_tick());
        let start = (self.tick - half).max(first);
        let end = (self.tick + half).min(last);
        let step = 1.0 / f64::from(format.fps()) / interval;
        let folder = super::clip::clips_folder();
        if let Err(error) = std::fs::create_dir_all(&folder) {
            self.clip_status = Some((format!("Couldn't make the clips folder: {error}"), false));
            return;
        }
        let demo = self.demo.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let path = folder.join(format!(
            "{}_tick{}_{}s.{}",
            super::report::safe_file_name(&demo),
            self.tick as u32,
            self.clip_seconds as u32,
            format.extension()
        ));
        self.clip = Some(ClipRecording {
            next: start,
            end,
            step,
            waiting: false,
            frames: 0,
            total: ((end - start) / step) as usize + 1,
            writer: super::clip::ClipWriter::start(path, format),
        });
        self.clip_status = None;
    }

    fn clip_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(clip) = &self.clip {
            ui.label(RichText::new(format!("Recording clip {}/{}", clip.frames, clip.total)).color(AMBER));
            if ui.button("Cancel").clicked() {
                if let Some(mut clip) = self.clip.take() {
                    clip.writer.close();
                }
                self.clip_status = Some(("Clip cancelled".to_string(), false));
            }
            return;
        }
        let busy = self.encoding.is_some();
        ui.add_enabled_ui(!busy, |ui| {
            ui.menu_button("Save clip…", |ui| {
                ui.label(RichText::new("Records the moments around now, with everything on screen.").small().color(MUTED));
                ui.horizontal(|ui| {
                    for seconds in CLIP_LENGTHS {
                        ui.selectable_value(&mut self.clip_seconds, seconds, format!("{seconds:.0} s"));
                    }
                });
                let video = super::clip::ffmpeg_available();
                if video && ui.button("Save as MP4").clicked() {
                    self.start_clip(super::clip::ClipFormat::Mp4);
                    ui.close_menu();
                }
                if ui.button("Save as GIF").clicked() {
                    self.start_clip(super::clip::ClipFormat::Gif);
                    ui.close_menu();
                }
                if !video {
                    ui.label(RichText::new("Install ffmpeg to save MP4 videos.").small().color(MUTED));
                }
            });
        });
        if let Some((message, ok)) = &self.clip_status {
            let color = if *ok { MUTED } else { Color32::from_rgb(235, 105, 90) };
            ui.label(RichText::new(message).small().color(color));
            if *ok && !busy && ui.small_button("Open folder").clicked() {
                super::report::open_externally(&super::clip::clips_folder().to_string_lossy());
            }
        }
    }

    fn show(&mut self, ctx: &egui::Context) -> bool {
        self.drive_clip(ctx);
        if !self.options_restored {
            self.options_restored = true;
            if let Some(options) = ctx.data(|data| data.get_temp::<ViewOptions>(view_options_id())) {
                self.set_view_options(options);
            }
        }
        self.drive_review(ctx);
        self.advance(ctx);
        self.play_sounds();
        let mut keep_open = true;

        egui::TopBottomPanel::top("player_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Close player").clicked() {
                    keep_open = false;
                }
                ui.separator();
                ui.strong(&self.recording.map);
                ui.label(RichText::new(&self.recording.recorded_by).color(MUTED));
                ui.separator();
                self.view_picker(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::Slider::new(&mut self.fov, 60.0..=120.0).text("FOV").integer());
                    ui.separator();
                    self.volume_controls(ui);
                    ui.separator();
                    egui::ComboBox::from_id_salt("player_glow")
                        .selected_text(self.glow.label())
                        .width(90.0)
                        .show_ui(ui, |ui| {
                            for mode in GlowMode::ALL {
                                ui.selectable_value(&mut self.glow, mode, mode.label());
                            }
                        });
                    ui.label(RichText::new("Glow").color(MUTED)).on_hover_text(
                        "Team-coloured outlines that show players through walls. Enemies and teammates are judged from the player you're watching. G switches between settings.",
                    );
                    ui.separator();
                    ui.menu_button("Show…", |ui| self.show_menu(ui));
                });
            });
        });

        if self.review.is_some() {
            egui::TopBottomPanel::top("review_bar").show(ctx, |ui| self.review_bar(ui));
        }

        egui::TopBottomPanel::bottom("player_controls").show(ctx, |ui| {
            ui.add_space(4.0);
            if self.show_aim_graph {
                if let Some(slot) = self.watched_slot() {
                    self.aim_graph(ui, slot);
                    ui.add_space(2.0);
                }
            }
            self.timeline(ui);
            ui.add_space(2.0);
            self.controls(ui);
            ui.add_space(2.0);
        });

        egui::SidePanel::right("player_side")
            .resizable(false)
            .exact_width(236.0)
            .show(ctx, |ui| self.side_panel(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.viewport(ui));

        let options = self.view_options();
        ctx.data_mut(|data| data.insert_temp(view_options_id(), options));
        keep_open
    }

    fn view_picker(&mut self, ui: &mut egui::Ui) {
        let label = match self.camera_mode {
            CameraMode::Free => "Free camera".to_string(),
            CameraMode::Eyes(slot) => self.recording.name_at(slot, self.tick).to_string(),
        };
        let tick = self.tick;
        ui.label("View");
        egui::ComboBox::from_id_salt("player_view")
            .selected_text(label)
            .width(190.0)
            .show_ui(ui, |ui| {
                let camera = self.camera();
                if ui.selectable_label(self.camera_mode == CameraMode::Free, "Free camera").clicked() {
                    self.go_free(&camera);
                }
                for slot in self.people_at(tick) {
                    let name = self.recording.name_at(slot, tick).to_string();
                    let selected = self.camera_mode == CameraMode::Eyes(slot);
                    if ui.selectable_label(selected, name).clicked() {
                        self.camera_mode = CameraMode::Eyes(slot);
                    }
                }
            });
        let watching = self.watched_slot().is_some();
        for (third, label, tip) in [
            (false, "First person", "Through the player's eyes, with their hands and crosshair"),
            (true, "Third person", "From behind the player, so you see them move and aim. C switches."),
        ] {
            let chosen = watching && self.third_person == third;
            let button = ui
                .add_enabled(watching, egui::SelectableLabel::new(chosen, label))
                .on_hover_text(tip)
                .on_disabled_hover_text("Pick a player in View to watch them in first or third person");
            if button.clicked() {
                self.third_person = third;
                self.orbit = None;
            }
        }
    }

    fn go_free(&mut self, camera: &Camera) {
        self.free_camera = FreeCamera {
            position: camera.eye,
            yaw: camera.yaw,
            pitch: camera.pitch,
        };
        self.camera_mode = CameraMode::Free;
    }

    /// Slots of everyone on a team right now, red before blue.
    fn people_at(&self, tick: f64) -> Vec<usize> {
        let frame = self.recording.frame_at(tick);
        let mut slots: Vec<usize> = (0..self.recording.slot_count)
            .filter(|&slot| self.recording.player(frame, slot).is_on_a_team())
            .collect();
        slots.sort_by_key(|&slot| {
            let player = self.recording.player(frame, slot);
            (player.team, player.class, slot)
        });
        slots
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        let tick = self.tick;
        let frame = self.recording.frame_at(tick);
        let roster_height = if self.markers.is_empty() {
            ui.available_height()
        } else {
            ui.available_height() * 0.68
        };

        egui::ScrollArea::vertical()
            .id_salt("player_roster")
            .max_height(roster_height)
            .auto_shrink(false)
            .show(ui, |ui| {
                for (team, title) in [(TEAM_RED, "RED"), (TEAM_BLUE, "BLU")] {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(title).strong().color(team_color(team)));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_sized([PING_COLUMN, 14.0], egui::Label::new(RichText::new("Ping").small().color(MUTED)))
                                .on_hover_text(PING_TOOLTIP);
                            ui.add_sized([HP_COLUMN, 14.0], egui::Label::new(RichText::new("HP").small().color(MUTED)));
                        });
                    });
                    for slot in self.people_at(tick) {
                        let player = *self.recording.player(frame, slot);
                        if player.team != team {
                            continue;
                        }
                        let name = self.recording.name_at(slot, tick).to_string();
                        let state = if !player.is_alive() {
                            "dead".to_string()
                        } else if player.class == CLASS_MEDIC {
                            format!("{}%", player.charge)
                        } else {
                            player.health.to_string()
                        };
                        let selected = self.camera_mode == CameraMode::Eyes(slot);
                        let color = if player.is_alive() { Color32::LIGHT_GRAY } else { MUTED };
                        let text = RichText::new(format!("{} {name}", class_name(player.class))).color(color);
                        let bot = self.recording.steam_id_at(slot, tick) == 0;
                        let (ping, ping_color) = ping_label(player.ping, bot);
                        // The numbers go in first from the right, so the name is cut off before them.
                        ui.horizontal(|ui| {
                            ui.set_width(ui.available_width());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.add_sized([PING_COLUMN, 16.0], egui::Label::new(RichText::new(ping).color(ping_color)))
                                    .on_hover_text(PING_TOOLTIP);
                                ui.add_sized([HP_COLUMN, 16.0], egui::Label::new(RichText::new(state).color(MUTED)));
                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                    let steam_id = self.recording.steam_id_at(slot, tick);
                                    if let Some((label, color)) = self.marks.get(&steam_id) {
                                        ui.label(RichText::new(label).small().color(*color)).on_hover_text("Tagged in your TF2BD playerlists");
                                    }
                                    let button = egui::Button::new(text).selected(selected).frame(selected).truncate();
                                    let response = ui.add(button).on_hover_text(format!("{name}\nRight-click for more"));
                                    if response.clicked() {
                                        self.camera_mode = CameraMode::Eyes(slot);
                                    }
                                    response.context_menu(|ui| self.player_menu(ui, slot, steam_id, &name));
                                });
                            });
                        });
                    }
                }
            });

        if let Some((message, ok, _)) = &self.report_status {
            ui.separator();
            let color = if *ok { MUTED } else { Color32::from_rgb(235, 105, 90) };
            ui.add(egui::Label::new(RichText::new(message).small().color(color)).wrap());
        }
        if self.markers.is_empty() {
            return;
        }
        ui.separator();
        let mut review_pick = None;
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Detections ({})", self.markers.len())).strong().color(AMBER));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.menu_button("Review…", |ui| {
                    ui.label(RichText::new("Pick a player to step through their detections").small().color(MUTED));
                    for (steam_id, count) in super::review::flagged_players(&self.markers) {
                        if ui.button(format!("{} ({count})", self.name_of_steam_id(steam_id))).clicked() {
                            review_pick = Some(steam_id);
                            ui.close_menu();
                        }
                    }
                })
                .response
                .on_hover_text("Plays one player's detections in slow motion from their eyes, one after another");
            });
        });
        if let Some(steam_id) = review_pick {
            self.start_review(steam_id);
        }
        let reviewing: &[usize] = match &self.review {
            Some(review) => &review.moments[review.index].markers,
            None => &[],
        };
        let first = f64::from(self.recording.first_tick());
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let mut jump = None;
        egui::ScrollArea::vertical()
            .id_salt("player_detections")
            .auto_shrink(false)
            .show_rows(ui, row_height, self.markers.len(), |ui, rows| {
                for index in rows {
                    let marker = &self.markers[index];
                    let seconds = self.recording.seconds_between(first, f64::from(marker.tick));
                    let text = format!("{}  {}", format_clock(seconds), marker.label);
                    let current = reviewing.contains(&index);
                    if ui.add(egui::Button::new(text).selected(current).frame(current).truncate()).clicked() {
                        jump = Some(index);
                    }
                }
            });
        if let Some(index) = jump {
            self.jump_to_marker_at(index);
        }
    }

    /// Jumps to a detection from the list. While reviewing that player it goes to
    /// that moment of the review, otherwise it leaves review mode first.
    fn jump_to_marker_at(&mut self, index: usize) {
        if let Some(review) = &self.review {
            let moment = review.moments.iter().position(|moment| moment.markers.contains(&index));
            if let Some(moment) = moment {
                self.review_moment(moment);
                return;
            }
            self.stop_review();
        }
        if let Some(marker) = self.markers.get(index).cloned() {
            self.jump_to_marker(&marker);
        }
    }

    /// A player's name at the time of their first detection.
    fn name_of_steam_id(&self, steam_id: u64) -> String {
        let at = self
            .markers
            .iter()
            .find(|marker| marker.steam_id == steam_id)
            .map_or(self.tick, |marker| f64::from(marker.tick));
        self.recording
            .slot_of_steam_id(steam_id, at)
            .map_or_else(|| steam_id.to_string(), |slot| self.recording.name_at(slot, at).to_string())
    }

    /// Starts review mode: each of this player's detections in turn, in slow motion
    /// from their eyes, with aim lines and the aim graph on. False if they have none.
    pub fn start_review(&mut self, steam_id: u64) -> bool {
        let gap = (REVIEW_GROUP_SECONDS / f64::from(self.recording.tick_interval)) as u32;
        let moments = super::review::moments(&self.markers, steam_id, gap);
        if moments.is_empty() {
            return false;
        }
        let before = match self.review.take() {
            Some(review) => review.before,
            None => ReviewRestore {
                speed_index: self.speed_index,
                show_aim_lines: self.show_aim_lines,
                show_aim_graph: self.show_aim_graph,
                show_ghosts: self.show_ghosts,
            },
        };
        let name = self.name_of_steam_id(steam_id);
        self.review = Some(Review {
            steam_id,
            name,
            moments,
            index: 0,
            phase: ReviewPhase::LeadIn,
            wait_until: 0.0,
            auto_advance: true,
            before,
        });
        self.review_moment(0);
        true
    }

    /// Plays one moment of the review from its start.
    fn review_moment(&mut self, index: usize) {
        let interval = f64::from(self.recording.tick_interval);
        let Some(review) = &mut self.review else { return };
        let index = index.min(review.moments.len().saturating_sub(1));
        review.index = index;
        review.phase = ReviewPhase::LeadIn;
        let moment = review.moments[index].clone();
        let steam_id = review.steam_id;
        let ghosts_before = review.before.show_ghosts;
        // Backtrack checks are about where the target was, which the ghosts show.
        let backtrack = moment.markers.iter().any(|&marker| {
            self.markers.get(marker).is_some_and(|marker| {
                super::review::check_of(marker).contains("backtrack")
                    || marker.data.get("backtrack_flick").and_then(serde_json::Value::as_bool) == Some(true)
            })
        });
        let tick = f64::from(moment.tick);
        self.seek(tick - REVIEW_LEAD_SECONDS / interval);
        if let Some(slot) = self.recording.slot_of_steam_id(steam_id, tick) {
            self.camera_mode = CameraMode::Eyes(slot);
        }
        self.speed_index = REVIEW_SPEED;
        self.show_aim_lines = true;
        self.show_aim_graph = true;
        self.show_ghosts = ghosts_before || backtrack;
        self.playing = true;
    }

    /// Leaves review mode and puts back what it turned on. The match stays where it is.
    pub fn stop_review(&mut self) {
        let Some(review) = self.review.take() else { return };
        self.speed_index = review.before.speed_index;
        self.show_aim_lines = review.before.show_aim_lines;
        self.show_aim_graph = review.before.show_aim_graph;
        self.show_ghosts = review.before.show_ghosts;
        self.playing = false;
    }

    /// Whether review mode is on, and where it is: (detection number, how many, paused on it).
    pub fn review_state(&self) -> Option<(usize, usize, bool)> {
        self.review
            .as_ref()
            .map(|review| (review.index + 1, review.moments.len(), review.phase == ReviewPhase::Hold))
    }

    /// Moves review mode along: slow motion into the detection, a pause on it, a
    /// little after, then on to the next one.
    fn drive_review(&mut self, ctx: &egui::Context) {
        if self.clip.is_some() {
            return;
        }
        let now = ctx.input(|input| input.time);
        let interval = f64::from(self.recording.tick_interval);
        let playing = self.playing;
        let Some(review) = &mut self.review else { return };
        let moment = f64::from(review.moments[review.index].tick);
        let end = moment + REVIEW_FOLLOW_SECONDS / interval;
        let mut next = None;
        match review.phase {
            // Skipped past it by hand.
            ReviewPhase::LeadIn if self.tick > end => {
                review.phase = ReviewPhase::Done;
                review.wait_until = now + REVIEW_NEXT_SECONDS;
            }
            ReviewPhase::LeadIn if self.tick >= moment => {
                self.tick = moment;
                self.playing = false;
                review.phase = ReviewPhase::Hold;
                review.wait_until = now + REVIEW_HOLD_SECONDS;
            }
            ReviewPhase::Hold if playing || now >= review.wait_until => {
                review.phase = ReviewPhase::Follow;
                self.playing = true;
            }
            ReviewPhase::Follow if self.tick >= end => {
                review.phase = ReviewPhase::Done;
                self.playing = false;
                review.wait_until = now + REVIEW_NEXT_SECONDS;
            }
            ReviewPhase::Done
                if review.auto_advance && !playing && now >= review.wait_until && review.index + 1 < review.moments.len() =>
            {
                next = Some(review.index + 1);
            }
            _ => {}
        }
        if let Some(index) = next {
            self.review_moment(index);
        }
        // The pauses are timed, so keep drawing while they count down.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }

    /// The bar over the view while reviewing: which detection this is, what the
    /// check looks for and what it measured.
    fn review_bar(&mut self, ui: &mut egui::Ui) {
        let Some(review) = &self.review else { return };
        let index = review.index;
        let count = review.moments.len();
        let moment = review.moments[index].clone();
        let phase = review.phase;
        let name = review.name.clone();
        let mut auto_advance = review.auto_advance;
        let wait_left = (review.wait_until - ui.ctx().input(|input| input.time)).max(0.0);
        let first = f64::from(self.recording.first_tick());
        let clock = format_clock(self.recording.seconds_between(first, f64::from(moment.tick)));
        let playing = self.playing;
        let mut action = None;
        ui.add_space(3.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Review").strong().color(AMBER));
            ui.strong(&name);
            ui.label(format!("detection {} of {count}", index + 1));
            ui.label(RichText::new(format!("at {clock}")).color(MUTED));
            ui.separator();
            let (status, color) = match phase {
                ReviewPhase::LeadIn => ("Slow motion up to the moment it was flagged".to_string(), MUTED),
                ReviewPhase::Hold => ("This is the moment it was flagged".to_string(), AMBER),
                ReviewPhase::Follow => ("Playing on a little".to_string(), MUTED),
                ReviewPhase::Done if index + 1 == count => ("That was the last one".to_string(), MUTED),
                ReviewPhase::Done if auto_advance && !playing => (format!("Next one in {:.0} s", wait_left.ceil()), MUTED),
                ReviewPhase::Done => ("Press Next for the next one".to_string(), MUTED),
            };
            ui.label(RichText::new(status).color(color));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Exit review").on_hover_text("Esc").clicked() {
                    action = Some(ReviewAction::Exit);
                }
                if ui.add_enabled(index + 1 < count, egui::Button::new("Next")).clicked() {
                    action = Some(ReviewAction::Go(index + 1));
                }
                if ui.button("Replay").clicked() {
                    action = Some(ReviewAction::Go(index));
                }
                if ui.add_enabled(index > 0, egui::Button::new("Previous")).clicked() {
                    action = Some(ReviewAction::Go(index.saturating_sub(1)));
                }
                ui.checkbox(&mut auto_advance, "Go on by itself")
                    .on_hover_text("Moves on to the next detection a moment after each one ends");
            });
        });
        let recording = &self.recording;
        for &marker_index in &moment.markers {
            let Some(marker) = self.markers.get(marker_index) else { continue };
            let check = super::review::check_of(marker);
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(super::review::check_title(check)).strong().color(REVIEW_CHECK_COLOR));
                ui.label(super::review::what_to_watch(check));
            });
            if let Some(about) = crate::lib::algorithm::algorithm_tooltip(check) {
                ui.add(egui::Label::new(RichText::new(about).small().color(MUTED)).wrap());
            }
            let tick = f64::from(marker.tick);
            let name_of = |steam_id: u64| {
                recording
                    .slot_of_steam_id(steam_id, tick)
                    .map(|slot| recording.name_at(slot, tick).to_string())
            };
            let lines = super::review::measured(&marker.data, &name_of);
            if lines.is_empty() {
                continue;
            }
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Measured").small().strong().color(MUTED));
                for (label, value) in lines.iter().take(REVIEW_MAX_FIELDS) {
                    ui.label(RichText::new(format!("{label}:")).small().color(MUTED));
                    ui.label(RichText::new(value).small().color(Color32::from_rgb(230, 230, 230)));
                    ui.add_space(6.0);
                }
                if lines.len() > REVIEW_MAX_FIELDS {
                    let rest: Vec<String> = lines[REVIEW_MAX_FIELDS..]
                        .iter()
                        .map(|(label, value)| format!("{label}: {value}"))
                        .collect();
                    ui.label(RichText::new(format!("and {} more", rest.len())).small().color(MUTED))
                        .on_hover_text(rest.join("\n"));
                }
            });
        }
        if let Some((degrees, flicks)) = self.review.as_ref().and_then(|review| self.aimbot_fov(review.steam_id)) {
            ui.add_space(2.0);
            ui.label(
                RichText::new(format!(
                    "Aimbot FOV: at least {degrees:.1}\u{b0}, the biggest flick in {flicks} aim detection{}. The circle around the crosshair shows it.",
                    if flicks == 1 { "" } else { "s" }
                ))
                .small()
                .color(REVIEW_CHECK_COLOR),
            );
        }
        ui.add_space(3.0);
        if let Some(review) = &mut self.review {
            review.auto_advance = auto_advance;
        }
        match action {
            Some(ReviewAction::Exit) => self.stop_review(),
            Some(ReviewAction::Go(index)) => self.review_moment(index),
            None => {}
        }
    }

    /// A banner in the view counting down to the detection, then marking it.
    fn draw_review_banner(&self, ui: &egui::Ui, rect: Rect) {
        let Some(review) = &self.review else { return };
        let moment = f64::from(review.moments[review.index].tick);
        let (text, color) = match review.phase {
            ReviewPhase::LeadIn => {
                let seconds = self.recording.seconds_between(self.tick, moment).max(0.0);
                (format!("Flagged in {seconds:.1} s"), Color32::from_rgb(225, 225, 225))
            }
            ReviewPhase::Hold => ("Flagged here".to_string(), AMBER),
            ReviewPhase::Follow | ReviewPhase::Done => return,
        };
        let painter = ui.painter_at(rect);
        let galley = painter.layout_no_wrap(text, FontId::proportional(18.0), color);
        let at = Pos2::new(rect.center().x, rect.top() + 30.0);
        let back = Rect::from_center_size(at, galley.size() + Vec2::new(22.0, 10.0));
        painter.rect_filled(back, 4.0, Color32::from_black_alpha(170));
        painter.galley(back.min + Vec2::new(11.0, 5.0), galley, color);
    }

    fn jump_to_marker(&mut self, marker: &Marker) {
        let lead = DETECTION_LEAD_SECONDS / f64::from(self.recording.tick_interval);
        self.seek(f64::from(marker.tick) - lead);
        if let Some(slot) = self.recording.slot_of_steam_id(marker.steam_id, self.tick) {
            self.camera_mode = CameraMode::Eyes(slot);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let tick_seconds = f64::from(self.recording.tick_interval);
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() {
                self.playing = !self.playing;
            }
            if ui.button("-5s").clicked() {
                self.seek(self.tick - SKIP_SECONDS / tick_seconds);
            }
            if ui.button("+5s").clicked() {
                self.seek(self.tick + SKIP_SECONDS / tick_seconds);
            }
            ui.separator();
            let first = f64::from(self.recording.first_tick());
            let last = f64::from(self.recording.last_tick());
            ui.label(format!(
                "{} / {}",
                format_clock(self.recording.seconds_between(first, self.tick)),
                format_clock(self.recording.seconds_between(first, last)),
            ));
            ui.label(RichText::new(format!("tick {}", self.tick as u32)).color(MUTED));
            ui.separator();
            for (index, speed) in SPEEDS.iter().enumerate() {
                if ui.selectable_label(index == self.speed_index, format!("{speed}x")).clicked() {
                    self.speed_index = index;
                }
            }
            ui.separator();
            self.clip_controls(ui);
            ui.separator();
            let watching = self.watched_slot().is_some();
            ui.add_enabled(watching, egui::Checkbox::new(&mut self.only_their_kills, "Only their kills and deaths"))
                .on_hover_text("On the timeline, show only the watched player's kills (up) and deaths (down), not every kill in the match")
                .on_disabled_hover_text("Watch a player to see only their kills and deaths on the timeline");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new("Space play   Left/Right skip   , . frame   [ ] speed   T traces   G glow   H hitboxes   C 3rd person   M mute")
                        .small()
                        .color(MUTED),
                );
            });
        });
    }

    /// How fast the watched player's view turns around now, with their shots, hits and
    /// detections marked. An aimbot's snap is a spike no hand can make.
    fn aim_graph(&mut self, ui: &mut egui::Ui, slot: usize) {
        let width = ui.available_width();
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, AIM_GRAPH_HEIGHT), Sense::click_and_drag());
        let recording = &self.recording;
        let window = AIM_GRAPH_SECONDS / f64::from(recording.tick_interval);
        let (from, to) = (self.tick - window, self.tick + window);
        let x_of = |tick: f64| rect.left() + ((tick - from) / (to - from)) as f32 * rect.width();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_rgb(20, 24, 29));

        let first = recording.frame_at(from).saturating_sub(1);
        let last = (recording.frame_at(to) + 1).min(recording.ticks.len() - 1);
        let ticks = &recording.ticks[first..=last];
        let angles: Vec<(f32, f32)> = (first..=last)
            .map(|frame| {
                let player = recording.player(frame, slot);
                (normalize_pitch(player.pitch), player.yaw)
            })
            .collect();
        let alive: Vec<bool> = (first..=last).map(|frame| recording.player(frame, slot).is_alive()).collect();
        let speeds: Vec<(u32, f32)> = aim_speeds(ticks, &angles)
            .into_iter()
            .filter(|(tick, _)| {
                let index = ticks.partition_point(|t| t < tick);
                alive.get(index).copied().unwrap_or(false)
            })
            .collect();
        let peak = speeds.iter().map(|(_, speed)| *speed).fold(0.0, f32::max);
        let scale = peak.max(AIM_GRAPH_MIN_SCALE);
        let plot = rect.shrink2(Vec2::new(0.0, 6.0));
        let y_of = |speed: f32| plot.bottom() - (speed / scale).min(1.0) * plot.height();

        // Detections for this player, shots and hits.
        let steam_id = recording.steam_id_at(slot, self.tick);
        for marker in self.markers.iter().filter(|m| m.steam_id == steam_id && steam_id != 0) {
            let x = x_of(f64::from(marker.tick));
            if rect.x_range().contains(x) {
                painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(2.0_f32, AMBER));
            }
        }
        for shot in recording.shots.iter().filter(|shot| shot.slot == slot) {
            let x = x_of(f64::from(shot.tick));
            if rect.x_range().contains(x) {
                painter.line_segment([Pos2::new(x, rect.bottom() - 6.0), Pos2::new(x, rect.bottom())], Stroke::new(1.0_f32, MUTED));
            }
        }
        for hurt in recording.hurts.iter().filter(|hurt| hurt.attacker == Some(slot)) {
            let x = x_of(f64::from(hurt.tick));
            if rect.x_range().contains(x) {
                painter.circle_filled(Pos2::new(x, rect.bottom() - 9.0), 2.5, Color32::from_rgb(255, 210, 70));
            }
        }

        // The speed line.
        let points: Vec<Pos2> = speeds
            .iter()
            .map(|(tick, speed)| Pos2::new(x_of(f64::from(*tick)), y_of(*speed)))
            .filter(|point| rect.x_range().contains(point.x))
            .collect();
        if points.len() > 1 {
            painter.add(egui::Shape::line(points, Stroke::new(1.5_f32, Color32::from_rgb(120, 200, 255))));
        }
        let now = x_of(self.tick);
        painter.line_segment([Pos2::new(now, rect.top()), Pos2::new(now, rect.bottom())], Stroke::new(1.0_f32, Color32::WHITE));
        painter.text(
            rect.left_top() + Vec2::new(4.0, 2.0),
            Align2::LEFT_TOP,
            format!("Aim speed (peak {peak:.1}°/tick).  Grey ticks: shots.  Yellow dots: hits.  Orange lines: detections."),
            FontId::proportional(11.0),
            MUTED,
        );

        // Clicking or dragging in the graph moves through time, like the timeline.
        if let Some(pointer) = response.interact_pointer_pos() {
            if response.clicked() || response.dragged() {
                let share = f64::from((pointer.x - rect.left()) / rect.width());
                self.seek(from + share * (to - from));
            }
        }
        response.on_hover_text("How fast the watched player's view turned between updates. Human aim speeds up and slows down; an aimbot snap is one tall spike.");
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(width, TIMELINE_HEIGHT), Sense::click_and_drag());
        let recording = &self.recording;
        let first = f64::from(recording.first_tick());
        let last = f64::from(recording.last_tick());
        let span = (last - first).max(1.0);
        let x_of = |tick: f64| rect.left() + ((tick - first) / span) as f32 * rect.width();
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 2.0, Color32::from_rgb(24, 28, 33));
        for (index, round) in recording.rounds.iter().enumerate() {
            let left = x_of(f64::from(round.start_tick));
            let right = x_of(f64::from(round.end_tick.unwrap_or(recording.last_tick())));
            let fill = if index % 2 == 0 {
                Color32::from_rgb(32, 37, 43)
            } else {
                Color32::from_rgb(40, 46, 53)
            };
            let band = Rect::from_min_max(Pos2::new(left, rect.top()), Pos2::new(right, rect.bottom()));
            painter.rect_filled(band, 0.0, fill);
            if round.winner == TEAM_RED || round.winner == TEAM_BLUE {
                let strip = Rect::from_min_max(band.min, Pos2::new(right, rect.top() + 3.0));
                painter.rect_filled(strip, 0.0, team_color(round.winner));
            }
        }

        let watched = self.watched_slot();
        let theirs = watched.filter(|_| self.only_their_kills);
        let middle = rect.top() + SEEK_LANE + (rect.height() - SEEK_LANE - PRESENCE_LANE) * 0.5;
        let mut shown_kills: Vec<&Kill> = Vec::new();
        for kill in &recording.kills {
            if kill.feigned {
                continue;
            }
            // Every kill goes up for RED and down for BLU; the watched player's own go
            // up for their kills and down for their deaths, a little taller.
            let (up, color, length, width) = match theirs {
                None => (kill.attacker_team == TEAM_RED, team_color(kill.attacker_team), 9.0_f32, 1.5_f32),
                Some(slot) if kill.victim == Some(slot) => {
                    let by_someone = kill.attacker.is_some_and(|attacker| attacker != slot);
                    (false, if by_someone { team_color(kill.attacker_team) } else { MUTED }, 11.0, 2.0)
                }
                Some(slot) if kill.attacker == Some(slot) => (true, team_color(kill.attacker_team), 11.0, 2.0),
                Some(_) => continue,
            };
            let x = x_of(f64::from(kill.tick));
            let end = if up { middle - length } else { middle + length };
            painter.line_segment([Pos2::new(x, middle), Pos2::new(x, end)], Stroke::new(width, color));
            shown_kills.push(kill);
        }

        // When the demo has the watched player's data: green alive, grey dead, nothing
        // while they're out of the recorder's view.
        let runs: &[PresenceRun] = watched.and_then(|slot| self.presence.get(slot)).map_or(&[], Vec::as_slice);
        for run in runs {
            let color = if run.presence == Presence::Alive { PRESENCE_ALIVE } else { PRESENCE_DEAD };
            let left = x_of(f64::from(run.from));
            let right = x_of(f64::from(run.to)).max(left + 1.0);
            let bar = Rect::from_min_max(Pos2::new(left, rect.bottom() - PRESENCE_LANE), Pos2::new(right, rect.bottom()));
            painter.rect_filled(bar, 0.0, color);
        }

        for marker in &self.markers {
            let x = x_of(f64::from(marker.tick));
            let top = rect.top() + 3.0;
            painter.add(egui::Shape::convex_polygon(
                vec![
                    Pos2::new(x - 4.0, top),
                    Pos2::new(x + 4.0, top),
                    Pos2::new(x, top + 7.0),
                ],
                AMBER,
                Stroke::NONE,
            ));
        }

        let head = x_of(self.tick);
        painter.line_segment(
            [Pos2::new(head, rect.top()), Pos2::new(head, rect.bottom())],
            Stroke::new(2.0_f32, Color32::WHITE),
        );

        let tick_at = |x: f32| first + f64::from(((x - rect.left()) / rect.width()).clamp(0.0, 1.0)) * span;
        if let Some(pointer) = response.hover_pos() {
            let tick = tick_at(pointer.x);
            let mut lines = vec![format_clock(recording.seconds_between(first, tick))];
            if let Some(marker) = self.markers.iter().find(|marker| (x_of(f64::from(marker.tick)) - pointer.x).abs() < 5.0) {
                lines[0] = format!("{}  {}", lines[0], marker.label);
            }
            let near_kill = shown_kills
                .iter()
                .filter(|kill| (x_of(f64::from(kill.tick)) - pointer.x).abs() < KILL_HOVER_REACH)
                .min_by(|a, b| {
                    let distance = |kill: &&&Kill| (x_of(f64::from(kill.tick)) - pointer.x).abs();
                    distance(a).total_cmp(&distance(b))
                });
            if let Some(kill) = near_kill {
                lines.push(describe_kill(kill));
            }
            if let Some(slot) = watched {
                let name = recording.name_at(slot, tick);
                let state = match runs.iter().find(|run| f64::from(run.from) <= tick && tick < f64::from(run.to)) {
                    Some(run) if run.presence == Presence::Alive => "in the demo, alive",
                    Some(_) => "in the demo, dead",
                    None => "not in the demo here: out of the recorder's view, so frozen or missing",
                };
                lines.push(format!("{name}: {state}"));
            }
            response.clone().on_hover_text_at_pointer(lines.join("\n"));
        }
        if response.drag_started() {
            self.resume_after_scrub = self.playing;
            self.playing = false;
        }
        if response.clicked() || response.dragged() {
            if let Some(pointer) = response.interact_pointer_pos() {
                self.tick = tick_at(pointer.x);
            }
        }
        if response.drag_stopped() {
            self.playing = self.resume_after_scrub;
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        self.view_rect = rect;
        let response = ui.allocate_rect(rect, Sense::click_and_drag());

        let orbiting = self.third_person && matches!(self.camera_mode, CameraMode::Eyes(_));
        if response.dragged_by(PointerButton::Primary) || response.dragged_by(PointerButton::Secondary) {
            let camera = self.camera();
            let delta = response.drag_delta();
            if orbiting {
                // Like spectating: the camera swings around the player and keeps following them.
                let (yaw, pitch) = self.orbit.unwrap_or((camera.yaw, camera.pitch));
                self.orbit = Some((
                    yaw - delta.x * LOOK_SENSITIVITY,
                    (pitch + delta.y * LOOK_SENSITIVITY).clamp(-ORBIT_PITCH_LIMIT, ORBIT_PITCH_LIMIT),
                ));
            } else {
                if self.camera_mode != CameraMode::Free {
                    self.go_free(&camera);
                }
                self.free_camera.yaw -= delta.x * LOOK_SENSITIVITY;
                self.free_camera.pitch = (self.free_camera.pitch + delta.y * LOOK_SENSITIVITY).clamp(-89.0, 89.0);
            }
        }
        if orbiting {
            if response.double_clicked() {
                self.orbit = None;
                self.chase_distance = CHASE_BEHIND;
            }
            let scroll = if response.hovered() { ui.input(|input| input.smooth_scroll_delta.y) } else { 0.0 };
            if scroll != 0.0 {
                self.chase_distance = (self.chase_distance * (1.0 - scroll * 0.002)).clamp(CHASE_NEAREST, CHASE_FURTHEST);
            }
        }

        let camera = self.camera();
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
        let (view, projection) = self.matrices(&camera, aspect);
        let view_projection = {
            let vertical = 2.0 * ((VIEW_MODEL_FOV.to_radians() * 0.5).tan() * 0.75).atan();
            Mat4::perspective_rh_gl(vertical, aspect, 1.0, 2000.0)
        };
        let Scene { shapes, actors, boxes } = self.scene();
        let view_actors = self.view_model(&camera);
        let sky_view = self.sky_camera.map(|sky| {
            let backdrop_camera = Camera {
                eye: sky.origin + camera.eye / sky.scale,
                ..camera
            };
            self.matrices(&backdrop_camera, aspect).0
        });
        let frame = Frame {
            view,
            projection,
            shapes,
            actors,
            view_actors,
            view_projection,
            sky_view,
            traces: self.traces(&camera),
            overlay: self.overlay(&camera, &boxes),
        };

        let renderer = Arc::clone(&self.renderer);
        let pending = Arc::clone(&self.pending_map);
        let errors = Arc::clone(&self.render_error);
        let callback = egui_glow::CallbackFn::new(move |info, painter| {
            let Ok(mut renderer) = renderer.lock() else { return };
            if renderer.is_none() {
                let map = pending.lock().ok().and_then(|mut map| map.take());
                if let Some(map) = map {
                    match Renderer::new(painter.gl(), &map) {
                        Ok(created) => *renderer = Some(created),
                        Err(error) => {
                            if let Ok(mut slot) = errors.lock() {
                                *slot = Some(error);
                            }
                        }
                    }
                }
            }
            if let Some(renderer) = renderer.as_mut() {
                let area = info.viewport_in_pixels();
                renderer.paint(
                    painter.gl(),
                    &frame,
                    painter.intermediate_fbo(),
                    [area.left_px, area.from_bottom_px, area.width_px, area.height_px],
                );
            }
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(callback),
        });

        let projected = projection * view;
        if self.scoped() {
            self.draw_scope(ui, rect);
        } else if self.shows_crosshair() {
            self.draw_crosshair(ui, rect);
        }
        if self.show_fov_circle {
            self.draw_fov_circle(ui, rect);
        }
        if self.show_names && !self.scoped() {
            self.draw_names(ui, rect, &projected);
        }
        if self.show_damage {
            self.draw_damage(ui, rect, &projected);
        }
        self.draw_overlays(ui, rect);
        self.draw_review_banner(ui, rect);

        if let Ok(error) = self.render_error.lock() {
            if let Some(error) = error.as_ref() {
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    format!("Your graphics driver couldn't draw the map:\n{error}"),
                    FontId::proportional(14.0),
                    Color32::WHITE,
                );
            }
        }
        if !self.renderer_ready() {
            ui.ctx().request_repaint();
        }
    }

    /// Whether the view is through living eyes, which is when TF2 shows a crosshair.
    fn shows_crosshair(&self) -> bool {
        self.first_person_slot().is_some_and(|slot| {
            let player = self.recording.player(self.recording.frame_at(self.tick), slot);
            player.is_alive() && !player.is_out_of_view()
        })
    }

    fn draw_crosshair(&mut self, ui: &egui::Ui, rect: Rect) {
        if self.crosshair_texture.is_none() {
            if let Some(image) = &self.crosshair_image {
                let pixels = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width as usize, image.height as usize],
                    &image.rgba,
                );
                self.crosshair_texture = Some(ui.ctx().load_texture("crosshair", pixels, egui::TextureOptions::LINEAR));
            }
        }
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let size = (rect.height() * CROSSHAIR_SCALE).round().max(16.0);
        match &self.crosshair_texture {
            Some(texture) => {
                let area = Rect::from_center_size(center, Vec2::splat(size));
                let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                painter.image(texture.id(), area, uv, Color32::from_rgba_unmultiplied(200, 200, 200, 230));
            }
            None => {
                // A plain cross if the game's image couldn't be read.
                let arm = size * 0.35;
                let line = Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(220, 220, 220, 230));
                painter.line_segment([center - Vec2::new(arm, 0.0), center + Vec2::new(arm, 0.0)], line);
                painter.line_segment([center - Vec2::new(0.0, arm), center + Vec2::new(0.0, arm)], line);
            }
        }
    }

    /// TF2's sniper scope over the view, with black bars filling the sides.
    fn draw_scope(&mut self, ui: &egui::Ui, rect: Rect) {
        if self.scope_texture.is_none() {
            if let Some(image) = &self.scope_image {
                let pixels = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width as usize, image.height as usize],
                    &image.rgba,
                );
                self.scope_texture = Some(ui.ctx().load_texture("sniper_scope", pixels, egui::TextureOptions::LINEAR));
            }
        }
        let painter = ui.painter_at(rect);
        // The game draws the scope 4:3 wide, with its image squeezed to suit.
        let height = rect.height();
        let width = (height * 4.0 / 3.0).min(rect.width());
        let square = Rect::from_center_size(rect.center(), Vec2::new(width, height));
        painter.rect_filled(Rect::from_min_max(rect.min, Pos2::new(square.left(), rect.bottom())), 0.0, Color32::BLACK);
        painter.rect_filled(Rect::from_min_max(Pos2::new(square.right(), rect.top()), rect.max), 0.0, Color32::BLACK);

        let center = rect.center();
        if let Some(texture) = &self.scope_texture {
            // One quarter, mirrored into each corner the way the game does it.
            let quarter = Vec2::new(width / 2.0, height / 2.0);
            let corners = [
                (Pos2::new(square.left(), square.top()), [0.0, 0.0, 1.0, 1.0]),
                (Pos2::new(center.x, square.top()), [1.0, 0.0, 0.0, 1.0]),
                (Pos2::new(square.left(), center.y), [0.0, 1.0, 1.0, 0.0]),
                (Pos2::new(center.x, center.y), [1.0, 1.0, 0.0, 0.0]),
            ];
            for (corner, [u0, v0, u1, v1]) in corners {
                let uv = Rect::from_min_max(Pos2::new(u0, v0), Pos2::new(u1, v1));
                painter.image(texture.id(), Rect::from_min_size(corner, quarter), uv, Color32::WHITE);
            }
        }
        // The crosshair.
        let line = Stroke::new(1.0_f32, Color32::BLACK);
        painter.line_segment([Pos2::new(square.left(), center.y), Pos2::new(square.right(), center.y)], line);
        painter.line_segment([Pos2::new(center.x, square.top()), Pos2::new(center.x, square.bottom())], line);
    }

    /// Floating damage numbers over hurt players. When watching someone, only their own
    /// hits are shown, the way TF2 shows them to the shooter.
    fn draw_damage(&self, ui: &egui::Ui, rect: Rect, projected: &Mat4) {
        let painter = ui.painter_at(rect);
        let recording = &self.recording;
        let window = DAMAGE_SECONDS / f64::from(recording.tick_interval);
        let first = recording.hurts.partition_point(|hurt| f64::from(hurt.tick) < self.tick - window);
        let watched = self.watched_slot();
        for hurt in &recording.hurts[first..] {
            let tick = f64::from(hurt.tick);
            if tick > self.tick {
                break;
            }
            let Some(victim) = hurt.victim else { continue };
            if watched.is_some() && hurt.attacker != watched {
                continue;
            }
            let frame = recording.frame_at(tick);
            let target = recording.player(frame, victim);
            if target.is_out_of_view() || Some(victim) == watched {
                continue;
            }
            let age = ((self.tick - tick) / window) as f32;
            let point = Vec3::from(target.position) + Vec3::Z * (target.eye_height() + 18.0 + age * 30.0);
            let clip = *projected * Vec4::new(point.x, point.y, point.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let x = rect.center().x + clip.x / clip.w * rect.width() * 0.5;
            let y = rect.center().y - clip.y / clip.w * rect.height() * 0.5;
            let (text, color) = if hurt.crit {
                (format!("-{} CRIT", hurt.damage), Color32::from_rgb(120, 230, 90))
            } else if hurt.mini_crit {
                (format!("-{} MINI-CRIT", hurt.damage), Color32::from_rgb(200, 230, 90))
            } else {
                (format!("-{}", hurt.damage), Color32::from_rgb(255, 210, 70))
            };
            let alpha = ((1.0 - age) * 255.0).clamp(0.0, 255.0) as u8;
            let fade = |c: Color32| Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha);
            let font = FontId::proportional(16.0);
            painter.text(Pos2::new(x + 1.0, y + 1.0), Align2::CENTER_BOTTOM, &text, font.clone(), fade(Color32::BLACK));
            painter.text(Pos2::new(x, y), Align2::CENTER_BOTTOM, &text, font, fade(color));
        }
    }

    fn draw_names(&self, ui: &egui::Ui, rect: Rect, projected: &Mat4) {
        let painter = ui.painter_at(rect);
        for slot in 0..self.recording.slot_count {
            if Some(slot) == self.watched_slot() {
                continue;
            }
            let player = self.blended_player(slot);
            if !player.is_alive() || !player.is_on_a_team() || player.is_out_of_view() {
                continue;
            }
            let head = Vec3::from(player.position) + Vec3::Z * 92.0;
            let clip = *projected * Vec4::new(head.x, head.y, head.z, 1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let x = rect.center().x + clip.x / clip.w * rect.width() * 0.5;
            let y = rect.center().y - clip.y / clip.w * rect.height() * 0.5;
            let name = self.recording.name_at(slot, self.tick);
            painter.text(
                Pos2::new(x + 1.0, y + 1.0),
                Align2::CENTER_BOTTOM,
                name,
                FontId::proportional(12.0),
                Color32::from_black_alpha(200),
            );
            painter.text(
                Pos2::new(x, y),
                Align2::CENTER_BOTTOM,
                name,
                FontId::proportional(12.0),
                team_color(player.team),
            );
        }
    }

    fn recent_kills(&self) -> Vec<&Kill> {
        let window = KILL_FEED_SECONDS / f64::from(self.recording.tick_interval);
        let mut recent: Vec<&Kill> = self
            .recording
            .kills
            .iter()
            .filter(|kill| {
                let tick = f64::from(kill.tick);
                tick <= self.tick && tick >= self.tick - window
            })
            .collect();
        let excess = recent.len().saturating_sub(KILL_FEED_LINES);
        recent.drain(..excess);
        recent
    }

    fn draw_overlays(&self, ui: &egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        let font = FontId::proportional(13.0);
        let backing = Color32::from_black_alpha(170);

        let mut y = rect.top() + 10.0;
        for kill in self.recent_kills() {
            let by = if kill.attacker.is_some() && kill.attacker != kill.victim {
                kill.attacker_name.as_str()
            } else {
                ""
            };
            let weapon = kill.weapon.replace("tf_projectile_", "").replace('_', " ");
            let text = format!("{by}  [{weapon}]  {}", kill.victim_name);
            let galley = painter.layout_no_wrap(text, font.clone(), Color32::WHITE);
            let size = galley.size();
            let spot = Pos2::new(rect.right() - size.x - 16.0, y);
            painter.rect_filled(
                Rect::from_min_size(spot - Vec2::new(6.0, 2.0), size + Vec2::new(12.0, 4.0)),
                3.0,
                backing,
            );
            painter.galley(spot, galley, Color32::WHITE);
            y += size.y + 8.0;
        }

        let window = CHAT_SECONDS / f64::from(self.recording.tick_interval);
        let lines: Vec<_> = self
            .recording
            .chat
            .iter()
            .filter(|line| {
                let tick = f64::from(line.tick);
                tick <= self.tick && tick >= self.tick - window
            })
            .collect();
        let mut y = rect.bottom() - 12.0;
        for line in lines.iter().rev().take(CHAT_LINES) {
            let prefix = if line.team_only { "(team) " } else { "" };
            let text = format!("{prefix}{}: {}", line.from, line.text);
            let galley = painter.layout(text, font.clone(), Color32::WHITE, rect.width() * 0.5);
            let size = galley.size();
            y -= size.y + 6.0;
            let spot = Pos2::new(rect.left() + 14.0, y);
            painter.rect_filled(
                Rect::from_min_size(spot - Vec2::new(6.0, 2.0), size + Vec2::new(12.0, 4.0)),
                3.0,
                backing,
            );
            painter.galley(spot, galley, Color32::WHITE);
        }

        // Who's talking on voice chat right now.
        if self.voice_chat {
            let mut y = rect.bottom() - 12.0;
            for line in self.voice_lines.iter().filter(|line| line.start <= self.tick && self.tick < line.end) {
                let (name, color) = match line.slot {
                    Some(slot) => {
                        let frame = self.recording.frame_at(self.tick);
                        (self.recording.name_at(slot, self.tick).to_string(), team_color(self.recording.player(frame, slot).team))
                    }
                    None => ("Someone".to_string(), Color32::LIGHT_GRAY),
                };
                let galley = painter.layout_no_wrap(format!("Talking: {name}"), font.clone(), color);
                let size = galley.size();
                y -= size.y + 6.0;
                let spot = Pos2::new(rect.right() - 14.0 - size.x, y);
                painter.rect_filled(Rect::from_min_size(spot - Vec2::new(6.0, 2.0), size + Vec2::new(12.0, 4.0)), 3.0, backing);
                painter.galley(spot, galley, color);
            }
        }

        if let CameraMode::Eyes(slot) = self.camera_mode {
            let frame = self.recording.frame_at(self.tick);
            let player = self.recording.player(frame, slot);
            let is_recorder = Some(slot) == self.recording.recorder_slot;
            let status = if is_recorder {
                None
            } else if player.is_out_of_view() {
                Some("This player is out of range in this demo, so their view has stopped.")
            } else if !player.is_alive() {
                Some("This player is dead.")
            } else {
                None
            };
            if let Some(status) = status {
                painter.text(
                    Pos2::new(rect.center().x, rect.top() + 26.0),
                    Align2::CENTER_CENTER,
                    status,
                    font.clone(),
                    AMBER,
                );
            }
            if self.third_person {
                let hint = if self.orbit.is_some() {
                    "Drag to look around them   Scroll to zoom   Double-click to go back behind them   C first person"
                } else {
                    "Drag to look around them   Scroll to zoom   C first person"
                };
                painter.text(
                    Pos2::new(rect.center().x, rect.bottom() - 20.0),
                    Align2::CENTER_CENTER,
                    hint,
                    FontId::proportional(12.0),
                    Color32::from_white_alpha(170),
                );
            }
        } else {
            painter.text(
                Pos2::new(rect.center().x, rect.bottom() - 20.0),
                Align2::CENTER_CENTER,
                "Drag to look   W A S D to move   Q E down and up   Shift for speed",
                FontId::proportional(12.0),
                Color32::from_white_alpha(170),
            );
        }
    }
}

/// Adds a line as two triangles, widened sideways to the viewer so it never looks edge-on.
fn push_strip(floats: &mut Vec<f32>, from: Vec3, to: Vec3, eye: Vec3, color: [f32; 3], alpha: [f32; 2], widen: f32) {
    let along = (to - from).normalize_or_zero();
    let toward_eye = (eye - (from + to) * 0.5).normalize_or_zero();
    let side = along.cross(toward_eye).normalize_or_zero();
    if side == Vec3::ZERO {
        return;
    }
    let half_width = |point: Vec3| (point.distance(eye) * TRACE_WIDTH).max(TRACE_MIN_WIDTH) * 0.5 * widen;
    let (near, far) = (side * half_width(from), side * half_width(to));
    let corners = [from + near, from - near, to + far, to - far];
    for index in [0, 1, 2, 2, 1, 3] {
        let point = corners[index];
        let strength = if index >= 2 { alpha[1] } else { alpha[0] };
        floats.extend_from_slice(&[point.x, point.y, point.z, color[0], color[1], color[2], strength]);
    }
    debug_assert_eq!(floats.len() % TRACE_FLOATS, 0);
}

/// The parts of the line from `from` to `to` outside a ball around `center`: none,
/// one, or two when the line passes through it.
fn outside_sphere(from: Vec3, to: Vec3, center: Vec3, radius: f32) -> [Option<(Vec3, Vec3)>; 2] {
    let along = to - from;
    let offset = from - center;
    let a = along.length_squared();
    let b = 2.0 * offset.dot(along);
    let c = offset.length_squared() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if a < 1e-6 || discriminant <= 0.0 {
        // It never touches the ball, or it's a single point.
        return if c > 0.0 { [Some((from, to)), None] } else { [None, None] };
    }
    let root = discriminant.sqrt();
    let (enter, leave) = ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a));
    let before = (enter > 0.0).then(|| (from, from + along * enter.min(1.0)));
    let after = (leave < 1.0).then(|| (from + along * leave.max(0.0), to));
    [before, after]
}

/// The colour of a team's energy shots, brighter than its trails.
fn energy_color(team: u8) -> [f32; 3] {
    match team {
        TEAM_RED => [1.0, 0.32, 0.25],
        TEAM_BLUE => [0.3, 0.62, 1.0],
        _ => [0.85, 0.85, 1.0],
    }
}

/// A soft round glow facing the camera: brightest in the middle, fading to nothing
/// at `radius`, which is in world units so it shrinks with distance like a real thing.
fn push_glow(floats: &mut Vec<f32>, center: Vec3, radius: f32, eye: Vec3, color: [f32; 3], alpha: f32) {
    let toward = (eye - center).normalize_or_zero();
    let mut right = toward.cross(Vec3::Z).normalize_or_zero();
    if right == Vec3::ZERO {
        right = Vec3::X;
    }
    let up = right.cross(toward);
    let point = |step: usize| {
        let angle = step as f32 / GLOW_STEPS as f32 * std::f32::consts::TAU;
        center + (right * angle.cos() + up * angle.sin()) * radius
    };
    for step in 0..GLOW_STEPS {
        for (corner, strength) in [(center, alpha), (point(step), 0.0), (point(step + 1), 0.0)] {
            floats.extend_from_slice(&[corner.x, corner.y, corner.z, color[0], color[1], color[2], strength]);
        }
    }
}

/// A glowing hoop at right angles to where it's going, like the Righteous Bison's shot.
fn push_ring(floats: &mut Vec<f32>, center: Vec3, heading: Vec3, radius: f32, color: [f32; 3], alpha: f32) {
    let mut right = heading.cross(Vec3::Z).normalize_or_zero();
    if right == Vec3::ZERO {
        right = Vec3::X;
    }
    let up = right.cross(heading).normalize_or_zero();
    let point = |step: usize, distance: f32| {
        let angle = step as f32 / GLOW_STEPS as f32 * std::f32::consts::TAU;
        center + (right * angle.cos() + up * angle.sin()) * distance
    };
    let band = radius * 0.35;
    // Two bands, dark at the inner and outer edge and bright along the middle.
    for (inner, outer, inner_alpha, outer_alpha) in [(radius - band, radius, 0.0, alpha), (radius, radius + band, alpha, 0.0)] {
        for step in 0..GLOW_STEPS {
            let corners = [
                (point(step, inner), inner_alpha),
                (point(step, outer), outer_alpha),
                (point(step + 1, inner), inner_alpha),
                (point(step + 1, outer), outer_alpha),
            ];
            for index in [0, 1, 2, 2, 1, 3] {
                let (corner, strength) = corners[index];
                floats.extend_from_slice(&[corner.x, corner.y, corner.z, color[0], color[1], color[2], strength]);
            }
        }
    }
}

/// How fast each class runs, in units per second.
fn run_speed(class: u8) -> f32 {
    match class {
        1 => 400.0,
        3 => 240.0,
        4 => 280.0,
        5 | 8 => 320.0,
        6 => 230.0,
        _ => 300.0,
    }
}

fn first_player(recording: &Recording) -> Option<usize> {
    (0..recording.slot_count).find(|&slot| {
        (0..recording.ticks.len())
            .step_by(500)
            .any(|frame| recording.player(frame, slot).is_on_a_team())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles_blend_the_short_way_round() {
        assert_eq!(lerp_angle(350.0, 10.0, 0.5), 360.0);
        assert_eq!(lerp_angle(10.0, 350.0, 0.5), 0.0);
        assert_eq!(lerp_angle(0.0, 90.0, 0.5), 45.0);
    }

    #[test]
    fn pitch_is_stored_from_0_to_360_but_used_from_minus_180_to_180() {
        assert_eq!(normalize_pitch(350.0), -10.0);
        assert_eq!(normalize_pitch(20.0), 20.0);
    }

    #[test]
    fn looking_down_points_the_camera_down() {
        assert!((forward_from(90.0, 0.0) + Vec3::Z).length() < 1e-5);
        assert!((forward_from(0.0, 90.0) - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn traces_become_two_triangles_facing_the_viewer() {
        let mut floats = Vec::new();
        let eye = Vec3::new(0.0, 0.0, 1000.0);
        push_strip(&mut floats, Vec3::ZERO, Vec3::X * 2000.0, eye, [1.0; 3], [1.0, 1.0], 1.0);
        assert_eq!(floats.len(), 6 * TRACE_FLOATS);
        // Widened along Y, the only direction square to both the line and the viewer.
        assert!(floats.chunks(TRACE_FLOATS).all(|vertex| vertex[2].abs() < 1e-4));
        // The far end is further from the viewer, so it's widened more to look the same.
        let width_at = |x: f32| {
            floats
                .chunks(TRACE_FLOATS)
                .filter(|vertex| (vertex[0] - x).abs() < 1.0)
                .map(|vertex| vertex[1].abs())
                .fold(0.0, f32::max)
        };
        assert!(width_at(2000.0) > width_at(0.0));
    }

    #[test]
    fn a_line_pointing_at_the_viewer_is_skipped() {
        let mut floats = Vec::new();
        push_strip(&mut floats, Vec3::ZERO, Vec3::X * 100.0, Vec3::X * 200.0, [1.0; 3], [1.0, 1.0], 1.0);
        assert!(floats.is_empty());
    }

    fn projectile(id: u16, x: f32) -> ObjectFrame {
        ObjectFrame {
            id,
            kind: ObjectKind::Rocket,
            team: TEAM_RED,
            level: 0,
            crit: false,
            sapped: false,
            mini: false,
            under_construction: false,
            position: [x, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            health: 0,
            max_health: 0,
            model: 0,
            animation: super::super::recording::ObjectAnimation::default(),
        }
    }

    #[test]
    fn presence_runs_follow_a_player_in_and_out_of_the_demo() {
        use super::super::recording::{PLAYER_ALIVE, PLAYER_OUT_OF_VIEW};
        let alive = PLAYER_ALIVE;
        let away = PLAYER_ALIVE | PLAYER_OUT_OF_VIEW;
        let flags = [alive, alive, away, away, alive, 0, 0, alive];
        let mut recording = Recording {
            ticks: (0..flags.len() as u32).map(|frame| 100 + frame * 2).collect(),
            slot_count: 1,
            ..Recording::default()
        };
        for flag in flags {
            recording.players.push(PlayerFrame { team: TEAM_RED, flags: flag, ..PlayerFrame::default() });
        }
        let runs = presence_runs(&recording);
        let shape: Vec<(u32, u32, Presence)> = runs[0].iter().map(|run| (run.from, run.to, run.presence)).collect();
        assert_eq!(
            shape,
            vec![
                (100, 104, Presence::Alive),
                (108, 110, Presence::Alive),
                (110, 114, Presence::Dead),
                (114, 114, Presence::Alive),
            ],
            "out of view is left out, and each stretch runs until the next one starts"
        );
    }

    #[test]
    fn the_third_person_camera_sits_behind_and_keeps_out_of_walls() {
        let eye = Vec3::new(0.0, 0.0, 64.0);
        // Looking along +x: the camera goes back along -x, a little to the right (-y) and up.
        let open = chase_eye(eye, 0.0, 0.0, CHASE_BEHIND, |_, _| None);
        assert!((open.x + CHASE_BEHIND).abs() < 1e-3, "{open:?}");
        assert!((open.y + CHASE_RIGHT).abs() < 1e-3 && (open.z - eye.z - CHASE_UP).abs() < 1e-3, "{open:?}");
        // A wall 50 units behind: the camera stops short of it.
        let walled = chase_eye(eye, 0.0, 0.0, CHASE_BEHIND, |from, to| Some(from + (to - from).normalize() * 50.0));
        assert!((walled.distance(eye) - (50.0 - CHASE_WALL_GAP)).abs() < 1e-3, "{walled:?}");
        // A wall right behind the head: the camera stays at the eyes rather than go through it.
        assert_eq!(chase_eye(eye, 0.0, 0.0, CHASE_BEHIND, |from, _| Some(from + Vec3::X * -5.0)), eye);
        // Swung round to look along -y from above: the camera is on the +y side, up high.
        let swung = orbit_eye(eye, 45.0, 270.0, 100.0, |_, _| None);
        assert!(swung.y > 60.0 && swung.z > eye.z + 60.0 && swung.x.abs() < 1e-3, "{swung:?}");
    }

    #[test]
    fn weapons_get_their_usual_sound_and_shots_are_told_from_reloads() {
        assert_eq!(usual_fire_sound("models/weapons/c_models/c_scattergun.mdl", false), Some("sound/weapons/scatter_gun_shoot.wav"));
        assert_eq!(usual_fire_sound("models/workshop/weapons/c_models/c_crossbow/c_crossbow.mdl", false), Some("sound/weapons/crusaders_crossbow_shoot.wav"));
        assert_eq!(usual_fire_sound("models/weapons/c_models/c_directhit/c_directhit.mdl", false), Some("sound/weapons/rocket_directhit_shoot.wav"));
        assert_eq!(usual_fire_sound("models/weapons/c_models/c_frying_pan/c_frying_pan.mdl", true), Some(USUAL_SWING_SOUND));
        assert_eq!(usual_fire_sound("models/weapons/c_models/c_medigun/c_medigun.mdl", false), None);
        assert!(is_fire_sound(")weapons/pistol_shoot.wav") && is_fire_sound("weapons/wrench_swing.wav"));
        assert!(!is_fire_sound("weapons/shotgun_worldreload.wav") && !is_fire_sound(")weapons/pistol_shoot_crit.wav"));
        assert!(is_attack_activity("ACT_PRIMARY_VM_PRIMARYATTACK") && is_attack_activity("ACT_MELEE_VM_SWINGHARD"));
        assert!(!is_attack_activity("ACT_PRIMARY_VM_RELOAD") && !is_attack_activity("ACT_MELEE_VM_DRAW"));
    }

    #[test]
    fn kills_read_as_sentences() {
        let kill = Kill {
            attacker: Some(1),
            victim: Some(2),
            attacker_name: "Vegeta".into(),
            victim_name: "Oldbutter".into(),
            weapon: "ambassador".into(),
            ..Kill::default()
        };
        assert_eq!(describe_kill(&kill), "Vegeta killed Oldbutter (ambassador)");
        let fall = Kill { attacker: None, victim: Some(2), victim_name: "Oldbutter".into(), weapon: "world".into(), ..Kill::default() };
        assert_eq!(describe_kill(&fall), "Oldbutter died (world)");
    }

    #[test]
    fn trails_stop_short_of_the_camera() {
        let eye = Vec3::ZERO;
        // Starting beside the camera: only the far part is kept.
        let [first, second] = outside_sphere(Vec3::new(10.0, 0.0, 0.0), Vec3::new(200.0, 0.0, 0.0), eye, 48.0);
        let (from, to) = first.or(second).expect("the far part");
        assert!((from.x - 48.0).abs() < 1e-3 && to.x == 200.0, "{from:?} {to:?}");
        assert!(first.is_none() || second.is_none());
        // Passing by the camera: both ends are kept, the middle is cut out.
        let [near, far] = outside_sphere(Vec3::new(-100.0, 10.0, 0.0), Vec3::new(100.0, 10.0, 0.0), eye, 48.0);
        assert!(near.is_some() && far.is_some());
        // Far away: untouched.
        let far_away = (Vec3::new(500.0, 0.0, 0.0), Vec3::new(600.0, 0.0, 0.0));
        let kept: Vec<_> = outside_sphere(far_away.0, far_away.1, eye, 48.0).into_iter().flatten().collect();
        assert_eq!(kept, vec![far_away]);
        // Entirely inside: nothing.
        assert_eq!(outside_sphere(Vec3::X, Vec3::Y, eye, 48.0), [None, None]);
    }

    #[test]
    fn an_energy_ring_stands_across_its_path() {
        let mut floats = Vec::new();
        push_ring(&mut floats, Vec3::ZERO, Vec3::X, 10.0, [1.0, 0.0, 0.0], 1.0);
        assert!(!floats.is_empty());
        assert_eq!(floats.len() % TRACE_FLOATS, 0);
        for vertex in floats.chunks(TRACE_FLOATS) {
            let point = Vec3::new(vertex[0], vertex[1], vertex[2]);
            assert!(point.x.abs() < 1e-3, "flat across the direction of travel: {point:?}");
            // Bright along the ring itself, fading out to either side.
            let bright = (point.length() - 10.0).abs() < 1e-3;
            assert_eq!(vertex[6], if bright { 1.0 } else { 0.0 }, "{point:?}");
        }
    }

    #[test]
    fn a_glow_faces_the_camera_and_fades_out() {
        let mut floats = Vec::new();
        let eye = Vec3::new(0.0, -500.0, 0.0);
        push_glow(&mut floats, Vec3::ZERO, 8.0, eye, [1.0, 0.5, 0.1], 0.6);
        assert_eq!(floats.len(), GLOW_STEPS * 3 * TRACE_FLOATS);
        for vertex in floats.chunks(TRACE_FLOATS) {
            let point = Vec3::new(vertex[0], vertex[1], vertex[2]);
            assert!(point.y.abs() < 1e-3, "faces the camera: {point:?}");
            let center = point.length() < 1e-3;
            assert_eq!(vertex[6], if center { 0.6 } else { 0.0 });
        }
    }

    #[test]
    fn projectiles_are_followed_and_reused_ids_start_new_trails() {
        let mut recording = Recording {
            ticks: vec![10, 12, 14, 16, 18],
            tick_interval: 0.015,
            ..Recording::default()
        };
        // Rocket 7 flies for three frames, vanishes, then id 7 is reused for a new rocket.
        let frames: [&[ObjectFrame]; 5] = [
            &[projectile(7, 0.0)],
            &[projectile(7, 50.0)],
            &[projectile(7, 100.0)],
            &[],
            &[projectile(7, 500.0)],
        ];
        for objects in frames {
            recording.object_offsets.push(recording.objects.len() as u32);
            recording.objects.extend_from_slice(objects);
        }
        recording.object_offsets.push(recording.objects.len() as u32);

        let trails = build_trails(&recording);
        assert_eq!(trails.len(), 1, "the single sighting of the reused id has no path yet");
        assert_eq!(trails[0].ticks, vec![10, 12, 14]);
        assert_eq!(trails[0].points[2], Vec3::new(100.0, 0.0, 0.0));
    }

    #[test]
    fn an_id_taken_over_on_the_next_frame_is_a_new_projectile() {
        let mut recording = Recording {
            ticks: vec![10, 11, 12, 13],
            tick_interval: 0.015,
            ..Recording::default()
        };
        // Rocket 7 explodes and a rocket fired far away gets its number straight away.
        let frames: [&[ObjectFrame]; 4] = [
            &[projectile(7, 0.0)],
            &[projectile(7, 20.0)],
            &[projectile(7, 3000.0)],
            &[projectile(7, 3020.0)],
        ];
        for objects in frames {
            recording.object_offsets.push(recording.objects.len() as u32);
            recording.objects.extend_from_slice(objects);
        }
        recording.object_offsets.push(recording.objects.len() as u32);

        let trails = build_trails(&recording);
        assert_eq!(trails.len(), 2);
        assert_eq!(trails[0].points, vec![Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0)]);
        assert_eq!(trails[1].ticks, vec![12, 13]);
    }

    #[test]
    fn first_person_animations_follow_the_weapon_and_its_slot() {
        // A Demoman's sword falls back to how he holds melee weapons, not his sticky launcher.
        let sets = first_person_sets(WeaponPose::Item1, 2);
        assert_eq!(&sets[..2], &[WeaponPose::Item1, WeaponPose::Melee]);
        assert!(first_person_actions(Gesture::Attack, WeaponPose::Melee).contains(&"ACT_MELEE_VM_HITCENTER".to_string()));
        assert_eq!(first_person_actions(Gesture::ReloadEnd, WeaponPose::Primary), vec!["ACT_PRIMARY_RELOAD_FINISH".to_string()]);
        // Alternate attacks try their own animation, then the usual one.
        let alternate = first_person_actions(Gesture::AttackSecondary, WeaponPose::Primary);
        assert_eq!(alternate[0], "ACT_PRIMARY_VM_SECONDARYATTACK");
        assert!(alternate.contains(&"ACT_PRIMARY_VM_PRIMARYATTACK".to_string()));
        assert!(first_person_actions(Gesture::Attack, WeaponPose::Primary).iter().all(|name| !name.is_empty()));
    }

    #[test]
    fn a_bullet_line_stops_at_the_player_it_hit() {
        let shooter = PlayerFrame { class: 1, flags: super::super::recording::PLAYER_ALIVE, ..PlayerFrame::default() };
        let victim = PlayerFrame { position: [500.0, 0.0, 0.0], ..shooter };
        let mut recording = Recording {
            slot_count: 2,
            ticks: vec![100],
            players: vec![shooter, victim],
            shots: vec![super::super::recording::Shot { tick: 100, slot: 0, origin: [0.0, 0.0, 65.0], ..Default::default() }],
            hurts: vec![super::super::recording::Hurt { tick: 100, victim: Some(1), attacker: Some(0), damage: 50, ..Default::default() }],
            ..Recording::default()
        };
        let wall = |from: Vec3, to: Vec3| Some(from.lerp(to, 0.5));
        let ends = shot_ends(&recording, wall);
        assert!((ends[0].x - 500.0).abs() < 1.0 && ends[0].y.abs() < 1.0, "stops at the victim: {:?}", ends[0]);
        // Without the hit, it carries on to the wall.
        recording.hurts.clear();
        assert!(shot_ends(&recording, wall)[0].x > 1000.0);
        // A hit on someone well off the line is a different bullet, like splash or a second pellet.
        recording.players[1].position = [500.0, 300.0, 0.0];
        recording.hurts.push(super::super::recording::Hurt { tick: 100, victim: Some(1), attacker: Some(0), ..Default::default() });
        assert!(shot_ends(&recording, wall)[0].x > 1000.0);
    }

    #[test]
    fn buildings_and_carts_get_no_trail() {
        let mut sentry = projectile(3, 0.0);
        sentry.kind = ObjectKind::Sentry;
        let mut moved = sentry;
        moved.position[0] = 10.0;
        let recording = Recording {
            ticks: vec![1, 2],
            object_offsets: vec![0, 1, 2],
            objects: vec![sentry, moved],
            ..Recording::default()
        };
        assert!(build_trails(&recording).is_empty());
    }

    #[test]
    fn aim_speed_is_measured_between_view_updates() {
        // The view turns 10 degrees, holds still for a while, then turns 4 more.
        let ticks = [0, 1, 2, 3, 5];
        let angles = [(0.0, 0.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 14.0)];
        let speeds = aim_speeds(&ticks, &angles);
        assert_eq!(speeds.len(), 2);
        assert!((speeds[0].1 - 10.0).abs() < 1e-3);
        assert!((speeds[1].1 - 1.0).abs() < 1e-3, "4 degrees since the last change, four ticks earlier");
    }

    #[test]
    fn box_edges_follow_the_bone() {
        let bone = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let edges = box_edges(&bone, Vec3::splat(-1.0), Vec3::splat(1.0));
        assert_eq!(edges.len(), 12);
        for (a, b) in edges {
            assert!((a.distance(b) - 2.0).abs() < 1e-4, "every edge is 2 long");
            assert!((a.x - 10.0).abs() <= 1.0 + 1e-4 && (b.x - 10.0).abs() <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn ping_is_coloured_by_how_high_it_is() {
        assert_eq!(ping_label(45, false), ("45".to_string(), PING_GOOD));
        assert_eq!(ping_label(120, false).1, AMBER);
        assert_eq!(ping_label(220, false).1, PING_HIGH);
        assert_eq!(ping_label(0, false).0, "-", "not received yet");
        assert_eq!(ping_label(5, true).0, "BOT");
    }

    #[test]
    fn the_clock_shows_minutes_and_seconds() {
        assert_eq!(format_clock(65.9), "1:05");
        assert_eq!(format_clock(-3.0), "0:00");
    }
}
