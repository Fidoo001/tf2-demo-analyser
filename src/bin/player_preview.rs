//! Opens a demo straight in the 3D player, without the rest of the tool.
//!
//! `player_preview <demo.dem> [--tick N] [--slot N] [--near SLOT] [--kill N]
//! [--projectile N] [--traces all|bullets|projectiles|none] [--glow off|everyone|enemies|teammates]
//! [--find-state aiming|jump] [--show aim,hitboxes,ghosts,nodamage,noimpacts,nograph] [--free "x y z yaw pitch"]
//! [--markers cli_output.json] [--review STEAMID]
//! [--survey DIR [--filter Soldier,Jump] [--limit N] [--around DEGREES] [--shots TICK:SLOT,...]]
//! [--play SECONDS] [--shot out.png] [--info]`
//!
//! With `--shot` it saves one screenshot and exits, which is handy for checking
//! rendering without clicking through the app.

use std::path::PathBuf;

use analysis_template::player::{CameraMode, DemoPlayer, FreeCamera, GlowMode, Marker, OpenRequest, Session};
use eframe::egui;
use glam::Vec3;

const FRAMES_BEFORE_SHOT: u32 = 12;

#[derive(Default)]
struct Options {
    demo: PathBuf,
    tick: Option<u32>,
    slot: Option<usize>,
    free: Option<[f32; 5]>,
    near: Option<usize>,
    kill: Option<usize>,
    projectile: Option<usize>,
    /// Only projectiles of this kind for --projectile, like EnergyRing.
    projectile_kind: Option<String>,
    /// Watch the --slot player from behind instead of through their eyes.
    third: bool,
    /// Records what the player plays to this WAV file while it plays for `--play` seconds.
    sound: Option<PathBuf>,
    /// Prints every object of a kind frame by frame: (kind, from tick, to tick).
    dump: Option<(String, u32, u32)>,
    traces: Option<String>,
    find_state: Option<String>,
    glow: Option<String>,
    show: Vec<String>,
    steam_id: Option<u64>,
    report: Option<u64>,
    clip: Option<(f64, bool)>,
    find_weapon: Option<String>,
    play_seconds: Option<f32>,
    shot: Option<PathBuf>,
    info: bool,
    /// The analyser's JSON output, for detections on the timeline.
    markers: Option<PathBuf>,
    review: Option<u64>,
    /// Photographs one example of every class, stance and animation set into this folder.
    survey: Option<PathBuf>,
    /// Only survey poses whose description contains one of these.
    filter: Vec<String>,
    limit: Option<usize>,
    around: Option<f32>,
    /// Exact moments to photograph instead: (tick, slot).
    shots: Vec<(u32, usize)>,
    /// Prints a player's movement frame by frame: (slot, from tick, to tick).
    trace: Option<(usize, u32, u32)>,
    /// Lists every kill: tick, victim slot, attacker slot and weapon.
    kills: bool,
    /// Lists buildings: when each appears, where, and how it animates.
    buildings: bool,
    /// Survey shots through the player's own eyes instead of from outside.
    pov: bool,
    /// Prints one building's animation frame by frame: (id, from tick, to tick).
    object: Option<(u16, u32, u32)>,
}

fn parse_options() -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--tick" => options.tick = value()?.parse().ok(),
            "--slot" => options.slot = value()?.parse().ok(),
            "--near" => options.near = value()?.parse().ok(),
            "--kill" => options.kill = value()?.parse().ok(),
            "--projectile" => options.projectile = value()?.parse().ok(),
            "--projectile-kind" => options.projectile_kind = Some(value()?),
            "--sound" => options.sound = Some(PathBuf::from(value()?)),
            "--third" => options.third = true,
            "--dump" => {
                let text = value()?;
                let parts: Vec<&str> = text.split(':').collect();
                if let [kind, from, to] = parts[..] {
                    options.dump = Some((kind.to_string(), from.parse().map_err(|_| "bad tick")?, to.parse().map_err(|_| "bad tick")?));
                }
            }
            "--traces" => options.traces = Some(value()?),
            "--find-state" => options.find_state = Some(value()?),
            "--glow" => options.glow = Some(value()?),
            "--show" => options.show.extend(value()?.split(',').map(str::to_string)),
            "--steam" => options.steam_id = value()?.parse().ok(),
            "--report" => options.report = value()?.parse().ok(),
            "--clip" => {
                let text = value()?;
                let (seconds, format) = text.split_once(':').unwrap_or((text.as_str(), "mp4"));
                options.clip = seconds.parse().ok().map(|s| (s, format == "mp4"));
            }
            "--find-weapon" => options.find_weapon = Some(value()?),
            "--play" => options.play_seconds = value()?.parse().ok(),
            "--shot" => options.shot = Some(PathBuf::from(value()?)),
            "--free" => {
                let numbers: Vec<f32> = value()?
                    .split_whitespace()
                    .filter_map(|part| part.parse().ok())
                    .collect();
                options.free = numbers.try_into().ok();
            }
            "--info" => options.info = true,
            "--kills" => options.kills = true,
            "--buildings" => options.buildings = true,
            "--pov" => options.pov = true,
            "--object" => {
                let text = value()?;
                let parts: Vec<&str> = text.split(':').collect();
                if let [id, from, to] = parts[..] {
                    options.object = Some((id.parse().map_err(|_| "bad id")?, from.parse().map_err(|_| "bad tick")?, to.parse().map_err(|_| "bad tick")?));
                }
            }
            "--markers" => options.markers = Some(PathBuf::from(value()?)),
            "--review" => options.review = value()?.parse().ok(),
            "--survey" => options.survey = Some(PathBuf::from(value()?)),
            "--filter" => options.filter.extend(value()?.split(',').map(str::to_string)),
            "--limit" => options.limit = value()?.parse().ok(),
            "--around" => options.around = value()?.parse().ok(),
            "--trace" => {
                let text = value()?;
                let parts: Vec<&str> = text.split(':').collect();
                if let [slot, from, to] = parts[..] {
                    options.trace = Some((slot.parse().map_err(|_| "bad slot")?, from.parse().map_err(|_| "bad tick")?, to.parse().map_err(|_| "bad tick")?));
                }
            }
            "--shots" => {
                for item in value()?.split(',') {
                    let (tick, slot) = item.split_once(':').ok_or("--shots wants tick:slot,tick:slot")?;
                    options.shots.push((tick.parse().map_err(|_| "bad tick")?, slot.parse().map_err(|_| "bad slot")?));
                }
            }
            path => options.demo = PathBuf::from(path),
        }
    }
    if options.demo.as_os_str().is_empty() {
        return Err("usage: player_preview <demo.dem> [--tick N] [--slot N] [--shot out.png]".into());
    }
    Ok(options)
}

/// One moment to photograph in a survey.
struct SurveyShot {
    tick: u32,
    slot: usize,
    details: String,
}

/// Photographs players one after another, cropped around them.
struct Survey {
    folder: PathBuf,
    shots: Vec<SurveyShot>,
    index: usize,
    frames: u32,
    requested: bool,
    lines: Vec<String>,
}

/// One example of each pose the demo shows, found by stepping through it.
fn plan_survey(session: &mut Session, filter: &[String], limit: usize) -> Vec<SurveyShot> {
    let recording = session.recording();
    let interval = f64::from(recording.tick_interval);
    let (first, last) = (f64::from(recording.first_tick()), f64::from(recording.last_tick()));
    let slots = recording.slot_count;
    let mut found: std::collections::BTreeMap<String, SurveyShot> = std::collections::BTreeMap::new();
    let mut tick = first + 2.0 / interval;
    while tick < last {
        session.seek(tick);
        for slot in 0..slots {
            let Some((key, details)) = session.pose_summary(slot) else { continue };
            let wanted = filter.is_empty() || filter.iter().any(|part| key.contains(part.as_str()));
            if wanted && !found.contains_key(&key) {
                found.insert(key, SurveyShot { tick: tick as u32, slot, details });
            }
        }
        tick += 0.25 / interval;
    }
    println!("survey: {} different poses found", found.len());
    found.into_values().take(limit).collect()
}

/// Puts the free camera in front of a player, turned about `around` degrees, looking back
/// at them, from the first angle and distance with nothing in the way.
fn look_at_player(session: &mut Session, slot: usize, around: f32) {
    let recording = session.recording();
    let now = recording.frame_at(session.tick());
    // A dead player is watched where they died, from further back to see the body fall.
    let alive_at = (0..=now).rev().take(600).find(|&frame| recording.player(frame, slot).is_alive()).unwrap_or(now);
    let dead = alive_at != now;
    let player = *recording.player(alive_at, slot);
    let chest = Vec3::from(player.position) + Vec3::new(0.0, 0.0, if dead { 20.0 } else { 45.0 });
    let distances: &[f32] = if dead { &[230.0, 180.0, 130.0] } else { &[110.0, 85.0, 65.0] };
    let mut best = (around, distances[0]);
    'search: for &distance in distances {
        for turn in [0.0_f32, 50.0, -50.0, 100.0, -100.0, 150.0, -150.0] {
            let yaw = (player.yaw + around + turn).to_radians();
            let eye = chest + Vec3::new(yaw.cos(), yaw.sin(), 0.0) * distance + Vec3::new(0.0, 0.0, 5.0);
            if session.line_is_clear(eye, chest) {
                best = (around + turn, distance);
                break 'search;
            }
        }
    }
    let (turn, distance) = best;
    let yaw = (player.yaw + turn).to_radians();
    let facing = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
    let (height, pitch) = if dead { (110.0, 22.0) } else { (50.0, 8.0) };
    session.free_camera = FreeCamera {
        position: Vec3::from(player.position) + facing * distance + Vec3::new(0.0, 0.0, height),
        yaw: player.yaw + turn + 180.0,
        pitch,
    };
    session.camera_mode = CameraMode::Free;
}

struct App {
    survey: Option<Survey>,
    player: DemoPlayer,
    options: Options,
    configured: bool,
    frames_ready: u32,
    played: f32,
    shot_requested: bool,
    script: Option<analysis_template::player::Script>,
}

impl App {
    fn configure(&mut self) {
        let Some(session) = self.player.session_mut() else { return };
        if self.configured {
            return;
        }
        self.configured = true;

        if self.options.info {
            println!("sound: {}", session.sound_summary());
            let recording = session.recording();
            println!("map {}  recorded by {}", recording.map, recording.recorded_by);
            println!(
                "ticks {}..{}  frames {}  slots {}  interval {}",
                recording.first_tick(),
                recording.last_tick(),
                recording.ticks.len(),
                recording.slot_count,
                recording.tick_interval
            );
            println!(
                "kills {}  rounds {}  chat {}  objects {}  recorder slot {:?}  sky {:?}",
                recording.kills.len(),
                recording.rounds.len(),
                recording.chat.len(),
                recording.objects.len(),
                recording.recorder_slot,
                recording.sky_name
            );
            let middle = recording.ticks.len() / 2;
            for slot in 0..recording.slot_count {
                let player = recording.player(middle, slot);
                if player.is_on_a_team() {
                    println!(
                        "  slot {slot:2} team {} class {} alive {} away {} {:?} yaw {:.0} pitch {:.0} hp {} {}",
                        player.team,
                        player.class,
                        player.is_alive(),
                        player.is_out_of_view(),
                        player.position.map(|v| v as i32),
                        player.yaw,
                        player.pitch,
                        player.health,
                        recording.name_at(slot, f64::from(recording.ticks[middle]))
                    );
                }
            }
            println!("view at middle {:?}", recording.views[middle]);
            println!("models: {}", recording.models.len());
            println!("map objects {}, shots {}", recording.props.len(), recording.shots.len());
            let mut events = std::collections::BTreeMap::new();
            for anim in &recording.anims {
                *events.entry(anim.event).or_insert(0) += 1;
            }
            println!("hurts {}, animation events by kind {:?}", recording.hurts.len(), events);
            // Busy moments: shots by players in view, a tenth of the way through and the middle.
            for share in [10, 50] {
                let index = recording.shots.len() * share / 100;
                if let Some(shot) = recording.shots.get(index) {
                    println!("  shot {index} at tick {} by slot {} crit {}", shot.tick, shot.slot, shot.crit);
                }
            }
            for slot in 0..recording.slot_count {
                let zoomed: Vec<u32> = (0..recording.ticks.len())
                    .filter(|&frame| recording.player(frame, slot).is_zoomed())
                    .map(|frame| recording.ticks[frame])
                    .collect();
                if let Some(first) = zoomed.first() {
                    println!("  slot {slot} scoped in {} frames, first at tick {first}, middle at {}", zoomed.len(), zoomed[zoomed.len() / 2]);
                }
            }
            for track in recording.props.iter().take(12) {
                println!("  object {} keys {} until {}", recording.models[usize::from(track.model)], track.keys.len(), track.until);
            }
            println!("view arms {} sequence {}", recording.models[usize::from(recording.views[middle].arms)], recording.views[middle].sequence);
            let middle_frame = &recording.players[middle * recording.slot_count..(middle + 1) * recording.slot_count];
            println!(
                "weapons at middle {:?}",
                middle_frame.iter().map(|p| (p.weapon_slot, p.weapon_model, p.flags)).collect::<Vec<_>>()
            );
        }

        if self.options.info {
            let recording = session.recording();
            let frame = recording.frame_at(session.tick());
            let view = recording.views[frame];
            println!("view at start tick {}: {:?}", recording.ticks[frame], view);
            if let Some(slot) = recording.recorder_slot {
                let player = recording.player(frame, slot);
                println!("recorder body {:?} alive {} flags {} class {}", player.position, player.is_alive(), player.flags, player.class);
            }
            for slot in self.options.slot.iter().copied() {
                let player = recording.player(frame, slot);
                println!(
                    "watched slot {slot}: class {} weapon slot {} model {}",
                    player.class,
                    player.weapon_slot,
                    recording.models[usize::from(player.weapon_model)]
                );
            }
        }
        if let Some(wanted) = &self.options.find_weapon {
            let recording = session.recording();
            let mut found = 0;
            for frame in (0..recording.ticks.len()).step_by(66) {
                for slot in 0..recording.slot_count {
                    let player = recording.player(frame, slot);
                    let model = &recording.models[usize::from(player.weapon_model)];
                    if player.is_alive() && !player.is_out_of_view() && model.contains(wanted.as_str()) && found < 6 {
                        println!("found {model} with slot {slot} (class {}) at tick {}", player.class, recording.ticks[frame]);
                        found += 1;
                    }
                }
            }
        }
        if self.options.find_state.as_deref() == Some("reload") {
            let recording = session.recording();
            let mut found = 0;
            for anim in recording.anims.iter().filter(|anim| anim.event == 3) {
                let frame = recording.frame_at(f64::from(anim.tick));
                let player = recording.player(frame, anim.slot);
                if player.is_alive() && !player.is_out_of_view() && player.class == 3 && found < 4 {
                    println!("state reload slot {} class {} at tick {}", anim.slot, player.class, anim.tick);
                    found += 1;
                }
            }
        }
        if let Some(state) = &self.options.find_state {
            // Moments a player in view is aiming (revved, scoped, drawing a bow) or jumps.
            let recording = session.recording();
            let mut found = 0;
            let mut last_hit = vec![0u32; recording.slot_count];
            for frame in 1..recording.ticks.len() {
                for slot in 0..recording.slot_count {
                    let (before, player) = (recording.player(frame - 1, slot), recording.player(frame, slot));
                    if !player.is_alive() || player.is_out_of_view() || found >= 8 {
                        continue;
                    }
                    let tick = recording.ticks[frame];
                    let hit = match state.as_str() {
                        "aiming" => player.is_aiming() && !before.is_aiming(),
                        "jump" => before.is_on_ground() && !player.is_on_ground() && player.position[2] - before.position[2] > 3.0,
                        _ => false,
                    };
                    if hit && tick > last_hit[slot] + 2000 {
                        last_hit[slot] = tick;
                        println!("state {state} slot {slot} class {} at tick {tick}", player.class);
                        found += 1;
                    }
                }
            }
        }
        session.third_person = self.options.third;
        if let Some(slot) = self.options.slot {
            session.camera_mode = CameraMode::Eyes(slot);
        }
        if let Some([x, y, z, yaw, pitch]) = self.options.free {
            session.free_camera = FreeCamera {
                position: Vec3::new(x, y, z),
                yaw,
                pitch,
            };
            session.camera_mode = CameraMode::Free;
        }
        if let Some(slot) = self.options.near {
            // Stand in front of the player and look back at them.
            let recording = session.recording();
            let player = *recording.player(recording.frame_at(session.tick()), slot);
            let around: f32 = std::env::var("DP_AROUND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let yaw = (player.yaw + around).to_radians();
            let facing = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
            session.free_camera = FreeCamera {
                position: Vec3::from(player.position) + facing * 110.0 + Vec3::new(0.0, 0.0, 50.0),
                yaw: player.yaw + around + 180.0,
                pitch: 8.0,
            };
            println!("near camera {:?} yaw {:.0}", session.free_camera.position, session.free_camera.yaw);
            session.camera_mode = CameraMode::Free;
        }
        if let Some((seconds, mp4)) = self.options.clip {
            session.record_clip(seconds, mp4);
        }
        if let Some(steam_id) = self.options.report {
            println!("report: {:?}", session.save_report(steam_id));
        }
        for item in &self.options.show {
            match item.as_str() {
                "aim" => session.show_aim_lines = true,
                "hitboxes" => session.show_hitboxes = true,
                "ghosts" => session.show_ghosts = true,
                "nodamage" => session.show_damage = false,
                "noimpacts" => session.show_impacts = false,
                "nograph" => session.show_aim_graph = false,
                other => eprintln!("unknown --show item {other}"),
            }
        }
        if let Some(glow) = &self.options.glow {
            session.glow = match glow.as_str() {
                "off" => GlowMode::Off,
                "enemies" => GlowMode::Enemies,
                "teammates" => GlowMode::Teammates,
                _ => GlowMode::Everyone,
            };
        }
        if let Some(traces) = &self.options.traces {
            session.show_bullets = matches!(traces.as_str(), "all" | "bullets");
            session.show_projectiles = matches!(traces.as_str(), "all" | "projectiles");
        }
        if let Some((kind, from, to)) = &self.options.dump {
            let recording = session.recording();
            for frame in 0..recording.ticks.len() {
                let tick = recording.ticks[frame];
                if tick < *from || tick > *to {
                    continue;
                }
                for object in recording.objects_at(frame) {
                    if format!("{:?}", object.kind) == *kind {
                        println!(
                            "dump tick {tick} id {} team {} at {:?} pitch {:.0} yaw {:.0} model {}",
                            object.id, object.team, object.position, object.pitch, object.yaw, object.model
                        );
                    }
                }
            }
        }
        if let Some(number) = self.options.projectile {
            // Partway through a long flight, seen from the side.
            let recording = session.recording();
            let mut runs: std::collections::HashMap<u16, (usize, Vec<Vec3>, String)> = Default::default();
            let mut found = None;
            let mut seen = 0;
            'frames: for frame in 0..recording.ticks.len() {
                for object in recording.objects_at(frame) {
                    if object.kind.is_building() || format!("{:?}", object.kind) == "Cart" {
                        continue;
                    }
                    if self.options.projectile_kind.as_ref().is_some_and(|kind| *kind != format!("{:?}", object.kind)) {
                        continue;
                    }
                    let entry = runs.entry(object.id).or_insert((frame, Vec::new(), String::new()));
                    if entry.0 + entry.1.len() != frame {
                        *entry = (frame, Vec::new(), String::new());
                    }
                    entry.1.push(Vec3::from(object.position));
                    entry.2 = format!("{:?} team {}", object.kind, object.team);
                    let path = &entry.1;
                    // Rare kinds are often only seen for a moment, so a short flight will do.
                    let needed = if self.options.projectile_kind.is_some() { 5 } else { 14 };
                    if path.len() == needed && path[0].distance(path[needed - 1]) > 60.0 {
                        if seen == number {
                            found = Some((recording.ticks[frame], path.clone(), entry.2.clone()));
                            break 'frames;
                        }
                        seen += 1;
                    }
                }
            }
            if found.is_none() {
                let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
                for object in &recording.objects {
                    *kinds.entry(format!("{:?}", object.kind)).or_default() += 1;
                }
                let longest = runs.values().map(|run| run.1.len()).max().unwrap_or(0);
                println!("no long flight found; objects by kind {kinds:?}, longest run {longest} frames");
            }
            if let Some((tick, path, what)) = found {
                let (first, last) = (path[0], path[path.len() - 1]);
                let along = (last - first).normalize_or_zero();
                let side = along.cross(Vec3::Z).normalize_or_zero();
                // Just behind and beside where it was fired, which is open space.
                let eye = first - along * 140.0 + side * 60.0 + Vec3::Z * 50.0;
                let look = (last - eye).normalize_or_zero();
                println!("{what} at tick {tick} flying from {first:?} to {last:?}");
                session.seek(f64::from(tick));
                session.free_camera = FreeCamera {
                    position: eye,
                    yaw: look.y.atan2(look.x).to_degrees(),
                    pitch: -look.z.asin().to_degrees(),
                };
                session.camera_mode = CameraMode::Free;
            }
        }
        if let Some(number) = self.options.kill {
            // A second after a kill, looking at where the victim fell.
            let recording = session.recording();
            let found = recording
                .kills
                .iter()
                .filter(|kill| kill.victim.is_some() && !kill.feigned)
                .filter(|kill| {
                    let frame = recording.frame_at(f64::from(kill.tick)).saturating_sub(1);
                    !recording.player(frame, kill.victim.unwrap_or(0)).is_out_of_view()
                })
                .nth(number)
                .map(|kill| {
                    let frame = recording.frame_at(f64::from(kill.tick)).saturating_sub(1);
                    (kill.tick, *recording.player(frame, kill.victim.unwrap_or(0)), kill.weapon.clone())
                });
            if let Some((tick, victim, weapon)) = found {
                println!("kill at tick {tick} by {weapon}, victim class {} at {:?}", victim.class, victim.position);
                session.seek(f64::from(tick) + std::env::var("DP_KILL_DELAY").ok().and_then(|v| v.parse().ok()).unwrap_or(70.0));
                session.free_camera = FreeCamera {
                    position: Vec3::from(victim.position) + Vec3::new(-150.0, 0.0, 90.0),
                    yaw: 0.0,
                    pitch: 25.0,
                };
                session.camera_mode = CameraMode::Free;
            }
        }
        if self.options.play_seconds.is_some() {
            session.set_playing(true);
        }
        if self.options.sound.is_some() {
            println!("sound recording: {} from tick {:.0}", session.start_sound_recording(), session.tick());
        }
        if let Some(steam_id) = self.options.review {
            println!("review started: {}", session.start_review(steam_id));
        }
        if let Some((id, from, to)) = self.options.object {
            let recording = session.recording();
            for frame in 0..recording.ticks.len() {
                let tick = recording.ticks[frame];
                if tick < from || tick > to {
                    continue;
                }
                for object in recording.objects_at(frame).iter().filter(|object| object.id == id) {
                    println!("object {tick} level {} model {} {:?}", object.level, recording.models[usize::from(object.model)], object.animation);
                }
            }
        }
        if self.options.buildings {
            let shots = session.sentry_shot_list();
            println!("sentry shots: {}", shots.len());
            for (tick, id, from) in shots.iter().step_by((shots.len() / 12).max(1)) {
                println!("sentry shot {tick} by {id} from {:.0?}", from.to_array());
            }
            let recording = session.recording();
            let mut seen: std::collections::HashMap<u16, (u32, u32, usize)> = std::collections::HashMap::new();
            for frame in 0..recording.ticks.len() {
                for object in recording.objects_at(frame) {
                    if !object.kind.is_building() {
                        continue;
                    }
                    let entry = seen.entry(object.id).or_insert((recording.ticks[frame], 0, 0));
                    entry.1 = recording.ticks[frame];
                    entry.2 += 1;
                    if entry.2 % 400 == 1 {
                        println!(
                            "building {} {:?} tick {} at {:.0?} yaw {:.0} model {} building {} {:?}",
                            object.id,
                            object.kind,
                            recording.ticks[frame],
                            object.position,
                            object.yaw,
                            recording.models[usize::from(object.model)],
                            object.under_construction,
                            object.animation
                        );
                    }
                }
            }
        }
        if self.options.kills {
            for kill in &session.recording().kills {
                println!(
                    "kill {} victim {:?} attacker {:?} {} headshot {} backstab {}",
                    kill.tick, kill.victim, kill.attacker, kill.weapon, kill.headshot, kill.backstab
                );
            }
        }
        if let Some((slot, from, to)) = self.options.trace {
            let recording = session.recording();
            for frame in 0..recording.ticks.len() {
                let tick = recording.ticks[frame];
                if tick < from || tick > to {
                    continue;
                }
                let player = recording.player(frame, slot);
                let events: Vec<u8> = recording.anims.iter().filter(|a| a.slot == slot && a.tick == tick).map(|a| a.event).collect();
                println!(
                    "trace {tick} at {:.0} {:.0} {:.1} yaw {:.0} pitch {:.0} ground {} duck {} alive {} weapon {} slot {} events {events:?}",
                    player.position[0],
                    player.position[1],
                    player.position[2],
                    player.yaw,
                    player.pitch,
                    player.is_on_ground(),
                    player.is_ducking(),
                    player.is_alive(),
                    recording.models[usize::from(player.weapon_model)],
                    player.weapon_slot
                );
            }
        }
        if let Some(folder) = self.options.survey.clone() {
            let shots = if self.options.shots.is_empty() {
                plan_survey(session, &self.options.filter, self.options.limit.unwrap_or(60))
            } else {
                self.options
                    .shots
                    .iter()
                    .map(|&(tick, slot)| {
                        session.seek(f64::from(tick));
                        let details = session.pose_summary(slot).map_or_else(|| "not visible".to_string(), |(_, details)| details);
                        SurveyShot { tick, slot, details }
                    })
                    .collect()
            };
            let _ = std::fs::create_dir_all(&folder);
            self.survey = Some(Survey { folder, shots, index: 0, frames: 0, requested: false, lines: Vec::new() });
        }
    }

    /// Photographs each survey pose in turn, then writes an index and closes.
    fn drive_survey(&mut self, ctx: &egui::Context) {
        let Some(survey) = &mut self.survey else { return };
        ctx.request_repaint();
        if !self.player.is_ready() {
            return;
        }
        let around = self.options.around.unwrap_or(40.0);
        let Some(session) = self.player.session_mut() else { return };
        if survey.index >= survey.shots.len() {
            let _ = std::fs::write(survey.folder.join("index.txt"), survey.lines.join("\n"));
            println!("survey saved {} shots to {}", survey.lines.len(), survey.folder.display());
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        let shot = &survey.shots[survey.index];
        if survey.requested {
            let image = ctx.input(|input| {
                input.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            let Some(image) = image else { return };
            let scale = ctx.pixels_per_point();
            let view = session.view_rect();
            let side = (view.width().min(view.height()) * 0.85 * scale) as u32;
            // Through the eyes, the weapon sits low on the right.
            let center = if self.options.pov { view.center() + egui::vec2(view.width() * 0.18, view.height() * 0.07) } else { view.center() };
            let left = ((center.x * scale) as u32).saturating_sub(side / 2);
            let top = ((center.y * scale) as u32).saturating_sub(side / 2);
            let [width, height] = image.size;
            let bytes: Vec<u8> = image.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
            if let Some(full) = image::RgbaImage::from_raw(width as u32, height as u32, bytes) {
                let side = side.min(width as u32 - left).min(height as u32 - top);
                let cell = image::imageops::crop_imm(&full, left, top, side, side).to_image();
                let cell = image::imageops::resize(&cell, 360, 360, image::imageops::FilterType::Triangle);
                let _ = cell.save(survey.folder.join(format!("{:03}.png", survey.index)));
            }
            survey.lines.push(format!("{:03}\t{}\t{}\t{}", survey.index, shot.tick, shot.slot, shot.details));
            survey.index += 1;
            survey.frames = 0;
            survey.requested = false;
            return;
        }
        if survey.frames == 0 {
            session.glow = GlowMode::Off;
            session.set_playing(false);
            session.seek(f64::from(shot.tick));
            if self.options.pov {
                session.camera_mode = CameraMode::Eyes(shot.slot);
            } else {
                look_at_player(session, shot.slot, around);
            }
        }
        survey.frames += 1;
        if survey.frames >= 4 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            survey.requested = true;
        }
    }

    fn save_screenshots(&mut self, ctx: &egui::Context) {
        if self.options.clip.is_some() {
            if self.configured {
                if let Some((message, ok)) = self.player.session_mut().and_then(|session| session.clip_outcome()) {
                    println!("clip: {message} ({ok})");
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            return;
        }
        let Some(path) = self.options.shot.clone() else { return };
        let shot = ctx.input(|input| {
            input.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = shot {
            let [width, height] = image.size;
            let bytes: Vec<u8> = image.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
            let saved = image::RgbaImage::from_raw(width as u32, height as u32, bytes)
                .map(|buffer| buffer.save(&path));
            println!("saved {} ({:?})", path.display(), saved.map(|result| result.is_ok()));
            if let Some(session) = self.player.session_mut() {
                let view = session.view_rect();
                println!("view rect {} {} {} {} scale {}", view.left(), view.top(), view.right(), view.bottom(), ctx.pixels_per_point());
            }
            if let Some(session) = self.player.session_mut() {
                println!("review at shot: {:?} tick {:.0}", session.review_state(), session.tick());
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        if let Some(failure) = self.player.failure() {
            println!("failed: {failure}");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if !self.player.is_ready() {
            return;
        }
        if let Some(seconds) = self.options.play_seconds {
            self.played += ctx.input(|input| input.stable_dt);
            if self.played < seconds {
                return;
            }
        }
        if let Some(path) = self.options.sound.take() {
            if let Some((rate, samples)) = self.player.session_mut().and_then(|session| session.take_sound_recording()) {
                let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
                let loud = samples.iter().filter(|sample| sample.abs() > 0.01).count();
                if let Some(session) = self.player.session_mut() {
                    println!("sound ended at tick {:.0}", session.tick());
                }
                println!(
                    "sound: {:.1} s at {rate} Hz, peak {peak:.2}, {:.0}% not silent",
                    samples.len() as f64 / 2.0 / f64::from(rate.max(1)),
                    loud as f64 * 100.0 / samples.len().max(1) as f64
                );
                let pcm: Vec<i16> = samples.iter().map(|sample| (sample * 32767.0) as i16).collect();
                println!("saved {} ({:?})", path.display(), std::fs::write(&path, stereo_wav(rate, &pcm)).is_ok());
            }
        }
        self.frames_ready += 1;
        if self.frames_ready > FRAMES_BEFORE_SHOT && !self.shot_requested {
            self.shot_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
    }
}

impl eframe::App for App {
    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        if let Some(script) = &mut self.script {
            script.feed(ctx, raw_input);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.player.show(ctx) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        self.configure();
        if self.survey.is_some() {
            self.drive_survey(ctx);
            return;
        }
        self.save_screenshots(ctx);
        if self.options.shot.is_some() {
            ctx.request_repaint();
        }
    }
}

/// 16-bit stereo PCM as a WAV file.
fn stereo_wav(rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let mut file = Vec::with_capacity(44 + samples.len() * 2);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(36 + data_size).to_le_bytes());
    file.extend_from_slice(b"WAVEfmt ");
    file.extend_from_slice(&16u32.to_le_bytes());
    file.extend_from_slice(&1u16.to_le_bytes());
    file.extend_from_slice(&2u16.to_le_bytes());
    file.extend_from_slice(&rate.to_le_bytes());
    file.extend_from_slice(&(rate * 4).to_le_bytes());
    file.extend_from_slice(&4u16.to_le_bytes());
    file.extend_from_slice(&16u16.to_le_bytes());
    file.extend_from_slice(b"data");
    file.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        file.extend_from_slice(&sample.to_le_bytes());
    }
    file
}

/// Detections from the analyser's JSON output (`cli -i demo -q -n`), or a plain list of them.
fn read_markers(path: &std::path::Path) -> Result<Vec<Marker>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let start = text.find(['{', '[']).ok_or("no JSON in the markers file")?;
    let value: serde_json::Value = serde_json::from_str(&text[start..]).map_err(|error| error.to_string())?;
    let list = value.get("detections").cloned().unwrap_or(value);
    let detections = list.as_array().ok_or("the markers file has no list of detections")?;
    let id = |value: &serde_json::Value| value.as_u64().or_else(|| value.as_str().and_then(|text| text.parse().ok()));
    Ok(detections
        .iter()
        .filter_map(|detection| {
            let algorithm = detection.get("algorithm")?.as_str()?.to_string();
            Some(Marker {
                tick: u32::try_from(detection.get("tick")?.as_u64()?).ok()?,
                label: algorithm.clone(),
                steam_id: id(detection.get("player")?)?,
                check: algorithm,
                data: detection.get("data").cloned().unwrap_or_default(),
            })
        })
        .collect())
}

fn main() -> Result<(), String> {
    let options = parse_options()?;
    let mut markers = match &options.markers {
        Some(path) => read_markers(path)?,
        None => Vec::new(),
    };
    markers.sort_by_key(|marker| marker.tick);
    println!("markers: {}", markers.len());
    let request = OpenRequest {
        demo: options.demo.clone(),
        tick: options.tick,
        steam_id: options.steam_id,
        markers,
    };
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1360.0, 860.0]),
        ..eframe::NativeOptions::default()
    };
    eframe::run_native(
        "Demo Player",
        native,
        Box::new(move |context| {
            context.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(App {
                survey: None,
                player: DemoPlayer::open(request),
                options,
                configured: false,
                frames_ready: 0,
                played: 0.0,
                shot_requested: false,
                script: analysis_template::player::Script::from_env(),
            }))
        }),
    )
    .map_err(|error| error.to_string())
}
