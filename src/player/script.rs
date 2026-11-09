//! Plays a list of input steps into the window and saves screenshots, so the user
//! guide's pictures can be taken again after the interface changes. While a script
//! runs, the real mouse and keyboard are ignored.
//!
//! Set `DEMO_ANALYSIS_SCRIPT` to a text file with one step per line:
//!
//! ```text
//! size 1360 860          resize the window (points)
//! wait 2.5               do nothing for 2.5 seconds
//! drop C:\demos\a.dem    drop a file on the window
//! move 640 400           move the mouse (screenshot pixels)
//! click 640 400          left click; `rclick` right clicks
//! drag 600 400 700 420   hold the left button from one point to another
//! key Space              press a key; `key Shift+ArrowRight` with modifiers
//! type hello             type text
//! scroll -120            turn the mouse wheel (points; negative scrolls down)
//! shot out.png           save a screenshot
//! close                  close the window
//! ```
//!
//! Lines starting with # are comments.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

/// The environment variable naming the script file.
pub const SCRIPT_VARIABLE: &str = "DEMO_ANALYSIS_SCRIPT";

/// Frames a drag takes from start to end.
const DRAG_FRAMES: u8 = 12;

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Size(f32, f32),
    Wait(Duration),
    Drop(PathBuf),
    Move(egui::Pos2),
    Click(egui::Pos2, egui::PointerButton),
    Drag(egui::Pos2, egui::Pos2),
    Key(egui::Key, egui::Modifiers),
    Type(String),
    Scroll(f32),
    Shot(PathBuf),
    Close,
}

/// A running script.
pub struct Script {
    steps: VecDeque<Step>,
    /// When the current wait ends.
    until: Option<Instant>,
    /// How far into the current click or drag we are, in frames.
    stage: u8,
    /// Where a screenshot that was asked for, and hasn't arrived yet, goes.
    shot: Option<PathBuf>,
    pointer: egui::Pos2,
}

impl Script {
    /// The script named by `DEMO_ANALYSIS_SCRIPT`, if it's set.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var_os(SCRIPT_VARIABLE)?;
        let text = std::fs::read_to_string(&path)
            .map_err(|error| eprintln!("script {}: {error}", PathBuf::from(&path).display()))
            .ok()?;
        match parse(&text) {
            Ok(steps) => Some(Self { steps: steps.into(), until: None, stage: 0, shot: None, pointer: egui::Pos2::ZERO }),
            Err(error) => {
                eprintln!("script: {error}");
                None
            }
        }
    }

    /// Replaces this frame's real mouse and keyboard input with the script's.
    /// Call it from `eframe::App::raw_input_hook`.
    pub fn feed(&mut self, ctx: &egui::Context, raw: &mut egui::RawInput) {
        ctx.request_repaint();
        raw.focused = true;
        raw.modifiers = egui::Modifiers::NONE;
        let arrived = raw.events.iter().find_map(|event| match event {
            egui::Event::Screenshot { image, .. } => Some(image.clone()),
            _ => None,
        });
        raw.events.retain(|event| matches!(event, egui::Event::Screenshot { .. }));
        raw.dropped_files.clear();
        raw.hovered_files.clear();

        if let Some(path) = self.shot.take() {
            match arrived {
                Some(image) => {
                    save(&path, &image);
                    self.steps.pop_front();
                }
                None => self.shot = Some(path),
            }
            return;
        }
        let scale = ctx.pixels_per_point();
        let Some(step) = self.steps.front().cloned() else { return };
        let mut done = true;
        let takes_frames = matches!(step, Step::Click(..) | Step::Drag(..));
        match step {
            Step::Size(width, height) => ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(width, height))),
            Step::Wait(duration) => {
                let until = *self.until.get_or_insert_with(|| Instant::now() + duration);
                done = Instant::now() >= until;
                if done {
                    self.until = None;
                }
            }
            Step::Drop(path) => raw.dropped_files.push(egui::DroppedFile { path: Some(path), ..Default::default() }),
            Step::Move(at) => {
                self.pointer = (at.to_vec2() / scale).to_pos2();
                raw.events.push(egui::Event::PointerMoved(self.pointer));
            }
            Step::Click(at, button) => {
                self.pointer = (at.to_vec2() / scale).to_pos2();
                let pos = self.pointer;
                // Move, press and release on separate frames, as a real click would.
                raw.events.push(match self.stage {
                    0 => egui::Event::PointerMoved(pos),
                    stage => egui::Event::PointerButton { pos, button, pressed: stage == 1, modifiers: egui::Modifiers::NONE },
                });
                done = self.stage == 2;
            }
            Step::Drag(from, to) => {
                let from = (from.to_vec2() / scale).to_pos2();
                let to = (to.to_vec2() / scale).to_pos2();
                let button = egui::PointerButton::Primary;
                let modifiers = egui::Modifiers::NONE;
                match self.stage {
                    0 => raw.events.push(egui::Event::PointerMoved(from)),
                    1 => raw.events.push(egui::Event::PointerButton { pos: from, button, pressed: true, modifiers }),
                    stage if stage < DRAG_FRAMES + 2 => {
                        let along = f32::from(stage - 1) / f32::from(DRAG_FRAMES);
                        raw.events.push(egui::Event::PointerMoved(from.lerp(to, along)));
                    }
                    _ => raw.events.push(egui::Event::PointerButton { pos: to, button, pressed: false, modifiers }),
                }
                self.pointer = to;
                done = self.stage >= DRAG_FRAMES + 2;
            }
            Step::Key(key, modifiers) => {
                raw.modifiers = modifiers;
                for pressed in [true, false] {
                    raw.events.push(egui::Event::Key { key, physical_key: Some(key), pressed, repeat: false, modifiers });
                }
            }
            Step::Type(text) => raw.events.push(egui::Event::Text(text)),
            Step::Scroll(amount) => raw.events.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, amount),
                modifiers: egui::Modifiers::NONE,
            }),
            Step::Shot(path) => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.shot = Some(path);
                return;
            }
            Step::Close => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
        if done {
            self.stage = 0;
            self.steps.pop_front();
        } else if takes_frames {
            self.stage += 1;
        }
    }
}

fn save(path: &std::path::Path, image: &egui::ColorImage) {
    let [width, height] = image.size;
    let bytes: Vec<u8> = image.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
    let saved = image::RgbaImage::from_raw(width as u32, height as u32, bytes)
        .ok_or_else(|| "bad image size".to_string())
        .and_then(|buffer| buffer.save(path).map_err(|error| error.to_string()));
    match saved {
        Ok(()) => println!("script: saved {} ({width}x{height})", path.display()),
        Err(error) => eprintln!("script: couldn't save {}: {error}", path.display()),
    }
}

fn parse(text: &str) -> Result<Vec<Step>, String> {
    text.lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .map(|(number, line)| parse_step(line).map_err(|error| format!("line {number}: {error}")))
        .collect()
}

fn parse_step(line: &str) -> Result<Step, String> {
    let (command, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    let rest = rest.trim();
    let numbers = || -> Result<Vec<f32>, String> {
        rest.split_whitespace().map(|part| part.parse().map_err(|_| format!("{part} isn't a number"))).collect()
    };
    let point = |numbers: &[f32]| match numbers {
        [x, y] => Ok(egui::pos2(*x, *y)),
        _ => Err(format!("{command} wants x y")),
    };
    Ok(match command {
        "size" => match numbers()?[..] {
            [width, height] => Step::Size(width, height),
            _ => return Err("size wants width height".into()),
        },
        "wait" => Step::Wait(Duration::from_secs_f32(rest.parse().map_err(|_| "wait wants seconds")?)),
        "drop" => Step::Drop(PathBuf::from(rest)),
        "move" => Step::Move(point(&numbers()?)?),
        "click" => Step::Click(point(&numbers()?)?, egui::PointerButton::Primary),
        "rclick" => Step::Click(point(&numbers()?)?, egui::PointerButton::Secondary),
        "drag" => match numbers()?[..] {
            [x1, y1, x2, y2] => Step::Drag(egui::pos2(x1, y1), egui::pos2(x2, y2)),
            _ => return Err("drag wants x1 y1 x2 y2".into()),
        },
        "key" => {
            let mut modifiers = egui::Modifiers::NONE;
            let mut parts: Vec<&str> = rest.split('+').collect();
            let name = parts.pop().unwrap_or_default();
            for part in parts {
                match part.to_ascii_lowercase().as_str() {
                    "shift" => modifiers.shift = true,
                    "ctrl" => {
                        modifiers.ctrl = true;
                        modifiers.command = true;
                    }
                    "alt" => modifiers.alt = true,
                    other => return Err(format!("unknown modifier {other}")),
                }
            }
            Step::Key(egui::Key::from_name(name).ok_or_else(|| format!("unknown key {name}"))?, modifiers)
        }
        "type" => Step::Type(rest.to_string()),
        "scroll" => Step::Scroll(rest.parse().map_err(|_| "scroll wants an amount")?),
        "shot" => Step::Shot(PathBuf::from(rest)),
        "close" => Step::Close,
        other => return Err(format!("unknown step {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_reads_one_step_per_line() {
        let steps = parse(
            "# open a demo\nsize 1360 860\nwait 1.5\ndrop C:\\demos\\a b.dem\nclick 10 20\nrclick 5 6\nkey Shift+ArrowRight\nshot out.png\nclose\n",
        )
        .unwrap();
        assert_eq!(steps.len(), 8);
        assert_eq!(steps[2], Step::Drop(PathBuf::from("C:\\demos\\a b.dem")));
        assert_eq!(steps[4], Step::Click(egui::pos2(5.0, 6.0), egui::PointerButton::Secondary));
        assert_eq!(steps[5], Step::Key(egui::Key::ArrowRight, egui::Modifiers::SHIFT));
        assert!(parse("jump 1").unwrap_err().contains("line 1"));
    }
}
