//! Dead bodies as ragdolls. The skeleton's main bones become weighted points joined
//! by sticks, which gravity and the killing blow move and the map stops. The rest of
//! the skeleton (fingers, weapon, clothing) follows the nearest simulated bone.

use glam::{Mat3, Mat4, Quat, Vec3};

use super::skeleton::Skeleton;

/// The simulated bones and how heavy each is.
const POINTS: [(&str, f32); 16] = [
    ("bip_pelvis", 3.0),
    ("bip_spine_1", 2.0),
    ("bip_spine_3", 2.0),
    ("bip_head", 1.0),
    ("bip_upperarm_l", 1.0),
    ("bip_lowerarm_l", 0.8),
    ("bip_hand_l", 0.5),
    ("bip_upperarm_r", 1.0),
    ("bip_lowerarm_r", 0.8),
    ("bip_hand_r", 0.5),
    ("bip_hip_l", 1.5),
    ("bip_knee_l", 1.0),
    ("bip_foot_l", 0.6),
    ("bip_hip_r", 1.5),
    ("bip_knee_r", 1.0),
    ("bip_foot_r", 0.6),
];
const PELVIS: usize = 0;
const SPINE_1: usize = 1;
const SPINE_3: usize = 2;
const HEAD: usize = 3;
const UPPER_ARM_L: usize = 4;
const LOWER_ARM_L: usize = 5;
const HAND_L: usize = 6;
const UPPER_ARM_R: usize = 7;
const LOWER_ARM_R: usize = 8;
const HAND_R: usize = 9;
const HIP_L: usize = 10;
const KNEE_L: usize = 11;
const FOOT_L: usize = 12;
const HIP_R: usize = 13;
const KNEE_R: usize = 14;
const FOOT_R: usize = 15;

/// Points held at their starting distance: the limbs, then braces that keep the
/// torso in one piece.
const STICKS: [(usize, usize); 25] = [
    (PELVIS, SPINE_1),
    (SPINE_1, SPINE_3),
    (SPINE_3, HEAD),
    (SPINE_3, UPPER_ARM_L),
    (UPPER_ARM_L, LOWER_ARM_L),
    (LOWER_ARM_L, HAND_L),
    (SPINE_3, UPPER_ARM_R),
    (UPPER_ARM_R, LOWER_ARM_R),
    (LOWER_ARM_R, HAND_R),
    (PELVIS, HIP_L),
    (HIP_L, KNEE_L),
    (KNEE_L, FOOT_L),
    (PELVIS, HIP_R),
    (HIP_R, KNEE_R),
    (KNEE_R, FOOT_R),
    (HIP_L, HIP_R),
    (UPPER_ARM_L, UPPER_ARM_R),
    (PELVIS, SPINE_3),
    (HIP_L, SPINE_1),
    (HIP_R, SPINE_1),
    (UPPER_ARM_L, SPINE_1),
    (UPPER_ARM_R, SPINE_1),
    (HIP_L, SPINE_3),
    (HIP_R, SPINE_3),
    (UPPER_ARM_L, PELVIS),
];

/// Points kept at least this share of their starting distance apart, so elbows,
/// knees and the neck bend but don't fold through themselves.
const MIN_GAPS: [(usize, usize, f32); 7] = [
    (UPPER_ARM_L, HAND_L, 0.35),
    (UPPER_ARM_R, HAND_R, 0.35),
    (HIP_L, FOOT_L, 0.45),
    (HIP_R, FOOT_R, 0.45),
    (HEAD, SPINE_1, 0.8),
    (HEAD, UPPER_ARM_L, 0.7),
    (HEAD, UPPER_ARM_R, 0.7),
];

/// How each simulated bone turns: toward one point, twisted so a pair of points
/// lies across it (with a second pair for when the first lines up, like a straight
/// arm), or the same way as another bone.
#[derive(Clone, Copy)]
enum Turn {
    Toward { from: usize, to: usize, across: (usize, usize), fallback: (usize, usize) },
    Like(usize),
}

const TURNS: [Turn; 16] = [
    Turn::Toward { from: PELVIS, to: SPINE_1, across: (HIP_L, HIP_R), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Toward { from: SPINE_1, to: SPINE_3, across: (HIP_L, HIP_R), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Toward { from: SPINE_3, to: HEAD, across: (UPPER_ARM_L, UPPER_ARM_R), fallback: (HIP_L, HIP_R) },
    Turn::Toward { from: SPINE_3, to: HEAD, across: (UPPER_ARM_L, UPPER_ARM_R), fallback: (HIP_L, HIP_R) },
    Turn::Toward { from: UPPER_ARM_L, to: LOWER_ARM_L, across: (UPPER_ARM_L, HAND_L), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Toward { from: LOWER_ARM_L, to: HAND_L, across: (LOWER_ARM_L, UPPER_ARM_L), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Like(LOWER_ARM_L),
    Turn::Toward { from: UPPER_ARM_R, to: LOWER_ARM_R, across: (UPPER_ARM_R, HAND_R), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Toward { from: LOWER_ARM_R, to: HAND_R, across: (LOWER_ARM_R, UPPER_ARM_R), fallback: (UPPER_ARM_L, UPPER_ARM_R) },
    Turn::Like(LOWER_ARM_R),
    Turn::Toward { from: HIP_L, to: KNEE_L, across: (HIP_L, FOOT_L), fallback: (HIP_L, HIP_R) },
    Turn::Toward { from: KNEE_L, to: FOOT_L, across: (KNEE_L, HIP_L), fallback: (HIP_L, HIP_R) },
    Turn::Like(KNEE_L),
    Turn::Toward { from: HIP_R, to: KNEE_R, across: (HIP_R, FOOT_R), fallback: (HIP_L, HIP_R) },
    Turn::Toward { from: KNEE_R, to: FOOT_R, across: (KNEE_R, HIP_R), fallback: (HIP_L, HIP_R) },
    Turn::Like(KNEE_R),
];

/// TF2's gravity, in units per second squared.
const GRAVITY: f32 = 800.0;
/// Air drag: the share of speed kept each step.
const DRAG: f32 = 0.99;
/// Share of sideways speed kept when a point touches the map, so bodies slide a little.
const FRICTION: f32 = 0.3;
/// Rounds of fixing stick lengths each step. More is stiffer.
const ITERATIONS: usize = 8;
/// A point resting on the map stays this far off it.
const SKIN: f32 = 0.5;
/// Below this movement per step for half a second, the body has come to rest.
const REST_SPEED: f32 = 0.02;

/// A body's fall, worked out once, as the points' positions over time.
pub struct Ragdoll {
    /// The skeleton bone of each point.
    bones: [usize; 16],
    /// Every bone's world matrix when the body started falling.
    start: Vec<Mat4>,
    /// The points' positions, one entry per simulation step until the body is at rest.
    frames: Vec<[Vec3; 16]>,
    step_seconds: f32,
}

/// An orientation aimed along `toward` with `across` lying across it, or None when
/// the two line up and the twist can't be told.
fn frame(toward: Vec3, across: Vec3) -> Option<Mat3> {
    let x = toward.try_normalize()?;
    let z = x.cross(across);
    if z.length() < 0.2 * across.length() {
        return None;
    }
    let z = z.normalize();
    Some(Mat3::from_cols(x, z.cross(x), z))
}

/// How a bone has turned between its start and now, from where its points are.
fn turn_between(turn: Turn, start: &[Vec3; 16], now: &[Vec3; 16]) -> Option<Quat> {
    let Turn::Toward { from, to, across, fallback } = turn else { return None };
    let toward_start = start[to] - start[from];
    let toward_now = now[to] - now[from];
    for (a, b) in [across, fallback] {
        if let (Some(f0), Some(f1)) = (frame(toward_start, start[b] - start[a]), frame(toward_now, now[b] - now[a])) {
            return Some(Quat::from_mat3(&(f1 * f0.transpose())).normalize());
        }
    }
    let (Some(a), Some(b)) = (toward_start.try_normalize(), toward_now.try_normalize()) else {
        return Some(Quat::IDENTITY);
    };
    Some(Quat::from_rotation_arc(a, b))
}

/// Moves points by one step of gravity and their own speed, stopping at the map.
fn step(points: &mut [Vec3; 16], previous: &mut [Vec3; 16], dt: f32, solid: &impl Fn(Vec3, Vec3) -> Option<Vec3>) {
    for i in 0..16 {
        let current = points[i];
        let moved = current + (current - previous[i]) * DRAG + Vec3::NEG_Z * GRAVITY * dt * dt;
        // Looking a skin's width further keeps a point resting on the map still,
        // instead of sinking a little each step and being pushed back.
        let ahead = moved + (moved - current).normalize_or_zero() * SKIN;
        match solid(current, ahead) {
            Some(hit) => {
                let back = (current - moved).normalize_or_zero();
                points[i] = hit + back * SKIN;
                // Touching the map stops falling and slows sliding.
                let motion = moved - current;
                previous[i] = points[i] - Vec3::new(motion.x, motion.y, 0.0) * FRICTION;
            }
            None => {
                previous[i] = current;
                points[i] = moved;
            }
        }
    }
}

/// Pulls points back to their stick lengths, heavier points moving less.
fn hold_together(points: &mut [Vec3; 16], lengths: &[f32], gaps: &[f32], inverse_mass: &[f32; 16]) {
    let mut fix = |a: usize, b: usize, wanted: f32, at_least: bool| {
        let delta = points[b] - points[a];
        let distance = delta.length();
        if distance < 1e-4 || (at_least && distance >= wanted) {
            return;
        }
        let total = inverse_mass[a] + inverse_mass[b];
        let correction = delta * ((distance - wanted) / distance / total);
        points[a] += correction * inverse_mass[a];
        points[b] -= correction * inverse_mass[b];
    };
    for _ in 0..ITERATIONS {
        for (&(a, b), &length) in STICKS.iter().zip(lengths) {
            fix(a, b, length, false);
        }
        for (&(a, b, _), &gap) in MIN_GAPS.iter().zip(gaps) {
            fix(a, b, gap, true);
        }
    }
}

/// Simulates points from a starting shape and speed until they come to rest or
/// time runs out. `push` is extra speed for the upper body from the killing blow.
fn simulate(
    start: [Vec3; 16],
    velocity: Vec3,
    push: Vec3,
    seconds: f32,
    dt: f32,
    solid: &impl Fn(Vec3, Vec3) -> Option<Vec3>,
) -> Vec<[Vec3; 16]> {
    let lengths: Vec<f32> = STICKS.iter().map(|&(a, b)| start[a].distance(start[b])).collect();
    let gaps: Vec<f32> = MIN_GAPS.iter().map(|&(a, b, share)| start[a].distance(start[b]) * share).collect();
    let inverse_mass: [f32; 16] = std::array::from_fn(|i| 1.0 / POINTS[i].1);
    let mut points = start;
    // Points that start just inside the floor are lifted onto it.
    for point in &mut points {
        if let Some(surface) = solid(*point + Vec3::Z * 8.0, *point) {
            *point = surface + Vec3::Z * SKIN;
        }
    }
    let upper = [SPINE_3, HEAD, UPPER_ARM_L, LOWER_ARM_L, HAND_L, UPPER_ARM_R, LOWER_ARM_R, HAND_R];
    let mut previous: [Vec3; 16] = std::array::from_fn(|i| {
        let extra = if upper.contains(&i) { push * 1.3 } else { push * 0.8 };
        points[i] - (velocity + extra) * dt
    });
    let steps = (seconds / dt).ceil() as usize;
    let rest_steps = (0.5 / dt).ceil() as usize;
    let mut still = 0;
    let mut frames = vec![points];
    for _ in 0..steps {
        let before_step = points;
        step(&mut points, &mut previous, dt, solid);
        let before_fix = points;
        hold_together(&mut points, &lengths, &gaps, &inverse_mass);
        // Holding together mustn't push points through the map.
        for i in 0..16 {
            if before_fix[i].distance_squared(points[i]) > 0.01 {
                if let Some(hit) = solid(before_fix[i], points[i]) {
                    points[i] = hit + (before_fix[i] - points[i]).normalize_or_zero() * SKIN;
                }
            }
        }
        frames.push(points);
        let fastest = (0..16).map(|i| before_step[i].distance(points[i])).fold(0.0, f32::max);
        still = if fastest < REST_SPEED { still + 1 } else { 0 };
        if still >= rest_steps {
            break;
        }
    }
    frames
}

impl Ragdoll {
    /// Works out a body's fall. `start` is every bone's world matrix as it died,
    /// `velocity` how it was moving, `push` the killing blow's shove, and `solid`
    /// where a line first hits the map. None if the skeleton lacks the main bones.
    pub fn simulate(
        skeleton: &Skeleton,
        start: Vec<Mat4>,
        velocity: Vec3,
        push: Vec3,
        seconds: f32,
        step_seconds: f32,
        solid: impl Fn(Vec3, Vec3) -> Option<Vec3>,
    ) -> Option<Ragdoll> {
        let mut bones = [0; 16];
        for (slot, (name, _)) in POINTS.iter().enumerate() {
            bones[slot] = skeleton.bones.iter().position(|bone| bone.name.eq_ignore_ascii_case(name))?;
        }
        let points: [Vec3; 16] = std::array::from_fn(|i| start[bones[i]].w_axis.truncate());
        let frames = simulate(points, velocity, push, seconds, step_seconds, &solid);
        Some(Ragdoll { bones, start, frames, step_seconds })
    }

    /// The points' positions some seconds into the fall.
    fn points_at(&self, seconds: f32) -> [Vec3; 16] {
        let exact = (seconds.max(0.0) / self.step_seconds).min((self.frames.len() - 1) as f32);
        let low = exact.floor() as usize;
        let high = (low + 1).min(self.frames.len() - 1);
        let t = exact - low as f32;
        std::array::from_fn(|i| self.frames[low][i].lerp(self.frames[high][i], t))
    }

    /// Every bone's world matrix some seconds into the fall.
    pub fn world_matrices(&self, skeleton: &Skeleton, seconds: f32) -> Vec<Mat4> {
        let now = self.points_at(seconds);
        let first = &self.frames[0];
        // The simulated bones first: where their point is, turned like their points.
        let mut turns = [Quat::IDENTITY; 16];
        for (i, turn) in TURNS.iter().enumerate() {
            turns[i] = match *turn {
                Turn::Like(other) => turn_between(TURNS[other], first, &now).unwrap_or(Quat::IDENTITY),
                toward => turn_between(toward, first, &now).unwrap_or(Quat::IDENTITY),
            };
        }
        let mut world: Vec<Option<Mat4>> = vec![None; skeleton.bones.len()];
        for (i, &bone) in self.bones.iter().enumerate() {
            let (scale, rotation, _) = self.start[bone].to_scale_rotation_translation();
            world[bone] = Some(Mat4::from_scale_rotation_translation(scale, turns[i] * rotation, now[i]));
        }
        // Everything else keeps its place relative to the nearest simulated bone above it.
        let mut result = Vec::with_capacity(skeleton.bones.len());
        for (index, bone) in skeleton.bones.iter().enumerate() {
            let matrix = match world[index] {
                Some(matrix) => matrix,
                None => {
                    let mut ancestor = usize::try_from(bone.parent).ok();
                    while let Some(at) = ancestor {
                        if self.bones.contains(&at) {
                            break;
                        }
                        ancestor = usize::try_from(skeleton.bones[at].parent).ok();
                    }
                    let anchor = ancestor.unwrap_or(self.bones[PELVIS]);
                    let anchor_now = world[anchor].unwrap_or(self.start[anchor]);
                    anchor_now * self.start[anchor].inverse() * self.start[index]
                }
            };
            result.push(matrix);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A standing figure about 80 units tall, feet at `base`.
    fn figure(base: Vec3) -> [Vec3; 16] {
        let at = |x: f32, y: f32, z: f32| base + Vec3::new(x, y, z);
        [
            at(0.0, 0.0, 40.0),
            at(0.0, 0.0, 50.0),
            at(0.0, 0.0, 62.0),
            at(0.0, 0.0, 75.0),
            at(0.0, 10.0, 62.0),
            at(0.0, 12.0, 50.0),
            at(0.0, 13.0, 40.0),
            at(0.0, -10.0, 62.0),
            at(0.0, -12.0, 50.0),
            at(0.0, -13.0, 40.0),
            at(0.0, 5.0, 38.0),
            at(2.0, 5.0, 20.0),
            at(0.0, 5.0, 2.0),
            at(0.0, -5.0, 38.0),
            at(2.0, -5.0, 20.0),
            at(0.0, -5.0, 2.0),
        ]
    }

    /// A flat floor at height 0.
    fn floor(from: Vec3, to: Vec3) -> Option<Vec3> {
        if from.z >= 0.0 && to.z < 0.0 {
            let t = from.z / (from.z - to.z);
            Some(from.lerp(to, t))
        } else if from.z < 0.0 {
            Some(from)
        } else {
            None
        }
    }

    #[test]
    fn a_body_falls_over_and_comes_to_rest_on_the_floor() {
        let start = figure(Vec3::ZERO);
        let frames = simulate(start, Vec3::ZERO, Vec3::new(150.0, 0.0, 60.0), 6.0, 1.0 / 66.0, &floor);
        let last = frames.last().unwrap();
        assert!(frames.len() < 6 * 66, "it settles before time runs out");
        assert!(last.iter().all(|point| point.z >= -0.01), "nothing sinks through the floor");
        assert!(last[HEAD].z < 20.0, "the body ends up lying down, head at {}", last[HEAD].z);
        assert!(last[HEAD].x > 10.0, "it falls the way it was pushed");
        // The limbs stay the same length.
        for &(a, b) in &STICKS[..15] {
            let before = start[a].distance(start[b]);
            let after = last[a].distance(last[b]);
            assert!((before - after).abs() < before * 0.1 + 0.5, "stick {a}-{b}: {before} -> {after}");
        }
    }

    #[test]
    fn turns_follow_the_points() {
        let start = figure(Vec3::ZERO);
        // The whole figure turned a quarter around Z.
        let spin = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let now: [Vec3; 16] = std::array::from_fn(|i| spin * start[i]);
        for turn in TURNS {
            if let Turn::Toward { .. } = turn {
                let found = turn_between(turn, &start, &now).unwrap();
                assert!(found.abs_diff_eq(spin, 1e-4) || found.abs_diff_eq(-spin, 1e-4), "{found:?}");
            }
        }
    }
}
