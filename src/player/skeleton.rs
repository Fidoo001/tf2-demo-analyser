//! Reads Source model skeletons and animations, and poses them.
//!
//! This reads the `.mdl` format directly because animation values are signed
//! and heavily packed, and only a handful of the hundreds of animations in each
//! file are ever needed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::{Mat4, Quat, Vec3, Vec4};

const BONE_SIZE: usize = 216;
const ANIM_DESC_SIZE: usize = 100;
const SEQUENCE_SIZE: usize = 212;
const POSE_PARAM_SIZE: usize = 20;
const AUTOLAYER_SIZE: usize = 24;

const ANIM_RAWPOS: u8 = 0x01;
const ANIM_RAWROT: u8 = 0x02;
const ANIM_ANIMPOS: u8 = 0x04;
const ANIM_ANIMROT: u8 = 0x08;
const ANIM_DELTA: u8 = 0x10;
const ANIM_RAWROT2: u8 = 0x20;

const SEQUENCE_DELTA: i32 = 0x04;
const SEQUENCE_POST: i32 = 0x10;

/// Bone matrices are sent to the GPU as three rows of four floats.
pub type BoneMatrix = [f32; 12];

/// A view of an `.mdl` file. Reads past the end return zeros rather than
/// panicking, since a damaged model shouldn't take the whole player down.
#[derive(Clone, Copy)]
struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn u8(self, at: usize) -> u8 {
        self.0.get(at).copied().unwrap_or(0)
    }

    fn i16(self, at: usize) -> i16 {
        i16::from_le_bytes([self.u8(at), self.u8(at + 1)])
    }

    fn u16(self, at: usize) -> u16 {
        u16::from_le_bytes([self.u8(at), self.u8(at + 1)])
    }

    fn i32(self, at: usize) -> i32 {
        i32::from_le_bytes([self.u8(at), self.u8(at + 1), self.u8(at + 2), self.u8(at + 3)])
    }

    fn u64(self, at: usize) -> u64 {
        let mut bytes = [0; 8];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = self.u8(at + index);
        }
        u64::from_le_bytes(bytes)
    }

    fn f32(self, at: usize) -> f32 {
        f32::from_bits(self.i32(at) as u32)
    }

    fn vec3(self, at: usize) -> Vec3 {
        Vec3::new(self.f32(at), self.f32(at + 4), self.f32(at + 8))
    }

    fn text(self, at: usize) -> String {
        let rest = self.0.get(at..).unwrap_or(&[]);
        let end = rest.iter().position(|byte| *byte == 0).unwrap_or(0);
        String::from_utf8_lossy(&rest[..end]).into_owned()
    }
}

pub(super) fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => 0.0,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn quaternion48(bytes: Bytes, at: usize) -> Quat {
    let x = (f32::from(bytes.u16(at)) - 32768.0) / 32768.0;
    let y = (f32::from(bytes.u16(at + 2)) - 32768.0) / 32768.0;
    let packed = bytes.u16(at + 4);
    let z = (f32::from(packed & 0x7FFF) - 16384.0) / 16384.0;
    let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
    Quat::from_xyzw(x, y, z, if packed & 0x8000 != 0 { -w } else { w })
}

fn quaternion64(bytes: Bytes, at: usize) -> Quat {
    let packed = bytes.u64(at);
    let unpack = |shift: u32| ((packed >> shift) & 0x1F_FFFF) as f32;
    let x = (unpack(0) - 1048576.0) / 1048576.5;
    let y = (unpack(21) - 1048576.0) / 1048576.5;
    let z = (unpack(42) - 1048576.0) / 1048576.5;
    let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
    Quat::from_xyzw(x, y, z, if packed >> 63 != 0 { -w } else { w })
}

/// Source's Euler convention: Z, then Y, then X.
fn quaternion_from_euler(angles: Vec3) -> Quat {
    let (sy, cy) = (angles.z * 0.5).sin_cos();
    let (sp, cp) = (angles.y * 0.5).sin_cos();
    let (sr, cr) = (angles.x * 0.5).sin_cos();
    Quat::from_xyzw(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
}

/// Reads one frame from Source's run-length packed animation values.
fn animated_value(bytes: Bytes, start: usize, frame: usize) -> f32 {
    let mut at = start;
    let mut remaining = frame;
    loop {
        let valid = usize::from(bytes.u8(at));
        let total = usize::from(bytes.u8(at + 1));
        if total == 0 {
            return 0.0;
        }
        if total > remaining {
            let index = if valid > remaining { remaining + 1 } else { valid };
            return f32::from(bytes.i16(at + index * 2));
        }
        remaining -= total;
        at += (valid + 1) * 2;
    }
}

#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent: i32,
    pub position: Vec3,
    pub rotation: Quat,
    euler: Vec3,
    position_scale: Vec3,
    rotation_scale: Vec3,
    pose_to_bone: Mat4,
}

fn read_bones(bytes: Bytes) -> Vec<Bone> {
    let count = bytes.i32(156).max(0) as usize;
    let index = bytes.i32(160).max(0) as usize;
    (0..count)
        .map(|bone| {
            let at = index + bone * BONE_SIZE;
            let row = |row: usize| {
                let start = at + 96 + row * 16;
                Vec4::new(
                    bytes.f32(start),
                    bytes.f32(start + 4),
                    bytes.f32(start + 8),
                    bytes.f32(start + 12),
                )
            };
            let (x, y, z) = (row(0), row(1), row(2));
            Bone {
                name: bytes.text(at + bytes.i32(at) as usize).to_ascii_lowercase(),
                parent: bytes.i32(at + 4),
                position: bytes.vec3(at + 32),
                rotation: Quat::from_xyzw(
                    bytes.f32(at + 44),
                    bytes.f32(at + 48),
                    bytes.f32(at + 52),
                    bytes.f32(at + 56),
                ),
                euler: bytes.vec3(at + 60),
                position_scale: bytes.vec3(at + 72),
                rotation_scale: bytes.vec3(at + 84),
                pose_to_bone: Mat4::from_cols(
                    Vec4::new(x.x, y.x, z.x, 0.0),
                    Vec4::new(x.y, y.y, z.y, 0.0),
                    Vec4::new(x.z, y.z, z.z, 0.0),
                    Vec4::new(x.w, y.w, z.w, 1.0),
                ),
            }
        })
        .collect()
}

/// One of a model's hitboxes: a box in a bone's space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hitbox {
    pub bone: usize,
    /// Which body part it is: 1 is the head.
    pub group: i32,
    pub min: Vec3,
    pub max: Vec3,
}

/// The game's hit group for heads.
pub const HITGROUP_HEAD: i32 = 1;

/// A skeleton's bones in their resting pose.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    /// The model's first hitbox set, the one the game uses for bullets.
    pub hitboxes: Vec<Hitbox>,
    /// Named points on the model, like a weapon's muzzle.
    pub attachments: Vec<Attachment>,
}

/// A named point fixed to a bone, like the muzzle bullets come out of.
#[derive(Clone, Debug)]
pub struct Attachment {
    pub name: String,
    pub bone: usize,
    /// Where it sits relative to its bone.
    pub local: Mat4,
}

/// Size of an mstudioattachment_t: name, flags, bone, a 3x4 matrix and padding.
const ATTACHMENT_SIZE: usize = 92;

/// Reads a model's attachment points.
fn read_attachments(bytes: Bytes, bone_count: usize) -> Vec<Attachment> {
    let count = bytes.i32(240).clamp(0, 256) as usize;
    let start = bytes.i32(244).max(0) as usize;
    (0..count)
        .filter_map(|index| {
            let at = start + index * ATTACHMENT_SIZE;
            let bone = usize::try_from(bytes.i32(at + 8)).ok().filter(|&bone| bone < bone_count)?;
            // A 3x4 matrix stored row by row: rotation, then the offset in the last column.
            let m = |row: usize, column: usize| bytes.f32(at + 12 + (row * 4 + column) * 4);
            let local = Mat4::from_cols(
                glam::Vec4::new(m(0, 0), m(1, 0), m(2, 0), 0.0),
                glam::Vec4::new(m(0, 1), m(1, 1), m(2, 1), 0.0),
                glam::Vec4::new(m(0, 2), m(1, 2), m(2, 2), 0.0),
                glam::Vec4::new(m(0, 3), m(1, 3), m(2, 3), 1.0),
            );
            Some(Attachment { name: bytes.text(at + bytes.i32(at).max(0) as usize).to_ascii_lowercase(), bone, local })
        })
        .collect()
}

/// Reads the first hitbox set (mstudiohitboxset_t, then mstudiobbox_t entries of 68 bytes).
fn read_hitboxes(bytes: Bytes, bone_count: usize) -> Vec<Hitbox> {
    let (set_count, set_index) = (bytes.i32(172), bytes.i32(176));
    if set_count <= 0 || set_index <= 0 {
        return Vec::new();
    }
    let set = set_index as usize;
    let (count, index) = (bytes.i32(set + 4).clamp(0, 256) as usize, bytes.i32(set + 8));
    if index <= 0 {
        return Vec::new();
    }
    (0..count)
        .map(|n| set + index as usize + n * 68)
        .filter_map(|at| {
            let bone = bytes.i32(at);
            (bone >= 0 && (bone as usize) < bone_count).then(|| Hitbox {
                bone: bone as usize,
                group: bytes.i32(at + 4),
                min: bytes.vec3(at + 8),
                max: bytes.vec3(at + 20),
            })
        })
        .collect()
}

impl Skeleton {
    pub fn read(data: &[u8]) -> Self {
        let bones = read_bones(Bytes(data));
        let hitboxes = read_hitboxes(Bytes(data), bones.len());
        let attachments = read_attachments(Bytes(data), bones.len());
        Self { bones, hitboxes, attachments }
    }

    /// Where a named attachment point is, given the bones' world matrices.
    pub fn attachment_point(&self, name: &str, world: &[Mat4]) -> Option<Vec3> {
        let attachment = self.attachments.iter().find(|attachment| attachment.name == name)?;
        Some((*world.get(attachment.bone)? * attachment.local).w_axis.truncate())
    }

    /// Whether any of this model's bones exist on the carrier, so it has something
    /// to hang on. Worn items like the Pyro's jetpack don't share any with the hands.
    pub fn attaches_to(&self, carrier: &Skeleton) -> bool {
        self.bones.iter().any(|bone| carrier.bone_index(&bone.name).is_some())
    }

    pub fn bone_index(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|bone| bone.name == name)
    }

    fn index_by_name(&self) -> HashMap<&str, usize> {
        self.bones
            .iter()
            .enumerate()
            .map(|(index, bone)| (bone.name.as_str(), index))
            .collect()
    }

    pub fn rest_pose(&self) -> Pose {
        Pose {
            positions: self.bones.iter().map(|bone| bone.position).collect(),
            rotations: self.bones.iter().map(|bone| bone.rotation).collect(),
        }
    }

    /// How a model sits when nothing animates it: the first frame of its first
    /// animation. Its raw resting pose can be lying on its side.
    pub fn idle_pose(&self, model: &[u8]) -> Pose {
        let bytes = Bytes(model);
        if bytes.i32(180) <= 0 {
            return self.rest_pose();
        }
        let to_target: Vec<Option<usize>> = (0..self.bones.len()).map(Some).collect();
        let clip = read_clip(bytes, 0, &self.bones, self, &to_target);
        let count = self.bones.len();
        Pose {
            positions: clip.positions[..count].to_vec(),
            rotations: clip.rotations[..count].to_vec(),
        }
    }

    /// Turns bone positions into where each bone ends up in the world.
    pub fn world_matrices(&self, pose: &Pose, root: Mat4) -> Vec<Mat4> {
        let mut world: Vec<Mat4> = Vec::with_capacity(self.bones.len());
        for (index, bone) in self.bones.iter().enumerate() {
            let local = Mat4::from_rotation_translation(pose.rotations[index], pose.positions[index]);
            let parent = usize::try_from(bone.parent).ok().and_then(|parent| world.get(parent));
            world.push(parent.copied().unwrap_or(root) * local);
        }
        world
    }

    /// Converts world bone matrices into what the skinning shader needs.
    pub fn skinning_matrices(&self, world: &[Mat4]) -> Vec<BoneMatrix> {
        self.bones
            .iter()
            .zip(world)
            .map(|(bone, world)| pack_matrix(&(*world * bone.pose_to_bone)))
            .collect()
    }

    /// Poses another model's skeleton by copying the bones it shares with this one,
    /// the way a held weapon follows the hand holding it.
    /// Every bone's world matrix for a model hung on a carrier: bones the carrier also
    /// has follow it, the rest keep their place below them.
    pub fn attached_world(&self, carrier: &Skeleton, carrier_world: &[Mat4], root: Mat4) -> Vec<Mat4> {
        let carrier_bones = carrier.index_by_name();
        let mut world: Vec<Mat4> = Vec::with_capacity(self.bones.len());
        for bone in &self.bones {
            let name = bone.name.as_str();
            let shared = carrier_bones.get(name).and_then(|index| carrier_world.get(*index));
            let matrix = match shared {
                Some(matrix) => *matrix,
                None => {
                    let local = Mat4::from_rotation_translation(bone.rotation, bone.position);
                    let parent = usize::try_from(bone.parent).ok().and_then(|parent| world.get(parent));
                    parent.copied().unwrap_or(root) * local
                }
            };
            world.push(matrix);
        }
        world
    }
}

fn pack_matrix(matrix: &Mat4) -> BoneMatrix {
    let columns = matrix.to_cols_array_2d();
    let mut rows = [0.0; 12];
    for row in 0..3 {
        for column in 0..4 {
            rows[row * 4 + column] = columns[column][row];
        }
    }
    rows
}

/// Every bone's position and rotation relative to its parent.
#[derive(Clone, Debug)]
pub struct Pose {
    pub positions: Vec<Vec3>,
    pub rotations: Vec<Quat>,
}

impl Pose {
    /// Part way from this pose to `other`: 0 is this one, 1 is `other`.
    pub fn blend(&self, other: &Pose, amount: f32) -> Pose {
        Pose {
            positions: self.positions.iter().zip(&other.positions).map(|(a, b)| a.lerp(*b, amount)).collect(),
            rotations: self.rotations.iter().zip(&other.rotations).map(|(a, b)| a.slerp(*b, amount)).collect(),
        }
    }
}

/// One animation, unpacked and matched up to a skeleton's bones.
pub struct Clip {
    frames: usize,
    fps: f32,
    positions: Vec<Vec3>,
    rotations: Vec<Quat>,
    bone_count: usize,
}

impl Clip {
    fn sample(&self, cycle: f32, bone: usize) -> (Vec3, Quat) {
        let last = self.frames.saturating_sub(1);
        let exact = cycle.clamp(0.0, 1.0) * last as f32;
        let frame = (exact as usize).min(last);
        let next = (frame + 1).min(last);
        let blend = exact - frame as f32;
        let a = frame * self.bone_count + bone;
        let b = next * self.bone_count + bone;
        (
            self.positions[a].lerp(self.positions[b], blend),
            self.rotations[a].slerp(self.rotations[b], blend),
        )
    }

    fn seconds(&self) -> f32 {
        if self.fps > 0.0 {
            self.frames.saturating_sub(1).max(1) as f32 / self.fps
        } else {
            1.0
        }
    }
}

/// Where an animation description keeps its block number. Animations in any block
/// but 0 keep their data in a separate .ani file.
const ANIM_DESC_BLOCK: usize = 52;
/// More animations than any real model has: a count past this is a broken file.
const MAX_ANIMATIONS: usize = 4096;

/// Whether the animation at `index` keeps its data in a separate .ani file.
fn is_external_animation(bytes: Bytes, index: usize) -> bool {
    let desc = bytes.i32(184).max(0) as usize + index * ANIM_DESC_SIZE;
    bytes.i32(desc + ANIM_DESC_BLOCK) != 0
}

/// Reads a model with vmdl, which crashes on animations kept in a separate .ani file
/// instead of saying it can't. Only the meshes are needed from vmdl, so such a
/// model is read as if it had no animations.
pub fn read_mdl(data: &[u8]) -> Option<vmdl::Mdl> {
    let bytes = Bytes(data);
    let count = (bytes.i32(180).max(0) as usize).min(MAX_ANIMATIONS);
    if (0..count).any(|index| is_external_animation(bytes, index)) {
        let mut copy = data.to_vec();
        copy.get_mut(180..184)?.copy_from_slice(&0i32.to_le_bytes());
        return vmdl::Mdl::read(&copy).ok();
    }
    vmdl::Mdl::read(data).ok()
}

fn read_clip(bytes: Bytes, index: usize, source: &[Bone], target: &Skeleton, to_target: &[Option<usize>]) -> Clip {
    let desc = bytes.i32(184).max(0) as usize + index * ANIM_DESC_SIZE;
    let fps = bytes.f32(desc + 8);
    let is_delta = bytes.i32(desc + 12) & SEQUENCE_DELTA != 0;
    // Data kept in a separate .ani file isn't read: the clip holds the resting pose.
    let external = is_external_animation(bytes, index);
    let frames = if external { 1 } else { bytes.i32(desc + 16).max(1) as usize };
    let section_frames = bytes.i32(desc + 84).max(0) as usize;
    let bone_count = target.bones.len();

    let mut positions = Vec::with_capacity(frames * bone_count);
    let mut rotations = Vec::with_capacity(frames * bone_count);

    for frame in 0..frames {
        // Deltas start from nothing, everything else from the resting pose.
        for bone in &target.bones {
            positions.push(if is_delta { Vec3::ZERO } else { bone.position });
            rotations.push(if is_delta { Quat::IDENTITY } else { bone.rotation });
        }
        let row = frame * bone_count;

        for (from, to) in to_target.iter().enumerate() {
            if let (Some(to), false) = (to, is_delta) {
                positions[row + *to] = source[from].position;
                rotations[row + *to] = source[from].rotation;
            }
        }

        if external {
            continue;
        }
        let (mut at, local_frame) = if section_frames > 0 {
            let section = frame / section_frames;
            let entry = desc + bytes.i32(desc + 80).max(0) as usize + section * 8;
            (desc + bytes.i32(entry + 4).max(0) as usize, frame - section * section_frames)
        } else {
            (desc + bytes.i32(desc + 56).max(0) as usize, frame)
        };

        loop {
            let bone = usize::from(bytes.u8(at));
            let flags = bytes.u8(at + 1);
            let next = bytes.i16(at + 2);
            let data = at + 4;

            if let (Some(info), Some(Some(to))) = (source.get(bone), to_target.get(bone)) {
                let delta = flags & ANIM_DELTA != 0;

                let mut position_at = data;
                let rotation = if flags & ANIM_RAWROT != 0 {
                    position_at += 6;
                    Some(quaternion48(bytes, data))
                } else if flags & ANIM_RAWROT2 != 0 {
                    position_at += 8;
                    Some(quaternion64(bytes, data))
                } else if flags & ANIM_ANIMROT != 0 {
                    position_at += 6;
                    let channel = |axis: usize| {
                        let offset = bytes.i16(data + axis * 2);
                        if offset == 0 {
                            return 0.0;
                        }
                        let start = (data as i64 + i64::from(offset)) as usize;
                        animated_value(bytes, start, local_frame)
                    };
                    let mut angles = Vec3::new(channel(0), channel(1), channel(2)) * info.rotation_scale;
                    if !delta {
                        angles += info.euler;
                    }
                    Some(quaternion_from_euler(angles))
                } else {
                    None
                };
                match rotation {
                    Some(rotation) => rotations[row + *to] = rotation,
                    // A relative bone with no rotation data doesn't turn at all.
                    None if delta => rotations[row + *to] = Quat::IDENTITY,
                    None => {}
                }

                if flags & (ANIM_RAWPOS | ANIM_ANIMPOS) == 0 && delta {
                    positions[row + *to] = Vec3::ZERO;
                } else if flags & ANIM_RAWPOS != 0 {
                    positions[row + *to] = Vec3::new(
                        half_to_f32(bytes.u16(position_at)),
                        half_to_f32(bytes.u16(position_at + 2)),
                        half_to_f32(bytes.u16(position_at + 4)),
                    );
                } else if flags & ANIM_ANIMPOS != 0 {
                    let channel = |axis: usize| {
                        let offset = bytes.i16(position_at + axis * 2);
                        if offset == 0 {
                            return 0.0;
                        }
                        let start = (position_at as i64 + i64::from(offset)) as usize;
                        animated_value(bytes, start, local_frame)
                    };
                    let mut value = Vec3::new(channel(0), channel(1), channel(2)) * info.position_scale;
                    if !delta {
                        value += info.position;
                    }
                    positions[row + *to] = value;
                }
            }

            if next <= 0 {
                break;
            }
            at += next as usize;
        }
    }

    Clip {
        frames,
        fps,
        positions,
        rotations,
        bone_count,
    }
}

/// Plays a model's animations by the number the game uses for them, including
/// animations that live in the files the model includes.
pub struct Animations {
    files: Vec<Vec<u8>>,
    file_bones: Vec<Vec<Bone>>,
    to_target: Vec<Vec<Option<usize>>>,
    /// Each sequence as (file, offset of its description).
    sequences: Vec<(usize, usize)>,
    clips: Mutex<HashMap<(usize, usize), Arc<Clip>>>,
}

fn sequence_label(bytes: Bytes, sequence: usize) -> String {
    bytes.text(sequence + bytes.i32(sequence + 4).max(0) as usize).to_ascii_lowercase()
}

fn sequence_offsets(bytes: Bytes) -> Vec<usize> {
    let count = bytes.i32(188).max(0) as usize;
    let start = bytes.i32(192).max(0) as usize;
    (0..count).map(|index| start + index * SEQUENCE_SIZE).collect()
}

impl Animations {
    /// The other model files a model pulls animations from.
    pub fn include_paths(model: &[u8]) -> Vec<String> {
        let bytes = Bytes(model);
        let count = bytes.i32(336).max(0) as usize;
        let start = bytes.i32(340).max(0) as usize;
        (0..count.min(16))
            .map(|index| {
                let at = start + index * 8;
                bytes.text(at + bytes.i32(at + 4).max(0) as usize).replace('\\', "/").to_ascii_lowercase()
            })
            .filter(|path| !path.is_empty())
            .collect()
    }

    pub fn read(model: Vec<u8>, includes: Vec<Vec<u8>>, target: &Skeleton) -> Self {
        let mut files: Vec<Vec<u8>> = vec![model];
        files.extend(includes);
        let by_name = target.index_by_name();
        let file_bones: Vec<Vec<Bone>> = files.iter().map(|file| read_bones(Bytes(file))).collect();
        let to_target = file_bones
            .iter()
            .map(|bones| bones.iter().map(|bone| by_name.get(bone.name.as_str()).copied()).collect())
            .collect();

        // The model's own sequences come first. Ones without animation of their own
        // are placeholders for a sequence of the same name in an included file.
        let mut sequences: Vec<(usize, usize)> = Vec::new();
        let mut labels: HashMap<String, usize> = HashMap::new();
        for (file, bytes) in files.iter().enumerate() {
            let bytes = Bytes(bytes);
            for sequence in sequence_offsets(bytes) {
                let label = sequence_label(bytes, sequence);
                let has_animation = bytes.i32(sequence + 56) > 0;
                match labels.get(&label) {
                    Some(&existing) => {
                        let (existing_file, existing_offset) = sequences[existing];
                        let existing_bytes: &[u8] = files[existing_file].as_slice();
                        let placeholder = Bytes(existing_bytes).i32(existing_offset + 56) <= 0;
                        if placeholder && has_animation {
                            sequences[existing] = (file, sequence);
                        }
                    }
                    None => {
                        labels.insert(label, sequences.len());
                        sequences.push((file, sequence));
                    }
                }
            }
        }

        Self {
            files,
            file_bones,
            to_target,
            sequences,
            clips: Mutex::new(HashMap::new()),
        }
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// The first sequence playing an activity, such as `ACT_PRIMARY_VM_IDLE`.
    /// The activity a sequence plays for, like `ACT_VM_PRIMARYATTACK`.
    pub fn activity_of(&self, sequence: usize) -> Option<String> {
        let &(file, offset) = self.sequences.get(sequence)?;
        let bytes = Bytes(self.files[file].as_slice());
        Some(bytes.text(offset + bytes.i32(offset + 8).max(0) as usize))
    }

    pub fn find_activity(&self, activity: &str) -> Option<usize> {
        self.sequences.iter().position(|&(file, offset)| {
            let bytes = Bytes(self.files[file].as_slice());
            bytes.text(offset + bytes.i32(offset + 8).max(0) as usize).eq_ignore_ascii_case(activity)
        })
    }

    fn clip(&self, target: &Skeleton, file: usize, animation: usize) -> Option<Arc<Clip>> {
        let bytes = Bytes(&self.files[file]);
        let mut clips = self.clips.lock().ok()?;
        Some(Arc::clone(clips.entry((file, animation)).or_insert_with(|| {
            Arc::new(read_clip(bytes, animation, &self.file_bones[file], target, &self.to_target[file]))
        })))
    }

    /// Whether a sequence only adds to another one, like a sentry's firing, rather
    /// than being a whole pose.
    pub fn is_overlay(&self, sequence: usize) -> bool {
        self.sequences.get(sequence).is_some_and(|&(file, offset)| Bytes(&self.files[file]).i32(offset + 12) & SEQUENCE_DELTA != 0)
    }

    /// How long a sequence lasts at normal speed, in seconds.
    pub fn sequence_seconds(&self, target: &Skeleton, sequence: usize) -> Option<f32> {
        let &(file, offset) = self.sequences.get(sequence)?;
        let bytes = Bytes(&self.files[file]);
        let list = offset + bytes.i32(offset + 60).max(0) as usize;
        let animation = bytes.i16(list).max(0) as usize;
        Some(self.clip(target, file, animation)?.seconds())
    }

    /// One sequence's bones at a point in its cycle, blended across its grid by the
    /// pose parameters, which are 0 to 1 of each parameter's range as the game sends them.
    fn sample(&self, target: &Skeleton, file: usize, offset: usize, cycle: f32, parameters: &[f32]) -> Option<Vec<(Vec3, Quat)>> {
        let bytes = Bytes(&self.files[file]);
        let width = bytes.i32(offset + 68).max(1) as usize;
        let height = bytes.i32(offset + 72).max(1) as usize;
        let list = offset + bytes.i32(offset + 60).max(0) as usize;
        let parameter_count = bytes.i32(300).max(0) as usize;
        let parameter_list = bytes.i32(304).max(0) as usize;
        // Where a parameter puts this sequence along one of its axes, 0 to 1.
        let share = |slot: usize| -> f32 {
            let Ok(index) = usize::try_from(bytes.i32(offset + 76 + slot * 4)) else { return 0.0 };
            if index >= parameter_count {
                return 0.0;
            }
            let at = parameter_list + index * POSE_PARAM_SIZE;
            let (low, high) = (bytes.f32(at + 8), bytes.f32(at + 12));
            let value = low + parameters.get(index).copied().unwrap_or(0.5) * (high - low);
            let (start, end) = (bytes.f32(offset + 84 + slot * 4), bytes.f32(offset + 92 + slot * 4));
            if (end - start).abs() <= f32::EPSILON {
                0.0
            } else {
                ((value - start) / (end - start)).clamp(0.0, 1.0)
            }
        };
        let cell = |size: usize, share: f32| -> (usize, usize, f32) {
            if size <= 1 {
                return (0, 0, 0.0);
            }
            let exact = share * (size - 1) as f32;
            let low = (exact as usize).min(size - 2);
            (low, low + 1, exact - low as f32)
        };
        let (x0, x1, tx) = cell(width, share(0));
        let (y0, y1, ty) = cell(height, share(1));
        let corners = [(x0, y0, (1.0 - tx) * (1.0 - ty)), (x1, y0, tx * (1.0 - ty)), (x0, y1, (1.0 - tx) * ty), (x1, y1, tx * ty)];
        let mut samples: Vec<(Arc<Clip>, f32)> = Vec::new();
        for (x, y, weight) in corners {
            if weight <= 0.0 && !samples.is_empty() {
                continue;
            }
            let animation = bytes.i16(list + (y * width + x) * 2).max(0) as usize;
            samples.push((self.clip(target, file, animation)?, weight));
        }
        let mut bones = Vec::with_capacity(target.bones.len());
        for bone in 0..target.bones.len() {
            let mut position = Vec3::ZERO;
            let mut rotation: Option<Quat> = None;
            let mut total = 0.0;
            for (clip, weight) in &samples {
                let (p, q) = clip.sample(cycle, bone);
                if samples.len() == 1 {
                    position = p;
                    rotation = Some(q);
                    break;
                }
                if *weight <= 0.0 {
                    continue;
                }
                position += p * *weight;
                total += *weight;
                rotation = Some(match rotation {
                    Some(current) => current.slerp(q, *weight / total),
                    None => q,
                });
            }
            bones.push((position, rotation.unwrap_or(Quat::IDENTITY)));
        }
        Some(bones)
    }

    /// The model posed at a point in one of its sequences (0 to 1 through it), with
    /// its pose parameters, including the layers the sequence brings along, such as
    /// the ones that turn a sentry's gun to its aim.
    pub fn pose_at(&self, target: &Skeleton, sequence: usize, cycle: f32, parameters: &[f32]) -> Option<Pose> {
        let &(file, offset) = self.sequences.get(sequence)?;
        let bytes = Bytes(&self.files[file]);
        if bytes.i32(offset + 56) <= 0 {
            return None;
        }
        let mut pose = target.rest_pose();
        for (bone, (position, rotation)) in self.sample(target, file, offset, cycle, parameters)?.into_iter().enumerate() {
            pose.positions[bone] = position;
            pose.rotations[bone] = rotation;
        }
        let starts = sequence_offsets(bytes);
        let count = bytes.i32(offset + 148).max(0) as usize;
        let layer_list = offset + bytes.i32(offset + 152).max(0) as usize;
        for layer in 0..count {
            let index = bytes.i16(layer_list + layer * AUTOLAYER_SIZE).max(0) as usize;
            if let Some(&layer_offset) = starts.get(index) {
                self.apply_layer(target, &mut pose, file, layer_offset, 0.0, parameters);
            }
        }
        Some(pose)
    }

    /// Plays another sequence over a pose, the way the game layers a sentry's
    /// firing over its aim. `cycle` is 0 to 1 through the layer.
    pub fn add_layer(&self, target: &Skeleton, pose: &mut Pose, sequence: usize, cycle: f32, parameters: &[f32]) {
        if let Some(&(file, offset)) = self.sequences.get(sequence) {
            self.apply_layer(target, pose, file, offset, cycle, parameters);
        }
    }

    fn apply_layer(&self, target: &Skeleton, pose: &mut Pose, file: usize, offset: usize, cycle: f32, parameters: &[f32]) {
        let bytes = Bytes(&self.files[file]);
        let flags = bytes.i32(offset + 12);
        let Some(samples) = self.sample(target, file, offset, cycle, parameters) else { return };
        let weight_list = offset + bytes.i32(offset + 156).max(0) as usize;
        let mut weights = vec![0.0_f32; target.bones.len()];
        for (from, to) in self.to_target[file].iter().enumerate() {
            if let Some(to) = to {
                weights[*to] = bytes.f32(weight_list + from * 4);
            }
        }
        for (bone, (position, rotation)) in samples.into_iter().enumerate() {
            let strength = weights[bone];
            if strength <= 0.0 {
                continue;
            }
            if flags & SEQUENCE_DELTA != 0 {
                let rotation = Quat::IDENTITY.slerp(rotation, strength);
                pose.positions[bone] += position * strength;
                pose.rotations[bone] = if flags & SEQUENCE_POST != 0 {
                    pose.rotations[bone] * rotation
                } else {
                    rotation * pose.rotations[bone]
                };
            } else {
                pose.positions[bone] = pose.positions[bone].lerp(position, strength);
                pose.rotations[bone] = pose.rotations[bone].slerp(rotation, strength);
            }
        }
    }

    /// The model posed partway through one of its sequences.
    pub fn pose(&self, target: &Skeleton, sequence: usize, seconds: f32) -> Option<Pose> {
        let &(file, offset) = self.sequences.get(sequence)?;
        let bytes = Bytes(&self.files[file]);
        if bytes.i32(offset + 56) <= 0 {
            return None;
        }
        let list = offset + bytes.i32(offset + 60).max(0) as usize;
        let animation = bytes.i16(list).max(0) as usize;
        let looping = bytes.i32(offset + 12) & 1 != 0;

        let clip = {
            let mut clips = self.clips.lock().ok()?;
            Arc::clone(clips.entry((file, animation)).or_insert_with(|| {
                Arc::new(read_clip(bytes, animation, &self.file_bones[file], target, &self.to_target[file]))
            }))
        };
        let progress = seconds.max(0.0) / clip.seconds();
        let cycle = if looping { progress.rem_euclid(1.0) } else { progress.min(1.0) };
        let mut pose = target.rest_pose();
        for bone in 0..target.bones.len() {
            let (position, rotation) = clip.sample(cycle, bone);
            pose.positions[bone] = position;
            pose.rotations[bone] = rotation;
        }
        Some(pose)
    }
}

/// A grid of clips blended by up to two pose parameters.
struct BlendGrid {
    width: usize,
    height: usize,
    clips: Vec<usize>,
    axes: [Option<Axis>; 2],
    /// How strongly each bone takes part, for layers that only touch part of the body.
    weights: Vec<f32>,
    delta: bool,
    post: bool,
}

#[derive(Clone, Copy, Debug)]
struct Axis {
    parameter: Parameter,
    start: f32,
    end: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Parameter {
    MoveX,
    MoveY,
    BodyPitch,
    BodyYaw,
    Other,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PoseInput {
    pub stance: Stance,
    pub weapon: WeaponPose,
    /// Shooting or reloading, and how many seconds ago it started.
    pub gesture: Option<(Gesture, f32)>,
    /// A revved minigun, a scoped rifle or a drawn bow, which TF2 calls deployed.
    pub deployed: bool,
    /// Movement relative to the body, each from -1 to 1.
    pub move_x: f32,
    pub move_y: f32,
    pub body_pitch: f32,
    pub body_yaw: f32,
    /// Seconds into the animation. Only ever read modulo its length.
    pub time: f32,
}

impl PoseInput {
    fn parameter(&self, parameter: Parameter) -> f32 {
        match parameter {
            Parameter::MoveX => self.move_x,
            Parameter::MoveY => self.move_y,
            Parameter::BodyPitch => self.body_pitch,
            Parameter::BodyYaw => self.body_yaw,
            Parameter::Other => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Stance {
    #[default]
    Stand,
    Run,
    Crouch,
    CrouchWalk,
    /// Flung through the air, like a rocket jump.
    Air,
    Swim,
    /// The first half second of a jump.
    JumpStart,
    /// The rest of a jump, until landing.
    JumpFloat,
}

/// An upper-body animation TF2 plays over the movement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gesture {
    Attack,
    /// A weapon's alternate attack: the Spy's backstab, the Heavy's other fist, the
    /// Sandman's ball, the Pyro's airblast.
    AttackSecondary,
    Reload,
    ReloadLoop,
    ReloadEnd,
}

impl Gesture {
    const ALL: [Gesture; 5] = [
        Gesture::Attack,
        Gesture::AttackSecondary,
        Gesture::Reload,
        Gesture::ReloadLoop,
        Gesture::ReloadEnd,
    ];
    const POSES: [&'static str; 4] = ["STAND", "CROUCH", "SWIM", "AIRWALK"];

    fn activity(self, pose: &str, weapon: WeaponPose) -> String {
        let suffix = weapon.suffix();
        match self {
            Gesture::Attack => format!("ACT_MP_ATTACK_{pose}_{suffix}"),
            Gesture::AttackSecondary => format!("ACT_MP_ATTACK_{pose}_{suffix}_SECONDARY"),
            Gesture::Reload => format!("ACT_MP_RELOAD_{pose}_{suffix}"),
            Gesture::ReloadLoop => format!("ACT_MP_RELOAD_{pose}_{suffix}_LOOP"),
            Gesture::ReloadEnd => format!("ACT_MP_RELOAD_{pose}_{suffix}_END"),
        }
    }
}

/// The alternate fire of a primary weapon, which only the Pyro's flamethrower has.
fn airblast_activity(pose: &str) -> String {
    format!("ACT_MP_ATTACK_{pose}_SECONDARYFIRE")
}

/// The gesture pose a stance uses.
fn gesture_pose(stance: Stance) -> &'static str {
    match stance {
        Stance::Crouch | Stance::CrouchWalk => "CROUCH",
        Stance::Swim => "SWIM",
        Stance::Air | Stance::JumpStart | Stance::JumpFloat => "AIRWALK",
        Stance::Stand | Stance::Run => "STAND",
    }
}

/// How much of a gesture's end blends back out, so it doesn't pop.
const GESTURE_BLEND_OUT: f32 = 0.15;

/// Which of a class's animation sets goes with the weapon in hand. TF2's item
/// schema picks it per weapon (see `items.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeaponPose {
    #[default]
    Primary,
    Secondary,
    Melee,
    MeleeAllClass,
    Item1,
    Item2,
    Item3,
    Item4,
    Building,
    Pda,
    Primary2,
    Secondary2,
}

impl WeaponPose {
    pub const ALL: [WeaponPose; 12] = [
        Self::Primary,
        Self::Secondary,
        Self::Melee,
        Self::MeleeAllClass,
        Self::Item1,
        Self::Item2,
        Self::Item3,
        Self::Item4,
        Self::Building,
        Self::Pda,
        Self::Primary2,
        Self::Secondary2,
    ];

    /// The name TF2's activities end with, as in `ACT_MP_RUN_ITEM1`.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Primary => "PRIMARY",
            Self::Secondary => "SECONDARY",
            Self::Melee => "MELEE",
            Self::MeleeAllClass => "MELEE_ALLCLASS",
            Self::Item1 => "ITEM1",
            Self::Item2 => "ITEM2",
            Self::Item3 => "ITEM3",
            Self::Item4 => "ITEM4",
            Self::Building => "BUILDING",
            Self::Pda => "PDA",
            Self::Primary2 => "PRIMARY2",
            Self::Secondary2 => "SECONDARY2",
        }
    }

    /// The sets whose shooting and reloading fit this one: itself and its near twin.
    /// Borrowing any other set's reload or swing puts the hands in the wrong place,
    /// like a shotgun reload played while carrying a building.
    fn gesture_sets(self) -> Vec<WeaponPose> {
        match self {
            Self::MeleeAllClass => vec![self, Self::Melee],
            Self::Primary2 => vec![self, Self::Primary],
            Self::Secondary2 => vec![self, Self::Secondary],
            _ => vec![self],
        }
    }

    /// This set, then the closest ones to use when a class doesn't have it.
    pub fn chain(self) -> Vec<WeaponPose> {
        let closest = match self {
            Self::Primary2 | Self::Item2 => Some(Self::Primary),
            Self::Secondary2 | Self::Item1 | Self::Item3 | Self::Item4 => Some(Self::Secondary),
            Self::MeleeAllClass | Self::Building | Self::Pda => Some(Self::Melee),
            Self::Primary | Self::Secondary | Self::Melee => None,
        };
        let mut chain = vec![self];
        chain.extend(closest);
        for general in [Self::Primary, Self::Secondary, Self::Melee] {
            if !chain.contains(&general) {
                chain.push(general);
            }
        }
        chain
    }
}

impl Stance {
    pub const ALL: [Stance; 8] = [
        Self::Stand,
        Self::Run,
        Self::Crouch,
        Self::CrouchWalk,
        Self::Air,
        Self::Swim,
        Self::JumpStart,
        Self::JumpFloat,
    ];

    /// TF2's activity name for this stance with a weapon, if it has one.
    fn activity(self, weapon: WeaponPose, deployed: bool) -> Option<String> {
        if deployed {
            let name = match (self, weapon) {
                (Self::Stand, WeaponPose::Primary) => "ACT_MP_DEPLOYED_IDLE",
                (Self::Run, WeaponPose::Primary) => "ACT_MP_DEPLOYED_PRIMARY",
                (Self::Crouch, WeaponPose::Primary) => "ACT_MP_CROUCH_DEPLOYED_IDLE",
                (Self::CrouchWalk, WeaponPose::Primary) => "ACT_MP_CROUCHWALK_DEPLOYED",
                (Self::Swim, WeaponPose::Primary) => "ACT_MP_SWIM_DEPLOYED_PRIMARY",
                (Self::Stand, WeaponPose::Item2) => "ACT_MP_DEPLOYED_IDLE_ITEM2",
                (Self::Run, WeaponPose::Item2) => "ACT_MP_DEPLOYED_ITEM2",
                (Self::Crouch, WeaponPose::Item2) => "ACT_MP_CROUCH_DEPLOYED_IDLE_ITEM2",
                (Self::CrouchWalk, WeaponPose::Item2) => "ACT_MP_CROUCHWALK_DEPLOYED_ITEM2",
                _ => return None,
            };
            return Some(name.to_string());
        }
        let stance = match self {
            Self::Stand => "STAND",
            Self::Run => "RUN",
            Self::Crouch => "CROUCH",
            Self::CrouchWalk => "CROUCHWALK",
            Self::Air => "AIRWALK",
            Self::Swim => "SWIM",
            Self::JumpStart => "JUMP_START",
            Self::JumpFloat => "JUMP_FLOAT",
        };
        Some(format!("ACT_MP_{stance}_{}", weapon.suffix()))
    }

    /// Stances to try in order when a class lacks this one.
    fn chain(self) -> &'static [Stance] {
        match self {
            Self::JumpStart => &[Self::JumpStart, Self::Air, Self::Stand],
            Self::JumpFloat => &[Self::JumpFloat, Self::Air, Self::Stand],
            Self::Stand => &[Self::Stand],
            Self::Run => &[Self::Run, Self::Stand],
            Self::Crouch => &[Self::Crouch, Self::Stand],
            Self::CrouchWalk => &[Self::CrouchWalk, Self::Crouch, Self::Stand],
            Self::Air => &[Self::Air, Self::Stand],
            Self::Swim => &[Self::Swim, Self::Stand],
        }
    }
}

struct Motion {
    base: BlendGrid,
    layers: Vec<BlendGrid>,
    /// Loops like a walk, or plays once and holds like the start of a jump.
    looping: bool,
}

/// The death animations TF2 plays for special kills.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeathKind {
    Headshot,
    Backstab,
    Burning,
}

/// A class's skeleton and the movement animations it plays.
pub struct Rig {
    pub skeleton: Skeleton,
    clips: Vec<Clip>,
    motions: HashMap<(Stance, WeaponPose, bool), Motion>,
    /// Shooting and reloading layers, by activity name.
    gestures: HashMap<String, Motion>,
    deaths: HashMap<DeathKind, usize>,
}

struct AnimationFile<'a> {
    bytes: Bytes<'a>,
    bones: Vec<Bone>,
    parameters: Vec<(Parameter, f32, f32)>,
    to_target: Vec<Option<usize>>,
    loaded: HashMap<usize, usize>,
}

impl AnimationFile<'_> {
    fn sequences(&self) -> impl Iterator<Item = usize> {
        let count = self.bytes.i32(188).max(0) as usize;
        let start = self.bytes.i32(192).max(0) as usize;
        (0..count).map(move |index| start + index * SEQUENCE_SIZE)
    }

    fn label(&self, sequence: usize) -> String {
        sequence_label(self.bytes, sequence)
    }

    fn activity_name(&self, sequence: usize) -> String {
        self.bytes.text(sequence + self.bytes.i32(sequence + 8) as usize)
    }

    fn clip(&mut self, target: &Skeleton, clips: &mut Vec<Clip>, index: usize) -> usize {
        if let Some(&loaded) = self.loaded.get(&index) {
            return loaded;
        }
        clips.push(read_clip(self.bytes, index, &self.bones, target, &self.to_target));
        self.loaded.insert(index, clips.len() - 1);
        clips.len() - 1
    }

    fn grid(&mut self, target: &Skeleton, clips: &mut Vec<Clip>, sequence: usize) -> BlendGrid {
        let bytes = self.bytes;
        let flags = bytes.i32(sequence + 12);
        let width = bytes.i32(sequence + 68).max(1) as usize;
        let height = bytes.i32(sequence + 72).max(1) as usize;
        let list = sequence + bytes.i32(sequence + 60).max(0) as usize;

        let mut grid_clips = Vec::with_capacity(width * height);
        for slot in 0..width * height {
            let index = bytes.i16(list + slot * 2).max(0) as usize;
            grid_clips.push(self.clip(target, clips, index));
        }

        // Each sequence states the range it covers, which can run backwards,
        // so it's read from the sequence rather than from the parameter.
        let axis = |slot: usize| {
            let index = usize::try_from(bytes.i32(sequence + 76 + slot * 4)).ok()?;
            let &(parameter, _, _) = self.parameters.get(index)?;
            Some(Axis {
                parameter,
                start: bytes.f32(sequence + 84 + slot * 4),
                end: bytes.f32(sequence + 92 + slot * 4),
            })
        };
        let axes = [axis(0), axis(1)];

        let weight_list = sequence + bytes.i32(sequence + 156).max(0) as usize;
        let mut weights = vec![0.0; target.bones.len()];
        for (from, to) in self.to_target.iter().enumerate() {
            if let Some(to) = to {
                weights[*to] = bytes.f32(weight_list + from * 4);
            }
        }

        BlendGrid {
            width,
            height,
            clips: grid_clips,
            axes,
            weights,
            delta: flags & SEQUENCE_DELTA != 0,
            post: flags & SEQUENCE_POST != 0,
        }
    }

    fn motion(&mut self, target: &Skeleton, clips: &mut Vec<Clip>, sequence: usize, starts: &[usize]) -> Motion {
        let base = self.grid(target, clips, sequence);
        let bytes = self.bytes;
        let layer_count = bytes.i32(sequence + 148).max(0) as usize;
        let layer_list = sequence + bytes.i32(sequence + 152).max(0) as usize;
        let mut layers = Vec::new();
        for layer in 0..layer_count {
            let index = bytes.i16(layer_list + layer * AUTOLAYER_SIZE).max(0) as usize;
            // Only the aim layers matter here. They're what make a body look around.
            if let Some(&start) = starts.get(index) {
                let grid = self.grid(target, clips, start);
                let aims = grid.axes.iter().flatten().any(|axis| {
                    matches!(axis.parameter, Parameter::BodyPitch | Parameter::BodyYaw)
                });
                if grid.delta && aims {
                    layers.push(grid);
                }
            }
        }
        let looping = bytes.i32(sequence + 12) & 1 != 0;
        Motion { base, layers, looping }
    }
}

fn parameter_from_name(name: &str) -> Parameter {
    match name {
        "move_x" => Parameter::MoveX,
        "move_y" => Parameter::MoveY,
        "body_pitch" => Parameter::BodyPitch,
        "body_yaw" => Parameter::BodyYaw,
        _ => Parameter::Other,
    }
}

impl Rig {
    /// Builds a rig from a class model and its companion animation file.
    pub fn read(model: &[u8], animations: &[u8]) -> Self {
        let skeleton = Skeleton::read(model);
        let bytes = Bytes(animations);
        let bones = read_bones(bytes);

        let by_name = skeleton.index_by_name();
        let to_target = bones.iter().map(|bone| by_name.get(bone.name.as_str()).copied()).collect();

        let parameter_count = bytes.i32(300).max(0) as usize;
        let parameter_list = bytes.i32(304).max(0) as usize;
        let parameters = (0..parameter_count)
            .map(|index| {
                let at = parameter_list + index * POSE_PARAM_SIZE;
                let name = bytes.text(at + bytes.i32(at) as usize);
                (parameter_from_name(&name), bytes.f32(at + 8), bytes.f32(at + 12))
            })
            .collect();

        let mut file = AnimationFile {
            bytes,
            bones,
            parameters,
            to_target,
            loaded: HashMap::new(),
        };

        let starts: Vec<usize> = file.sequences().collect();
        let by_activity: HashMap<String, usize> = starts
            .iter()
            .map(|&sequence| (file.activity_name(sequence).to_ascii_uppercase(), sequence))
            .collect();

        let mut clips = Vec::new();
        let mut motions = HashMap::new();
        let mut gestures = HashMap::new();
        // The Pyro's airblast, which TF2 names without a weapon set.
        for pose in Gesture::POSES {
            let name = airblast_activity(pose);
            if let Some(&sequence) = by_activity.get(&name) {
                let motion = file.motion(&skeleton, &mut clips, sequence, &starts);
                gestures.insert(name, motion);
            }
        }
        for weapon in WeaponPose::ALL {
            for gesture in Gesture::ALL {
                for pose in Gesture::POSES {
                    let name = gesture.activity(pose, weapon);
                    // The Sniper fires and reloads differently while scoped in.
                    for name in [format!("{name}_DEPLOYED"), name] {
                        if gestures.contains_key(&name) {
                            continue;
                        }
                        if let Some(&sequence) = by_activity.get(&name) {
                            let motion = file.motion(&skeleton, &mut clips, sequence, &starts);
                            gestures.insert(name, motion);
                        }
                    }
                }
            }
            for stance in Stance::ALL {
                for deployed in [false, true] {
                    let Some(name) = stance.activity(weapon, deployed) else { continue };
                    if let Some(&sequence) = by_activity.get(&name) {
                        let motion = file.motion(&skeleton, &mut clips, sequence, &starts);
                        motions.insert((stance, weapon, deployed), motion);
                    }
                }
            }
        }

        let mut deaths = HashMap::new();
        for (kind, label) in [
            (DeathKind::Headshot, "primary_death_headshot"),
            (DeathKind::Backstab, "primary_death_backstab"),
            (DeathKind::Burning, "primary_death_burning"),
        ] {
            if let Some(&sequence) = starts.iter().find(|&&sequence| file.label(sequence) == label) {
                let list = sequence + bytes.i32(sequence + 60).max(0) as usize;
                let animation = bytes.i16(list).max(0) as usize;
                let clip = file.clip(&skeleton, &mut clips, animation);
                deaths.insert(kind, clip);
            }
        }

        Self {
            skeleton,
            clips,
            motions,
            gestures,
            deaths,
        }
    }

    /// The body partway through falling over, holding the last frame once it's down.
    pub fn death_pose(&self, kind: DeathKind, seconds: f32) -> Option<Pose> {
        let clip = self
            .deaths
            .get(&kind)
            .or_else(|| self.deaths.get(&DeathKind::Headshot))
            .or_else(|| self.deaths.values().next())?;
        let clip = &self.clips[*clip];
        let cycle = (seconds.max(0.0) / clip.seconds()).min(1.0);
        let mut pose = self.skeleton.rest_pose();
        for bone in 0..self.skeleton.bones.len() {
            let (position, rotation) = clip.sample(cycle, bone);
            pose.positions[bone] = position;
            pose.rotations[bone] = rotation;
        }
        Some(pose)
    }

    #[cfg(test)]
    pub fn has_motion(&self) -> bool {
        !self.motions.is_empty()
    }

    /// The gesture layer for a stance and weapon. None when the weapon has no such
    /// animation, like a reload for a melee weapon.
    fn gesture(&self, gesture: Gesture, stance: Stance, weapon: WeaponPose, deployed: bool) -> Option<&Motion> {
        let pose = gesture_pose(stance);
        if gesture == Gesture::AttackSecondary {
            // The weapon's own alternate attack, the airblast for a primary, or else
            // its usual attack, which is what TF2 falls back to.
            let find = |name: &dyn Fn(&str) -> String| [pose, "STAND"].iter().find_map(|pose| self.gestures.get(&name(pose)));
            let own = weapon
                .gesture_sets()
                .into_iter()
                .find_map(|weapon| find(&|pose| gesture.activity(pose, weapon)))
                .or_else(|| (weapon == WeaponPose::Primary).then(|| find(&airblast_activity)).flatten());
            return own.or_else(|| self.gesture(Gesture::Attack, stance, weapon, deployed));
        }
        weapon.gesture_sets().into_iter().find_map(|weapon| {
            [pose, "STAND"].iter().find_map(|pose| {
                let name = gesture.activity(pose, weapon);
                deployed
                    .then(|| self.gestures.get(&format!("{name}_DEPLOYED")))
                    .flatten()
                    .or_else(|| self.gestures.get(&name))
            })
        })
    }

    #[cfg(test)]
    fn has_gesture(&self, gesture: Gesture, weapon: WeaponPose) -> bool {
        self.gesture(gesture, Stance::Stand, weapon, false).is_some()
    }

    /// The closest animation the class has. Keeping the weapon's own set comes
    /// first, since the wrong set puts the hands in the wrong place.
    fn motion(&self, stance: Stance, weapon: WeaponPose, deployed: bool) -> Option<&Motion> {
        for weapon in weapon.chain() {
            for &stance in stance.chain() {
                let found = deployed
                    .then(|| self.motions.get(&(stance, weapon, true)))
                    .flatten()
                    .or_else(|| self.motions.get(&(stance, weapon, false)));
                if found.is_some() {
                    return found;
                }
            }
        }
        None
    }

    /// Works out a grid's four nearest clips and how much each one counts.
    fn grid_weights(&self, grid: &BlendGrid, input: &PoseInput) -> [(usize, f32); 4] {
        let cell = |axis: Option<Axis>, size: usize| -> (usize, usize, f32) {
            if size <= 1 {
                return (0, 0, 0.0);
            }
            let Some(axis) = axis else { return (0, 0, 0.0) };
            let range = axis.end - axis.start;
            let share = if range.abs() > f32::EPSILON {
                ((input.parameter(axis.parameter) - axis.start) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let exact = share * (size - 1) as f32;
            let low = (exact as usize).min(size - 2);
            (low, low + 1, exact - low as f32)
        };
        let (x0, x1, tx) = cell(grid.axes[0], grid.width);
        let (y0, y1, ty) = cell(grid.axes[1], grid.height);
        let at = |x: usize, y: usize| grid.clips[(y * grid.width + x).min(grid.clips.len() - 1)];
        [
            (at(x0, y0), (1.0 - tx) * (1.0 - ty)),
            (at(x1, y0), tx * (1.0 - ty)),
            (at(x0, y1), (1.0 - tx) * ty),
            (at(x1, y1), tx * ty),
        ]
    }

    fn sample_grid(&self, weights: &[(usize, f32); 4], cycle: f32, bone: usize) -> (Vec3, Quat) {
        let mut position = Vec3::ZERO;
        let mut rotation: Option<Quat> = None;
        let mut total = 0.0;
        for &(clip, weight) in weights {
            if weight <= 0.0 {
                continue;
            }
            let (p, q) = self.clips[clip].sample(cycle, bone);
            position += p * weight;
            total += weight;
            rotation = Some(match rotation {
                Some(current) => current.slerp(q, weight / total),
                None => q,
            });
        }
        (position, rotation.unwrap_or(Quat::IDENTITY))
    }

    /// Poses the skeleton for a moment in a player's movement.
    pub fn pose(&self, input: &PoseInput) -> Pose {
        let mut pose = self.skeleton.rest_pose();
        let Some(motion) = self.motion(input.stance, input.weapon, input.deployed) else {
            return pose;
        };

        let weights = self.grid_weights(&motion.base, input);
        let (main_clip, _) = weights
            .iter()
            .copied()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((motion.base.clips[0], 1.0));
        let length = self.clips[main_clip].seconds();
        let cycle = if motion.looping {
            (input.time / length).rem_euclid(1.0)
        } else {
            (input.time / length).clamp(0.0, 1.0)
        };

        for bone in 0..self.skeleton.bones.len() {
            let (position, rotation) = self.sample_grid(&weights, cycle, bone);
            pose.positions[bone] = position;
            pose.rotations[bone] = rotation;
        }

        for layer in &motion.layers {
            let weights = self.grid_weights(layer, input);
            for bone in 0..self.skeleton.bones.len() {
                let strength = layer.weights[bone];
                if strength <= 0.0 {
                    continue;
                }
                let (position, rotation) = self.sample_grid(&weights, 0.0, bone);
                let rotation = Quat::IDENTITY.slerp(rotation, strength);
                pose.positions[bone] += position * strength;
                pose.rotations[bone] = if layer.post {
                    pose.rotations[bone] * rotation
                } else {
                    rotation * pose.rotations[bone]
                };
            }
        }

        // Shooting or reloading plays over the upper body, then blends back out.
        if let Some((gesture, seconds)) = input.gesture {
            if let Some(layer) = self.gesture(gesture, input.stance, input.weapon, input.deployed) {
                let weights = self.grid_weights(&layer.base, input);
                let main = weights.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(layer.base.clips[0], |w| w.0);
                let progress = seconds.max(0.0) / self.clips[main].seconds().max(0.001);
                if progress < 1.0 {
                    let fade = ((1.0 - progress) / GESTURE_BLEND_OUT).clamp(0.0, 1.0);
                    for bone in 0..self.skeleton.bones.len() {
                        let strength = layer.base.weights[bone] * fade;
                        if strength <= 0.0 {
                            continue;
                        }
                        let (position, rotation) = self.sample_grid(&weights, progress, bone);
                        if layer.base.delta {
                            let rotation = Quat::IDENTITY.slerp(rotation, strength);
                            pose.positions[bone] += position * strength;
                            pose.rotations[bone] = if layer.base.post {
                                pose.rotations[bone] * rotation
                            } else {
                                rotation * pose.rotations[bone]
                            };
                        } else {
                            pose.positions[bone] = pose.positions[bone].lerp(position, strength);
                            pose.rotations[bone] = pose.rotations[bone].slerp(rotation, strength);
                        }
                    }
                }
            }
        }
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Quat, b: Quat) -> bool {
        a.dot(b).abs() > 0.9999
    }

    #[test]
    fn euler_conversion_matches_the_axes() {
        let turn = std::f32::consts::FRAC_PI_2;
        assert!(close(quaternion_from_euler(Vec3::new(turn, 0.0, 0.0)), Quat::from_rotation_x(turn)));
        assert!(close(quaternion_from_euler(Vec3::new(0.0, turn, 0.0)), Quat::from_rotation_y(turn)));
        assert!(close(quaternion_from_euler(Vec3::new(0.0, 0.0, turn)), Quat::from_rotation_z(turn)));
    }

    #[test]
    fn euler_applies_z_after_y_after_x() {
        let (x, y, z) = (0.3, -0.7, 1.1);
        let expected = Quat::from_rotation_z(z) * Quat::from_rotation_y(y) * Quat::from_rotation_x(x);
        assert!(close(quaternion_from_euler(Vec3::new(x, y, z)), expected));
    }

    #[test]
    fn half_floats_decode() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
    }

    #[test]
    fn a_packed_identity_quaternion_decodes() {
        let data = [0x00, 0x80, 0x00, 0x80, 0x00, 0x40];
        assert!(close(quaternion48(Bytes(&data), 0), Quat::IDENTITY));
    }

    #[test]
    fn packed_values_handle_negatives_and_runs() {
        // Header: 2 stored values covering 4 frames, then the values -5 and 7.
        // Frames past the stored values repeat the last one.
        let mut data = vec![2u8, 4];
        data.extend_from_slice(&(-5i16).to_le_bytes());
        data.extend_from_slice(&7i16.to_le_bytes());
        // A second run: 1 stored value covering 2 frames.
        data.extend_from_slice(&[1, 2]);
        data.extend_from_slice(&9i16.to_le_bytes());
        let bytes = Bytes(&data);
        assert_eq!(animated_value(bytes, 0, 0), -5.0);
        assert_eq!(animated_value(bytes, 0, 1), 7.0);
        assert_eq!(animated_value(bytes, 0, 3), 7.0);
        assert_eq!(animated_value(bytes, 0, 4), 9.0);
        assert_eq!(animated_value(bytes, 0, 5), 9.0);
    }

    #[test]
    fn running_off_the_end_of_packed_values_gives_zero() {
        let data = [0u8, 0];
        assert_eq!(animated_value(Bytes(&data), 0, 3), 0.0);
    }

    #[test]
    fn reading_damaged_data_never_panics() {
        let empty = Skeleton::read(&[]);
        assert!(empty.bones.is_empty());
        let rig = Rig::read(&[0; 64], &[0; 64]);
        assert!(!rig.has_motion());
    }

    #[test]
    fn child_bones_follow_their_parent() {
        let bone = |parent: i32, position: Vec3| Bone {
            name: String::new(),
            parent,
            position,
            rotation: Quat::IDENTITY,
            euler: Vec3::ZERO,
            position_scale: Vec3::ONE,
            rotation_scale: Vec3::ONE,
            pose_to_bone: Mat4::IDENTITY,
        };
        let skeleton = Skeleton {
            attachments: Vec::new(),
            hitboxes: Vec::new(),
            bones: vec![bone(-1, Vec3::X), bone(0, Vec3::Y)],
        };
        let world = skeleton.world_matrices(&skeleton.rest_pose(), Mat4::from_translation(Vec3::Z));
        assert_eq!(world[1].w_axis.truncate(), Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn matrices_pack_as_rows() {
        let packed = pack_matrix(&Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(packed[3], 1.0);
        assert_eq!(packed[7], 2.0);
        assert_eq!(packed[11], 3.0);
        assert_eq!(packed[0], 1.0);
    }
}

#[cfg(test)]
mod real_files {
    use super::*;

    /// These need TF2 installed, so they quietly pass without it.
    fn load(path: &str) -> Option<Vec<u8>> {
        tf_asset_loader::Loader::new().ok()?.load(path).ok()?
    }

    #[test]
    fn models_with_animations_in_another_file_still_load() {
        let Some(mut model) = load("models/weapons/c_models/c_minigun/c_minigun.mdl") else { return };
        assert!(Bytes(&model).i32(180) > 0, "the minigun has animations");
        // Pretend its first animation lives in block 1 of an .ani file, as some map
        // props' do. vmdl panics on those instead of returning an error.
        let desc = Bytes(&model).i32(184) as usize;
        model[desc + ANIM_DESC_BLOCK..desc + ANIM_DESC_BLOCK + 4].copy_from_slice(&1i32.to_le_bytes());
        assert!(is_external_animation(Bytes(&model), 0));
        let mdl = read_mdl(&model).expect("the model still loads, without animations");
        assert!(!mdl.body_parts.is_empty());
    }

    #[test]
    fn weapons_have_a_muzzle_down_the_barrel() {
        let Some(model) = load("models/weapons/c_models/c_minigun/c_minigun.mdl") else { return };
        let skeleton = Skeleton::read(&model);
        let muzzle = skeleton.attachments.iter().find(|attachment| attachment.name == "muzzle").expect("a muzzle");
        assert_eq!(skeleton.bones[muzzle.bone].name, "weapon_bone");
        // The minigun's barrel runs along the weapon bone's z axis, about 58 units out.
        let offset = muzzle.local.w_axis.truncate();
        assert!((offset.z - 57.8).abs() < 0.5, "{offset:?}");
        let world = vec![Mat4::IDENTITY; skeleton.bones.len()];
        assert!(skeleton.attachment_point("muzzle", &world).is_some());
    }

    #[test]
    fn class_models_have_hitboxes_with_a_head() {
        let Some(model) = load("models/player/heavy.mdl") else { return };
        let skeleton = Skeleton::read(&model);
        assert!(skeleton.hitboxes.len() >= 10, "{} hitboxes", skeleton.hitboxes.len());
        let head = skeleton.hitboxes.iter().find(|h| h.group == HITGROUP_HEAD).expect("a head hitbox");
        assert_eq!(skeleton.bones[head.bone].name, "bip_head");
        assert!((head.max - head.min).length() > 5.0);
    }

    #[test]
    fn the_resting_pose_reproduces_the_model() {
        let Some(model) = load("models/player/heavy.mdl") else { return };
        let skeleton = Skeleton::read(&model);
        let world = skeleton.world_matrices(&skeleton.rest_pose(), Mat4::IDENTITY);
        for (index, bone) in skeleton.bones.iter().enumerate() {
            let product = world[index] * bone.pose_to_bone;
            assert!(product.abs_diff_eq(Mat4::IDENTITY, 1e-3), "{} is off", bone.name);
        }
    }

    #[test]
    fn every_weapon_set_finds_an_animation_for_every_class() {
        for class in ["scout", "soldier", "pyro", "demo", "heavy", "engineer", "medic", "sniper", "spy"] {
            let (Some(model), Some(animations)) = (
                load(&format!("models/player/{class}.mdl")),
                load(&format!("models/player/{class}_animations.mdl")),
            ) else {
                return;
            };
            let rig = Rig::read(&model, &animations);
            for weapon in WeaponPose::ALL {
                for stance in Stance::ALL {
                    for deployed in [false, true] {
                        assert!(rig.motion(stance, weapon, deployed).is_some(), "{class} {stance:?} {weapon:?}");
                    }
                }
            }
            // The Spy has no primary set at all, so his revolver uses the secondary one.
            if class == "spy" {
                assert!(rig.motions.get(&(Stance::Stand, WeaponPose::Primary, false)).is_none());
                assert!(rig.motions.contains_key(&(Stance::Stand, WeaponPose::Secondary, false)));
            }
            assert!(rig.has_gesture(Gesture::Attack, WeaponPose::Melee), "{class} swings");
            // No class reloads a melee weapon or a carried building, so nothing is borrowed.
            assert!(!rig.has_gesture(Gesture::Reload, WeaponPose::Melee), "{class} reloads a melee weapon");
            assert!(!rig.has_gesture(Gesture::Reload, WeaponPose::Building), "{class} reloads a building");
            if class == "sniper" {
                let scoped = rig.gesture(Gesture::Attack, Stance::Stand, WeaponPose::Primary, true).map(|m| m as *const Motion);
                let unscoped = rig.gesture(Gesture::Attack, Stance::Stand, WeaponPose::Primary, false).map(|m| m as *const Motion);
                assert!(scoped.is_some() && scoped != unscoped, "a scoped shot has its own animation");
            }
            if matches!(class, "spy" | "heavy" | "scout") {
                let special = rig.gesture(Gesture::AttackSecondary, Stance::Stand, WeaponPose::Melee, false).map(|m| m as *const Motion);
                let swing = rig.gesture(Gesture::Attack, Stance::Stand, WeaponPose::Melee, false).map(|m| m as *const Motion);
                assert!(special.is_some() && special != swing, "{class} has its own alternate melee attack");
            }
            if class == "soldier" {
                // No alternate melee attack, so it swings as usual.
                let special = rig.gesture(Gesture::AttackSecondary, Stance::Stand, WeaponPose::Melee, false).map(|m| m as *const Motion);
                let swing = rig.gesture(Gesture::Attack, Stance::Stand, WeaponPose::Melee, false).map(|m| m as *const Motion);
                assert_eq!(special, swing);
            }
            if class == "pyro" {
                let airblast = rig.gesture(Gesture::AttackSecondary, Stance::Stand, WeaponPose::Primary, false).map(|m| m as *const Motion);
                let flames = rig.gesture(Gesture::Attack, Stance::Stand, WeaponPose::Primary, false).map(|m| m as *const Motion);
                assert!(airblast.is_some() && airblast != flames, "airblast has its own animation");
            }
            if class == "heavy" {
                assert!(rig.motions.contains_key(&(Stance::Stand, WeaponPose::Primary, true)), "revved minigun");
                assert!(rig.motions.contains_key(&(Stance::JumpStart, WeaponPose::Primary, false)));
            }
        }
    }

    #[test]
    fn blending_poses_goes_from_one_to_the_other() {
        let a = Pose { positions: vec![Vec3::ZERO], rotations: vec![Quat::IDENTITY] };
        let b = Pose { positions: vec![Vec3::new(10.0, 0.0, 0.0)], rotations: vec![Quat::from_rotation_z(1.0)] };
        assert_eq!(a.blend(&b, 0.0).positions[0], Vec3::ZERO);
        assert!(a.blend(&b, 1.0).rotations[0].abs_diff_eq(b.rotations[0], 1e-6));
        let half = a.blend(&b, 0.5);
        assert!((half.positions[0].x - 5.0).abs() < 1e-5);
        assert!(half.rotations[0].abs_diff_eq(Quat::from_rotation_z(0.5), 1e-5));
    }

    #[test]
    fn animated_bodies_stand_upright_with_hands_below_the_head() {
        let (Some(model), Some(animations)) =
            (load("models/player/heavy.mdl"), load("models/player/heavy_animations.mdl"))
        else {
            return;
        };
        let rig = Rig::read(&model, &animations);
        assert!(rig.has_motion());
        for stance in [Stance::Stand, Stance::Run, Stance::Crouch] {
            let input = PoseInput { stance, move_x: 1.0, ..PoseInput::default() };
            let world = rig.skeleton.world_matrices(&rig.pose(&input), Mat4::IDENTITY);
            let height = |name: &str| {
                let index = rig.skeleton.bones.iter().position(|bone| bone.name == name).unwrap();
                world[index].w_axis.z
            };
            assert!(height("bip_head") > height("bip_pelvis"), "{stance:?} is upside down");
            assert!(height("bip_hand_r") < height("bip_head"), "{stance:?} has raised hands");
            assert!(height("bip_pelvis") > 15.0, "{stance:?} is sunk into the floor");
        }
    }
}
