//! Turns a TF2 `.bsp` into meshes, textures and lighting ready for the GPU.

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use tf_asset_loader::Loader;
use vbsp::{Bsp, TextureFlags};

use super::assets::{Assets, Image, Vmt};
use super::characters::{load_characters, Characters, MeshData};

const LUMP_LEAFS: usize = 10;
const LUMP_LEAF_FACES: usize = 16;
const LUMP_LIGHTING: usize = 8;
const LUMP_LEAF_AMBIENT_INDEX_HDR: usize = 51;
const LUMP_LEAF_AMBIENT_INDEX: usize = 52;
const LUMP_LIGHTING_HDR: usize = 53;
const LUMP_LEAF_AMBIENT_LIGHTING_HDR: usize = 55;
const LUMP_LEAF_AMBIENT_LIGHTING: usize = 56;

const CONTENTS_SOLID: i32 = 1;
const RENDER_MODE_NONE: i32 = 10;
const ATLAS_SIZE: usize = 4096;
const ATLAS_PADDING: usize = 1;
const SKY_SUFFIXES: [&str; 6] = ["rt", "lf", "bk", "ft", "up", "dn"];
const SUN_DISTANCE: f32 = 12000.0;

/// Brush entities that are meant to be seen. Everything else with a brush
/// model is a trigger, a volume or some other invisible helper.
const VISIBLE_BRUSH_ENTITIES: [&str; 12] = [
    "func_brush",
    "func_wall",
    "func_wall_toggle",
    "func_illusionary",
    "func_door",
    "func_door_rotating",
    "func_movelinear",
    "func_rotating",
    "func_lod",
    "func_tracktrain",
    "func_breakable",
    "func_physbox",
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WorldVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub lightmap_uv: [f32; 2],
    /// How far a two-texture blend has moved towards the second texture.
    pub blend: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Where this vertex sits in the map's baked lighting for the prop, or -1.
    pub light: f32,
}

/// How a surface gets drawn. Texture fields index `MapData::textures`.
#[derive(Clone, Debug, Default)]
pub struct Material {
    pub base: Option<usize>,
    pub base2: Option<usize>,
    pub translucent: bool,
    pub additive: bool,
    pub alpha_test: Option<f32>,
    pub opacity: f32,
    pub tint: [f32; 3],
    pub unlit: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub material: usize,
    pub first_index: u32,
    pub index_count: u32,
    /// Part of the miniature backdrop that gets drawn enlarged behind the map.
    pub sky: bool,
}

/// Where the miniature backdrop sits and how much it's scaled down.
#[derive(Clone, Copy, Debug)]
pub struct SkyCamera {
    pub origin: Vec3,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelPart {
    pub material: usize,
    pub first_index: u32,
    pub index_count: u32,
}

pub struct ModelData {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<ModelPart>,
    /// How many baked lighting colours each mesh of the first detail level
    /// takes, which a map's lighting file has to match.
    pub light_counts: Vec<u32>,
}

/// Light arriving from each axis direction, the way Source lights models.
pub type LightCube = [[f32; 3]; 6];

#[derive(Clone, Copy, Debug)]
pub struct ModelLighting {
    pub cube: LightCube,
    pub sun: f32,
}

#[derive(Clone, Debug)]
pub struct PropInstance {
    pub model: usize,
    pub transform: Mat4,
    pub lighting: ModelLighting,
    pub sky: bool,
    /// Where this prop's colours start in `MapData::prop_light`, when the map
    /// baked lighting for it.
    pub baked: Option<u32>,
}

/// Static prop colours are packed this many to a row.
pub const PROP_LIGHT_WIDTH: usize = 4096;

#[derive(Clone, Copy, Debug, Default)]
pub struct Sun {
    /// Points from the world towards the sun.
    pub direction: [f32; 3],
    pub color: [f32; 3],
}

pub struct MapData {
    pub world_vertices: Vec<WorldVertex>,
    pub world_indices: Vec<u32>,
    pub batches: Vec<Batch>,
    pub materials: Vec<Material>,
    pub textures: Vec<Image>,
    pub lightmap_width: usize,
    pub lightmap_height: usize,
    /// Linear RGB, three floats per texel.
    pub lightmap: Vec<f32>,
    pub models: Vec<ModelData>,
    pub props: Vec<PropInstance>,
    /// Baked vertex lighting for static props, RGBA, `PROP_LIGHT_WIDTH` to a row.
    pub prop_light: Vec<u8>,
    /// Faces in the order +X, -X, +Y, -Y, +Z, -Z. Each indexes `textures`.
    pub sky: Option<[usize; 6]>,
    pub lighting: Arc<LightProbe>,
    pub sky_camera: Option<SkyCamera>,
    pub bounds: (Vec3, Vec3),
    /// Animated models: players, weapons, projectiles and buildings.
    pub meshes: Vec<MeshData>,
    pub characters: Characters,
    pub map_props: Vec<MapProp>,
    /// The top left quarter of TF2's sniper scope, as a black mask. The other
    /// corners are mirror images of it.
    pub scope: Option<Image>,
    /// TF2's default crosshair.
    pub crosshair: Option<Image>,
    /// The animation set for each demo model, per class, when it's a weapon.
    pub weapon_roles: super::items::WeaponRoles,
}

/// An object placed in the map file that can move, such as a door or cart.
/// Demos only include the ones near the recording player, so the rest are
/// drawn from here where the map put them.
#[derive(Clone, Debug)]
pub struct MapProp {
    /// Index into the model list, the same numbering as `Recording::models`.
    pub model: usize,
    pub transform: Mat4,
    pub origin: Vec3,
    pub skin: usize,
    /// Attached to something that moves, like the bomb on a payload cart.
    pub attached: bool,
}

const MAP_PROP_CLASSES: [&str; 6] = [
    "prop_dynamic",
    "prop_dynamic_override",
    "prop_physics",
    "prop_physics_override",
    "prop_physics_multiplayer",
    "prop_dynamic_ornament",
];

/// What a demo needs on top of the map itself.
#[derive(Clone, Debug, Default)]
pub struct DemoContent {
    pub sky_name: String,
    pub classes: Vec<u8>,
    pub models: Vec<String>,
}

/// Lets anything in the world be lit according to where it stands.
pub struct LightProbe {
    /// Terrain and solid brush entities, which the BSP tree below doesn't know about.
    terrain: Terrain,
    planes: Vec<(Vec3, f32)>,
    nodes: Vec<(i32, [i32; 2])>,
    /// Whether each node's plane carries sky faces, so a ray stopping there saw the sky.
    node_is_sky: Vec<bool>,
    leaf_contents: Vec<i32>,
    leaf_areas: Vec<u16>,
    leaf_cubes: Vec<Vec<AmbientSample>>,
    sun: Sun,
    default_cube: LightCube,
}

/// Size of the grid cells terrain triangles are filed under, in units.
const TERRAIN_CELL: f32 = 256.0;

/// Displacement terrain (hills, rocky ground) and solid brush entities (bridges,
/// walls) as triangles filed in a grid, so rays can stop on them. The BSP tree's
/// solid shape has neither, which would let bullets, aim lines and bodies pass
/// straight through.
#[derive(Default)]
struct Terrain {
    triangles: Vec<[Vec3; 3]>,
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl Terrain {
    fn new(triangles: Vec<[Vec3; 3]>) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let cell = |value: f32| (value / TERRAIN_CELL).floor() as i32;
        for (index, triangle) in triangles.iter().enumerate() {
            let low = triangle[0].min(triangle[1]).min(triangle[2]);
            let high = triangle[0].max(triangle[1]).max(triangle[2]);
            for x in cell(low.x)..=cell(high.x) {
                for y in cell(low.y)..=cell(high.y) {
                    cells.entry((x, y)).or_default().push(index as u32);
                }
            }
        }
        Self { triangles, cells }
    }

    /// How far along the line from `from` to `to` (0 to 1) it first touches terrain.
    fn first_hit(&self, from: Vec3, to: Vec3) -> Option<f32> {
        if self.triangles.is_empty() {
            return None;
        }
        let direction = to - from;
        let mut best: Option<f32> = None;
        // Walks the grid cells the line crosses, seen from above.
        let cell = |value: f32| (value / TERRAIN_CELL).floor() as i32;
        let (mut x, mut y) = (cell(from.x), cell(from.y));
        let (end_x, end_y) = (cell(to.x), cell(to.y));
        let step_x = if direction.x > 0.0 { 1 } else { -1 };
        let step_y = if direction.y > 0.0 { 1 } else { -1 };
        let crossing = |start: f32, length: f32, at: i32, step: i32| -> (f32, f32) {
            if length.abs() < 1e-6 {
                return (f32::INFINITY, f32::INFINITY);
            }
            let boundary = if step > 0 { (at + 1) as f32 } else { at as f32 } * TERRAIN_CELL;
            ((boundary - start) / length, TERRAIN_CELL / length.abs())
        };
        let (mut next_x, delta_x) = crossing(from.x, direction.x, x, step_x);
        let (mut next_y, delta_y) = crossing(from.y, direction.y, y, step_y);
        let limit = (end_x - x).abs() + (end_y - y).abs() + 2;
        for _ in 0..=limit {
            if let Some(indices) = self.cells.get(&(x, y)) {
                for &index in indices {
                    if let Some(t) = segment_hits_triangle(from, direction, &self.triangles[index as usize]) {
                        if best.map_or(true, |best| t < best) {
                            best = Some(t);
                        }
                    }
                }
            }
            // Nothing further along can be nearer than a hit already found in this cell.
            if best.is_some_and(|best| best <= next_x.min(next_y)) || (x == end_x && y == end_y) {
                break;
            }
            if next_x < next_y {
                x += step_x;
                next_x += delta_x;
            } else {
                y += step_y;
                next_y += delta_y;
            }
        }
        best
    }
}

/// Where a line segment `from + direction * t`, t in 0..=1, crosses a triangle
/// from either side.
fn segment_hits_triangle(from: Vec3, direction: Vec3, triangle: &[Vec3; 3]) -> Option<f32> {
    let edge1 = triangle[1] - triangle[0];
    let edge2 = triangle[2] - triangle[0];
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant.abs() < 1e-8 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let offset = from - triangle[0];
    let u = offset.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = offset.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inverse;
    (0.0..=1.0).contains(&t).then_some(t)
}

/// Brush entities that stay put and block things: bridges, walls, breakable panels.
/// Doors and moving parts are left out, since where they are changes, and so are
/// the ones nothing collides with.
const SOLID_BRUSH_ENTITIES: [&str; 6] = ["func_brush", "func_wall", "func_wall_toggle", "func_lod", "func_breakable", "func_physbox"];

/// Surfaces the BSP tree's solid shape leaves out: the map's displacement terrain,
/// and solid brush entities, which are separate models.
fn collision_triangles(bsp: &Bsp) -> Vec<[Vec3; 3]> {
    let mut triangles = displacement_triangles(bsp);
    let models: Vec<_> = bsp.models().collect();
    for entity in bsp.entities.iter() {
        let class = entity.prop("classname").unwrap_or_default();
        // A func_brush set to never be solid doesn't block anything.
        if !SOLID_BRUSH_ENTITIES.contains(&class) || entity.prop("solidity").map(str::trim) == Some("1") {
            continue;
        }
        let Some(model) = entity
            .prop("model")
            .and_then(|model| model.strip_prefix('*'))
            .and_then(|index| index.parse::<usize>().ok())
            .and_then(|index| models.get(index))
        else {
            continue;
        };
        let origin = entity.prop("origin").and_then(parse_vector).unwrap_or(Vec3::ZERO);
        for face in model.faces() {
            let corners: Vec<Vec3> = face.vertices().map(|v| vec3(v.position) + origin).collect();
            for i in 1..corners.len().saturating_sub(1) {
                triangles.push([corners[0], corners[i], corners[i + 1]]);
            }
        }
    }
    triangles
}

/// The map's displacement terrain as triangles.
fn displacement_triangles(bsp: &Bsp) -> Vec<[Vec3; 3]> {
    let mut triangles = Vec::new();
    let Some(world) = bsp.models().next() else { return triangles };
    for face in world.faces() {
        let Some(displacement) = face.displacement() else { continue };
        let corners: Vec<Vec3> = face.vertices().map(|v| vec3(v.position)).collect();
        let Some(corners) = displacement_corners(&corners, vec3(displacement.start_position)) else {
            continue;
        };
        let side = 2usize.pow(displacement.power as u32) + 1;
        let offsets: Vec<_> = displacement.displacement_vertices().collect();
        if offsets.len() < side * side {
            continue;
        }
        let step = 1.0 / (side - 1) as f32;
        let mut points = Vec::with_capacity(side * side);
        for row in 0..side {
            let left = corners[0].lerp(corners[1], row as f32 * step);
            let right = corners[3].lerp(corners[2], row as f32 * step);
            for column in 0..side {
                let base = left.lerp(right, column as f32 * step);
                points.push(base + vec3(offsets[row * side + column].displacement()));
            }
        }
        for row in 0..side - 1 {
            for column in 0..side - 1 {
                let a = row * side + column;
                let (b, c) = (a + 1, a + side);
                let d = c + 1;
                triangles.push([points[a], points[c], points[b]]);
                triangles.push([points[b], points[c], points[d]]);
            }
        }
    }
    triangles
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RayEnd {
    Clear,
    Sky,
    Wall,
    EnteredSolid,
}

#[derive(Clone)]
struct AmbientSample {
    position: Vec3,
    cube: LightCube,
}

impl LightProbe {
    pub fn sun(&self) -> Sun {
        self.sun
    }

    /// The sealed region of the map a point is in.
    fn area_at(&self, point: Vec3) -> Option<u16> {
        self.leaf_areas.get(self.leaf_index(point)?).copied()
    }

    fn leaf_index(&self, point: Vec3) -> Option<usize> {
        let mut node = 0i32;
        let mut steps = 0;
        while node >= 0 {
            let (plane, children) = *self.nodes.get(node as usize)?;
            let (normal, distance) = *self.planes.get(plane as usize)?;
            node = if normal.dot(point) < distance {
                children[1]
            } else {
                children[0]
            };
            steps += 1;
            if steps > 4096 {
                return None;
            }
        }
        Some((!node) as usize)
    }

    /// Follows a ray through the map until it enters something solid.
    fn trace(&self, node: i32, from: Vec3, to: Vec3, depth: u32) -> RayEnd {
        if node < 0 {
            let contents = self.leaf_contents.get((!node) as usize).copied().unwrap_or(0);
            return if contents & CONTENTS_SOLID != 0 {
                RayEnd::EnteredSolid
            } else {
                RayEnd::Clear
            };
        }
        let plane = self.nodes.get(node as usize).filter(|_| depth < 256);
        let Some(&(plane, children)) = plane else {
            return RayEnd::Clear;
        };
        let Some(&(normal, distance)) = self.planes.get(plane as usize) else {
            return RayEnd::Clear;
        };
        let start = normal.dot(from) - distance;
        let end = normal.dot(to) - distance;
        if start >= 0.0 && end >= 0.0 {
            return self.trace(children[0], from, to, depth + 1);
        }
        if start < 0.0 && end < 0.0 {
            return self.trace(children[1], from, to, depth + 1);
        }
        let split = from + (to - from) * (start / (start - end));
        let (near, far) = if start >= 0.0 {
            (children[0], children[1])
        } else {
            (children[1], children[0])
        };
        match self.trace(near, from, split, depth + 1) {
            RayEnd::Clear => {}
            hit => return hit,
        }
        match self.trace(far, split, to, depth + 1) {
            // The ray crossed into something solid right at this plane.
            RayEnd::EnteredSolid => {
                if self.node_is_sky.get(node as usize).copied().unwrap_or(false) {
                    RayEnd::Sky
                } else {
                    RayEnd::Wall
                }
            }
            other => other,
        }
    }

    /// Where a straight line from `from` to `to` first hits something solid,
    /// brushes or terrain.
    pub fn first_hit(&self, from: Vec3, to: Vec3) -> Option<Vec3> {
        let wall = self.hit_in(0, from, to, 0);
        let end = wall.unwrap_or(to);
        match self.terrain.first_hit(from, end) {
            Some(t) => Some(from + (end - from) * t),
            None => wall,
        }
    }

    fn hit_in(&self, node: i32, from: Vec3, to: Vec3, depth: u32) -> Option<Vec3> {
        if node < 0 {
            let contents = self.leaf_contents.get((!node) as usize).copied().unwrap_or(0);
            return (contents & CONTENTS_SOLID != 0).then_some(from);
        }
        let &(plane, children) = self.nodes.get(node as usize).filter(|_| depth < 256)?;
        let &(normal, distance) = self.planes.get(plane as usize)?;
        let start = normal.dot(from) - distance;
        let end = normal.dot(to) - distance;
        if start >= 0.0 && end >= 0.0 {
            return self.hit_in(children[0], from, to, depth + 1);
        }
        if start < 0.0 && end < 0.0 {
            return self.hit_in(children[1], from, to, depth + 1);
        }
        let split = from + (to - from) * (start / (start - end));
        let (near, far) = if start >= 0.0 {
            (children[0], children[1])
        } else {
            (children[1], children[0])
        };
        self.hit_in(near, from, split, depth + 1)
            .or_else(|| self.hit_in(far, split, to, depth + 1))
    }

    /// The light falling on something at this point.
    pub fn at(&self, point: Vec3) -> ModelLighting {
        let cube = self
            .leaf_index(point)
            .and_then(|leaf| self.leaf_cubes.get(leaf))
            .and_then(|samples| {
                samples.iter().min_by(|a, b| {
                    let a = a.position.distance_squared(point);
                    let b = b.position.distance_squared(point);
                    a.total_cmp(&b)
                })
            })
            .map_or(self.default_cube, |sample| sample.cube);

        let to_sun = Vec3::from(self.sun.direction);
        let sees_sun = to_sun != Vec3::ZERO
            && matches!(self.trace(0, point, point + to_sun * SUN_DISTANCE, 0), RayEnd::Sky);
        ModelLighting {
            cube,
            sun: if sees_sun { 1.0 } else { 0.0 },
        }
    }
}

/// Raw access to the lumps `vbsp` doesn't expose.
struct Lumps<'a> {
    data: &'a [u8],
}

impl<'a> Lumps<'a> {
    fn entry(&self, index: usize) -> Option<(usize, usize, i32, i32)> {
        let start = 8 + index * 16;
        let field = |offset: usize| -> Option<i32> {
            let bytes = self.data.get(start + offset..start + offset + 4)?;
            Some(i32::from_le_bytes(bytes.try_into().ok()?))
        };
        Some((
            field(0)? as usize,
            field(4)? as usize,
            field(8)?,
            field(12)?,
        ))
    }

    fn version(&self, index: usize) -> i32 {
        self.entry(index).map_or(0, |entry| entry.2)
    }

    fn get(&self, index: usize) -> Cow<'a, [u8]> {
        let Some((offset, length, _, uncompressed)) = self.entry(index) else {
            return Cow::Borrowed(&[]);
        };
        let Some(bytes) = self.data.get(offset..offset + length) else {
            return Cow::Borrowed(&[]);
        };
        if uncompressed == 0 {
            return Cow::Borrowed(bytes);
        }
        decompress_lump(bytes).map_or(Cow::Borrowed(&[]), Cow::Owned)
    }
}

/// Valve's LZMA lumps use their own small header, so rebuild a standard one.
fn decompress_lump(bytes: &[u8]) -> Option<Vec<u8>> {
    let expected = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
    let mut stream = Vec::with_capacity(bytes.len() + 8);
    stream.extend_from_slice(bytes.get(12..17)?);
    stream.extend_from_slice(&u64::from(expected).to_le_bytes());
    stream.extend_from_slice(bytes.get(17..)?);
    let mut out = Vec::with_capacity(expected as usize);
    lzma_rs::lzma_decompress(&mut Cursor::new(stream), &mut out).ok()?;
    Some(out)
}

fn color_from_rgbe(bytes: &[u8]) -> [f32; 3] {
    let scale = 2f32.powi(i32::from(bytes[3] as i8)) / 255.0;
    [
        f32::from(bytes[0]) * scale,
        f32::from(bytes[1]) * scale,
        f32::from(bytes[2]) * scale,
    ]
}

/// A simple shelf packer: rows of rectangles, tallest first.
struct Atlas {
    width: usize,
    x: usize,
    y: usize,
    row_height: usize,
}

impl Atlas {
    fn place(&mut self, width: usize, height: usize) -> (usize, usize) {
        if self.x + width > self.width {
            self.x = 0;
            self.y += self.row_height;
            self.row_height = 0;
        }
        let spot = (self.x, self.y);
        self.x += width;
        self.row_height = self.row_height.max(height);
        spot
    }
}

fn dot4(row: [f32; 4], point: Vec3) -> f32 {
    row[0] * point.x + row[1] * point.y + row[2] * point.z + row[3]
}

fn vec3(v: vbsp::Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Source angles (pitch, yaw, roll in degrees) as a rotation matrix.
pub fn rotation_from_angles(pitch: f32, yaw: f32, roll: f32) -> Mat4 {
    Mat4::from_rotation_z(yaw.to_radians())
        * Mat4::from_rotation_y(pitch.to_radians())
        * Mat4::from_rotation_x(roll.to_radians())
}

fn parse_vector(text: &str) -> Option<Vec3> {
    let mut numbers = text.split_whitespace().filter_map(|part| part.parse().ok());
    Some(Vec3::new(numbers.next()?, numbers.next()?, numbers.next()?))
}

/// Keeps one copy of each material and remembers which textures they need.
#[derive(Default)]
struct MaterialTable {
    by_name: HashMap<String, usize>,
    materials: Vec<Material>,
    texture_names: Vec<String>,
    texture_ids: HashMap<String, usize>,
}

impl MaterialTable {
    fn texture(&mut self, name: &str) -> usize {
        let key = name.replace('\\', "/").to_ascii_lowercase();
        if let Some(&id) = self.texture_ids.get(&key) {
            return id;
        }
        let id = self.texture_names.len();
        self.texture_names.push(key.clone());
        self.texture_ids.insert(key, id);
        id
    }

    fn material(&mut self, assets: &Assets, name: &str) -> usize {
        let key = name.to_ascii_lowercase();
        if let Some(&id) = self.by_name.get(&key) {
            return id;
        }
        let material = match assets.material(name) {
            Some(vmt) => self.from_vmt(&vmt),
            None => Material {
                opacity: 1.0,
                tint: [1.0; 3],
                ..Material::default()
            },
        };
        let id = self.materials.len();
        self.materials.push(material);
        self.by_name.insert(key, id);
        id
    }

    fn from_vmt(&mut self, vmt: &Vmt) -> Material {
        let water = vmt.shader == "water";
        let additive = vmt.flag("$additive");
        let base = vmt.get("$basetexture").map(|name| self.texture(name));
        let base2 = vmt.get("$basetexture2").map(|name| self.texture(name));
        let mut tint = vmt.color("$color").unwrap_or([1.0; 3]);
        if water {
            tint = vmt.color("$fogcolor").unwrap_or([0.18, 0.28, 0.32]);
        }
        Material {
            base,
            base2,
            translucent: water || additive || vmt.flag("$translucent"),
            additive,
            alpha_test: vmt
                .flag("$alphatest")
                .then(|| vmt.number("$alphatestreference").filter(|r| *r > 0.0).unwrap_or(0.5)),
            opacity: if water { 0.65 } else { vmt.number("$alpha").unwrap_or(1.0) },
            tint,
            unlit: vmt.shader.starts_with("unlit") || vmt.shader == "sprite",
        }
    }
}

/// Decodes textures on every core, since this is the slowest part of loading.
fn decode_textures(assets: &Assets, names: &[String], on_progress: &(impl Fn(f32) + Sync)) -> Vec<Image> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |count| count.get()).min(12);

    let mut decoded: Vec<(usize, Image)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(name) = names.get(index) else { break };
                        out.push((index, assets.texture(name).unwrap_or_else(missing_texture)));
                        let finished = done.fetch_add(1, Ordering::Relaxed) + 1;
                        on_progress(finished as f32 / names.len().max(1) as f32);
                    }
                    out
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap_or_default())
            .collect()
    });
    decoded.sort_by_key(|(index, _)| *index);
    decoded.into_iter().map(|(_, image)| image).collect()
}

fn missing_texture() -> Image {
    Image {
        width: 1,
        height: 1,
        rgba: vec![128, 128, 128, 255],
    }
}

/// The order a map's baked prop lighting lists colours in: every vertex of every
/// strip group of the first detail level's meshes. Returns where each model
/// vertex first appears in that list, and how many colours each mesh takes.
fn baked_light_order(mdl: &vmdl::Mdl, vtx: &vmdl::Vtx) -> (HashMap<usize, u32>, Vec<u32>) {
    let mut order = HashMap::new();
    let mut counts = Vec::new();
    let mut next = 0u32;
    for (mdl_part, vtx_part) in mdl.body_parts.iter().zip(&vtx.body_parts) {
        for (mdl_model, vtx_model) in mdl_part.models.iter().zip(&vtx_part.models) {
            let Some(lod) = vtx_model.lods.first() else { continue };
            for (mdl_mesh, vtx_mesh) in mdl_model.meshes.iter().zip(&lod.meshes) {
                let base = mdl_mesh.vertex_offset as usize + mdl_model.vertex_offset as usize;
                let mut count = 0;
                for group in &vtx_mesh.strip_groups {
                    for vertex in &group.vertices {
                        order.entry(base + usize::from(vertex.original_mesh_vertex_id)).or_insert(next);
                        next += 1;
                        count += 1;
                    }
                }
                counts.push(count);
            }
        }
    }
    (order, counts)
}

/// Reads the colours a map baked for one static prop (`sp_N.vhv`), checking
/// they line up with the model's own meshes.
fn read_baked_light(data: &[u8], light_counts: &[u32]) -> Option<Vec<u8>> {
    let int = |at: usize| -> Option<u32> { Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?)) };
    let vertex_size = int(12)? as usize;
    let mesh_count = int(20)? as usize;
    if vertex_size != 4 || mesh_count > 4096 {
        return None;
    }
    let mut rgba = Vec::new();
    let mut lod_meshes = 0;
    for mesh in 0..mesh_count {
        let header = 40 + mesh * 28;
        if int(header)? != 0 {
            continue;
        }
        let (count, offset) = (int(header + 4)? as usize, int(header + 8)? as usize);
        if light_counts.get(lod_meshes).copied() != Some(count as u32) {
            return None;
        }
        lod_meshes += 1;
        // Stored blue, green, red, alpha.
        for color in data.get(offset..offset + count * 4)?.chunks_exact(4) {
            rgba.extend_from_slice(&[color[2], color[1], color[0], 255]);
        }
    }
    (lod_meshes == light_counts.len()).then_some(rgba)
}

/// Reads a model's first detail level as a plain triangle mesh.
pub fn load_model(
    assets: &Assets,
    table: &mut impl FnMut(&Assets, &str) -> usize,
    path: &str,
    skin: i32,
) -> Option<ModelData> {
    let mdl = super::skeleton::read_mdl(&assets.load(path)?)?;
    let vtx = vmdl::Vtx::read(&assets.load(&path.replace(".mdl", ".dx90.vtx"))?).ok()?;
    let vvd = vmdl::Vvd::read(&assets.load(&path.replace(".mdl", ".vvd"))?).ok()?;
    let (light_order, light_counts) = baked_light_order(&mdl, &vtx);
    let model = vmdl::Model::from_parts(mdl, vtx, vvd);

    let skins: Vec<_> = model.skin_tables().collect();
    let skin_table = skins.get(skin.max(0) as usize).or(skins.first())?;
    let directories = model.texture_directories().to_vec();

    let mut data = ModelData {
        vertices: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
        light_counts,
    };
    let source = model.vertices();

    for mesh in model.meshes() {
        let Some(texture) = skin_table.texture(mesh.material_index()) else {
            continue;
        };
        let material_name = directories
            .iter()
            .map(|directory| format!("{}{}", directory.replace('\\', "/"), texture))
            .find(|name| assets.load(&format!("materials/{}.vmt", name.to_ascii_lowercase())).is_some())
            .unwrap_or_else(|| texture.to_string());
        let material = table(assets, &material_name);

        let first_index = data.indices.len() as u32;
        let mut remap: HashMap<usize, u32> = HashMap::new();
        for strip in mesh.vertex_strip_indices() {
            for index in strip {
                let Some(vertex) = source.get(index) else { continue };
                let mapped = *remap.entry(index).or_insert_with(|| {
                    let position = model.apply_root_transform(vertex.position);
                    let normal = model.apply_root_transform(vertex.normal);
                    data.vertices.push(ModelVertex {
                        position: [position.x, position.y, position.z],
                        normal: [normal.x, normal.y, normal.z],
                        uv: vertex.texture_coordinates,
                        light: light_order.get(&index).map_or(-1.0, |&at| at as f32),
                    });
                    (data.vertices.len() - 1) as u32
                });
                data.indices.push(mapped);
            }
        }
        data.parts.push(ModelPart {
            material,
            first_index,
            index_count: data.indices.len() as u32 - first_index,
        });
    }
    (!data.parts.is_empty()).then_some(data)
}

#[derive(Default)]
struct WorldBuilder {
    batches: HashMap<(usize, bool), (Vec<WorldVertex>, Vec<u32>)>,
}

impl WorldBuilder {
    fn add_triangles(
        &mut self,
        material: usize,
        sky: bool,
        vertices: &[WorldVertex],
        indices: impl Iterator<Item = u32>,
    ) {
        let (target_vertices, target_indices) = self.batches.entry((material, sky)).or_default();
        let base = target_vertices.len() as u32;
        target_vertices.extend_from_slice(vertices);
        target_indices.extend(indices.map(|index| base + index));
    }
}

struct FaceLightmap {
    x: usize,
    y: usize,
    min: [f32; 2],
}

/// The corners of a displacement's base face, starting from its start position.
fn displacement_corners(corners: &[Vec3], start: Vec3) -> Option<[Vec3; 4]> {
    let mut corners: [Vec3; 4] = corners.try_into().ok()?;
    let first = (0..4).min_by(|&a, &b| {
        let a = corners[a].distance_squared(start);
        let b = corners[b].distance_squared(start);
        a.total_cmp(&b)
    })?;
    corners.rotate_left(first);
    Some(corners)
}

/// Copies one face's lighting into the atlas, repeating its edges into the
/// padding so neighbouring faces can't bleed into it.
fn write_lightmap(
    atlas: &mut [f32],
    lighting: &[u8],
    offset: usize,
    size: (usize, usize),
    spot: (usize, usize),
) {
    let (width, height) = size;
    for y in 0..height + 2 * ATLAS_PADDING {
        for x in 0..width + 2 * ATLAS_PADDING {
            let source_x = x.saturating_sub(ATLAS_PADDING).min(width - 1);
            let source_y = y.saturating_sub(ATLAS_PADDING).min(height - 1);
            let at = offset + (source_y * width + source_x) * 4;
            let Some(bytes) = lighting.get(at..at + 4) else { continue };
            let target = ((spot.1 + y) * ATLAS_SIZE + spot.0 + x) * 3;
            atlas[target..target + 3].copy_from_slice(&color_from_rgbe(bytes));
        }
    }
}

fn build_world(
    bsp: &Bsp,
    assets: &Assets,
    table: &mut MaterialTable,
    lumps: &Lumps,
    sky_faces: &std::collections::HashSet<usize>,
) -> (WorldBuilder, Vec<f32>, usize) {
    // Faces store offsets into the plain lighting data. The HDR copy has its own
    // face list, so it's only a fallback for maps without any plain lighting.
    let ldr = lumps.get(LUMP_LIGHTING);
    let lighting = if ldr.is_empty() { lumps.get(LUMP_LIGHTING_HDR) } else { ldr };

    let mut models: Vec<(usize, Vec3)> = vec![(0, Vec3::ZERO)];
    for entity in bsp.entities.iter() {
        let class = entity.prop("classname").unwrap_or_default();
        if !VISIBLE_BRUSH_ENTITIES.contains(&class) {
            continue;
        }
        // Render mode 10 means "don't draw", which maps use for hidden doors and blockers.
        if entity.prop("rendermode").and_then(|mode| mode.trim().parse::<i32>().ok()) == Some(RENDER_MODE_NONE) {
            continue;
        }
        let index = entity
            .prop("model")
            .and_then(|model| model.strip_prefix('*'))
            .and_then(|index| index.parse::<usize>().ok());
        if let Some(index) = index {
            let origin = entity.prop("origin").and_then(parse_vector).unwrap_or(Vec3::ZERO);
            models.push((index, origin));
        }
    }

    // Tallest first keeps the shelves tight.
    let mut faces = Vec::new();
    let all_models: Vec<_> = bsp.models().collect();
    for (index, origin) in models {
        let Some(model) = all_models.get(index) else { continue };
        let first = model.first_face.max(0) as usize;
        for (offset, face) in model.faces().enumerate() {
            let in_sky = index == 0 && sky_faces.contains(&(first + offset));
            let name = face.texture().name();
            if !face.is_visible() || name.to_ascii_lowercase().starts_with("tools/") {
                continue;
            }
            faces.push((face, origin, in_sky));
        }
    }
    let face_size = |face: &vbsp::Handle<vbsp::Face>| {
        (
            face.light_map_texture_size[0] as usize + 1,
            face.light_map_texture_size[1] as usize + 1,
        )
    };
    faces.sort_by_key(|(face, _, _)| std::cmp::Reverse(face_size(face).1));

    // One white texel in the corner serves faces with no lighting data.
    let mut atlas = vec![0f32; ATLAS_SIZE * ATLAS_SIZE * 3];
    atlas[..9].fill(1.0);
    let mut packer = Atlas {
        width: ATLAS_SIZE,
        x: 4,
        y: 0,
        row_height: 4,
    };
    let mut world = WorldBuilder::default();

    for (face, origin, in_sky) in &faces {
        let texture = face.texture();
        let material = table.material(assets, texture.name());
        let data = texture.texture_data();
        let (texture_width, texture_height) = (data.width.max(1) as f32, data.height.max(1) as f32);
        let (u_row, v_row) = (texture.texture_transforms_u, texture.texture_transforms_v);
        let (lm_u_row, lm_v_row) = (texture.light_map_scale, texture.light_map_transform);

        let (width, height) = face_size(face);
        let has_light = face.light_offset >= 0
            && face.styles[0] != 255
            && !texture.flags.intersects(TextureFlags::NOLIGHT);
        let lightmap = has_light.then(|| {
            let spot = packer.place(width + 2 * ATLAS_PADDING, height + 2 * ATLAS_PADDING);
            if spot.1 + height + 2 * ATLAS_PADDING <= ATLAS_SIZE {
                write_lightmap(&mut atlas, &lighting, face.light_offset as usize, (width, height), spot);
            }
            FaceLightmap {
                x: spot.0 + ATLAS_PADDING,
                y: spot.1 + ATLAS_PADDING,
                min: [
                    face.light_map_texture_min[0] as f32,
                    face.light_map_texture_min[1] as f32,
                ],
            }
        });

        let lightmap_uv = |base: Vec3| -> [f32; 2] {
            match &lightmap {
                Some(spot) => [
                    (spot.x as f32 + 0.5 + dot4(lm_u_row, base) - spot.min[0]) / ATLAS_SIZE as f32,
                    (spot.y as f32 + 0.5 + dot4(lm_v_row, base) - spot.min[1]) / ATLAS_SIZE as f32,
                ],
                None => [1.5 / ATLAS_SIZE as f32, 1.5 / ATLAS_SIZE as f32],
            }
        };
        let vertex = |position: Vec3, base: Vec3, blend: f32| WorldVertex {
            position: (position + *origin).to_array(),
            uv: [
                dot4(u_row, base) / texture_width,
                dot4(v_row, base) / texture_height,
            ],
            lightmap_uv: lightmap_uv(base),
            blend,
        };

        let corners: Vec<Vec3> = face.vertices().map(|v| vec3(v.position)).collect();

        if let Some(displacement) = face.displacement() {
            let Some(corners) = displacement_corners(&corners, vec3(displacement.start_position)) else {
                continue;
            };
            let side = 2usize.pow(displacement.power as u32) + 1;
            let offsets: Vec<_> = displacement.displacement_vertices().collect();
            if offsets.len() < side * side {
                continue;
            }
            let step = 1.0 / (side - 1) as f32;
            let mut vertices = Vec::with_capacity(side * side);
            for row in 0..side {
                let left = corners[0].lerp(corners[1], row as f32 * step);
                let right = corners[3].lerp(corners[2], row as f32 * step);
                for column in 0..side {
                    let base = left.lerp(right, column as f32 * step);
                    let offset = &offsets[row * side + column];
                    let moved = base + vec3(offset.displacement());
                    let mut point = vertex(moved, base, offset.alpha / 255.0);
                    // A displacement's lighting is laid out on its own grid, from the
                    // start corner, rather than projected the way flat faces are.
                    if let Some(spot) = &lightmap {
                        let s = column as f32 * step * (width - 1) as f32;
                        let t = row as f32 * step * (height - 1) as f32;
                        point.lightmap_uv = [
                            (spot.x as f32 + 0.5 + s) / ATLAS_SIZE as f32,
                            (spot.y as f32 + 0.5 + t) / ATLAS_SIZE as f32,
                        ];
                    }
                    vertices.push(point);
                }
            }
            let mut indices = Vec::with_capacity((side - 1) * (side - 1) * 6);
            for row in 0..side - 1 {
                for column in 0..side - 1 {
                    let a = (row * side + column) as u32;
                    let b = a + 1;
                    let c = a + side as u32;
                    let d = c + 1;
                    indices.extend_from_slice(&[a, c, b, b, c, d]);
                }
            }
            world.add_triangles(material, *in_sky, &vertices, indices.into_iter());
        } else {
            let vertices: Vec<_> = corners.iter().map(|&p| vertex(p, p, 0.0)).collect();
            let fan = (1..vertices.len().saturating_sub(1) as u32).flat_map(|i| [0, i, i + 1]);
            world.add_triangles(material, *in_sky, &vertices, fan);
        }
    }

    let used_height = (packer.y + packer.row_height).min(ATLAS_SIZE);
    (world, atlas, used_height)
}

fn read_light_probe(bsp: &Bsp, lumps: &Lumps) -> LightProbe {
    let planes = bsp.planes.iter().map(|p| (vec3(p.normal), p.dist)).collect();
    let nodes = bsp.nodes.iter().map(|n| (n.plane_index, n.children)).collect();
    let is_sky_face = |index: usize| {
        bsp.faces.get(index).is_some_and(|face| {
            let info = bsp.textures_info.get(face.texture_info as usize);
            info.is_some_and(|info| info.flags.intersects(TextureFlags::SKY | TextureFlags::SKY2D))
        })
    };
    let node_is_sky = bsp
        .nodes
        .iter()
        .map(|node| {
            let first = usize::from(node.first_face);
            (first..first + usize::from(node.face_count)).any(is_sky_face)
        })
        .collect();

    let leaves = lumps.get(LUMP_LEAFS);
    let leaf_size = if lumps.version(LUMP_LEAFS) == 0 { 56 } else { 32 };
    let leaf_bounds = |leaf: &[u8]| -> (Vec3, Vec3) {
        let read = |offset: usize| f32::from(i16::from_le_bytes([leaf[offset], leaf[offset + 1]]));
        (
            Vec3::new(read(8), read(10), read(12)),
            Vec3::new(read(14), read(16), read(18)),
        )
    };
    let leaf_contents: Vec<i32> = leaves
        .chunks_exact(leaf_size)
        .map(|leaf| i32::from_le_bytes([leaf[0], leaf[1], leaf[2], leaf[3]]))
        .collect();
    let leaf_areas: Vec<u16> = leaves
        .chunks_exact(leaf_size)
        .map(|leaf| u16::from_le_bytes([leaf[6], leaf[7]]) & 0x1FF)
        .collect();

    let plain_index = lumps.get(LUMP_LEAF_AMBIENT_INDEX);
    let (index, samples) = if plain_index.is_empty() {
        (lumps.get(LUMP_LEAF_AMBIENT_INDEX_HDR), lumps.get(LUMP_LEAF_AMBIENT_LIGHTING_HDR))
    } else {
        (plain_index, lumps.get(LUMP_LEAF_AMBIENT_LIGHTING))
    };

    let mut leaf_cubes = vec![Vec::new(); leaf_contents.len()];
    for (leaf, entry) in index.chunks_exact(4).enumerate() {
        let count = usize::from(u16::from_le_bytes([entry[0], entry[1]]));
        let first = usize::from(u16::from_le_bytes([entry[2], entry[3]]));
        let Some(bounds) = leaves.chunks_exact(leaf_size).nth(leaf).map(leaf_bounds) else {
            continue;
        };
        for sample in samples.chunks_exact(28).skip(first).take(count) {
            let mut cube = [[0.0; 3]; 6];
            for (side, value) in cube.iter_mut().enumerate() {
                // Unlike lightmaps, these are stored on a 0 to 1 scale rather than 0 to 255.
                *value = color_from_rgbe(&sample[side * 4..side * 4 + 4]).map(|channel| channel * 255.0);
            }
            let fraction = Vec3::new(
                f32::from(sample[24]) / 255.0,
                f32::from(sample[25]) / 255.0,
                f32::from(sample[26]) / 255.0,
            );
            leaf_cubes[leaf].push(AmbientSample {
                position: bounds.0 + (bounds.1 - bounds.0) * fraction,
                cube,
            });
        }
    }

    let sun = bsp
        .entities
        .iter()
        .find(|entity| entity.prop("classname") == Some("light_environment"))
        .map_or(Sun::default(), |entity| read_sun(&entity));

    LightProbe {
        terrain: Terrain::default(),
        planes,
        nodes,
        node_is_sky,
        leaf_contents,
        leaf_areas,
        leaf_cubes,
        sun,
        default_cube: [[0.35; 3]; 6],
    }
}

fn read_sun(entity: &vbsp::RawEntity) -> Sun {
    let angles = entity.prop("angles").and_then(parse_vector).unwrap_or(Vec3::ZERO);
    let pitch = entity
        .prop("pitch")
        .and_then(|text| text.trim().parse::<f32>().ok())
        .filter(|pitch| *pitch != 0.0)
        .unwrap_or(angles.x)
        .to_radians();
    let yaw = angles.y.to_radians();
    let shines = Vec3::new(yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin());

    let mut numbers = entity
        .prop("_light")
        .unwrap_or("255 255 255 200")
        .split_whitespace()
        .filter_map(|part| part.parse::<f32>().ok());
    let channel = |value: f32| (value / 255.0).powf(2.2);
    let (r, g, b) = (
        channel(numbers.next().unwrap_or(255.0)),
        channel(numbers.next().unwrap_or(255.0)),
        channel(numbers.next().unwrap_or(255.0)),
    );
    let brightness = numbers.next().unwrap_or(200.0) / 255.0;

    Sun {
        direction: (-shines).normalize_or_zero().to_array(),
        color: [r * brightness, g * brightness, b * brightness],
    }
}

pub fn load_map(
    loader: &Loader,
    map_name: &str,
    content: &DemoContent,
    on_progress: impl Fn(f32) + Sync,
) -> Result<MapData, String> {
    let sky_name = content.sky_name.as_str();
    let file = loader
        .load(&format!("maps/{map_name}.bsp"))
        .ok()
        .flatten()
        .ok_or_else(|| format!("The map \"{map_name}\" isn't in your TF2 folder, so this demo can't be shown."))?;
    let bsp = Bsp::read(&file).map_err(|error| format!("The map \"{map_name}\" couldn't be read: {error}"))?;
    let lumps = Lumps { data: &file };
    let assets = Assets::new(loader, Some(&bsp));
    let mut table = MaterialTable::default();

    let mut lighting = read_light_probe(&bsp, &lumps);
    lighting.terrain = Terrain::new(collision_triangles(&bsp));
    let sky_camera = bsp
        .entities
        .iter()
        .find(|entity| entity.prop("classname") == Some("sky_camera"))
        .and_then(|entity| {
            let origin = entity.prop("origin").and_then(parse_vector)?;
            let scale = entity
                .prop("scale")
                .and_then(|scale| scale.trim().parse::<f32>().ok())
                .filter(|scale| *scale > 0.0)
                .unwrap_or(16.0);
            Some(SkyCamera { origin, scale })
        });
    let sky_area = sky_camera.and_then(|camera| lighting.area_at(camera.origin));

    // Every face in the backdrop's sealed region belongs to the backdrop.
    let mut sky_faces = std::collections::HashSet::new();
    if let Some(area) = sky_area {
        let leaves = lumps.get(LUMP_LEAFS);
        let leaf_size = if lumps.version(LUMP_LEAFS) == 0 { 56 } else { 32 };
        let leaf_faces = lumps.get(LUMP_LEAF_FACES);
        for leaf in leaves.chunks_exact(leaf_size) {
            if u16::from_le_bytes([leaf[6], leaf[7]]) & 0x1FF != area {
                continue;
            }
            let first = usize::from(u16::from_le_bytes([leaf[20], leaf[21]]));
            let count = usize::from(u16::from_le_bytes([leaf[22], leaf[23]]));
            for entry in leaf_faces.chunks_exact(2).skip(first).take(count) {
                sky_faces.insert(usize::from(u16::from_le_bytes([entry[0], entry[1]])));
            }
        }
    }

    let (world, atlas, atlas_height) = build_world(&bsp, &assets, &mut table, &lumps, &sky_faces);
    on_progress(0.1);

    // Static props, loaded once per model and skin however often they're placed.
    let mut model_ids: HashMap<(String, i32), Option<usize>> = HashMap::new();
    let mut models: Vec<ModelData> = Vec::new();
    let mut props = Vec::new();
    let mut prop_light: Vec<u8> = Vec::new();
    for (prop_index, prop) in bsp.static_props().enumerate() {
        let path = prop.model().replace('\\', "/").to_ascii_lowercase();
        let model = *model_ids.entry((path.clone(), prop.skin)).or_insert_with(|| {
            let mut resolve = |assets: &Assets, name: &str| table.material(assets, name);
            load_model(&assets, &mut resolve, &path, prop.skin).map(|data| {
                models.push(data);
                models.len() - 1
            })
        });
        let Some(model) = model else { continue };

        let origin = vec3(prop.origin);
        let angles = prop.angles;
        let transform = Mat4::from_translation(origin)
            * rotation_from_angles(angles.pitch, angles.yaw, angles.roll);
        let lighting_at = origin + Vec3::Z * 16.0;
        let baked = assets
            .load(&format!("sp_{prop_index}.vhv"))
            .and_then(|data| read_baked_light(&data, &models[model].light_counts))
            .map(|colors| {
                let start = (prop_light.len() / 4) as u32;
                prop_light.extend_from_slice(&colors);
                start
            });
        props.push(PropInstance {
            model,
            transform,
            lighting: lighting.at(lighting_at),
            sky: sky_area.is_some() && lighting.area_at(origin + Vec3::Z) == sky_area,
            baked,
        });
    }
    on_progress(0.25);

    // Movable objects from the map file join the demo's own models.
    let mut model_paths = content.models.clone();
    if model_paths.is_empty() {
        model_paths.push(String::new());
    }
    let mut map_props = Vec::new();
    for entity in bsp.entities.iter() {
        let class = entity.prop("classname").unwrap_or_default();
        if !MAP_PROP_CLASSES.contains(&class) {
            continue;
        }
        let hidden = entity.prop("rendermode").and_then(|mode| mode.trim().parse::<i32>().ok()) == Some(RENDER_MODE_NONE)
            || entity.prop("StartDisabled").is_some_and(|value| value.trim() == "1");
        let Some(path) = entity.prop("model").map(|path| path.replace('\\', "/").to_ascii_lowercase()) else {
            continue;
        };
        if hidden || !path.ends_with(".mdl") {
            continue;
        }
        let origin = entity.prop("origin").and_then(parse_vector).unwrap_or(Vec3::ZERO);
        let angles = entity.prop("angles").and_then(parse_vector).unwrap_or(Vec3::ZERO);
        let model = match model_paths.iter().position(|known| *known == path) {
            Some(index) => index,
            None => {
                model_paths.push(path);
                model_paths.len() - 1
            }
        };
        map_props.push(MapProp {
            model,
            transform: Mat4::from_translation(origin) * rotation_from_angles(angles.x, angles.y, angles.z),
            origin,
            skin: entity.prop("skin").and_then(|skin| skin.trim().parse().ok()).unwrap_or(0),
            attached: entity.prop("parentname").is_some_and(|parent| !parent.is_empty()),
        });
    }

    let mut meshes = Vec::new();
    let characters = {
        let mut resolve = |assets: &Assets, name: &str| table.material(assets, name);
        load_characters(&assets, &mut resolve, &content.classes, &model_paths, &mut meshes)
    };

    // The scope multiplies the view by this image at double strength, so its dark
    // edge turns black and the middle stays clear. That becomes a black mask.
    let scope = assets.material("hud/scope_sniper_ul").and_then(|vmt| {
        let mut image = assets.texture(vmt.get("$refracttinttexture").or(vmt.get("$basetexture"))?)?;
        for pixel in image.rgba.chunks_exact_mut(4) {
            let light = u32::from(pixel[0].max(pixel[1]).max(pixel[2]));
            let shown = (light * 2 * u32::from(pixel[3]) / 255).min(255);
            pixel.copy_from_slice(&[0, 0, 0, (255 - shown) as u8]);
        }
        Some(image)
    });

    let sky = (!sky_name.is_empty())
        .then(|| {
            let mut faces = [0usize; 6];
            for (slot, suffix) in faces.iter_mut().zip(SKY_SUFFIXES) {
                let vmt = assets.material(&format!("skybox/{sky_name}{suffix}"))?;
                *slot = table.texture(vmt.get("$basetexture")?);
            }
            Some(faces)
        })
        .flatten();

    let textures = decode_textures(&assets, &table.texture_names, &|share| {
        on_progress(0.25 + share * 0.75);
    });

    // Opaque surfaces go first, then anything see-through.
    let mut order: Vec<_> = world.batches.into_iter().collect();
    order.sort_by_key(|((material, _), _)| table.materials[*material].translucent);

    let mut world_vertices = Vec::new();
    let mut world_indices = Vec::new();
    let mut batches = Vec::new();
    for ((material, sky), (vertices, indices)) in order {
        let base = world_vertices.len() as u32;
        let first_index = world_indices.len() as u32;
        world_vertices.extend_from_slice(&vertices);
        world_indices.extend(indices.iter().map(|index| base + index));
        batches.push(Batch {
            material,
            first_index,
            index_count: indices.len() as u32,
            sky,
        });
    }

    let mut low = Vec3::splat(f32::MAX);
    let mut high = Vec3::splat(f32::MIN);
    for vertex in batches
        .iter()
        .filter(|batch| !batch.sky)
        .flat_map(|batch| &world_indices[batch.first_index as usize..(batch.first_index + batch.index_count) as usize])
        .map(|index| &world_vertices[*index as usize])
    {
        low = low.min(Vec3::from(vertex.position));
        high = high.max(Vec3::from(vertex.position));
    }

    // The atlas only keeps the rows it used, so heights have to be measured against that.
    let lightmap_height = atlas_height.max(4);
    let mut lightmap = atlas;
    lightmap.truncate(ATLAS_SIZE * lightmap_height * 3);
    let v_scale = ATLAS_SIZE as f32 / lightmap_height as f32;
    for vertex in &mut world_vertices {
        vertex.lightmap_uv[1] *= v_scale;
    }

    Ok(MapData {
        world_vertices,
        world_indices,
        batches,
        materials: table.materials,
        textures,
        lightmap_width: ATLAS_SIZE,
        lightmap_height,
        lightmap,
        models,
        props,
        prop_light,
        sky,
        lighting: Arc::new(lighting),
        sky_camera,
        bounds: (low, high),
        meshes,
        characters,
        map_props,
        scope,
        crosshair: assets
            .material("vgui/crosshairs/default")
            .and_then(|vmt| assets.texture(vmt.get("$basetexture")?)),
        weapon_roles: super::items::weapon_roles(&assets, &content.models),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgbe_colors_scale_by_their_exponent() {
        assert_eq!(color_from_rgbe(&[255, 0, 0, 0]), [1.0, 0.0, 0.0]);
        assert_eq!(color_from_rgbe(&[255, 0, 0, 1]), [2.0, 0.0, 0.0]);
        assert_eq!(color_from_rgbe(&[255, 0, 0, 0xFF]), [0.5, 0.0, 0.0]);
    }

    #[test]
    fn the_packer_starts_a_new_row_when_full() {
        let mut atlas = Atlas { width: 10, x: 0, y: 0, row_height: 0 };
        assert_eq!(atlas.place(6, 3), (0, 0));
        assert_eq!(atlas.place(3, 2), (6, 0));
        assert_eq!(atlas.place(5, 1), (0, 3));
    }

    #[test]
    fn displacement_corners_start_at_the_nearest_corner() {
        let square = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let rotated = displacement_corners(&square, Vec3::new(0.9, 1.1, 0.0)).unwrap();
        assert_eq!(rotated[0], Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(rotated[1], Vec3::new(0.0, 1.0, 0.0));
        assert!(displacement_corners(&square[..3], Vec3::ZERO).is_none());
    }

    #[test]
    fn yaw_turns_forward_towards_positive_y() {
        let forward = rotation_from_angles(0.0, 90.0, 0.0).transform_vector3(Vec3::X);
        assert!((forward - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn positive_pitch_looks_down() {
        let forward = rotation_from_angles(90.0, 0.0, 0.0).transform_vector3(Vec3::X);
        assert!((forward + Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn rays_stop_on_terrain() {
        // A 512 x 512 floor at height 10, as two triangles.
        let (a, b, c, d) = (
            Vec3::new(0.0, 0.0, 10.0),
            Vec3::new(512.0, 0.0, 10.0),
            Vec3::new(0.0, 512.0, 10.0),
            Vec3::new(512.0, 512.0, 10.0),
        );
        let terrain = Terrain::new(vec![[a, b, c], [b, d, c]]);
        let down = terrain.first_hit(Vec3::new(300.0, 300.0, 50.0), Vec3::new(300.0, 300.0, -50.0));
        assert!((down.unwrap() - 0.4).abs() < 1e-4, "{down:?}");
        // A long slanted line crossing several cells.
        let slant = terrain.first_hit(Vec3::new(-100.0, 10.0, 110.0), Vec3::new(900.0, 400.0, -90.0));
        assert!((slant.unwrap() - 0.5).abs() < 1e-4, "{slant:?}");
        // Above the floor the whole way, and beside it.
        assert_eq!(terrain.first_hit(Vec3::new(0.0, 0.0, 50.0), Vec3::new(500.0, 500.0, 20.0)), None);
        assert_eq!(terrain.first_hit(Vec3::new(700.0, 700.0, 50.0), Vec3::new(700.0, 700.0, -50.0)), None);
    }

    #[test]
    fn light_probe_blocks_rays_through_solid_leaves() {
        let probe = |is_sky: bool| LightProbe {
            terrain: Terrain::default(),
            planes: vec![(Vec3::NEG_X, -100.0)],
            nodes: vec![(0, [!0, !1])],
            node_is_sky: vec![is_sky],
            leaf_contents: vec![0, CONTENTS_SOLID],
            leaf_areas: vec![1, 1],
            leaf_cubes: vec![Vec::new(), Vec::new()],
            sun: Sun::default(),
            default_cube: [[0.0; 3]; 6],
        };
        let near = Vec3::new(50.0, 0.0, 0.0);
        let far = Vec3::new(200.0, 0.0, 0.0);
        assert_eq!(probe(false).trace(0, Vec3::ZERO, near, 0), RayEnd::Clear);
        assert_eq!(probe(false).trace(0, Vec3::ZERO, far, 0), RayEnd::Wall);
        assert_eq!(probe(true).trace(0, Vec3::ZERO, far, 0), RayEnd::Sky);
        assert_eq!(probe(false).first_hit(Vec3::ZERO, near), None);
        assert_eq!(probe(false).first_hit(Vec3::ZERO, far), Some(Vec3::new(100.0, 0.0, 0.0)));
    }
}


