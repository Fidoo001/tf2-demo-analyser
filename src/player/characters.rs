//! Loads the animated models: the nine classes, the weapons they hold, and
//! anything else the demo shows with a model, such as rockets and sentries.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use super::assets::Assets;
use super::skeleton::{Animations, Pose, Rig, Skeleton};

/// Model file names by class number, as the game numbers them.
const CLASS_FILES: [&str; 10] = [
    "", "scout", "sniper", "soldier", "demo", "medic", "heavy", "pyro", "spy", "engineer",
];
/// Red and blue are the first two skins of team-coloured models.
const TEAM_SKINS: usize = 2;
pub const MAX_BONES: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub bones: [f32; 3],
    pub weights: [f32; 3],
}

/// One drawable piece of a model, and which body group option it belongs to.
#[derive(Clone, Copy, Debug)]
pub struct MeshPart {
    pub material: usize,
    pub first_index: u32,
    pub index_count: u32,
    pub group: u8,
    pub option: u8,
}

/// What the GPU needs to draw one model.
pub struct MeshData {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
    /// Parts to draw for each skin. Skin 0 is red and skin 1 blue where a model has both.
    pub skins: Vec<Vec<MeshPart>>,
    /// For each body group, its place value in the body number and how many options it has.
    pub body_groups: Vec<(u32, u32)>,
}

pub struct ClassBody {
    pub mesh: usize,
    pub rig: Rig,
    /// The class's first-person arms, indexed like `Characters::skeletons`.
    pub arms: Option<usize>,
}

/// The CPU side of the animated models: skeletons and animations.
#[derive(Default)]
pub struct Characters {
    pub skeletons: Vec<Skeleton>,
    /// Indexed by class number.
    pub classes: Vec<Option<ClassBody>>,
    /// Indexed like `Recording::models`.
    pub by_model: Vec<Option<usize>>,
    /// How each mesh sits when nothing animates it, indexed like `skeletons`.
    pub idle_poses: Vec<Pose>,
    /// Each mesh's own animations, for things like arms and map objects.
    pub animations: Vec<Option<Animations>>,
}

/// Reads a model and its skeleton. Each skin gets its own list of parts.
fn load_mesh(
    assets: &Assets,
    material: &mut dyn FnMut(&Assets, &str) -> usize,
    path: &str,
) -> Option<(MeshData, Skeleton)> {
    let bytes = assets.load(path)?;
    let skeleton = Skeleton::read(&bytes);
    if skeleton.bones.is_empty() || skeleton.bones.len() > MAX_BONES {
        return None;
    }
    let mdl = super::skeleton::read_mdl(&bytes)?;
    // Which body group option each mesh belongs to, in the order `meshes()` walks them.
    let mut groups = Vec::new();
    let mut body_groups = Vec::new();
    let mut place = 1u32;
    for (group, part) in mdl.body_parts.iter().enumerate() {
        let options = part.models.len().max(1) as u32;
        body_groups.push((place, options));
        place = place.saturating_mul(options);
        for (option, model) in part.models.iter().enumerate() {
            groups.extend(std::iter::repeat_n((group as u8, option as u8), model.meshes.len()));
        }
    }
    let vtx = vmdl::Vtx::read(&assets.load(&path.replace(".mdl", ".dx90.vtx"))?).ok()?;
    let vvd = vmdl::Vvd::read(&assets.load(&path.replace(".mdl", ".vvd"))?).ok()?;
    let model = vmdl::Model::from_parts(mdl, vtx, vvd);

    let directories: Vec<String> = model
        .texture_directories()
        .iter()
        .map(|directory| directory.replace('\\', "/").trim_start_matches('/').to_string())
        .collect();
    let mut resolved: HashMap<String, usize> = HashMap::new();
    let mut resolve = |name: &str| -> usize {
        if let Some(&id) = resolved.get(name) {
            return id;
        }
        let full = directories
            .iter()
            .map(|directory| format!("{directory}{name}"))
            .find(|candidate| assets.material(candidate).is_some())
            .unwrap_or_else(|| name.to_string());
        let id = material(assets, &full);
        resolved.insert(name.to_string(), id);
        id
    };

    let source = model.vertices();
    let mut data = MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
        skins: Vec::new(),
        body_groups,
    };
    let skin_tables: Vec<_> = model.skin_tables().take(TEAM_SKINS).collect();
    data.skins = vec![Vec::new(); skin_tables.len().max(1)];

    for (mesh_number, mesh) in model.meshes().enumerate() {
        let (group, option) = groups.get(mesh_number).copied().unwrap_or((0, 0));
        let first_index = data.indices.len() as u32;
        let mut remap: HashMap<usize, u32> = HashMap::new();
        for strip in mesh.vertex_strip_indices() {
            for index in strip {
                let Some(vertex) = source.get(index) else { continue };
                let mapped = *remap.entry(index).or_insert_with(|| {
                    let mut bones = [0.0; 3];
                    let mut weights = [0.0; 3];
                    for (slot, weight) in vertex.bone_weights.weights().take(3).enumerate() {
                        bones[slot] = f32::from(weight.bone_id);
                        weights[slot] = weight.weight;
                    }
                    if weights.iter().sum::<f32>() <= 0.0 {
                        weights[0] = 1.0;
                    }
                    data.vertices.push(SkinnedVertex {
                        position: [vertex.position.x, vertex.position.y, vertex.position.z],
                        normal: [vertex.normal.x, vertex.normal.y, vertex.normal.z],
                        uv: vertex.texture_coordinates,
                        bones,
                        weights,
                    });
                    (data.vertices.len() - 1) as u32
                });
                data.indices.push(mapped);
            }
        }
        let index_count = data.indices.len() as u32 - first_index;
        if index_count == 0 {
            continue;
        }
        for (skin, table) in skin_tables.iter().enumerate() {
            if let Some(texture) = table.texture(mesh.material_index()) {
                data.skins[skin].push(MeshPart {
                    material: resolve(texture),
                    first_index,
                    index_count,
                    group,
                    option,
                });
            }
        }
    }
    let has_parts = data.skins.iter().any(|parts| !parts.is_empty());
    has_parts.then_some((data, skeleton))
}

/// Loads a model along with its idle pose and its own animations.
fn load_animated(
    assets: &Assets,
    material: &mut dyn FnMut(&Assets, &str) -> usize,
    path: &str,
    meshes: &mut Vec<MeshData>,
    characters: &mut Characters,
) -> Option<usize> {
    let (mesh, skeleton) = load_mesh(assets, material, path)?;
    let bytes = assets.load(path).unwrap_or_default();
    let idle = skeleton.idle_pose(&bytes);
    let includes = Animations::include_paths(&bytes)
        .iter()
        .filter_map(|include| assets.load(include))
        .collect();
    let animations = Animations::read(bytes, includes, &skeleton);
    meshes.push(mesh);
    characters.idle_poses.push(idle);
    characters.animations.push((animations.sequence_count() > 0).then_some(animations));
    characters.skeletons.push(skeleton);
    Some(meshes.len() - 1)
}

/// Loads every class in `classes` and every model in `models`.
pub fn load_characters(
    assets: &Assets,
    material: &mut dyn FnMut(&Assets, &str) -> usize,
    classes: &[u8],
    models: &[String],
    meshes: &mut Vec<MeshData>,
) -> Characters {
    let mut characters = Characters {
        classes: (0..CLASS_FILES.len()).map(|_| None).collect(),
        ..Characters::default()
    };

    for &class in classes {
        let Some(name) = CLASS_FILES.get(usize::from(class)).filter(|name| !name.is_empty()) else {
            continue;
        };
        let path = format!("models/player/{name}.mdl");
        let Some((mesh, skeleton)) = load_mesh(assets, material, &path) else { continue };
        let Some(model_bytes) = assets.load(&path) else { continue };
        let animations = assets
            .load(&format!("models/player/{name}_animations.mdl"))
            .unwrap_or_default();
        let rig = Rig::read(&model_bytes, &animations);

        meshes.push(mesh);
        characters.idle_poses.push(skeleton.rest_pose());
        characters.animations.push(None);
        characters.skeletons.push(skeleton);
        let body = meshes.len() - 1;
        let arms_path = format!("models/weapons/c_models/c_{name}_arms.mdl");
        let arms = load_animated(assets, material, &arms_path, meshes, &mut characters);
        characters.classes[usize::from(class)] = Some(ClassBody { mesh: body, rig, arms });
    }

    for path in models {
        let loaded = if path.is_empty() {
            None
        } else {
            load_animated(assets, material, path, meshes, &mut characters)
        };
        characters.by_model.push(loaded);
    }
    characters
}

