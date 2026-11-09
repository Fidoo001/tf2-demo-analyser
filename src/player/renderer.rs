//! Draws the world with OpenGL into an offscreen image, which then gets copied
//! into the egui window. Everything here runs on the UI thread.

use eframe::glow::{self, HasContext};
use glam::Mat4;

use super::assets::Image;
use super::characters::{MeshData, SkinnedVertex, MAX_BONES};
use super::skeleton::BoneMatrix;
use super::map::{LightCube, MapData, Material, ModelVertex, Sun, WorldVertex, PROP_LIGHT_WIDTH};

const MSAA_SAMPLES: i32 = 4;
const INSTANCE_FLOATS: usize = 16 + 18 + 4 + 1;
const TEXTURE_MAX_ANISOTROPY: u32 = 0x84FE;
const MAX_ANISOTROPY_LIMIT: u32 = 0x84FF;

const WORLD_VERTEX: &str = r#"#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec2 lightmap_uv;
layout(location = 3) in float blend;
uniform mat4 view_projection;
out vec2 v_uv;
out vec2 v_lightmap_uv;
out float v_blend;
void main() {
    v_uv = uv;
    v_lightmap_uv = lightmap_uv;
    v_blend = blend;
    gl_Position = view_projection * vec4(position, 1.0);
}"#;

const WORLD_FRAGMENT: &str = r#"#version 330 core
const float EXPOSURE = 2.2;
in vec2 v_uv;
in vec2 v_lightmap_uv;
in float v_blend;
uniform sampler2D base;
uniform sampler2D base2;
uniform sampler2D lightmap;
uniform int use_blend;
uniform int unlit;
uniform float alpha_test;
uniform float opacity;
uniform vec3 tint;
out vec4 color;
void main() {
    vec4 albedo = texture(base, v_uv);
    if (use_blend == 1) {
        albedo = mix(albedo, texture(base2, v_uv), v_blend);
    }
    if (albedo.a < alpha_test) discard;
    vec3 linear = pow(albedo.rgb, vec3(2.2)) * tint;
    vec3 light = unlit == 1 ? vec3(1.0) : texture(lightmap, v_lightmap_uv).rgb;
    vec3 lit = 1.0 - exp(-linear * light * EXPOSURE);
    color = vec4(pow(lit, vec3(1.0 / 2.2)), albedo.a * opacity);
}"#;

const MODEL_VERTEX: &str = r#"#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in mat4 instance_transform;
layout(location = 7) in vec3 cube0;
layout(location = 8) in vec3 cube1;
layout(location = 9) in vec3 cube2;
layout(location = 10) in vec3 cube3;
layout(location = 11) in vec3 cube4;
layout(location = 12) in vec3 cube5;
layout(location = 13) in vec4 instance_tint;
layout(location = 14) in float instance_baked;
layout(location = 15) in float vertex_baked;
uniform mat4 view_projection;
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform sampler2D baked;
const int BAKED_WIDTH = 4096;
out vec2 v_uv;
out vec3 v_light;
out vec3 v_tint;
void main() {
    vec3 n = normalize(mat3(instance_transform) * normal);
    if (instance_baked >= 0.0 && vertex_baked >= 0.0) {
        // The map's own lighting for this vertex: gamma space with 2x overbright.
        int at = int(instance_baked + 0.5) + int(vertex_baked + 0.5);
        vec3 stored = texelFetch(baked, ivec2(at % BAKED_WIDTH, at / BAKED_WIDTH), 0).rgb;
        v_light = pow(stored * 2.0, vec3(2.2));
    } else {
        vec3 weights = n * n;
        vec3 ambient = weights.x * (n.x >= 0.0 ? cube0 : cube1)
                     + weights.y * (n.y >= 0.0 ? cube2 : cube3)
                     + weights.z * (n.z >= 0.0 ? cube4 : cube5);
        float facing = max(dot(n, sun_direction), 0.0);
        v_light = ambient + sun_color * facing * instance_tint.a;
    }
    v_tint = instance_tint.rgb;
    v_uv = uv;
    gl_Position = view_projection * instance_transform * vec4(position, 1.0);
}"#;

const MODEL_FRAGMENT: &str = r#"#version 330 core
const float EXPOSURE = 2.2;
in vec2 v_uv;
in vec3 v_light;
in vec3 v_tint;
uniform sampler2D base;
uniform float alpha_test;
uniform float opacity;
uniform vec3 tint;
uniform int unlit;
out vec4 color;
void main() {
    vec4 albedo = texture(base, v_uv);
    if (albedo.a < alpha_test) discard;
    vec3 linear = pow(albedo.rgb, vec3(2.2)) * tint * v_tint;
    vec3 lit = 1.0 - exp(-(unlit == 1 ? linear : linear * v_light) * EXPOSURE);
    color = vec4(pow(lit, vec3(1.0 / 2.2)), albedo.a * opacity);
}"#;

const SKINNED_VERTEX: &str = r#"#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 bone_ids;
layout(location = 4) in vec3 bone_weights;
uniform mat4 view_projection;
uniform vec4 bones[384];
uniform vec3 cube[6];
uniform vec3 sun_direction;
uniform vec3 sun_color;
uniform float sun_visible;
out vec2 v_uv;
out vec3 v_light;
out vec3 v_tint;
mat4 bone(float id) {
    int i = int(id + 0.5) * 3;
    vec4 r0 = bones[i];
    vec4 r1 = bones[i + 1];
    vec4 r2 = bones[i + 2];
    return mat4(r0.x, r1.x, r2.x, 0.0, r0.y, r1.y, r2.y, 0.0, r0.z, r1.z, r2.z, 0.0, r0.w, r1.w, r2.w, 1.0);
}
void main() {
    mat4 skin = bone(bone_ids.x) * bone_weights.x
              + bone(bone_ids.y) * bone_weights.y
              + bone(bone_ids.z) * bone_weights.z;
    vec3 n = normalize(mat3(skin) * normal);
    vec3 weights = n * n;
    vec3 ambient = weights.x * (n.x >= 0.0 ? cube[0] : cube[1])
                 + weights.y * (n.y >= 0.0 ? cube[2] : cube[3])
                 + weights.z * (n.z >= 0.0 ? cube[4] : cube[5]);
    float facing = max(dot(n, sun_direction), 0.0);
    v_light = ambient + sun_color * facing * sun_visible;
    v_tint = vec3(1.0);
    v_uv = uv;
    gl_Position = view_projection * skin * vec4(position, 1.0);
}"#;

/// Glowing actors drawn flat into their own image, walls or not.
const GLOW_MASK_FRAGMENT: &str = r#"#version 330 core
in vec2 v_uv;
in vec3 v_light;
in vec3 v_tint;
uniform sampler2D base;
uniform float alpha_test;
uniform vec3 glow_color;
out vec4 color;
void main() {
    if (texture(base, v_uv).a < alpha_test) discard;
    color = vec4(glow_color, 1.0);
}"#;

/// One triangle that covers the whole screen.
const SCREEN_VERTEX: &str = r#"#version 330 core
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}"#;

/// Draws a soft outline around the glow image, leaving the shapes themselves clear.
const GLOW_FRAGMENT: &str = r#"#version 330 core
uniform sampler2D mask;
uniform int radius;
out vec4 color;
bool filled(vec3 c) { return max(c.r, max(c.g, c.b)) > 0.0; }
void main() {
    ivec2 size = textureSize(mask, 0);
    ivec2 at = ivec2(gl_FragCoord.xy);
    if (filled(texelFetch(mask, at, 0).rgb)) discard;
    vec3 best = vec3(0.0);
    float strength = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            vec3 c = texelFetch(mask, clamp(at + ivec2(x, y), ivec2(0), size - 1), 0).rgb;
            float s = 1.0 - length(vec2(x, y)) / float(radius + 1);
            if (filled(c) && s > strength) {
                strength = s;
                best = c;
            }
        }
    }
    if (strength <= 0.0) discard;
    color = vec4(best, min(1.0, strength * 1.6));
}"#;

const TRACE_VERTEX: &str = r#"#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 view_projection;
out vec4 v_color;
void main() {
    v_color = color;
    gl_Position = view_projection * vec4(position, 1.0);
}"#;

const TRACE_FRAGMENT: &str = r#"#version 330 core
in vec4 v_color;
uniform float opacity;
out vec4 color;
void main() {
    color = vec4(v_color.rgb, v_color.a * opacity);
}"#;

/// Floats per trace vertex: position then colour.
pub const TRACE_FLOATS: usize = 7;
/// How strongly overlay lines show where a wall is in the way.
const OVERLAY_HIDDEN_OPACITY: f32 = 0.45;

const SKY_VERTEX: &str = r#"#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
uniform mat4 view_projection;
out vec2 v_uv;
void main() {
    v_uv = uv;
    vec4 clip = view_projection * vec4(position, 1.0);
    gl_Position = clip.xyww;
}"#;

const SKY_FRAGMENT: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D base;
out vec4 color;
void main() {
    color = vec4(texture(base, v_uv).rgb, 1.0);
}"#;

/// Something to draw this frame using a mesh from the shape set.
#[derive(Clone, Copy, Debug)]
pub struct ShapeInstance {
    pub shape: Shape,
    pub transform: Mat4,
    pub tint: [f32; 3],
    pub cube: LightCube,
    pub sun: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Figure,
    Rocket,
    Ball,
    Box,
}

/// An animated model to draw this frame, already posed.
#[derive(Clone, Debug)]
pub struct Actor {
    pub mesh: usize,
    pub skin: usize,
    /// Which body group options to show, as the game numbers them.
    pub body: u32,
    pub bones: Vec<BoneMatrix>,
    pub cube: LightCube,
    pub sun: f32,
    /// An outline in this colour that shows through walls.
    pub glow: Option<[f32; 3]>,
}

/// Everything that changes from one frame to the next.
#[derive(Clone, Debug)]
pub struct Frame {
    pub view: Mat4,
    pub projection: Mat4,
    pub shapes: Vec<ShapeInstance>,
    pub actors: Vec<Actor>,
    /// First-person arms and weapon, drawn last with their own field of view.
    pub view_actors: Vec<Actor>,
    pub view_projection: Mat4,
    /// Camera for the miniature backdrop, placed where the real camera would be at its scale.
    pub sky_view: Option<Mat4>,
    /// Bullet traces as triangles, `TRACE_FLOATS` per vertex.
    pub traces: Vec<f32>,
    /// Lines that stay visible through walls (aim lines, hitboxes), in the same layout as
    /// `traces`. Parts behind walls are drawn fainter.
    pub overlay: Vec<f32>,
}

struct Program {
    program: glow::Program,
    view_projection: Option<glow::UniformLocation>,
    alpha_test: Option<glow::UniformLocation>,
    opacity: Option<glow::UniformLocation>,
    tint: Option<glow::UniformLocation>,
    unlit: Option<glow::UniformLocation>,
    use_blend: Option<glow::UniformLocation>,
    sun_direction: Option<glow::UniformLocation>,
    sun_color: Option<glow::UniformLocation>,
}

unsafe fn compile(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<Program, String> {
    let program = gl.create_program()?;
    let mut shaders = Vec::new();
    for (kind, source) in [(glow::VERTEX_SHADER, vertex), (glow::FRAGMENT_SHADER, fragment)] {
        let shader = gl.create_shader(kind)?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            return Err(gl.get_shader_info_log(shader));
        }
        gl.attach_shader(program, shader);
        shaders.push(shader);
    }
    gl.link_program(program);
    if !gl.get_program_link_status(program) {
        return Err(gl.get_program_info_log(program));
    }
    for shader in shaders {
        gl.detach_shader(program, shader);
        gl.delete_shader(shader);
    }
    let uniform = |name: &str| gl.get_uniform_location(program, name);
    Ok(Program {
        program,
        view_projection: uniform("view_projection"),
        alpha_test: uniform("alpha_test"),
        opacity: uniform("opacity"),
        tint: uniform("tint"),
        unlit: uniform("unlit"),
        use_blend: uniform("use_blend"),
        sun_direction: uniform("sun_direction"),
        sun_color: uniform("sun_color"),
    })
}

unsafe fn upload_texture(gl: &glow::Context, image: &Image, anisotropy: f32, repeat: bool) -> glow::Texture {
    let texture = gl.create_texture().expect("the driver is out of texture names");
    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    gl.tex_image_2d(
        glow::TEXTURE_2D,
        0,
        glow::RGBA8 as i32,
        image.width as i32,
        image.height as i32,
        0,
        glow::RGBA,
        glow::UNSIGNED_BYTE,
        glow::PixelUnpackData::Slice(Some(&image.rgba)),
    );
    gl.generate_mipmap(glow::TEXTURE_2D);
    let wrap = if repeat { glow::REPEAT } else { glow::CLAMP_TO_EDGE } as i32;
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
    if anisotropy > 1.0 {
        gl.tex_parameter_f32(glow::TEXTURE_2D, TEXTURE_MAX_ANISOTROPY, anisotropy);
    }
    texture
}

struct Targets {
    width: i32,
    height: i32,
    multisampled: glow::Framebuffer,
    color: glow::Renderbuffer,
    depth: glow::Renderbuffer,
    resolved: glow::Framebuffer,
    resolved_color: glow::Texture,
    /// Glowing actors as flat colour, for the outline pass.
    glow: glow::Framebuffer,
    glow_color: glow::Texture,
}

impl Targets {
    unsafe fn new(gl: &glow::Context, width: i32, height: i32) -> Self {
        let color = gl.create_renderbuffer().expect("no renderbuffer");
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(color));
        gl.renderbuffer_storage_multisample(glow::RENDERBUFFER, MSAA_SAMPLES, glow::RGBA8, width, height);
        let depth = gl.create_renderbuffer().expect("no renderbuffer");
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth));
        gl.renderbuffer_storage_multisample(
            glow::RENDERBUFFER,
            MSAA_SAMPLES,
            glow::DEPTH_COMPONENT32F,
            width,
            height,
        );
        let multisampled = gl.create_framebuffer().expect("no framebuffer");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(multisampled));
        gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::RENDERBUFFER, Some(color));
        gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(depth));

        let resolved_color = gl.create_texture().expect("no texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(resolved_color));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        let resolved = gl.create_framebuffer().expect("no framebuffer");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(resolved));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(resolved_color),
            0,
        );

        let glow_color = gl.create_texture().expect("no texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(glow_color));
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            width,
            height,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(None),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        let glow = gl.create_framebuffer().expect("no framebuffer");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(glow));
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(glow_color), 0);

        Self {
            width,
            height,
            multisampled,
            color,
            depth,
            resolved,
            resolved_color,
            glow,
            glow_color,
        }
    }

    unsafe fn destroy(&self, gl: &glow::Context) {
        gl.delete_framebuffer(self.multisampled);
        gl.delete_framebuffer(self.resolved);
        gl.delete_renderbuffer(self.color);
        gl.delete_renderbuffer(self.depth);
        gl.delete_texture(self.resolved_color);
        gl.delete_framebuffer(self.glow);
        gl.delete_texture(self.glow_color);
    }
}

struct ModelGpu {
    vao: glow::VertexArray,
    vertex_buffer: glow::Buffer,
    index_buffer: glow::Buffer,
    instance_buffer: glow::Buffer,
    parts: Vec<(usize, u32, u32)>,
    instance_count: i32,
}

struct MeshGpu {
    vao: glow::VertexArray,
    vertex_buffer: glow::Buffer,
    index_buffer: glow::Buffer,
    skins: Vec<Vec<super::characters::MeshPart>>,
    body_groups: Vec<(u32, u32)>,
}

unsafe fn mesh_vao(gl: &glow::Context, mesh: &MeshData) -> MeshGpu {
    let vao = gl.create_vertex_array().expect("no vertex array");
    gl.bind_vertex_array(Some(vao));
    let vertex_buffer = gl.create_buffer().expect("no buffer");
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&mesh.vertices), glow::STATIC_DRAW);
    let stride = std::mem::size_of::<SkinnedVertex>() as i32;
    for (location, size, offset) in [(0, 3, 0), (1, 3, 12), (2, 2, 24), (3, 3, 32), (4, 3, 44)] {
        gl.enable_vertex_attrib_array(location);
        gl.vertex_attrib_pointer_f32(location, size, glow::FLOAT, false, stride, offset);
    }
    let index_buffer = gl.create_buffer().expect("no buffer");
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
    gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytemuck::cast_slice(&mesh.indices), glow::STATIC_DRAW);
    gl.bind_vertex_array(None);
    MeshGpu {
        vao,
        vertex_buffer,
        index_buffer,
        skins: mesh.skins.clone(),
        body_groups: mesh.body_groups.clone(),
    }
}

struct ShapeGpu {
    model: ModelGpu,
    shape: Shape,
}

/// Lays out a vertex array for model drawing, with per-instance data at the end.
unsafe fn model_vao(gl: &glow::Context, vertices: &[ModelVertex], indices: &[u32]) -> ModelGpu {
    let vao = gl.create_vertex_array().expect("no vertex array");
    gl.bind_vertex_array(Some(vao));

    let vertex_buffer = gl.create_buffer().expect("no buffer");
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(vertices), glow::STATIC_DRAW);
    let stride = std::mem::size_of::<ModelVertex>() as i32;
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
    gl.enable_vertex_attrib_array(1);
    gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 12);
    gl.enable_vertex_attrib_array(2);
    gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, 24);
    gl.enable_vertex_attrib_array(15);
    gl.vertex_attrib_pointer_f32(15, 1, glow::FLOAT, false, stride, 32);

    let index_buffer = gl.create_buffer().expect("no buffer");
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
    gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, bytemuck::cast_slice(indices), glow::STATIC_DRAW);

    let instance_buffer = gl.create_buffer().expect("no buffer");
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(instance_buffer));
    let stride = (INSTANCE_FLOATS * 4) as i32;
    for column in 0..4 {
        let location = 3 + column;
        gl.enable_vertex_attrib_array(location);
        gl.vertex_attrib_pointer_f32(location, 4, glow::FLOAT, false, stride, (column * 16) as i32);
        gl.vertex_attrib_divisor(location, 1);
    }
    for side in 0..6 {
        let location = 7 + side;
        gl.enable_vertex_attrib_array(location);
        gl.vertex_attrib_pointer_f32(location, 3, glow::FLOAT, false, stride, (64 + side * 12) as i32);
        gl.vertex_attrib_divisor(location, 1);
    }
    gl.enable_vertex_attrib_array(13);
    gl.vertex_attrib_pointer_f32(13, 4, glow::FLOAT, false, stride, 64 + 72);
    gl.vertex_attrib_divisor(13, 1);
    gl.enable_vertex_attrib_array(14);
    gl.vertex_attrib_pointer_f32(14, 1, glow::FLOAT, false, stride, 64 + 72 + 16);
    gl.vertex_attrib_divisor(14, 1);

    gl.bind_vertex_array(None);
    ModelGpu {
        vao,
        vertex_buffer,
        index_buffer,
        instance_buffer,
        parts: Vec::new(),
        instance_count: 0,
    }
}

/// One instance's data. `baked` is where its baked colours start, or -1.
fn instance_floats(transform: &Mat4, cube: &LightCube, tint: [f32; 3], sun: f32, baked: f32, out: &mut Vec<f32>) {
    out.extend_from_slice(&transform.to_cols_array());
    for side in cube {
        out.extend_from_slice(side);
    }
    out.extend_from_slice(&tint);
    out.push(sun);
    out.push(baked);
}

struct SkyGpu {
    vao: glow::VertexArray,
    buffer: glow::Buffer,
    faces: [usize; 6],
}

fn sky_vertices() -> Vec<f32> {
    // Each face runs left to right and top to bottom as seen from inside.
    let faces: [[[f32; 3]; 4]; 6] = [
        [[1., 1., 1.], [1., -1., 1.], [1., -1., -1.], [1., 1., -1.]],
        [[-1., -1., 1.], [-1., 1., 1.], [-1., 1., -1.], [-1., -1., -1.]],
        [[-1., 1., 1.], [1., 1., 1.], [1., 1., -1.], [-1., 1., -1.]],
        [[1., -1., 1.], [-1., -1., 1.], [-1., -1., -1.], [1., -1., -1.]],
        [[-1., 1., 1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.]],
        [[1., 1., -1.], [1., -1., -1.], [-1., -1., -1.], [-1., 1., -1.]],
    ];
    let corners = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
    let mut out = Vec::new();
    for face in faces {
        for index in [0, 1, 2, 0, 2, 3] {
            out.extend_from_slice(&face[index]);
            out.extend_from_slice(&corners[index]);
        }
    }
    out
}

/// A box-shaped placeholder person, built around a standing height of 82 units.
fn figure_mesh() -> (Vec<ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    add_cylinder(&mut vertices, &mut indices, [0., 0., 0.], 15.0, 62.0, 12);
    add_sphere(&mut vertices, &mut indices, [0., 0., 71.], 10.5, 12, 8);
    add_box(&mut vertices, &mut indices, [9., -3., 64.], [11., 6., 6.]);
    (vertices, indices)
}

fn add_box(vertices: &mut Vec<ModelVertex>, indices: &mut Vec<u32>, center: [f32; 3], half: [f32; 3]) {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
        ([-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]),
        ([0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]),
        ([0., -1., 0.], [1., 0., 0.], [0., 0., 1.]),
        ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
        ([0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]),
    ];
    for (normal, u, v) in faces {
        let base = vertices.len() as u32;
        for (a, b) in [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)] {
            let corner = |axis: usize| {
                center[axis] + (normal[axis] + u[axis] * a + v[axis] * b) * half[axis]
            };
            vertices.push(ModelVertex {
                position: [corner(0), corner(1), corner(2)],
                normal,
                uv: [0.5, 0.5],
                light: -1.0,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn add_cylinder(
    vertices: &mut Vec<ModelVertex>,
    indices: &mut Vec<u32>,
    base: [f32; 3],
    radius: f32,
    height: f32,
    segments: u32,
) {
    let start = vertices.len() as u32;
    for step in 0..=segments {
        let angle = step as f32 / segments as f32 * std::f32::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        for z in [0.0, height] {
            vertices.push(ModelVertex {
                position: [base[0] + cos * radius, base[1] + sin * radius, base[2] + z],
                normal: [cos, sin, 0.0],
                uv: [0.5, 0.5],
                light: -1.0,
            });
        }
    }
    for step in 0..segments {
        let a = start + step * 2;
        indices.extend_from_slice(&[a, a + 2, a + 1, a + 1, a + 2, a + 3]);
    }
    let top = vertices.len() as u32;
    vertices.push(ModelVertex {
        position: [base[0], base[1], base[2] + height],
        normal: [0., 0., 1.],
        uv: [0.5, 0.5],
        light: -1.0,
    });
    for step in 0..segments {
        let a = start + step * 2 + 1;
        indices.extend_from_slice(&[top, a, a + 2]);
    }
}

fn add_sphere(
    vertices: &mut Vec<ModelVertex>,
    indices: &mut Vec<u32>,
    center: [f32; 3],
    radius: f32,
    segments: u32,
    rings: u32,
) {
    let start = vertices.len() as u32;
    for ring in 0..=rings {
        let phi = ring as f32 / rings as f32 * std::f32::consts::PI;
        for step in 0..=segments {
            let theta = step as f32 / segments as f32 * std::f32::consts::TAU;
            let normal = [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()];
            vertices.push(ModelVertex {
                position: [
                    center[0] + normal[0] * radius,
                    center[1] + normal[1] * radius,
                    center[2] + normal[2] * radius,
                ],
                normal,
                uv: [0.5, 0.5],
                light: -1.0,
            });
        }
    }
    let row = segments + 1;
    for ring in 0..rings {
        for step in 0..segments {
            let a = start + ring * row + step;
            indices.extend_from_slice(&[a, a + row, a + 1, a + 1, a + row, a + row + 1]);
        }
    }
}

fn shape_mesh(shape: Shape) -> (Vec<ModelVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    match shape {
        Shape::Figure => return figure_mesh(),
        Shape::Rocket => add_box(&mut vertices, &mut indices, [0., 0., 0.], [11., 2.5, 2.5]),
        Shape::Ball => add_sphere(&mut vertices, &mut indices, [0., 0., 0.], 4.5, 10, 6),
        Shape::Box => add_box(&mut vertices, &mut indices, [0., 0., 0.], [1., 1., 1.]),
    }
    (vertices, indices)
}

pub struct Renderer {
    world: Program,
    model: Program,
    sky_program: Program,
    skinned: Program,
    trace_program: Program,
    glow_mask: Program,
    glow_mask_bones: Option<glow::UniformLocation>,
    glow_program: Program,
    /// Bound for draws that make their own vertices.
    empty_vao: glow::VertexArray,
    trace_vao: glow::VertexArray,
    trace_buffer: glow::Buffer,
    skinned_bones: Option<glow::UniformLocation>,
    skinned_cube: Option<glow::UniformLocation>,
    skinned_sun: Option<glow::UniformLocation>,
    meshes: Vec<MeshGpu>,
    world_vao: glow::VertexArray,
    world_buffers: [glow::Buffer; 2],
    batches: Vec<(usize, u32, u32, bool)>,
    materials: Vec<Material>,
    textures: Vec<glow::Texture>,
    white: glow::Texture,
    lightmap: glow::Texture,
    /// Baked vertex lighting for static props.
    prop_light: glow::Texture,
    models: Vec<ModelGpu>,
    /// Props inside the miniature backdrop, drawn in their own pass.
    sky_models: Vec<ModelGpu>,
    shapes: Vec<ShapeGpu>,
    sky: Option<SkyGpu>,
    sun: Sun,
    targets: Option<Targets>,
}

impl Renderer {
    /// Uploads a loaded map. Must run with the window's GL context current.
    pub fn new(gl: &glow::Context, map: &MapData) -> Result<Self, String> {
        unsafe {
            let world = compile(gl, WORLD_VERTEX, WORLD_FRAGMENT)?;
            let model = compile(gl, MODEL_VERTEX, MODEL_FRAGMENT)?;
            let sky_program = compile(gl, SKY_VERTEX, SKY_FRAGMENT)?;
            let skinned = compile(gl, SKINNED_VERTEX, MODEL_FRAGMENT)?;
            let trace_program = compile(gl, TRACE_VERTEX, TRACE_FRAGMENT)?;
            let glow_mask = compile(gl, SKINNED_VERTEX, GLOW_MASK_FRAGMENT)?;
            let glow_mask_bones = gl.get_uniform_location(glow_mask.program, "bones");
            let glow_program = compile(gl, SCREEN_VERTEX, GLOW_FRAGMENT)?;
            let empty_vao = gl.create_vertex_array()?;
            let trace_vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(trace_vao));
            let trace_buffer = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(trace_buffer));
            let trace_stride = (TRACE_FLOATS * 4) as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, trace_stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, trace_stride, 12);
            gl.bind_vertex_array(None);
            let skinned_bones = gl.get_uniform_location(skinned.program, "bones");
            let skinned_cube = gl.get_uniform_location(skinned.program, "cube");
            let skinned_sun = gl.get_uniform_location(skinned.program, "sun_visible");
            let meshes = map.meshes.iter().map(|mesh| mesh_vao(gl, mesh)).collect();

            let anisotropy = if gl.supported_extensions().contains("GL_EXT_texture_filter_anisotropic") {
                gl.get_parameter_f32(MAX_ANISOTROPY_LIMIT).min(8.0)
            } else {
                1.0
            };

            let textures: Vec<_> = map
                .textures
                .iter()
                .map(|image| upload_texture(gl, image, anisotropy, true))
                .collect();
            let white = upload_texture(
                gl,
                &Image {
                    width: 1,
                    height: 1,
                    rgba: vec![255; 4],
                },
                1.0,
                true,
            );

            let lightmap = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(lightmap));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB16F as i32,
                map.lightmap_width as i32,
                map.lightmap_height as i32,
                0,
                glow::RGB,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&map.lightmap))),
            );
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
            }

            let prop_light = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(prop_light));
            let mut colors = map.prop_light.clone();
            if colors.is_empty() {
                colors.extend_from_slice(&[255; 4]);
            }
            let texels = colors.len() / 4;
            let (light_width, light_height) = if texels <= PROP_LIGHT_WIDTH {
                (texels, 1)
            } else {
                (PROP_LIGHT_WIDTH, texels.div_ceil(PROP_LIGHT_WIDTH))
            };
            colors.resize(light_width * light_height * 4, 255);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                light_width as i32,
                light_height as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&colors)),
            );
            for (name, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::NEAREST),
                (glow::TEXTURE_MAG_FILTER, glow::NEAREST),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, name, value as i32);
            }

            let world_vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(world_vao));
            let vertex_buffer = gl.create_buffer()?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&map.world_vertices),
                glow::STATIC_DRAW,
            );
            let stride = std::mem::size_of::<WorldVertex>() as i32;
            for (location, size, offset) in [(0, 3, 0), (1, 2, 12), (2, 2, 20), (3, 1, 28)] {
                gl.enable_vertex_attrib_array(location);
                gl.vertex_attrib_pointer_f32(location, size, glow::FLOAT, false, stride, offset);
            }
            let index_buffer = gl.create_buffer()?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(index_buffer));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&map.world_indices),
                glow::STATIC_DRAW,
            );
            gl.bind_vertex_array(None);

            let mut models: Vec<ModelGpu> = map
                .models
                .iter()
                .map(|data| {
                    let mut gpu = model_vao(gl, &data.vertices, &data.indices);
                    gpu.parts = data
                        .parts
                        .iter()
                        .map(|part| (part.material, part.first_index, part.index_count))
                        .collect();
                    gpu
                })
                .collect();

            let mut per_model: Vec<Vec<f32>> = vec![Vec::new(); models.len()];
            let mut per_sky_model: Vec<Vec<f32>> = vec![Vec::new(); models.len()];
            for prop in &map.props {
                let target = if prop.sky { &mut per_sky_model } else { &mut per_model };
                instance_floats(
                    &prop.transform,
                    &prop.lighting.cube,
                    [1.0; 3],
                    prop.lighting.sun,
                    prop.baked.map_or(-1.0, |start| start as f32),
                    &mut target[prop.model],
                );
            }
            let mut sky_models = Vec::new();
            for (index, floats) in per_sky_model.iter().enumerate() {
                if floats.is_empty() {
                    continue;
                }
                let data = &map.models[index];
                let mut gpu = model_vao(gl, &data.vertices, &data.indices);
                gpu.parts = models[index].parts.clone();
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu.instance_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(floats), glow::STATIC_DRAW);
                gpu.instance_count = (floats.len() / INSTANCE_FLOATS) as i32;
                sky_models.push(gpu);
            }
            for (gpu, floats) in models.iter_mut().zip(&per_model) {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu.instance_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(floats), glow::STATIC_DRAW);
                gpu.instance_count = (floats.len() / INSTANCE_FLOATS) as i32;
            }

            let shapes = [Shape::Figure, Shape::Rocket, Shape::Ball, Shape::Box]
                .into_iter()
                .map(|shape| {
                    let (vertices, indices) = shape_mesh(shape);
                    let mut gpu = model_vao(gl, &vertices, &indices);
                    gpu.parts = vec![(usize::MAX, 0, indices.len() as u32)];
                    ShapeGpu { model: gpu, shape }
                })
                .collect();

            let sky = match map.sky {
                Some(faces) => {
                    let vao = gl.create_vertex_array()?;
                    gl.bind_vertex_array(Some(vao));
                    let buffer = gl.create_buffer()?;
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
                    gl.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&sky_vertices()),
                        glow::STATIC_DRAW,
                    );
                    gl.enable_vertex_attrib_array(0);
                    gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 20, 0);
                    gl.enable_vertex_attrib_array(1);
                    gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 20, 12);
                    gl.bind_vertex_array(None);
                    for face in faces {
                        gl.bind_texture(glow::TEXTURE_2D, Some(textures[face]));
                        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                    }
                    Some(SkyGpu { vao, buffer, faces })
                }
                None => None,
            };

            Ok(Self {
                world,
                model,
                sky_program,
                skinned,
                trace_program,
                glow_mask,
                glow_mask_bones,
                glow_program,
                empty_vao,
                trace_vao,
                trace_buffer,
                skinned_bones,
                skinned_cube,
                skinned_sun,
                meshes,
                world_vao,
                world_buffers: [vertex_buffer, index_buffer],
                batches: map
                    .batches
                    .iter()
                    .map(|batch| (batch.material, batch.first_index, batch.index_count, batch.sky))
                    .collect(),
                materials: map.materials.clone(),
                textures,
                white,
                lightmap,
                prop_light,
                models,
                sky_models,
                shapes,
                sky,
                sun: map.lighting.sun(),
                targets: None,
            })
        }
    }

    fn texture(&self, id: Option<usize>) -> glow::Texture {
        id.and_then(|id| self.textures.get(id).copied()).unwrap_or(self.white)
    }

    unsafe fn bind_on(&self, gl: &glow::Context, unit: u32, texture: glow::Texture) {
        gl.active_texture(glow::TEXTURE0 + unit);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
    }

    unsafe fn set_blending(&self, gl: &glow::Context, material: &Material) {
        if material.translucent {
            gl.enable(glow::BLEND);
            gl.depth_mask(false);
            // The image's own alpha always stays solid, so only colour is blended.
            if material.additive {
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE, glow::ZERO, glow::ONE);
            } else {
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE);
            }
        } else {
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
        }
    }

    unsafe fn draw_sky(&self, gl: &glow::Context, frame: &Frame) {
        let Some(sky) = &self.sky else { return };
        let rotation_only = Mat4::from_mat3(glam::Mat3::from_mat4(frame.view));
        let matrix = frame.projection * rotation_only;
        gl.use_program(Some(self.sky_program.program));
        gl.uniform_matrix_4_f32_slice(
            self.sky_program.view_projection.as_ref(),
            false,
            &matrix.to_cols_array(),
        );
        gl.disable(glow::DEPTH_TEST);
        gl.depth_mask(false);
        gl.bind_vertex_array(Some(sky.vao));
        for (face, texture) in sky.faces.iter().enumerate() {
            self.bind_on(gl, 0, self.textures[*texture]);
            gl.draw_arrays(glow::TRIANGLES, (face * 6) as i32, 6);
        }
        gl.enable(glow::DEPTH_TEST);
        gl.depth_mask(true);
    }

    unsafe fn draw_world(&self, gl: &glow::Context, view_projection: &Mat4, sky: bool) {
        let p = &self.world;
        gl.use_program(Some(p.program));
        gl.uniform_matrix_4_f32_slice(p.view_projection.as_ref(), false, &view_projection.to_cols_array());
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "base").as_ref(), 0);
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "base2").as_ref(), 1);
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "lightmap").as_ref(), 2);
        self.bind_on(gl, 2, self.lightmap);
        gl.bind_vertex_array(Some(self.world_vao));

        for &(material, first, count, in_sky) in &self.batches {
            if in_sky != sky {
                continue;
            }
            let m = &self.materials[material];
            self.bind_on(gl, 0, self.texture(m.base));
            self.bind_on(gl, 1, self.texture(m.base2.or(m.base)));
            gl.uniform_1_i32(p.use_blend.as_ref(), i32::from(m.base2.is_some()));
            gl.uniform_1_i32(p.unlit.as_ref(), i32::from(m.unlit));
            gl.uniform_1_f32(p.alpha_test.as_ref(), m.alpha_test.unwrap_or(0.0));
            gl.uniform_1_f32(p.opacity.as_ref(), m.opacity);
            gl.uniform_3_f32_slice(p.tint.as_ref(), &m.tint);
            self.set_blending(gl, m);
            gl.draw_elements(glow::TRIANGLES, count as i32, glow::UNSIGNED_INT, (first * 4) as i32);
        }
        gl.disable(glow::BLEND);
        gl.depth_mask(true);
    }

    unsafe fn prepare_models(&self, gl: &glow::Context, view_projection: &Mat4) {
        let p = &self.model;
        gl.use_program(Some(p.program));
        gl.uniform_matrix_4_f32_slice(p.view_projection.as_ref(), false, &view_projection.to_cols_array());
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "base").as_ref(), 0);
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "baked").as_ref(), 3);
        self.bind_on(gl, 3, self.prop_light);
        gl.uniform_3_f32_slice(p.sun_direction.as_ref(), &self.sun.direction);
        gl.uniform_3_f32_slice(p.sun_color.as_ref(), &self.sun.color);
    }

    unsafe fn draw_model_parts(&self, gl: &glow::Context, model: &ModelGpu, instances: i32, translucent: bool) {
        let p = &self.model;
        gl.bind_vertex_array(Some(model.vao));
        for &(material, first, count) in &model.parts {
            let plain = Material {
                opacity: 1.0,
                tint: [1.0; 3],
                ..Material::default()
            };
            let m = self.materials.get(material).unwrap_or(&plain);
            if m.translucent != translucent {
                continue;
            }
            self.bind_on(gl, 0, self.texture(m.base));
            gl.uniform_1_i32(p.unlit.as_ref(), i32::from(m.unlit));
            gl.uniform_1_f32(p.alpha_test.as_ref(), m.alpha_test.unwrap_or(0.0));
            gl.uniform_1_f32(p.opacity.as_ref(), m.opacity);
            gl.uniform_3_f32_slice(p.tint.as_ref(), &m.tint);
            self.set_blending(gl, m);
            gl.draw_elements_instanced(
                glow::TRIANGLES,
                count as i32,
                glow::UNSIGNED_INT,
                (first * 4) as i32,
                instances,
            );
        }
    }

    /// Draws glowing actors flat into the glow image, ignoring walls, then
    /// outlines what's there over the scene.
    unsafe fn draw_glow(&self, gl: &glow::Context, targets: &Targets, actors: &[Actor], view_projection: &Mat4) {
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(targets.glow));
        gl.color_mask(true, true, true, true);
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(glow::COLOR_BUFFER_BIT);
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::BLEND);

        let p = &self.glow_mask;
        gl.use_program(Some(p.program));
        gl.uniform_matrix_4_f32_slice(p.view_projection.as_ref(), false, &view_projection.to_cols_array());
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "base").as_ref(), 0);
        let glow_color = gl.get_uniform_location(p.program, "glow_color");
        let mut floats: Vec<f32> = Vec::with_capacity(MAX_BONES * 12);
        for actor in actors {
            let Some(color) = actor.glow else { continue };
            let Some(mesh) = self.meshes.get(actor.mesh) else { continue };
            let Some(parts) = mesh.skins.get(actor.skin).or(mesh.skins.first()) else { continue };
            floats.clear();
            for matrix in actor.bones.iter().take(MAX_BONES) {
                floats.extend_from_slice(matrix);
            }
            gl.uniform_4_f32_slice(self.glow_mask_bones.as_ref(), &floats);
            gl.uniform_3_f32_slice(glow_color.as_ref(), &color);
            gl.bind_vertex_array(Some(mesh.vao));
            for part in parts {
                let shown = mesh.body_groups.get(usize::from(part.group)).is_none_or(|&(place, options)| {
                    (actor.body / place.max(1)) % options.max(1) == u32::from(part.option)
                });
                if !shown {
                    continue;
                }
                let m = self.materials.get(part.material);
                self.bind_on(gl, 0, self.texture(m.and_then(|m| m.base)));
                let alpha_test = m.map_or(0.0, |m| m.alpha_test.unwrap_or(0.0).max(if m.translucent { 0.3 } else { 0.0 }));
                gl.uniform_1_f32(p.alpha_test.as_ref(), alpha_test);
                gl.draw_elements(glow::TRIANGLES, part.index_count as i32, glow::UNSIGNED_INT, (part.first_index * 4) as i32);
            }
        }

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(targets.multisampled));
        gl.color_mask(true, true, true, false);
        gl.use_program(Some(self.glow_program.program));
        gl.uniform_1_i32(gl.get_uniform_location(self.glow_program.program, "mask").as_ref(), 0);
        // About two pixels at 720p, wider on bigger screens.
        let radius = (targets.height / 360).clamp(2, 6);
        gl.uniform_1_i32(gl.get_uniform_location(self.glow_program.program, "radius").as_ref(), radius);
        self.bind_on(gl, 0, targets.glow_color);
        gl.enable(glow::BLEND);
        gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE);
        gl.depth_mask(false);
        gl.bind_vertex_array(Some(self.empty_vao));
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
        gl.disable(glow::BLEND);
        gl.depth_mask(true);
        gl.enable(glow::DEPTH_TEST);
    }

    unsafe fn draw_actors(&self, gl: &glow::Context, actors: &[Actor], view_projection: &Mat4) {
        if actors.is_empty() {
            return;
        }
        let p = &self.skinned;
        gl.use_program(Some(p.program));
        gl.uniform_matrix_4_f32_slice(p.view_projection.as_ref(), false, &view_projection.to_cols_array());
        gl.uniform_1_i32(gl.get_uniform_location(p.program, "base").as_ref(), 0);
        gl.uniform_3_f32_slice(p.sun_direction.as_ref(), &self.sun.direction);
        gl.uniform_3_f32_slice(p.sun_color.as_ref(), &self.sun.color);
        gl.disable(glow::BLEND);
        gl.depth_mask(true);

        let mut floats: Vec<f32> = Vec::with_capacity(MAX_BONES * 12);
        for actor in actors {
            let Some(mesh) = self.meshes.get(actor.mesh) else { continue };
            let Some(parts) = mesh.skins.get(actor.skin).or(mesh.skins.first()) else { continue };
            floats.clear();
            for matrix in actor.bones.iter().take(MAX_BONES) {
                floats.extend_from_slice(matrix);
            }
            gl.uniform_4_f32_slice(self.skinned_bones.as_ref(), &floats);
            let cube: Vec<f32> = actor.cube.iter().flatten().copied().collect();
            gl.uniform_3_f32_slice(self.skinned_cube.as_ref(), &cube);
            gl.uniform_1_f32(self.skinned_sun.as_ref(), actor.sun);
            gl.bind_vertex_array(Some(mesh.vao));
            for part in parts {
                let shown = mesh.body_groups.get(usize::from(part.group)).is_none_or(|&(place, options)| {
                    (actor.body / place.max(1)) % options.max(1) == u32::from(part.option)
                });
                if !shown {
                    continue;
                }
                let (material, first, count) = (part.material, part.first_index, part.index_count);
                let plain = Material {
                    opacity: 1.0,
                    tint: [1.0; 3],
                    ..Material::default()
                };
                let m = self.materials.get(material).unwrap_or(&plain);
                self.bind_on(gl, 0, self.texture(m.base));
                gl.uniform_1_i32(p.unlit.as_ref(), i32::from(m.unlit));
                gl.uniform_1_f32(p.alpha_test.as_ref(), m.alpha_test.unwrap_or(0.0).max(if m.translucent { 0.3 } else { 0.0 }));
                gl.uniform_1_f32(p.opacity.as_ref(), 1.0);
                gl.uniform_3_f32_slice(p.tint.as_ref(), &m.tint);
                gl.draw_elements(glow::TRIANGLES, count as i32, glow::UNSIGNED_INT, (first * 4) as i32);
            }
        }
    }

    unsafe fn draw_shapes(&self, gl: &glow::Context, frame: &Frame) {
        for gpu in &self.shapes {
            let mut floats = Vec::new();
            let mut count = 0;
            for instance in frame.shapes.iter().filter(|i| i.shape == gpu.shape) {
                instance_floats(&instance.transform, &instance.cube, instance.tint, instance.sun, -1.0, &mut floats);
                count += 1;
            }
            if count == 0 {
                continue;
            }
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu.model.instance_buffer));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&floats), glow::STREAM_DRAW);
            self.draw_model_parts(gl, &gpu.model, count, false);
        }
    }

    /// Renders a frame and copies it into the given window rectangle, measured
    /// in pixels from the bottom left.
    pub fn paint(
        &mut self,
        gl: &glow::Context,
        frame: &Frame,
        target: Option<glow::Framebuffer>,
        rect: [i32; 4],
    ) {
        let [left, bottom, width, height] = rect;
        if width <= 0 || height <= 0 {
            return;
        }
        unsafe {
            let stale = self.targets.as_ref().is_none_or(|t| t.width != width || t.height != height);
            if stale {
                if let Some(old) = self.targets.take() {
                    old.destroy(gl);
                }
                self.targets = Some(Targets::new(gl, width, height));
            }
            let targets = self.targets.as_ref().expect("targets were just created");

            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(targets.multisampled));
            gl.viewport(0, 0, width, height);
            gl.color_mask(true, true, true, true);
            gl.depth_mask(true);
            gl.clear_color(0.33, 0.42, 0.5, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            // Keep every pixel fully solid. See-through textures still blend their colour.
            gl.color_mask(true, true, true, false);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.disable(glow::BLEND);

            let view_projection = frame.projection * frame.view;
            self.draw_sky(gl, frame);

            // The miniature backdrop goes behind everything, seen from its own camera.
            if let Some(sky_view) = frame.sky_view {
                let sky_projection = frame.projection * sky_view;
                self.draw_world(gl, &sky_projection, true);
                self.prepare_models(gl, &sky_projection);
                for model in &self.sky_models {
                    self.draw_model_parts(gl, model, model.instance_count, false);
                }
                for model in &self.sky_models {
                    self.draw_model_parts(gl, model, model.instance_count, true);
                }
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
                gl.clear(glow::DEPTH_BUFFER_BIT);
            }

            self.draw_world(gl, &view_projection, false);

            self.prepare_models(gl, &view_projection);
            for model in &self.models {
                self.draw_model_parts(gl, model, model.instance_count, false);
            }
            self.draw_shapes(gl, frame);
            self.draw_actors(gl, &frame.actors, &view_projection);
            self.prepare_models(gl, &view_projection);
            for model in &self.models {
                self.draw_model_parts(gl, model, model.instance_count, true);
            }

            // Bullet traces glow over whatever is behind them, but walls still hide them.
            if !frame.traces.is_empty() {
                gl.use_program(Some(self.trace_program.program));
                gl.uniform_matrix_4_f32_slice(
                    self.trace_program.view_projection.as_ref(),
                    false,
                    &view_projection.to_cols_array(),
                );
                gl.uniform_1_f32(self.trace_program.opacity.as_ref(), 1.0);
                gl.enable(glow::BLEND);
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE, glow::ZERO, glow::ONE);
                gl.depth_mask(false);
                gl.bind_vertex_array(Some(self.trace_vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.trace_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&frame.traces), glow::STREAM_DRAW);
                gl.draw_arrays(glow::TRIANGLES, 0, (frame.traces.len() / TRACE_FLOATS) as i32);
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
            }

            // Overlay lines: faint where something hides them, full strength where they're in view.
            if !frame.overlay.is_empty() {
                let p = &self.trace_program;
                gl.use_program(Some(p.program));
                gl.uniform_matrix_4_f32_slice(p.view_projection.as_ref(), false, &view_projection.to_cols_array());
                gl.enable(glow::BLEND);
                gl.blend_func_separate(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA, glow::ZERO, glow::ONE);
                gl.depth_mask(false);
                gl.bind_vertex_array(Some(self.trace_vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.trace_buffer));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&frame.overlay), glow::STREAM_DRAW);
                let count = (frame.overlay.len() / TRACE_FLOATS) as i32;
                gl.disable(glow::DEPTH_TEST);
                gl.uniform_1_f32(p.opacity.as_ref(), OVERLAY_HIDDEN_OPACITY);
                gl.draw_arrays(glow::TRIANGLES, 0, count);
                gl.enable(glow::DEPTH_TEST);
                gl.uniform_1_f32(p.opacity.as_ref(), 1.0);
                gl.draw_arrays(glow::TRIANGLES, 0, count);
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
            }

            // Outlines that show through walls.
            if frame.actors.iter().any(|actor| actor.glow.is_some()) {
                self.draw_glow(gl, targets, &frame.actors, &view_projection);
            }

            // The first-person view model sits on top of everything else.
            if !frame.view_actors.is_empty() {
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
                gl.clear(glow::DEPTH_BUFFER_BIT);
                let view_model = frame.view_projection * frame.view;
                self.draw_actors(gl, &frame.view_actors, &view_model);
            }

            gl.disable(glow::BLEND);
            gl.depth_mask(true);
            gl.color_mask(true, true, true, true);
            gl.disable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
            gl.active_texture(glow::TEXTURE0);

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(targets.multisampled));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(targets.resolved));
            gl.blit_framebuffer(0, 0, width, height, 0, 0, width, height, glow::COLOR_BUFFER_BIT, glow::NEAREST);

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(targets.resolved));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, target);
            gl.blit_framebuffer(
                0,
                0,
                width,
                height,
                left,
                bottom,
                left + width,
                bottom + height,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.bind_framebuffer(glow::FRAMEBUFFER, target);
        }
    }

    /// Frees everything on the GPU. Must run with the GL context current.
    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some(targets) = self.targets.take() {
                targets.destroy(gl);
            }
            for texture in self.textures.drain(..) {
                gl.delete_texture(texture);
            }
            gl.delete_texture(self.white);
            gl.delete_texture(self.lightmap);
            gl.delete_texture(self.prop_light);
            gl.delete_vertex_array(self.world_vao);
            for buffer in self.world_buffers {
                gl.delete_buffer(buffer);
            }
            let shape_models = self.shapes.drain(..).map(|shape| shape.model);
            let sky_models = self.sky_models.drain(..).collect::<Vec<_>>();
            for model in self.models.drain(..).chain(sky_models).chain(shape_models) {
                gl.delete_vertex_array(model.vao);
                gl.delete_buffer(model.vertex_buffer);
                gl.delete_buffer(model.index_buffer);
                gl.delete_buffer(model.instance_buffer);
            }
            for mesh in self.meshes.drain(..) {
                gl.delete_vertex_array(mesh.vao);
                gl.delete_buffer(mesh.vertex_buffer);
                gl.delete_buffer(mesh.index_buffer);
            }
            if let Some(sky) = self.sky.take() {
                gl.delete_vertex_array(sky.vao);
                gl.delete_buffer(sky.buffer);
            }
            gl.delete_vertex_array(self.trace_vao);
            gl.delete_buffer(self.trace_buffer);
            gl.delete_vertex_array(self.empty_vao);
            for program in [
                &self.world,
                &self.model,
                &self.sky_program,
                &self.skinned,
                &self.trace_program,
                &self.glow_mask,
                &self.glow_program,
            ] {
                gl.delete_program(program.program);
            }
        }
    }
}
