// Source Engine BSP v20 Constants and Lump Structures

pub const LUMP_ENTITIES: usize = 0;
pub const LUMP_PLANES: usize = 1;
pub const LUMP_TEXDATA: usize = 2;
pub const LUMP_NODES: usize = 5;
pub const LUMP_TEXINFO: usize = 6;
pub const LUMP_LEAFS: usize = 10;
pub const LUMP_LEAFBRUSHES: usize = 17;
pub const LUMP_BRUSHES: usize = 18;
pub const LUMP_BRUSHSIDES: usize = 19;
pub const LUMP_TEXDATA_STRING_TABLE: usize = 43;
pub const LUMP_TEXDATA_STRING_DATA: usize = 44;

// Content Flags (from Source SDK public/bspflags.h)
pub const CONTENTS_SOLID: i32 = 0x1;
pub const CONTENTS_WINDOW: i32 = 0x2;
pub const CONTENTS_AUX: i32 = 0x4;
pub const CONTENTS_GRATE: i32 = 0x8;
pub const CONTENTS_SLIME: i32 = 0x10;
pub const CONTENTS_WATER: i32 = 0x20;
pub const CONTENTS_BLOCKLOS: i32 = 0x40;
pub const CONTENTS_OPAQUE: i32 = 0x80;
pub const CONTENTS_TESTFOGVOLUME: i32 = 0x100;
pub const CONTENTS_TEAM4: i32 = 0x200;
pub const CONTENTS_TEAM3: i32 = 0x400;
pub const CONTENTS_TEAM1: i32 = 0x800;
pub const CONTENTS_TEAM2: i32 = 0x1000;
pub const CONTENTS_IGNORE_NODRAW_OPAQUE: i32 = 0x2000;
pub const CONTENTS_MOVEABLE: i32 = 0x4000;
pub const CONTENTS_AREAPORTAL: i32 = 0x8000;
pub const CONTENTS_PLAYERCLIP: i32 = 0x10000;
pub const CONTENTS_MONSTERCLIP: i32 = 0x20000;

// Surface Flags
pub const SURF_LIGHT: i32 = 0x0001;
pub const SURF_SKY2D: i32 = 0x0002;
pub const SURF_SKY: i32 = 0x0004;
pub const SURF_WARP: i32 = 0x0008;
pub const SURF_TRANS: i32 = 0x0010;
pub const SURF_NOPORTAL: i32 = 0x0020;
pub const SURF_TRIGGER: i32 = 0x0040;
pub const SURF_NODRAW: i32 = 0x0080;
pub const SURF_HINT: i32 = 0x0100;
pub const SURF_SKIP: i32 = 0x0200;

#[derive(Clone, Copy, Debug, Default)]
pub struct Plane {
    pub normal: [f32; 3],
    pub dist: f32,
    pub plane_type: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Node {
    pub plane_num: i32,
    pub children: [i32; 2],
    pub mins: [i16; 3],
    pub maxs: [i16; 3],
    pub first_face: u16,
    pub num_faces: u16,
    pub area: i16,
    pub padding: i16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Leaf {
    pub contents: i32,
    pub cluster: i16,
    pub area_flags: i16,
    pub mins: [i16; 3],
    pub maxs: [i16; 3],
    pub first_leafface: u16,
    pub num_leaffaces: u16,
    pub first_leafbrush: u16,
    pub num_leafbrushes: u16,
    pub leaf_water_data_id: i16,
    pub padding: i16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Brush {
    pub first_side: i32,
    pub num_sides: i32,
    pub contents: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BrushSide {
    pub planenum: u16,
    pub texinfo: i16,
    pub dispinfo: i16,
    pub bevel: i16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TexInfo {
    pub flags: i32,
    pub texdata: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TexData {
    pub name_string_table_id: i32,
    pub width: i32,
    pub height: i32,
}
