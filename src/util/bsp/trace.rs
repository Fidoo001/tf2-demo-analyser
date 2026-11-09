use super::lumps::*;
use std::cell::RefCell;

// Stack entry for the iterative BSP traversal.
struct StackEntry {
    node_idx: i32,
    t_start: f32,
    t_end: f32,
}

/// Per-thread buffers reused by every trace, so a trace allocates nothing.
#[derive(Default)]
struct TraceScratch {
    stack: Vec<StackEntry>,
    // A brush is listed in every leaf it touches; brushes whose stamp equals the current trace's
    // stamp were already tested by this trace (testing one again cannot change the result).
    brush_stamps: Vec<u32>,
    stamp: u32,
}

thread_local! {
    static TRACE_SCRATCH: RefCell<TraceScratch> = RefCell::new(TraceScratch::default());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SightlineStatus {
    Clear,
    BlockedByOpaqueWall,
    BlockedByTransparentWindow,
    BlockedByGrate,
}

/// Result of an axis-aligned box sweep (see [`BspCollisionData::trace_hull`]).
#[derive(Clone, Copy, Debug)]
pub struct HullTrace {
    /// Fraction of the sweep completed before the first hit (1.0 = no hit).
    pub fraction: f32,
    /// Normal of the plane that was hit (zero when nothing was hit).
    pub normal: [f32; 3],
    pub start_solid: bool,
}

/// Brush contents that block player movement (MASK_PLAYERSOLID without monsters).
pub const MASK_PLAYERSOLID_BRUSH: i32 = CONTENTS_SOLID | CONTENTS_MOVEABLE | CONTENTS_WINDOW | CONTENTS_GRATE | CONTENTS_PLAYERCLIP;

#[derive(Clone, Debug, Default)]
pub struct BspCollisionData {
    pub planes: Vec<Plane>,
    pub nodes: Vec<Node>,
    pub leafs: Vec<Leaf>,
    pub leafbrushes: Vec<u16>,
    pub brushes: Vec<Brush>,
    pub brushsides: Vec<BrushSide>,
    pub texinfo: Vec<TexInfo>,
    pub texdata: Vec<TexData>,
    pub material_names: Vec<String>,
}

impl BspCollisionData {
    fn check_side_transparency(&self, side_idx: usize) -> Option<SightlineStatus> {
        if side_idx < self.brushsides.len() {
            let side = &self.brushsides[side_idx];
            let texinfo_idx = side.texinfo as usize;
            if texinfo_idx < self.texinfo.len() {
                let ti = &self.texinfo[texinfo_idx];
                if (ti.flags & SURF_TRANS) != 0 {
                    return Some(SightlineStatus::BlockedByTransparentWindow);
                }

                let texdata_idx = ti.texdata as usize;
                if texdata_idx < self.texdata.len() {
                    let td = &self.texdata[texdata_idx];
                    let str_id = td.name_string_table_id as usize;
                    if str_id < self.material_names.len() {
                        let mat = self.material_names[str_id].to_ascii_lowercase();
                        if mat.contains("glass") || mat.contains("window") {
                            return Some(SightlineStatus::BlockedByTransparentWindow);
                        }
                        if mat.contains("grate") || mat.contains("fence") {
                            return Some(SightlineStatus::BlockedByGrate);
                        }
                    }
                }
            }
        }
        None
    }

    /// Determines whether a brush represents a transparent window/glass, a grate, or an opaque solid.
    fn classify_brush(&self, brush: &Brush, hit_side_idx: Option<usize>) -> SightlineStatus {
        // 1. Check for wire fence / grate
        if (brush.contents & CONTENTS_GRATE) != 0 {
            return SightlineStatus::BlockedByGrate;
        }

        // 2. Check for glass window content flag
        if (brush.contents & CONTENTS_WINDOW) != 0 {
            return SightlineStatus::BlockedByTransparentWindow;
        }

        // 3. Inspect surface flags and texture name of the hit brushside
        if let Some(side_idx) = hit_side_idx {
            if let Some(status) = self.check_side_transparency(side_idx) {
                return status;
            }
        }

        // 4. If hit side was nodraw or didn't match, check all brush sides
        let first_side = brush.first_side as usize;
        let num_sides = brush.num_sides as usize;
        for s in 0..num_sides {
            let side_idx = first_side + s;
            if let Some(status) = self.check_side_transparency(side_idx) {
                return status;
            }
        }

        // 5. Solid or opaque
        if (brush.contents & (CONTENTS_SOLID | CONTENTS_OPAQUE | CONTENTS_PLAYERCLIP)) != 0 {
            return SightlineStatus::BlockedByOpaqueWall;
        }

        SightlineStatus::Clear
    }

    /// Traces a ray from start to end through the BSP tree, checking for wall occlusion.
    pub fn trace_sightline(&self, start: [f32; 3], end: [f32; 3]) -> SightlineStatus {
        if self.nodes.is_empty() {
            return SightlineStatus::Clear;
        }

        TRACE_SCRATCH.with(|scratch| self.trace_sightline_with(start, end, &mut scratch.borrow_mut()))
    }

    fn trace_sightline_with(&self, start: [f32; 3], end: [f32; 3], scratch: &mut TraceScratch) -> SightlineStatus {
        scratch.stamp = scratch.stamp.wrapping_add(1);
        if scratch.stamp == 0 {
            scratch.brush_stamps.fill(0);
            scratch.stamp = 1;
        }
        if scratch.brush_stamps.len() < self.brushes.len() {
            scratch.brush_stamps.resize(self.brushes.len(), 0);
        }
        let stamp = scratch.stamp;
        let brush_stamps = &mut scratch.brush_stamps;

        // Stack-based iterative traversal to avoid recursion overhead
        let stack = &mut scratch.stack;
        stack.clear();
        stack.push(StackEntry {
            node_idx: 0,
            t_start: 0.0,
            t_end: 1.0,
        });

        let mut hit_window = false;
        let mut hit_grate = false;

        while let Some(entry) = stack.pop() {
            if entry.node_idx < 0 {
                // Leaf reached
                let leaf_idx = (-entry.node_idx - 1) as usize;
                if leaf_idx < self.leafs.len() {
                    let leaf = &self.leafs[leaf_idx];

                    // Check all brushes in this leaf
                    let start_b = leaf.first_leafbrush as usize;
                    let end_b = start_b + leaf.num_leafbrushes as usize;

                    for lb_idx in start_b..end_b {
                        if lb_idx >= self.leafbrushes.len() {
                            break;
                        }
                        let b_idx = self.leafbrushes[lb_idx] as usize;
                        if b_idx >= self.brushes.len() {
                            continue;
                        }
                        if brush_stamps[b_idx] == stamp {
                            continue;
                        }
                        brush_stamps[b_idx] = stamp;

                        let brush = &self.brushes[b_idx];
                        if (brush.contents & (CONTENTS_SOLID | CONTENTS_WINDOW | CONTENTS_OPAQUE | CONTENTS_GRATE)) == 0 {
                            continue;
                        }

                        // Clip ray against convex brush planes
                        let mut enter_frac = -1.0f32;
                        let mut exit_frac = 1.0f32;
                        let mut hit_side_idx = None;
                        let mut starts_out = false;

                        let first_side = brush.first_side as usize;
                        let num_sides = brush.num_sides as usize;

                        for s in 0..num_sides {
                            let side_idx = first_side + s;
                            if side_idx >= self.brushsides.len() {
                                break;
                            }
                            let side = &self.brushsides[side_idx];
                            let plane_idx = side.planenum as usize;
                            if plane_idx >= self.planes.len() {
                                continue;
                            }
                            let plane = &self.planes[plane_idx];

                            let d0 = start[0] * plane.normal[0]
                                + start[1] * plane.normal[1]
                                + start[2] * plane.normal[2]
                                - plane.dist;
                            let d1 = end[0] * plane.normal[0]
                                + end[1] * plane.normal[1]
                                + end[2] * plane.normal[2]
                                - plane.dist;

                            if d0 > 0.0 {
                                starts_out = true;
                            }

                            // If both are in front of plane, ray misses brush
                            if d0 > 0.0 && d1 > 0.0 {
                                enter_frac = 1.0;
                                exit_frac = -1.0;
                                break;
                            }
                            // If both are behind plane, continue
                            if d0 <= 0.0 && d1 <= 0.0 {
                                continue;
                            }

                            if d0 > d1 {
                                // Entering brush
                                let f = (d0 - 0.03125) / (d0 - d1);
                                if f > enter_frac {
                                    enter_frac = f;
                                    hit_side_idx = Some(side_idx);
                                }
                            } else {
                                // Exiting brush
                                let f = (d0 + 0.03125) / (d0 - d1);
                                if f < exit_frac {
                                    exit_frac = f;
                                }
                            }
                        }

                        if starts_out && enter_frac < exit_frac && enter_frac >= 0.0 && enter_frac < 1.0 {
                            let status = self.classify_brush(brush, hit_side_idx);
                            match status {
                                SightlineStatus::BlockedByOpaqueWall => {
                                    // Opaque wall stops sightline completely
                                    return SightlineStatus::BlockedByOpaqueWall;
                                }
                                SightlineStatus::BlockedByTransparentWindow => {
                                    hit_window = true;
                                }
                                SightlineStatus::BlockedByGrate => {
                                    hit_grate = true;
                                }
                                SightlineStatus::Clear => {}
                            }
                        }
                    }
                }
                continue;
            }

            let node_idx = entry.node_idx as usize;
            if node_idx >= self.nodes.len() {
                continue;
            }
            let node = &self.nodes[node_idx];
            let plane_idx = node.plane_num as usize;
            if plane_idx >= self.planes.len() {
                continue;
            }
            let plane = &self.planes[plane_idx];

            let p_start = [
                start[0] + entry.t_start * (end[0] - start[0]),
                start[1] + entry.t_start * (end[1] - start[1]),
                start[2] + entry.t_start * (end[2] - start[2]),
            ];
            let p_end = [
                start[0] + entry.t_end * (end[0] - start[0]),
                start[1] + entry.t_end * (end[1] - start[1]),
                start[2] + entry.t_end * (end[2] - start[2]),
            ];

            let d0 = p_start[0] * plane.normal[0]
                + p_start[1] * plane.normal[1]
                + p_start[2] * plane.normal[2]
                - plane.dist;
            let d1 = p_end[0] * plane.normal[0]
                + p_end[1] * plane.normal[1]
                + p_end[2] * plane.normal[2]
                - plane.dist;

            if d0 >= 0.0 && d1 >= 0.0 {
                // Entire segment is in front of plane
                stack.push(StackEntry {
                    node_idx: node.children[0],
                    t_start: entry.t_start,
                    t_end: entry.t_end,
                });
            } else if d0 < 0.0 && d1 < 0.0 {
                // Entire segment is behind plane
                stack.push(StackEntry {
                    node_idx: node.children[1],
                    t_start: entry.t_start,
                    t_end: entry.t_end,
                });
            } else {
                // Segment crosses plane
                let (near_child, far_child) = if d0 >= 0.0 {
                    (node.children[0], node.children[1])
                } else {
                    (node.children[1], node.children[0])
                };

                let frac = (d0 / (d0 - d1)).clamp(0.0, 1.0);
                let t_mid = entry.t_start + frac * (entry.t_end - entry.t_start);

                // Push far child first, then near child (so near child pops and executes first)
                stack.push(StackEntry {
                    node_idx: far_child,
                    t_start: t_mid,
                    t_end: entry.t_end,
                });
                stack.push(StackEntry {
                    node_idx: near_child,
                    t_start: entry.t_start,
                    t_end: t_mid,
                });
            }
        }

        if hit_window {
            SightlineStatus::BlockedByTransparentWindow
        } else if hit_grate {
            SightlineStatus::BlockedByGrate
        } else {
            SightlineStatus::Clear
        }
    }

    /// Sweeps an axis-aligned box (`mins`/`maxs` relative to the origin) from `start` to `end` through
    /// the world brushes, like the engine's CM_ClipBoxToBrush: every brush side (bevels included) is
    /// pushed out by the box corner facing it. Static props and displacements are not part of the
    /// brush data, so they never block the box.
    pub fn trace_hull(&self, start: [f32; 3], end: [f32; 3], mins: [f32; 3], maxs: [f32; 3], mask: i32) -> HullTrace {
        const DIST_EPSILON: f32 = 0.03125;
        let mut result = HullTrace { fraction: 1.0, normal: [0.0; 3], start_solid: false };
        if self.nodes.is_empty() {
            return result;
        }
        let dot = |a: &[f32; 3], b: &[f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        // Bounds of the whole sweep, used to find the leaves it can touch.
        let mut center = [0.0f32; 3];
        let mut extent = [0.0f32; 3];
        for i in 0..3 {
            let lo = start[i].min(end[i]) + mins[i] - 1.0;
            let hi = start[i].max(end[i]) + maxs[i] + 1.0;
            center[i] = (lo + hi) * 0.5;
            extent[i] = (hi - lo) * 0.5;
        }

        let mut tested = vec![false; self.brushes.len()];
        let mut stack = vec![0i32];
        while let Some(node_idx) = stack.pop() {
            if node_idx < 0 {
                let Some(leaf) = self.leafs.get((-node_idx - 1) as usize) else { continue };
                let first = leaf.first_leafbrush as usize;
                for lb in first..first + leaf.num_leafbrushes as usize {
                    let Some(&b_idx) = self.leafbrushes.get(lb) else { break };
                    let b_idx = b_idx as usize;
                    if b_idx >= self.brushes.len() || tested[b_idx] {
                        continue;
                    }
                    tested[b_idx] = true;
                    let brush = &self.brushes[b_idx];
                    if brush.contents & mask == 0 {
                        continue;
                    }

                    let mut enter_frac = -1.0f32;
                    let mut exit_frac = 1.0f32;
                    let mut hit_normal = [0.0f32; 3];
                    let mut starts_out = false;
                    let mut ends_out = false;
                    let mut missed = false;
                    for side_idx in brush.first_side as usize..(brush.first_side + brush.num_sides) as usize {
                        let Some(side) = self.brushsides.get(side_idx) else { break };
                        let Some(plane) = self.planes.get(side.planenum as usize) else { continue };
                        let n = plane.normal;
                        let ofs = [
                            if n[0] < 0.0 { maxs[0] } else { mins[0] },
                            if n[1] < 0.0 { maxs[1] } else { mins[1] },
                            if n[2] < 0.0 { maxs[2] } else { mins[2] },
                        ];
                        let dist = plane.dist - dot(&ofs, &n);
                        let d1 = dot(&start, &n) - dist;
                        let d2 = dot(&end, &n) - dist;
                        if d1 > 0.0 {
                            starts_out = true;
                        }
                        if d2 > 0.0 {
                            ends_out = true;
                        }
                        if d1 > 0.0 && (d2 >= DIST_EPSILON || d2 >= d1) {
                            missed = true;
                            break;
                        }
                        if d1 <= 0.0 && d2 <= 0.0 {
                            continue;
                        }
                        if d1 > d2 {
                            let f = ((d1 - DIST_EPSILON) / (d1 - d2)).max(0.0);
                            if f > enter_frac {
                                enter_frac = f;
                                hit_normal = n;
                            }
                        } else {
                            let f = ((d1 + DIST_EPSILON) / (d1 - d2)).min(1.0);
                            if f < exit_frac {
                                exit_frac = f;
                            }
                        }
                    }
                    if missed {
                        continue;
                    }
                    if !starts_out {
                        result.start_solid = true;
                        if !ends_out {
                            result.fraction = 0.0;
                        }
                        continue;
                    }
                    if enter_frac > -1.0 && enter_frac < exit_frac && enter_frac < result.fraction {
                        result.fraction = enter_frac.max(0.0);
                        result.normal = hit_normal;
                    }
                }
                continue;
            }
            let Some(node) = self.nodes.get(node_idx as usize) else { continue };
            let Some(plane) = self.planes.get(node.plane_num as usize) else { continue };
            let n = plane.normal;
            let dc = dot(&center, &n) - plane.dist;
            let r = n[0].abs() * extent[0] + n[1].abs() * extent[1] + n[2].abs() * extent[2];
            if dc - r >= 0.0 {
                stack.push(node.children[0]);
            } else if dc + r < 0.0 {
                stack.push(node.children[1]);
            } else {
                stack.push(node.children[0]);
                stack.push(node.children[1]);
            }
        }
        result
    }
}
