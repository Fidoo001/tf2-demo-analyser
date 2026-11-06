pub mod cache;
pub mod lumps;
pub mod parser;
pub mod trace;

pub use cache::{find_map_file, get_map};
pub use trace::{BspCollisionData, HullTrace, SightlineStatus, MASK_PLAYERSOLID_BRUSH};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_pl_badwater() {
        if let Some(path) = find_map_file("pl_badwater") {
            println!("Found badwater at {:?}", path);
            let bsp = get_map("pl_badwater").expect("Failed to load pl_badwater");
            println!(
                "Loaded badwater: {} planes, {} nodes, {} leafs, {} brushes, {} materials",
                bsp.planes.len(),
                bsp.nodes.len(),
                bsp.leafs.len(),
                bsp.brushes.len(),
                bsp.material_names.len()
            );
            assert!(!bsp.nodes.is_empty());
            assert!(!bsp.leafs.is_empty());

            // Test raycast through open air (e.g. slight offset in sky/open courtyard)
            // Or test raycast right through a solid plane
            let status = bsp.trace_sightline([0.0, 0.0, 0.0], [0.0, 0.0, 100.0]);
            println!("Trace status: {:?}", status);
        } else {
            println!("pl_badwater.bsp not found on system, skipping test.");
        }

        if let Some(path) = find_map_file("pl_barnblitz") {
            println!("Found barnblitz at {:?}", path);
            let bsp = get_map("pl_barnblitz").expect("Failed to load pl_barnblitz");
            let t374 = bsp.trace_sightline([1199.375, 821.0, -52.875], [1007.6, -1399.0, -21.875]);
            let t16911 = bsp.trace_sightline([537.5, -717.25, -52.875], [429.125, -1605.875, -152.25]);
            let t18916 = bsp.trace_sightline([292.0, -1555.0, 59.0], [-332.0, -1277.0, 103.0]);
            println!("Trace tick 374: {:?}", t374);
            println!("Trace tick 16911: {:?}", t16911);
            println!("Trace tick 18916: {:?}", t18916);
        }
    }
}

