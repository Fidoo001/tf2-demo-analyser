use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::parser::parse_bsp;
use super::trace::BspCollisionData;
use lazy_static::lazy_static;

lazy_static! {
    static ref MAP_CACHE: RwLock<HashMap<String, Option<Arc<BspCollisionData>>>> =
        RwLock::new(HashMap::new());
}

/// Resolves a map name to an absolute file path on disk.
pub fn find_map_file(map_name: &str) -> Option<PathBuf> {
    let clean_name = map_name
        .trim()
        .trim_end_matches(".bsp");

    if clean_name.is_empty() {
        return None;
    }

    let bsp_filename = format!("{}.bsp", clean_name);

    // 1. Explicit environment variable
    if let Ok(dir) = env::var("TF2_MAPS_DIR") {
        let p = Path::new(&dir).join(&bsp_filename);
        if p.is_file() {
            return Some(p);
        }
    }

    if let Ok(tf2_dir) = env::var("TF2_DIR") {
        let p = Path::new(&tf2_dir).join("tf").join("maps").join(&bsp_filename);
        if p.is_file() {
            return Some(p);
        }
    }

    // 2. Common Windows paths
    let default_windows_paths = [
        r"C:\Program Files (x86)\Steam\steamapps\common\Team Fortress 2\tf\maps",
        r"C:\Program Files\Steam\steamapps\common\Team Fortress 2\tf\maps",
        r"D:\SteamLibrary\steamapps\common\Team Fortress 2\tf\maps",
        r"E:\SteamLibrary\steamapps\common\Team Fortress 2\tf\maps",
        r"F:\SteamLibrary\steamapps\common\Team Fortress 2\tf\maps",
        r"D:\Steam\steamapps\common\Team Fortress 2\tf\maps",
        r"E:\Steam\steamapps\common\Team Fortress 2\tf\maps",
    ];

    for base in default_windows_paths {
        let p = Path::new(base).join(&bsp_filename);
        if p.is_file() {
            return Some(p);
        }
    }

    // 3. Local working directory / relative paths
    let local_paths = [
        PathBuf::from("maps").join(&bsp_filename),
        PathBuf::from(&bsp_filename),
    ];
    for p in &local_paths {
        if p.is_file() {
            return Some(p.clone());
        }
    }

    // 4. Linux / Steam Deck paths
    if let Ok(home) = env::var("HOME") {
        let linux_paths = [
            format!("{}/.steam/steam/steamapps/common/Team Fortress 2/tf/maps", home),
            format!("{}/.local/share/Steam/steamapps/common/Team Fortress 2/tf/maps", home),
        ];
        for base in linux_paths {
            let p = Path::new(&base).join(&bsp_filename);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    None
}

/// Retrieves or parses and caches the collision data for the requested map.
/// Returns None if the map file is not found or fails to parse.
pub fn get_map(map_name: &str) -> Option<Arc<BspCollisionData>> {
    let clean_name = map_name
        .trim()
        .trim_end_matches(".bsp")
        .to_string();

    if clean_name.is_empty() {
        return None;
    }

    // Read lock fast-path
    {
        let cache = MAP_CACHE.read().unwrap();
        if let Some(cached) = cache.get(&clean_name) {
            return cached.clone();
        }
    }

    // Write lock slow-path (parse map)
    let mut cache = MAP_CACHE.write().unwrap();
    if let Some(cached) = cache.get(&clean_name) {
        return cached.clone();
    }

    let bsp_data = if let Some(path) = find_map_file(&clean_name) {
        match parse_bsp(&path) {
            Ok(data) => Some(Arc::new(data)),
            Err(e) => {
                eprintln!("[BSP] Failed to parse map {}: {}", clean_name, e);
                None
            }
        }
    } else {
        None
    };

    cache.insert(clean_name, bsp_data.clone());
    bsp_data
}
