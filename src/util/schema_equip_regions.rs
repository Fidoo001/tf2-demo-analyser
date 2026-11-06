use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::HashMap;

pub const REGIONS: [&str; 66] = [
    "Back", "arm_tattoos", "arms", "back", "beard", "belt_misc", "demo_belt",
    "demo_eyepatch", "demo_head_replacement", "demoman_collar", "disconnected_floating_item",
    "ears", "engineer_belt", "engineer_hair", "engineer_left_arm", "engineer_pocket",
    "engineer_wings", "face", "feet", "flair", "glasses", "grenades", "hat",
    "head_skin", "heavy_belt", "heavy_belt_back", "heavy_bullets", "heavy_hair",
    "heavy_hip", "heavy_pocket", "heavy_towel", "left_shoulder", "lenses", "medal",
    "medic_gloves", "medic_hip", "medic_pipe", "medigun_accessories", "necklace",
    "pants", "pyro_head_replacement", "pyro_spikes", "pyro_tail", "pyro_wings",
    "right_shoulder", "scout_backpack", "scout_bandages", "scout_hands", "scout_pants",
    "scout_wings", "shirt", "sleeves", "sniper_bullets", "sniper_headband", "sniper_legs",
    "sniper_pocket", "sniper_pocket_left", "sniper_quiver", "sniper_vest", "soldier_cigar",
    "soldier_coat", "soldier_legs", "soldier_pocket", "spy_coat", "whole_head", "zombie_body"
];

#[derive(Debug, Deserialize)]
struct RawCosmetic {
    name: String,
    equip_regions: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CosmeticInfo {
    pub name: String,
    pub regions: Vec<String>,
    pub region_mask: u128,
}

pub fn is_weapon_wearable(id: u16) -> bool {
    matches!(
        id,
        57 | 131 | 133 | 231 | 405 | 406 | 444 | 608 | 642 | 1099 | 1101 | 1144 | 1179 | 1180 | 1185
    )
}

pub fn is_action_item(id: u16) -> bool {
    matches!(
        id,
        280 | 281 | 282 | 283 | 284 | 286 | 288 | 362 | 489 | 493 | 542 | 673 | 788 | 1039 | 1066 | 1069 | 1130 | 1131 | 1132 | 1152 | 1163 | 1167 | 5869
    )
}

pub fn is_dummy_or_invalid(id: u16) -> bool {
    id == 0 || id == 65535
}

pub fn region_mask_to_names(mask: u128) -> Vec<&'static str> {
    let mut names = Vec::new();
    for (i, &r) in REGIONS.iter().enumerate() {
        if (mask & (1u128 << i)) != 0 {
            names.push(r);
        }
    }
    names
}

lazy_static! {
    static ref REGION_INDEX_MAP: HashMap<&'static str, u8> = {
        let mut m = HashMap::new();
        for (i, &r) in REGIONS.iter().enumerate() {
            m.insert(r, i as u8);
        }
        m
    };

    static ref COSMETICS_MAP: HashMap<u16, CosmeticInfo> = {
        let raw_json = include_str!("../../cosmetics.json");
        let parsed: HashMap<String, RawCosmetic> = serde_json::from_str(raw_json).expect("Invalid cosmetics.json");
        let mut map = HashMap::with_capacity(parsed.len());

        for (id_str, raw) in parsed {
            if let Ok(id) = id_str.parse::<u16>() {
                let regions = raw.equip_regions.unwrap_or_default();
                let mut mask = 0u128;
                for r in &regions {
                    if let Some(&idx) = REGION_INDEX_MAP.get(r.as_str()) {
                        mask |= 1u128 << idx;
                    }
                }
                map.insert(id, CosmeticInfo {
                    name: raw.name,
                    regions,
                    region_mask: mask,
                });
            }
        }
        map
    };
}

/// Resolves cosmetic info by item definition index from official TF2 schema.
pub fn get_cosmetic_info(id: u16) -> Option<CosmeticInfo> {
    if is_dummy_or_invalid(id) || is_weapon_wearable(id) || is_action_item(id) {
        return None;
    }
    COSMETICS_MAP.get(&id).cloned()
}

/// Checks whether two cosmetic items conflict according to the official TF2 equip region conflict rules:
/// - Any shared equip region (e.g. Hat + Hat, Beard + Beard, Shirt + Shirt)
/// - Glasses conflicts with: Face, Lenses, Whole Head (ignored if ignore_quickswitch is true)
/// - Whole Head conflicts with: Hat, Face, Glasses (ignored if ignore_quickswitch is true)
pub fn check_cosmetic_conflict(
    info_a: &CosmeticInfo,
    info_b: &CosmeticInfo,
    ignore_quickswitch: bool,
) -> Option<Vec<&'static str>> {
    let direct_overlap = info_a.region_mask & info_b.region_mask;
    let mut conflicts = Vec::new();

    if direct_overlap != 0 {
        conflicts.extend(region_mask_to_names(direct_overlap));
    }

    if !ignore_quickswitch {
        let a_has = |name: &str| info_a.regions.iter().any(|r| r == name);
        let b_has = |name: &str| info_b.regions.iter().any(|r| r == name);

        // Glasses conflicts with: Face, Lenses, Whole Head
        if a_has("glasses") && (b_has("face") || b_has("lenses") || b_has("whole_head")) {
            conflicts.push("glasses");
        } else if b_has("glasses") && (a_has("face") || a_has("lenses") || a_has("whole_head")) {
            conflicts.push("glasses");
        }

        // Whole Head conflicts with: Hat, Face, Glasses
        if a_has("whole_head") && (b_has("hat") || b_has("face") || b_has("glasses")) {
            conflicts.push("whole_head");
        } else if b_has("whole_head") && (a_has("hat") || a_has("face") || a_has("glasses")) {
            conflicts.push("whole_head");
        }
    }

    if conflicts.is_empty() {
        None
    } else {
        conflicts.sort_unstable();
        conflicts.dedup();
        Some(conflicts)
    }
}

