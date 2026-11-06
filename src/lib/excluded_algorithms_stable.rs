//! Hardcoded list of stable algorithms excluded from the global sliding window filter.
//!
//! Excluded categories:
//! - Algorithms with 100% accuracy (e.g. impossible angles or illegal equip regions)
//! - Learned algorithms that already aggregate evidence per player over the whole demo

/// Stable algorithms that are excluded from the global sliding window filter.
/// Detections from these algorithms are never hidden by the sliding window filter.
pub const STABLE_EXCLUDED_ALGORITHMS: &[&str] = &[
    // Algorithms with 100% accuracy
    "nocrex/oob_pitch",
    "oob_pitch",
    "fidoo/invalid_equip_region",
    "invalid_equip_region",

    // Learned detections: per-player score over all events of the demo, already aggregated
    "fidoo/ml_aimbot",
    "ml_aimbot",
    "fidoo/ml_triggerbot",
    "ml_triggerbot",
    "fidoo/ml_autobackstab",
    "ml_autobackstab",
    "fidoo/ml_psilent",
    "ml_psilent",
    "fidoo/ml_silent_aim",
    "ml_silent_aim",
    "fidoo/ml_backtrack",
    "ml_backtrack",
];

/// Returns whether a given algorithm name is in the stable excluded list.
pub fn is_stable_excluded(algorithm_name: &str) -> bool {
    let name_trimmed = algorithm_name.trim();
    let short_name = name_trimmed
        .strip_prefix("fidoo/")
        .or_else(|| name_trimmed.strip_prefix("nocrex/"))
        .unwrap_or(name_trimmed);

    STABLE_EXCLUDED_ALGORITHMS
        .iter()
        .any(|&excluded| excluded.eq_ignore_ascii_case(name_trimmed) || excluded.eq_ignore_ascii_case(short_name))
}
