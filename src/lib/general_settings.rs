use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::lib::algorithm::{is_excluded_from_sliding_window, Detection};
use crate::lib::parameters::{Parameter, Parameters};

/// General program settings stored under "general_settings" in params.json.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneralSettings {
    /// Whether the global sliding window filter is enabled.
    pub global_sliding_window_enabled: bool,
    /// Span of the sliding window in ticks (default: 20000, ~5 minutes at 66.67 tps).
    pub sliding_window_ticks: u32,
    /// Minimum detections required within a sliding window for a player (or algorithm) to qualify.
    pub sliding_window_detections: usize,
    /// When true, the sliding window threshold is evaluated per algorithm individually.
    /// When false, detections from all algorithms for a player are aggregated into the window.
    pub per_algorithm: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: false,
        }
    }
}

impl GeneralSettings {
    /// Converts these settings into a `Parameters` map compatible with `params.json`.
    pub fn to_parameters(&self) -> Parameters {
        let mut map = HashMap::new();
        map.insert(
            "global_sliding_window_enabled".to_string(),
            Parameter::Bool(self.global_sliding_window_enabled),
        );
        map.insert(
            "sliding_window_ticks".to_string(),
            Parameter::Int(self.sliding_window_ticks as i32),
        );
        map.insert(
            "sliding_window_detections".to_string(),
            Parameter::Int(self.sliding_window_detections as i32),
        );
        map.insert(
            "per_algorithm".to_string(),
            Parameter::Bool(self.per_algorithm),
        );
        map
    }

    /// Parses general settings from a `Parameters` map. Supports snake_case, spaces, and case-insensitivity.
    pub fn from_parameters(params: &Parameters) -> Self {
        let mut settings = Self::default();

        for (k, v) in params {
            let key = k.to_lowercase().replace(' ', "_");
            match key.as_str() {
                "global_sliding_window_enabled" => {
                    if let Ok(b) = bool::try_from(v) {
                        settings.global_sliding_window_enabled = b;
                    }
                }
                "sliding_window_ticks" => {
                    if let Ok(i) = i32::try_from(v) {
                        settings.sliding_window_ticks = i.max(0) as u32;
                    } else if let Ok(u) = u64::try_from(v) {
                        settings.sliding_window_ticks = u as u32;
                    }
                }
                "sliding_window_detections" => {
                    if let Ok(i) = i32::try_from(v) {
                        settings.sliding_window_detections = i.max(1) as usize;
                    } else if let Ok(u) = u64::try_from(v) {
                        settings.sliding_window_detections = u.max(1) as usize;
                    }
                }
                "per_algorithm" => {
                    if let Ok(b) = bool::try_from(v) {
                        settings.per_algorithm = b;
                    }
                }
                _ => {}
            }
        }

        settings
    }

    /// Extracts `GeneralSettings` from a full config map (e.g., loaded from `params.json`).
    pub fn from_config(config: &HashMap<String, Parameters>) -> Self {
        for (key, params) in config {
            let normalized = key.to_lowercase().replace([' ', '-'], "_");
            if normalized == "general_settings" {
                return Self::from_parameters(params);
            }
        }
        Self::default()
    }

    /// Extracts `GeneralSettings` from an optional config map.
    pub fn from_config_opt(config: &Option<HashMap<String, Parameters>>) -> Self {
        match config {
            Some(cfg) => Self::from_config(cfg),
            None => Self::load_default(),
        }
    }

    /// Attempts to find and load `general_settings` from `params.json` in cwd or exe directory.
    /// Falls back to `GeneralSettings::default()` if not found or on error.
    pub fn load_default() -> Self {
        let candidate_path = if std::path::Path::new("params.json").exists() {
            Some(std::path::PathBuf::from("params.json"))
        } else if let Ok(mut exe) = std::env::current_exe() {
            exe.pop();
            let p = exe.join("params.json");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(p) = candidate_path {
            if let Ok(content) = std::fs::read_to_string(p) {
                if let Ok(cfg) = serde_json::from_str::<HashMap<String, Parameters>>(&content) {
                    return Self::from_config(&cfg);
                }
            }
        }

        Self::default()
    }

    /// Applies the global sliding window filter to a slice of detections.
    ///
    /// Semantics:
    /// - If `!global_sliding_window_enabled` or `sliding_window_detections <= 1`, all detections are retained.
    /// - Detections from excluded algorithms (`CheatAlgorithm::excluded_from_sliding_window`, set in each
    ///   algorithm's own file) are ALWAYS retained.
    /// - When `per_algorithm == false`: If ANY sliding window of `sliding_window_ticks` contains >= `sliding_window_detections`
    ///   for a player across any algorithms (or if the player has an excluded algorithm detection), the player is considered qualified,
    ///   and ALL detections for that player are kept (including earlier and later ones). If no window reaches the threshold and no
    ///   excluded algorithms triggered, non-excluded detections for that player are hidden.
    /// - When `per_algorithm == true`: Excluded algorithms are always retained. Non-excluded algorithms are evaluated per
    ///   `(player, algorithm)` pair: if ANY window contains >= threshold detections for that specific algorithm on that player,
    ///   ALL detections for that pair are retained.
    pub fn filter_detections(&self, detections: &[Detection]) -> Vec<Detection> {
        if !self.global_sliding_window_enabled || self.sliding_window_detections <= 1 {
            return detections.to_vec();
        }

        let k = self.sliding_window_detections;
        let window_ticks = self.sliding_window_ticks;

        if !self.per_algorithm {
            // Group by player
            let mut player_ticks: HashMap<u64, Vec<u32>> = HashMap::new();
            let mut qualified_players = HashSet::new();

            for d in detections {
                if is_excluded_from_sliding_window(&d.algorithm) {
                    // Excluded algorithm detections immediately qualify the player
                    qualified_players.insert(d.player);
                }
                player_ticks.entry(d.player).or_default().push(d.tick);
            }

            for (player, mut ticks) in player_ticks {
                if qualified_players.contains(&player) {
                    continue;
                }
                if ticks.len() < k {
                    continue;
                }
                ticks.sort_unstable();
                for i in (k - 1)..ticks.len() {
                    if ticks[i].saturating_sub(ticks[i - (k - 1)]) <= window_ticks {
                        qualified_players.insert(player);
                        break;
                    }
                }
            }

            detections
                .iter()
                .filter(|d| is_excluded_from_sliding_window(&d.algorithm) || qualified_players.contains(&d.player))
                .cloned()
                .collect()
        } else {
            // Group by (player, algorithm)
            let mut group_ticks: HashMap<(u64, String), Vec<u32>> = HashMap::new();
            for d in detections {
                if !is_excluded_from_sliding_window(&d.algorithm) {
                    group_ticks
                        .entry((d.player, d.algorithm.clone()))
                        .or_default()
                        .push(d.tick);
                }
            }

            let mut qualified_groups = HashSet::new();
            for (group, mut ticks) in group_ticks {
                if ticks.len() < k {
                    continue;
                }
                ticks.sort_unstable();
                for i in (k - 1)..ticks.len() {
                    if ticks[i].saturating_sub(ticks[i - (k - 1)]) <= window_ticks {
                        qualified_groups.insert(group);
                        break;
                    }
                }
            }

            detections
                .iter()
                .filter(|d| is_excluded_from_sliding_window(&d.algorithm) || qualified_groups.contains(&(d.player, d.algorithm.clone())))
                .cloned()
                .collect()
        }
    }
}

/// GUI color settings, stored as hex codes alongside the other keys under "general_settings".
/// Only the GUI reads these; the CLI ignores them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorSettings {
    /// When true, the automatic light/dark color scheme is disabled and the colors below are used.
    pub custom_colors_enabled: bool,
    /// Default text / element color.
    pub text_color: [u8; 3],
    /// Background color.
    pub background_color: [u8; 3],
    /// Background of selected fields (e.g. the selected player or detection).
    pub selected_color: [u8; 3],
    /// Color of players that are neither marked nor deleted.
    pub normal_player_color: [u8; 3],
    /// Color of players marked in `reported_ids.txt`.
    pub marked_player_color: [u8; 3],
    /// Color of players whose Steam account no longer exists.
    pub deleted_player_color: [u8; 3],
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            custom_colors_enabled: false,
            text_color: [0xB4, 0xB4, 0xB4],
            background_color: [0x1B, 0x1B, 0x1B],
            selected_color: [0x00, 0x5C, 0x80],
            normal_player_color: [0x4C, 0xE0, 0x3C],
            marked_player_color: [0xEB, 0x46, 0x46],
            deleted_player_color: [0x6E, 0x6E, 0x6E],
        }
    }
}

impl ColorSettings {
    pub const ENABLED_KEY: &'static str = "custom_colors_enabled";
    pub const TEXT_KEY: &'static str = "text_color";
    pub const BACKGROUND_KEY: &'static str = "background_color";
    pub const SELECTED_KEY: &'static str = "selected_color";
    pub const NORMAL_KEY: &'static str = "normal_player_color";
    pub const MARKED_KEY: &'static str = "marked_player_color";
    pub const DELETED_KEY: &'static str = "deleted_player_color";

    /// Converts these settings into `Parameters` entries (hex strings) for `params.json`.
    pub fn to_parameters(&self) -> Parameters {
        let mut map = HashMap::new();
        map.insert(Self::ENABLED_KEY.to_string(), Parameter::Bool(self.custom_colors_enabled));
        map.insert(Self::TEXT_KEY.to_string(), Parameter::String(format_hex_rgb(self.text_color)));
        map.insert(Self::BACKGROUND_KEY.to_string(), Parameter::String(format_hex_rgb(self.background_color)));
        map.insert(Self::SELECTED_KEY.to_string(), Parameter::String(format_hex_rgb(self.selected_color)));
        map.insert(Self::NORMAL_KEY.to_string(), Parameter::String(format_hex_rgb(self.normal_player_color)));
        map.insert(Self::MARKED_KEY.to_string(), Parameter::String(format_hex_rgb(self.marked_player_color)));
        map.insert(Self::DELETED_KEY.to_string(), Parameter::String(format_hex_rgb(self.deleted_player_color)));
        map
    }

    /// Parses color settings from a `general_settings` map. Invalid hex codes fall back to the default.
    pub fn from_parameters(params: &Parameters) -> Self {
        let mut settings = Self::default();

        for (k, v) in params {
            let key = k.to_lowercase().replace(' ', "_");
            let target = match key.as_str() {
                Self::ENABLED_KEY => {
                    if let Ok(b) = bool::try_from(v) {
                        settings.custom_colors_enabled = b;
                    }
                    continue;
                }
                Self::TEXT_KEY => &mut settings.text_color,
                Self::BACKGROUND_KEY => &mut settings.background_color,
                Self::SELECTED_KEY => &mut settings.selected_color,
                Self::NORMAL_KEY => &mut settings.normal_player_color,
                Self::MARKED_KEY => &mut settings.marked_player_color,
                Self::DELETED_KEY => &mut settings.deleted_player_color,
                _ => continue,
            };
            if let Parameter::String(s) = v {
                if let Some(rgb) = parse_hex_rgb(s) {
                    *target = rgb;
                }
            }
        }

        settings
    }

    /// Extracts `ColorSettings` from a full config map (e.g., loaded from `params.json`).
    pub fn from_config(config: &HashMap<String, Parameters>) -> Self {
        for (key, params) in config {
            let normalized = key.to_lowercase().replace([' ', '-'], "_");
            if normalized == "general_settings" {
                return Self::from_parameters(params);
            }
        }
        Self::default()
    }
}

/// GUI UI scale, stored as a percentage alongside the other keys under "general_settings".
/// Only the GUI reads this; the CLI ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiScaleSettings {
    /// UI scale in percent, between `MIN_PERCENT` and `MAX_PERCENT`.
    pub percent: i32,
}

impl Default for UiScaleSettings {
    fn default() -> Self {
        Self { percent: Self::DEFAULT_PERCENT }
    }
}

impl UiScaleSettings {
    pub const KEY: &'static str = "ui_scale_percent";
    pub const MIN_PERCENT: i32 = 80;
    pub const DEFAULT_PERCENT: i32 = 100;
    pub const MAX_PERCENT: i32 = 150;

    /// Clamps a percentage into the supported range.
    pub fn clamp_percent(percent: i32) -> i32 {
        percent.clamp(Self::MIN_PERCENT, Self::MAX_PERCENT)
    }

    /// The scale as a multiplier (1.0 = 100%).
    pub fn factor(&self) -> f32 {
        self.percent as f32 / 100.0
    }

    /// Converts these settings into a `Parameters` entry for `params.json`.
    pub fn to_parameters(&self) -> Parameters {
        let mut map = HashMap::new();
        map.insert(Self::KEY.to_string(), Parameter::Int(self.percent));
        map
    }

    /// Parses the scale from a `general_settings` map. Missing or invalid values fall back to the default,
    /// out-of-range values are clamped.
    pub fn from_parameters(params: &Parameters) -> Self {
        let mut settings = Self::default();
        for (k, v) in params {
            if k.to_lowercase().replace(' ', "_") != Self::KEY {
                continue;
            }
            let percent = match v {
                Parameter::Int(i) => Some(*i as i64),
                Parameter::UInt(u) => Some(*u as i64),
                Parameter::Float(f) if f.is_finite() => Some(f.round() as i64),
                _ => None,
            };
            if let Some(p) = percent {
                settings.percent = Self::clamp_percent(p.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
            }
        }
        settings
    }

    /// Extracts `UiScaleSettings` from a full config map (e.g., loaded from `params.json`).
    pub fn from_config(config: &HashMap<String, Parameters>) -> Self {
        for (key, params) in config {
            let normalized = key.to_lowercase().replace([' ', '-'], "_");
            if normalized == "general_settings" {
                return Self::from_parameters(params);
            }
        }
        Self::default()
    }
}

/// Parses a `#RRGGBB` (or `RRGGBB`) hex color code.
pub fn parse_hex_rgb(s: &str) -> Option<[u8; 3]> {
    let hex = s.trim();
    let hex = hex.strip_prefix('#').unwrap_or(hex);
    if hex.len() != 6 || !hex.is_ascii() {
        return None;
    }
    let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some([channel(0)?, channel(2)?, channel(4)?])
}

/// Formats an RGB color as a `#RRGGBB` hex code.
pub fn format_hex_rgb(rgb: [u8; 3]) -> String {
    format!("#{:02X}{:02X}{:02X}", rgb[0], rgb[1], rgb[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_det(tick: u32, player: u64, algorithm: &str) -> Detection {
        Detection {
            tick,
            algorithm: algorithm.to_string(),
            player,
            data: json!({}),
        }
    }

    #[test]
    fn test_default_settings() {
        let settings = GeneralSettings::default();
        assert!(settings.global_sliding_window_enabled);
        assert_eq!(settings.sliding_window_ticks, 20000);
        assert_eq!(settings.sliding_window_detections, 2);
        assert!(!settings.per_algorithm);

        // When detections threshold is 2, single isolated detections are filtered
        let dets = vec![make_det(100, 1, "test_algo")];
        assert_eq!(settings.filter_detections(&dets).len(), 0);

        // 2 detections within 20000 ticks are kept
        let dets2 = vec![
            make_det(100, 1, "test_algo"),
            make_det(500, 1, "test_algo"),
        ];
        assert_eq!(settings.filter_detections(&dets2).len(), 2);
    }

    #[test]
    fn test_filter_isolated_false_flags() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: false,
        };

        // Player 1 has 2 isolated detections 50,000 ticks apart -> should be hidden
        let isolated = vec![
            make_det(1000, 1, "algo_a"),
            make_det(51000, 1, "algo_b"),
        ];
        assert_eq!(settings.filter_detections(&isolated).len(), 0);

        // Player 2 has 2 detections within 500 ticks -> both should be kept
        let cheater = vec![
            make_det(1000, 2, "algo_a"),
            make_det(1500, 2, "algo_b"),
        ];
        assert_eq!(settings.filter_detections(&cheater).len(), 2);
    }

    #[test]
    fn test_keeps_all_past_and_future_detections_when_qualified() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: false,
        };

        // Player 1 has:
        // - past detection at tick 100
        // - burst of 2 detections at tick 20000, 20500 (meets window of 20000)
        // - future isolated detection at tick 90000
        // ALL 4 detections must be kept!
        let dets = vec![
            make_det(100, 1, "algo_a"),
            make_det(20000, 1, "algo_b"),
            make_det(20500, 1, "algo_a"),
            make_det(90000, 1, "algo_c"),
        ];

        let filtered = settings.filter_detections(&dets);
        assert_eq!(filtered.len(), 4);
        assert_eq!(filtered[0].tick, 100);
        assert_eq!(filtered[1].tick, 20000);
        assert_eq!(filtered[2].tick, 20500);
        assert_eq!(filtered[3].tick, 90000);
    }

    #[test]
    fn test_per_algorithm_flag() {
        // With per_algorithm = false, two different algorithms on same player combine:
        let settings_global = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: false,
        };

        let dets = vec![
            make_det(1000, 1, "algo_a"),
            make_det(1500, 1, "algo_b"),
        ];
        assert_eq!(settings_global.filter_detections(&dets).len(), 2);

        // With per_algorithm = true, different algorithms do NOT combine:
        let settings_per_algo = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: true,
        };
        assert_eq!(settings_per_algo.filter_detections(&dets).len(), 0);

        // But if algo_a has 2 detections within window:
        let dets_matching = vec![
            make_det(1000, 1, "algo_a"),
            make_det(1500, 1, "algo_a"),
            make_det(1600, 1, "algo_b"), // algo_b only has 1
        ];
        let filtered = settings_per_algo.filter_detections(&dets_matching);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].algorithm, "algo_a");
        assert_eq!(filtered[1].algorithm, "algo_a");
    }

    #[test]
    fn test_disabled_sliding_window() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: false,
            sliding_window_ticks: 20000,
            sliding_window_detections: 2,
            per_algorithm: false,
        };

        let dets = vec![make_det(1000, 1, "algo_a")];
        assert_eq!(settings.filter_detections(&dets).len(), 1);
    }

    #[test]
    fn test_stable_excluded_algorithms_never_filtered() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 3,
            per_algorithm: false,
        };

        // 100% accuracy algorithms (stable) should NEVER be filtered, even as a single detection
        let dets = vec![
            make_det(1000, 1, "skiesred/oob_pitch"),
            make_det(2000, 2, "fidoo/invalid_equip_region"),
        ];

        let filtered = settings.filter_detections(&dets);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_stable_excluded_qualifies_player_globally() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 3,
            per_algorithm: false,
        };

        // Player 1 has 1 oob_pitch detection and 1 un-excluded isolated detection.
        // Because oob_pitch is excluded (and qualifies player 1), all detections on player 1 are kept.
        let dets = vec![
            make_det(1000, 1, "skiesred/oob_pitch"),
            make_det(50000, 1, "some_aimbot"),
            make_det(1000, 2, "some_aimbot"), // player 2 has only 1 isolated non-excluded detection -> filtered out
        ];

        let filtered = settings.filter_detections(&dets);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|d| d.player == 1));
    }

    #[test]
    fn test_color_settings_round_trip() {
        assert_eq!(parse_hex_rgb("#EB4646"), Some([0xEB, 0x46, 0x46]));
        assert_eq!(parse_hex_rgb("6e6e6e"), Some([0x6E, 0x6E, 0x6E]));
        assert_eq!(parse_hex_rgb("#12345"), None);
        assert_eq!(parse_hex_rgb("#GG0000"), None);

        let colors = ColorSettings {
            custom_colors_enabled: true,
            text_color: [1, 2, 3],
            ..ColorSettings::default()
        };
        assert_eq!(ColorSettings::from_parameters(&colors.to_parameters()), colors);

        // Invalid hex falls back to the default for that field only
        let mut params = colors.to_parameters();
        params.insert("text_color".to_string(), Parameter::String("nope".to_string()));
        let parsed = ColorSettings::from_parameters(&params);
        assert_eq!(parsed.text_color, ColorSettings::default().text_color);
        assert!(parsed.custom_colors_enabled);

        // Color keys do not disturb the sliding window settings
        let mut gen = GeneralSettings::default().to_parameters();
        gen.extend(colors.to_parameters());
        assert_eq!(GeneralSettings::from_parameters(&gen), GeneralSettings::default());
    }

    #[test]
    fn test_ui_scale_settings() {
        assert_eq!(UiScaleSettings::default().percent, 100);
        assert_eq!(UiScaleSettings::default().factor(), 1.0);

        let scale = UiScaleSettings { percent: 150 };
        assert_eq!(UiScaleSettings::from_parameters(&scale.to_parameters()), scale);

        // Out-of-range and wrongly typed values are clamped / ignored
        let mut params = HashMap::new();
        params.insert(UiScaleSettings::KEY.to_string(), Parameter::Int(500));
        assert_eq!(UiScaleSettings::from_parameters(&params).percent, 150);
        params.insert(UiScaleSettings::KEY.to_string(), Parameter::Int(175));
        assert_eq!(UiScaleSettings::from_parameters(&params).percent, 150);
        params.insert(UiScaleSettings::KEY.to_string(), Parameter::Int(-3));
        assert_eq!(UiScaleSettings::from_parameters(&params).percent, 80);
        params.insert(UiScaleSettings::KEY.to_string(), Parameter::Int(80));
        assert_eq!(UiScaleSettings::from_parameters(&params).percent, 80);
        params.insert(UiScaleSettings::KEY.to_string(), Parameter::Bool(true));
        assert_eq!(UiScaleSettings::from_parameters(&params).percent, 100);

        // The scale key does not disturb the sliding window settings
        let mut gen = GeneralSettings::default().to_parameters();
        gen.extend(scale.to_parameters());
        assert_eq!(GeneralSettings::from_parameters(&gen), GeneralSettings::default());
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn test_experimental_excluded_algorithms() {
        let settings = GeneralSettings {
            global_sliding_window_enabled: true,
            sliding_window_ticks: 20000,
            sliding_window_detections: 3,
            per_algorithm: true,
        };

        let dets = vec![
            make_det(1000, 1, "fidoo/tampered"),
            make_det(2000, 1, "fidoo/recorder_bhop"),
            make_det(3000, 1, "fidoo/demoswap_precision"),
            make_det(4000, 1, "sv_cheats_bypass"),
            make_det(5000, 1, "fidoo/lboxaimbot"),
            make_det(6000, 1, "fidoo/triggerbot_double_inputs"),
            make_det(7000, 1, "suspicious_motd"),
            make_det(8000, 1, "fidoo/votekick_votes"),
            make_det(9000, 1, "isolated_generic_algo"), // should be filtered out
        ];

        let filtered = settings.filter_detections(&dets);
        assert_eq!(filtered.len(), 8);
        assert!(!filtered.iter().any(|d| d.algorithm == "isolated_generic_algo"));
        assert!(filtered.iter().any(|d| d.algorithm == "fidoo/votekick_votes"));
    }
}

