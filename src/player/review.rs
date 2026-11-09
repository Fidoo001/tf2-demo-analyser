//! Review mode's bookkeeping: grouping a player's detections into moments to step
//! through, and turning what each check measured into readable lines.

use serde_json::Value;

use super::view::Marker;

/// One moment to review: a tick and the detections made around it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Moment {
    pub tick: u32,
    /// Indexes into the marker list.
    pub markers: Vec<usize>,
}

/// A player's detections in time order, with ones closer than `gap_ticks` (like two
/// checks firing on the same shot) merged into one moment.
pub fn moments(markers: &[Marker], steam_id: u64, gap_ticks: u32) -> Vec<Moment> {
    let mut mine: Vec<usize> = (0..markers.len()).filter(|&i| markers[i].steam_id == steam_id).collect();
    mine.sort_by_key(|&i| markers[i].tick);
    let mut moments: Vec<Moment> = Vec::new();
    for index in mine {
        let tick = markers[index].tick;
        match moments.last_mut() {
            Some(moment) if tick.saturating_sub(moment.tick) <= gap_ticks => moment.markers.push(index),
            _ => moments.push(Moment { tick, markers: vec![index] }),
        }
    }
    moments
}

/// Players with detections, most detections first: (steam id, how many).
pub fn flagged_players(markers: &[Marker]) -> Vec<(u64, usize)> {
    let mut counts: Vec<(u64, usize)> = Vec::new();
    for marker in markers.iter().filter(|m| m.steam_id != 0) {
        match counts.iter_mut().find(|(id, _)| *id == marker.steam_id) {
            Some((_, count)) => *count += 1,
            None => counts.push((marker.steam_id, 1)),
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
}

/// Checks whose detections are snaps or flicks onto a target, which show how far
/// from the crosshair an aimbot reaches.
fn is_aim_check(check: &str) -> bool {
    (check.contains("aim") || check.contains("psilent")) && !check.contains("triggerbot")
}

/// How far a detection's aim flicked, in degrees, from what the check measured:
/// a silent flick's way out, or a snap's size.
fn reported_flick(data: &Value) -> Option<f32> {
    let field = |name: &str| data.get(name).and_then(Value::as_f64).map(|value| value as f32);
    field("out_angle").or_else(|| {
        let snaps = [field("snap_angle"), field("eye_angle_snap")];
        snaps.into_iter().flatten().reduce(f32::max)
    })
}

/// Flicks bigger than this are something else, like a turn to a new fight.
const LARGEST_FLICK: f32 = 90.0;

/// The smallest aimbot FOV that explains a player's aim detections: the biggest
/// flick among them, in degrees, and how many flicks it's based on. `measure` works
/// a flick out from the demo for checks that don't report one. The real setting
/// can be bigger; a player who never needed a big flick looks smaller than it.
pub fn aimbot_fov(markers: &[Marker], steam_id: u64, measure: &dyn Fn(&Marker) -> Option<f32>) -> Option<(f32, usize)> {
    let flicks: Vec<f32> = markers
        .iter()
        .filter(|marker| marker.steam_id == steam_id && is_aim_check(check_of(marker)))
        .filter_map(|marker| reported_flick(&marker.data).or_else(|| measure(marker)))
        .filter(|angle| angle.is_finite() && *angle > 0.0 && *angle <= LARGEST_FLICK)
        .collect();
    let largest = flicks.iter().copied().reduce(f32::max)?;
    Some((largest, flicks.len()))
}

/// The check that made a detection, or its label when that isn't known.
pub fn check_of(marker: &Marker) -> &str {
    if marker.check.is_empty() {
        &marker.label
    } else {
        &marker.check
    }
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// A check's name for people: "fidoo/ml_silent_aim" becomes "Silent aim (learned model)".
pub fn check_title(check: &str) -> String {
    let name = check.rsplit('/').next().unwrap_or(check);
    let (name, learned) = match name.strip_prefix("ml_") {
        Some(rest) => (rest, true),
        None => (name, false),
    };
    let words = capitalized(&name.replace('_', " "));
    if learned {
        format!("{words} (learned model)")
    } else {
        words
    }
}

/// What to look at on screen to see what a check saw.
pub fn what_to_watch(check: &str) -> &'static str {
    let has = |part: &str| check.contains(part);
    if has("psilent") {
        "Watch where the projectile leaves: it flies off in a direction their view never pointed."
    } else if has("silent") || has("sniperaim") {
        "Watch the aim line at the shot: the view jumps onto the target for one update and straight back."
    } else if has("triggerbot") {
        "Watch the crosshair: the shot goes off the instant it crosses the head, even mid-sweep."
    } else if has("backtrack") {
        "Watch the ghosts: the hit lands where the target was several ticks ago, not where they are."
    } else if has("backstab") {
        "Watch the knife: it stabs on the first possible tick, often with the view snapping onto the victim."
    } else if has("aimbot") || has("aimlock") || has("plain_aim") {
        "Watch the aim graph: a turn spike no hand makes, landing right on the target."
    } else if has("high_hit_rate") {
        "This one is about the whole match: they hit far more rifle shots than anyone can."
    } else if has("nospread") {
        "Watch the bullet lines: they land dead centre where weapon spread should scatter them."
    } else if has("autostrafe") {
        "Watch them in the air: speed builds and curves while the view stays still."
    } else if has("bunnyhop") || has("bhop") {
        "Watch the landings: every one jumps again on the very first tick."
    } else if has("duckspeed") {
        "Watch their movement: full running speed while fully crouched."
    } else if has("antiaim") || has("180") || has("pitch") {
        "Watch their head: the view flips to angles a mouse can't make."
    } else if has("fakelag") {
        "Watch their movement: they freeze, then jump ahead, over and over."
    } else if has("doubletap") {
        "Watch the shots: two go out faster than the weapon can fire."
    } else if has("cosmetic") || has("equip") {
        "This one is about their loadout: items that can't be worn together."
    } else {
        "Watch the aim line and the aim graph around the moment."
    }
}

/// A field name as words: "flick_miss_hu" becomes "Flick miss (units)".
pub fn readable_key(key: &str) -> String {
    let clean: String = key.chars().filter(|&c| c != '\u{200B}').collect();
    let words: Vec<String> = clean
        .split(['_', ' '])
        .filter(|word| !word.is_empty())
        .map(|word| match word.to_ascii_lowercase().as_str() {
            "hu" => "(units)".to_string(),
            "dmg" => "damage".to_string(),
            "vel" => "velocity".to_string(),
            "pos" => "position".to_string(),
            "num" => "number of".to_string(),
            "fov" => "FOV".to_string(),
            "hp" => "health".to_string(),
            "id" => "ID".to_string(),
            "radii" => "(head sizes)".to_string(),
            other => other.to_string(),
        })
        .collect();
    capitalized(&words.join(" "))
}

fn number(value: f64) -> String {
    if (value - value.round()).abs() < 1e-9 && value.abs() < 1e12 {
        let whole = value.round() as i64;
        let digits = whole.unsigned_abs().to_string();
        let mut grouped = String::new();
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(c);
        }
        if whole < 0 {
            format!("-{grouped}")
        } else {
            grouped
        }
    } else {
        let text = format!("{value:.2}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// A number with the unit its field's name suggests.
fn with_unit(key: &str, value: f64) -> String {
    let key = key.to_ascii_lowercase();
    let has = |part: &str| key.contains(part);
    let shown = number(value);
    // Names that say their unit already, like "_hu" and "_radii".
    if key.ends_with("_hu") || has("radii") {
        shown
    } else if key.ends_with("score") && (0.0..=1.0).contains(&value) {
        format!("{:.0}%", value * 100.0)
    } else if has("tick") {
        if shown == "1" { "1 tick".to_string() } else { format!("{shown} ticks") }
    } else if has("angle") || has("snap") || has("deviation") || has("degrees") || has("cone") || has("pitch") || has("yaw") || key.ends_with("_deg") {
        format!("{shown}°")
    } else if key.ends_with("_ms") || key.ends_with("ping") {
        format!("{shown} ms")
    } else if has("distance") {
        format!("{shown} units")
    } else {
        shown
    }
}

/// A field's value in words, with a unit when its name suggests one.
pub fn readable_value(key: &str, value: &Value) -> String {
    match value {
        Value::Bool(true) => "yes".to_string(),
        Value::Bool(false) => "no".to_string(),
        Value::Number(n) => n.as_f64().map_or_else(|| n.to_string(), |v| with_unit(key, v)),
        Value::String(text) => text.replace('_', " "),
        Value::Array(items) => {
            let shown: Vec<String> = items.iter().take(6).map(|item| readable_value(key, item)).collect();
            let more = if items.len() > 6 { format!(" … ({} in all)", items.len()) } else { String::new() };
            format!("{}{more}", shown.join(", "))
        }
        Value::Object(_) => {
            let text = value.to_string();
            if text.chars().count() > 60 {
                format!("{}…", text.chars().take(60).collect::<String>())
            } else {
                text
            }
        }
        Value::Null => "-".to_string(),
    }
}

/// Fields shown first, in this order, when a check reports them.
const FIRST_FIELDS: [&str; 7] = ["class", "weapon", "weapon_name", "target", "type", "score", "shot_score"];
/// Fields that mean nothing to a person.
const HIDDEN_FIELDS: [&str; 2] = ["server_name", "weapon_id"];

/// What a check measured, as (label, value) lines, with the most telling ones first.
/// `name_of` turns a SteamID64 (like the "target" field) into a player's name.
pub fn measured(data: &Value, name_of: &dyn Fn(u64) -> Option<String>) -> Vec<(String, String)> {
    let Value::Object(fields) = data else { return Vec::new() };
    let line = |key: &str, value: &Value| {
        let shown = match (key, value.as_u64()) {
            ("target" | "victim", Some(id)) => name_of(id).unwrap_or_else(|| id.to_string()),
            _ => readable_value(key, value),
        };
        let label = match key {
            "score" => "Player score".to_string(),
            "shot_score" => "This shot's score".to_string(),
            "weapon_name" => "Weapon".to_string(),
            _ => readable_key(key),
        };
        (label, shown)
    };
    let clean = |key: &str| key.chars().filter(|&c| c != '\u{200B}').collect::<String>();
    let mut lines: Vec<(String, String)> = Vec::new();
    for wanted in FIRST_FIELDS {
        if let Some((key, value)) = fields.iter().find(|(key, value)| clean(key) == wanted && !value.is_null()) {
            lines.push(line(&clean(key), value));
        }
    }
    for (key, value) in fields {
        let key = clean(key);
        if FIRST_FIELDS.contains(&key.as_str()) || HIDDEN_FIELDS.contains(&key.as_str()) || value.is_null() {
            continue;
        }
        lines.push(line(&key, value));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn marker(tick: u32, steam_id: u64) -> Marker {
        Marker { tick, steam_id, ..Marker::default() }
    }

    #[test]
    fn close_detections_become_one_moment() {
        let markers = [marker(500, 1), marker(100, 1), marker(110, 1), marker(105, 2), marker(400, 1)];
        let found = moments(&markers, 1, 66);
        assert_eq!(found, vec![
            Moment { tick: 100, markers: vec![1, 2] },
            Moment { tick: 400, markers: vec![4] },
            Moment { tick: 500, markers: vec![0] },
        ]);
    }

    #[test]
    fn flagged_players_are_ranked_by_detections() {
        let markers = [marker(1, 7), marker(2, 9), marker(3, 9), marker(4, 0)];
        assert_eq!(flagged_players(&markers), vec![(9, 2), (7, 1)]);
    }

    #[test]
    fn the_aimbot_fov_is_the_biggest_flick_in_aim_detections() {
        let flag = |steam_id: u64, check: &str, data: Value| Marker {
            steam_id,
            check: check.to_string(),
            data,
            ..Marker::default()
        };
        let markers = [
            flag(1, "fidoo/ml_silent_aim", json!({"out_angle": 12.0, "back_angle": 12.97})),
            flag(1, "fidoo/sniperaim", json!({"snap_angle": 1.6, "eye_angle_snap": 1.75})),
            flag(1, "fidoo/ml_aimbot", json!({"target": 5})),
            // Not about aim, or someone else.
            flag(1, "fidoo/nospread", json!({"out_angle": 40.0})),
            flag(2, "fidoo/ml_silent_aim", json!({"out_angle": 30.0})),
        ];
        // The aimbot check reports no angle, so it's measured from the demo.
        let measure = |marker: &Marker| (marker.check == "fidoo/ml_aimbot").then_some(7.5);
        assert_eq!(aimbot_fov(&markers, 1, &measure), Some((12.0, 3)));
        assert_eq!(aimbot_fov(&markers, 2, &measure), Some((30.0, 1)));
        assert_eq!(aimbot_fov(&markers, 3, &measure), None);
        // A turn too big to be a flick is left out.
        let turn = [flag(4, "fidoo/plain_aimbot", json!({"snap_angle": 150.0}))];
        assert_eq!(aimbot_fov(&turn, 4, &measure), None);
    }

    #[test]
    fn checks_have_readable_titles() {
        assert_eq!(check_title("fidoo/ml_silent_aim"), "Silent aim (learned model)");
        assert_eq!(check_title("fidoo/bunnyhop"), "Bunnyhop");
        assert_eq!(check_title("viewangles_180degrees"), "Viewangles 180degrees");
        assert!(what_to_watch("fidoo/ml_silent_aim").contains("aim line"));
        assert!(what_to_watch("fidoo/ml_psilent").contains("projectile"));
        assert!(what_to_watch("fidoo/ml_backtrack").contains("ghosts"));
    }

    #[test]
    fn fields_read_as_words_with_units() {
        assert_eq!(readable_key("flick_miss_hu"), "Flick miss (units)");
        assert_eq!(readable_key("\u{200B}\u{200B}gap_ticks"), "Gap ticks");
        assert_eq!(readable_key("snap_head_radii"), "Snap head (head sizes)");
        assert_eq!(readable_value("back_angle", &json!(12.970000267)), "12.97°");
        assert_eq!(readable_value("distance", &json!(2224.0)), "2,224 units");
        assert_eq!(readable_value("gap_ticks", &json!(1)), "1 tick");
        assert_eq!(readable_value("rewind_ticks", &json!(29.0)), "29 ticks");
        assert_eq!(readable_value("flick_tick_offset", &json!(0.0)), "0 ticks");
        assert_eq!(readable_value("impulse_head_radii", &json!(5.73)), "5.73");
        assert_eq!(readable_value("shot_score", &json!(0.9987)), "100%");
        assert_eq!(readable_value("headshot", &json!(true)), "yes");
        assert_eq!(readable_value("flick_miss_hu", &json!(4.07)), "4.07");
        assert_eq!(readable_value("type", &json!("silent_flick")), "silent flick");
    }

    #[test]
    fn telling_fields_come_first() {
        let data = json!({
            "back_angle": 1.5,
            "weapon": "Sniper Rifle",
            "class": "sniper",
            "server_name": "x",
            "target": 76561198259133340_u64,
            "shot_score": 0.88,
            "path_miss_hu": null,
        });
        let name_of = |id: u64| (id == 76561198259133340).then(|| "Pyro Main".to_string());
        let lines = measured(&data, &name_of);
        assert_eq!(lines, vec![
            ("Class".to_string(), "sniper".to_string()),
            ("Weapon".to_string(), "Sniper Rifle".to_string()),
            ("Target".to_string(), "Pyro Main".to_string()),
            ("This shot's score".to_string(), "88%".to_string()),
            ("Back angle".to_string(), "1.5°".to_string()),
        ]);
    }
}
