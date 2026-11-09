//! A one-page HTML report about one player in a demo: their detections, how they
//! did, and their ping over the match. It opens in any browser and can be shared
//! as proof.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use super::recording::Recording;
use super::view::Marker;

/// Ping is sampled about this often for the chart, in seconds.
const PING_SAMPLE_SECONDS: f64 = 2.0;
const CHART_WIDTH: f64 = 640.0;
const CHART_HEIGHT: f64 = 140.0;

/// Makes text safe to put in HTML. Player names can contain anything.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// What the report says about the player, gathered from the recording.
#[derive(Debug, Default, PartialEq)]
pub struct Summary {
    pub kills: usize,
    pub deaths: usize,
    pub assists: usize,
    pub headshots: usize,
    pub backstabs: usize,
    pub damage_dealt: u64,
    pub damage_taken: u64,
    pub shots: usize,
    /// Bullet shots that hurt someone in the same tick or the next.
    pub shots_that_hit: usize,
    /// (seconds into the demo, ping) samples.
    pub ping: Vec<(f64, u16)>,
    pub classes: Vec<&'static str>,
}

const CLASS_NAMES: [&str; 10] = ["", "Scout", "Sniper", "Soldier", "Demoman", "Medic", "Heavy", "Pyro", "Spy", "Engineer"];

/// Adds up a player's match. Only ticks where the slot belonged to `steam_id`
/// count, since slots are reused when people leave.
pub fn summarize(recording: &Recording, slot: usize, steam_id: u64) -> Summary {
    let mine = |tick: u32| recording.steam_id_at(slot, f64::from(tick)) == steam_id;
    let name_at = |tick: u32| recording.name_at(slot, f64::from(tick)).to_string();
    let mut summary = Summary::default();

    for kill in &recording.kills {
        if kill.feigned {
            continue;
        }
        if kill.attacker == Some(slot) && kill.victim != Some(slot) && mine(kill.tick) {
            summary.kills += 1;
            summary.headshots += usize::from(kill.headshot);
            summary.backstabs += usize::from(kill.backstab);
        }
        if kill.victim == Some(slot) && mine(kill.tick) {
            summary.deaths += 1;
        }
        if !kill.assister_name.is_empty() && kill.assister_name == name_at(kill.tick) && mine(kill.tick) {
            summary.assists += 1;
        }
    }
    for hurt in &recording.hurts {
        if hurt.attacker == Some(slot) && mine(hurt.tick) {
            summary.damage_dealt += u64::from(hurt.damage);
        }
        if hurt.victim == Some(slot) && mine(hurt.tick) {
            summary.damage_taken += u64::from(hurt.damage);
        }
    }
    let my_hits: Vec<u32> = recording
        .hurts
        .iter()
        .filter(|hurt| hurt.attacker == Some(slot) && mine(hurt.tick))
        .map(|hurt| hurt.tick)
        .collect();
    for shot in recording.shots.iter().filter(|shot| shot.slot == slot && mine(shot.tick)) {
        summary.shots += 1;
        let start = my_hits.partition_point(|&tick| tick < shot.tick);
        if my_hits.get(start).is_some_and(|&tick| tick <= shot.tick + 1) {
            summary.shots_that_hit += 1;
        }
    }

    let interval = f64::from(recording.tick_interval);
    let step = (PING_SAMPLE_SECONDS / interval).max(1.0) as u32;
    let first = recording.first_tick();
    let mut next_sample = first;
    let mut classes = BTreeMap::new();
    for (frame, &tick) in recording.ticks.iter().enumerate() {
        if !mine(tick) {
            continue;
        }
        let player = recording.player(frame, slot);
        if player.is_alive() {
            *classes.entry(usize::from(player.class)).or_insert(0usize) += 1;
        }
        if tick >= next_sample && player.ping > 0 {
            summary.ping.push((recording.seconds_between(f64::from(first), f64::from(tick)), player.ping));
            next_sample = tick + step;
        }
    }
    let mut played: Vec<(usize, usize)> = classes.into_iter().collect();
    played.sort_by_key(|&(_, frames)| std::cmp::Reverse(frames));
    summary.classes = played
        .into_iter()
        .filter_map(|(class, _)| CLASS_NAMES.get(class).copied().filter(|name| !name.is_empty()))
        .collect();
    summary
}

/// An inline SVG line chart of ping over the match.
fn ping_chart(ping: &[(f64, u16)]) -> String {
    if ping.len() < 2 {
        return "<p class=\"muted\">Not enough ping data in the demo.</p>".to_string();
    }
    let end = ping.last().map_or(1.0, |&(t, _)| t.max(1.0));
    let top = ping.iter().map(|&(_, p)| f64::from(p)).fold(100.0, f64::max) * 1.1;
    let points: Vec<String> = ping
        .iter()
        .map(|&(t, p)| format!("{:.1},{:.1}", t / end * CHART_WIDTH, CHART_HEIGHT - f64::from(p) / top * CHART_HEIGHT))
        .collect();
    let mut svg = format!(
        "<svg viewBox=\"0 0 {CHART_WIDTH} {CHART_HEIGHT}\" class=\"chart\" role=\"img\" aria-label=\"Ping over the match\">"
    );
    for line in [50.0, 100.0, 150.0, 200.0] {
        if line < top {
            let y = CHART_HEIGHT - line / top * CHART_HEIGHT;
            let _ = write!(
                svg,
                "<line x1=\"0\" x2=\"{CHART_WIDTH}\" y1=\"{y:.1}\" y2=\"{y:.1}\" class=\"grid\"/><text x=\"4\" y=\"{:.1}\" class=\"axis\">{line} ms</text>",
                y - 3.0
            );
        }
    }
    let _ = write!(svg, "<polyline points=\"{}\" class=\"line\"/></svg>", points.join(" "));
    svg
}

/// Writes the report as HTML.
pub fn render(
    recording: &Recording,
    markers: &[Marker],
    slot: usize,
    steam_id: u64,
    demo: &Path,
    tick: f64,
    aimbot_fov: Option<(f32, usize)>,
) -> String {
    let summary = summarize(recording, slot, steam_id);
    let name = escape(recording.name_at(slot, tick));
    let demo_name = escape(&demo.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    let first = f64::from(recording.first_tick());
    let mine: Vec<&Marker> = markers.iter().filter(|m| m.steam_id == steam_id && steam_id != 0).collect();
    let mut by_check: BTreeMap<String, usize> = BTreeMap::new();
    let check_of = |marker: &Marker| super::review::check_of(marker).to_string();
    for marker in &mine {
        *by_check.entry(check_of(marker)).or_default() += 1;
    }
    let accuracy = if summary.shots > 0 {
        format!("{:.0}%", summary.shots_that_hit as f64 * 100.0 / summary.shots as f64)
    } else {
        "-".to_string()
    };
    let (ping_min, ping_avg, ping_max) = if summary.ping.is_empty() {
        ("-".to_string(), "-".to_string(), "-".to_string())
    } else {
        let values: Vec<u16> = summary.ping.iter().map(|&(_, p)| p).collect();
        let average = values.iter().map(|&p| f64::from(p)).sum::<f64>() / values.len() as f64;
        (
            values.iter().min().unwrap_or(&0).to_string(),
            format!("{average:.0}"),
            values.iter().max().unwrap_or(&0).to_string(),
        )
    };

    let mut html = String::new();
    let _ = write!(
        html,
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} - demo report</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --text:#1d2329; --muted:#66707a; --line:#d9dee3; --accent:#c8551b; --plot:#2f7fd1; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15191d; --card:#1e2328; --text:#e7eaed; --muted:#9aa4ad; --line:#323a42; --accent:#f08a4b; --plot:#6bb0ff; }} }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ margin:0 0 4px; font-size:26px; }} h2 {{ font-size:17px; margin:28px 0 10px; }}
.muted {{ color:var(--muted); }} a {{ color:var(--plot); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.card b {{ display:block; font-size:22px; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-weight:600; }}
.flag {{ color:var(--accent); font-weight:600; }}
.chart {{ width:100%; height:auto; background:var(--card); border:1px solid var(--line); border-radius:8px; }}
.chart .grid {{ stroke:var(--line); }} .chart .axis {{ fill:var(--muted); font-size:10px; }}
.chart .line {{ fill:none; stroke:var(--plot); stroke-width:1.5; }}
</style></head><body><main>
<h1>{name}</h1>
<p class="muted">SteamID64 {steam_id} &middot; <a href="https://steamcommunity.com/profiles/{steam_id}">Steam profile</a> &middot; demo {demo_name} &middot; map {map} &middot; recorded by {recorder}</p>
<p class="muted">Played: {classes}</p>
<h2>Detections</h2>
"#,
        map = escape(&recording.map),
        recorder = escape(&recording.recorded_by),
        classes = if summary.classes.is_empty() { "-".to_string() } else { summary.classes.join(", ") },
    );
    if mine.is_empty() {
        html.push_str("<p class=\"muted\">No detections for this player in this demo.</p>\n");
    } else {
        html.push_str("<div class=\"cards\">");
        for (check, count) in &by_check {
            let _ = write!(html, "<div class=\"card\"><b class=\"flag\">{count}</b>{}</div>", escape(check));
        }
        html.push_str("</div>\n<p></p><table><tr><th>Time</th><th>Tick</th><th>Check</th></tr>\n");
        for marker in &mine {
            let seconds = recording.seconds_between(first, f64::from(marker.tick));
            let _ = writeln!(
                html,
                "<tr><td>{}</td><td>{}</td><td class=\"flag\">{}</td></tr>",
                clock(seconds),
                marker.tick,
                escape(&check_of(marker))
            );
        }
        html.push_str("</table>\n");
        if let Some((degrees, flicks)) = aimbot_fov {
            let _ = write!(
                html,
                "<h2>Aimbot FOV</h2>\n<div class=\"cards\"><div class=\"card\"><b class=\"flag\">&ge; {degrees:.1}&deg;</b>estimated aimbot FOV</div></div>\n\
                 <p class=\"muted\">The biggest flick onto a target in {flicks} aim detection{}. An aimbot only grabs targets \
                 within this angle of the crosshair, so its setting is at least this big.</p>\n",
                if flicks == 1 { "" } else { "s" }
            );
        }
    }
    let _ = write!(
        html,
        r#"<h2>Match</h2>
<div class="cards">
<div class="card"><b>{kills}</b>kills</div><div class="card"><b>{deaths}</b>deaths</div><div class="card"><b>{assists}</b>assists</div>
<div class="card"><b>{headshots}</b>headshot kills</div><div class="card"><b>{backstabs}</b>backstabs</div>
<div class="card"><b>{dealt}</b>damage dealt</div><div class="card"><b>{taken}</b>damage taken</div>
<div class="card"><b>{accuracy}</b>bullet accuracy <span class="muted">({hit} of {shots} shots)</span></div>
</div>
<h2>Ping</h2>
<div class="cards"><div class="card"><b>{ping_min}</b>lowest ms</div><div class="card"><b>{ping_avg}</b>average ms</div><div class="card"><b>{ping_max}</b>highest ms</div></div>
<p></p>{chart}
<p class="muted">Ping is what the server measured, as on TF2's scoreboard. Bullet accuracy counts a shot as a hit when the player hurt someone in the same tick or the next, so pellets and splash make it approximate.</p>
<p class="muted">Made by the Demo Analysis 3D player.</p>
</main></body></html>
"#,
        kills = summary.kills,
        deaths = summary.deaths,
        assists = summary.assists,
        headshots = summary.headshots,
        backstabs = summary.backstabs,
        dealt = summary.damage_dealt,
        taken = summary.damage_taken,
        hit = summary.shots_that_hit,
        shots = summary.shots,
        chart = ping_chart(&summary.ping),
    );
    html
}

/// Where reports are saved: a "reports" folder next to the program.
pub fn reports_folder() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("reports")
}

/// A file name made only of safe characters.
pub fn safe_file_name(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "player".to_string()
    } else {
        trimmed.chars().take(40).collect()
    }
}

/// Opens a file or web address with whatever the system uses for it.
pub fn open_externally(target: &str) {
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("explorer").arg(target).spawn();
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(target).spawn();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let result = std::process::Command::new("xdg-open").arg(target).spawn();
    let _ = result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::recording::{Hurt, Kill, PlayerFrame, Shot, Slot};

    #[test]
    fn names_are_escaped() {
        assert_eq!(escape("<b>\"x\" & 'y'</b>"), "&lt;b&gt;&quot;x&quot; &amp; &#39;y&#39;&lt;/b&gt;");
    }

    #[test]
    fn file_names_are_made_safe() {
        assert_eq!(safe_file_name("a/b:c?"), "a_b_c");
        assert_eq!(safe_file_name("???"), "player");
    }

    #[test]
    fn a_summary_counts_only_this_player() {
        let recording = Recording {
            ticks: vec![10],
            slots: vec![Slot::default(), Slot::default()],
            slot_count: 2,
            players: vec![PlayerFrame::default(); 2],
            kills: vec![
                Kill { tick: 10, attacker: Some(0), victim: Some(1), headshot: true, ..Kill::default() },
                Kill { tick: 10, attacker: Some(1), victim: Some(0), ..Kill::default() },
            ],
            hurts: vec![
                Hurt { tick: 20, attacker: Some(0), victim: Some(1), damage: 50, ..Hurt::default() },
                Hurt { tick: 30, attacker: Some(1), victim: Some(0), damage: 20, ..Hurt::default() },
            ],
            shots: vec![
                Shot { tick: 20, slot: 0, ..Shot::default() },
                Shot { tick: 40, slot: 0, ..Shot::default() },
            ],
            ..Recording::default()
        };
        // With no slot names recorded, every slot reads as Steam ID 0.
        let summary = summarize(&recording, 0, 0);
        assert_eq!((summary.kills, summary.deaths, summary.headshots), (1, 1, 1));
        assert_eq!((summary.damage_dealt, summary.damage_taken), (50, 20));
        assert_eq!((summary.shots, summary.shots_that_hit), (2, 1));
    }
}
