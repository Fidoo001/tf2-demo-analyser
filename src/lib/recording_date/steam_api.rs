//! Method 1: item drops looked up with the Steam Web API.
//!
//! An `item_found` event means the Game Coordinator created a new item for that player at that
//! moment of the match, but the event carries no item ID. `IEconItems_440/GetPlayerItems` lists the
//! player's backpack with each item's `original_id` (the ID it was created with, kept through later
//! trades and modifications). The found item is the one with the event's definition index and
//! quality whose `original_id` is the smallest one newer than every item ID seen in the demo; its
//! creation date is the recording date, read off the item ID timeline.
//!
//! Steam has no API that returns an item's creation date, so the timeline is still what turns the
//! ID into a date; the API only finds the exact ID created during the match.
//!
//! The lookups are rate limited and capped per demo. The finder may have traded the item away or
//! have a private backpack, so finders are tried one after another, newest-looking drops first,
//! until one item is found. A 429 response pauses all lookups for a while and the caller falls
//! back to the offline estimate.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{date_of_item_id, estimate_offline, format_date, DateMethod, DemoItemFacts, ItemDrop, RecordingDate};

const API_URL: &str = "https://api.steampowered.com/IEconItems_440/GetPlayerItems/v0001/";
/// Backpack requests per demo at most.
const MAX_LOOKUPS_PER_DEMO: usize = 6;
/// Minimum time between two requests.
const REQUEST_INTERVAL: Duration = Duration::from_millis(1100);
/// Pause after a 429 before the API is tried again.
const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(15 * 60);
/// Backpacks are cached this long (a session usually looks at a handful of demos from one match).
const CACHE_TTL: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone)]
struct BackpackItem {
    /// Current ID: assigned at the last trade/purchase/modification.
    id: u64,
    /// ID the item was created with.
    original_id: u64,
    defindex: u32,
    quality: u32,
    origin: Option<u32>,
}

#[derive(Debug, Clone)]
enum Backpack {
    Items(Vec<BackpackItem>),
    /// Private backpack, deleted account, ...: nothing to find.
    Unavailable,
}

#[derive(Debug)]
enum LookupError {
    RateLimited,
    BadKey,
    Network(String),
}

/// Shared state of the Steam API lookups: pacing, rate limit backoff and a backpack cache. Keep one
/// for the whole program run.
#[derive(Default)]
pub struct SteamApiLimiter {
    last_request: Option<Instant>,
    paused_until: Option<Instant>,
    cache: HashMap<u64, (Instant, Backpack)>,
}

impl SteamApiLimiter {
    pub fn is_paused(&self) -> bool {
        self.paused_until.is_some_and(|t| Instant::now() < t)
    }

    fn backpack(&mut self, key: &str, steam_id: u64) -> Result<Backpack, LookupError> {
        if let Some((at, bp)) = self.cache.get(&steam_id) {
            if at.elapsed() < CACHE_TTL {
                return Ok(bp.clone());
            }
        }
        if self.is_paused() {
            return Err(LookupError::RateLimited);
        }
        if let Some(wait) = self.last_request.and_then(|t| REQUEST_INTERVAL.checked_sub(t.elapsed())) {
            std::thread::sleep(wait);
        }
        self.last_request = Some(Instant::now());
        let response = ureq::get(API_URL)
            .query("key", key)
            .query("steamid", &steam_id.to_string())
            .timeout(Duration::from_secs(15))
            .call();
        let body: serde_json::Value = match response {
            Ok(r) => {
                let text = r.into_string().map_err(|e| LookupError::Network(e.to_string()))?;
                serde_json::from_str(&text).map_err(|e| LookupError::Network(e.to_string()))?
            }
            Err(ureq::Error::Status(429, _)) => {
                self.paused_until = Some(Instant::now() + RATE_LIMIT_COOLDOWN);
                return Err(LookupError::RateLimited);
            }
            Err(ureq::Error::Status(401 | 403, _)) => return Err(LookupError::BadKey),
            // 5xx/404: Steam's item servers are down now and then, treat like a private backpack
            Err(ureq::Error::Status(code, _)) => return Err(LookupError::Network(format!("HTTP {code}"))),
            // ureq's transport errors contain the request URL, i.e. the key
            Err(e) => return Err(LookupError::Network(mask_key(&e.to_string(), key))),
        };
        let result = &body["result"];
        let backpack = if result["status"].as_i64() == Some(1) {
            let items = result["items"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|i| {
                            Some(BackpackItem {
                                id: i["id"].as_u64()?,
                                original_id: i["original_id"].as_u64()?,
                                defindex: i["defindex"].as_u64()? as u32,
                                quality: i["quality"].as_u64().unwrap_or(0) as u32,
                                origin: i["origin"].as_u64().map(|o| o as u32),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            Backpack::Items(items)
        } else {
            Backpack::Unavailable
        };
        self.cache.insert(steam_id, (Instant::now(), backpack.clone()));
        Ok(backpack)
    }
}

/// Removes the API key from a message (error texts may contain the request URL).
fn mask_key(message: &str, key: &str) -> String {
    if key.is_empty() {
        message.to_string()
    } else {
        message.replace(key, "***")
    }
}

/// The backpack `origin` an `item_found` method produces, where it is unambiguous.
fn origin_of_method(method: u8) -> Option<u32> {
    match method {
        0 => Some(0),  // timed drop
        1 => Some(4),  // crafted
        4 => Some(8),  // found in crate (unboxed)
        _ => None,
    }
}

/// Length of a demo tick in seconds (66.67 ticks per second).
const TICK_SECS: f64 = 0.015;
/// Bounds on how fast item IDs are created (2.9-5.4 M/day = 33-62/s since 2024; wide margins for
/// bursts and quiet hours).
const MIN_IDS_PER_SEC: f64 = 10.0;
const MAX_IDS_PER_SEC: f64 = 200.0;
/// IDs the Game Coordinator may hand out between two drops beyond the rate bounds (latency,
/// batching): about ten minutes of IDs.
const ID_SLACK: f64 = 40_000.0;

/// Candidate IDs for the item of `drop`: matching items (definition, quality, origin) with an ID
/// newer than `newer_than`. A dropped, crafted or unboxed item was created then (`original_id`,
/// which survives later trades and modifications); a traded one got a new current `id` then (its
/// `original_id` is old). Duplicates (the same item found several times) all match.
fn drop_candidates(drop: &ItemDrop, items: &[BackpackItem], newer_than: u64) -> Vec<u64> {
    let origin = origin_of_method(drop.method);
    let mut ids: Vec<u64> = items
        .iter()
        .filter(|i| i.defindex == drop.item_def && i.quality == drop.quality as u32)
        .filter(|i| origin.is_none() || i.origin.is_none() || i.origin == origin)
        .flat_map(|i| [i.original_id, i.id])
        .filter(|id| *id > newer_than)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Whether item `y` (found at tick `tick_y`) can have been created in the same match as item `x`
/// (found at `tick_x`): IDs increase with time, by a bounded amount per second.
fn same_match(x: u64, tick_x: u32, y: u64, tick_y: u32) -> bool {
    if x == y {
        return false;
    }
    let dt = (tick_y as f64 - tick_x as f64) * TICK_SECS;
    let diff = y as f64 - x as f64;
    let (lo, hi) = if dt >= 0.0 {
        (dt * MIN_IDS_PER_SEC, dt * MAX_IDS_PER_SEC)
    } else {
        (dt * MAX_IDS_PER_SEC, dt * MIN_IDS_PER_SEC)
    };
    diff >= lo - ID_SLACK && diff <= hi + ID_SLACK
}

/// Candidates of one `item_found` event.
struct DropCandidates<'a> {
    drop: &'a ItemDrop,
    ids: Vec<u64>,
}

#[derive(Debug, PartialEq)]
enum Consensus {
    /// Finds of at least two players agree on one match: the earliest agreeing ID, the index of
    /// its drop and how many players agree.
    Confirmed { id: u64, drop: usize, agreeing: usize },
    /// Several groups of drops agree on different matches (duplicates found together again later).
    Ambiguous(Vec<u64>),
    /// Candidates, but no second drop to confirm any of them.
    Unconfirmed(Vec<u64>),
    None,
}

/// Picks the candidates that belong to one match. A duplicate from another day does not fit the
/// IDs of the other drops of the match, so a candidate is only accepted when a candidate of
/// *another player* agrees with it, and only when exactly one such group exists. One player's finds
/// cannot confirm each other: someone crafting metal makes evenly spaced IDs in every session, so
/// a later session fits just as well as the match (seen in the database: 24 crafted scrap/reclaimed
/// metal "agreeing" on a date 20 months after the match).
fn consensus(drops: &[DropCandidates]) -> Consensus {
    // (support = other players with a fitting candidate, id, drop index) for every candidate
    let mut scored = Vec::new();
    for (a, da) in drops.iter().enumerate() {
        for &x in &da.ids {
            let mut players: Vec<u64> = drops
                .iter()
                .filter(|db| db.drop.steam_id != da.drop.steam_id)
                .filter(|db| db.ids.iter().any(|&y| same_match(x, da.drop.tick, y, db.drop.tick)))
                .map(|db| db.drop.steam_id)
                .collect();
            players.sort_unstable();
            players.dedup();
            scored.push((players.len(), x, a));
        }
    }
    let Some(best) = scored.iter().map(|s| s.0).max() else { return Consensus::None };
    if best == 0 {
        return Consensus::Unconfirmed(scored.iter().map(|s| s.1).collect());
    }
    let tops: Vec<&(usize, u64, usize)> = scored.iter().filter(|s| s.0 == best).collect();
    // All best-supported candidates must belong to one match
    let anchor = **tops.iter().min_by_key(|s| s.1).unwrap();
    let others: Vec<u64> = tops
        .iter()
        .filter(|s| s.1 != anchor.1 && !same_match(anchor.1, drops[anchor.2].drop.tick, s.1, drops[s.2].drop.tick))
        .map(|s| s.1)
        .collect();
    if !others.is_empty() {
        let mut ids = vec![anchor.1];
        ids.extend(others);
        return Consensus::Ambiguous(ids);
    }
    Consensus::Confirmed { id: anchor.1, drop: anchor.2, agreeing: best + 1 }
}

/// The best estimate, trying item drops through the Steam Web API first. Without drops, without a
/// key or when the API fails, this is the offline estimate (with `api_note` saying why, if the API
/// was tried).
///
/// The trade hold window is deliberately not used to pick or reject drops: the window assumes the
/// newest hold was still running, and when it was not, a filter on it would reject the true drop
/// and could accept a duplicate inside the wrong window, confirming the wrong date.
pub fn estimate_with_steam_api(facts: &DemoItemFacts, key: &str, limiter: &mut SteamApiLimiter) -> Option<RecordingDate> {
    let offline = estimate_offline(facts);
    let key = key.trim();
    if key.is_empty() || facts.drops.is_empty() {
        return offline;
    }
    let with_note = |note: String| {
        offline.clone().map(|mut d| {
            d.api_note = Some(note);
            d
        })
    };

    // One lookup per finder; a finder with several drops is tried first (more chances to match).
    let mut finders: Vec<u64> = Vec::new();
    for d in &facts.drops {
        if !finders.contains(&d.steam_id) {
            finders.push(d.steam_id);
        }
    }
    finders.sort_by_key(|sid| std::cmp::Reverse(facts.drops.iter().filter(|d| d.steam_id == *sid).count()));

    let mut candidates: Vec<DropCandidates> = Vec::new();
    let mut tried = 0;
    let mut network_error = None;
    let mut api_stopped = None;
    for sid in finders {
        if tried >= MAX_LOOKUPS_PER_DEMO {
            break;
        }
        tried += 1;
        let backpack = match limiter.backpack(key, sid) {
            Ok(bp) => bp,
            Err(LookupError::RateLimited) => {
                api_stopped = Some("Steam API rate limited");
                break;
            }
            Err(LookupError::BadKey) => {
                api_stopped = Some("Steam API key was rejected");
                break;
            }
            Err(LookupError::Network(e)) => {
                network_error = Some(e);
                continue;
            }
        };
        let Backpack::Items(items) = backpack else { continue };
        for drop in facts.drops.iter().filter(|d| d.steam_id == sid) {
            // The drop is newer than every item that existed before it in the match
            let bound = facts.newest_item_id_before(drop.tick).unwrap_or(0);
            let ids = drop_candidates(drop, &items, bound);
            if !ids.is_empty() {
                candidates.push(DropCandidates { drop, ids });
            }
        }
        // Three players agreeing is enough; otherwise look further for confirmation or ambiguity
        if matches!(consensus(&candidates), Consensus::Confirmed { agreeing, .. } if agreeing >= 3) {
            break;
        }
    }

    let dates = |ids: &[u64]| {
        let mut ds: Vec<String> = ids.iter().map(|id| format_date(date_of_item_id(*id).timestamp)).collect();
        ds.sort();
        ds.dedup();
        ds.join(", ")
    };
    match consensus(&candidates) {
        Consensus::Confirmed { id, drop, agreeing } => {
            let drop = candidates[drop].drop;
            let date = date_of_item_id(id);
            Some(RecordingDate {
                method: DateMethod::ItemDrop,
                estimate: date.timestamp,
                earliest: date.timestamp - date.uncertainty_secs,
                latest: Some(date.timestamp + date.uncertainty_secs),
                detail: format!(
                    "Finds of {agreeing} players during the match agree: {} found item #{id} (def {}), created around {}{}",
                    drop.player_name,
                    drop.item_def,
                    format_date(date.timestamp),
                    if date.extrapolated { " (extrapolated)" } else { "" }
                ),
                api_note: api_stopped.map(|n| format!("{n} before all finders were checked")),
            })
        }
        Consensus::Ambiguous(ids) => with_note(format!(
            "Items found during the match fit several dates ({}), duplicates cannot be told apart; used the offline estimate",
            dates(&ids)
        )),
        Consensus::Unconfirmed(ids) => with_note(format!(
            "{} candidate item{} for the match's drops ({}), but no other player's find confirms one; used the offline estimate",
            ids.len(),
            if ids.len() == 1 { "" } else { "s" },
            dates(&ids)
        )),
        Consensus::None => with_note(match (api_stopped, network_error) {
            (Some(n), _) => format!("{n}, used the offline estimate"),
            (None, Some(e)) => format!("Steam API lookup failed ({e}), used the offline estimate"),
            (None, None) => "No item found during the match is still in the finders' backpacks".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drop_at(def: u32, quality: u8, method: u8, tick: u32) -> ItemDrop {
        found_by(def, quality, method, tick, 1)
    }

    fn found_by(def: u32, quality: u8, method: u8, tick: u32, steam_id: u64) -> ItemDrop {
        ItemDrop {
            tick,
            steam_id,
            account_id: 1,
            player_name: String::new(),
            item_def: def,
            quality,
            method,
            is_strange: false,
            is_unusual: false,
        }
    }

    fn drop(def: u32, quality: u8, method: u8) -> ItemDrop {
        drop_at(def, quality, method, 0)
    }

    fn item(original_id: u64, defindex: u32, quality: u32, origin: u32) -> BackpackItem {
        BackpackItem { id: original_id, original_id, defindex, quality, origin: Some(origin) }
    }

    #[test]
    fn candidates_are_matching_items_newer_than_the_bound() {
        let items = [
            item(100, 810, 6, 0),  // older than the bound: dropped before the match
            item(205, 810, 6, 0),
            item(300, 810, 6, 0),  // duplicate found later
            item(201, 810, 6, 2),  // purchased, wrong origin
            item(202, 811, 6, 0),  // other item
        ];
        assert_eq!(drop_candidates(&drop(810, 6, 0), &items, 200), vec![205, 300]);
        assert!(drop_candidates(&drop(810, 6, 0), &items, 400).is_empty());
        // unknown method: origin not checked
        assert_eq!(drop_candidates(&drop(810, 6, 9), &items, 200), vec![201, 205, 300]);
    }

    #[test]
    fn traded_item_matches_on_its_current_id() {
        let mut traded = item(50, 30000, 6, 0);
        traded.id = 250;
        let mut modified_drop = item(260, 30001, 6, 0);
        modified_drop.id = 900;
        let items = [traded, modified_drop];
        assert_eq!(drop_candidates(&drop(30000, 6, 2), &items, 200), vec![250]);
        assert_eq!(drop_candidates(&drop(30001, 6, 0), &items, 200), vec![260, 900]);
    }

    #[test]
    fn errors_do_not_contain_the_key() {
        let msg = mask_key("https://api.steampowered.com/x?key=ABC123&steamid=1: Connection Failed", "ABC123");
        assert!(!msg.contains("ABC123"));
    }

    #[test]
    fn ids_must_follow_the_drop_order() {
        // 600 s apart: 6k..120k IDs later (plus slack) fit, earlier ones do not
        assert!(same_match(5_000_000, 0, 5_030_000, 40_000));
        assert!(!same_match(5_000_000, 0, 4_900_000, 40_000));
        assert!(!same_match(5_000_000, 0, 5_500_000, 40_000));
        assert!(!same_match(5_000_000, 0, 5_000_000, 40_000));
    }

    #[test]
    fn duplicates_are_resolved_by_a_second_drop() {
        let a = drop_at(810, 6, 0, 1_000);
        let b = found_by(237, 6, 0, 40_000, 2);
        // a's finder has three of the item (an older one past the bound, the match's, a later one)
        let drops = [
            DropCandidates { drop: &a, ids: vec![1_000_000, 5_000_000, 9_000_000] },
            DropCandidates { drop: &b, ids: vec![5_023_400, 20_000_000] },
        ];
        assert_eq!(consensus(&drops), Consensus::Confirmed { id: 5_000_000, drop: 0, agreeing: 2 });
    }

    #[test]
    fn two_fitting_groups_are_ambiguous() {
        let a = drop_at(810, 6, 0, 1_000);
        let b = found_by(237, 6, 0, 40_000, 2);
        // both items were found together again in a later match
        let drops = [
            DropCandidates { drop: &a, ids: vec![5_000_000, 9_000_000] },
            DropCandidates { drop: &b, ids: vec![5_023_400, 9_020_000] },
        ];
        assert!(matches!(consensus(&drops), Consensus::Ambiguous(_)));
    }

    #[test]
    fn one_players_finds_do_not_confirm_each_other() {
        // a metal crafting session: evenly spaced IDs from one player fit any session
        let a = drop_at(5000, 6, 1, 1_000);
        let b = drop_at(5000, 6, 1, 1_600);
        let drops = [
            DropCandidates { drop: &a, ids: vec![9_000_000] },
            DropCandidates { drop: &b, ids: vec![9_000_300] },
        ];
        assert!(matches!(consensus(&drops), Consensus::Unconfirmed(_)));
    }

    #[test]
    fn a_single_drop_is_not_confirmed() {
        let a = drop_at(810, 6, 0, 1_000);
        let drops = [DropCandidates { drop: &a, ids: vec![5_000_000] }];
        assert_eq!(consensus(&drops), Consensus::Unconfirmed(vec![5_000_000]));
        assert_eq!(consensus(&[]), Consensus::None);
    }
}
