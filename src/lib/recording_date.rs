//! Recording date estimation for demos.
//!
//! Demos store no wall-clock date, but the econ items networked in them do carry time information:
//!
//! - **Item IDs** are assigned by the Game Coordinator from one global, monotonically increasing
//!   counter whenever an item is created *or changes owner* (a purchase gives the item a new ID).
//!   The newest item ID in a demo therefore bounds the recording date from below.
//! - **Trade holds**: a purchased item (Steam Community Market) gets attribute 211
//!   (`tradable_after_date`), the UNIX time its 7-day trade hold ends (rounded to midnight Pacific).
//!   The attribute stays on the item after the hold has expired, so only the newest one is useful.
//! - **Item drops**: `item_found` events (timed drops, crafts, unboxes, ...) mark a new item ID being
//!   created at that exact moment of the match. The event carries no ID, but the Steam Web API can
//!   find the item in the finder's backpack (its `original_id`), which converts to a date through
//!   the item ID timeline.
//!
//! The item ID → date timeline (`ITEM_ID_TIMELINE`) is fitted offline from trade-held items in the
//! demo database: a purchased item's ID was assigned when it was bought, 7–8 days before its
//! `tradable_after_date`. Regenerate it with `scripts/fit_item_id_timeline.py`.
//!
//! This module is UI independent: [`collect_item_facts`] parses a demo, [`estimate_offline`] and
//! [`estimate_with_steam_api`] turn the facts into a [`RecordingDate`].

use std::collections::HashMap;

use anyhow::Result;
use bitbuffer::BitRead;
use serde::Serialize;
use steamid_ng::SteamID;
use tf_demo_parser::demo::data::userinfo::UserInfo;
use tf_demo_parser::demo::data::DemoTick;
use tf_demo_parser::demo::gameevent_gen::GameEvent;
use tf_demo_parser::demo::header::Header;
use tf_demo_parser::demo::message::packetentities::UpdateType;
use tf_demo_parser::demo::message::Message;
use tf_demo_parser::demo::packet::stringtable::StringTableEntry;
use tf_demo_parser::demo::packet::Packet;
use tf_demo_parser::demo::parser::handler::{DemoHandler, MessageHandler};
use tf_demo_parser::demo::sendprop::SendPropValue;
use tf_demo_parser::{Demo, MessageType, Parse, ParserState};

pub use timeline::{date_of_item_id, IdDate, TIMELINE_LAST_DATE};

mod timeline;
mod timeline_data;

/// Attribute 211, `tradable_after_date`.
const ATTR_TRADABLE_AFTER: i64 = 211;
/// The trade hold length. Steam ends holds at midnight Pacific, so the item was acquired up to a day
/// more before the stored date.
pub const TRADE_HOLD_SECS: i64 = 7 * 86400;
const DAY: i64 = 86400;
/// Item IDs at or above this are not real Game Coordinator IDs (stock items use 0xFFFFFFFF_xxxxxxxx).
const MAX_REAL_ITEM_ID: u64 = 1 << 40;
/// Sanity range for `tradable_after_date` values (2011-07 .. 2038).
const MIN_VALID_TS: i64 = 1_310_000_000;

/// One econ item seen in the demo.
#[derive(Debug, Clone, Serialize)]
pub struct ItemFact {
    pub item_id: u64,
    pub def: u32,
    pub account_id: u32,
    pub quality: u32,
    /// `tradable_after_date` (attribute 211), if the item has it.
    pub tradable_after: Option<i64>,
    pub class: String,
    /// Demo tick the item was first seen: it existed by then.
    pub first_tick: u32,
}

/// An `item_found` event: an item was created for a player during the match.
#[derive(Debug, Clone, Serialize)]
pub struct ItemDrop {
    pub tick: u32,
    pub steam_id: u64,
    pub account_id: u32,
    pub player_name: String,
    pub item_def: u32,
    pub quality: u8,
    /// `item_found` method: 0 = timed drop, 1 = crafted, 2 = traded, 3 = purchased, 4 = unboxed, ...
    pub method: u8,
    pub is_strange: bool,
    pub is_unusual: bool,
}

/// Everything about a demo's items that the date estimate uses.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DemoItemFacts {
    /// Unique items by item ID.
    pub items: Vec<ItemFact>,
    pub drops: Vec<ItemDrop>,
    pub map: String,
    pub duration_secs: f32,
}

impl DemoItemFacts {
    /// The newest real item ID in the demo. IDs that would date the item after now are corrupt
    /// (seen twice in 5399 demos: a sent low word combined with another item's high word) and are
    /// ignored, since a demo cannot be recorded in the future.
    pub fn newest_item_id(&self) -> Option<u64> {
        self.items.iter().map(|i| i.item_id).filter(|id| plausible_item_id(*id)).max()
    }

    /// The newest real item ID among items seen before `tick`: an item created at `tick` (a drop)
    /// must be newer. Items first seen later may have been created after it.
    pub fn newest_item_id_before(&self, tick: u32) -> Option<u64> {
        self.items.iter().filter(|i| i.first_tick < tick).map(|i| i.item_id).filter(|id| plausible_item_id(*id)).max()
    }

    /// The newest `tradable_after_date` in the demo.
    pub fn newest_trade_hold(&self) -> Option<&ItemFact> {
        self.items.iter().filter(|i| i.tradable_after.is_some()).max_by_key(|i| i.tradable_after)
    }
}

/// Whether an item ID can exist yet: its earliest possible creation date is not after now.
fn plausible_item_id(id: u64) -> bool {
    let date = date_of_item_id(id);
    date.timestamp - date.uncertainty_secs <= chrono::Utc::now().timestamp()
}

/// Which method produced a [`RecordingDate`], from the most to the least precise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DateMethod {
    /// An item found during the match, looked up in the finder's backpack.
    ItemDrop,
    /// The newest trade hold (7-day window).
    TradeHold,
    /// The newest item ID converted to a date (a lower bound).
    NewestItemId,
    /// The newest item ID, extrapolated past the end of the timeline.
    Extrapolated,
}

impl DateMethod {
    pub fn label(self) -> &'static str {
        match self {
            DateMethod::ItemDrop => "item drop",
            DateMethod::TradeHold => "trade hold",
            DateMethod::NewestItemId => "newest item ID",
            DateMethod::Extrapolated => "item ID extrapolation",
        }
    }
}

/// The estimated recording date of a demo. All times are UNIX seconds (UTC).
#[derive(Debug, Clone, Serialize)]
pub struct RecordingDate {
    pub method: DateMethod,
    /// Best guess.
    pub estimate: i64,
    /// The demo was recorded no earlier than this.
    pub earliest: i64,
    /// The demo was recorded no later than this, when known.
    pub latest: Option<i64>,
    /// One line explaining the basis, for a tooltip.
    pub detail: String,
    /// Set when a Steam API lookup was wanted but did not succeed (e.g. rate limited).
    pub api_note: Option<String>,
}

impl RecordingDate {
    /// Half width of the uncertainty window in days, when it is bounded.
    pub fn precision_days(&self) -> Option<f64> {
        self.latest.map(|l| (l - self.earliest) as f64 / 2.0 / DAY as f64)
    }
}

pub fn format_date(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts.to_string())
}

pub fn format_date_time(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| ts.to_string())
}

// ---------------------------------------------------------------------------------------------
// Demo parsing
// ---------------------------------------------------------------------------------------------

/// Flattened prop indices of an econ item server class.
#[derive(Default, Clone)]
struct ItemClassLayout {
    is_item: bool,
    class: String,
    def: Option<u32>,
    id_high: Option<u32>,
    id_low: Option<u32>,
    account: Option<u32>,
    quality: Option<u32>,
    /// Every attribute slot, in order. All slots share one prop identifier, so they can only be
    /// told apart by their flattened index.
    attr_slots: Vec<AttrSlot>,
}

/// One slot of an attribute vector. An item networks two vectors (`m_AttributeList` and
/// `m_NetworkedDynamicAttributesForDemos`), each led by an unnamed length prop. Slots past the
/// length are not sent, so they keep whatever the baseline had: another item's attributes.
#[derive(Clone, Copy)]
struct AttrSlot {
    def: u32,
    raw: u32,
    /// The length prop of this slot's vector.
    count: Option<u32>,
    /// Position in its vector.
    position: usize,
}

/// Where an entity's prop value came from. Props equal to the baseline are not sent, so a value is
/// either sent for this entity or inherited from the instance or static baseline of its class.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Static,
    Instance,
    Sent,
}

#[derive(Default)]
struct FactCollector {
    layouts: HashMap<usize, ItemClassLayout>,
    /// Live item entities: entity index -> (server class, flattened prop index -> (value, layer)).
    entities: HashMap<u32, (usize, HashMap<u32, (i64, Layer)>)>,
    items: HashMap<u64, ItemFact>,
    users: HashMap<u32, UserInfo>,
    drops: Vec<ItemDrop>,
}

impl FactCollector {
    fn layout(&mut self, class: usize, state: &ParserState) -> &ItemClassLayout {
        self.layouts.entry(class).or_insert_with(|| {
            let mut layout = ItemClassLayout::default();
            let Some(table) = state.send_tables.get(class) else { return layout };
            layout.class = state.server_classes.get(class).map(|c| c.name.to_string()).unwrap_or_default();
            let mut pending_def = None;
            let mut last_unnamed = None;
            let mut vector_count: Option<u32> = None;
            let mut position = 0;
            for (i, prop) in table.flattened_props.iter().enumerate() {
                let i = i as u32;
                let Some((_, name)) = prop.identifier.names() else {
                    last_unnamed = Some(i);
                    continue;
                };
                match name.as_str() {
                    "m_iItemDefinitionIndex" => layout.def = Some(i),
                    "m_iItemIDHigh" => layout.id_high = Some(i),
                    "m_iItemIDLow" => layout.id_low = Some(i),
                    "m_iAccountID" => layout.account = Some(i),
                    "m_iEntityQuality" => layout.quality = Some(i),
                    "m_iAttributeDefinitionIndex" => {
                        // a slot right after an unnamed prop starts a new vector
                        if last_unnamed == Some(i.wrapping_sub(1)) {
                            vector_count = last_unnamed;
                            position = 0;
                        }
                        pending_def = Some(i);
                    }
                    "m_iRawValue32" => {
                        if let Some(def) = pending_def.take() {
                            layout.attr_slots.push(AttrSlot { def, raw: i, count: vector_count, position });
                            position += 1;
                        }
                    }
                    _ => {}
                }
            }
            // Dropped weapons are skipped: when one re-enters the PVS its attribute slots come from the
            // class's instance baseline, i.e. from another dropped weapon (foreign 211 dates). The
            // weapon is still read from its owner's weapon entity.
            layout.is_item = layout.id_low.is_some() && layout.def.is_some() && layout.class != "CTFDroppedWeapon";
            layout
        })
    }

    fn record_item(&mut self, entity: u32, tick: u32) {
        let Some((class, values)) = self.entities.get(&entity) else { return };
        let Some(layout) = self.layouts.get(class) else { return };
        let get = |i: Option<u32>| i.and_then(|i| values.get(&i)).map(|v| v.0).unwrap_or(0);
        let high = get(layout.id_high) as u64 & 0xFFFF_FFFF;
        let low = get(layout.id_low) as u64 & 0xFFFF_FFFF;
        let item_id = (high << 32) | low;
        let def = get(layout.def) as u32;
        if item_id == 0 || item_id >= MAX_REAL_ITEM_ID || def == 0 {
            return;
        }
        // Only slots inside their vector's length belong to this item; without a known length no
        // slot is trusted (they may all be baseline leftovers)
        let in_use = |slot: &&AttrSlot| {
            slot.count.and_then(|c| values.get(&c)).is_some_and(|len| (slot.position as i64) < len.0)
        };
        // A baseline describes one item. Its values are this item's only if the item ID came from
        // that baseline too; when the ID was sent, the baseline was another item and an inherited
        // date is that item's (the parser does not always pick the baseline the server delta-encoded
        // against, so this happens in practice: ~0.3 % of items showed a foreign date).
        let id_layer = layout.id_low.and_then(|i| values.get(&i)).map(|v| v.1).unwrap_or(Layer::Static);
        let tradable_after = layout
            .attr_slots
            .iter()
            .filter(in_use)
            .find(|slot| values.get(&slot.def).map(|v| v.0) == Some(ATTR_TRADABLE_AFTER))
            .and_then(|slot| values.get(&slot.raw).copied())
            .filter(|(_, layer)| *layer == Layer::Sent || *layer >= id_layer)
            .map(|(v, _)| v as u32 as i64)
            .filter(|v| *v >= MIN_VALID_TS);
        let fact = self.items.entry(item_id).or_insert_with(|| ItemFact {
            item_id,
            def,
            account_id: 0,
            quality: 0,
            tradable_after: None,
            class: layout.class.clone(),
            first_tick: tick,
        });
        fact.def = def;
        let account = get(layout.account) as u32;
        if account != 0 {
            fact.account_id = account;
        }
        fact.quality = get(layout.quality) as u32;
        if tradable_after.is_some() {
            fact.tradable_after = tradable_after;
        }
    }

    fn user_by_entity(&self, entity: u32) -> Option<(u64, u32, String)> {
        let user = self.users.values().find(|u| u32::from(u.entity_id) == entity)?;
        let sid = SteamID::from_steam3(&user.player_info.steam_id).ok()?;
        Some((u64::from(sid), sid.account_id(), user.player_info.name.clone()))
    }
}

impl MessageHandler for FactCollector {
    type Output = Self;

    fn does_handle(message_type: MessageType) -> bool {
        matches!(message_type, MessageType::PacketEntities | MessageType::GameEvent)
    }

    fn handle_message(&mut self, message: &Message, tick: DemoTick, state: &ParserState) {
        match message {
            Message::PacketEntities(packet) => {
                for removed in &packet.removed_entities {
                    self.entities.remove(&u32::from(*removed));
                }
                for entity in &packet.entities {
                    let index = u32::from(entity.entity_index);
                    let class = usize::from(entity.server_class);
                    if entity.update_type == UpdateType::Enter {
                        if !self.layout(class, state).is_item {
                            self.entities.remove(&index);
                            continue;
                        }
                        // Props equal to the baseline are not sent. The instance baseline only holds
                        // what was sent for it; the rest (usually m_iItemIDHigh) is in the class's
                        // static baseline.
                        let mut values = HashMap::new();
                        if let Some(table) = state.send_tables.get(class) {
                            if let Ok(statics) = state.get_static_baseline(entity.server_class, table) {
                                for prop in &statics {
                                    if let Some(v) = int_value(&prop.value) {
                                        values.insert(prop.index, (v, Layer::Static));
                                    }
                                }
                            }
                        }
                        for prop in entity.get_baseline_props(state).iter() {
                            if let Some(v) = int_value(&prop.value) {
                                values.insert(prop.index, (v, Layer::Instance));
                            }
                        }
                        // An entity re-entering the PVS (after Leave, not deleted) is sent without the
                        // props the client still has from before; the baselines hold another item's
                        // values there. Keep the previous state when it is the same item.
                        let layout = &self.layouts[&class];
                        let sent_id = |i: Option<u32>| {
                            i.and_then(|i| entity.props.iter().find(|p| p.index == i)).and_then(|p| int_value(&p.value))
                        };
                        if let Some((prev_class, prev)) = self.entities.get(&index) {
                            let prev_id = |i: Option<u32>| i.and_then(|i| prev.get(&i)).map(|v| v.0);
                            let same_item = *prev_class == class
                                && [layout.id_low, layout.id_high]
                                    .iter()
                                    .all(|i| sent_id(*i).is_none_or(|v| Some(v) == prev_id(*i)))
                                && prev_id(layout.id_low).is_some();
                            if same_item {
                                values = prev.clone();
                            }
                        }
                        self.entities.insert(index, (class, values));
                    }
                    let Some((_, values)) = self.entities.get_mut(&index) else { continue };
                    let mut changed = entity.update_type == UpdateType::Enter;
                    for prop in &entity.props {
                        if let Some(v) = int_value(&prop.value) {
                            changed |= values.insert(prop.index, (v, Layer::Sent)).map(|old| old.0) != Some(v);
                        }
                    }
                    if changed {
                        self.record_item(index, u32::from(tick));
                    }
                }
            }
            Message::GameEvent(event) => {
                if let GameEvent::ItemFound(found) = &event.event {
                    let Some((steam_id, account_id, player_name)) = self.user_by_entity(found.player as u32) else {
                        return;
                    };
                    self.drops.push(ItemDrop {
                        tick: u32::from(tick),
                        steam_id,
                        account_id,
                        player_name,
                        item_def: found.item_def,
                        quality: found.quality,
                        method: found.method,
                        is_strange: found.is_strange != 0,
                        is_unusual: found.is_unusual != 0,
                    });
                }
            }
            _ => {}
        }
    }

    fn handle_string_entry(&mut self, table: &str, index: usize, entry: &StringTableEntry, _state: &ParserState) {
        if table != "userinfo" {
            return;
        }
        if let Ok(Some(info)) = UserInfo::parse_from_string_table(
            index as u16,
            entry.text.as_ref().map(|s| s.as_ref()),
            entry.extra_data.as_ref().map(|d| d.data.clone()),
        ) {
            self.users.insert(index as u32, info);
        }
    }

    fn into_output(self, _state: &ParserState) -> Self::Output {
        self
    }
}

fn int_value(value: &SendPropValue) -> Option<i64> {
    match value {
        SendPropValue::Integer(i) => Some(*i),
        _ => None,
    }
}

/// Parses a demo and collects its items and item drops. Truncated demos return what was read.
pub fn collect_item_facts(bytes: &[u8]) -> Result<DemoItemFacts> {
    let demo = Demo::new(bytes);
    let mut stream = demo.get_stream();
    let header = Header::read(&mut stream)?;
    let mut handler = DemoHandler::with_analyser(FactCollector::default());
    handler.handle_header(&header);
    while let Ok(packet) = Packet::parse(&mut stream, &handler.state_handler) {
        let stop = matches!(packet, Packet::Stop(_));
        if handler.handle_packet(packet).is_err() || stop {
            break;
        }
    }
    let collector = handler.into_output();
    let mut items: Vec<ItemFact> = collector.items.into_values().collect();
    items.sort_by_key(|i| i.item_id);
    Ok(DemoItemFacts {
        items,
        drops: collector.drops,
        map: header.map.clone(),
        duration_secs: header.duration,
    })
}

// ---------------------------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------------------------

/// Players needed with a purchase in the newest hold's week for the hold to bound the date from
/// above too (see `estimate_offline`).
const MIN_BUYERS_FOR_WINDOW: usize = 2;

/// Distinct accounts with a purchase whose hold ends within 8 days before `newest_hold_end`, i.e.
/// holds that could all be running at the same time as the newest one.
fn recent_buyers(facts: &DemoItemFacts, newest_hold_end: i64) -> usize {
    let mut accounts: Vec<u32> = facts
        .items
        .iter()
        .filter(|i| i.tradable_after.is_some_and(|t| t >= newest_hold_end - TRADE_HOLD_SECS - DAY))
        .map(|i| i.account_id)
        .collect();
    accounts.sort_unstable();
    accounts.dedup();
    accounts.len()
}

/// The estimate from the demo alone (methods 2–4).
pub fn estimate_offline(facts: &DemoItemFacts) -> Option<RecordingDate> {
    let newest_id = facts.newest_item_id()?;
    // The newest item ID's creation time: the demo cannot be older (within the timeline's precision)
    let bound = date_of_item_id(newest_id);
    let id_earliest = bound.timestamp - bound.uncertainty_secs;
    let held = facts.items.iter().filter(|i| i.tradable_after.is_some()).count();

    let mut expired_hold_note = String::new();
    if let Some(hold) = facts.newest_trade_hold() {
        let ts = hold.tradable_after.unwrap_or_default();
        if id_earliest <= ts {
            // Recorded after the purchase (7-8 days before the hold ends). The newest hold is
            // usually still running, so the demo was also recorded before the hold end, but only
            // when several players bought something that week: with a single buyer that one
            // purchase may be old and its hold over (validated against the Discord thread time:
            // 1 buyer 57 % provably running, 2+ buyers 77-87 %), so it is only a lower bound.
            let earliest = (ts - TRADE_HOLD_SECS - DAY).max(id_earliest);
            let buyers = recent_buyers(facts, ts);
            let bounded = buyers >= MIN_BUYERS_FOR_WINDOW;
            return Some(RecordingDate {
                method: DateMethod::TradeHold,
                estimate: if bounded { earliest + (ts - earliest) / 2 } else { earliest },
                earliest,
                latest: bounded.then_some(ts),
                detail: format!(
                    "Newest trade hold ends {} (item #{}); {buyers} player{} bought items in that week, {held} trade-held items in the demo{}",
                    format_date_time(ts),
                    hold.item_id,
                    if buyers == 1 { "" } else { "s" },
                    if bounded { "" } else { ". With a single buyer the hold may have ended before the match, so this is a lower bound" }
                ),
                api_note: None,
            });
        }
        // An item created after the newest hold ended: that hold was over when the demo was recorded
        expired_hold_note = format!(" The newest trade hold ({}) had already ended.", format_date(ts));
    }

    let extrapolated = bound.extrapolated && bound.timestamp > TIMELINE_LAST_DATE;
    Some(RecordingDate {
        method: if extrapolated { DateMethod::Extrapolated } else { DateMethod::NewestItemId },
        estimate: bound.timestamp,
        earliest: id_earliest,
        latest: None,
        detail: format!(
            "Newest item ID #{newest_id} was created around {}{}, so the demo cannot be older.{expired_hold_note}",
            format_date(bound.timestamp),
            if extrapolated {
                format!(" (extrapolated {} days past the timeline)", (bound.timestamp - TIMELINE_LAST_DATE) / DAY)
            } else {
                String::new()
            }
        ),
        api_note: None,
    })
}

mod steam_api;
pub use steam_api::{estimate_with_steam_api, SteamApiLimiter};

#[cfg(test)]
mod tests {
    use super::*;

    fn item(item_id: u64, tradable_after: Option<i64>) -> ItemFact {
        ItemFact { item_id, def: 1, account_id: 1, quality: 6, tradable_after, class: String::new(), first_tick: 0 }
    }

    fn bought_by(mut item: ItemFact, account_id: u32) -> ItemFact {
        item.account_id = account_id;
        item
    }

    fn facts(items: Vec<ItemFact>) -> DemoItemFacts {
        DemoItemFacts { items, ..Default::default() }
    }

    #[test]
    fn trade_hold_window_ends_with_the_newest_hold() {
        let (t, id) = timeline_data::KNOTS[timeline_data::KNOTS.len() - 30];
        let hold_end = t + TRADE_HOLD_SECS + DAY / 2;
        let est = estimate_offline(&facts(vec![
            item(id, Some(hold_end)),
            bought_by(item(id - 2_000_000, Some(hold_end - DAY)), 2),
            item(id - 1_000_000_000, Some(hold_end - 300 * DAY)),
        ]))
        .unwrap();
        assert_eq!(est.method, DateMethod::TradeHold);
        assert_eq!(est.latest, Some(hold_end));
        assert!(est.earliest >= hold_end - TRADE_HOLD_SECS - DAY && est.earliest < hold_end);
        assert!(est.precision_days().unwrap() <= 4.0);
    }

    #[test]
    fn single_recent_buyer_is_only_a_lower_bound() {
        let (t, id) = timeline_data::KNOTS[timeline_data::KNOTS.len() - 30];
        let hold_end = t + TRADE_HOLD_SECS + DAY / 2;
        // the other player's purchase is older than the newest hold's week
        let est = estimate_offline(&facts(vec![
            item(id, Some(hold_end)),
            item(id - 1_000_000, Some(hold_end - DAY)),
            bought_by(item(id - 100_000_000, Some(hold_end - 20 * DAY)), 2),
        ]))
        .unwrap();
        assert_eq!(est.method, DateMethod::TradeHold);
        assert!(est.latest.is_none());
        assert_eq!(est.estimate, est.earliest);
    }

    #[test]
    fn expired_hold_falls_back_to_the_newest_item_id() {
        let (t, id) = timeline_data::KNOTS[timeline_data::KNOTS.len() - 30];
        // an item created 30 days after the only hold ended
        let est = estimate_offline(&facts(vec![item(id, None), item(id - 500_000_000, Some(t - 30 * DAY))])).unwrap();
        assert_eq!(est.method, DateMethod::NewestItemId);
        assert!(est.latest.is_none());
        assert!((est.estimate - t).abs() < DAY);
    }

    #[test]
    fn ids_past_the_timeline_are_extrapolated() {
        let (_, id) = timeline_data::KNOTS[timeline_data::KNOTS.len() - 1];
        let est = estimate_offline(&facts(vec![item(id + 3_000_000, None)])).unwrap(); // ~0.7 days past the end
        assert_eq!(est.method, DateMethod::Extrapolated);
        assert!(est.estimate > TIMELINE_LAST_DATE);
    }

    #[test]
    fn item_ids_from_the_future_are_ignored() {
        let (_, id) = timeline_data::KNOTS[timeline_data::KNOTS.len() - 30];
        let corrupt = id + 4_000_000_000; // years past now
        let est = estimate_offline(&facts(vec![item(id, None), item(corrupt, None)])).unwrap();
        assert_eq!(est.method, DateMethod::NewestItemId);
        assert!(est.detail.contains(&format!("#{id}")));
    }

    #[test]
    fn no_items_no_estimate() {
        assert!(estimate_offline(&facts(vec![])).is_none());
    }
}
