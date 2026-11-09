//! Which animation set TF2 plays for each weapon, read from the game's item
//! schema (`scripts/items/items_game.txt`). The inventory slot alone isn't
//! enough: the Demoman's launchers swap their animations, all-class melee
//! weapons have their own set, and items like the Huntsman, the Sandvich and
//! the banners each use an "item" set.

use std::collections::HashMap;

use super::assets::Assets;
use super::skeleton::WeaponPose;

const ITEMS_GAME: &str = "scripts/items/items_game.txt";
/// TF2's class numbers, in the order the schema's class names are listed.
const CLASS_NAMES: [(&str, usize); 9] = [
    ("scout", 1),
    ("sniper", 2),
    ("soldier", 3),
    ("demoman", 4),
    ("medic", 5),
    ("heavy", 6),
    ("pyro", 7),
    ("spy", 8),
    ("engineer", 9),
];
/// How deep prefabs can refer to other prefabs before we stop following.
const PREFAB_DEPTH: u32 = 8;

/// The animation set for each of `models`, per class number.
pub type WeaponRoles = Vec<[Option<WeaponPose>; 10]>;

/// Looks up every model in `models` in the item schema. Models that aren't
/// weapons, or a schema that can't be read, leave `None`s.
pub fn weapon_roles(assets: &Assets, models: &[String]) -> WeaponRoles {
    match assets.load(ITEMS_GAME) {
        Some(text) => roles_from_schema(&String::from_utf8_lossy(&text), models),
        None => vec![[None; 10]; models.len()],
    }
}

fn roles_from_schema(text: &str, models: &[String]) -> WeaponRoles {
    let mut roles = vec![[None; 10]; models.len()];
    let root = parse(text);
    let Some(schema) = root.block("items_game") else {
        return roles;
    };
    let wanted: HashMap<String, usize> = models
        .iter()
        .enumerate()
        .map(|(index, path)| (normalize(path), index))
        .collect();
    let prefabs = schema.block("prefabs");
    let Some(items) = schema.block("items") else {
        return roles;
    };

    for (_, item) in items.entries() {
        let Some(item) = item.as_block() else { continue };
        let lookup = Lookup { item, prefabs };
        let item_class = lookup.text("item_class").unwrap_or_default();
        if !item_class.starts_with("tf_weapon") && item_class != "saxxy" {
            continue;
        }
        let used_by = lookup.pairs("used_by_classes");
        let anim_slot = lookup.text("anim_slot").and_then(role_from_anim_slot);
        let item_slot = lookup.text("item_slot");

        let mut paths: Vec<String> = ["model_player", "model_world"]
            .iter()
            .filter_map(|key| lookup.text(key))
            .map(str::to_string)
            .collect();
        for (class, path) in lookup.pairs("model_player_per_class") {
            if class == "basename" {
                for (name, _) in CLASS_NAMES {
                    paths.push(path.replace("%s", name));
                }
            } else {
                paths.push(path.to_string());
            }
        }

        for path in paths {
            let Some(&model) = wanted.get(&normalize(&path)) else { continue };
            for (class_name, slot_for_class) in &used_by {
                let Some(class) = CLASS_NAMES.iter().find(|(name, _)| name == class_name).map(|(_, id)| *id) else {
                    continue;
                };
                // Items used by several classes can sit in a different slot for each.
                let role = anim_slot
                    .or_else(|| role_from_slot(slot_for_class))
                    .or_else(|| item_slot.and_then(role_from_slot));
                if roles[model][class].is_none() {
                    roles[model][class] = role;
                }
            }
        }
    }
    roles
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

fn role_from_anim_slot(name: &str) -> Option<WeaponPose> {
    Some(match name.to_ascii_lowercase().as_str() {
        "primary" => WeaponPose::Primary,
        "secondary" => WeaponPose::Secondary,
        "melee" => WeaponPose::Melee,
        "melee_allclass" => WeaponPose::MeleeAllClass,
        "item1" => WeaponPose::Item1,
        "item2" => WeaponPose::Item2,
        "item3" => WeaponPose::Item3,
        "item4" => WeaponPose::Item4,
        "building" => WeaponPose::Building,
        "pda" => WeaponPose::Pda,
        "primary2" => WeaponPose::Primary2,
        "secondary2" => WeaponPose::Secondary2,
        // "FORCE_NOT_USED", empty and the PASS Time ball fall back to the slot.
        _ => return None,
    })
}

fn role_from_slot(name: &str) -> Option<WeaponPose> {
    Some(match name.to_ascii_lowercase().as_str() {
        "primary" => WeaponPose::Primary,
        "secondary" => WeaponPose::Secondary,
        "melee" => WeaponPose::Melee,
        "pda" | "pda2" => WeaponPose::Pda,
        "building" => WeaponPose::Building,
        _ => return None,
    })
}

/// Reads an item's keys, falling back to the prefabs it's built from.
struct Lookup<'a> {
    item: &'a Block,
    prefabs: Option<&'a Block>,
}

impl<'a> Lookup<'a> {
    fn text(&self, key: &str) -> Option<&'a str> {
        self.find(self.item, key, 0)
    }

    fn find(&self, block: &'a Block, key: &str, depth: u32) -> Option<&'a str> {
        if let Some(value) = block.text(key) {
            return Some(value);
        }
        if depth >= PREFAB_DEPTH {
            return None;
        }
        let names = block.text("prefab")?;
        names
            .split_whitespace()
            .filter_map(|name| self.prefabs?.block(name))
            .find_map(|prefab| self.find(prefab, key, depth + 1))
    }

    /// Every key and value in a sub-block, merged across the prefabs.
    fn pairs(&self, key: &str) -> Vec<(String, &'a str)> {
        let mut pairs = Vec::new();
        self.collect_pairs(self.item, key, 0, &mut pairs);
        pairs
    }

    fn collect_pairs(&self, block: &'a Block, key: &str, depth: u32, out: &mut Vec<(String, &'a str)>) {
        if let Some(found) = block.block(key) {
            for (name, value) in found.entries() {
                if let Some(value) = value.as_text() {
                    if !out.iter().any(|(existing, _)| existing == name) {
                        out.push((name.clone(), value));
                    }
                }
            }
        }
        if depth >= PREFAB_DEPTH {
            return;
        }
        let Some(names) = block.text("prefab") else { return };
        for prefab in names.split_whitespace().filter_map(|name| self.prefabs?.block(name)) {
            self.collect_pairs(prefab, key, depth + 1, out);
        }
    }
}

/// A KeyValues block: keys are lowercased, and repeated blocks are merged.
#[derive(Debug, Default)]
struct Block {
    entries: Vec<(String, Node)>,
    index: HashMap<String, usize>,
}

#[derive(Debug)]
enum Node {
    Text(String),
    Block(Block),
}

impl Node {
    fn as_block(&self) -> Option<&Block> {
        match self {
            Node::Block(block) => Some(block),
            Node::Text(_) => None,
        }
    }

    fn as_text(&self) -> Option<&str> {
        match self {
            Node::Text(text) => Some(text),
            Node::Block(_) => None,
        }
    }
}

impl Block {
    fn get(&self, key: &str) -> Option<&Node> {
        self.index.get(key).map(|&at| &self.entries[at].1)
    }

    fn text(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_text()
    }

    fn block(&self, key: &str) -> Option<&Block> {
        self.get(key)?.as_block()
    }

    fn entries(&self) -> impl Iterator<Item = &(String, Node)> {
        self.entries.iter()
    }

    fn insert(&mut self, key: String, node: Node) {
        match (self.index.get(&key).copied(), node) {
            (Some(at), Node::Block(more)) => {
                if let Node::Block(existing) = &mut self.entries[at].1 {
                    for (key, node) in more.entries {
                        existing.insert(key, node);
                    }
                }
            }
            (Some(at), node) => self.entries[at].1 = node,
            (None, node) => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, node));
            }
        }
    }
}

enum Token<'a> {
    Text(&'a str),
    Open,
    Close,
}

fn tokens(text: &str) -> Vec<Token<'_>> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b' ' | b'\t' | b'\r' | b'\n' => at += 1,
            b'/' if bytes.get(at + 1) == Some(&b'/') => {
                while at < bytes.len() && bytes[at] != b'\n' {
                    at += 1;
                }
            }
            b'{' => {
                out.push(Token::Open);
                at += 1;
            }
            b'}' => {
                out.push(Token::Close);
                at += 1;
            }
            b'"' => {
                let start = at + 1;
                let mut end = start;
                while end < bytes.len() && bytes[end] != b'"' {
                    end += if bytes[end] == b'\\' { 2 } else { 1 };
                }
                let end = end.min(bytes.len());
                out.push(Token::Text(&text[start..end]));
                at = end + 1;
            }
            _ => {
                let start = at;
                while at < bytes.len() && !matches!(bytes[at], b' ' | b'\t' | b'\r' | b'\n' | b'{' | b'}' | b'"') {
                    at += 1;
                }
                out.push(Token::Text(&text[start..at]));
            }
        }
    }
    out
}

fn parse(text: &str) -> Block {
    let tokens = tokens(text);
    let mut at = 0;
    parse_block(&tokens, &mut at)
}

fn parse_block(tokens: &[Token], at: &mut usize) -> Block {
    let mut block = Block::default();
    while *at < tokens.len() {
        let key = match tokens[*at] {
            Token::Close => {
                *at += 1;
                return block;
            }
            Token::Open => {
                // A block with no key: read it and drop it.
                *at += 1;
                parse_block(tokens, at);
                continue;
            }
            Token::Text(key) => key.to_ascii_lowercase(),
        };
        *at += 1;
        match tokens.get(*at) {
            Some(Token::Open) => {
                *at += 1;
                let child = parse_block(tokens, at);
                block.insert(key, Node::Block(child));
            }
            Some(Token::Text(value)) => {
                *at += 1;
                // Conditions like [$WIN32] trail some values. They're skipped.
                if let Some(Token::Text(condition)) = tokens.get(*at) {
                    if condition.starts_with('[') {
                        *at += 1;
                    }
                }
                block.insert(key, Node::Text((*value).to_string()));
            }
            _ => {}
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
"items_game"
{
    "prefabs"
    {
        "weapon_launcher" { "item_class" "tf_weapon_grenadelauncher" "item_slot" "primary" }
        "valve" { "prefab" "weapon_launcher" "anim_slot" "secondary" }
    }
    "items"
    {
        "19" { "prefab" "valve" "model_player" "models\Weapons\c_models\c_grenadelauncher.mdl"
               "used_by_classes" { "demoman" "1" } }
        "9"  { "item_class" "tf_weapon_shotgun" "item_slot" "primary" // comment
               "model_world" "models/weapons/c_models/c_shotgun.mdl"
               "used_by_classes" { "engineer" "1" "soldier" "secondary" } }
        "42" { "item_class" "tf_wearable" "model_player" "models/hat.mdl" "used_by_classes" { "heavy" "1" } }
    }
}
"#;

    #[test]
    fn schema_text_parses_with_comments_and_merging() {
        let root = parse(SAMPLE);
        let items = root.block("items_game").and_then(|game| game.block("items")).unwrap();
        assert_eq!(items.entries().count(), 3);
        assert_eq!(items.block("9").and_then(|item| item.text("item_slot")), Some("primary"));
    }

    #[test]
    fn roles_follow_prefabs_and_per_class_slots() {
        let root = parse(SAMPLE);
        let schema = root.block("items_game").unwrap();
        let lookup = Lookup {
            item: schema.block("items").unwrap().block("19").unwrap(),
            prefabs: schema.block("prefabs"),
        };
        assert_eq!(lookup.text("item_class"), Some("tf_weapon_grenadelauncher"));
        assert_eq!(lookup.text("anim_slot").and_then(role_from_anim_slot), Some(WeaponPose::Secondary));
        assert_eq!(role_from_anim_slot("FORCE_NOT_USED"), None);
        assert_eq!(role_from_slot("pda2"), Some(WeaponPose::Pda));
    }

    #[test]
    fn each_class_gets_the_weapons_animation_set() {
        let models = [
            "models/weapons/c_models/c_grenadelauncher.mdl",
            "models/weapons/c_models/c_shotgun.mdl",
            "models/hat.mdl",
        ]
        .map(String::from);
        let roles = roles_from_schema(SAMPLE, &models);
        // The Demoman's grenade launcher plays his "secondary" set.
        assert_eq!(roles[0][4], Some(WeaponPose::Secondary));
        assert_eq!(roles[1][9], Some(WeaponPose::Primary), "engineer shotgun");
        assert_eq!(roles[1][3], Some(WeaponPose::Secondary), "soldier shotgun");
        assert_eq!(roles[2], [None; 10], "hats aren't weapons");
    }
}
