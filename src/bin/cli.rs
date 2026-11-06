#[cfg(feature = "rcon")]
compile_error!("The 'rcon' feature is GUI-only and cannot be enabled for the CLI target.");

use analysis_template::{
    base::cheat_analyser_base::CheatAnalyser,
    chat_view::parse_chat,
    dev_print,
    header_view::HeaderData,
    lib::{
        algorithm::{analyse_bytes_with_settings, analyse_with_settings, get_algorithms, is_experimental_algorithm_name, CheatAlgorithm},
        general_settings::GeneralSettings,
        parameters::{strip_session_only_parameters, Parameter, Parameters},
    },
    playerlist_view::parse_playerlist,
    util::helpers::{
        convert_demo_to_replay, convert_demo_to_replay_with_takes, find_default_tf2_dir,
        format_sus_demo_filename, strip_consolecmds_from_bytes,
        write_stripped_demo_to_replays,
        ReplaySuspect, ReplayTakeInput,
    },
    SILENT,
};

use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self},
    path::{Path, PathBuf},
};

use anyhow::Error;

pub use tf_demo_parser::{Demo, DemoParser, Parse, ParseError, ParserState, Stream};

use getopts::Options;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn move_demo_file(src: &Path, dst: &Path) -> Result<(), Error> {
    if src == dst {
        return Ok(());
    }
    if dst.exists() {
        let _ = fs::remove_file(dst);
    }
    if fs::rename(src, dst).is_err() {
        fs::copy(src, dst)?;
        fs::remove_file(src)?;
    }
    Ok(())
}

fn get_reported_ids_path(custom_file: Option<&str>) -> (Option<PathBuf>, PathBuf) {
    if let Some(f) = custom_file {
        let p = PathBuf::from(f);
        let existing = if p.exists() { Some(p.clone()) } else { None };
        return (existing, p);
    }

    let cwd_path = PathBuf::from("reported_ids.txt");
    let exe_path = env::current_exe().ok().and_then(|mut p| {
        p.pop();
        let candidate = p.join("reported_ids.txt");
        if candidate.exists() {
            Some(candidate)
        } else {
            None
        }
    });

    if cwd_path.exists() {
        (Some(cwd_path), PathBuf::from("reported_ids.txt"))
    } else if let Some(ep) = exe_path {
        (Some(ep), PathBuf::from("reported_ids.txt"))
    } else {
        (None, PathBuf::from("reported_ids.txt"))
    }
}

fn parse_steam_id_line(line: &str) -> Option<u64> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
        return None;
    }
    if let Ok(id) = trimmed.parse::<u64>() {
        return Some(id);
    }
    if let Ok(sid) = steamid_ng::SteamID::from_steam2(trimmed) {
        return Some(u64::from(sid));
    }
    if let Ok(sid) = steamid_ng::SteamID::from_steam3(trimmed) {
        return Some(u64::from(sid));
    }
    None
}

pub fn load_or_download_reported_ids(custom_file: Option<&str>, silent: bool) -> Result<HashSet<u64>, Error> {
    let (existing, save_path) = get_reported_ids_path(custom_file);
    if let Some(p) = existing {
        let content = fs::read_to_string(&p)?;
        let mut ids = HashSet::new();
        for line in content.lines() {
            if let Some(id) = parse_steam_id_line(line) {
                ids.insert(id);
            }
        }
        if !silent {
            println!("Loaded {} marked cheater IDs from existing '{}' (no download)", ids.len(), p.display());
        }
        return Ok(ids);
    }

    // File not found locally -> download once
    if !silent {
        println!("reported_ids.txt not found locally. Downloading marked cheater list...");
    }
    const URL: &str = "https://raw.githubusercontent.com/Nocrex/Tom/refs/heads/main/reported_ids.txt";
    let resp = ureq::get(URL)
        .timeout(std::time::Duration::from_secs(15))
        .call()
        .map_err(|e| anyhow::anyhow!("Failed to download reported_ids.txt from {}: {}", URL, e))?;
    let body = resp.into_string()?;

    let _ = fs::write(&save_path, &body);
    if let Ok(mut exe) = env::current_exe() {
        exe.pop();
        let exe_file = exe.join("reported_ids.txt");
        if exe_file != save_path {
            let _ = fs::write(&exe_file, &body);
        }
    }

    let mut ids = HashSet::new();
    for line in body.lines() {
        if let Some(id) = parse_steam_id_line(line) {
            ids.insert(id);
        }
    }
    if !silent {
        println!("Downloaded and saved {} marked cheater IDs to '{}'", ids.len(), save_path.display());
    }

    Ok(ids)
}

/// Whether a flagged player's Steam account still exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountStatus {
    Exists,
    Deleted,
    /// The request failed or the response was not recognised; retried on the next analysis.
    Unknown,
}

/// Queries the public Steam community XML profile. Deleted accounts answer with
/// `<response><error>The specified profile could not be found.</error></response>`,
/// while every existing account (including never set up profiles) returns a `<profile>`.
pub fn parse_steam_profile_xml(body: &str) -> AccountStatus {
    if body.contains("<profile>") {
        AccountStatus::Exists
    } else if body.contains("<response>") && body.contains("The specified profile could not be found") {
        AccountStatus::Deleted
    } else {
        AccountStatus::Unknown
    }
}

/// Queries the public Steam community XML profile. Deleted accounts answer with
/// `<response><error>The specified profile could not be found.</error></response>`,
/// while every existing account (including never set up profiles) returns a `<profile>`.
pub fn check_steam_account(steam_id: u64) -> AccountStatus {
    let Ok(resp) = analysis_template::util::helpers::request_steam_profile_xml(steam_id) else {
        return AccountStatus::Unknown;
    };
    let Ok(body) = resp.into_string() else {
        return AccountStatus::Unknown;
    };
    parse_steam_profile_xml(&body)
}

pub struct SteamAccountChecker {
    pub cache: HashMap<u64, AccountStatus>,
    last_request: Option<std::time::Instant>,
    request_interval: std::time::Duration,
    /// When an `Unknown` lookup failed: the same account is not requested again until
    /// `unknown_retry_after` passed, since `get_status` is called once per detection.
    failed_at: HashMap<u64, std::time::Instant>,
    unknown_retry_after: std::time::Duration,
}

impl SteamAccountChecker {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            last_request: None,
            // Stay well below the Steam community rate limit
            request_interval: std::time::Duration::from_millis(300),
            failed_at: HashMap::new(),
            unknown_retry_after: std::time::Duration::from_secs(300),
        }
    }

    pub fn get_status(&mut self, steam_id: u64) -> AccountStatus {
        if steam_id == 0 {
            return AccountStatus::Exists;
        }
        if let Some(&status) = self.cache.get(&steam_id) {
            if status != AccountStatus::Unknown {
                return status;
            }
            if self.failed_at.get(&steam_id).is_some_and(|t| t.elapsed() < self.unknown_retry_after) {
                return status;
            }
        }

        if let Some(wait) = self.last_request.and_then(|t| self.request_interval.checked_sub(t.elapsed())) {
            std::thread::sleep(wait);
        }
        self.last_request = Some(std::time::Instant::now());

        let status = check_steam_account(steam_id);
        self.cache.insert(steam_id, status);
        if status == AccountStatus::Unknown {
            self.failed_at.insert(steam_id, std::time::Instant::now());
        }
        status
    }
}


fn prepare_algorithms(
    config: &Option<HashMap<String, Parameters>>,
    specified_algorithms: &[String],
) -> Vec<Box<dyn CheatAlgorithm<'static> + Send>> {
    let mut algorithms: Vec<Box<dyn CheatAlgorithm<'static> + Send>> = get_algorithms();
    if specified_algorithms.is_empty() {
        algorithms.retain(|a| {
            if let Some(ref cfg) = config {
                if let Some(algo_cfg) = cfg.get(a.algorithm_name()) {
                    if let Some(Parameter::Bool(b)) = algo_cfg.get("enabled") {
                        return *b;
                    }
                }
            }
            a.default()
        });
    } else {
        algorithms.retain(|a| specified_algorithms.contains(&a.algorithm_name().to_string()));
    }

    if let Some(ref cfg) = config {
        for algo in algorithms.iter_mut() {
            let algorithm_name: String = algo.algorithm_name().to_owned();
            if let Some(algo_params) = algo.params() {
                algo_params.iter_mut().for_each(|(k, v)| {
                    if let Some(config_params) = cfg.get(&algorithm_name) {
                        if let Some(param_value) = config_params.get(k) {
                            match (&*v, param_value) {
                                (Parameter::String(_), Parameter::Int(i)) => {
                                    *v = Parameter::String(if *i <= 0 { String::new() } else { i.to_string() });
                                }
                                (Parameter::String(_), Parameter::UInt(u)) => {
                                    *v = Parameter::String(if *u == 0 { String::new() } else { u.to_string() });
                                }
                                (Parameter::UInt(_), Parameter::Int(i)) if *i >= 0 => {
                                    *v = Parameter::UInt(*i as u64);
                                }
                                (Parameter::UInt(_), Parameter::String(_)) => {
                                    if let Ok(u) = u64::try_from(param_value) {
                                        *v = Parameter::UInt(u);
                                    } else {
                                        *v = param_value.clone();
                                    }
                                }
                                (_, src) => {
                                    *v = src.clone();
                                }
                            }
                        }
                    }
                });
            }
        }
    }

    algorithms
}

fn print_algorithms_list(
    config: &Option<HashMap<String, Parameters>>,
    silent: bool,
    pretty: bool,
) -> Result<(), Error> {
    let mut algos = get_algorithms();
    algos.sort_by(|a, b| a.algorithm_name().cmp(b.algorithm_name()));

    let mut proven_count = 0;
    let mut experimental_count = 0;

    struct AlgoDisplayItem {
        name: String,
        tier: &'static str,
        default_enabled: bool,
        configured_enabled: Option<bool>,
        params: Vec<(String, Parameter)>,
    }

    let mut display_items = Vec::new();

    for algo in algos.iter_mut() {
        let name = algo.algorithm_name().to_string();
        let is_exp = is_experimental_algorithm_name(&name);
        let tier = if is_exp {
            experimental_count += 1;
            "experimental"
        } else {
            proven_count += 1;
            "proven"
        };
        let default_enabled = algo.default();
        let configured_enabled = config.as_ref().and_then(|cfg| {
            cfg.get(&name).and_then(|p| {
                if let Some(Parameter::Bool(b)) = p.get("enabled") {
                    Some(*b)
                } else {
                    None
                }
            })
        });

        let mut param_list = Vec::new();
        if let Some(params) = algo.params() {
            if let Some(cfg) = config {
                if let Some(cfg_params) = cfg.get(&name) {
                    for (k, v) in params.iter_mut() {
                        if let Some(val) = cfg_params.get(k) {
                            *v = val.clone();
                        }
                    }
                }
            }
            let mut sorted_keys: Vec<String> = params.keys().cloned().collect();
            sorted_keys.sort();
            for k in sorted_keys {
                if let Some(val) = params.get(&k) {
                    param_list.push((k, val.clone()));
                }
            }
        }

        display_items.push(AlgoDisplayItem {
            name,
            tier,
            default_enabled,
            configured_enabled,
            params: param_list,
        });
    }

    if silent {
        let json_items: Vec<serde_json::Value> = display_items
            .iter()
            .map(|item| {
                let params_map: Option<serde_json::Map<String, serde_json::Value>> = if item.params.is_empty() {
                    None
                } else {
                    let mut m = serde_json::Map::new();
                    for (k, v) in &item.params {
                        if let Ok(val) = serde_json::to_value(v) {
                            m.insert(k.clone(), val);
                        }
                    }
                    Some(m)
                };
                serde_json::json!({
                    "name": item.name,
                    "tier": item.tier,
                    "default": item.default_enabled,
                    "configured_enabled": item.configured_enabled,
                    "parameters": params_map,
                })
            })
            .collect();

        let root = serde_json::json!({
            "total": display_items.len(),
            "proven_count": proven_count,
            "experimental_count": experimental_count,
            "algorithms": json_items,
        });

        let json_str = if pretty {
            serde_json::to_string_pretty(&root)?
        } else {
            serde_json::to_string(&root)?
        };
        println!("{}", json_str);
        return Ok(());
    }

    println!();
    println!(
        "Available Cheat Detection Algorithms ({} total: {} proven, {} experimental):",
        display_items.len(),
        proven_count,
        experimental_count
    );
    println!("{:-<100}", "");
    println!("{:<32} {:<15} {:<10} {}", "ALGORITHM", "TIER", "DEFAULT", "PARAMETERS");
    println!("{:-<100}", "");

    for item in &display_items {
        let default_str = match item.configured_enabled {
            Some(true) => "on (config)",
            Some(false) => "off (config)",
            None => if item.default_enabled { "on" } else { "off" },
        };
        let params_str = {
            let non_enabled_params: Vec<_> = item.params
                .iter()
                .filter(|(k, _)| k != "enabled")
                .collect();
            if non_enabled_params.is_empty() {
                "none".to_string()
            } else {
                non_enabled_params
                    .into_iter()
                    .map(|(k, v)| {
                        let val_str = match v {
                            Parameter::Float(f) => format!("{}", f),
                            Parameter::Int(i) => format!("{}", i),
                            Parameter::UInt(u) => format!("{}", u),
                            Parameter::Bool(b) => format!("{}", b),
                            Parameter::String(s) => format!("\"{}\"", s),
                        };
                        format!("{}={}", k, val_str)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };

        println!("{:<32} {:<15} {:<10} {}", item.name, item.tier, default_str, params_str);
    }
    println!("{:-<100}", "");
    if experimental_count == 0 {
        println!("Note: No experimental algorithms are compiled in this build.");
        println!("      Enable all: cargo run --bin cli --features experimental -- [options]");
        println!("      Enable specific: $env:EXPERIMENTAL_ALGOS=\"algo_name\"; cargo run --bin cli -- [options]");
    }
    println!();

    Ok(())
}

enum BulkDemoResult {
    Analysed {
        analyser: CheatAnalyser<'static>,
        stripped_count: Option<usize>,
        stripped_bytes: Option<Vec<u8>>,
    },
    StrippedOnly {
        stripped_count: usize,
    },
    /// Not analysed because of --valve-only and counted as clean; the server name is empty when none was found.
    NotValveServer {
        server_name: String,
        stripped_count: Option<usize>,
    },
    ReadFailed(std::io::Error),
    AnalysisFailed(Error),
    StripFailed(String),
}

/// Analyses one demo of a bulk scan on the calling thread (bulk scans run several demos at once,
/// which uses the CPU better than splitting each demo's algorithms across workers). A demo that makes
/// the parser or an algorithm panic is reported as failed instead of taking the whole scan down.
fn analyse_bulk_demo(
    demo_path: &Path,
    config: &Option<HashMap<String, Parameters>>,
    specified_algorithms: &[String],
    strip: bool,
    strip_only: bool,
    valve_only: bool,
    scratch_dir: &Option<PathBuf>,
    keep_stripped_bytes: bool,
) -> BulkDemoResult {
    let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        analyse_bulk_demo_unguarded(demo_path, config, specified_algorithms, strip, strip_only, valve_only, scratch_dir, keep_stripped_bytes)
    }));
    match guarded {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            BulkDemoResult::AnalysisFailed(anyhow::anyhow!("analysis panicked (corrupted demo?): {message}"))
        }
    }
}

fn analyse_bulk_demo_unguarded(
    demo_path: &Path,
    config: &Option<HashMap<String, Parameters>>,
    specified_algorithms: &[String],
    strip: bool,
    strip_only: bool,
    valve_only: bool,
    scratch_dir: &Option<PathBuf>,
    keep_stripped_bytes: bool,
) -> BulkDemoResult {
    let file_bytes = match fs::read(demo_path) {
        Ok(bytes) => bytes,
        Err(e) => return BulkDemoResult::ReadFailed(e),
    };

    let (bytes_to_analyse, stripped_count) = if strip || strip_only {
        let (stripped_bytes, count) = match strip_consolecmds_from_bytes(&file_bytes) {
            Ok(res) => res,
            Err(e) => return BulkDemoResult::StripFailed(e),
        };
        if let Some(scratch) = scratch_dir {
            let filename = demo_path
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("unknown.dem"));
            let target_scratch = scratch.join(filename);
            if let Err(e) = fs::write(&target_scratch, &stripped_bytes) {
                return BulkDemoResult::StripFailed(format!(
                    "Failed to write to '{}': {e}",
                    target_scratch.display()
                ));
            }
        }
        (stripped_bytes, Some(count))
    } else {
        (file_bytes, None)
    };

    if strip_only {
        return BulkDemoResult::StrippedOnly {
            stripped_count: stripped_count.unwrap_or(0),
        };
    }

    if valve_only {
        match HeaderData::parse_bytes(&bytes_to_analyse, String::new()) {
            Ok(header) if header.is_valve_server() => {}
            Ok(header) => return BulkDemoResult::NotValveServer { server_name: header.server_name, stripped_count },
            Err(e) => return BulkDemoResult::AnalysisFailed(e),
        }
    }

    let algorithms = prepare_algorithms(config, specified_algorithms);
    let general_settings = GeneralSettings::from_config_opt(config);
    match analyse_with_settings(&Demo::new(&bytes_to_analyse), algorithms, &general_settings) {
        Ok(mut analyser) => {
            // Free the algorithms' state while the result waits to be handled in file order.
            analyser.algorithms.clear();
            let stripped_bytes_opt = if keep_stripped_bytes {
                Some(bytes_to_analyse)
            } else {
                None
            };
            BulkDemoResult::Analysed {
                analyser,
                stripped_count,
                stripped_bytes: stripped_bytes_opt,
            }
        }
        Err(e) => BulkDemoResult::AnalysisFailed(e),
    }
}

fn run_bulk_scan(
    dir_path: &Path,
    config: &Option<HashMap<String, Parameters>>,
    specified_algorithms: &[String],
    reported_ids: &Option<HashSet<u64>>,
    ignore_marked: bool,
    ignore_deleted: bool,
    no_move: bool,
    replay_old: bool,
    replay_new: bool,
    silent: bool,
    pretty: bool,
    strip: bool,
    strip_only: bool,
    valve_only: bool,
    custom_scratch_dir: Option<PathBuf>,
    cleanup_jsons: bool,
) -> Result<(), Error> {
    let replay_creation = replay_old || replay_new;
    let use_scratch = strip_only || (strip && !replay_creation);
    let scratch_dir = custom_scratch_dir.unwrap_or_else(|| dir_path.join("scratch"));
    if use_scratch {
        fs::create_dir_all(&scratch_dir)?;
    }

    let sus_dir = if use_scratch { scratch_dir.join("sus") } else { dir_path.join("sus") };
    let clean_dir = if use_scratch { scratch_dir.join("clean") } else { dir_path.join("clean") };
    if !no_move && !strip_only {
        fs::create_dir_all(&sus_dir)?;
        fs::create_dir_all(&clean_dir)?;
    }

    let mut demo_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext.eq_ignore_ascii_case("dem") {
                    demo_files.push(path);
                }
            }
        }
    }
    demo_files.sort();

    let total = demo_files.len();
    if !silent {
        println!("Bulk Scan Directory: {}", dir_path.display());
        println!("Found {} demo(s) to analyze", total);
        if strip {
            if use_scratch {
                println!("Strip mode active: consolecmds will be stripped into '{}' (originals untouched)", scratch_dir.display());
            } else {
                println!("Strip mode active: consolecmds will be stripped directly into TF2 replay directory (no scratch folder used, originals untouched in content)");
            }
        } else if strip_only {
            println!("Strip-only mode active: consolecmds will be stripped into '{}' (no analysis)", scratch_dir.display());
        }
        if ignore_marked {
            if let Some(ids) = reported_ids {
                println!("Filtering active: {} marked cheater IDs will be ignored", ids.len());
            }
        }
        if ignore_deleted {
            println!("Filtering active: deleted Steam accounts will be ignored");
        }
        if valve_only {
            println!("Filtering active: demos from non-Valve servers will not be analysed and count as clean");
        }
        if no_move {
            if strip && use_scratch {
                println!("Mode: In-place scan (stripped demos will remain flat in '{}')", scratch_dir.display());
            } else {
                println!("Mode: In-place scan (demos will NOT be moved or sorted)");
            }
        } else if strip && use_scratch {
            println!("Sorting destination: sus/ -> {}, clean/ -> {}", sus_dir.display(), clean_dir.display());
            println!("(Original demos in '{}' remain untouched)", dir_path.display());
        } else {
            println!("Sorting destination: sus/ -> {}, clean/ -> {}", sus_dir.display(), clean_dir.display());
        }
        println!("--------------------------------------------------");
    }

    if total == 0 {
        if !silent {
            println!("No .dem files found directly in '{}'.", dir_path.display());
        } else {
            let mut res = serde_json::json!({
                "directory": dir_path.to_string_lossy(),
                "total": 0,
                "clean": 0,
                "sus": 0,
                "sus_demos": [],
                "clean_demos": []
            });
            if strip || strip_only {
                if use_scratch {
                    res["scratch_directory"] = serde_json::json!(scratch_dir.to_string_lossy());
                }
                res["total_consolecmds_stripped"] = serde_json::json!(0);
            }
            println!("{}", if pretty { serde_json::to_string_pretty(&res)? } else { serde_json::to_string(&res)? });
        }
        return Ok(());
    }

    let mut clean_demos = Vec::new();
    let mut sus_demos = Vec::new();
    // Demos that could not be analysed at all, and demos analysed only up to a parse error or a cut-off.
    let mut failed_demos: Vec<serde_json::Value> = Vec::new();
    let mut partial_demos: Vec<serde_json::Value> = Vec::new();
    let mut non_valve_count: usize = 0;
    let mut total_consolecmds_stripped: usize = 0;
    let start_scan = std::time::Instant::now();

    // Several demos are analysed at once, one thread each. Results are still handled (printed,
    // moved, converted to replays) in file order on this thread.
    let analysis_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(total)
        .max(1);
    // The per-demo analyser output (map, duration, tick progress) would interleave between the
    // demos being analysed at the same time, so it is silenced during the scan.
    SILENT.store(true, std::sync::atomic::Ordering::Relaxed);
    let next_demo = std::sync::atomic::AtomicUsize::new(0);
    let scratch_dir_opt = if use_scratch {
        Some(scratch_dir.clone())
    } else {
        None
    };
    let keep_stripped_bytes = strip && replay_creation;

    std::thread::scope(|s| {
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        for _ in 0..analysis_threads {
            let result_sender = result_sender.clone();
            let next_demo = &next_demo;
            let demo_files = &demo_files;
            let scratch_dir_ref = &scratch_dir_opt;
            s.spawn(move || loop {
                let index = next_demo.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(demo_path) = demo_files.get(index) else {
                    break;
                };
                let result = analyse_bulk_demo(demo_path, config, specified_algorithms, strip, strip_only, valve_only, scratch_dir_ref, keep_stripped_bytes);
                if result_sender.send((index, result)).is_err() {
                    break;
                }
            });
        }
        drop(result_sender);
        let mut finished: HashMap<usize, BulkDemoResult> = HashMap::new();
        let mut account_checker = SteamAccountChecker::new();

        for (idx, demo_path) in demo_files.iter().enumerate() {
            let result = loop {
                if let Some(result) = finished.remove(&idx) {
                    break Some(result);
                }
                match result_receiver.recv() {
                    Ok((index, result)) => {
                        finished.insert(index, result);
                    }
                    // An analysis thread panicked; the scope re-raises the panic.
                    Err(_) => break None,
                }
            };
            let Some(result) = result else {
                break;
            };

            let file_name_str = demo_path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown.dem".to_string());

            let (analyser, stripped_count, stripped_bytes) = match result {
                BulkDemoResult::Analysed { analyser, stripped_count, stripped_bytes } => {
                    if !silent {
                        if let Some(sc) = stripped_count {
                            println!("[{}/{}] Stripped {} consolecmds, analyzing {} ...", idx + 1, total, sc, file_name_str);
                        } else {
                            println!("[{}/{}] Analyzing {} ...", idx + 1, total, file_name_str);
                        }
                    }
                    if let Some(reason) = &analyser.parse_error {
                        if !silent {
                            println!("WARNING: partial analysis, {}", reason);
                        }
                        partial_demos.push(serde_json::json!({ "demo": file_name_str, "reason": reason }));
                    }
                    (analyser, stripped_count, stripped_bytes)
                }
                BulkDemoResult::StrippedOnly { stripped_count } => {
                    total_consolecmds_stripped += stripped_count;
                    if !silent {
                        println!("[{}/{}] Stripped {} consolecmds from {} -> {}", idx + 1, total, stripped_count, file_name_str, scratch_dir.join(&file_name_str).display());
                    }
                    clean_demos.push(serde_json::json!({
                        "demo": file_name_str,
                        "dest": scratch_dir.join(&file_name_str).to_string_lossy(),
                        "stripped_consolecmds": stripped_count,
                    }));
                    continue;
                }
                BulkDemoResult::NotValveServer { server_name, stripped_count } => {
                    if let Some(sc) = stripped_count {
                        total_consolecmds_stripped += sc;
                    }
                    let source_demo_to_move = if use_scratch {
                        scratch_dir.join(&file_name_str)
                    } else {
                        demo_path.clone()
                    };
                    let dest_path = if no_move {
                        source_demo_to_move.clone()
                    } else {
                        clean_dir.join(&file_name_str)
                    };
                    if !no_move {
                        if let Err(e) = move_demo_file(&source_demo_to_move, &dest_path) {
                            if !silent {
                                println!("ERROR: failed to move to clean: {}", e);
                            }
                            continue;
                        }
                    }
                    non_valve_count += 1;
                    if !silent {
                        let shown = if server_name.is_empty() { "unknown" } else { server_name.as_str() };
                        let reason = format!("non-Valve server: {}", shown);
                        println!("[{}/{}] Not analyzing {} ...", idx + 1, total, file_name_str);
                        if no_move {
                            if strip && use_scratch {
                                println!("CLEAN -> clean (stripped copy in scratch/, {})", reason);
                            } else {
                                println!("CLEAN -> clean (not moved, {})", reason);
                            }
                        } else if strip && use_scratch {
                            println!("CLEAN -> moved to scratch/clean/ ({})", reason);
                        } else {
                            println!("CLEAN -> moved to clean/ ({})", reason);
                        }
                    }
                    clean_demos.push(serde_json::json!({
                        "demo": file_name_str,
                        "dest": dest_path.to_string_lossy(),
                        "stripped_consolecmds": stripped_count,
                        "ignored_marked_detections": 0,
                        "ignored_deleted_detections": 0,
                        "non_valve_server": server_name,
                    }));
                    continue;
                }
                BulkDemoResult::StripFailed(e) => {
                    if !silent {
                        println!("[{}/{}] ERROR: failed to strip consolecmds from {}: {}", idx + 1, total, file_name_str, e);
                    }
                    failed_demos.push(serde_json::json!({ "demo": file_name_str, "reason": format!("failed to strip consolecmds: {e}") }));
                    continue;
                }
                BulkDemoResult::ReadFailed(e) => {
                    if !silent {
                        println!("[{}/{}] ERROR: failed to read {}: {}", idx + 1, total, file_name_str, e);
                    }
                    failed_demos.push(serde_json::json!({ "demo": file_name_str, "reason": format!("failed to read file: {e}") }));
                    continue;
                }
                BulkDemoResult::AnalysisFailed(e) => {
                    if !silent {
                        println!("[{}/{}] ERROR: failed to analyze {}: {}", idx + 1, total, file_name_str, e);
                    }
                    failed_demos.push(serde_json::json!({ "demo": file_name_str, "reason": format!("{e}") }));
                    continue;
                }
            };

            if let Some(sc) = stripped_count {
                total_consolecmds_stripped += sc;
            }

            let all_detections = analyser.detections;
            let mut qualifying_detections = Vec::new();
            let mut ignored_marked_count = 0;
            let mut ignored_deleted_count = 0;

            if ignore_deleted {
                let mut unique_players: HashSet<u64> = all_detections.iter().map(|d| d.player).collect();
                unique_players.remove(&0);
                for &player in &unique_players {
                    let is_marked = reported_ids.as_ref().map_or(false, |ids| ids.contains(&player));
                    if !is_marked {
                        account_checker.get_status(player);
                    }
                }
            }

            for d in all_detections {
                let is_marked = reported_ids.as_ref().map_or(false, |ids| ids.contains(&d.player));
                if is_marked {
                    if ignore_marked {
                        ignored_marked_count += 1;
                    } else {
                        qualifying_detections.push(d);
                    }
                } else if ignore_deleted && d.player != 0 && account_checker.get_status(d.player) == AccountStatus::Deleted {
                    ignored_deleted_count += 1;
                } else {
                    qualifying_detections.push(d);
                }
            }

            let is_sus = !qualifying_detections.is_empty();
            let target_filename = if is_sus {
                format_sus_demo_filename(&file_name_str, qualifying_detections.len())
            } else {
                file_name_str.clone()
            };
            let (dest_dir, category_str) = if is_sus {
                (&sus_dir, "sus")
            } else {
                (&clean_dir, "clean")
            };

            let source_demo_to_move = if use_scratch {
                scratch_dir.join(&file_name_str)
            } else {
                demo_path.clone()
            };
            let dest_path = if no_move {
                source_demo_to_move.clone()
            } else {
                dest_dir.join(&target_filename)
            };

            if !no_move {
                if let Err(e) = move_demo_file(&source_demo_to_move, &dest_path) {
                    if !silent {
                        println!("ERROR: failed to move to {}: {}", category_str, e);
                    }
                    continue;
                }
            }

            if is_sus {
                let mut player_algos: HashMap<u64, Vec<String>> = HashMap::new();
                for d in &qualifying_detections {
                    player_algos.entry(d.player).or_default().push(d.algorithm.clone());
                }

                // When strip is run with replay creation, write the stripped demo directly into the replay folder
                if strip && (replay_old || replay_new) {
                    if let Some(bytes) = &stripped_bytes {
                        if let Err(e) = write_stripped_demo_to_replays(&dest_path, bytes) {
                            if !silent {
                                println!("       Warning: Failed to write stripped demo to replays folder: {}", e);
                            }
                        }
                    }
                }

                // Optional replay creation in tf/replay/client/replays
                let header_info = analyser.header.as_ref().map(|h| (h.map.as_str(), h.duration));
                let replay_handle = if replay_old {
                    match convert_demo_to_replay(&dest_path, header_info) {
                        Ok(handle) => Some(handle),
                        Err(e) => {
                            if !silent {
                                println!("       Warning: Failed to convert replay (old): {}", e);
                            }
                            None
                        }
                    }
                } else {
                    None
                };

                let replay_handles_new: Vec<u32> = if replay_new {
                    let mut suspects: Vec<ReplaySuspect> = Vec::new();
                    for (sid, _) in &player_algos {
                        let slot = analyser
                            .state
                            .get_player_slot(*sid)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "-1".to_string());
                        let name = analyser.state.get_player_name(*sid).unwrap_or("unknown").to_string();
                        let mut dets: Vec<_> = qualifying_detections.iter().filter(|d| d.player == *sid).collect();
                        dets.sort_by_key(|d| d.tick);
                        suspects.push(ReplaySuspect {
                            steam_id: *sid,
                            slot: slot.clone(),
                            takes: dets
                                .into_iter()
                                .map(|d| ReplayTakeInput {
                                    tick: d.tick,
                                    player_name: name.clone(),
                                    algorithm: d.algorithm.clone(),
                                    player_slot: slot.clone(),
                                })
                                .collect(),
                        });
                    }
                    suspects.sort_by_key(|s| s.steam_id);
                    let all_detections: Vec<analysis_template::util::helpers::ReplayDetection> = qualifying_detections
                        .iter()
                        .map(|d| analysis_template::util::helpers::ReplayDetection {
                            steam_id: d.player,
                            tick: d.tick,
                            algorithm: d.algorithm.clone(),
                        })
                        .collect();
                    match convert_demo_to_replay_with_takes(&dest_path, header_info, &suspects, &all_detections) {
                        Ok(handles) => handles,
                        Err(e) => {
                            if !silent {
                                println!("       Warning: Failed to convert replay (new): {}", e);
                            }
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };

                if !silent {
                    let mut player_summaries = Vec::new();
                    for (sid, algos) in &player_algos {
                        let name = analyser.state.get_player_name(*sid).unwrap_or("unknown");
                        player_summaries.push(format!("{} ({}) [{} detections]", name, sid, algos.len()));
                    }
                    let ignored_suffix = if ignored_marked_count > 0 && ignored_deleted_count > 0 {
                        format!(" ({} detections on marked cheaters, {} on deleted accounts ignored)", ignored_marked_count, ignored_deleted_count)
                    } else if ignored_marked_count > 0 {
                        format!(" ({} detections on marked cheaters ignored)", ignored_marked_count)
                    } else if ignored_deleted_count > 0 {
                        format!(" ({} detections on deleted accounts ignored)", ignored_deleted_count)
                    } else {
                        String::new()
                    };
                    let mut replay_suffix = String::new();
                    if let Some(h) = replay_handle {
                        replay_suffix.push_str(&format!(" [Replay (old) #{}]", h));
                    }
                    if !replay_handles_new.is_empty() {
                        let list: Vec<String> = replay_handles_new.iter().map(|h| format!("#{}", h)).collect();
                        replay_suffix.push_str(&format!(" [Replay (new) {}]", list.join(", ")));
                    }
                    if no_move {
                        if strip && use_scratch {
                            println!("SUS -> flagged (stripped copy in scratch/){}\n       New suspect(s): {}{}", replay_suffix, player_summaries.join(", "), ignored_suffix);
                        } else {
                            println!("SUS -> flagged (not moved){}\n       New suspect(s): {}{}", replay_suffix, player_summaries.join(", "), ignored_suffix);
                        }
                    } else if strip && use_scratch {
                        println!("SUS -> moved to scratch/sus/ as {}{}\n       New suspect(s): {}{}", target_filename, replay_suffix, player_summaries.join(", "), ignored_suffix);
                    } else {
                        println!("SUS -> moved to sus/ as {}{}\n       New suspect(s): {}{}", target_filename, replay_suffix, player_summaries.join(", "), ignored_suffix);
                    }
                }

                sus_demos.push(serde_json::json!({
                    "demo": file_name_str,
                    "renamed": if no_move { None } else { Some(&target_filename) },
                    "dest": dest_path.to_string_lossy(),
                    "replay_handle": replay_handle,
                    "replay_handles_new": replay_handles_new,
                    "stripped_consolecmds": stripped_count,
                    "suspects": player_algos.iter().map(|(sid, algos)| {
                        serde_json::json!({
                            "steam_id": sid.to_string(),
                            "name": analyser.state.get_player_name(*sid).unwrap_or("unknown"),
                            "detections": algos.len(),
                            "algorithms": algos,
                        })
                    }).collect::<Vec<_>>(),
                    "ignored_marked_detections": ignored_marked_count,
                    "ignored_deleted_detections": ignored_deleted_count,
                }));
            } else {
                let total_ignored = ignored_marked_count + ignored_deleted_count;
                if !silent {
                    if total_ignored > 0 {
                        let reason = if ignored_marked_count > 0 && ignored_deleted_count > 0 {
                            format!("all {} detection(s) were ignored: {} marked, {} deleted", total_ignored, ignored_marked_count, ignored_deleted_count)
                        } else if ignored_marked_count > 0 {
                            format!("all {} detection(s) were from marked cheaters", ignored_marked_count)
                        } else {
                            format!("all {} detection(s) were from deleted accounts", ignored_deleted_count)
                        };
                        if no_move {
                            if strip && use_scratch {
                                println!("CLEAN -> clean (stripped copy in scratch/, {})", reason);
                            } else {
                                println!("CLEAN -> clean (not moved, {})", reason);
                            }
                        } else if strip && use_scratch {
                            println!("CLEAN -> moved to scratch/clean/ ({})", reason);
                        } else {
                            println!("CLEAN -> moved to clean/ ({})", reason);
                        }
                    } else {
                        if no_move {
                            if strip && use_scratch {
                                println!("CLEAN -> clean (stripped copy in scratch/, 0 detections)");
                            } else {
                                println!("CLEAN -> clean (not moved, 0 detections)");
                            }
                        } else if strip && use_scratch {
                            println!("CLEAN -> moved to scratch/clean/ (0 detections)");
                        } else {
                            println!("CLEAN -> moved to clean/ (0 detections)");
                        }
                    }
                }
                clean_demos.push(serde_json::json!({
                    "demo": file_name_str,
                    "dest": dest_path.to_string_lossy(),
                    "stripped_consolecmds": stripped_count,
                    "ignored_marked_detections": ignored_marked_count,
                    "ignored_deleted_detections": ignored_deleted_count,
                }));
            }
        }
    });
    SILENT.store(silent, std::sync::atomic::Ordering::Relaxed);

    // Only the .json files directly in the demo directory are removed, not those in subfolders.
    let mut deleted_jsons: usize = 0;
    if cleanup_jsons {
        for entry in fs::read_dir(dir_path)? {
            let path = entry?.path();
            let is_json = path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("json"));
            if is_json {
                match fs::remove_file(&path) {
                    Ok(()) => deleted_jsons += 1,
                    Err(e) if !silent => eprintln!("ERROR: failed to delete '{}': {}", path.display(), e),
                    Err(_) => {}
                }
            }
        }
    }

    let elapsed = start_scan.elapsed().as_secs_f64();
    if !silent {
        println!("--------------------------------------------------");
        if cleanup_jsons {
            println!("Cleanup: deleted {} json file(s) from '{}'", deleted_jsons, dir_path.display());
        }
        if strip || strip_only {
            if use_scratch {
                println!(
                    "Bulk Scan Completed in {:.2}s (Stripped {} total consolecmds into '{}')",
                    elapsed,
                    total_consolecmds_stripped,
                    scratch_dir.display()
                );
            } else {
                println!(
                    "Bulk Scan Completed in {:.2}s (Stripped {} total consolecmds directly into replays)",
                    elapsed,
                    total_consolecmds_stripped
                );
            }
        } else {
            println!("Bulk Scan Completed in {:.2}s", elapsed);
        }
        if valve_only {
            println!("Total: {} | Clean: {} ({} non-Valve) | Sus: {}", total, clean_demos.len(), non_valve_count, sus_demos.len());
        } else {
            println!("Total: {} | Clean: {} | Sus: {}", total, clean_demos.len(), sus_demos.len());
        }
        if !failed_demos.is_empty() {
            println!("Failed to analyze: {} (not counted as clean or sus)", failed_demos.len());
            for d in &failed_demos {
                println!("  {}: {}", d["demo"].as_str().unwrap_or("?"), d["reason"].as_str().unwrap_or("?"));
            }
        }
        if !partial_demos.is_empty() {
            println!("Partially analyzed: {} (already counted above; corrupted or cut-off recordings)", partial_demos.len());
            for d in &partial_demos {
                println!("  {}: {}", d["demo"].as_str().unwrap_or("?"), d["reason"].as_str().unwrap_or("?"));
            }
        }
        println!("--------------------------------------------------");
    } else {
        let mut res = serde_json::json!({
            "directory": dir_path.to_string_lossy(),
            "elapsed_seconds": elapsed,
            "total": total,
            "clean_count": clean_demos.len(),
            "sus_count": sus_demos.len(),
            "sus_demos": sus_demos,
            "clean_demos": clean_demos,
            "failed_count": failed_demos.len(),
            "failed_demos": failed_demos,
            "partial_count": partial_demos.len(),
            "partial_demos": partial_demos
        });
        if strip || strip_only {
            if use_scratch {
                res["scratch_directory"] = serde_json::json!(scratch_dir.to_string_lossy());
            }
            res["total_consolecmds_stripped"] = serde_json::json!(total_consolecmds_stripped);
        }
        if valve_only {
            res["non_valve_count"] = serde_json::json!(non_valve_count);
        }
        if cleanup_jsons {
            res["deleted_jsons"] = serde_json::json!(deleted_jsons);
        }
        println!("{}", if pretty { serde_json::to_string_pretty(&res)? } else { serde_json::to_string(&res)? });
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let start = std::time::Instant::now();

    let mut opts = Options::new();
    opts.optopt("i", "input", "set input file path", "PATH");
    opts.optflagopt(
        "b",
        "bulk",
        "auto scan bulk demos in TF2 folder (or optional PATH) and sort into 'sus' and 'clean' subfolders",
        "DIR",
    );
    opts.optflagopt(
        "",
        "scan",
        "alias for --bulk",
        "DIR",
    );
    opts.optflagopt(
        "m",
        "ignore-marked",
        "ignore detections for cheaters in reported_ids.txt (downloads list if not found locally, or custom PATH)",
        "FILE",
    );
    opts.optflagopt(
        "",
        "ignore-reported",
        "alias for --ignore-marked",
        "FILE",
    );
    opts.optflag(
        "d",
        "ignore-deleted",
        "ignore detections for players whose Steam account no longer exists / was deleted",
    );
    opts.optflag(
        "q",
        "quiet",
        "silence all output except for the final JSON string",
    );
    opts.optflag(
        "Q",
        "quiet-pretty",
        "same as -q, but with more human-readable json",
    );
    opts.optflag("H", "header", "parse and display demo header information");
    opts.optflag(
        "P",
        "players",
        "parse and display player list with teams, classes, and events",
    );
    opts.optflag(
        "C",
        "chat",
        "parse and display all and team chat messages",
    );
    opts.optmulti("a", "algorithm", "specify the algorithm to run. Include multiple -a flags to run multiple algorithms. If not specified, the default algorithms are run.", "ALGORITHM [-a ALGORITHM]...");
    opts.optflag("c", "count", "only print the number of detections");
    opts.optflag("h", "help", "print this help menu");
    opts.optopt(
        "p",
        "params",
        "Load a JSON config FILE (e.g. params.json) with algorithm settings. This is the only option that loads a file; only algorithms enabled in it run (default: built-in algorithm defaults)",
        "FILE_PATH",
    );
    opts.optopt(
        "",
        "config",
        "alias for -p/--params (JSON config FILE path)",
        "FILE_PATH",
    );
    opts.optflag(
        "l",
        "list-algorithms",
        "list all available cheat detection algorithms and exit",
    );
    opts.optflag(
        "",
        "list",
        "alias for --list-algorithms",
    );
    opts.optflag(
        "",
        "list-algos",
        "alias for --list-algorithms",
    );
    opts.optflag(
        "n",
        "no-move",
        "do not move or sort demos into 'sus' and 'clean' folders during bulk scan",
    );
    opts.optflag(
        "",
        "no-sort",
        "alias for --no-move",
    );
    opts.optflag(
        "",
        "dry-run",
        "alias for --no-move",
    );
    opts.optflag(
        "",
        "cleanup-jsons",
        "during bulk scan, delete all .json files directly in the demo directory after the demos have been moved",
    );
    opts.optflag(
        "",
        "replay-new",
        "during bulk scan, create TF2 replays with takes per suspect (the GUI's \"Convert to replay\")",
    );
    opts.optflag(
        "",
        "replay-old",
        "during bulk scan, create basic TF2 replays without takes (the GUI's \"Convert to replay (old)\")",
    );
    opts.optmulti(
        "",
        "set",
        "Override ONE algorithm parameter inline (NOT a file). Format: ALGO.KEY=VALUE or ALGO:KEY=VALUE, e.g. --set fidoo/psilent5.enabled=true. Repeatable; applied on top of -p/--params if given",
        "ALGO.KEY=VALUE",
    );
    opts.optmulti(
        "",
        "override-json",
        "Override parameters with an inline JSON STRING (NOT a file), e.g. --override-json '{\"fidoo/psilent5\":{\"enabled\":true}}'. Repeatable; applied on top of -p/--params if given",
        "JSON_STRING",
    );
    opts.optflag(
        "s",
        "strip",
        "strip consolecmds from demo(s) into a scratch directory (or directly into TF2 replay directory when combined with replay creation) without directly modifying original demos",
    );
    opts.optflag(
        "",
        "strip-only",
        "only strip consolecmds into a scratch directory and exit without running analysis",
    );
    opts.optopt(
        "",
        "scratch-dir",
        "custom scratch directory where stripped demo copies will be written (default: ./scratch for single demo, <DIR>/scratch for bulk scan)",
        "DIR",
    );
    opts.optflag(
        "",
        "valve-only",
        "during bulk scan, do not analyse demos from non-Valve servers and count them as clean (like demos whose detections were all ignored)",
    );

    fn print_help(opts: &getopts::Options) {
        println!("{}", opts.usage("Usage: analysis-template [options]"));
    }

    let matches = match opts.parse(env::args().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: {}", e);
            print_help(&opts);
            return Ok(());
        }
    };

    if matches.opt_present("h") {
        print_help(&opts);
        return Ok(());
    }

    let silent = matches.opt_present("q") || matches.opt_present("Q");
    let pretty = matches.opt_present("Q");
    SILENT.store(silent, std::sync::atomic::Ordering::Relaxed);

    let strip = matches.opt_present("s") || matches.opt_present("strip");
    let strip_only = matches.opt_present("strip-only");
    let custom_scratch_dir = matches.opt_str("scratch-dir").map(PathBuf::from);

    // Check if ignoring marked cheaters is requested
    let ignore_marked = matches.opt_present("m")
        || matches.opt_present("ignore-marked")
        || matches.opt_present("ignore-reported");
    let marked_file_arg = matches
        .opt_str("m")
        .or_else(|| matches.opt_str("ignore-marked"))
        .or_else(|| matches.opt_str("ignore-reported"));
    let ignore_deleted = matches.opt_present("d") || matches.opt_present("ignore-deleted");

    let reported_ids: Option<HashSet<u64>> = if ignore_marked || ignore_deleted {
        Some(load_or_download_reported_ids(marked_file_arg.as_deref(), silent)?)
    } else {
        None
    };

    // Load parameters / config ONLY if explicitly passed via -p, --params, or --config.
    // When a file is given, only algorithms enabled in that file run.
    let mut config: Option<HashMap<String, Parameters>> = None;
    let explicit_param_path = matches
        .opt_str("p")
        .or_else(|| matches.opt_str("params"))
        .or_else(|| matches.opt_str("config"));

    if let Some(ref path) = explicit_param_path {
        dev_print!("Loading parameters from {}:", path);
        match fs::read(path) {
            Ok(c) => match serde_json::from_slice::<HashMap<String, Parameters>>(&c) {
                Ok(mut parsed_config) => {
                    // Session-only overrides (recorder_id, target_player) come from --set, never from a file
                    strip_session_only_parameters(&mut parsed_config);
                    // A specified file is authoritative: algorithms that don't
                    // explicitly set `enabled` in it are turned off.
                    for algo in get_algorithms() {
                        parsed_config
                            .entry(algo.algorithm_name().to_string())
                            .or_default()
                            .entry("enabled".to_string())
                            .or_insert(Parameter::Bool(false));
                    }
                    config = Some(parsed_config);
                }
                Err(e) => {
                    eprintln!("Error: Failed to parse parameter file '{}': {}", path, e);
                    return Err(e.into());
                }
            },
            Err(e) => {
                eprintln!("Error: Failed to read parameter file '{}': {}", path, e);
                return Err(e.into());
            }
        }
    }

    let specified_algorithms = matches.opt_strs("a");

    // Apply CLI parameter overrides
    let cli_params: Vec<String> = matches.opt_strs("set");
    let cli_param_jsons: Vec<String> = matches.opt_strs("override-json");

    if (!cli_params.is_empty() || !cli_param_jsons.is_empty()) && config.is_none() {
        config = Some(HashMap::new());
    }

    if let Some(ref mut cfg) = config {
        for json_str in cli_param_jsons {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(obj) = val.as_object() {
                    for (k, v) in obj {
                        if let Some(inner_obj) = v.as_object() {
                            let entry = cfg.entry(k.clone()).or_default();
                            for (ik, iv) in inner_obj {
                                if let Ok(param) = serde_json::from_value::<Parameter>(iv.clone()) {
                                    entry.insert(ik.clone(), param);
                                }
                            }
                        } else if let Ok(param) = serde_json::from_value::<Parameter>(v.clone()) {
                            if !specified_algorithms.is_empty() {
                                for algo in &specified_algorithms {
                                    cfg.entry(algo.clone()).or_default().insert(k.clone(), param.clone());
                                }
                            } else {
                                cfg.entry(k.clone()).or_default().insert(k.clone(), param.clone());
                            }
                        }
                    }
                }
            } else {
                eprintln!("Warning: Failed to parse --override-json string as JSON: {}", json_str);
            }
        }

        fn parse_cli_param_val(val_str: &str) -> Parameter {
            let trimmed = val_str.trim();
            if trimmed.eq_ignore_ascii_case("true") {
                Parameter::Bool(true)
            } else if trimmed.eq_ignore_ascii_case("false") {
                Parameter::Bool(false)
            } else if trimmed.contains('.') {
                if let Ok(f) = trimmed.parse::<f32>() {
                    Parameter::Float(f)
                } else {
                    Parameter::String(trimmed.to_string())
                }
            } else if let Ok(u) = trimmed.parse::<u64>() {
                Parameter::UInt(u)
            } else if let Ok(i) = trimmed.parse::<i32>() {
                Parameter::Int(i)
            } else {
                Parameter::String(trimmed.to_string())
            }
        }

        for param_str in cli_params {
            if let Some((raw_k, raw_v)) = param_str.split_once('=') {
                let parsed_val = parse_cli_param_val(raw_v);
                let trimmed_k = raw_k.trim();
                let is_general_setting = matches!(
                    trimmed_k,
                    "sliding_window_ticks" | "sliding_window_detections" | "global_sliding_window_enabled" | "per_algorithm"
                );

                if let Some((algo, param_name)) = trimmed_k.split_once('.').or_else(|| trimmed_k.split_once(':')) {
                    cfg.entry(algo.to_string()).or_default().insert(param_name.to_string(), parsed_val);
                } else if is_general_setting {
                    cfg.entry("general_settings".to_string()).or_default().insert(trimmed_k.to_string(), parsed_val);
                } else if !specified_algorithms.is_empty() {
                    for algo in &specified_algorithms {
                        cfg.entry(algo.clone()).or_default().insert(trimmed_k.to_string(), parsed_val.clone());
                    }
                } else {
                    cfg.entry(trimmed_k.to_string()).or_default().insert(trimmed_k.to_string(), parsed_val);
                }
            } else {
                eprintln!("Warning: Invalid --set format '{}', expected KEY=VALUE or ALGO.KEY=VALUE", param_str);
            }
        }
    }

    let list_requested = matches.opt_present("l")
        || matches.opt_present("list-algorithms")
        || matches.opt_present("list")
        || matches.opt_present("list-algos");

    if list_requested {
        return print_algorithms_list(&config, silent, pretty);
    }

    // Validate algorithm names against all available algorithms
    let all_available = get_algorithms();
    let unknown_algorithms: Vec<String> = specified_algorithms
        .iter()
        .filter(|a| all_available.iter().all(|b| b.algorithm_name() != a.as_str()))
        .cloned()
        .collect();
    if !unknown_algorithms.is_empty() {
        panic!(
            "Unknown algorithms specified: {}",
            unknown_algorithms.join(", ")
        );
    }

    // Check whether bulk mode is active
    let bulk_flag_present = matches.opt_present("b")
        || matches.opt_present("bulk")
        || matches.opt_present("scan");
    let input_opt = matches.opt_str("i");

    // Also detect if -i was passed pointing to an existing directory
    let input_is_dir = input_opt.as_ref().map_or(false, |p| Path::new(p).is_dir());

    if bulk_flag_present || input_is_dir {
        // Resolve directory to scan
        let bulk_dir_arg = matches
            .opt_str("b")
            .or_else(|| matches.opt_str("bulk"))
            .or_else(|| matches.opt_str("scan"))
            .or_else(|| matches.free.first().cloned())
            .or_else(|| {
                if input_is_dir {
                    input_opt.clone()
                } else {
                    None
                }
            });

        let target_dir = match bulk_dir_arg {
            Some(dir_str) => {
                let p = PathBuf::from(dir_str);
                if !p.exists() {
                    eprintln!("Error: Directory '{}' does not exist.", p.display());
                    return Ok(());
                }
                if !p.is_dir() {
                    eprintln!("Error: '{}' is not a directory.", p.display());
                    return Ok(());
                }
                p
            }
            None => match find_default_tf2_dir() {
                Some(dir) => dir,
                None => {
                    eprintln!("Error: Could not locate default Team Fortress 2 demo directory ('tf').");
                    eprintln!("Please specify the demo directory explicitly: --bulk <PATH>");
                    return Ok(());
                }
            },
        };

        let no_move = matches.opt_present("n")
            || matches.opt_present("no-move")
            || matches.opt_present("no-sort")
            || matches.opt_present("dry-run");

        return run_bulk_scan(
            &target_dir,
            &config,
            &specified_algorithms,
            &reported_ids,
            ignore_marked,
            ignore_deleted,
            no_move,
            matches.opt_present("replay-old"),
            matches.opt_present("replay-new"),
            silent,
            pretty,
            strip,
            strip_only,
            matches.opt_present("valve-only"),
            custom_scratch_dir,
            matches.opt_present("cleanup-jsons"),
        );
    }

    // Single demo analysis mode
    let demo_path = match input_opt {
        Some(p) => p,
        None => {
            eprintln!("Error: No input demo file provided.");
            eprintln!("Usage:");
            eprintln!("  Single demo: cli -i <path/to/demo.dem> [options]");
            eprintln!("  Bulk scan:   cli --bulk [optional/path/to/tf] [options]");
            print_help(&opts);
            return Ok(());
        }
    };

    // If --strip or --strip-only was specified:
    let (file_bytes, target_demo_path) = if strip || strip_only {
        let raw = fs::read(&demo_path)?;
        let (stripped_bytes, count) = strip_consolecmds_from_bytes(&raw)
            .map_err(|e| anyhow::anyhow!("Failed to strip consolecmds from '{}': {}", demo_path, e))?;

        let scratch_dir = custom_scratch_dir.clone().unwrap_or_else(|| PathBuf::from("scratch"));
        fs::create_dir_all(&scratch_dir)?;
        let filename = Path::new(&demo_path)
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("stripped.dem"));
        let scratch_path = scratch_dir.join(filename);
        fs::write(&scratch_path, &stripped_bytes)?;

        if !silent {
            println!(
                "Stripped {} consolecmd(s) from '{}' -> '{}'",
                count,
                demo_path,
                scratch_path.display()
            );
        }

        if strip_only {
            if silent {
                let res = serde_json::json!({
                    "demo": demo_path,
                    "stripped_to": scratch_path.to_string_lossy(),
                    "stripped_consolecmds": count,
                });
                println!("{}", if pretty { serde_json::to_string_pretty(&res)? } else { serde_json::to_string(&res)? });
            }
            return Ok(());
        }

        (stripped_bytes, scratch_path.to_string_lossy().to_string())
    } else {
        let raw = fs::read(&demo_path)?;
        (raw, demo_path.clone())
    };

    // Feature 1: Header CLI
    if matches.opt_present("H") {
        let header = HeaderData::parse_file(&target_demo_path)?;
        let json_str = if silent && !pretty {
            serde_json::to_string(&header)?
        } else {
            serde_json::to_string_pretty(&header)?
        };
        println!("{}", json_str);
        if !matches.opt_present("P") && !matches.opt_present("C") && !matches.opt_present("a") {
            return Ok(());
        }
    }

    // Feature 2: Playerlist CLI
    if matches.opt_present("P") {
        let players = parse_playerlist(&target_demo_path)?;
        let json_str = if silent && !pretty {
            serde_json::to_string(&players)?
        } else {
            serde_json::to_string_pretty(&players)?
        };
        println!("{}", json_str);
        if !matches.opt_present("C") && !matches.opt_present("a") {
            return Ok(());
        }
    }

    // Feature 3: Chat CLI
    if matches.opt_present("C") {
        let messages = parse_chat(&target_demo_path)?;
        let json_str = if silent && !pretty {
            serde_json::to_string(&messages)?
        } else {
            serde_json::to_string_pretty(&messages)?
        };
        println!("{}", json_str);
        if !matches.opt_present("a") {
            return Ok(());
        }
    }

    let algorithms = prepare_algorithms(&config, &specified_algorithms);
    if algorithms.is_empty() {
        panic!("No algorithms specified or enabled");
    }

    let general_settings = GeneralSettings::from_config_opt(&config);
    let mut analyser = analyse_bytes_with_settings(&file_bytes, algorithms, &general_settings)?;

    // If ignoring marked cheaters, filter detections
    if ignore_marked {
        if let Some(ref marked_ids) = reported_ids {
            let before_count = analyser.detections.len();
            analyser.detections.retain(|d| !marked_ids.contains(&d.player));
            let ignored = before_count - analyser.detections.len();
            if ignored > 0 && !silent {
                dev_print!("Ignored {} detection(s) from already marked cheaters", ignored);
            }
        }
    }

    if ignore_deleted {
        let mut account_checker = SteamAccountChecker::new();
        let unique_players: HashSet<u64> = analyser.detections.iter().map(|d| d.player).filter(|&p| p != 0).collect();
        for &player in &unique_players {
            let is_marked = reported_ids.as_ref().map_or(false, |ids| ids.contains(&player));
            if !is_marked {
                account_checker.get_status(player);
            }
        }
        let before_count = analyser.detections.len();
        analyser.detections.retain(|d| {
            let is_marked = reported_ids.as_ref().map_or(false, |ids| ids.contains(&d.player));
            if !is_marked && d.player != 0 && account_checker.get_status(d.player) == AccountStatus::Deleted {
                false
            } else {
                true
            }
        });
        let ignored = before_count - analyser.detections.len();
        if ignored > 0 && !silent {
            dev_print!("Ignored {} detection(s) from deleted Steam accounts", ignored);
        }
    }

    if start.elapsed().as_secs() >= 10 {
        analyser.print_metadata();
    }

    if SILENT.load(std::sync::atomic::Ordering::Relaxed) {
        analyser.print_detection_json(pretty);
    } else if matches.opt_present("c") {
        analyser.print_detection_summary();
    } else {
        analyser.print_detection_json(true);
    }

    let total_ticks = analyser.get_tick_count_u32();
    let total_time = start.elapsed().as_secs_f64();
    let total_tps = (total_ticks as f64) / total_time;
    dev_print!(
        "Done! (Processed {} ticks in {:.2} seconds averaging {:.2} tps)",
        total_ticks,
        total_time,
        total_tps
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steam_account_status_deleted_and_exists() {
        let existing_xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><profile><steamID64>76561197960287930</steamID64></profile>";
        assert_eq!(parse_steam_profile_xml(existing_xml), AccountStatus::Exists);

        let deleted_xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><response><error><![CDATA[The specified profile could not be found.]]></error></response>";
        assert_eq!(parse_steam_profile_xml(deleted_xml), AccountStatus::Deleted);

        let transient_error_xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><response><error><![CDATA[Failed loading profile data, please try again later.]]></error></response>";
        assert_eq!(parse_steam_profile_xml(transient_error_xml), AccountStatus::Unknown);

        let html_error = "<html><body>Error</body></html>";
        assert_eq!(parse_steam_profile_xml(html_error), AccountStatus::Unknown);
    }

    #[test]
    fn test_steam_account_checker_caching_and_player_zero() {
        let mut checker = SteamAccountChecker::new();
        // ID 0 should never query steam and always be Exists
        assert_eq!(checker.get_status(0), AccountStatus::Exists);
        assert!(!checker.cache.contains_key(&0));

        // Insert into cache manually to verify cache hit
        checker.cache.insert(12345, AccountStatus::Deleted);
        assert_eq!(checker.get_status(12345), AccountStatus::Deleted);
    }

    #[test]
    fn test_steam_live_76561198666787621() {
        let url = format!("https://steamcommunity.com/profiles/76561198666787621?xml=1");
        let resp = ureq::get(&url).timeout(std::time::Duration::from_secs(10)).call();
        match resp {
            Ok(r) => {
                let status_code = r.status();
                let body = r.into_string().unwrap_or_default();
                println!("HTTP Status: {}, Body: {:?}", status_code, body);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
        let status = check_steam_account(76561198666787621);
        println!("check_steam_account status: {:?}", status);
    }

    #[test]
    fn test_filtering_logic_marked_and_deleted() {
        use analysis_template::lib::algorithm::Detection;

        let marked_id = 11111u64;
        let deleted_id = 22222u64;
        let legit_id = 33333u64;

        let mut reported_ids = HashSet::new();
        reported_ids.insert(marked_id);
        let reported_ids_opt = Some(reported_ids);

        let mut checker = SteamAccountChecker::new();
        checker.cache.insert(deleted_id, AccountStatus::Deleted);
        checker.cache.insert(legit_id, AccountStatus::Exists);

        let detections = vec![
            Detection {
                tick: 100,
                player: marked_id,
                algorithm: "aimbot".to_string(),
                data: serde_json::json!({}),
            },
            Detection {
                tick: 200,
                player: deleted_id,
                algorithm: "aimbot".to_string(),
                data: serde_json::json!({}),
            },
            Detection {
                tick: 300,
                player: legit_id,
                algorithm: "aimbot".to_string(),
                data: serde_json::json!({}),
            },
        ];

        // Scenario 1: ignore_marked=true, ignore_deleted=true
        {
            let mut qual = Vec::new();
            let mut ign_marked = 0;
            let mut ign_deleted = 0;

            for d in &detections {
                let is_marked = reported_ids_opt.as_ref().map_or(false, |ids| ids.contains(&d.player));
                if is_marked {
                    ign_marked += 1;
                } else if d.player != 0 && checker.get_status(d.player) == AccountStatus::Deleted {
                    ign_deleted += 1;
                } else {
                    qual.push(d.clone());
                }
            }

            assert_eq!(ign_marked, 1);
            assert_eq!(ign_deleted, 1);
            assert_eq!(qual.len(), 1);
            assert_eq!(qual[0].player, legit_id);
        }

        // Scenario 2: ignore_marked=false, ignore_deleted=true
        // Marked players should NOT be queried on steam, nor should they be ignored
        {
            let mut qual = Vec::new();
            let mut ign_deleted = 0;

            for d in &detections {
                let is_marked = reported_ids_opt.as_ref().map_or(false, |ids| ids.contains(&d.player));
                if is_marked {
                    // Not ignored because ignore_marked is false
                    qual.push(d.clone());
                } else if d.player != 0 && checker.get_status(d.player) == AccountStatus::Deleted {
                    ign_deleted += 1;
                } else {
                    qual.push(d.clone());
                }
            }

            assert_eq!(ign_deleted, 1);
            assert_eq!(qual.len(), 2);
            assert!(qual.iter().any(|d| d.player == marked_id));
            assert!(qual.iter().any(|d| d.player == legit_id));
        }

        // Scenario 3: ignore_marked=true, ignore_deleted=false
        {
            let mut qual = Vec::new();
            let mut ign_marked = 0;

            for d in &detections {
                let is_marked = reported_ids_opt.as_ref().map_or(false, |ids| ids.contains(&d.player));
                if is_marked {
                    ign_marked += 1;
                } else {
                    qual.push(d.clone());
                }
            }

            assert_eq!(ign_marked, 1);
            assert_eq!(qual.len(), 2);
            assert!(qual.iter().any(|d| d.player == deleted_id));
            assert!(qual.iter().any(|d| d.player == legit_id));
        }
    }

    #[test]
    fn test_strip_single_demo_flow() {
        let temp_dir = std::env::temp_dir();
        let run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let test_demo = temp_dir.join(format!("test_single_{}.dem", run_id));
        let scratch_dir = temp_dir.join(format!("scratch_single_{}", run_id));

        // Create mock demo with 1 consolecmd
        let mut mock_data = vec![0u8; 1072];
        mock_data[0..8].copy_from_slice(b"HL2DEMO\0");
        mock_data[8..12].copy_from_slice(&4i32.to_le_bytes());
        mock_data.push(4); // consolecmd
        mock_data.extend_from_slice(&10i32.to_le_bytes());
        mock_data.push(0);
        mock_data.extend_from_slice(&4i32.to_le_bytes());
        mock_data.extend_from_slice(b"say\0");
        mock_data.push(7); // stop

        fs::write(&test_demo, &mock_data).unwrap();

        // Simulate single demo stripping
        let raw = fs::read(&test_demo).unwrap();
        let (stripped_bytes, count) = strip_consolecmds_from_bytes(&raw).unwrap();
        assert_eq!(count, 1);

        fs::create_dir_all(&scratch_dir).unwrap();
        let scratch_path = scratch_dir.join("test_single.dem");
        fs::write(&scratch_path, &stripped_bytes).unwrap();

        // Original demo must remain byte-for-byte untouched
        let original_check = fs::read(&test_demo).unwrap();
        assert_eq!(original_check, mock_data);

        // Stripped file exists in scratch and is smaller
        assert!(scratch_path.exists());
        assert!(stripped_bytes.len() < mock_data.len());

        let _ = fs::remove_file(&test_demo);
        let _ = fs::remove_dir_all(&scratch_dir);
    }

    #[test]
    fn test_strip_bulk_scan_non_destructive() {
        let temp_dir = std::env::temp_dir();
        let run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let bulk_dir = temp_dir.join(format!("test_bulk_{}", run_id));
        fs::create_dir_all(&bulk_dir).unwrap();

        let demo1_path = bulk_dir.join("demo1.dem");
        let demo2_path = bulk_dir.join("demo2.dem");

        // demo1: contains consolecmd
        let mut mock_data1 = vec![0u8; 1072];
        mock_data1[0..8].copy_from_slice(b"HL2DEMO\0");
        mock_data1[8..12].copy_from_slice(&4i32.to_le_bytes());
        mock_data1.push(4); // consolecmd
        mock_data1.extend_from_slice(&10i32.to_le_bytes());
        mock_data1.push(0);
        mock_data1.extend_from_slice(&4i32.to_le_bytes());
        mock_data1.extend_from_slice(b"say\0");
        mock_data1.push(7); // stop
        fs::write(&demo1_path, &mock_data1).unwrap();

        // demo2: no consolecmds
        let mut mock_data2 = vec![0u8; 1072];
        mock_data2[0..8].copy_from_slice(b"HL2DEMO\0");
        mock_data2[8..12].copy_from_slice(&4i32.to_le_bytes());
        mock_data2.push(3); // synctick
        mock_data2.extend_from_slice(&5i32.to_le_bytes());
        mock_data2.push(0);
        mock_data2.push(7); // stop
        fs::write(&demo2_path, &mock_data2).unwrap();

        // Run bulk scan with strip = true and strip_only = true (so cheat analysis isn't needed on mock demos)
        let res = run_bulk_scan(
            &bulk_dir,
            &None,
            &[],
            &None,
            false,
            false,
            false,
            false,
            false,
            true, // silent
            false,
            true, // strip
            true, // strip_only
            false, // valve_only
            None,
            false, // cleanup_jsons
        );
        assert!(res.is_ok());

        // CRITICAL CHECK: Originals must still exist in bulk_dir untouched!
        assert!(demo1_path.exists(), "Original demo1.dem must not be moved or deleted");
        assert!(demo2_path.exists(), "Original demo2.dem must not be moved or deleted");
        assert_eq!(fs::read(&demo1_path).unwrap(), mock_data1, "Original demo1.dem content must remain identical");
        assert_eq!(fs::read(&demo2_path).unwrap(), mock_data2, "Original demo2.dem content must remain identical");

        // The stripped demos should exist in bulk_dir/scratch
        let scratch_demo1 = bulk_dir.join("scratch").join("demo1.dem");
        let scratch_demo2 = bulk_dir.join("scratch").join("demo2.dem");
        assert!(scratch_demo1.exists(), "Stripped demo1 must exist in scratch/");
        assert!(scratch_demo2.exists(), "Stripped demo2 must exist in scratch/");
        assert!(fs::metadata(&scratch_demo1).unwrap().len() < mock_data1.len() as u64);

        let _ = fs::remove_dir_all(&bulk_dir);
    }

    #[test]
    fn test_strip_with_replay_no_scratch() {
        let temp_dir = std::env::temp_dir();
        let run_id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root = temp_dir.join(format!("test_replay_strip_{}", run_id));
        let tf_dir = root.join("tf");
        let bulk_dir = tf_dir.join("demos");
        let replays_dir = tf_dir.join("replay").join("client").join("replays");
        fs::create_dir_all(&bulk_dir).unwrap();
        fs::create_dir_all(&replays_dir).unwrap();

        let demo_path = bulk_dir.join("demo_sus.dem");

        // Mock demo containing a consolecmd
        let mut mock_data = vec![0u8; 1072];
        mock_data[0..8].copy_from_slice(b"HL2DEMO\0");
        mock_data[8..12].copy_from_slice(&4i32.to_le_bytes());
        mock_data.push(4); // consolecmd
        mock_data.extend_from_slice(&10i32.to_le_bytes());
        mock_data.push(0);
        mock_data.extend_from_slice(&4i32.to_le_bytes());
        mock_data.extend_from_slice(b"say\0");
        mock_data.push(7); // stop
        fs::write(&demo_path, &mock_data).unwrap();

        // 1. Verify write_stripped_demo_to_replays writes stripped bytes directly into replays_dir
        let (stripped_bytes, count) = strip_consolecmds_from_bytes(&mock_data).unwrap();
        assert_eq!(count, 1);
        let written_target = write_stripped_demo_to_replays(&demo_path, &stripped_bytes).unwrap();
        assert_eq!(written_target, replays_dir.join("demo_sus.dem"));
        assert!(written_target.exists());
        assert_eq!(fs::read(&written_target).unwrap(), stripped_bytes);

        // 2. Verify that run_bulk_scan with strip = true and replay_new = true does NOT create scratch
        let res = run_bulk_scan(
            &bulk_dir,
            &None,
            &[],
            &None,
            false,
            false,
            true, // no_move
            false, // replay_old
            true,  // replay_new
            true,  // silent
            false,
            true,  // strip
            false, // strip_only
            false, // valve_only
            None,
            false, // cleanup_jsons
        );
        assert!(res.is_ok());

        // CRITICAL CHECK: Scratch directory must NOT exist when running strip with replay creation!
        let scratch_dir = bulk_dir.join("scratch");
        assert!(!scratch_dir.exists(), "Scratch directory must NOT be created when running strip with replay creation");

        // Original demo must still exist untouched in content
        assert!(demo_path.exists());
        assert_eq!(fs::read(&demo_path).unwrap(), mock_data, "Original demo content must remain intact and unstripped");

        let _ = fs::remove_dir_all(&root);
    }
}
