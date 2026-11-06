// Per-algorithm cost estimates used to balance algorithms across the parallel workers.
//
// Builds that include experimental algorithms learn the costs at runtime: the analyser times every
// algorithm on a sample of ticks and the results are averaged into `algo_costs.json` in the project
// root (next to the executable if the project folder does not exist).
// Stable builds embed a snapshot of that file at compile time and never touch the disk.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

include!(concat!(env!("OUT_DIR"), "/algorithm_hashes.rs"));

/// True for builds that learn algorithm costs and keep them in `algo_costs.json`.
pub const COST_LEARNING: bool = cfg!(algo_cost_learning);

/// Algorithms are timed on one out of every `SAMPLE_INTERVAL` ticks.
pub const SAMPLE_INTERVAL: u32 = 8;

/// Weight of a new demo's measurement in the running average.
const NEW_SAMPLE_WEIGHT: f64 = 0.25;

const PROFILE_FILE: &str = "algo_costs.json";

#[derive(Serialize, Deserialize, Default, Clone)]
struct CostProfile {
    algorithms: BTreeMap<String, AlgorithmCost>,
}

#[derive(Serialize, Deserialize, Clone)]
struct AlgorithmCost {
    /// CPU time spent in the algorithm per 1000 demo ticks.
    ms_per_1k_ticks: f64,
    /// Number of demos averaged into `ms_per_1k_ticks`.
    samples: u32,
    /// Hash of the algorithm's source file when it was measured.
    source_hash: String,
}

static PROFILE: Mutex<Option<CostProfile>> = Mutex::new(None);

fn with_profile<R>(f: impl FnOnce(&mut CostProfile) -> R) -> R {
    let mut guard = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let profile = guard.get_or_insert_with(load_profile);
    f(profile)
}

fn builtin_profile() -> CostProfile {
    serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/builtin_costs.json"))).unwrap_or_default()
}

fn load_profile() -> CostProfile {
    if !COST_LEARNING {
        return builtin_profile();
    }
    profile_path()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn profile_path() -> Option<PathBuf> {
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    if project_dir.is_dir() {
        return Some(project_dir.join(PROFILE_FILE));
    }
    Some(std::env::current_exe().ok()?.parent()?.join(PROFILE_FILE))
}

/// Source hashes of every compiled-in algorithm, keyed by algorithm name.
fn source_hashes() -> &'static HashMap<String, String> {
    static HASHES: OnceLock<HashMap<String, String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let stable = crate::lib::algorithm::get_proven_algorithms();
        let experimental = crate::lib::algorithm::get_experimental_algorithms();
        stable
            .iter()
            .zip(STABLE_SOURCE_HASHES)
            .chain(experimental.iter().zip(EXPERIMENTAL_SOURCE_HASHES))
            .map(|(algorithm, hash)| (algorithm.algorithm_name().to_string(), format!("{:016x}", hash)))
            .collect()
    })
}

/// Estimated cost of each named algorithm. Algorithms without a (current) measurement get the
/// median of the known costs, or 1.0 if nothing is known.
pub fn estimate_costs(names: &[&str]) -> Vec<f64> {
    let known: Vec<Option<f64>> = with_profile(|profile| {
        names
            .iter()
            .map(|name| {
                let cost = profile.algorithms.get(*name)?;
                if COST_LEARNING && source_hashes().get(*name) != Some(&cost.source_hash) {
                    return None;
                }
                Some(cost.ms_per_1k_ticks).filter(|ms| ms.is_finite() && *ms > 0.0)
            })
            .collect()
    });

    let mut sorted: Vec<f64> = known.iter().flatten().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let fallback = sorted.get(sorted.len() / 2).copied().unwrap_or(1.0);
    known.into_iter().map(|cost| cost.unwrap_or(fallback)).collect()
}

/// A worker count is only worth it if its slowest worker finishes within this factor of the best
/// achievable; every extra worker parses the whole demo again.
const WORKER_COUNT_TOLERANCE: f64 = 1.05;

/// Number of workers for the given algorithm costs: at most the physical CPU cores (hyperthreads
/// share execution units and only add parsing overhead) and the number of algorithms, and no more
/// than the costs can keep busy. The most expensive algorithm always occupies one worker on its
/// own, so beyond some point extra workers only sit idle.
pub fn worker_count(costs: &[f64]) -> usize {
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let max_workers = num_cpus::get_physical()
        .min(logical)
        .min(costs.len())
        .min(crate::MAX_WORKERS)
        .max(1);

    let slowest: Vec<f64> = (1..=max_workers).map(|workers| slowest_worker(costs, workers)).collect();
    let best = slowest.iter().copied().fold(f64::INFINITY, f64::min);
    slowest
        .iter()
        .position(|load| *load <= best * WORKER_COUNT_TOLERANCE)
        .map_or(1, |index| index + 1)
}

/// Load of the most loaded worker when the costs are assigned with `assign_workers`.
fn slowest_worker(costs: &[f64], workers: usize) -> f64 {
    let mut loads = vec![0.0f64; workers.max(1)];
    for (index, worker) in assign_workers(costs, workers).into_iter().enumerate() {
        loads[worker] += costs[index];
    }
    loads.into_iter().fold(0.0, f64::max)
}

/// Assigns algorithms to workers, most expensive first, each to the least loaded worker.
/// Returns the worker index for each algorithm.
pub fn assign_workers(costs: &[f64], workers: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by(|a, b| costs[*b].total_cmp(&costs[*a]));

    let mut loads = vec![0.0f64; workers.max(1)];
    let mut assignment = vec![0; costs.len()];
    for index in order {
        let (worker, _) = loads
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        loads[worker] += costs[index];
        assignment[index] = worker;
    }
    assignment
}

/// Averages one demo's measured algorithm times (seconds) into the learned profile.
pub(crate) fn record(measurements: &[(String, f64)], ticks: u32) {
    if !COST_LEARNING || ticks == 0 {
        return;
    }
    let hashes = source_hashes();
    with_profile(|profile| {
        for (name, seconds) in measurements {
            // Only learn compiled-in algorithms, not ad-hoc inspectors from the debug binaries.
            let Some(hash) = hashes.get(name) else {
                continue;
            };
            let observed = seconds * 1000.0 * 1000.0 / f64::from(ticks);
            if !observed.is_finite() {
                continue;
            }
            match profile.algorithms.get_mut(name) {
                Some(cost) if cost.source_hash == *hash && cost.samples > 0 => {
                    cost.ms_per_1k_ticks += (observed - cost.ms_per_1k_ticks) * NEW_SAMPLE_WEIGHT;
                    cost.samples = cost.samples.saturating_add(1);
                }
                _ => {
                    profile.algorithms.insert(
                        name.clone(),
                        AlgorithmCost {
                            ms_per_1k_ticks: observed,
                            samples: 1,
                            source_hash: hash.clone(),
                        },
                    );
                }
            }
        }
    });
}

/// Writes the learned profile to disk.
pub(crate) fn save() {
    if !COST_LEARNING {
        return;
    }
    let Some(path) = profile_path() else {
        return;
    };
    // The file is written while holding the profile lock, so threads analysing different demos
    // at the same time (bulk scans) never write it concurrently.
    with_profile(|profile| {
        let Ok(json) = serde_json::to_string_pretty(profile) else {
            return;
        };
        // Write to a temporary file first so another process never reads a partial file.
        let tmp_path = path.with_extension(format!("json.{}.tmp", std::process::id()));
        if std::fs::write(&tmp_path, json).is_ok() && std::fs::rename(&tmp_path, &path).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_most_expensive_first_to_least_loaded() {
        let costs = [1.0, 5.0, 2.0, 2.0, 1.0];
        let assignment = assign_workers(&costs, 2);
        let mut loads = [0.0; 2];
        for (index, worker) in assignment.iter().enumerate() {
            loads[*worker] += costs[index];
        }
        // The expensive algorithm does not share a worker with the two medium ones.
        assert_ne!(assignment[1], assignment[2]);
        assert_ne!(assignment[1], assignment[3]);
        assert_eq!(loads[0] + loads[1], 11.0);
        assert!((loads[0] - loads[1]).abs() <= 1.0);
    }

    #[test]
    fn worker_count_is_limited_by_costs() {
        let cores = num_cpus::get_physical()
            .min(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
            .min(crate::MAX_WORKERS);
        assert_eq!(worker_count(&[1.0]), 1);
        // One dominant algorithm: a second worker would barely help.
        assert_eq!(worker_count(&[10.0, 0.1, 0.1]), 1);
        // Many equal algorithms: use every core.
        assert_eq!(worker_count(&[1.0; 64]), cores);
    }
}
