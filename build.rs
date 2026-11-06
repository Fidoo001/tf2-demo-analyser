use std::env;
use std::fs;
use std::path::Path;

fn extract_struct_name(content: &str) -> Option<String> {
    // Look for `impl ... CheatAlgorithm ... for StructName`
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("impl") && trimmed.contains("CheatAlgorithm") && trimmed.contains("for ") {
            if let Some(pos) = trimmed.rfind("for ") {
                let after = trimmed[pos + 4..].trim();
                let struct_name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !struct_name.is_empty() {
                    return Some(struct_name);
                }
            }
        }
    }
    // Fallback: look for `pub struct StructName`
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("pub struct ") {
            let after = trimmed["pub struct ".len()..].trim();
            let struct_name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !struct_name.is_empty() {
                return Some(struct_name);
            }
        }
    }
    None
}

fn write_if_changed(path: &Path, content: &str) {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == content {
            return;
        }
    }
    fs::write(path, content).unwrap_or_else(|e| panic!("Failed to write {}: {}", path.display(), e));
}

/// FNV-1a hash of an algorithm's source file, used to reset its learned cost when it changes.
fn source_hash(folder_path: &Path, module: &str) -> u64 {
    let content = fs::read(folder_path.join(format!("{}.rs", module))).unwrap_or_default();
    content.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn scan_folder(folder_path: &Path) -> Vec<(String, String)> {
    let mut results = Vec::new();
    if folder_path.exists() {
        if let Ok(entries) = fs::read_dir(folder_path) {
            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().is_file() && e.path().extension().map(|ext| ext == "rs").unwrap_or(false)
                })
                .collect();
            files.sort_by_key(|e| e.path());
            for entry in files {
                let path = entry.path();
                println!("cargo:rerun-if-changed={}", path.display());
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Some(struct_name) = extract_struct_name(&content) {
                        results.push((stem, struct_name));
                    }
                }
            }
        }
    }
    results
}

fn parse_selected_experimental_algos(manifest_dir: &Path) -> (bool, Vec<String>) {
    if env::var("CARGO_FEATURE_EXPERIMENTAL").is_ok() {
        return (true, Vec::new());
    }

    let mut tokens = Vec::new();

    let env_vars = [
        "EXPERIMENTAL_ALGOS",
        "EXPERIMENTAL_ALGORITHMS",
        "INCLUDE_EXPERIMENTAL",
        "EXTRA_ALGOS",
        "EXTRA_ALGORITHMS",
    ];

    for var in &env_vars {
        if let Ok(val) = env::var(var) {
            for part in val.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    tokens.push(trimmed.to_string());
                }
            }
        }
    }

    for rustflags_var in &["CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS"] {
        if let Ok(flags) = env::var(rustflags_var) {
            let parts: Vec<&str> = if flags.contains('\x1f') {
                flags.split('\x1f').collect()
            } else {
                flags.split_whitespace().collect()
            };

            for part in parts {
                let trimmed = part.trim();
                if let Some(rest) = trimmed.strip_prefix("--cfg=")
                    .or_else(|| trimmed.strip_prefix("--cfg")) {
                    let cfg_expr = rest.trim().trim_matches('"').trim_matches('\'');
                    for prefix in &["exp=", "experimental=", "algo="] {
                        if let Some(val) = cfg_expr.strip_prefix(prefix) {
                            let val = val.trim_matches('"').trim_matches('\'');
                            for item in val.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
                                let item_trimmed = item.trim();
                                if !item_trimmed.is_empty() {
                                    tokens.push(item_trimmed.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let candidate_files = [
        manifest_dir.join("experimental_algos.txt"),
        manifest_dir.join(".experimental_algos"),
    ];

    for file in &candidate_files {
        if file.exists() {
            if let Ok(content) = fs::read_to_string(file) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.starts_with('#') || line.starts_with("//") {
                        continue;
                    }
                    for part in line.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
                        let trimmed = part.trim();
                        if !trimmed.is_empty() {
                            tokens.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }

    if tokens.iter().any(|t| t == "*" || t.eq_ignore_ascii_case("all")) {
        return (true, Vec::new());
    }

    (false, tokens)
}

fn matches_algo(
    category: &str,
    stem: &str,
    struct_name: &str,
    tokens: &[String],
) -> bool {
    let full_name = if category.is_empty() {
        stem.to_string()
    } else {
        format!("{}/{}", category, stem)
    };

    let path_stem = if category.is_empty() {
        format!("experimental/{}", stem)
    } else {
        format!("{}_experimental/{}", category, stem)
    };

    let underscore_name = if category.is_empty() {
        stem.to_string()
    } else {
        format!("{}_{}", category, stem)
    };

    for token in tokens {
        let t = token.trim();
        if t == "*" || t.eq_ignore_ascii_case("all") {
            return true;
        }
        if !category.is_empty() && (t.eq_ignore_ascii_case(&format!("{}/*", category)) || t.eq_ignore_ascii_case(category)) {
            return true;
        }
        if t.eq_ignore_ascii_case(stem)
            || t.eq_ignore_ascii_case(&format!("{}.rs", stem))
            || t.eq_ignore_ascii_case(&full_name)
            || t.eq_ignore_ascii_case(&format!("{}.rs", full_name))
            || t.eq_ignore_ascii_case(&path_stem)
            || t.eq_ignore_ascii_case(&format!("{}.rs", path_stem))
            || t.eq_ignore_ascii_case(&underscore_name)
            || t.eq_ignore_ascii_case(struct_name)
        {
            return true;
        }
    }
    false
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    let algo_dir = Path::new(&manifest_dir).join("src").join("algorithms");
    let fidoo_dir = algo_dir.join("fidoo");
    let nocrex_dir = algo_dir.join("nocrex");
    let exp_dir = algo_dir.join("experimental");
    let fidoo_exp_dir = algo_dir.join("fidoo_experimental");

    println!("cargo:rerun-if-changed={}", algo_dir.display());
    println!("cargo:rerun-if-changed={}", fidoo_dir.display());
    println!("cargo:rerun-if-changed={}", nocrex_dir.display());
    println!("cargo:rerun-if-changed={}", exp_dir.display());
    println!("cargo:rerun-if-changed={}", fidoo_exp_dir.display());

    println!("cargo:rerun-if-env-changed=EXPERIMENTAL_ALGOS");
    println!("cargo:rerun-if-env-changed=EXPERIMENTAL_ALGORITHMS");
    println!("cargo:rerun-if-env-changed=INCLUDE_EXPERIMENTAL");
    println!("cargo:rerun-if-env-changed=EXTRA_ALGOS");
    println!("cargo:rerun-if-env-changed=EXTRA_ALGORITHMS");
    let exp_txt = Path::new(&manifest_dir).join("experimental_algos.txt");
    if exp_txt.exists() {
        println!("cargo:rerun-if-changed={}", exp_txt.display());
    }
    let dot_exp_txt = Path::new(&manifest_dir).join(".experimental_algos");
    if dot_exp_txt.exists() {
        println!("cargo:rerun-if-changed={}", dot_exp_txt.display());
    }

    let root_algos = scan_folder(&algo_dir);
    let fidoo_algos = scan_folder(&fidoo_dir);
    let nocrex_algos = scan_folder(&nocrex_dir);
    let exp_algos = scan_folder(&exp_dir);
    let fidoo_exp_algos = scan_folder(&fidoo_exp_dir);

    let (include_all_exp, exp_tokens) = parse_selected_experimental_algos(Path::new(&manifest_dir));

    let (selected_exp_algos, selected_fidoo_exp_algos) = if include_all_exp {
        (exp_algos.clone(), fidoo_exp_algos.clone())
    } else if exp_tokens.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        for token in &exp_tokens {
            let mut matched = false;
            for (stem, struct_name) in &exp_algos {
                if matches_algo("", stem, struct_name, &[token.clone()]) {
                    matched = true;
                    break;
                }
            }
            if !matched {
                for (stem, struct_name) in &fidoo_exp_algos {
                    if matches_algo("fidoo", stem, struct_name, &[token.clone()]) {
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                println!(
                    "cargo:warning=Requested experimental algorithm '{}' did not match any known experimental algorithm",
                    token
                );
            }
        }

        let sel_exp: Vec<_> = exp_algos
            .into_iter()
            .filter(|(stem, struct_name)| {
                let matches = matches_algo("", stem, struct_name, &exp_tokens);
                if matches {
                    println!("cargo:warning=Including selective experimental algorithm: {}", stem);
                }
                matches
            })
            .collect();

        let sel_fidoo_exp: Vec<_> = fidoo_exp_algos
            .into_iter()
            .filter(|(stem, struct_name)| {
                let matches = matches_algo("fidoo", stem, struct_name, &exp_tokens);
                if matches {
                    println!("cargo:warning=Including selective experimental algorithm: fidoo/{}", stem);
                }
                matches
            })
            .collect();

        (sel_exp, sel_fidoo_exp)
    };

    // 1. Generate stable_mods.rs
    let mut stable_mods = String::new();
    for (module, _) in &root_algos {
        let abs_path = algo_dir
            .join(format!("{}.rs", module))
            .to_string_lossy()
            .replace('\\', "/");
        stable_mods.push_str(&format!(
            "#[path = \"{}\"]\npub mod {};\n",
            abs_path, module
        ));
    }

    stable_mods.push_str("pub mod fidoo {\n");
    for (module, _) in &fidoo_algos {
        let abs_path = fidoo_dir
            .join(format!("{}.rs", module))
            .to_string_lossy()
            .replace('\\', "/");
        stable_mods.push_str(&format!(
            "    #[path = \"{}\"]\n    pub mod {};\n",
            abs_path, module
        ));
    }
    stable_mods.push_str("}\n");

    stable_mods.push_str("pub mod nocrex {\n");
    for (module, _) in &nocrex_algos {
        let abs_path = nocrex_dir
            .join(format!("{}.rs", module))
            .to_string_lossy()
            .replace('\\', "/");
        stable_mods.push_str(&format!(
            "    #[path = \"{}\"]\n    pub mod {};\n",
            abs_path, module
        ));
    }
    stable_mods.push_str("}\n");

    write_if_changed(&Path::new(&out_dir).join("stable_mods.rs"), &stable_mods);

    // 2. Generate stable_list.rs
    let mut stable_list = String::from("vec![\n");
    for (module, struct_name) in &root_algos {
        stable_list.push_str(&format!(
            "    Box::new(crate::algorithms::{}::{}::new()),\n",
            module, struct_name
        ));
    }
    for (module, struct_name) in &fidoo_algos {
        stable_list.push_str(&format!(
            "    Box::new(crate::algorithms::fidoo::{}::{}::new()),\n",
            module, struct_name
        ));
    }
    for (module, struct_name) in &nocrex_algos {
        stable_list.push_str(&format!(
            "    Box::new(crate::algorithms::nocrex::{}::{}::new()),\n",
            module, struct_name
        ));
    }
    stable_list.push_str("]");

    write_if_changed(&Path::new(&out_dir).join("stable_list.rs"), &stable_list);

    // 3. Generate experimental_mods.rs
    let mut exp_mods = String::new();
    exp_mods.push_str("#[allow(unused_imports, dead_code)]\n");
    exp_mods.push_str("pub mod experimental {\n");
    for (module, _) in &selected_exp_algos {
        let abs_path = exp_dir
            .join(format!("{}.rs", module))
            .to_string_lossy()
            .replace('\\', "/");
        exp_mods.push_str(&format!(
            "    #[path = \"{}\"]\n    pub mod {};\n",
            abs_path, module
        ));
    }
    exp_mods.push_str("}\n");

    exp_mods.push_str("#[allow(unused_imports, dead_code)]\n");
    exp_mods.push_str("pub mod fidoo_experimental {\n");
    for (module, _) in &selected_fidoo_exp_algos {
        let abs_path = fidoo_exp_dir
            .join(format!("{}.rs", module))
            .to_string_lossy()
            .replace('\\', "/");
        exp_mods.push_str(&format!(
            "    #[path = \"{}\"]\n    pub mod {};\n",
            abs_path, module
        ));
    }
    exp_mods.push_str("}\n");

    write_if_changed(&Path::new(&out_dir).join("experimental_mods.rs"), &exp_mods);

    // 4. Generate experimental_list.rs
    let mut exp_list = String::from("vec![\n");
    for (module, struct_name) in &selected_exp_algos {
        exp_list.push_str(&format!(
            "    Box::new(crate::algorithms::experimental::{}::{}::new()),\n",
            module, struct_name
        ));
    }
    for (module, struct_name) in &selected_fidoo_exp_algos {
        exp_list.push_str(&format!(
            "    Box::new(crate::algorithms::fidoo_experimental::{}::{}::new()),\n",
            module, struct_name
        ));
    }
    exp_list.push_str("]");

    write_if_changed(&Path::new(&out_dir).join("experimental_list.rs"), &exp_list);

    // 5. Generate algorithm_hashes.rs (same order as stable_list.rs / experimental_list.rs)
    let hash_list = |entries: &[(&Path, &Vec<(String, String)>)]| -> String {
        let mut list = String::from("&[
");
        for (folder, algos) in entries {
            for (module, _) in algos.iter() {
                list.push_str(&format!("    {}u64,
", source_hash(folder, module)));
            }
        }
        list.push(']');
        list
    };
    let algorithm_hashes = format!(
        "pub const STABLE_SOURCE_HASHES: &[u64] = {};
pub const EXPERIMENTAL_SOURCE_HASHES: &[u64] = {};
",
        hash_list(&[(&algo_dir, &root_algos), (&fidoo_dir, &fidoo_algos), (&nocrex_dir, &nocrex_algos)]),
        hash_list(&[(&exp_dir, &selected_exp_algos), (&fidoo_exp_dir, &selected_fidoo_exp_algos)]),
    );
    write_if_changed(&Path::new(&out_dir).join("algorithm_hashes.rs"), &algorithm_hashes);

    // 6. Algorithm cost profile used to balance algorithms across parallel workers.
    // Builds with experimental algorithms learn costs at runtime (algo_costs.json in the project
    // root). Stable builds embed a snapshot of that file and never read or write it at runtime.
    println!("cargo:rustc-check-cfg=cfg(algo_cost_learning)");
    let cost_learning = !selected_exp_algos.is_empty() || !selected_fidoo_exp_algos.is_empty();
    let builtin_costs = if cost_learning {
        println!("cargo:rustc-cfg=algo_cost_learning");
        String::from("{}")
    } else {
        // Watched even when missing (cargo then reruns this script on every stable build), so a
        // profile created after the last stable build is still picked up.
        let costs_file = Path::new(&manifest_dir).join("algo_costs.json");
        println!("cargo:rerun-if-changed={}", costs_file.display());
        fs::read_to_string(&costs_file).unwrap_or_else(|_| String::from("{}"))
    };
    write_if_changed(&Path::new(&out_dir).join("builtin_costs.json"), &builtin_costs);

    // 7. Check for experimental sliding window exclusions file
    println!("cargo:rustc-check-cfg=cfg(has_experimental_exclusions)");
    let exp_exclusions_file = Path::new(&manifest_dir)
        .join("src")
        .join("lib")
        .join("excluded_algorithms_experimental.rs");
    if exp_exclusions_file.exists() {
        println!("cargo:rerun-if-changed={}", exp_exclusions_file.display());
        println!("cargo:rustc-cfg=has_experimental_exclusions");
    }

    #[cfg(windows)]
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winres::WindowsResource::new();
        res.set("FileDescription", "TF2 Demo Analysis Tool");
        res.set("ProductName", "TF2 Demo Analyser");
        res.set("OriginalFilename", "analysis-template.exe");
        res.set("LegalCopyright", "Copyright (c) 2026");
        res.set("CompanyName", "TF2 Analysis Project");
        if Path::new("assets/app_icon.ico").exists() {
            res.set_icon("assets/app_icon.ico");
        }
        if let Err(e) = res.compile() {
            eprintln!("cargo:warning=Failed to compile Windows resource: {}", e);
        } else {
            let resource_o = Path::new(&out_dir).join("resource.o");
            if resource_o.exists() {
                println!("cargo:rustc-link-arg-bins={}", resource_o.display());
            }
        }
    }
}
