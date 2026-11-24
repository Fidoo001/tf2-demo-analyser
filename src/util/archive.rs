//! On-demand archive inspection and single-demo extraction.
//!
//! Supports `.zip`, `.7z`, `.rar`, `.tar`, `.tar.gz`/`.tgz`, `.tar.bz2`/`.tbz2`,
//! `.tar.xz`/`.txz`, `.gz`, `.bz2`, and `.xz` using pure-Rust crates so no
//! external `7z` or `unrar` binary is required on Windows or Linux.

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use bzip2::read::MultiBzDecoder;
use flate2::read::MultiGzDecoder;
use lzma_rust2::XzReader;

/// File extensions recognized as archives in CLI input and GUI file dialogs.
pub const ARCHIVE_EXTENSIONS: &[&str] = &[
    "zip", "7z", "rar", "tar", "tgz", "tbz2", "txz", "gz", "bz2", "xz",
];

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Supported archive and compressed stream formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    Zip,
    SevenZ,
    Rar,
    Tar,
    TarGz,
    TarBz2,
    TarXz,
    Gzip,
    Bzip2,
    Xz,
}

/// Metadata for a single `.dem` file inside an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveDemoEntry {
    /// 0-based index among the `.dem` entries inside the archive.
    pub index: usize,
    /// Internal entry index within the underlying container (for direct extraction).
    raw_index: usize,
    /// Exact entry name stored in the archive headers.
    raw_name: String,
    /// Display path normalized with forward slashes.
    pub name: String,
    /// Sanitized basename ending in `.dem`, safe on Windows and Linux filesystems.
    pub file_name: String,
    /// Uncompressed file size in bytes, when provided by the archive metadata.
    pub uncompressed_size: Option<u64>,
}

/// Temporary file holding a single extracted `.dem` entry.
///
/// The extracted file and its unique parent directory are deleted automatically
/// when this guard is dropped.
#[derive(Debug)]
pub struct ExtractedTempDemo {
    pub path: PathBuf,
    dir: PathBuf,
    pub entry: ArchiveDemoEntry,
    pub archive_path: PathBuf,
}

impl Drop for ExtractedTempDemo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
        // Remove the shared temp root only if it is now empty.
        let _ = fs::remove_dir(archive_temp_root());
    }
}

/// Detects the archive format from a path's compound or single extension.
pub fn detect_archive_format(path: &Path) -> Option<ArchiveFormat> {
    let lower_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if lower_name.ends_with(".tar.gz") {
        return Some(ArchiveFormat::TarGz);
    }
    if lower_name.ends_with(".tar.bz2") {
        return Some(ArchiveFormat::TarBz2);
    }
    if lower_name.ends_with(".tar.xz") {
        return Some(ArchiveFormat::TarXz);
    }

    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "zip" => Some(ArchiveFormat::Zip),
        "7z" => Some(ArchiveFormat::SevenZ),
        "rar" => Some(ArchiveFormat::Rar),
        "tar" => Some(ArchiveFormat::Tar),
        "tgz" => Some(ArchiveFormat::TarGz),
        "tbz2" => Some(ArchiveFormat::TarBz2),
        "txz" => Some(ArchiveFormat::TarXz),
        "gz" => Some(ArchiveFormat::Gzip),
        "bz2" => Some(ArchiveFormat::Bzip2),
        "xz" => Some(ArchiveFormat::Xz),
        _ => None,
    }
}

/// Returns `true` if `path` has a supported archive extension.
pub fn is_archive_path(path: &Path) -> bool {
    detect_archive_format(path).is_some()
}

/// Formats a byte count into a compact human-readable string (e.g. `"12.4 MB"`).
pub fn format_byte_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn archive_temp_root() -> PathBuf {
    std::env::temp_dir().join("tf2-demo-analyser-archives")
}

fn normalize_entry_name(raw: &str) -> String {
    let replaced = raw.replace('\\', "/");
    let trimmed = replaced.trim_start_matches("./").trim_start_matches('/');
    trimmed.to_string()
}

fn is_valid_demo_entry_name(raw: &str) -> bool {
    let normalized = normalize_entry_name(raw);
    if normalized.is_empty() || normalized.ends_with('/') {
        return false;
    }

    let mut segments = normalized.split('/').peekable();
    let mut last_segment = "";
    while let Some(seg) = segments.next() {
        if seg.eq_ignore_ascii_case("__MACOSX") || seg == ".." {
            return false;
        }
        if segments.peek().is_none() {
            last_segment = seg;
        }
    }

    if last_segment.is_empty() || last_segment.starts_with("._") {
        return false;
    }

    last_segment.to_ascii_lowercase().ends_with(".dem")
}

fn sanitize_filename(raw: &str) -> String {
    let normalized = raw.replace('\\', "/");
    let base = normalized.rsplit('/').next().unwrap_or("");
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*')
            {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c: char| c == ' ' || c == '.');
    if trimmed.is_empty() || trimmed == "." || trimmed == ".." {
        return "demo.dem".to_string();
    }
    if trimmed.to_ascii_lowercase().ends_with(".dem") {
        trimmed.to_string()
    } else {
        format!("{trimmed}.dem")
    }
}

fn push_demo_entry(
    out: &mut Vec<ArchiveDemoEntry>,
    raw_index: usize,
    raw_name: String,
    uncompressed_size: Option<u64>,
) {
    if !is_valid_demo_entry_name(&raw_name) {
        return;
    }
    let name = normalize_entry_name(&raw_name);
    let file_name = sanitize_filename(&name);
    let index = out.len();
    out.push(ArchiveDemoEntry {
        index,
        raw_index,
        raw_name,
        name,
        file_name,
        uncompressed_size,
    });
}

fn single_stream_demo_name(archive_path: &Path, format: ArchiveFormat) -> Option<String> {
    let ext_suffix = match format {
        ArchiveFormat::Gzip => ".gz",
        ArchiveFormat::Bzip2 => ".bz2",
        ArchiveFormat::Xz => ".xz",
        _ => return None,
    };
    let name = archive_path.file_name()?.to_string_lossy();
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(ext_suffix) {
        return None;
    }
    let inner_len = name.len().checked_sub(ext_suffix.len())?;
    let inner = &name[..inner_len];
    if is_valid_demo_entry_name(inner) {
        Some(inner.to_string())
    } else {
        None
    }
}

fn with_tar_archive<T>(
    archive_path: &Path,
    format: ArchiveFormat,
    f: impl FnOnce(&mut tar::Archive<Box<dyn Read>>) -> Result<T>,
) -> Result<T> {
    let file = BufReader::new(
        File::open(archive_path)
            .with_context(|| format!("Failed to open archive '{}'", archive_path.display()))?,
    );
    let reader: Box<dyn Read> = match format {
        ArchiveFormat::Tar => Box::new(file),
        ArchiveFormat::TarGz => Box::new(MultiGzDecoder::new(file)),
        ArchiveFormat::TarBz2 => Box::new(MultiBzDecoder::new(file)),
        ArchiveFormat::TarXz => Box::new(XzReader::new(file, true)),
        _ => bail!("Internal error: {:?} is not a tar format", format),
    };
    let mut archive = tar::Archive::new(reader);
    f(&mut archive)
}

fn read_single_stream(archive_path: &Path, format: ArchiveFormat) -> Result<Vec<u8>> {
    let file = BufReader::new(
        File::open(archive_path)
            .with_context(|| format!("Failed to open archive '{}'", archive_path.display()))?,
    );
    let mut reader: Box<dyn Read> = match format {
        ArchiveFormat::Gzip => Box::new(MultiGzDecoder::new(file)),
        ArchiveFormat::Bzip2 => Box::new(MultiBzDecoder::new(file)),
        ArchiveFormat::Xz => Box::new(XzReader::new(file, true)),
        _ => bail!("Internal error: {:?} is not a single-stream format", format),
    };
    let mut buf = Vec::new();
    reader
        .read_to_end(&mut buf)
        .with_context(|| format!("Failed to decompress '{}'", archive_path.display()))?;
    Ok(buf)
}

/// Lists all `.dem` entries contained inside `archive_path` without extracting them.
pub fn list_demos_in_archive(archive_path: &Path) -> Result<Vec<ArchiveDemoEntry>> {
    let format = detect_archive_format(archive_path)
        .ok_or_else(|| anyhow!("Unsupported archive format: '{}'", archive_path.display()))?;
    let mut entries = Vec::new();

    match format {
        ArchiveFormat::Zip => {
            let file = BufReader::new(
                File::open(archive_path)
                    .with_context(|| format!("Failed to open zip '{}'", archive_path.display()))?,
            );
            let archive = zip::ZipArchive::new(file)
                .with_context(|| format!("Failed to read zip '{}'", archive_path.display()))?;
            let meta = archive.metadata();
            for raw_idx in 0..meta.len() {
                let entry = meta.entry(raw_idx)?;
                if !entry.is_file() {
                    continue;
                }
                let raw_name = entry.name()?.into_owned();
                push_demo_entry(&mut entries, raw_idx, raw_name, Some(entry.size()));
            }
        }
        ArchiveFormat::SevenZ => {
            let reader = sevenz_rust2::ArchiveReader::open(
                archive_path,
                sevenz_rust2::Password::empty(),
            )
            .with_context(|| format!("Failed to open 7z '{}'", archive_path.display()))?;
            for (raw_idx, entry) in reader.archive().files.iter().enumerate() {
                if entry.is_directory() || entry.is_anti_item() {
                    continue;
                }
                push_demo_entry(&mut entries, raw_idx, entry.name.clone(), Some(entry.size));
            }
        }
        ArchiveFormat::Rar => {
            let file = BufReader::new(
                File::open(archive_path)
                    .with_context(|| format!("Failed to open rar '{}'", archive_path.display()))?,
            );
            let archive = unrar_rs::RarArchive::open(file)
                .with_context(|| format!("Failed to read rar '{}'", archive_path.display()))?;
            for (raw_idx, member) in archive.entries().enumerate() {
                if member.is_directory || member.is_symlink || member.is_hardlink {
                    continue;
                }
                push_demo_entry(
                    &mut entries,
                    raw_idx,
                    member.raw_name,
                    member.unpacked_size,
                );
            }
        }
        ArchiveFormat::Tar
        | ArchiveFormat::TarGz
        | ArchiveFormat::TarBz2
        | ArchiveFormat::TarXz => {
            with_tar_archive(archive_path, format, |archive| {
                for (raw_idx, item) in archive.entries()?.enumerate() {
                    let entry = item?;
                    if !entry.header().entry_type().is_file() {
                        continue;
                    }
                    let raw_name = entry.path()?.to_string_lossy().into_owned();
                    let size = entry.header().size().ok();
                    push_demo_entry(&mut entries, raw_idx, raw_name, size);
                }
                Ok(())
            })?;
        }
        ArchiveFormat::Gzip | ArchiveFormat::Bzip2 | ArchiveFormat::Xz => {
            if let Some(raw_name) = single_stream_demo_name(archive_path, format) {
                push_demo_entry(&mut entries, 0, raw_name, None);
            }
        }
    }

    Ok(entries)
}

/// Extracts a single `.dem` entry at 0-based `demo_index` from `archive_path` into memory.
pub fn extract_demo_bytes(
    archive_path: &Path,
    demo_index: usize,
) -> Result<(ArchiveDemoEntry, Vec<u8>)> {
    let format = detect_archive_format(archive_path)
        .ok_or_else(|| anyhow!("Unsupported archive format: '{}'", archive_path.display()))?;
    let mut entries = list_demos_in_archive(archive_path)?;
    if entries.is_empty() {
        bail!("No .dem files found in archive '{}'", archive_path.display());
    }
    if demo_index >= entries.len() {
        bail!(
            "Archive index {} is out of range for '{}' (contains {} demo(s), valid indices: 0..={})",
            demo_index,
            archive_path.display(),
            entries.len(),
            entries.len() - 1
        );
    }
    let entry = entries.swap_remove(demo_index);

    let bytes = match format {
        ArchiveFormat::Zip => {
            let file = BufReader::new(
                File::open(archive_path)
                    .with_context(|| format!("Failed to open zip '{}'", archive_path.display()))?,
            );
            let mut archive = zip::ZipArchive::new(file)
                .with_context(|| format!("Failed to read zip '{}'", archive_path.display()))?;
            let mut zf = archive.by_index(entry.raw_index).with_context(|| {
                format!(
                    "Failed to access '{}' in '{}'",
                    entry.name,
                    archive_path.display()
                )
            })?;
            let cap = entry.uncompressed_size.unwrap_or(0).min(128 * 1024 * 1024) as usize;
            let mut buf = Vec::with_capacity(cap);
            zf.read_to_end(&mut buf).with_context(|| {
                format!(
                    "Failed to decompress '{}' from '{}'",
                    entry.name,
                    archive_path.display()
                )
            })?;
            buf
        }
        ArchiveFormat::SevenZ => {
            let mut reader = sevenz_rust2::ArchiveReader::open(
                archive_path,
                sevenz_rust2::Password::empty(),
            )
            .with_context(|| format!("Failed to open 7z '{}'", archive_path.display()))?;
            reader.read_file(&entry.raw_name).with_context(|| {
                format!(
                    "Failed to decompress '{}' from '{}'",
                    entry.name,
                    archive_path.display()
                )
            })?
        }
        ArchiveFormat::Rar => {
            let file = BufReader::new(
                File::open(archive_path)
                    .with_context(|| format!("Failed to open rar '{}'", archive_path.display()))?,
            );
            let mut archive = unrar_rs::RarArchive::open(file)
                .with_context(|| format!("Failed to read rar '{}'", archive_path.display()))?;
            let cap = entry.uncompressed_size.unwrap_or(0).min(128 * 1024 * 1024) as usize;
            let mut buf = Vec::with_capacity(cap);
            archive
                .by_index(entry.raw_index)
                .and_then(|member| member.copy_to(&mut buf))
                .with_context(|| {
                    format!(
                        "Failed to decompress '{}' from '{}'",
                        entry.name,
                        archive_path.display()
                    )
                })?;
            buf
        }
        ArchiveFormat::Tar
        | ArchiveFormat::TarGz
        | ArchiveFormat::TarBz2
        | ArchiveFormat::TarXz => with_tar_archive(archive_path, format, |archive| {
            for (raw_idx, item) in archive.entries()?.enumerate() {
                let mut item = item?;
                if raw_idx == entry.raw_index {
                    let cap = entry.uncompressed_size.unwrap_or(0).min(128 * 1024 * 1024) as usize;
                    let mut buf = Vec::with_capacity(cap);
                    item.read_to_end(&mut buf)?;
                    return Ok(buf);
                }
            }
            bail!(
                "Demo '{}' was not found at tar entry {} in '{}'",
                entry.name,
                entry.raw_index,
                archive_path.display()
            )
        })?,
        ArchiveFormat::Gzip | ArchiveFormat::Bzip2 | ArchiveFormat::Xz => {
            read_single_stream(archive_path, format)?
        }
    };

    Ok((entry, bytes))
}

/// Extracts a single `.dem` entry at 0-based `demo_index` into an isolated
/// temporary directory and returns an RAII guard that deletes it on drop.
pub fn extract_demo_to_temp(archive_path: &Path, demo_index: usize) -> Result<ExtractedTempDemo> {
    let (entry, bytes) = extract_demo_bytes(archive_path, demo_index)?;

    let seq = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = archive_temp_root().join(format!("{}-{seq}", std::process::id()));
    fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create temp directory '{}'", dir.display()))?;

    let path = dir.join(&entry.file_name);
    if let Err(err) = fs::write(&path, &bytes) {
        let _ = fs::remove_dir_all(&dir);
        return Err(err)
            .with_context(|| format!("Failed to write extracted demo '{}'", path.display()));
    }

    Ok(ExtractedTempDemo {
        path,
        dir,
        entry,
        archive_path: archive_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    fn unique_test_dir(label: &str) -> PathBuf {
        let seq = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "tf2-demo-archive-tests-{}-{}-{seq}",
            label,
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn format_detection_and_name_filtering() {
        assert_eq!(detect_archive_format(Path::new("a.zip")), Some(ArchiveFormat::Zip));
        assert_eq!(detect_archive_format(Path::new("a.7Z")), Some(ArchiveFormat::SevenZ));
        assert_eq!(detect_archive_format(Path::new("a.rar")), Some(ArchiveFormat::Rar));
        assert_eq!(detect_archive_format(Path::new("a.tar.gz")), Some(ArchiveFormat::TarGz));
        assert_eq!(detect_archive_format(Path::new("a.TAR.BZ2")), Some(ArchiveFormat::TarBz2));
        assert_eq!(detect_archive_format(Path::new("a.tar.xz")), Some(ArchiveFormat::TarXz));
        assert_eq!(detect_archive_format(Path::new("a.dem.gz")), Some(ArchiveFormat::Gzip));
        assert_eq!(detect_archive_format(Path::new("a.dem")), None);

        assert!(is_valid_demo_entry_name("match1.dem"));
        assert!(is_valid_demo_entry_name("folder\\sub\\match2.DEM"));
        assert!(!is_valid_demo_entry_name("__MACOSX/._match1.dem"));
        assert!(!is_valid_demo_entry_name("folder/._match1.dem"));
        assert!(!is_valid_demo_entry_name("readme.txt"));
        assert!(!is_valid_demo_entry_name("folder/"));

        assert_eq!(sanitize_filename("folder/weird:name*.dem"), "weird_name_.dem");
    }

    #[test]
    fn zip_list_extract_and_drop_cleanup() {
        let dir = unique_test_dir("zip");
        let zip_path = dir.join("demos.zip");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(file);
            let opts = SimpleFileOptions::default();

            zw.add_directory("nested/", opts).unwrap();
            zw.start_file("readme.txt", opts).unwrap();
            zw.write_all(b"not a demo").unwrap();
            zw.start_file("__MACOSX/._ghost.dem", opts).unwrap();
            zw.write_all(b"junk").unwrap();
            zw.start_file("nested/first.dem", opts).unwrap();
            zw.write_all(b"DEMO_ONE").unwrap();
            zw.start_file("second.DEM", opts).unwrap();
            zw.write_all(b"DEMO_TWO").unwrap();
            zw.finish().unwrap();
        }

        let entries = list_demos_in_archive(&zip_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index, 0);
        assert_eq!(entries[0].name, "nested/first.dem");
        assert_eq!(entries[0].file_name, "first.dem");
        assert_eq!(entries[1].index, 1);
        assert_eq!(entries[1].name, "second.DEM");

        let extracted_path;
        let extracted_dir;
        {
            let temp = extract_demo_to_temp(&zip_path, 1).unwrap();
            assert!(temp.path.is_file());
            assert_eq!(fs::read(&temp.path).unwrap(), b"DEMO_TWO");
            extracted_path = temp.path.clone();
            extracted_dir = temp.dir.clone();
        }
        assert!(!extracted_path.exists());
        assert!(!extracted_dir.exists());

        let err = extract_demo_bytes(&zip_path, 2).unwrap_err();
        assert!(err.to_string().contains("out of range"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn zip_without_demos_returns_empty_and_errors_on_extract() {
        let dir = unique_test_dir("zip_empty");
        let zip_path = dir.join("empty.zip");
        {
            let file = File::create(&zip_path).unwrap();
            let mut zw = zip::ZipWriter::new(file);
            zw.start_file("notes.txt", SimpleFileOptions::default()).unwrap();
            zw.write_all(b"hello").unwrap();
            zw.finish().unwrap();
        }

        assert!(list_demos_in_archive(&zip_path).unwrap().is_empty());
        let err = extract_demo_bytes(&zip_path, 0).unwrap_err();
        assert!(err.to_string().contains("No .dem files found"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sevenz_single_and_multi_extraction() {
        let dir = unique_test_dir("7z");
        let sz_path = dir.join("bundle.7z");
        {
            let mut writer = sevenz_rust2::ArchiveWriter::create(&sz_path).unwrap();
            writer
                .push_archive_entry(
                    sevenz_rust2::ArchiveEntry::new_file("ignore.txt"),
                    Some(&b"ignore"[..]),
                )
                .unwrap();
            writer
                .push_archive_entry(
                    sevenz_rust2::ArchiveEntry::new_file("sub/alpha.dem"),
                    Some(&b"SEVENZ_ALPHA"[..]),
                )
                .unwrap();
            writer
                .push_archive_entry(
                    sevenz_rust2::ArchiveEntry::new_file("beta.dem"),
                    Some(&b"SEVENZ_BETA"[..]),
                )
                .unwrap();
            writer.finish().unwrap();
        }

        let entries = list_demos_in_archive(&sz_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "sub/alpha.dem");
        assert_eq!(entries[1].name, "beta.dem");

        let (_, bytes0) = extract_demo_bytes(&sz_path, 0).unwrap();
        assert_eq!(bytes0, b"SEVENZ_ALPHA");
        let (_, bytes1) = extract_demo_bytes(&sz_path, 1).unwrap();
        assert_eq!(bytes1, b"SEVENZ_BETA");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn tar_and_tar_gz_and_single_gz() {
        let dir = unique_test_dir("tar");

        // .tar
        let tar_path = dir.join("pack.tar");
        {
            let file = File::create(&tar_path).unwrap();
            let mut builder = tar::Builder::new(file);
            let mut header = tar::Header::new_gnu();
            header.set_size(8);
            header.set_cksum();
            builder
                .append_data(&mut header, "dir/one.dem", &b"TAR_DEMO"[..])
                .unwrap();
            builder.finish().unwrap();
        }
        assert_eq!(list_demos_in_archive(&tar_path).unwrap().len(), 1);
        assert_eq!(extract_demo_bytes(&tar_path, 0).unwrap().1, b"TAR_DEMO");

        // .tar.gz
        let tgz_path = dir.join("pack.tar.gz");
        {
            let file = File::create(&tgz_path).unwrap();
            let gz = GzEncoder::new(file, Compression::default());
            let mut builder = tar::Builder::new(gz);
            let mut header = tar::Header::new_gnu();
            header.set_size(8);
            header.set_cksum();
            builder
                .append_data(&mut header, "two.dem", &b"TGZ_DEMO"[..])
                .unwrap();
            builder.into_inner().unwrap().finish().unwrap();
        }
        assert_eq!(list_demos_in_archive(&tgz_path).unwrap().len(), 1);
        assert_eq!(extract_demo_bytes(&tgz_path, 0).unwrap().1, b"TGZ_DEMO");

        // .dem.gz
        let gz_path = dir.join("solo.dem.gz");
        {
            let file = File::create(&gz_path).unwrap();
            let mut gz = GzEncoder::new(file, Compression::default());
            gz.write_all(b"SOLO_GZ").unwrap();
            gz.finish().unwrap();
        }
        let gz_entries = list_demos_in_archive(&gz_path).unwrap();
        assert_eq!(gz_entries.len(), 1);
        assert_eq!(gz_entries[0].file_name, "solo.dem");
        assert_eq!(extract_demo_bytes(&gz_path, 0).unwrap().1, b"SOLO_GZ");

        let _ = fs::remove_dir_all(&dir);
    }
}
