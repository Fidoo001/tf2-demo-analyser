//! Saves a stretch of the 3D view as a video: MP4 through ffmpeg when it's
//! installed, otherwise an animated GIF. Frames arrive from the UI thread and are
//! encoded on a background thread so the window stays responsive.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::OnceLock;
use std::thread::JoinHandle;

/// GIFs get big quickly, so they're kept at most this wide.
const GIF_MAX_WIDTH: u32 = 800;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipFormat {
    Mp4,
    Gif,
}

impl ClipFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ClipFormat::Mp4 => "mp4",
            ClipFormat::Gif => "gif",
        }
    }

    /// Frames a second: smooth for video, lighter for GIFs.
    pub fn fps(self) -> u32 {
        match self {
            ClipFormat::Mp4 => 30,
            ClipFormat::Gif => 15,
        }
    }
}

/// Whether ffmpeg can be run, checked once.
pub fn ffmpeg_available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| {
        let mut command = Command::new("ffmpeg");
        command.arg("-version").stdout(Stdio::null()).stderr(Stdio::null());
        hide_window(&mut command);
        command.status().is_ok_and(|status| status.success())
    })
}

/// Keeps ffmpeg from flashing a console window on Windows.
fn hide_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = command;
}

/// Where clips are saved: a "clips" folder next to the program.
pub fn clips_folder() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("clips")
}

/// One frame of RGBA pixels.
pub struct ClipFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Cuts a frame to an even size, which video encoders need.
pub fn even_size(width: u32, height: u32) -> (u32, u32) {
    (width & !1, height & !1)
}

/// Crops `frame` to `width` x `height` from its top left, padding with black if it's smaller.
pub fn fit(frame: &ClipFrame, width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height.min(frame.height) {
        let row = (width.min(frame.width) * 4) as usize;
        let from = (y * frame.width * 4) as usize;
        let to = (y * width * 4) as usize;
        out[to..to + row].copy_from_slice(&frame.rgba[from..from + row]);
    }
    for pixel in out.chunks_exact_mut(4) {
        pixel[3] = 255;
    }
    out
}

/// Encodes frames on a background thread.
pub struct ClipWriter {
    sender: Option<Sender<ClipFrame>>,
    handle: Option<JoinHandle<Result<PathBuf, String>>>,
}

impl ClipWriter {
    pub fn start(path: PathBuf, format: ClipFormat) -> Self {
        let (sender, receiver) = mpsc::channel::<ClipFrame>();
        let handle = std::thread::spawn(move || match format {
            ClipFormat::Mp4 => encode_mp4(&path, format.fps(), receiver),
            ClipFormat::Gif => encode_gif(&path, format.fps(), receiver),
        });
        Self {
            sender: Some(sender),
            handle: Some(handle),
        }
    }

    pub fn push(&self, frame: ClipFrame) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(frame);
        }
    }

    /// No more frames are coming. Encoding finishes in the background.
    pub fn close(&mut self) {
        self.sender = None;
    }

    /// The result once encoding is done, without waiting.
    pub fn result(&mut self) -> Option<Result<PathBuf, String>> {
        if self.handle.as_ref().is_some_and(|handle| handle.is_finished()) {
            let handle = self.handle.take()?;
            return Some(handle.join().unwrap_or_else(|_| Err("the encoder stopped unexpectedly".to_string())));
        }
        None
    }
}

fn encode_mp4(path: &Path, fps: u32, frames: Receiver<ClipFrame>) -> Result<PathBuf, String> {
    let first = frames.recv().map_err(|_| "no frames were recorded".to_string())?;
    let (width, height) = even_size(first.width, first.height);
    let mut command = Command::new("ffmpeg");
    command
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .args(["-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-"])
        .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hide_window(&mut command);
    let mut child = command.spawn().map_err(|error| format!("couldn't start ffmpeg: {error}"))?;
    let mut stdin = child.stdin.take().ok_or("ffmpeg has no input")?;
    let mut write = |frame: &ClipFrame| stdin.write_all(&fit(frame, width, height));
    write(&first).map_err(|error| format!("ffmpeg stopped: {error}"))?;
    for frame in frames {
        write(&frame).map_err(|error| format!("ffmpeg stopped: {error}"))?;
    }
    drop(stdin);
    let output = child.wait_with_output().map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(path.to_path_buf())
    } else {
        Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr).trim()))
    }
}

fn encode_gif(path: &Path, fps: u32, frames: Receiver<ClipFrame>) -> Result<PathBuf, String> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame, RgbaImage};

    let first = frames.recv().map_err(|_| "no frames were recorded".to_string())?;
    let scale = (GIF_MAX_WIDTH as f32 / first.width as f32).min(1.0);
    let (width, height) = even_size(first.width, first.height);
    let target = (((width as f32 * scale) as u32).max(2), ((height as f32 * scale) as u32).max(2));
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = GifEncoder::new_with_speed(std::io::BufWriter::new(file), 10);
    encoder.set_repeat(Repeat::Infinite).map_err(|error| error.to_string())?;
    let delay = Delay::from_numer_denom_ms(1000, fps);
    for frame in std::iter::once(first).chain(frames) {
        let image = RgbaImage::from_raw(width, height, fit(&frame, width, height)).ok_or("bad frame size")?;
        let image = image::imageops::resize(&image, target.0, target.1, image::imageops::FilterType::Triangle);
        encoder
            .encode_frame(Frame::from_parts(image, 0, 0, delay))
            .map_err(|error| error.to_string())?;
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_cut_to_an_even_size() {
        assert_eq!(even_size(1125, 757), (1124, 756));
        let frame = ClipFrame { width: 3, height: 2, rgba: vec![9; 3 * 2 * 4] };
        let fitted = fit(&frame, 2, 2);
        assert_eq!(fitted.len(), 2 * 2 * 4);
        assert!(fitted.chunks_exact(4).all(|p| p == [9, 9, 9, 255]));
        let padded = fit(&frame, 4, 2);
        assert_eq!(&padded[12..16], &[0, 0, 0, 255], "missing pixels are black");
    }

    #[test]
    fn a_gif_is_written() {
        let path = std::env::temp_dir().join("demo_player_clip_test.gif");
        let mut writer = ClipWriter::start(path.clone(), ClipFormat::Gif);
        for shade in [0u8, 128, 255] {
            writer.push(ClipFrame { width: 8, height: 6, rgba: vec![shade; 8 * 6 * 4] });
        }
        writer.close();
        let result = loop {
            if let Some(result) = writer.result() {
                break result;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert_eq!(result, Ok(path.clone()));
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"GIF89a"));
        let _ = std::fs::remove_file(path);
    }
}
