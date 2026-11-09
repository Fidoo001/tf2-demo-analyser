//! Reads TF2's own files from disk. The game itself never has to run.

use std::collections::HashMap;

use tf_asset_loader::Loader;

/// Textures larger than this are shrunk, which keeps memory use reasonable.
const MAX_TEXTURE_SIZE: u32 = 2048;
const MAX_INCLUDE_DEPTH: u32 = 4;

/// Where assets come from: the map's embedded files first, then the game's.
pub struct Assets<'a> {
    loader: &'a Loader,
    pack: Option<&'a vbsp::Bsp>,
}

impl<'a> Assets<'a> {
    pub fn new(loader: &'a Loader, pack: Option<&'a vbsp::Bsp>) -> Self {
        Self { loader, pack }
    }

    pub fn load(&self, path: &str) -> Option<Vec<u8>> {
        let path = path.replace('\\', "/").to_ascii_lowercase();
        if let Some(bsp) = self.pack {
            if let Ok(Some(data)) = bsp.pack.get(&path) {
                return Some(data);
            }
        }
        self.loader.load(&path).ok().flatten()
    }

    /// Loads a `.vmt` by material name, following `patch` materials to their base.
    pub fn material(&self, name: &str) -> Option<Vmt> {
        self.material_at(&format!("materials/{}.vmt", clean_name(name)), 0)
    }

    fn material_at(&self, path: &str, depth: u32) -> Option<Vmt> {
        let data = self.load(path)?;
        let vmt = Vmt::parse(&String::from_utf8_lossy(&data))?;
        if vmt.shader != "patch" || depth >= MAX_INCLUDE_DEPTH {
            return Some(vmt);
        }
        let mut base = self.material_at(vmt.params.get("include")?, depth + 1)?;
        base.params.extend(vmt.params);
        Some(base)
    }

    pub fn texture(&self, name: &str) -> Option<Image> {
        let data = self.load(&format!("materials/{}.vtf", clean_name(name)))?;
        Image::from_vtf(&data)
    }
}

fn clean_name(name: &str) -> String {
    name.trim_start_matches(['/', '\\'])
        .trim_end_matches(".vmt")
        .trim_end_matches(".vtf")
        .replace('\\', "/")
        .to_ascii_lowercase()
}

#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Unpacks the simple pixel formats the `vtf` crate doesn't decode itself.
fn convert_raw(format: vtf::ImageFormat, width: u32, height: u32, bytes: &[u8]) -> Option<Vec<u8>> {
    use vtf::ImageFormat as F;
    let pixels = (width * height) as usize;
    let size = match format {
        F::I8 | F::A8 | F::P8 => 1,
        F::Ia88 | F::Bgr565 | F::Rgb565 | F::Bgra4444 | F::Bgra5551 | F::Bgrx5551 | F::Uv88 => 2,
        F::Abgr8888 | F::Argb8888 | F::Bgrx8888 | F::Uvwq8888 | F::Uvlx8888 => 4,
        F::Rgba16161616f | F::Rgba16161616 => 8,
        _ => return None,
    };
    let source = bytes.get(..pixels * size)?;
    let mut rgba = Vec::with_capacity(pixels * 4);
    let expand5 = |value: u16| ((value & 0x1F) * 255 / 31) as u8;
    let expand6 = |value: u16| ((value & 0x3F) * 255 / 63) as u8;
    for pixel in source.chunks_exact(size) {
        let word = || u16::from_le_bytes([pixel[0], pixel[1]]);
        let rgba_pixel: [u8; 4] = match format {
            F::I8 | F::P8 => [pixel[0], pixel[0], pixel[0], 255],
            // Alpha-only textures are tinted when drawn, so black is the neutral choice.
            F::A8 => [0, 0, 0, pixel[0]],
            F::Ia88 => [pixel[0], pixel[0], pixel[0], pixel[1]],
            F::Uv88 => [pixel[0], pixel[1], 255, 255],
            F::Rgb565 => {
                let w = word();
                [expand5(w >> 11), expand6(w >> 5), expand5(w), 255]
            }
            F::Bgr565 => {
                let w = word();
                [expand5(w), expand6(w >> 5), expand5(w >> 11), 255]
            }
            F::Bgra4444 => {
                let w = word();
                let nibble = |shift: u16| (((w >> shift) & 0xF) * 17) as u8;
                [nibble(8), nibble(4), nibble(0), nibble(12)]
            }
            F::Bgra5551 | F::Bgrx5551 => {
                let w = word();
                let alpha = if format == F::Bgrx5551 || w & 0x8000 != 0 { 255 } else { 0 };
                [expand5(w >> 10), expand5(w >> 5), expand5(w), alpha]
            }
            F::Abgr8888 => [pixel[3], pixel[2], pixel[1], pixel[0]],
            F::Argb8888 => [pixel[1], pixel[2], pixel[3], pixel[0]],
            F::Bgrx8888 => [pixel[2], pixel[1], pixel[0], 255],
            F::Uvwq8888 | F::Uvlx8888 => [pixel[0], pixel[1], pixel[2], 255],
            F::Rgba16161616f => {
                let half = |at: usize| {
                    let value = super::skeleton::half_to_f32(u16::from_le_bytes([pixel[at], pixel[at + 1]]));
                    // Bright HDR values get squeezed into range rather than clipped.
                    ((value / (1.0 + value)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
                };
                [half(0), half(2), half(4), 255]
            }
            F::Rgba16161616 => {
                let channel = |at: usize| pixel[at + 1];
                [channel(0), channel(2), channel(4), channel(6)]
            }
            _ => return None,
        };
        rgba.extend_from_slice(&rgba_pixel);
    }
    Some(rgba)
}

impl Image {
    fn from_vtf(data: &[u8]) -> Option<Self> {
        let vtf = vtf::from_bytes(data).ok()?;
        let image = match vtf.highres_image.decode(0) {
            Ok(image) => image,
            Err(_) => {
                let source = &vtf.highres_image;
                let (width, height) = (u32::from(source.width), u32::from(source.height));
                let rgba = convert_raw(source.format, width, height, source.get_frame(0).ok()?)?;
                ::image::DynamicImage::ImageRgba8(::image::RgbaImage::from_raw(width, height, rgba)?)
            }
        };
        let (mut width, mut height) = (image.width(), image.height());
        let image = if width > MAX_TEXTURE_SIZE || height > MAX_TEXTURE_SIZE {
            let scaled = image.thumbnail(MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE);
            width = scaled.width();
            height = scaled.height();
            scaled
        } else {
            image
        };
        Some(Self {
            width,
            height,
            rgba: image.to_rgba8().into_raw(),
        })
    }
}

/// A material's settings. Valve's format is loose enough that a strict parser
/// rejects plenty of real files, so this one keeps whatever it can read.
#[derive(Clone, Debug, Default)]
pub struct Vmt {
    pub shader: String,
    pub params: HashMap<String, String>,
}

impl Vmt {
    pub fn parse(text: &str) -> Option<Self> {
        let mut tokens = Tokens { text, position: 0 };
        let shader = tokens.next()?.to_ascii_lowercase();
        if tokens.next()? != "{" {
            return None;
        }
        let mut params = HashMap::new();
        read_block(&mut tokens, &mut params);
        Some(Self { shader, params })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(String::as_str)
    }

    pub fn flag(&self, key: &str) -> bool {
        self.get(key)
            .is_some_and(|value| value.trim().parse::<f32>().is_ok_and(|number| number != 0.0))
    }

    pub fn number(&self, key: &str) -> Option<f32> {
        self.get(key)?.trim().parse().ok()
    }

    /// Reads a colour written as `[r g b]` (0 to 1) or `{r g b}` (0 to 255).
    pub fn color(&self, key: &str) -> Option<[f32; 3]> {
        let value = self.get(key)?.trim();
        let scale = if value.starts_with('{') { 1.0 / 255.0 } else { 1.0 };
        let mut numbers = value
            .trim_matches(|c| matches!(c, '[' | ']' | '{' | '}'))
            .split_whitespace()
            .filter_map(|part| part.parse::<f32>().ok());
        Some([
            numbers.next()? * scale,
            numbers.next()? * scale,
            numbers.next()? * scale,
        ])
    }
}

fn read_block(tokens: &mut Tokens, params: &mut HashMap<String, String>) {
    while let Some(key) = tokens.next() {
        if key == "}" {
            return;
        }
        let Some(value) = tokens.next() else { return };
        let key = key.to_ascii_lowercase();
        if value != "{" {
            params.entry(key).or_insert_with(|| value.to_string());
            continue;
        }
        // Proxies animate materials, and `<dx90` style blocks target older hardware.
        if key == "proxies" || key.starts_with('<') {
            skip_block(tokens);
        } else {
            read_block(tokens, params);
        }
    }
}

fn skip_block(tokens: &mut Tokens) {
    let mut depth = 1;
    while let Some(token) = tokens.next() {
        match token {
            "{" => depth += 1,
            "}" => {
                depth -= 1;
                if depth == 0 {
                    return;
                }
            }
            _ => {}
        }
    }
}

struct Tokens<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Tokens<'a> {
    fn next(&mut self) -> Option<&'a str> {
        let bytes = self.text.as_bytes();
        loop {
            while bytes.get(self.position)?.is_ascii_whitespace() {
                self.position += 1;
            }
            if self.text[self.position..].starts_with("//") {
                let rest = &self.text[self.position..];
                self.position += rest.find('\n').unwrap_or(rest.len());
                continue;
            }
            break;
        }

        let start = self.position;
        match bytes[start] {
            b'{' | b'}' => {
                self.position += 1;
                Some(&self.text[start..start + 1])
            }
            b'"' => {
                let rest = &self.text[start + 1..];
                let end = rest.find('"').unwrap_or(rest.len());
                self.position = (start + 1 + end + 1).min(self.text.len());
                Some(&rest[..end])
            }
            _ => {
                let rest = &self.text[start..];
                let end = rest
                    .find(|c: char| c.is_ascii_whitespace() || matches!(c, '{' | '}' | '"'))
                    .unwrap_or(rest.len());
                self.position = start + end;
                Some(&rest[..end])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_quoted_and_bare_values() {
        let vmt = Vmt::parse(
            r#"LightmappedGeneric
            {
                "$basetexture" "Nature/Grass01"
                $translucent 1
                // a comment
                "$color" "[0.5 1 0.25]"
            }"#,
        )
        .unwrap();
        assert_eq!(vmt.shader, "lightmappedgeneric");
        assert_eq!(vmt.get("$basetexture"), Some("Nature/Grass01"));
        assert!(vmt.flag("$translucent"));
        assert_eq!(vmt.color("$color"), Some([0.5, 1.0, 0.25]));
    }

    #[test]
    fn ignores_proxies_and_old_hardware_blocks() {
        let vmt = Vmt::parse(
            r#""VertexLitGeneric" {
                "$basetexture" "a"
                "<dx90" { "$basetexture" "wrong" }
                "Proxies" { "Sine" { "resultVar" "$alpha" } }
            }"#,
        )
        .unwrap();
        assert_eq!(vmt.get("$basetexture"), Some("a"));
        assert_eq!(vmt.get("resultvar"), None);
    }

    #[test]
    fn keeps_patch_includes_and_inserted_values() {
        let vmt = Vmt::parse(
            r#"patch { include "materials/a.vmt" insert { "$envmap" "b" } }"#,
        )
        .unwrap();
        assert_eq!(vmt.get("include"), Some("materials/a.vmt"));
        assert_eq!(vmt.get("$envmap"), Some("b"));
    }

    #[test]
    fn survives_broken_files() {
        assert!(Vmt::parse("").is_none());
        assert!(Vmt::parse("shader").is_none());
        assert!(Vmt::parse("shader {").is_some());
        assert!(Vmt::parse("shader { \"$a\" \"unterminated").is_some());
    }

    #[test]
    fn colors_in_braces_are_0_to_255() {
        let vmt = Vmt::parse(r#"x { "$fogcolor" "{255 0 51}" }"#).unwrap();
        let [r, g, b] = vmt.color("$fogcolor").unwrap();
        assert!((r - 1.0).abs() < 1e-6 && g == 0.0 && (b - 0.2).abs() < 1e-6);
    }

    #[test]
    fn packed_pixel_formats_unpack() {
        use vtf::ImageFormat as F;
        assert_eq!(convert_raw(F::I8, 1, 1, &[200]), Some(vec![200, 200, 200, 255]));
        assert_eq!(convert_raw(F::A8, 1, 1, &[90]), Some(vec![0, 0, 0, 90]));
        assert_eq!(convert_raw(F::Ia88, 1, 1, &[10, 20]), Some(vec![10, 10, 10, 20]));
        assert_eq!(convert_raw(F::Bgrx8888, 1, 1, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
        let white_565 = 0xFFFFu16.to_le_bytes();
        assert_eq!(convert_raw(F::Bgr565, 1, 1, &white_565), Some(vec![255, 255, 255, 255]));
        assert_eq!(convert_raw(F::I8, 2, 2, &[1, 2]), None, "too little data");
        assert_eq!(convert_raw(F::Dxt1, 1, 1, &[0; 8]), None, "handled by the crate itself");
    }

    #[test]
    fn material_names_are_normalized() {
        assert_eq!(clean_name("\\Nature\\Rock.vmt"), "nature/rock");
    }
}


