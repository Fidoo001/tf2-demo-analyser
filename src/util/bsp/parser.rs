use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use super::lumps::*;
use super::trace::BspCollisionData;

#[derive(Clone, Copy, Debug, Default)]
struct LumpHeader {
    offset: i32,
    length: i32,
    _version: i32,
    _four_cc: [u8; 4],
}

pub fn parse_bsp<P: AsRef<Path>>(path: P) -> io::Result<BspCollisionData> {
    let mut file = File::open(path)?;

    // 1. Read header (1032 bytes: 4 ident, 4 version, 64 * 16 lumps)
    let mut header_buf = [0u8; 1032];
    file.read_exact(&mut header_buf)?;

    let ident = &header_buf[0..4];
    if ident != b"VBSP" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Not a valid Valve BSP file",
        ));
    }

    let version = i32::from_le_bytes(header_buf[4..8].try_into().unwrap());
    if !(19..=21).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported BSP version: {}", version),
        ));
    }

    let mut lumps = [LumpHeader::default(); 64];
    for i in 0..64 {
        let base = 8 + i * 16;
        lumps[i] = LumpHeader {
            offset: i32::from_le_bytes(header_buf[base..base + 4].try_into().unwrap()),
            length: i32::from_le_bytes(header_buf[base + 4..base + 8].try_into().unwrap()),
            _version: i32::from_le_bytes(header_buf[base + 8..base + 12].try_into().unwrap()),
            _four_cc: header_buf[base + 12..base + 16].try_into().unwrap(),
        };
    }

    let read_lump = |file: &mut File, lump_idx: usize| -> io::Result<Vec<u8>> {
        if lump_idx >= 64 {
            return Ok(Vec::new());
        }
        let lump = &lumps[lump_idx];
        if lump.length <= 0 || lump.offset <= 0 {
            return Ok(Vec::new());
        }

        file.seek(SeekFrom::Start(lump.offset as u64))?;
        let mut raw = vec![0u8; lump.length as usize];
        file.read_exact(&mut raw)?;

        // Check if lump is LZMA compressed
        if raw.len() >= 17 && &raw[0..4] == b"LZMA" {
            let actual_size = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
            let lzma_size = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
            let props = &raw[12..17];
            let compressed_payload_len = (raw.len() - 17).min(lzma_size);
            let payload = &raw[17..17 + compressed_payload_len];

            // Reconstruct standard 13-byte LZMA header:
            // 5 bytes props + 8 bytes uncompressed size (u64 le)
            let mut lzma_stream = Vec::with_capacity(13 + payload.len());
            lzma_stream.extend_from_slice(props);
            lzma_stream.extend_from_slice(&(actual_size as u64).to_le_bytes());
            lzma_stream.extend_from_slice(payload);

            let mut decompressed = Vec::with_capacity(actual_size);
            lzma_rs::lzma_decompress(&mut &lzma_stream[..], &mut decompressed).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to decompress lump {}: {}", lump_idx, e),
                )
            })?;
            Ok(decompressed)
        } else {
            Ok(raw)
        }
    };

    // 2. Read lumps required for collision & LOS
    let planes_bytes = read_lump(&mut file, LUMP_PLANES)?;
    let nodes_bytes = read_lump(&mut file, LUMP_NODES)?;
    let leafs_bytes = read_lump(&mut file, LUMP_LEAFS)?;
    let leafbrushes_bytes = read_lump(&mut file, LUMP_LEAFBRUSHES)?;
    let brushes_bytes = read_lump(&mut file, LUMP_BRUSHES)?;
    let brushsides_bytes = read_lump(&mut file, LUMP_BRUSHSIDES)?;
    let texinfo_bytes = read_lump(&mut file, LUMP_TEXINFO)?;
    let texdata_bytes = read_lump(&mut file, LUMP_TEXDATA)?;
    let stringtable_bytes = read_lump(&mut file, LUMP_TEXDATA_STRING_TABLE)?;
    let stringdata_bytes = read_lump(&mut file, LUMP_TEXDATA_STRING_DATA)?;

    // 3. Deserialize Planes (20 bytes each)
    let planes = planes_bytes
        .chunks_exact(20)
        .map(|c| Plane {
            normal: [
                f32::from_le_bytes(c[0..4].try_into().unwrap()),
                f32::from_le_bytes(c[4..8].try_into().unwrap()),
                f32::from_le_bytes(c[8..12].try_into().unwrap()),
            ],
            dist: f32::from_le_bytes(c[12..16].try_into().unwrap()),
            plane_type: i32::from_le_bytes(c[16..20].try_into().unwrap()),
        })
        .collect();

    // 4. Deserialize Nodes (32 bytes each)
    let nodes = nodes_bytes
        .chunks_exact(32)
        .map(|c| Node {
            plane_num: i32::from_le_bytes(c[0..4].try_into().unwrap()),
            children: [
                i32::from_le_bytes(c[4..8].try_into().unwrap()),
                i32::from_le_bytes(c[8..12].try_into().unwrap()),
            ],
            mins: [
                i16::from_le_bytes(c[12..14].try_into().unwrap()),
                i16::from_le_bytes(c[14..16].try_into().unwrap()),
                i16::from_le_bytes(c[16..18].try_into().unwrap()),
            ],
            maxs: [
                i16::from_le_bytes(c[18..20].try_into().unwrap()),
                i16::from_le_bytes(c[20..22].try_into().unwrap()),
                i16::from_le_bytes(c[22..24].try_into().unwrap()),
            ],
            first_face: u16::from_le_bytes(c[24..26].try_into().unwrap()),
            num_faces: u16::from_le_bytes(c[26..28].try_into().unwrap()),
            area: i16::from_le_bytes(c[28..30].try_into().unwrap()),
            padding: i16::from_le_bytes(c[30..32].try_into().unwrap()),
        })
        .collect();

    // 5. Deserialize Leafs (32 bytes each)
    let leafs = leafs_bytes
        .chunks_exact(32)
        .map(|c| Leaf {
            contents: i32::from_le_bytes(c[0..4].try_into().unwrap()),
            cluster: i16::from_le_bytes(c[4..6].try_into().unwrap()),
            area_flags: i16::from_le_bytes(c[6..8].try_into().unwrap()),
            mins: [
                i16::from_le_bytes(c[8..10].try_into().unwrap()),
                i16::from_le_bytes(c[10..12].try_into().unwrap()),
                i16::from_le_bytes(c[12..14].try_into().unwrap()),
            ],
            maxs: [
                i16::from_le_bytes(c[14..16].try_into().unwrap()),
                i16::from_le_bytes(c[16..18].try_into().unwrap()),
                i16::from_le_bytes(c[18..20].try_into().unwrap()),
            ],
            first_leafface: u16::from_le_bytes(c[20..22].try_into().unwrap()),
            num_leaffaces: u16::from_le_bytes(c[22..24].try_into().unwrap()),
            first_leafbrush: u16::from_le_bytes(c[24..26].try_into().unwrap()),
            num_leafbrushes: u16::from_le_bytes(c[26..28].try_into().unwrap()),
            leaf_water_data_id: i16::from_le_bytes(c[28..30].try_into().unwrap()),
            padding: i16::from_le_bytes(c[30..32].try_into().unwrap()),
        })
        .collect();

    // 6. Deserialize Leafbrushes (2 bytes each)
    let leafbrushes = leafbrushes_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // 7. Deserialize Brushes (12 bytes each)
    let brushes = brushes_bytes
        .chunks_exact(12)
        .map(|c| Brush {
            first_side: i32::from_le_bytes(c[0..4].try_into().unwrap()),
            num_sides: i32::from_le_bytes(c[4..8].try_into().unwrap()),
            contents: i32::from_le_bytes(c[8..12].try_into().unwrap()),
        })
        .collect();

    // 8. Deserialize Brushsides (8 bytes each)
    let brushsides = brushsides_bytes
        .chunks_exact(8)
        .map(|c| BrushSide {
            planenum: u16::from_le_bytes(c[0..2].try_into().unwrap()),
            texinfo: i16::from_le_bytes(c[2..4].try_into().unwrap()),
            dispinfo: i16::from_le_bytes(c[4..6].try_into().unwrap()),
            bevel: i16::from_le_bytes(c[6..8].try_into().unwrap()),
        })
        .collect();

    // 9. Deserialize TexInfo (72 bytes each)
    let texinfo = texinfo_bytes
        .chunks_exact(72)
        .map(|c| TexInfo {
            flags: i32::from_le_bytes(c[64..68].try_into().unwrap()),
            texdata: i32::from_le_bytes(c[68..72].try_into().unwrap()),
        })
        .collect();

    // 10. Deserialize TexData (32 bytes each)
    let texdata = texdata_bytes
        .chunks_exact(32)
        .map(|c| TexData {
            name_string_table_id: i32::from_le_bytes(c[12..16].try_into().unwrap()),
            width: i32::from_le_bytes(c[16..20].try_into().unwrap()),
            height: i32::from_le_bytes(c[20..24].try_into().unwrap()),
        })
        .collect();

    // 11. Parse String Table and String Data
    let string_table_offsets: Vec<i32> = stringtable_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut material_names = Vec::with_capacity(string_table_offsets.len());
    for &offset in &string_table_offsets {
        if offset < 0 || (offset as usize) >= stringdata_bytes.len() {
            material_names.push(String::new());
            continue;
        }
        let start = offset as usize;
        let end = stringdata_bytes[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|pos| start + pos)
            .unwrap_or(stringdata_bytes.len());
        let s = String::from_utf8_lossy(&stringdata_bytes[start..end]).to_string();
        material_names.push(s);
    }

    Ok(BspCollisionData {
        planes,
        nodes,
        leafs,
        leafbrushes,
        brushes,
        brushsides,
        texinfo,
        texdata,
        material_names,
        sight_opaque: Default::default(),
    })
}
