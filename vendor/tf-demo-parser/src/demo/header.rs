use bitbuffer::{BitRead, BitReadStream, BitWrite, Endianness};
use serde::{Deserialize, Serialize};

/// Reads a fixed-size, nul-padded text field, replacing invalid utf8 instead of failing
/// (demos can contain truncated multi-byte sequences in e.g. the nick field).
fn read_lossy_string<E: Endianness>(
    stream: &mut BitReadStream<'_, E>,
    size: usize,
) -> bitbuffer::Result<String> {
    let bytes: Vec<u8> = stream.read_sized(size)?;
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

impl<'a, E: Endianness> BitRead<'a, E> for Header {
    fn read(stream: &mut BitReadStream<'a, E>) -> bitbuffer::Result<Self> {
        Ok(Header {
            demo_type: read_lossy_string(stream, 8)?,
            version: stream.read()?,
            protocol: stream.read()?,
            server: read_lossy_string(stream, 260)?,
            nick: read_lossy_string(stream, 260)?,
            map: read_lossy_string(stream, 260)?,
            game: read_lossy_string(stream, 260)?,
            duration: stream.read()?,
            ticks: stream.read()?,
            frames: stream.read()?,
            signon: stream.read()?,
        })
    }
}

#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(BitWrite, Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Header {
    #[size = 8]
    pub demo_type: String,
    pub version: u32,
    pub protocol: u32,
    #[size = 260]
    pub server: String,
    #[size = 260]
    pub nick: String,
    #[size = 260]
    pub map: String,
    #[size = 260]
    pub game: String,
    pub duration: f32,
    pub ticks: u32,
    pub frames: u32,
    pub signon: u32,
}
