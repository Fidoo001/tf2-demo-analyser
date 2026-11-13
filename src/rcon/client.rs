//! Remote Console (RCON) Client for Team Fortress 2 / Source Engine
//!
//! Implements the standard Source Engine RCON protocol over TCP:
//! - 4-byte size (little-endian)
//! - 4-byte ID (little-endian)
//! - 4-byte Type (little-endian)
//! - Null-terminated ASCII command / password payload
//! - 1-byte null padding

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::debug_log;

pub const SERVERDATA_AUTH: i32 = 3;
pub const SERVERDATA_AUTH_RESPONSE: i32 = 2;
pub const SERVERDATA_EXECCOMMAND: i32 = 2;
pub const SERVERDATA_RESPONSE_VALUE: i32 = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RconConnectionState {
    Disconnected,
    Connected,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct RconConfig {
    pub server_address: String,
    pub password: String,
    pub timeout: Duration,
}

impl Default for RconConfig {
    fn default() -> Self {
        Self {
            server_address: "127.0.0.1:27015".to_string(),
            password: String::new(),
            timeout: Duration::from_millis(600),
        }
    }
}

pub struct RconClient {
    pub config: RconConfig,
    pub state: RconConnectionState,
    stream: Option<Arc<Mutex<TcpStream>>>,
}

fn send_packet(stream: &mut TcpStream, id: i32, packet_type: i32, body: &str) -> std::io::Result<()> {
    let body_bytes = body.as_bytes();
    let size = (4 + 4 + body_bytes.len() + 1 + 1) as i32;
    stream.write_all(&size.to_le_bytes())?;
    stream.write_all(&id.to_le_bytes())?;
    stream.write_all(&packet_type.to_le_bytes())?;
    stream.write_all(body_bytes)?;
    stream.write_all(b"\0\0")?;
    stream.flush()?;
    Ok(())
}

/// How long the rest of a packet that has started to arrive is waited for (TF2 can stall for a
/// few seconds, e.g. while loading a map) before the connection is considered broken.
const PARTIAL_PACKET_WAIT: Duration = Duration::from_secs(5);
/// Longest a command waits for its reply while other (stale) packets keep arriving.
const COMMAND_REPLY_WAIT: Duration = Duration::from_secs(3);

fn is_timeout(e: &std::io::Error) -> bool {
    matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
}

/// Fills `buf` completely. Returns `Ok(false)` if `idle_ok` and the read timed out before any byte
/// arrived. A packet that stops arriving halfway is an error: the stream would be out of sync.
fn fill(stream: &mut TcpStream, buf: &mut [u8], idle_ok: bool) -> std::io::Result<bool> {
    let mut read = 0;
    let started = std::time::Instant::now();
    while read < buf.len() {
        match stream.read(&mut buf[read..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed by TF2",
                ))
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) if is_timeout(&e) => {
                if read == 0 && idle_ok {
                    return Ok(false);
                }
                if started.elapsed() >= PARTIAL_PACKET_WAIT {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "incomplete RCON packet",
                    ));
                }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Reads one packet: `Ok(None)` if none arrived within the read timeout.
fn read_packet(stream: &mut TcpStream) -> std::io::Result<Option<(i32, i32, String)>> {
    let mut size_buf = [0u8; 4];
    if !fill(stream, &mut size_buf, true)? {
        return Ok(None);
    }
    let size = i32::from_le_bytes(size_buf);
    if size < 10 || size > 65536 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid RCON packet size: {size}"),
        ));
    }
    let mut payload = vec![0u8; size as usize];
    fill(stream, &mut payload, false)?;
    let id = i32::from_le_bytes(payload[0..4].try_into().unwrap());
    let packet_type = i32::from_le_bytes(payload[4..8].try_into().unwrap());
    let body_end = payload[8..]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(payload.len() - 8);
    let body = String::from_utf8_lossy(&payload[8..8 + body_end]).to_string();
    Ok(Some((id, packet_type, body)))
}

/// The command as written to the debug log: a password set through RCON is hidden.
fn redact_command(command: &str) -> String {
    if command.trim_start().to_ascii_lowercase().starts_with("rcon_password") {
        "rcon_password ***".to_string()
    } else {
        command.to_string()
    }
}

/// Id of the next command, so its reply can be told apart from late replies to earlier ones.
fn next_command_id() -> i32 {
    static NEXT_ID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(100);
    // Positive and never 1 (auth) or -1 (failed auth), also after wrapping around
    (NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0x3FFF_FFFF).max(100)
}

impl RconClient {
    pub fn new(config: RconConfig) -> Self {
        Self {
            config,
            state: RconConnectionState::Disconnected,
            stream: None,
        }
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, RconConnectionState::Connected) && self.stream.is_some()
    }

    /// Establishes an RCON connection to the configured TF2 server / local client.
    /// Performs the SERVERDATA_AUTH handshake.
    pub fn connect(&mut self) -> Result<(), String> {
        self.disconnect();
        self.state = RconConnectionState::Disconnected;

        let addr: SocketAddr = match self.config.server_address.parse() {
            Ok(a) => a,
            Err(_) => {
                use std::net::ToSocketAddrs;
                let addrs: Vec<SocketAddr> = self
                    .config
                    .server_address
                    .to_socket_addrs()
                    .map_err(|e| format!("Invalid server address '{}': {e}", self.config.server_address))?
                    .collect();
                addrs
                    .into_iter()
                    .find(|a| a.is_ipv4())
                    .ok_or_else(|| format!("Could not resolve IPv4 address for '{}'", self.config.server_address))?
            }
        };

        debug_log!(
            "[RCON] Connecting to {addr} (password {} characters, timeout {} ms)",
            self.config.password.chars().count(),
            self.config.timeout.as_millis()
        );
        let stream_res = TcpStream::connect_timeout(&addr, self.config.timeout);
        let mut stream = match stream_res {
            Ok(s) => s,
            Err(e) => {
                debug_log!(
                    "[RCON] TCP connect to {addr} failed: {e} (kind {:?}, os error {:?}). Refused usually means TF2 is not running or was started without -usercon / with another port.",
                    e.kind(),
                    e.raw_os_error()
                );
                let err = format!("Could not connect to TF2 on {addr}: {e}");
                self.state = RconConnectionState::Error(err.clone());
                return Err(err);
            }
        };

        let _ = stream.set_read_timeout(Some(self.config.timeout));
        let _ = stream.set_write_timeout(Some(self.config.timeout));
        debug_log!("[RCON] TCP connected (local {:?}), sending auth", stream.local_addr());

        // Send authentication packet (id = 1, type = 3)
        if let Err(e) = send_packet(&mut stream, 1, SERVERDATA_AUTH, &self.config.password) {
            debug_log!("[RCON] Sending auth failed: {e} (kind {:?}, os error {:?})", e.kind(), e.raw_os_error());
            let err = if e.kind() == std::io::ErrorKind::ConnectionReset || e.raw_os_error() == Some(10054) {
                "Connection reset by TF2 (os error 10054). If multiple failed connection attempts occurred, TF2 may have temporarily banned 127.0.0.1 (restart TF2 to clear).".to_string()
            } else {
                format!("Failed to send RCON auth packet: {e}")
            };
            self.state = RconConnectionState::Error(err.clone());
            return Err(err);
        }

        // Read responses. Server may reply with an empty SERVERDATA_RESPONSE_VALUE first,
        // then SERVERDATA_AUTH_RESPONSE.
        let mut authenticated = false;
        let auth_started = std::time::Instant::now();
        for _ in 0..5 {
            match read_packet(&mut stream) {
                // TF2 may be busy for a moment (e.g. loading a map)
                Ok(None) if auth_started.elapsed() < COMMAND_REPLY_WAIT => continue,
                Ok(None) => {
                    debug_log!("[RCON] No auth response within {} ms", auth_started.elapsed().as_millis());
                    let err = "Timed out waiting for the RCON auth response".to_string();
                    self.state = RconConnectionState::Error(err.clone());
                    return Err(err);
                }
                Ok(Some((id, packet_type, body))) => {
                    debug_log!("[RCON] Auth reply packet: id={id} type={packet_type} body={body:?}");
                    if packet_type == SERVERDATA_AUTH_RESPONSE {
                        if id == -1 {
                            let err = "RCON authentication failed: invalid password".to_string();
                            self.state = RconConnectionState::Error(err.clone());
                            return Err(err);
                        } else {
                            authenticated = true;
                            break;
                        }
                    }
                }
                Err(e) => {
                    debug_log!("[RCON] Reading auth response failed: {e} (kind {:?}, os error {:?})", e.kind(), e.raw_os_error());
                    let err = if e.kind() == std::io::ErrorKind::ConnectionReset || e.raw_os_error() == Some(10054) {
                        "Connection reset by TF2 (os error 10054). If multiple failed connection attempts occurred, TF2 may have temporarily banned 127.0.0.1 (restart TF2 to clear). Also verify 'rcon_password' is set in TF2 console.".to_string()
                    } else {
                        format!("Failed reading RCON auth response: {e}")
                    };
                    self.state = RconConnectionState::Error(err.clone());
                    return Err(err);
                }
            }
        }

        if !authenticated {
            let err = "No valid authentication response received from TF2".to_string();
            self.state = RconConnectionState::Error(err.clone());
            return Err(err);
        }

        debug_log!("[RCON] Authenticated in {} ms", auth_started.elapsed().as_millis());
        self.stream = Some(Arc::new(Mutex::new(stream)));
        self.state = RconConnectionState::Connected;
        Ok(())
    }

    /// Disconnects the active TCP stream.
    pub fn disconnect(&mut self) {
        if let Some(s) = self.stream.take() {
            if let Ok(stream) = s.lock() {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        }
        self.state = RconConnectionState::Disconnected;
    }

    /// Sends a console command to TF2 via RCON and returns the console output.
    pub fn send_command(&mut self, command: &str) -> Result<String, String> {
        let stream_arc = match &self.stream {
            Some(s) => s.clone(),
            None => {
                self.state = RconConnectionState::Disconnected;
                return Err("Not connected to TF2 RCON".to_string());
            }
        };

        match Self::send_command_shared(&stream_arc, command) {
            Ok(body) => Ok(body),
            Err(e) => {
                debug_log!("[RCON] Command {:?} failed, disconnecting: {e}", redact_command(command));
                self.disconnect();
                self.state = RconConnectionState::Error(e.clone());
                Err(e)
            }
        }
    }

    /// Sends a command across an already cloned Arc<Mutex<TcpStream>>, allowing
    /// background threads to safely execute commands without locking or mutating RconClient.
    pub fn send_command_shared(stream_arc: &Arc<Mutex<TcpStream>>, command: &str) -> Result<String, String> {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            return Ok(String::new());
        }

        let mut stream = match stream_arc.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let id = next_command_id();
        if let Err(e) = send_packet(&mut stream, id, SERVERDATA_EXECCOMMAND, trimmed) {
            return Err(format!("Failed to send command to TF2: {e}"));
        }

        // Packets with another id are replies to earlier commands that came late or in several
        // packets (long outputs such as `status`); they are skipped so they cannot pile up in the
        // socket or be taken for this command's reply.
        let started = std::time::Instant::now();
        let mut skipped = 0usize;
        loop {
            match read_packet(&mut stream) {
                Ok(Some((reply_id, _, body))) if reply_id == id => {
                    debug_log!(
                        "[RCON] > {:?}: reply after {} ms, {} bytes{}",
                        redact_command(trimmed),
                        started.elapsed().as_millis(),
                        body.len(),
                        if skipped > 0 { format!(" ({skipped} stale packets skipped)") } else { String::new() }
                    );
                    return Ok(body);
                }
                Ok(Some(_)) if started.elapsed() < COMMAND_REPLY_WAIT => skipped += 1,
                // No reply (yet): the command was still delivered to TF2 (e.g. playdemo or
                // demo_gototick have no console output)
                Ok(_) => {
                    debug_log!(
                        "[RCON] > {:?}: no reply within {} ms ({skipped} stale packets skipped)",
                        redact_command(trimmed),
                        started.elapsed().as_millis()
                    );
                    return Ok(String::new());
                }
                Err(e) => {
                    debug_log!("[RCON] > {:?}: connection lost: {e} (kind {:?}, os error {:?})", redact_command(trimmed), e.kind(), e.raw_os_error());
                    return Err(format!("Connection lost: {e}"));
                }
            }
        }
    }

    /// Clones the shared TCP stream reference for background tasks.
    pub fn clone_stream_arc(&self) -> Option<Arc<Mutex<TcpStream>>> {
        self.stream.clone()
    }

    /// Sends a message into TF2 party chat via `say_party`.
    pub fn send_party_chat(&mut self, msg: &str) -> Result<String, String> {
        let clean: String = msg
            .chars()
            .filter(|&c| c != '\r' && c != '\n' && c != '\0')
            .map(|c| if c == '"' { '\'' } else { c })
            .collect();
        let cmd = format!("say_party \"{}\"", clean);
        self.send_command(&cmd)
    }

    /// Sends a message into TF2 party chat across a shared TCP stream reference.
    pub fn send_party_chat_shared(stream_arc: &Arc<Mutex<TcpStream>>, msg: &str) -> Result<String, String> {
        let clean: String = msg
            .chars()
            .filter(|&c| c != '\r' && c != '\n' && c != '\0')
            .map(|c| if c == '"' { '\'' } else { c })
            .collect();
        let cmd = format!("say_party \"{}\"", clean);
        Self::send_command_shared(stream_arc, &cmd)
    }
}

impl Default for RconClient {
    fn default() -> Self {
        Self::new(RconConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn packet(id: i32, packet_type: i32, body: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(((4 + 4 + body.len() + 2) as i32).to_le_bytes());
        out.extend(id.to_le_bytes());
        out.extend(packet_type.to_le_bytes());
        out.extend(body.as_bytes());
        out.extend([0, 0]);
        out
    }

    /// Fake TF2: answers the auth, then runs `script` with each command's id.
    fn fake_tf2(script: impl FnOnce(&mut TcpStream, &mut dyn FnMut(&mut TcpStream) -> (i32, String)) + Send + 'static) -> RconClient {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut read_cmd = |s: &mut TcpStream| {
                let (id, _, body) = read_packet(s).unwrap().unwrap();
                (id, body)
            };
            let (auth_id, _) = read_cmd(&mut s);
            s.write_all(&packet(auth_id, SERVERDATA_RESPONSE_VALUE, "")).unwrap();
            s.write_all(&packet(auth_id, SERVERDATA_AUTH_RESPONSE, "")).unwrap();
            script(&mut s, &mut read_cmd);
            std::thread::sleep(Duration::from_secs(8));
        });
        let mut client = RconClient::new(RconConfig {
            server_address: addr.to_string(),
            password: "pw".to_string(),
            timeout: Duration::from_millis(300),
        });
        client.connect().unwrap();
        client
    }

    #[test]
    fn replies_are_matched_by_id() {
        let mut client = fake_tf2(|s, read_cmd| {
            // `status`: a reply in two packets
            let (id, body) = read_cmd(s);
            assert_eq!(body, "status");
            s.write_all(&packet(id, 0, "status part 1")).unwrap();
            s.write_all(&packet(id, 0, "status part 2")).unwrap();
            // A command without a reply, which comes late: after the next command was sent
            let (late_id, _) = read_cmd(s);
            let (id, body) = read_cmd(s);
            assert_eq!(body, "echo b");
            s.write_all(&packet(late_id, 0, "late")).unwrap();
            s.write_all(&packet(id, 0, "b")).unwrap();
        });
        assert_eq!(client.send_command("status").unwrap(), "status part 1");
        assert_eq!(client.send_command("playdemo x").unwrap(), "");
        // Skips the rest of `status` and the late reply
        assert_eq!(client.send_command("echo b").unwrap(), "b");
        assert!(client.is_connected());
    }

    #[test]
    fn half_sent_packet_disconnects() {
        let mut client = fake_tf2(|s, read_cmd| {
            let (id, _) = read_cmd(s);
            let full = packet(id, 0, "cut off");
            s.write_all(&full[..9]).unwrap();
        });
        assert!(client.send_command("echo a").is_err());
        assert!(!client.is_connected());
    }
}
