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

fn read_packet(stream: &mut TcpStream) -> std::io::Result<(i32, i32, String)> {
    let mut size_buf = [0u8; 4];
    stream.read_exact(&mut size_buf)?;
    let size = i32::from_le_bytes(size_buf);
    if size < 10 || size > 65536 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid RCON packet size: {size}"),
        ));
    }
    let mut payload = vec![0u8; size as usize];
    stream.read_exact(&mut payload)?;
    let id = i32::from_le_bytes(payload[0..4].try_into().unwrap());
    let packet_type = i32::from_le_bytes(payload[4..8].try_into().unwrap());
    let body_end = payload[8..]
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(payload.len() - 8);
    let body = String::from_utf8_lossy(&payload[8..8 + body_end]).to_string();
    Ok((id, packet_type, body))
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

        let stream_res = TcpStream::connect_timeout(&addr, self.config.timeout);
        let mut stream = match stream_res {
            Ok(s) => s,
            Err(e) => {
                let err = format!("Could not connect to TF2 on {addr}: {e}");
                self.state = RconConnectionState::Error(err.clone());
                return Err(err);
            }
        };

        let _ = stream.set_read_timeout(Some(self.config.timeout));
        let _ = stream.set_write_timeout(Some(self.config.timeout));

        // Send authentication packet (id = 1, type = 3)
        if let Err(e) = send_packet(&mut stream, 1, SERVERDATA_AUTH, &self.config.password) {
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
        for _ in 0..5 {
            match read_packet(&mut stream) {
                Ok((id, packet_type, _)) => {
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

        if let Err(e) = send_packet(&mut stream, 2, SERVERDATA_EXECCOMMAND, trimmed) {
            return Err(format!("Failed to send command to TF2: {e}"));
        }

        match read_packet(&mut stream) {
            Ok((_, _, body)) => Ok(body),
            Err(e) => {
                // If read timed out, command was still delivered to TF2 (e.g. playdemo or demo_gototick has no console output)
                if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock {
                    Ok(String::new())
                } else {
                    Err(format!("Connection lost: {e}"))
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
