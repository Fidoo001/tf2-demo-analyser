//! Remote Console (RCON) & Real-Time Demo Scanning Module
//!
//! Provides RCON client connectivity, real-time recording demo monitoring,
//! player death detection, and party chat alert notifications under the `rcon` feature flag.

pub mod client;
pub mod scanner;
pub mod vote_notifier;

pub use client::*;
pub use scanner::*;
