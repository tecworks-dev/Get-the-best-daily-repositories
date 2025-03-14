//! MCP schema definitions

pub mod client;
pub mod common;
pub mod json_rpc;
pub mod server;

// Re-export all schema types
pub use client::*;
pub use common::*;
pub use json_rpc::*;
pub use server::*;
