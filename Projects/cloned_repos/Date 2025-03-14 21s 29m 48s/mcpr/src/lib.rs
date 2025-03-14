//! # Model Context Protocol (MCP) for Rust
//!
//! This crate provides a Rust implementation of Anthropic's Model Context Protocol (MCP),
//! an open standard for connecting AI assistants to data sources and tools.
//!
//! The implementation includes:
//! - Schema definitions for MCP messages
//! - Transport layer for communication
//! - High-level client and server implementations
//! - CLI tools for generating server and client stubs
//! - Generator for creating MCP server and client stubs
//!
//! ## High-Level Client
//!
//! The high-level client provides a simple interface for communicating with MCP servers:
//!
//! ```rust,no_run
//! use mcpr::{
//!     client::Client,
//!     transport::stdio::StdioTransport,
//! };
//!
//! # fn main() -> Result<(), mcpr::error::MCPError> {
//! // Create a client with stdio transport
//! let transport = StdioTransport::new();
//! let mut client = Client::new(transport);
//!
//! // Initialize the client
//! client.initialize()?;
//!
//! // Call a tool (example with serde_json::Value)
//! let request = serde_json::json!({
//!     "param1": "value1",
//!     "param2": "value2"
//! });
//! let response: serde_json::Value = client.call_tool("my_tool", &request)?;
//!
//! // Shutdown the client
//! client.shutdown()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## High-Level Server
//!
//! The high-level server makes it easy to create MCP-compatible servers:
//!
//! ```rust
//! use mcpr::{
//!     error::MCPError,
//!     server::{Server, ServerConfig},
//!     transport::stdio::StdioTransport,
//!     Tool,
//! };
//! use serde_json::Value;
//!
//! # fn main() -> Result<(), mcpr::error::MCPError> {
//! // Configure the server
//! let server_config = ServerConfig::new()
//!     .with_name("My MCP Server")
//!     .with_version("1.0.0")
//!     .with_tool(Tool {
//!         name: "my_tool".to_string(),
//!         description: Some("My awesome tool".to_string()),
//!         input_schema: mcpr::schema::common::ToolInputSchema {
//!             r#type: "object".to_string(),
//!             properties: Some([
//!                 ("param1".to_string(), serde_json::json!({
//!                     "type": "string",
//!                     "description": "First parameter"
//!                 })),
//!                 ("param2".to_string(), serde_json::json!({
//!                     "type": "string",
//!                     "description": "Second parameter"
//!                 }))
//!             ].into_iter().collect()),
//!             required: Some(vec!["param1".to_string(), "param2".to_string()]),
//!         },
//!     });
//!
//! // Create the server
//! let mut server: Server<StdioTransport> = Server::new(server_config);
//!
//! // Register tool handlers
//! server.register_tool_handler("my_tool", |params: Value| {
//!     // Parse parameters and handle the tool call
//!     let param1 = params.get("param1")
//!         .and_then(|v| v.as_str())
//!         .ok_or_else(|| MCPError::Protocol("Missing param1".to_string()))?;
//!
//!     let param2 = params.get("param2")
//!         .and_then(|v| v.as_str())
//!         .ok_or_else(|| MCPError::Protocol("Missing param2".to_string()))?;
//!
//!     // Process the parameters and generate a response
//!     let response = serde_json::json!({
//!         "result": format!("Processed {} and {}", param1, param2)
//!     });
//!
//!     Ok(response)
//! })?;
//!
//! // In a real application, you would start the server with:
//! // let transport = StdioTransport::new();
//! // server.start(transport)?;
//! # Ok(())
//! # }
//! ```

pub mod cli;
pub mod client;
pub mod generator;
pub mod schema;
pub mod server;
pub mod transport;

// Re-export commonly used types
pub use schema::common::{Cursor, LoggingLevel, ProgressToken, Tool};
pub use schema::json_rpc::{JSONRPCMessage, RequestId};

/// Protocol version constants
pub mod constants {
    /// The latest supported MCP protocol version
    pub const LATEST_PROTOCOL_VERSION: &str = "2024-11-05";
    /// The JSON-RPC version used by MCP
    pub const JSONRPC_VERSION: &str = "2.0";
}

/// Error types for the MCP implementation
pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum MCPError {
        #[error("JSON serialization error: {0}")]
        Serialization(#[from] serde_json::Error),

        #[error("Transport error: {0}")]
        Transport(String),

        #[error("Protocol error: {0}")]
        Protocol(String),

        #[error("Unsupported feature: {0}")]
        UnsupportedFeature(String),
    }
}
