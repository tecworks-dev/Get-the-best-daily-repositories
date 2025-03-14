//! Templates for generating MCP server and client stubs

/// Template for server main.rs
pub const SERVER_MAIN_TEMPLATE: &str = r#"//! MCP Server: {{name}}

use clap::Parser;
use mcpr::schema::{
    CallToolParams, CallToolResult, Implementation, InitializeResult, JSONRPCError, JSONRPCMessage,
    JSONRPCResponse, ServerCapabilities, TextContent, Tool, ToolInputSchema, ToolResultContent,
    ToolsCapability,
};
use mcpr::transport::stdio::StdioTransport;
use serde_json::{json, Value};
use std::error::Error;
use std::collections::HashMap;
use log::{info, error, debug, warn};

/// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable debug output
    #[arg(short, long)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    if args.debug {
        println!("Debug mode enabled");
    }
    
    println!("Starting MCP server: {{name}}");
    
    // Create a transport for communication
    let mut transport = StdioTransport::new();
    
    // Wait for initialize request
    let message: JSONRPCMessage = transport.receive()?;
    
    // Server implementation here...
    
    Ok(())
}
"#;

/// Template for server Cargo.toml
pub const SERVER_CARGO_TEMPLATE: &str = r#"[package]
name = "{{name}}"
version = "0.1.0"
edition = "2021"
description = "MCP server generated from mcpr template"

[dependencies]
mcpr = "0.1.0"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"#;

/// Template for server README.md
pub const SERVER_README_TEMPLATE: &str = r#"# {{name}}

An MCP server generated using the mcpr CLI.

## Running the Server

```bash
cargo run
```

## Connecting to the Server

You can connect to this server using any MCP client. For example:

```bash
mcpr connect --uri stdio://./target/debug/{{name}}
```

## Available Tools

This server provides the following tools:

- `example`: A simple example tool that processes a query string
"#;

/// Template for client main.rs
pub const CLIENT_MAIN_TEMPLATE: &str = r#"//! MCP Client: {{name}}

use clap::Parser;
use mcpr::schema::{
    CallToolParams, CallToolResult, ClientCapabilities, Implementation, InitializeParams,
    JSONRPCMessage, JSONRPCRequest, RequestId, TextContent, ToolResultContent,
};
use mcpr::transport::stdio::StdioTransport;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::io::{self, Write};
use log::{info, error, debug};

/// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// URI of the server to connect to
    #[arg(short, long)]
    uri: String,

    /// Enable debug output
    #[arg(short, long)]
    debug: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    if args.debug {
        println!("Debug mode enabled");
    }
    
    println!("Starting MCP client: {{name}}");
    println!("Connecting to server: {}", args.uri);
    
    // Client implementation here...
    
    Ok(())
}
"#;

/// Template for client Cargo.toml
pub const CLIENT_CARGO_TEMPLATE: &str = r#"[package]
name = "{{name}}"
version = "0.1.0"
edition = "2021"
description = "MCP client generated from mcpr template"

[dependencies]
mcpr = "0.1.0"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
"#;

/// Template for client README.md
pub const CLIENT_README_TEMPLATE: &str = r#"# {{name}}

An MCP client generated using the mcpr CLI.

## Running the Client

```bash
cargo run -- --uri <server_uri>
```

For example, to connect to a local server:

```bash
cargo run -- --uri stdio://./path/to/server
```

## Usage

Once connected, you can enter queries that will be processed by the server's tools.
Type 'exit' to quit the client.
"#;

/// Template for project server main.rs
pub const PROJECT_SERVER_TEMPLATE: &str = r#"//! MCP Server for {{name}} project

use clap::Parser;
use mcpr::{
    error::MCPError,
    schema::common::{Tool, ToolInputSchema},
    transport::{
        {{#if transport_type == "stdio"}}
        stdio::StdioTransport,
        {{/if}}
        {{#if transport_type == "sse"}}
        sse::SSETransport,
        {{/if}}
        {{#if transport_type == "websocket"}}
        websocket::WebSocketTransport,
        {{/if}}
        Transport,
    },
};
use serde_json::Value;
use std::error::Error;
use std::collections::HashMap;
use log::{info, error, debug, warn};

/// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable debug output
    #[arg(short, long)]
    debug: bool,
    
    {{#if transport_type != "stdio"}}
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,
    {{/if}}
}

/// Server configuration
struct ServerConfig {
    /// Server name
    name: String,
    /// Server version
    version: String,
    /// Available tools
    tools: Vec<Tool>,
}

impl ServerConfig {
    /// Create a new server configuration
    fn new() -> Self {
        Self {
            name: "MCP Server".to_string(),
            version: "1.0.0".to_string(),
            tools: Vec::new(),
        }
    }

    /// Set the server name
    fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the server version
    fn with_version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    /// Add a tool to the server
    fn with_tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }
}

/// Tool handler function type
type ToolHandler = Box<dyn Fn(Value) -> Result<Value, MCPError> + Send + Sync>;

/// High-level MCP server
struct Server<T> {
    config: ServerConfig,
    tool_handlers: HashMap<String, ToolHandler>,
    transport: Option<T>,
}

impl<T> Server<T> 
where 
    T: Transport
{
    /// Create a new MCP server with the given configuration
    fn new(config: ServerConfig) -> Self {
        Self {
            config,
            tool_handlers: HashMap::new(),
            transport: None,
        }
    }

    /// Register a tool handler
    fn register_tool_handler<F>(&mut self, tool_name: &str, handler: F) -> Result<(), MCPError>
    where
        F: Fn(Value) -> Result<Value, MCPError> + Send + Sync + 'static,
    {
        // Check if the tool exists in the configuration
        if !self.config.tools.iter().any(|t| t.name == tool_name) {
            return Err(MCPError::Protocol(format!(
                "Tool '{}' not found in server configuration",
                tool_name
            )));
        }

        // Register the handler
        self.tool_handlers
            .insert(tool_name.to_string(), Box::new(handler));

        info!("Registered handler for tool '{}'", tool_name);
        Ok(())
    }

    /// Start the server with the given transport
    fn start(&mut self, mut transport: T) -> Result<(), MCPError> {
        // Start the transport
        info!("Starting transport...");
        transport.start()?;

        // Store the transport
        self.transport = Some(transport);

        // Process messages
        info!("Processing messages...");
        self.process_messages()
    }

    /// Process incoming messages
    fn process_messages(&mut self) -> Result<(), MCPError> {
        loop {
            let message = {
                let transport = self
                    .transport
                    .as_mut()
                    .ok_or_else(|| MCPError::Protocol("Transport not initialized".to_string()))?;

                // Receive a message
                match transport.receive() {
                    Ok(msg) => msg,
                    Err(e) => {
                        error!("Error receiving message: {}", e);
                        continue;
                    }
                }
            };

            // Handle the message
            match message {
                mcpr::schema::json_rpc::JSONRPCMessage::Request(request) => {
                    let id = request.id.clone();
                    let method = request.method.clone();
                    let params = request.params.clone();

                    match method.as_str() {
                        "initialize" => {
                            info!("Received initialization request");
                            self.handle_initialize(id, params)?;
                        }
                        "tool_call" => {
                            info!("Received tool call request");
                            self.handle_tool_call(id, params)?;
                        }
                        "shutdown" => {
                            info!("Received shutdown request");
                            self.handle_shutdown(id)?;
                            break;
                        }
                        _ => {
                            warn!("Unknown method: {}", method);
                            self.send_error(
                                id,
                                -32601,
                                format!("Method not found: {}", method),
                                None,
                            )?;
                        }
                    }
                }
                _ => {
                    warn!("Unexpected message type");
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Handle initialization request
    fn handle_initialize(&mut self, id: mcpr::schema::json_rpc::RequestId, _params: Option<Value>) -> Result<(), MCPError> {
        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| MCPError::Protocol("Transport not initialized".to_string()))?;

        // Create initialization response
        let response = mcpr::schema::json_rpc::JSONRPCResponse::new(
            id,
            serde_json::json!({
                "protocol_version": mcpr::constants::LATEST_PROTOCOL_VERSION,
                "server_info": {
                    "name": self.config.name,
                    "version": self.config.version
                },
                "tools": self.config.tools
            }),
        );

        // Send the response
        debug!("Sending initialization response");
        transport.send(&mcpr::schema::json_rpc::JSONRPCMessage::Response(response))?;

        Ok(())
    }

    /// Handle tool call request
    fn handle_tool_call(&mut self, id: mcpr::schema::json_rpc::RequestId, params: Option<Value>) -> Result<(), MCPError> {
        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| MCPError::Protocol("Transport not initialized".to_string()))?;

        // Extract tool name and parameters
        let params = params.ok_or_else(|| {
            MCPError::Protocol("Missing parameters in tool call request".to_string())
        })?;

        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MCPError::Protocol("Missing tool name in parameters".to_string()))?;

        let tool_params = params.get("parameters").cloned().unwrap_or(Value::Null);
        debug!("Tool call: {} with parameters: {:?}", tool_name, tool_params);

        // Find the tool handler
        let handler = self.tool_handlers.get(tool_name).ok_or_else(|| {
            MCPError::Protocol(format!("No handler registered for tool '{}'", tool_name))
        })?;

        // Call the handler
        match handler(tool_params) {
            Ok(result) => {
                // Create tool result response
                let response = mcpr::schema::json_rpc::JSONRPCResponse::new(
                    id,
                    serde_json::json!({
                        "result": result
                    }),
                );

                // Send the response
                debug!("Sending tool call response: {:?}", result);
                transport.send(&mcpr::schema::json_rpc::JSONRPCMessage::Response(response))?;
            }
            Err(e) => {
                // Send error response
                error!("Tool execution failed: {}", e);
                self.send_error(id, -32000, format!("Tool execution failed: {}", e), None)?;
            }
        }

        Ok(())
    }

    /// Handle shutdown request
    fn handle_shutdown(&mut self, id: mcpr::schema::json_rpc::RequestId) -> Result<(), MCPError> {
        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| MCPError::Protocol("Transport not initialized".to_string()))?;

        // Create shutdown response
        let response = mcpr::schema::json_rpc::JSONRPCResponse::new(id, serde_json::json!({}));

        // Send the response
        debug!("Sending shutdown response");
        transport.send(&mcpr::schema::json_rpc::JSONRPCMessage::Response(response))?;

        // Close the transport
        info!("Closing transport");
        transport.close()?;

        Ok(())
    }

    /// Send an error response
    fn send_error(
        &mut self,
        id: mcpr::schema::json_rpc::RequestId,
        code: i32,
        message: String,
        data: Option<Value>,
    ) -> Result<(), MCPError> {
        let transport = self
            .transport
            .as_mut()
            .ok_or_else(|| MCPError::Protocol("Transport not initialized".to_string()))?;

        // Create error response
        let error = mcpr::schema::json_rpc::JSONRPCMessage::Error(
            mcpr::schema::json_rpc::JSONRPCError::new(id, code, message.clone(), data),
        );

        // Send the error
        warn!("Sending error response: {}", message);
        transport.send(&error)?;

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Set log level based on debug flag
    if args.debug {
        log::set_max_level(log::LevelFilter::Debug);
        debug!("Debug logging enabled");
    }
    
    // Configure the server
    let server_config = ServerConfig::new()
        .with_name("{{name}}-server")
        .with_version("1.0.0")
        .with_tool(Tool {
            name: "hello".to_string(),
            description: Some("A simple hello world tool".to_string()),
            input_schema: ToolInputSchema {
                r#type: "object".to_string(),
                properties: Some([
                    ("name".to_string(), serde_json::json!({
                        "type": "string",
                        "description": "Name to greet"
                    }))
                ].into_iter().collect()),
                required: Some(vec!["name".to_string()]),
            },
        });
    
    // Create the server
    let mut server = Server::new(server_config);
    
    // Register tool handlers
    server.register_tool_handler("hello", |params: Value| {
        // Parse parameters
        let name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| MCPError::Protocol("Missing name parameter".to_string()))?;
        
        info!("Handling hello tool call for name: {}", name);
        
        // Generate response
        let response = serde_json::json!({
            "message": format!("Hello, {}!", name)
        });
        
        Ok(response)
    })?;
    
    // Create transport and start the server
    {{#if transport_type == "stdio"}}
    let transport = StdioTransport::new();
    {{/if}}
    {{#if transport_type == "sse"}}
    let transport = SSETransport::new(format!("http://localhost:{}", args.port));
    {{/if}}
    {{#if transport_type == "websocket"}}
    let transport = WebSocketTransport::new(format!("ws://localhost:{}", args.port));
    {{/if}}
    
    info!("Starting {{name}}-server...");
    server.start(transport)?;
    
    Ok(())
}"#;

/// Template for project server Cargo.toml
pub const PROJECT_SERVER_CARGO_TEMPLATE: &str = r#"[package]
name = "{{name}}-server"
version = "0.1.0"
edition = "2021"
description = "MCP server for {{name}} project"

[dependencies]
# For local development, use path dependency:
mcpr = { path = "../../../" }
# For production, use version from crates.io:
# mcpr = "0.1.0"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
env_logger = "0.10"
log = "0.4"
{{transport_deps}}
"#;

/// Template for project client main.rs
pub const PROJECT_CLIENT_TEMPLATE: &str = r#"//! MCP Client for {{name}} project

use clap::Parser;
use mcpr::{
    error::MCPError,
    schema::json_rpc::{JSONRPCMessage, JSONRPCRequest, RequestId},
    transport::{
        {{#if transport_type == "stdio"}}
        stdio::StdioTransport,
        {{/if}}
        {{#if transport_type == "sse"}}
        sse::SSETransport,
        {{/if}}
        {{#if transport_type == "websocket"}}
        websocket::WebSocketTransport,
        {{/if}}
        Transport,
    },
};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::error::Error;
use std::io::{self, Write};
use log::{info, error, debug};

/// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable debug output
    #[arg(short, long)]
    debug: bool,
    
    {{#if transport_type != "stdio"}}
    /// Server URI
    #[arg(short, long, default_value = "http://localhost:8080")]
    uri: String,
    {{/if}}
    
    /// Run in interactive mode
    #[arg(short, long)]
    interactive: bool,
    
    /// Name to greet (for non-interactive mode)
    #[arg(short, long)]
    name: Option<String>,
}

/// High-level MCP client
struct Client<T: Transport> {
    transport: T,
    next_request_id: i64,
}

impl<T: Transport> Client<T> {
    /// Create a new MCP client with the given transport
    fn new(transport: T) -> Self {
        Self {
            transport,
            next_request_id: 1,
        }
    }

    /// Initialize the client
    fn initialize(&mut self) -> Result<Value, MCPError> {
        // Start the transport
        debug!("Starting transport");
        self.transport.start()?;

        // Send initialization request
        let initialize_request = JSONRPCRequest::new(
            self.next_request_id(),
            "initialize".to_string(),
            Some(serde_json::json!({
                "protocol_version": mcpr::constants::LATEST_PROTOCOL_VERSION
            })),
        );

        let message = JSONRPCMessage::Request(initialize_request);
        debug!("Sending initialize request: {:?}", message);
        self.transport.send(&message)?;

        // Wait for response
        info!("Waiting for initialization response");
        let response: JSONRPCMessage = self.transport.receive()?;
        debug!("Received response: {:?}", response);

        match response {
            JSONRPCMessage::Response(resp) => Ok(resp.result),
            JSONRPCMessage::Error(err) => {
                error!("Initialization failed: {:?}", err);
                Err(MCPError::Protocol(format!(
                    "Initialization failed: {:?}",
                    err
                )))
            },
            _ => {
                error!("Unexpected response type");
                Err(MCPError::Protocol("Unexpected response type".to_string()))
            },
        }
    }

    /// Call a tool on the server
    fn call_tool<P: Serialize + std::fmt::Debug, R: DeserializeOwned>(
        &mut self,
        tool_name: &str,
        params: &P,
    ) -> Result<R, MCPError> {
        // Create tool call request
        let tool_call_request = JSONRPCRequest::new(
            self.next_request_id(),
            "tool_call".to_string(),
            Some(serde_json::json!({
                "name": tool_name,
                "parameters": serde_json::to_value(params)?
            })),
        );

        let message = JSONRPCMessage::Request(tool_call_request);
        info!("Calling tool '{}' with parameters: {:?}", tool_name, params);
        debug!("Sending tool call request: {:?}", message);
        self.transport.send(&message)?;

        // Wait for response
        info!("Waiting for tool call response");
        let response: JSONRPCMessage = self.transport.receive()?;
        debug!("Received response: {:?}", response);

        match response {
            JSONRPCMessage::Response(resp) => {
                // Extract the tool result from the response
                let result_value = resp.result;
                let result = result_value.get("result").ok_or_else(|| {
                    error!("Missing 'result' field in response");
                    MCPError::Protocol("Missing 'result' field in response".to_string())
                })?;

                // Parse the result
                debug!("Parsing result: {:?}", result);
                serde_json::from_value(result.clone()).map_err(|e| {
                    error!("Failed to parse result: {}", e);
                    MCPError::Serialization(e)
                })
            }
            JSONRPCMessage::Error(err) => {
                error!("Tool call failed: {:?}", err);
                Err(MCPError::Protocol(format!("Tool call failed: {:?}", err)))
            }
            _ => {
                error!("Unexpected response type");
                Err(MCPError::Protocol("Unexpected response type".to_string()))
            }
        }
    }

    /// Shutdown the client
    fn shutdown(&mut self) -> Result<(), MCPError> {
        // Send shutdown request
        let shutdown_request =
            JSONRPCRequest::new(self.next_request_id(), "shutdown".to_string(), None);

        let message = JSONRPCMessage::Request(shutdown_request);
        info!("Sending shutdown request");
        debug!("Shutdown request: {:?}", message);
        self.transport.send(&message)?;

        // Wait for response
        info!("Waiting for shutdown response");
        let response: JSONRPCMessage = self.transport.receive()?;
        debug!("Received response: {:?}", response);

        match response {
            JSONRPCMessage::Response(_) => {
                // Close the transport
                info!("Closing transport");
                self.transport.close()?;
                Ok(())
            }
            JSONRPCMessage::Error(err) => {
                error!("Shutdown failed: {:?}", err);
                Err(MCPError::Protocol(format!("Shutdown failed: {:?}", err)))
            }
            _ => {
                error!("Unexpected response type");
                Err(MCPError::Protocol("Unexpected response type".to_string()))
            }
        }
    }

    /// Generate the next request ID
    fn next_request_id(&mut self) -> RequestId {
        let id = self.next_request_id;
        self.next_request_id += 1;
        RequestId::Number(id)
    }
}

fn prompt_input(prompt: &str) -> Result<String, io::Error> {
    print!("{}: ", prompt);
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    Ok(input.trim().to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Set log level based on debug flag
    if args.debug {
        log::set_max_level(log::LevelFilter::Debug);
        debug!("Debug logging enabled");
    }
    
    // Create transport and client
    {{#if transport_type == "stdio"}}
    info!("Using stdio transport");
    let transport = StdioTransport::new();
    {{/if}}
    {{#if transport_type == "sse"}}
    info!("Using SSE transport with URI: {}", args.uri);
    let transport = SSETransport::new(args.uri.clone());
    {{/if}}
    {{#if transport_type == "websocket"}}
    info!("Using WebSocket transport with URI: {}", args.uri);
    let transport = WebSocketTransport::new(args.uri.clone());
    {{/if}}
    
    let mut client = Client::new(transport);
    
    // Initialize the client
    info!("Initializing client...");
    let init_result = client.initialize()?;
    info!("Server info: {:?}", init_result);
    
    if args.interactive {
        // Interactive mode
        info!("=== {{name}}-client Interactive Mode ===");
        println!("=== {{name}}-client Interactive Mode ===");
        println!("Type 'exit' or 'quit' to exit");
        
        loop {
            let name = prompt_input("Enter your name (or 'exit' to quit)")?;
            if name.to_lowercase() == "exit" || name.to_lowercase() == "quit" {
                info!("User requested exit");
                break;
            }
            
            // Call the hello tool
            let request = serde_json::json!({
                "name": name
            });
            
            match client.call_tool::<Value, Value>("hello", &request) {
                Ok(response) => {
                    if let Some(message) = response.get("message") {
                        let msg = message.as_str().unwrap_or("");
                        info!("Received message: {}", msg);
                        println!("{}", msg);
                    } else {
                        info!("Received response without message field: {:?}", response);
                        println!("Response: {:?}", response);
                    }
                },
                Err(e) => {
                    error!("Error calling tool: {}", e);
                    eprintln!("Error: {}", e);
                }
            }
            
            println!();
        }
        
        info!("Exiting interactive mode");
        println!("Exiting interactive mode");
    } else {
        // One-shot mode
        let name = args.name.ok_or_else(|| {
            error!("Name is required in non-interactive mode");
            "Name is required in non-interactive mode"
        })?;
        
        info!("Running in one-shot mode with name: {}", name);
        
        // Call the hello tool
        let request = serde_json::json!({
            "name": name
        });
        
        let response: Value = client.call_tool("hello", &request)?;
        
        if let Some(message) = response.get("message") {
            let msg = message.as_str().unwrap_or("");
            info!("Received message: {}", msg);
            println!("{}", msg);
        } else {
            info!("Received response without message field: {:?}", response);
            println!("Response: {:?}", response);
        }
    }
    
    // Shutdown the client
    info!("Shutting down client");
    client.shutdown()?;
    info!("Client shutdown complete");
    
    Ok(())
}"#;

/// Template for project client Cargo.toml
pub const PROJECT_CLIENT_CARGO_TEMPLATE: &str = r#"[package]
name = "{{name}}-client"
version = "0.1.0"
edition = "2021"
description = "MCP client for {{name}} project"

[dependencies]
# For local development, use path dependency:
mcpr = { path = "../../../" }
# For production, use version from crates.io:
# mcpr = "0.1.0"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
env_logger = "0.10"
log = "0.4"
{{transport_deps}}
"#;

/// Template for project README.md
pub const PROJECT_README_TEMPLATE: &str = r#"# {{name}} - MCP Project

A complete MCP project with both client and server components, using {{transport_type}} transport.

## Project Structure

- `server/`: The MCP server implementation
- `client/`: The MCP client implementation
- `test.sh`: A test script to run both client and server

## Building the Project

```bash
# Build the server
cd server
cargo build

# Build the client
cd ../client
cargo build
```

## Running the Server

```bash
cd server
cargo run
```

## Running the Client

```bash
cd client
cargo run -- --interactive
```

## Running the Test Script

```bash
./test.sh
```

## Available Tools

This server provides the following tools:

- `hello`: A simple tool that greets a person by name
"#;

/// Template for project server test script
pub const PROJECT_SERVER_TEST_TEMPLATE: &str = r#"#!/bin/bash

# Test script for {{name}}-server

# Exit on error
set -e

echo "Building server..."
cd server
cargo build
cd ..

# Create a simple test that runs the server directly
echo "Creating a simple test file..."
cat > test_input.json << EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05"}}
{"jsonrpc":"2.0","id":2,"method":"tool_call","params":{"name":"hello","parameters":{"name":"MCP User"}}}
{"jsonrpc":"2.0","id":3,"method":"shutdown","params":{}}
EOF

echo "Running server with test input..."
./server/target/debug/{{name}}-server-server < test_input.json > test_output.json

echo "Checking server output..."
if grep -q "Hello, MCP User" test_output.json; then
    echo "Server test completed successfully!"
else
    echo "Server test failed. Server output does not contain expected response."
    cat test_output.json
    exit 1
fi

# Clean up
rm test_input.json test_output.json
"#;

/// Template for project client test script
pub const PROJECT_CLIENT_TEST_TEMPLATE: &str = r#"#!/bin/bash

# Test script for {{name}}-client

# Exit on error
set -e

echo "Building client..."
cd client
cargo build
cd ..

# Create a mock server response
echo "Creating mock server responses..."
cat > server_responses.json << EOF
{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"2024-11-05","server_info":{"name":"{{name}}-server","version":"1.0.0"},"tools":[{"name":"hello","description":"A simple hello world tool","input_schema":{"type":"object","properties":{"name":{"type":"string","description":"Name to greet"}},"required":["name"]}}]}}
{"jsonrpc":"2.0","id":2,"result":{"result":{"message":"Hello, MCP User!"}}}
{"jsonrpc":"2.0","id":3,"result":{}}
EOF

# Run the client and capture its output
echo "Running client with mock server responses..."
./client/target/debug/{{name}}-client-client --name "MCP User" < server_responses.json > client_output.json

echo "Checking client output..."
if grep -q "initialize" client_output.json && grep -q "tool_call" client_output.json && grep -q "shutdown" client_output.json; then
    echo "Client test completed successfully!"
else
    echo "Client test failed. Client output does not contain expected requests."
    cat client_output.json
    exit 1
fi

# Clean up
rm server_responses.json client_output.json
"#;

/// Template for project combined test script
pub const PROJECT_RUN_TESTS_TEMPLATE: &str = r#"#!/bin/bash

# Combined test script for {{name}} MCP project

# Exit on error
set -e

echo "=== Running server test ==="
./test_server.sh

echo ""
echo "=== Running client test ==="
./test_client.sh

echo ""
echo "All tests completed successfully!"
"#;

/// Template for project test script (original, kept for backward compatibility)
pub const PROJECT_TEST_SCRIPT_TEMPLATE: &str = r#"#!/bin/bash

# Test script for {{name}} MCP project

# Exit on error
set -e

echo "Building server..."
cd server
cargo build

echo "Building client..."
cd ../client
cargo build

{{#if transport_type == "stdio"}}
echo "Creating a simple test file..."
cd ..
cat > test_input.json << EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05"}}
{"jsonrpc":"2.0","id":2,"method":"tool_call","params":{"name":"hello","parameters":{"name":"MCP User"}}}
{"jsonrpc":"2.0","id":3,"method":"shutdown","params":{}}
EOF

echo "Running server with test input..."
./server/target/debug/{{name}}-server < test_input.json > test_output.json

echo "Checking server output..."
if grep -q "Hello, MCP User" test_output.json; then
    echo "Test completed successfully!"
else
    echo "Test failed. Server output does not contain expected response."
    cat test_output.json
    exit 1
fi

# Clean up
rm test_input.json test_output.json
{{/if}}

{{#if transport_type != "stdio"}}
echo "Starting server in background..."
cd ..
./server/target/debug/{{name}}-server-server --port 8081 &
SERVER_PID=$!

# Give the server time to start
sleep 2

echo "Running client..."
./client/target/debug/{{name}}-client-client --uri "http://localhost:8081" --name "MCP User"

echo "Shutting down server..."
kill $SERVER_PID
{{/if}}

echo "Test completed successfully!"
"#;
