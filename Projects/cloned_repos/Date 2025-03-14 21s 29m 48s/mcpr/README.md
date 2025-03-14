# MCPR - Model Context Protocol for Rust

A Rust implementation of Anthropic's [Model Context Protocol (MCP)](https://docs.anthropic.com/claude/docs/model-context-protocol), an open standard for connecting AI assistants to data sources and tools.

## Features

- **Schema Definitions**: Complete implementation of the MCP schema
- **Transport Layer**: Multiple transport options including stdio, SSE, and WebSocket (all tested and working)
- **High-Level Client/Server**: Easy-to-use client and server implementations
- **CLI Tools**: Generate server and client stubs
- **Project Generator**: Quickly scaffold new MCP projects

## Installation

Add MCPR to your `Cargo.toml`:

```toml
[dependencies]
mcpr = "0.2.0"
```

For CLI tools, install globally:

```bash
cargo install mcpr
```

## Usage

### High-Level Client

The high-level client provides a simple interface for communicating with MCP servers:

```rust
use mcpr::{
    client::Client,
    transport::stdio::StdioTransport,
};

// Create a client with stdio transport
let transport = StdioTransport::new();
let mut client = Client::new(transport);

// Initialize the client
client.initialize()?;

// Call a tool
let request = MyToolRequest { /* ... */ };
let response: MyToolResponse = client.call_tool("my_tool", &request)?;

// Shutdown the client
client.shutdown()?;
```

### High-Level Server

The high-level server makes it easy to create MCP-compatible servers:

```rust
use mcpr::{
    server::{Server, ServerConfig},
    transport::stdio::StdioTransport,
    Tool,
};

// Configure the server
let server_config = ServerConfig::new()
    .with_name("My MCP Server")
    .with_version("1.0.0")
    .with_tool(Tool {
        name: "my_tool".to_string(),
        description: "My awesome tool".to_string(),
        parameters_schema: serde_json::json!({
            "type": "object",
            "properties": {
                // Tool parameters schema
            },
            "required": ["param1", "param2"]
        }),
    });

// Create the server
let mut server = Server::new(server_config);

// Register tool handlers
server.register_tool_handler("my_tool", |params| {
    // Parse parameters and handle the tool call
    // ...
    Ok(serde_json::to_value(response)?)
})?;

// Start the server with stdio transport
let transport = StdioTransport::new();
server.start(transport)?;
```

## Creating MCP Projects

MCPR includes a project generator to quickly scaffold new MCP projects with different transport types.

### Using the CLI

```bash
# Generate a project with stdio transport
mcpr generate --name my-stdio-project --transport stdio

# Generate a project with SSE transport
mcpr generate --name my-sse-project --transport sse

# Generate a project with WebSocket transport
mcpr generate --name my-websocket-project --transport websocket
```

### Project Structure

Each generated project includes:

```
my-project/
├── client/             # Client implementation
│   ├── src/
│   │   └── main.rs     # Client code
│   └── Cargo.toml      # Client dependencies
├── server/             # Server implementation
│   ├── src/
│   │   └── main.rs     # Server code
│   └── Cargo.toml      # Server dependencies
├── test.sh             # Combined test script
├── test_server.sh      # Server-only test script
├── test_client.sh      # Client-only test script
└── run_tests.sh        # Script to run all tests
```

### Building Projects

```bash
# Build the server
cd my-project/server
cargo build

# Build the client
cd my-project/client
cargo build
```

### Running Projects

#### Stdio Transport

For stdio transport, you typically run the server and pipe its output to the client:

```bash
# Run the server and pipe to client
./server/target/debug/my-stdio-project-server | ./client/target/debug/my-stdio-project-client
```

Or use the client to connect to the server:

```bash
# Run the server in one terminal
./server/target/debug/my-stdio-project-server

# Run the client in another terminal
./client/target/debug/my-stdio-project-client --uri "stdio://./server/target/debug/my-stdio-project-server"
```

#### SSE Transport

For SSE transport, you run the server first, then connect with the client:

```bash
# Run the server (default port is 8080)
./server/target/debug/my-sse-project-server --port 8080

# In another terminal, run the client
./client/target/debug/my-sse-project-client --uri "http://localhost:8080"
```

#### Interactive Mode

Clients support an interactive mode for manual testing:

```bash
./client/target/debug/my-project-client --interactive
```

### Running Tests

Each generated project includes test scripts:

```bash
# Run all tests
./run_tests.sh

# Run only server tests
./test_server.sh

# Run only client tests
./test_client.sh

# Run the combined test (original test script)
./test.sh
```

## Transport Options

MCPR supports multiple transport options:

### Stdio Transport

The simplest transport, using standard input/output:

```rust
use mcpr::transport::stdio::StdioTransport;

let transport = StdioTransport::new();
```

### SSE Transport

Server-Sent Events transport for web-based applications:

```rust
use mcpr::transport::sse::SSETransport;

// For server
let transport = SSETransport::new("http://localhost:8080");

// For client
let transport = SSETransport::new("http://localhost:8080");
```

### WebSocket Transport

WebSocket transport for full-duplex communication:

```rust
use mcpr::transport::websocket::WebSocketTransport;

// For server
let transport = WebSocketTransport::new("ws://localhost:8080");

// For client
let transport = WebSocketTransport::new("ws://localhost:8080");
```

## Debugging

Enable debug logging for detailed information:

```bash
# Set log level to debug
RUST_LOG=debug ./server/target/debug/my-project-server

# Capture logs to a file
RUST_LOG=debug ./server/target/debug/my-project-server > server.log 2>&1
```

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Implement additional transport options
- Add more examples
- Improve documentation
- Fix bugs and add features

## License

This project is licensed under the MIT License - see the LICENSE file for details. 