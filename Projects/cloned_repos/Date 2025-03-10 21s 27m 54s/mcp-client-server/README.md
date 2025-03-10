# MCP Test Client

An MCP middleware that acts as both a server (to Claude) and a client (to servers under test) for testing MCP servers during development.

## Architecture

The MCP Test Client has a dual role:
- It's a **server** registered with Claude that exposes tools for testing
- It's a **client** that connects to and tests other MCP servers

```
┌─────────────┐          ┌───────────────────┐          ┌────────────────┐
│             │  Tools   │                   │  Client  │                │
│   Claude    │─────────>│  MCP Test Client  │─────────>│  Server Under  │
│             │          │                   │          │     Test       │
└─────────────┘          └───────────────────┘          └────────────────┘
```

This architecture lets you test MCP servers without registering them directly with Claude.

## Features

- Deploy MCP servers to test environments
- Call individual tools with custom arguments
- Run automated test suites
- View server logs
- Test servers before formal registration with Claude

## Implementation

The MCP Test Client is implemented with:

- **Process Management**: Spawns and manages MCP server processes
- **MCP SDK Client**: Uses the official MCP SDK to communicate with servers
- **Custom Transport**: Implements a custom transport for stdio communication
- **Test Execution**: Runs tests and validates responses
- **CLI Interface**: Provides an interactive testing interface

The current implementation is Phase 1 of the design plan, with future enhancements planned for Phases 2 and 3.

## Installation

```bash
# Install dependencies
npm install

# Build the TypeScript project
npm run build
```

## Usage

### As an MCP Server

The MCP Test Client is registered with Claude via the `claude-mcp-local` script. You can use the following tools:

1. Deploy a server:
```typescript
mcp__mcp-test__mcp_test_deploy_server({
  name: "my-server",
  source_path: "/path/to/server",
  env_vars: {
    "API_KEY": "${API_KEY}"
  }
})
```

2. Call a tool on a deployed server:
```typescript
mcp__mcp-test__mcp_test_call_tool({
  server_name: "my-server",
  tool_name: "tool_name",
  arguments: {
    // Tool-specific arguments
  }
})
```

3. Run tests against a server:
```typescript
mcp__mcp-test__mcp_test_run_tests({
  server_name: "my-server"
})
```

4. View server logs:
```typescript
mcp__mcp-test__mcp_test_get_logs({
  server_name: "my-server",
  lines: 100
})
```

5. List deployed servers:
```typescript
mcp__mcp-test__mcp_test_list_servers({})
```

6. Stop a server:
```typescript
mcp__mcp-test__mcp_test_stop_server({
  server_name: "my-server"
})
```

### As a CLI Tool

Run the CLI interface for testing:

```bash
# Use npm script
npm run test

# Or run directly
node dist/test-runner.js
```

This provides an interactive menu for deploying, testing, and managing MCP servers.

## Development Workflow

The MCP Test Client supports this workflow:

1. Develop an MCP server in the playground directory
2. Deploy it to the test environment with MCP Test Client
3. Test functionality, call individual tools, and debug issues
4. Fix and iterate until the server works correctly
5. Migrate the server to mcp-servers/ when ready
6. Register with Claude through claude-mcp-local

## Future Enhancements

Planned enhancements include:

- **Phase 2**: Docker-based container management, comprehensive test suites
- **Phase 3**: Migration tools, more advanced test validation

See `notes/mcp_test_client_design.md` for the complete design document.
