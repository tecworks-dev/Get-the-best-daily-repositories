# MCP Test Client Example Usage

This document provides examples of how to use the MCP Test Client for testing MCP servers during development.

## Deploying a Server

```typescript
// Deploy a server to the test environment
mcp__mcp-test__mcp_test_deploy_server({
  name: "claude-chat",
  source_path: "/Users/williambrown/dev/claude-code/playground/claude-chat-mcp",
  env_vars: {
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
  },
  persistent: true
})
```

## Calling a Tool

```typescript
// Call a tool on the deployed server
mcp__mcp-test__mcp_test_call_tool({
  server_name: "claude-chat",
  tool_name: "claude_send_message",
  arguments: {
    prompt: "Tell me a fun fact about AI development"
  }
})
```

## Running Tests

```typescript
// Run automated tests against the server
mcp__mcp-test__mcp_test_run_tests({
  server_name: "claude-chat",
  test_suite: "basic"
})
```

## Getting Logs

```typescript
// View server logs for debugging
mcp__mcp-test__mcp_test_get_logs({
  server_name: "claude-chat",
  lines: 100
})
```

## Listing Servers

```typescript
// List all deployed servers
mcp__mcp-test__mcp_test_list_servers({
  status: "running"
})
```

## Stopping a Server

```typescript
// Stop a server when done testing
mcp__mcp-test__mcp_test_stop_server({
  server_name: "claude-chat"
})
```

## Complete Development Workflow

Here's an example of a complete development workflow:

1. **Develop MCP Server**: Create or modify a server in the playground directory

2. **Deploy for Testing**:
   ```typescript
   mcp__mcp-test__mcp_test_deploy_server({
     name: "my-server",
     source_path: "/Users/williambrown/dev/claude-code/playground/my-server"
   })
   ```

3. **Run Tests**:
   ```typescript
   mcp__mcp-test__mcp_test_run_tests({
     server_name: "my-server"
   })
   ```

4. **Debug Issues**:
   ```typescript
   mcp__mcp-test__mcp_test_get_logs({
     server_name: "my-server"
   })
   ```

5. **Fix and Iterate**: Make changes to your code (they will be immediately available)

6. **Re-run Tests**:
   ```typescript
   mcp__mcp-test__mcp_test_run_tests({
     server_name: "my-server"
   })
   ```

7. **Clean Up**:
   ```typescript
   mcp__mcp-test__mcp_test_stop_server({
     server_name: "my-server"
   })
   ```

This workflow allows for rapid iteration without needing to restart Claude sessions or formally register the server during development.