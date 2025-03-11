#!/bin/bash

# Check if the client is built
if [ ! -f "$(dirname "$0")/../dist/index.js" ]; then
  echo "Error: MCP Test Client is not built. Run 'npm run build' first."
  exit 1
fi

# Make sure it's executable
chmod +x "$(dirname "$0")/../dist/index.js"

# Register with Claude
echo "Registering MCP Test Client with Claude..."
cd "$(dirname "$0")/../../.."
claude mcp add mcp-test -- node "$(dirname "$0")/../dist/index.js"

echo "Successfully registered MCP Test Client!"
echo "You can now use the following tools in Claude:"
echo "- mcp_test_deploy_server"
echo "- mcp_test_call_tool"
echo "- mcp_test_get_logs"
echo "- mcp_test_list_servers"
echo "- mcp_test_run_tests"
echo "- mcp_test_stop_server"