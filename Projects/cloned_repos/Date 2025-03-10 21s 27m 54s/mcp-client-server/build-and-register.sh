#!/bin/bash

# Exit on error
set -e

# Get absolute path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install dependencies
echo "Installing dependencies..."
npm install

# Build TypeScript project
echo "Building project..."
npm run build

# Make sure the script is executable
chmod +x dist/index.js

# Check if mcp-test is already registered
if claude mcp list | grep -q "mcp-test:"; then
  echo "Removing existing mcp-test registration..."
  claude mcp remove mcp-test
fi

# Register the MCP Test Client
echo "Registering mcp-test with Claude..."
claude mcp add mcp-test -- node "$SCRIPT_DIR/dist/index.js"

echo "MCP Test Client successfully built and registered as 'mcp-test'"
echo "You can now use tools like 'mcp__mcp-test__mcp_test_deploy_server'"