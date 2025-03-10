#!/bin/bash

# Script to build and register the MCP Test Client with Claude

# Exit on error
set -e

echo "Building MCP Test Client..."
cd "$(dirname "$0")/.."
npm run build

echo "Making index.js executable..."
chmod +x dist/index.js

echo "Registering MCP Test Client with Claude..."
cd -
cd "$(dirname "$0")/../../"
claude mcp add mcp-test -- node "$(dirname "$0")/../dist/index.js"

echo "MCP Test Client registered successfully!"
echo "You can now use the mcp_test_* tools in Claude."