import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { tools, toolHandlers } from './tools/index.js';

export function createServer() {
  const server = new Server(
    {
      name: "browser-mcp",
      version: "0.1.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    console.error("[Tools] List available tools");
    return {
      tools
    };
  });

  /**
   * Handle tool call requests
   * Dispatch to the appropriate tool implementation
   */
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const toolName = request.params.name;
    const handler = toolHandlers[toolName];
    
    if (!handler) {
      throw new Error(`Unknown tool: ${toolName}`);
    }
    
    return handler(request.params.arguments);
  });

  return server;
}