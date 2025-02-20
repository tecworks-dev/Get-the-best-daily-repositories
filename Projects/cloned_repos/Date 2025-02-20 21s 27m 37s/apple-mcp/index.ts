#!/usr/bin/env bun
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
} from "@modelcontextprotocol/sdk/types.js";
import contacts from "./utils/contacts";
import notes from "./utils/notes";
import message from "./utils/message";

const CONTACTS_TOOL: Tool = {
  name: "contacts",
  description: "Search and retrieve contacts from Apple Contacts app",
  inputSchema: {
    type: "object",
    properties: {
      name: {
        type: "string",
        description: "Name to search for (optional - if not provided, returns all contacts)"
      }
    }
  }
};

const NOTES_TOOL: Tool = {
  name: "notes", 
  description: "Search and retrieve notes from Apple Notes app",
  inputSchema: {
    type: "object",
    properties: {
      searchText: {
        type: "string",
        description: "Text to search for in notes (optional - if not provided, returns all notes)"
      }
    }
  }
};

const MESSAGES_TOOL: Tool = {
  name: "messages",
  description: "Send and retrieve messages from Apple Messages app. you MUST provide a phone number to use this tool. Use the contacts tool to get the phone number.",
  inputSchema: {
    type: "object",
    properties: {
      phoneNumber: {
        type: "string",
        description: "Phone number to send message to"
      },
      message: {
        type: "string",
        description: "Message to send"
      }
    }
  }
};


const server = new Server(
  {
    name: "Apple MCP tools",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

function isContactsArgs(args: unknown): args is { name?: string } {
  return (
    typeof args === "object" &&
    args !== null &&
    (!("name" in args) || typeof (args as { name: string }).name === "string")
  );
}

function isNotesArgs(args: unknown): args is { searchText?: string } {
  return (
    typeof args === "object" &&
    args !== null &&
    (!("searchText" in args) || typeof (args as { searchText: string }).searchText === "string")
  );
}

function isMessagesArgs(args: unknown): args is { phoneNumber: string, message: string } {
  return (
    typeof args === "object" &&
    args !== null &&
    typeof (args as { phoneNumber: string, message: string }).phoneNumber === "string" &&
    typeof (args as { phoneNumber: string, message: string }).message === "string"
  );
}


server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [CONTACTS_TOOL, NOTES_TOOL, MESSAGES_TOOL],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  try {
    const { name, arguments: args } = request.params;

    if (!args) {
      throw new Error("No arguments provided");
    }

    switch (name) {
      case "contacts": {
        if (!isContactsArgs(args)) {
          throw new Error("Invalid arguments for contacts tool");
        }

        if (args.name) {
          const numbers = await contacts.findNumber(args.name);
          return {
            content: [{
              type: "text",
              text: numbers.length ? 
                `${args.name}: ${numbers.join(", ")}` :
                `No contact found for ${args.name}`
            }],
            isError: false
          };
        } else {
          const allNumbers = await contacts.getAllNumbers();
          return {
            content: [{
              type: "text",
              text: Object.entries(allNumbers)
                .map(([name, phones]) => `${name}: ${phones.join(", ")}`)
                .join("\n")
            }],
            isError: false
          };
        }
      }

      case "notes": {
        if (!isNotesArgs(args)) {
          throw new Error("Invalid arguments for notes tool");
        }

        if (args.searchText) {
          const foundNotes = await notes.findNote(args.searchText);
          return {
            content: [{
              type: "text",
              text: foundNotes.length ?
                foundNotes.map(note => `${note.name}:\n${note.content}`).join("\n\n") :
                `No notes found for "${args.searchText}"`
            }],
            isError: false
          };
        } else {
          const allNotes = await notes.getAllNotes();
          return {
            content: [{
              type: "text",
              text: Object.entries(allNotes)
                .map(([name, content]) => `${name}:\n${content}`)
                .join("\n\n")
            }],
            isError: false
          };
        }
      }

      case "messages": {
        if (!isMessagesArgs(args)) {
          throw new Error("Invalid arguments for messages tool");
        }

        const result = await message.sendMessage(args.phoneNumber, args.message);
        return {
          content: [{ type: "text", text: `Message sent to ${args.phoneNumber}` }],
          isError: false
        };
      }


      default:
        return {
          content: [{ type: "text", text: `Unknown tool: ${name}` }],
          isError: true,
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Apple MCP Server running on stdio");