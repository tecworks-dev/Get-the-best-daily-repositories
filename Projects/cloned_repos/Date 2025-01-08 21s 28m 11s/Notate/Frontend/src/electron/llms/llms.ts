import { BrowserWindow } from "electron";
import db from "../db.js";
import { AnthropicProvider } from "./providers/anthropic.js";
import { OpenAIProvider } from "./providers/openai.js";
import { GeminiProvider } from "./providers/gemini.js";
import { XAIProvider } from "./providers/xai.js";
import { generateTitle } from "./generateTitle.js";
import { vectorstoreQuery } from "../embedding/vectorstoreQuery.js";
import { LocalProvider } from "./providers/local.js";

interface ProviderResponse {
  id: bigint | number;
  messages: Message[];
  title: string;
  content: string;
  aborted: boolean;
}

let mainWindow: BrowserWindow | null = null;

export function setMainWindow(window: BrowserWindow) {
  mainWindow = window;
}

export async function chatRequest(
  messages: Message[],
  activeUser: User,
  conversationId?: bigint | number,
  title?: string,
  collectionId?: bigint | number,
  signal?: AbortSignal
): Promise<{
  messages: Message[];
  id: bigint | number;
  title: string;
  error?: string;
}> {
  try {
    let currentTitle = title;
    const userSettings = await db.getUserSettings(activeUser.id);
    if (!conversationId) {
      currentTitle = await generateTitle(
        messages[messages.length - 1].content,
        activeUser.id,
        userSettings.model
      );
    }
    let data: {
      top_k: number;
      results: {
        content: string;
        metadata: string;
      }[];
    } | null = null;
    if (collectionId) {
      const collectionName = await db.getCollectionName(Number(collectionId));
      const vectorstoreData = await vectorstoreQuery({
        query: messages[messages.length - 1].content,
        userId: activeUser.id,
        userName: activeUser.name,
        collectionId: Number(collectionId),
        collectionName: collectionName.name,
      });
      if (vectorstoreData) {
        data = {
          top_k: vectorstoreData.results.length,
          results: vectorstoreData.results,
        };
      }
    }
    if (!currentTitle) {
      currentTitle = messages[messages.length - 1].content.substring(0, 20);
    }
    if (!conversationId) {
      const addConversation = await db.addUserConversation(
        activeUser.id,
        currentTitle
      );
      conversationId = addConversation.id;
    }

    let prompt;
    const getPrompt = await db.getUserPrompt(
      activeUser.id,
      Number(userSettings.prompt)
    );
    if (getPrompt) {
      prompt = getPrompt.prompt;
    } else {
      prompt = "You are a helpful assistant.";
    }
    let provider;

    switch (userSettings.provider) {
      case "openai":
        provider = OpenAIProvider;
        break;
      case "anthropic":
        provider = AnthropicProvider;
        break;
      case "gemini":
        provider = GeminiProvider;
        break;
      case "xai":
        provider = XAIProvider;
        break;
      case "local":
        provider = LocalProvider;
        break;
      default:
        throw new Error(
          "No AI provider selected. Please open Settings (top right) make sure you add an API key and select a provider under the 'AI Provider' tab."
        );
    }
    if (!currentTitle) {
      currentTitle = messages[messages.length - 1].content.substring(0, 20);
    }
    const result = await provider(
      messages,
      activeUser,
      userSettings,
      prompt,
      conversationId,
      mainWindow,
      currentTitle,
      Number(collectionId),
      data ? data : undefined,
      signal
    ) as ProviderResponse;

    try {
      // Add the user's message first
      db.addUserMessage(
        activeUser.id,
        Number(conversationId),
        "user",
        messages[messages.length - 1].content
      );

      // Add the assistant's message
      const assistantMessageId = db.addUserMessage(
        activeUser.id,
        Number(conversationId),
        "assistant",
        result.content,
        collectionId ? Number(collectionId) : undefined
      ).lastInsertRowid;

      // If we have data from retrieval, add it
      if (data !== null) {
        db.addRetrievedData(Number(assistantMessageId), JSON.stringify(data));
      }
    } catch (error) {
      // If we get a foreign key constraint error, it likely means the message was already added
      // We can safely ignore this and continue
      if (!(error instanceof Error && 'code' in error && error.code === 'SQLITE_CONSTRAINT_FOREIGNKEY')) {
        throw error;
      }
    }

    return {
      ...result,
      title:
        currentTitle || messages[messages.length - 1].content.substring(0, 20),
    };
  } catch (error) {
    console.error("Error in chat request:", error);

    const newMessage = {
      role: "assistant",
      content: "Please add an API key and select an AI Model in Settings.",
      timestamp: new Date(),
      data_content: undefined,
    } as Message;
    return {
      id: -1,
      messages: [...messages, newMessage],
      title: "Need API Key",
    };
  }
}

export function sendMessageChunk(
  content: string,
  mainWindow: BrowserWindow | null
) {
  if (mainWindow) {
    mainWindow.webContents.send("messageChunk", content);
  } else {
    console.log("This no work cause Chunk not chunky");
  }
}
