import OpenAI from "openai";
import db from "../../db.js";
import { BrowserWindow } from "electron";
import { sendMessageChunk } from "../llms.js";
let openai: OpenAI;

async function initializeOpenAI(apiKey: string) {
  openai = new OpenAI({ apiKey });
}

export async function OpenAIProvider(
  messages: Message[],
  activeUser: User,
  userSettings: UserSettings,
  prompt: string,
  conversationId: bigint | number,
  mainWindow: BrowserWindow | null = null,
  currentTitle: string,
  collectionId?: number,
  data?: {
    top_k: number;
    results: {
      content: string;
      metadata: string;
    }[];
  } | null,
  signal?: AbortSignal
) {
  const apiKey = db.getApiKey(activeUser.id, "openai");

  if (!apiKey) {
    throw new Error("OpenAI API key not found for the active user");
  }

  await initializeOpenAI(apiKey);

  if (!openai) {
    throw new Error("OpenAI instance not initialized");
  }

  const newMessages = messages.map((msg) => ({
    role: msg.role,
    content: msg.content,
  }));
  let dataCollectionInfo;
  if (collectionId) {
    dataCollectionInfo = db.getCollection(collectionId) as Collection;
  }
  const sysPrompt: {
    role: "system";
    content: string;
  } = {
    role: "system",
    content:
      prompt +
      (data
        ? "The following is the data that the user has provided via their custom data collection: " +
          `\n\n${JSON.stringify(data)}` +
          `\n\nCollection/Store Name: ${dataCollectionInfo?.name}` +
          `\n\nCollection/Store Files: ${dataCollectionInfo?.files}` +
          `\n\nCollection/Store Description: ${dataCollectionInfo?.description}`
        : ""),
  };
  newMessages.unshift(sysPrompt);

  const stream = await openai.chat.completions.create(
    {
      model: userSettings.model as string,
      messages: newMessages,
      stream: true,
      temperature: Number(userSettings.temperature),
    },
    { signal }
  );

  const newMessage: Message = {
    role: "assistant",
    content: "",
    timestamp: new Date(),
    data_content: data ? JSON.stringify(data) : undefined,
  };

  try {
    for await (const chunk of stream) {
      if (signal?.aborted) {
        throw new Error("AbortError");
      }
      const content = chunk.choices[0]?.delta?.content || "";
      newMessage.content += content;
      sendMessageChunk(content, mainWindow);
    }

    if (mainWindow) {
      mainWindow.webContents.send("streamEnd");
    }

    return {
      id: conversationId,
      messages: [...messages, newMessage],
      title: currentTitle,
      content: newMessage.content,
      aborted: false
    };
  } catch (error) {
    if (
      signal?.aborted ||
      (error instanceof Error && error.message === "AbortError")
    ) {
      return {
        id: conversationId,
        messages: messages,
        title: currentTitle,
        content: "",
        aborted: true
      };
    }
    throw error;
  }
}
