import {
  GoogleGenerativeAI,
  GenerativeModel,
  ChatSession,
} from "@google/generative-ai";
import db from "../../db.js";
import { BrowserWindow } from "electron";
import { sendMessageChunk } from "../llms.js";

let genAI: GoogleGenerativeAI;

async function initializeGemini(apiKey: string) {
  genAI = new GoogleGenerativeAI(apiKey);
}

export async function GeminiProvider(
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
  },
  signal?: AbortSignal
) {
  const apiKey = db.getApiKey(activeUser.id, "gemini");
  if (!apiKey) {
    throw new Error("Gemini API key not found for the active user");
  }
  await initializeGemini(apiKey);

  if (!genAI) {
    throw new Error("Gemini instance not initialized");
  }

  let dataCollectionInfo;
  if (collectionId) {
    dataCollectionInfo = db.getCollection(collectionId) as Collection;
  }

  const model: GenerativeModel = genAI.getGenerativeModel({
    model: userSettings.model as string,
    systemInstruction:
      prompt +
      (data
        ? "The following is the data that the user has provided via their custom data collection: " +
          `\n\n${JSON.stringify(data)}` +
          `\n\nCollection/Store Name: ${dataCollectionInfo?.name}` +
          `\n\nCollection/Store Files: ${dataCollectionInfo?.files}` +
          `\n\nCollection/Store Description: ${dataCollectionInfo?.description}`
        : ""),
  });
  const temperature = Number(userSettings.temperature);
  const chat: ChatSession = model.startChat({
    history: messages.map((msg) => ({
      role: msg.role === "assistant" ? "model" : "user",
      parts: [{ text: msg.content }],
    })),
    generationConfig: {
      temperature: temperature,
    },
  });

  const newMessage: Message = {
    role: "assistant",
    content: "",
    timestamp: new Date(),
    data_content: data ? JSON.stringify(data) : undefined,
  };

  try {
    const result = await chat.sendMessageStream(
      messages[messages.length - 1].content,
      { signal }
    );

    let buffer = "";
    for await (const chunk of result.stream) {
      if (signal?.aborted) {
        throw new Error("AbortError");
      }
      let content = "";

      if (typeof chunk.text === "function") {
        content = chunk.text();
      } else if (chunk.candidates && chunk.candidates.length > 0) {
        const candidate = chunk.candidates[0];
        if (candidate.content && candidate.content.parts) {
          content = candidate.content.parts
            .filter((part) => part.text)
            .map((part) => part.text)
            .join("");
        }
      }

      if (content) {
        buffer += content;
        while (buffer.length >= 1) {
          const chunkToSend = buffer.slice(0, 1);
          buffer = buffer.slice(1);
          newMessage.content += chunkToSend;
          sendMessageChunk(chunkToSend, mainWindow);
        }
      }
    }

    if (buffer.length > 0) {
      newMessage.content += buffer;
      sendMessageChunk(buffer, mainWindow);
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
      if (mainWindow) {
        mainWindow.webContents.send("streamEnd");
      }
      return {
        id: conversationId,
        messages: [...messages, { ...newMessage }],
        title: currentTitle,
        content: newMessage.content,
        aborted: true
      };
    }
    throw error;
  }
}
