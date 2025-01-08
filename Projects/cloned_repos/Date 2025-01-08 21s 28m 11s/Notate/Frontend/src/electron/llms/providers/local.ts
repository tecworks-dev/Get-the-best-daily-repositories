import { BrowserWindow } from "electron";
import db from "../../db.js";
import { sendMessageChunk } from "../llms.js";

export async function LocalProvider(
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

  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: userSettings.model || "llama2",
      messages: newMessages,
      stream: true,
      keep_alive: -1,
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Ollama API error: ${response.status} ${response.statusText}`
    );
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Failed to get response reader");
  }

  const newMessage: Message = {
    role: "assistant",
    content: "",
    timestamp: new Date(),
    data_content: data ? JSON.stringify(data) : undefined,
  };

  try {
    let buffer = "";
    while (true) {
      if (signal?.aborted) {
        if (mainWindow) {
          mainWindow.webContents.send("streamEnd");
        }
        return {
          id: conversationId,
          messages: [...messages, { ...newMessage }],
          title: currentTitle,
          content: newMessage.content,
          aborted: true,
        };
      }

      const { done, value } = await reader.read();
      if (done) break;

      // Add new data to buffer and split by newlines
      buffer += new TextDecoder().decode(value);
      const lines = buffer.split("\n");

      // Process all complete lines
      for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        try {
          const parsed = JSON.parse(line);
          if (parsed.message?.content) {
            newMessage.content += parsed.message.content;
            sendMessageChunk(parsed.message.content, mainWindow);
          }
        } catch (e) {
          console.warn("Failed to parse line:", line, e);
        }
      }

      // Keep the last incomplete line in the buffer
      buffer = lines[lines.length - 1];
    }

    // Process any remaining data in the buffer
    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer);
        if (parsed.message?.content) {
          newMessage.content += parsed.message.content;
          sendMessageChunk(parsed.message.content, mainWindow);
        }
      } catch (e) {
        console.warn("Failed to parse final buffer:", buffer, e);
      }
    }

    if (mainWindow) {
      mainWindow.webContents.send("streamEnd");
    }

    // Only return message if we have content and weren't aborted
    if (newMessage.content) {
      return {
        id: conversationId,
        messages: [...messages, newMessage],
        title: currentTitle,
        content: newMessage.content,
        aborted: false,
      };
    }

    return {
      id: conversationId,
      messages: messages,
      title: currentTitle,
      content: "",
      aborted: false,
    };
  } catch (error) {
    if (mainWindow) {
      mainWindow.webContents.send("streamEnd");
    }

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
        aborted: true,
      };
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
}
