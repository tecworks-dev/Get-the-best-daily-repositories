import Anthropic from "@anthropic-ai/sdk";
import db from "../db.js";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";

async function generateTitleOpenAI(input: string, userId: number) {
  let apiKey = "";
  try {
    apiKey = db.getApiKey(userId, "openai");
  } catch (error) {
    console.error("Error getting API key:", error);
  }
  if (!apiKey) {
    throw new Error("OpenAI API key not found for the active user");
  }
  const openai = new OpenAI({ apiKey });
  const llmTitleRequest = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content:
          "Generate a short, concise title (5 words or less) for a conversation based on the following message: Return the Title only and nothing else example response: 'Meeting with John' Return: 'Meeting with John'",
      },
      {
        role: "user",
        content: input,
      },
    ],
    max_tokens: 20,
  });

  const generatedTitle = llmTitleRequest.choices[0]?.message?.content?.trim();
  return generatedTitle;
}

async function generateTitleAnthropic(input: string, userId: number) {
  let apiKey = "";
  try {
    apiKey = db.getApiKey(userId, "anthropic");
  } catch (error) {
    console.error("Error getting API key:", error);
  }
  if (!apiKey) {
    throw new Error("Anthropic API key not found for the active user");
  }
  const anthropic = new Anthropic({ apiKey });
  const llmTitleRequest = (await anthropic.messages.create({
    model: "claude-3-sonnet-20240229",
    max_tokens: 20,
    system:
      "Generate a short, concise title (5 words or less) for a conversation based on the following message: Return the Title only and nothing else example response: 'Meeting with John' Return: 'Meeting with John'",
    messages: [
      {
        role: "user",
        content: input,
      },
    ],
  })) as unknown as {
    content: { text: string }[];
  };

  const generatedTitle = llmTitleRequest.content[0].text;
  return generatedTitle || "New Conversation";
}

async function generateTitleGemini(input: string, userId: number) {
  let apiKey = "";
  try {
    apiKey = db.getApiKey(userId, "gemini");
  } catch (error) {
    console.error("Error getting API key:", error);
  }
  if (!apiKey) {
    throw new Error("Gemini API key not found for the active user");
  }
  const genAI = new GoogleGenerativeAI(apiKey);

  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  const titleResult = await model.generateContent(
    "Generate a short, concise title (5 words or less) for a conversation based on the following message: Return the Title only and nothing else example response: 'Meeting with John' Return: 'Meeting with John'\n\n" +
      input
  );
  const generatedTitle = titleResult.response.text().trim();

  return generatedTitle ?? "New Conversation";
}

async function generateTitleXAI(input: string, userId: number) {
  let apiKey = "";
  try {
    apiKey = db.getApiKey(userId, "xai");
  } catch (error) {
    console.error("Error getting API key:", error);
  }
  if (!apiKey) {
    throw new Error("XAI API key not found for the active user");
  }
  const openai = new OpenAI({ apiKey, baseURL: "https://api.x.ai/v1" });

  const llmTitleRequest = await openai.chat.completions.create({
    model: "grok-beta",
    messages: [
      {
        role: "system",
        content:
          "Generate a short, concise title (5 words or less) for a conversation based on the following message: Return the Title only and nothing else example response: 'Meeting with John' Return: 'Meeting with John'",
      },
      {
        role: "user",
        content: input,
      },
    ],
    max_tokens: 20,
  });

  const generatedTitle = llmTitleRequest.choices[0]?.message?.content?.trim();
  return generatedTitle;
}
/* http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}' */
async function generateTitleLocal(input: string, model: string) {
  try {
    const messages = [
      {
        role: "system",
        content:
          "Generate a short, concise title (5 words or less) for a conversation based on the following message: Return the Title only and nothing else example response: 'Meeting with John' Return: 'Meeting with John'",
      },
      { role: "user", content: input },
    ];
    const response = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model,
        messages: messages,
        stream: false, // Disable streaming to get a single response
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`
      );
    }

    const text = await response.text();
    // Ollama returns one JSON object per line
    const lines = text.split("\n").filter((line) => line.trim());
    const lastLine = lines[lines.length - 1];
    const lastResponse = JSON.parse(lastLine);
    if (!lastResponse.message?.content) {
      console.warn("Empty response from Ollama:", lastResponse);
      return "New Conversation";
    }

    return lastResponse.message.content.trim() || "New Conversation";
  } catch (error) {
    console.error("Error generating title:", error);
    return "New Conversation";
  }
}

export async function generateTitle(
  input: string,
  userId: number,
  model?: string
) {
  const userSettings = await db.getUserSettings(userId);
  switch (userSettings.provider) {
    case "openai":
      console.log("OpenAI");
      return generateTitleOpenAI(input, userId);
    case "anthropic":
      console.log("Anthropic");
      return generateTitleAnthropic(input, userId);
    case "gemini":
      console.log("Gemini");
      return generateTitleGemini(input, userId);
    case "xai":
      console.log("XAI");
      return generateTitleXAI(input, userId);
    case "local":
      console.log("Local");
      return generateTitleLocal(input, model || "llama3.2");
    default:
      return "New Conversation";
  }
}
