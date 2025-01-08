import { GoogleGenerativeAI } from "@google/generative-ai";

let genAI: GoogleGenerativeAI;
async function initializeGemini(apiKey: string) {
  genAI = new GoogleGenerativeAI(apiKey);
}

export async function GeminiProviderAPIKeyCheck(apiKey: string): Promise<{
  error?: string;
  success?: boolean;
}> {
  if (!apiKey) {
    throw new Error("Gemini API key not found for the active user");
  }
  await initializeGemini(apiKey);

  if (!genAI) {
    throw new Error("Gemini instance not initialized");
  }

  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const result = await model.generateContent("Hello, world!");
  const response = await result.response;
  if (response.text()) {
    return {
      success: true,
    };
  }

  return {
    error: "Gemini API key is invalid",
  };
}
