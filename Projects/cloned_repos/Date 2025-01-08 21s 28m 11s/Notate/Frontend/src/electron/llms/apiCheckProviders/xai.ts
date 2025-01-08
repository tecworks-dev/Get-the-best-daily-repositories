import OpenAI from "openai";

let openai: OpenAI;

async function initializeXAI(apiKey: string) {
  openai = new OpenAI({ apiKey, baseURL: "https://api.x.ai/v1" });
}

export async function XAIProviderAPIKeyCheck(apiKey: string): Promise<{
  error?: string;
  success?: boolean;
}> {
  if (!apiKey) {
    throw new Error("XAI API key not found for the active user");
  }
  await initializeXAI(apiKey);

  if (!openai) {
    throw new Error("XAI instance not initialized");
  }

  const response = await openai.chat.completions.create({
    model: "grok-beta",
    messages: [{ role: "user", content: "Hello, world!" }],
    max_tokens: 10,
  });

  if (response.choices[0]?.message?.content) {
    return {
      success: true,
    };
  }

  return {
    error: "XAI API key is invalid",
  };
}
