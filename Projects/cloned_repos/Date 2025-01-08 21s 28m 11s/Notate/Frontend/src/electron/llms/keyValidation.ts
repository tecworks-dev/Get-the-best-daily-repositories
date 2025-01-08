import { OpenAIProviderAPIKeyCheck } from "./apiCheckProviders/openai.js";
import { AnthropicProviderAPIKeyCheck } from "./apiCheckProviders/anthropic.js";
import { GeminiProviderAPIKeyCheck } from "./apiCheckProviders/gemini.js";
import { XAIProviderAPIKeyCheck } from "./apiCheckProviders/xai.js";

export async function keyValidation({
  apiKey,
  inputProvider,
}: {
  apiKey: string;
  inputProvider: string;
}): Promise<{
  error?: string;
  success?: boolean;
}> {
  try {
    let provider;
    switch (inputProvider) {
      case "openai":
        provider = OpenAIProviderAPIKeyCheck;
        break;
      case "anthropic":
        provider = AnthropicProviderAPIKeyCheck;
        break;
      case "gemini":
        provider = GeminiProviderAPIKeyCheck;
        break;
      case "xai":
        provider = XAIProviderAPIKeyCheck;
        break;
      default:
        throw new Error(
          "No AI provider selected. Please open Settings (top right) make sure you add an API key and select a provider under the 'AI Provider' tab."
        );
    }

    const result = await provider(apiKey);

    return {
      ...result,
    };
  } catch (error) {
    console.error("Error in chat request:", error);
    return {
      error: "Error in chat request",
    };
  }
}
