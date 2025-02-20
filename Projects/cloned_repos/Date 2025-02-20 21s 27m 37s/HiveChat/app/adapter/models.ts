import ChatGPTApi from '@/app/adapter/openai/api';
import Claude from '@/app/adapter/claude/api';
import GeminiApi from '@/app/adapter/gemini/api';

export const getLLMInstance = (providerId: string) => {
  let llmApi;
  switch (providerId) {
    case 'claude':
      llmApi = new Claude();
      break;
    case 'gemini':
      llmApi = new GeminiApi();
      break;
    default:
      llmApi = new ChatGPTApi(providerId);
      break;
  }
  return llmApi;
}