import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'ollama',
  providerName: 'Ollama',
}

export const modelList: LLMModel[] = [
  {
    'id': 'llama3.2:3b',
    'displayName': 'Llama3.2 3B',
    'supportVision': false,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
  {
    'id': 'llama3.2-vision',
    'displayName': 'Llama3.2 vision',
    'supportVision': true,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
]