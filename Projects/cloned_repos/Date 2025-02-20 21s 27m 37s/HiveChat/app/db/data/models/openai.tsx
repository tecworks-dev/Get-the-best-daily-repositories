import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'openai',
  providerName: 'Open AI',
}

export const modelList: LLMModel[] = [
  {
    'id': 'gpt-4o',
    'displayName': 'GPT 4o',
    'supportVision': true,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
  {
    'id': 'gpt-4o-mini',
    'displayName': 'GPT 4o mini',
    'supportVision': true,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
  {
    'id': 'o1',
    'displayName': 'o1',
    'supportVision': false,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
  {
    'id': 'o1-mini',
    'displayName': 'o1 mini',
    'supportVision': false,
    'maxTokens': 131072,
    'selected': true,
    provider
  },
  {
    'id': 'gpt-4-turbo-preview',
    'displayName': 'GPT 4 Turbo',
    'supportVision': true,
    'maxTokens': 131072,
    'selected': false,
    provider
  },
  {
    'id': 'gpt-4-32k',
    'displayName': 'GPT 4 32k',
    'supportVision': true,
    'maxTokens': 32768,
    'selected': false,
    provider
  },
  {
    'id': 'gpt-3.5-turbo',
    'displayName': 'GPT 3.5 Turbo',
    'maxTokens': 16384,
    'selected': true,
    provider
  },
]