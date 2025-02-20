import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'deepseek',
  providerName: 'Deepseek',
}

export const modelList: LLMModel[] = [
  {
    'id': 'deepseek-chat',
    'displayName': 'DeepSeek V3',
    'supportVision': false,
    'maxTokens': 65536,
    'selected': true,
    provider
  },
  {
    'id': 'deepseek-reasoner',
    'displayName': 'DeepSeek R1',
    'supportVision': false,
    'maxTokens': 65536,
    'selected': true,
    provider
  }
]