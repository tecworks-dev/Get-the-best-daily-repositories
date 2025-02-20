import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'siliconflow',
  providerName: '硅基流动',
}

export const modelList: LLMModel[] = [
  {
    'id': 'deepseek-ai/DeepSeek-V3',
    'displayName': 'DeepSeek V3',
    'supportVision': false,
    'maxTokens': 65536,
    'selected': true,
    provider
  },
  {
    'id': 'deepseek-ai/DeepSeek-R1',
    'displayName': 'DeepSeek R1',
    'supportVision': false,
    'maxTokens': 65536,
    'selected': true,
    provider
  },
]