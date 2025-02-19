import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'qianfan',
  providerName: '百度云千帆',
}

export const modelList: LLMModel[] = [
  {
    'id': 'ernie-4.0-8k-latest',
    'displayName': 'ERNIE 4.0',
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'ernie-4.0-turbo-8k-latest',
    'displayName': 'ERNIE 4.0 Turbo',
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'ernie-speed-pro-128k',
    'displayName': 'ERNIE Speed',
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'deepseek-v3',
    'displayName': 'DeepSeek V3',
    'supportVision': false,
    'selected': true,
    provider
  },
  {
    'id': 'deepseek-r1',
    'displayName': 'DeepSeek-R1',
    'supportVision': false,
    'selected': true,
    provider
  }
]