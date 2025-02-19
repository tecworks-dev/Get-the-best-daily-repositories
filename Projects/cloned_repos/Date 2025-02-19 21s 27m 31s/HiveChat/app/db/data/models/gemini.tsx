import { LLMModel } from "@/app/adapter/interface"
export const provider = {
  id: 'gemini',
  providerName: 'Gemini',
}

export const modelList: LLMModel[] = [
  {
    'id': 'gemini-1.5-flash',
    'displayName': 'Gemini 1.5 Flash',
    'supportVision': true,
    'selected': true,
    provider
  },
  {
    'id': 'gemini-1.5-flash-8b',
    'displayName': 'Gemini 1.5 Flash 8B',
    'supportVision': true,
    'selected': true,
    provider
  },
  {
    'id': 'gemini-1.5-pro',
    'displayName': 'Gemini 1.5 Pro',
    'supportVision': true,
    'selected': true,
    provider
  },

]