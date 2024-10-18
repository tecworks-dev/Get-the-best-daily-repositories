export enum ChatModels {
  QWEN_TURBO = 'qwen-turbo',
  QWEN_PLUS = 'qwen-plus',
  QWEN_MAX = 'qwen-max',
}

export type ChatModel = {
  chat: (data: Record<string, any>) => Promise<any>
  streamChat: (data: Record<string, any>) => any
}

export type ChatRole = 'user' | 'assistant'
