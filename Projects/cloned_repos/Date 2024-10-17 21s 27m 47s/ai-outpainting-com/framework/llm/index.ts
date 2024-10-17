import { ChatModel, ChatModels } from './types'

import QwenModel from './QwenModel'

export * from './types'
export * from './utils'
export * from './prompts'

export default function createChatModel(model: ChatModels): ChatModel {
  if (model.startsWith('qwen')) {
    return QwenModel(model)
  }
  throw new Error(`暂时不支持该模型：${model}`)
}
