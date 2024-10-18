import createChatModel, { ChatModels, NSFW_VIRTUAL_GIRL_FRIEND } from '@/framework/llm'
import { NextRequest } from 'next/server'
import { QwenChatApi, StreamResponse } from './QwenChatApi'

const chatModel = createChatModel(ChatModels.QWEN_TURBO)
const chatApi = new QwenChatApi({model: ChatModels.QWEN_MAX})

export async function POST(request: NextRequest) {
  const { content, stream = false } = await request.json()
  const messages = [
    {
      role: 'system',
      content: NSFW_VIRTUAL_GIRL_FRIEND,
    },
    {
      role: 'user',
      content,
    },
  ]
  const data = {}
  // if (stream) {
  // return new Response(chatModel.streamChat(data))
  // }
  // const res = chatModel.chat(data)
  // return Response.json(res)
  return StreamResponse(chatApi.stream(messages))
}
