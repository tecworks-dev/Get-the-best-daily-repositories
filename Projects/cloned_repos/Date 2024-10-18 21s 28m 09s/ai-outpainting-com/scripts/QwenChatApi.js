const GENERATION_URL =
  'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'

/**
 * 通义千问ai接口服务类
 */
class QwenChatApi {
  static API_KEY = 'sk-e291921d55444a72a3c5cd345452e525'
  model
  encoder = new TextEncoder()
  decoder = new TextDecoder('utf-8')
  chatStream
  constructor({ model = 'qwen-turbo', api_key = QwenChatApi.API_KEY } = {}) {
    this.model = model
    QwenChatApi.API_KEY = api_key
  }
  iteratorToStream(iterator) {
    return new ReadableStream({
      async pull(controller) {
        const { value, done } = await iterator.next()

        if (done) {
          controller.close()
        } else {
          controller.enqueue(value)
        }
      },
    })
  }
  async *makeIterator() {
    try {
      if (!this.chatStream) {
        yield 'none'
        return
      }
      for await (const message of this.chatStream) {
        const decodedMessage = this.decoder.decode(message)

        // Find the start of the JSON content
        const jsonStartIndex = decodedMessage.indexOf('{')
        if (jsonStartIndex === -1) {
          console.log('No JSON content found in the message.')
          continue
        }

        // Extract the JSON content
        const jsonString = decodedMessage.substring(jsonStartIndex)

        try {
          const parse = JSON.parse(jsonString)
          // Process each message
          yield this.encoder.encode(parse['output']['text'])
        } catch (error) {
          console.error('GlobalError parsing JSON:', error)
        }
      }
    } catch (err) {
      console.error('GlobalError during streaming:', err)
    }
  }

  streamChat(messages = [], history = []) {
    let resolveNext
    let rejectNext
    let pending = false
    let done = false
    let queue = []
    // 创建异步迭代器
    const asyncIterator = {
      next() {
        if (queue.length > 0) {
          // 如果队列中有数据，立即返回
          return Promise.resolve({ value: queue.shift(), done: false })
        } else if (done) {
          // 如果流已经结束，返回结束信号
          return Promise.resolve({ done: true })
        } else {
          // 否则，等待下一个值
          pending = true
          return new Promise((resolve, reject) => {
            resolveNext = resolve
            rejectNext = reject
          })
        }
      },

      // 可选：提供一个方法来手动关闭迭代器
      return() {
        done = true
        if (pending) {
          resolveNext({ done: true })
        }
        return Promise.resolve({ done: true })
      },
    }
    const requestData = {
      model: this.model,
      input: { messages: [...history, ...messages] },
      parameters: {},
    }
    async function fetchAi() {
      const response = await fetch(GENERATION_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          Authorization: `Bearer ${QwenChatApi.API_KEY}`,
        },
        body: JSON.stringify(requestData),
      })
      if (!response) {
        // @ts-ignore
        rejectNext(new Error('No response'))
      } else {
        // @ts-ignore
        for await (const chunk of response?.body) {
          // Do something with each chunk
          // Here we just accumulate the size of the response.
          if (pending) {
            // @ts-ignore
            resolveNext({ value: chunk, done: false })
            pending = false
          } else {
            queue.push(chunk)
          }
        }
        if (pending) {
          // @ts-ignore
          resolveNext({ done: true })
        }
      }
    }
    fetchAi().then((r) => {})
    return {
      [Symbol.asyncIterator]() {
        return asyncIterator
      },
    }
  }

  /**
   * 流式接口调用 api
   * @param messages
   * @param history
   */
  stream(messages = [], history = []) {
    this.chatStream = this.streamChat(messages, history)
    const iterator = this.makeIterator()
    return this.iteratorToStream(iterator)
  }

  /**
   * 非流式接口调用 api
   * @param messages
   * @param history
   */
  chat(messages = [], history = []) {
    const requestData = {
      model: this.model,
      input: { messages: [...history, ...messages] },
      parameters: {},
    }
    const requestStr = JSON.stringify(requestData)
    console.log('qwen request:', requestStr)
    return fetch(GENERATION_URL, {
      method: 'POST', // Assuming it's a POST request
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${QwenChatApi.API_KEY}`,
      },
      body: requestStr,
    }).then((response) => response.json())
  }
}
function StreamResponse(stream) {
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      Connection: 'keep-alive',
      'Cache-Control': 'no-cache, no-transform',
    },
  })
}

async function chat(prompt) {
  const qwenChatApi = new QwenChatApi({ model: 'qwen-turbo' })
  const msg = await qwenChatApi.chat([{ role: 'user', content: prompt }], [])
  console.log('##ai返回结果', JSON.stringify(msg))
  return msg.output.text
}

module.exports = {
  chat,
}
