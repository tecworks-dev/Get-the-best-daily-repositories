const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8')

export async function fetchChat<T = unknown>(
  url: string,
  apiKey: string,
  body: Record<string, any>,
): Promise<T> {
  return fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  }).then((res) => res.json())
}

function toResponseStream(iterator: any) {
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

async function* createIterator(stream: any) {
  try {
    if (!stream) {
      yield 'none'
      return
    }
    for await (const message of stream) {
      const decodedMessage = decoder.decode(message)

      // Find the start of the JSON content
      const jsonStartIndex = decodedMessage.indexOf('{')
      if (jsonStartIndex === -1) {
        console.log('No JSON content found in the message.')
        continue
      }

      // Extract the JSON content
      const jsonString = decodedMessage.substring(jsonStartIndex)

      try {
        // Process each message
        yield encoder.encode(JSON.parse(jsonString))
      } catch (error) {
        console.error('GlobalError parsing JSON:', error)
      }
    }
  } catch (err) {
    console.error('GlobalError during streaming:', err)
  }
}

export function fetchStreamChat(
  url: string,
  apiKey: string,
  body: Record<string, any>,
) {
  function createStream() {
    let resolveNext = null as any
    let rejectNext = null as any
    let pending = false
    let done = false
    let queue: any[] = []
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

    async function fetchAi() {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
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

  const stream = createStream()
  const iter = createIterator(stream)
  return toResponseStream(iter)
}
