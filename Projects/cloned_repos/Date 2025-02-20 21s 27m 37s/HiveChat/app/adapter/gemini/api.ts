'use client'
import { fetchEventSource, EventStreamContentType } from '@microsoft/fetch-event-source';
import { ChatOptions, LLMApi, LLMModel, LLMUsage, RequestMessage, ResponseContent } from '@/app/adapter/interface';
import { prettyObject } from '@/app/utils';
import { InvalidAPIKeyError, TimeoutError } from '@/app/adapter/errorTypes';

export type MessageType = {
  role: 'user' | 'model' | 'system';
  parts: Array<
    {
      text: string;
    }
    |
    {
      'inline_data': {
        mime_type: string,
        data: string,
      }
    }
  >;
}

export default class GeminiApi implements LLMApi {
  private controller: AbortController | null = null;
  private answer = '';
  private reasoning_content = '';
  private finished = false;

  prepareMessage<MessageType>(messages: RequestMessage[]): MessageType[] {
    return messages.map(msg => {
      let newRoleName = 'user';
      if (msg.role === 'system') {
        newRoleName = 'model'
      }
      // 处理文本消息
      if (typeof msg.content === 'string') {
        return {
          role: newRoleName,
          parts: [{
            text: msg.content
          }]
        } as MessageType;
      }

      // 处理包含图像的消息
      if (Array.isArray(msg.content)) {
        const formattedContent = msg.content.map(item => {
          if (item.type === 'text') {
            return {
              text: item.text
            }
          };
          if (item.type === 'image') {
            return {
              inline_data: {
                mime_type: item.mimeType || 'image/jpeg',
                data: item.data.replace(/^data:image\/\w+;base64,/, '')
              }
            };
          }
        }).filter(Boolean);

        return {
          role: newRoleName,
          parts: formattedContent
        } as MessageType;
      }

      // 默认返回文本消息
      return {
        role: newRoleName,
        parts: ['']
      } as MessageType;
    });
  }
  async chat(options: ChatOptions) {
    this.answer = '';
    const clear = () => {
      if (!this.finished) {
        this.finished = true;
        if (this.controller) {
          this.controller.abort();
          this.controller = null;
        }
        this.answer = '';
      }
    };
    this.controller = new AbortController();

    const timeoutId = setTimeout(() => {
      this.controller?.abort('timeout');
      options.onError?.(new TimeoutError('Timeout'));
    }, 30000);

    const messages = this.prepareMessage(options.messages)
    try {
      await fetchEventSource('/api/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Provider': 'gemini',
          'X-Model': options.config.model,
          'X-Chat-Id': options.chatId!,
        },
        body: JSON.stringify({
          "contents": messages,
        }),
        signal: this.controller.signal,
        onopen: async (res) => {
          clearTimeout(timeoutId);
          this.finished = false;
          if (
            !res.ok ||
            !res.headers.get("content-type")?.startsWith(EventStreamContentType) ||
            res.status !== 200
          ) {

            let resTextRaw = '';
            try {
              const resTextJson = await res.clone().json();
              resTextRaw = prettyObject(resTextJson);
            } catch {
              resTextRaw = await res.clone().text();
            }
            const responseTexts = [resTextRaw];
            if (res.status >= 400 && res.status < 500) {
              options.onError?.(new InvalidAPIKeyError('Invalid API Key'));
            } else {
              this.answer = responseTexts.join("\n\n");
              options.onError?.(new Error(this.answer));
            }
            clear();
          }
        },
        onmessage: (event) => {
          const text = event.data;
          try {
            const json = JSON.parse(text);
            const candidates = json?.candidates[0];
            const deltaContent = candidates.content.parts[0]?.text;
            if (deltaContent) {
              this.answer += deltaContent;
            }
            options.onUpdate({ content: this.answer });
            if (candidates.finishReason) {
              options.onFinish({ content: this.answer });
              clear();
              return;
            }
          } catch (e) {
            console.error("[Request] parse error", text, event);
          }
        },
        onclose: () => {
          clear();
        },
        onerror: (err) => {
          this.controller = null;
          this.finished = true;
          this.answer = '';
          // 需要 throw，不然框架会自动重试
          throw err;
        },
        openWhenHidden: true,
      });
    } catch (error) {
      if (error instanceof Error) {
        options.onError?.(new InvalidAPIKeyError('Invalid API Key'));
      } else {
        options.onError?.(new Error('An unknown error occurred'));
      }
      clear();
    } finally {
      clearTimeout(timeoutId);
    }
  }
  stopChat = (callback: (responseContent: ResponseContent) => void) => {
    this.finished = true;
    if (this.controller) {
      this.controller.abort();
      this.controller = null;
    }
    callback({
      content: this.answer,
      reasoning_content: this.reasoning_content
    });
    this.answer = '';
  }

  // 实现一个 check() 方法，用来检查 api 是否可用，如果不可用，返回详细的错误信息
  async check(modelId: string, apikey: string, apiUrl: string): Promise<{ status: 'success' | 'error', message?: string }> {
    const headers = {
      'Content-Type': 'application/json',
      'X-Provider': 'gemini',
      'X-Apikey': `${apikey}`,
      'X-Model': modelId
    };
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);
    try {
      const res = await fetch('/api/completions', {
        signal: controller.signal,
        method: 'POST',
        headers,
        body: JSON.stringify({
          "contents": [{
            "role": "user",
            "parts": [{ 'text': "ping" }]
          }],
        }),
      });
      if (!res.ok) {
        let resTextRaw = '';
        try {
          const resTextJson = await res.clone().json();
          resTextRaw = prettyObject(resTextJson);
        } catch {
          resTextRaw = await res.clone().text();
        }
        return {
          status: 'error',
          message: resTextRaw,
        }
      } else {
        clearTimeout(timeoutId);
        return {
          status: 'success'
        }
      }
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') {
        return {
          status: 'error',
          message: '网络连接超时',
        }
      }
      return {
        status: 'error',
        message: (error as Error)?.message || 'Unknown error occurred',
      }
    }
  }
  usage(): Promise<LLMUsage> {
    throw new Error('Method not implemented.');
  }

  models(): Promise<LLMModel[]> {
    throw new Error('Method not implemented.');
  }

}
