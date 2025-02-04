import { ChatRequestOptions as BaseChatRequestOptions } from 'ai';

export interface ChatRequestOptions extends BaseChatRequestOptions {
  experimental_deepResearch?: boolean;
}
