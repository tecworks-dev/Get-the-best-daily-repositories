import { create } from 'zustand';
import { updateChatTitleInServer, addChatInServer, updateChatInServer } from '@/app/chat/actions/chat';
import { addBotToChatInServer } from '@/app/chat/actions/bot';
import { ChatType, BotType } from '@/app/db/schema';

interface IChatListStore {
  chatList: ChatType[];
  setNewTitle: (chatId: string, newTitle: string) => void;
  setChatList: (chatList: ChatType[]) => void;
  updateChat: (chatId: string, chat: {
    title?: string;
    defaultModel?: string;
    historyType?: 'all' | 'none' | 'count';
    historyCount?: number;
    isStar?: boolean;
    isWithBot?: boolean;
    botId?: number;
    avatar?: string;
    avatarType?: 'emoji' | 'url' | 'none';
    prompt?: string;
    starAt?: Date;
  }) => void;
  addBot: (botId: number) => void;
}

const useChatListStore = create<IChatListStore>((set) => ({
  chatList: [],
  setNewTitle: (chatId: string, newTitle: string) => {
    set((state) => {
      updateChatTitleInServer(chatId, newTitle);
      // 同步更新聊天列表
      const chatList = state.chatList.map(chat => {
        if (chat.id === chatId) {
          return { ...chat, title: newTitle };
        }
        return chat;
      });
      return { chatList };
    });
  },
  updateChat: (chatId: string, newChatInfo) => {
    set((state) => {
      updateChatInServer(chatId, newChatInfo);
      const chatList = state.chatList.map(chat => {
        if (chat.id === chatId) {
          return { ...chat, ...newChatInfo };
        }
        return chat;
      });
      return { chatList };
    });
  },
  setChatList: (chatList: ChatType[]) => {
    set((state) => {
      return { chatList: chatList };
    });
  },

  addBot: async (botId: number) => {
    const result = await addBotToChatInServer(botId);
    set((state) => ({
      chatList: [result.data as ChatType, ...state.chatList],
    }));
  },

}))

export default useChatListStore
