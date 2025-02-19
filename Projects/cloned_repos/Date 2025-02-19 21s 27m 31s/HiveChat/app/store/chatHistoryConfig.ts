import { create } from 'zustand';
import { updateChatInServer, getChatInfoInServer } from '@/app/chat/actions/chat';

interface IChatHistoryConfigStore {
  historyType: 'all' | 'none' | 'count';
  historyCount: number;
  setHistoryType: (chatId: string, newType: 'all' | 'none' | 'count') => void;
  setHistoryCount: (chatId: string, newCount: number) => void;
  initializeHistory: (chatId: string) => void;
}

const chatHistoryConfig = create<IChatHistoryConfigStore>((set) => ({
  historyType: 'count',
  historyCount: 5,
  setHistoryType: (chatId: string, newType: 'all' | 'none' | 'count') => {
    set((state) => {
      updateChatInServer(chatId, { historyType: newType })
      return { historyType: newType }
    });
  },
  setHistoryCount: (chatId: string, newCount: number) => {
    set((state) => {
      updateChatInServer(chatId, { historyCount: newCount })
      return { historyCount: newCount }
    });
  },

  initializeHistory: async (chatId: string) => {
    const { data } = await getChatInfoInServer(chatId);
    set({
      historyType: data?.historyType || 'count',
      historyCount: data?.historyCount || 5
    });
  },

}))

export default chatHistoryConfig
