import { create } from 'zustand';
import { ChatType } from '@/app/db/schema';
import { getChatInfoInServer } from '@/app/chat/actions/chat';

interface IChatStore {
  chat: ChatType | null;
  setChat: (chat: ChatType) => void;
  initializeChat: (chatId: string) => void;
}

const useChatStore = create<IChatStore>((set) => ({
  chat: null,
  setChat: (chat: ChatType) => {
    set({ chat: chat });
  },

  initializeChat: async (chatId: string) => {
    const result = await getChatInfoInServer(chatId);
    set({ chat: result.data });
  },

}))

export default useChatStore
