import { useEffect, useState, useCallback } from 'react';
import { Message } from '@/app/db/schema';
import { ChatOptions, LLMApi, RequestMessage, MessageContent } from '@/app/adapter/interface';
import chatHistoryConfig from '@/app/store/chatHistoryConfig';
import useChatStore from '@/app/store/chat';
import useChatListStore from '@/app/store/chatList';
import { generateTitle } from '@/app/utils';
import { getLLMInstance } from '@/app/adapter/models';
import useModelListStore from '@/app/store/modelList';
import { ResponseContent } from '@/app/adapter/interface';
import { addMessageInServer, getMessagesInServer, deleteMessageInServer, clearMessageInServer } from '@/app/chat/actions/message';
import { localDb } from '@/app/db/localDb';

const useChat = (chatId: string) => {
  const { currentModel } = useModelListStore();
  const { historyType, historyCount, initializeHistory } = chatHistoryConfig();
  const [messageList, setMessageList] = useState<Message[]>([]);
  const [isPending, setIsPending] = useState(false);
  const [responseStatus, setResponseStatus] = useState<"done" | "pending">("done");
  const [chatBot, setChatBot] = useState<LLMApi | null>(null);
  const [responseMessage, setResponseMessage] = useState<ResponseContent>({ content: '', reasoning_content: '' });
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const [input, setInput] = useState('');
  const [userSendCount, setUserSendCount] = useState(0);
  const { chat, initializeChat } = useChatStore();
  const { setNewTitle } = useChatListStore();

  useEffect(() => {
    const llmApi = getLLMInstance(currentModel.provider.id);
    setChatBot(llmApi);

    return () => {
      // 清理 chatBot
      if (llmApi) {
        llmApi.stopChat?.(() => { });
        setChatBot(null);
      }
    };
  }, [currentModel]);

  useEffect(() => {
    initializeHistory(chatId);
  }, [initializeHistory, chatId]);

  useEffect(() => {
    async function fetchMessages() {
      try {
        let messageList: Message[] = [];
        const result = await getMessagesInServer(chatId);
        if (result.status === 'success') {
          messageList = result.data as Message[]
        }
        setMessageList(messageList);
        let tmpUserSendCount = 0;
        messageList.forEach((item) => {
          if (item.role === "user") {
            tmpUserSendCount = tmpUserSendCount + 1;
          }
        });
        setUserSendCount(tmpUserSendCount);
      } catch (error) {
        console.error('Error fetching items from database:', error);
      }
    }
    async function fetchLocalMessages() {
      const localMessage = await localDb.messages.where({ 'chatId': chatId }).toArray();
      setMessageList(localMessage);
      setUserSendCount(1);
      await localDb.messages.clear();
    }
    if (localStorage.getItem('f') === 'home') {
      fetchLocalMessages();
      localStorage.removeItem('f');
    } else {
      setIsPending(true);
      Promise.all([
        fetchMessages(),
        initializeChat(chatId)
      ]).finally(() => {
        setIsPending(false);
      });
    }
  }, [chatId, initializeChat]);

  const handleInputChange = (e: any) => {
    setInput(e.target.value);
  };

  const shouldSetNewTitle = useCallback((messages: RequestMessage[]) => {
    if (userSendCount === 0 && !chat?.isWithBot) {
      const renameModel = currentModel.id
      generateTitle(messages, renameModel, currentModel.provider.id, (message: string) => {
        setNewTitle(chatId, message);
      }, () => { })
    }
  }, [
    chat,
    chatId,
    currentModel,
    userSendCount,
    setNewTitle,
  ]);

  const sendMessage = useCallback(async (messages: RequestMessage[]) => {
    let lastUpdate = Date.now();
    setResponseStatus("pending");
    const options: ChatOptions = {
      messages: messages,
      config: { model: currentModel.id },
      chatId: chatId,
      onUpdate: (responseContent: ResponseContent) => {
        const now = Date.now();
        if (now - lastUpdate < 60) return; // 如果距离上次更新小于 60ms，则不更新
        setResponseMessage(responseContent);
        lastUpdate = now;
      },
      onFinish: async (responseContent: ResponseContent) => {
        const respMessage: Message = {
          role: "assistant",
          chatId: chatId,
          content: responseContent.content,
          reasoninContent: responseContent.reasoning_content,
          providerId: currentModel.provider.id,
          model: currentModel.id,
          type: 'text',
          createdAt: new Date()
        };
        setMessageList(prevList => [...prevList, respMessage]);
        setResponseStatus("done");
        setResponseMessage({ content: '', reasoning_content: '' });
        shouldSetNewTitle(messages);
      },
      onError: async (error) => {
        const respMessage: Message = {
          role: "assistant",
          chatId: chatId,
          content: error?.message || '',
          providerId: currentModel.provider.id,
          model: currentModel.id,
          type: 'error',
          errorType: error?.name || 'unknown error',
          errorMessage: error?.message || '',
          createdAt: new Date()
        };
        setMessageList((m) => ([...m, respMessage]));
        setResponseStatus("done");
        setResponseMessage({ content: '', reasoning_content: '' });
        addMessageInServer(respMessage);
      }
    }
    chatBot?.chat(options);
  }, [
    shouldSetNewTitle,
    chatBot,
    chatId,
    currentModel,
  ]);

  const stopChat = () => {
    setResponseStatus("done");
    chatBot?.stopChat((responseContent: ResponseContent) => {
      if (responseContent.content || responseContent.reasoning_content) {
        const respMessage: Message = {
          role: "assistant",
          chatId: chatId,
          content: responseContent.content,
          reasoninContent: responseContent.reasoning_content,
          providerId: currentModel.provider.id,
          model: currentModel.id,
          type: 'text',
          createdAt: new Date()
        };
        setMessageList((m) => ([...m, respMessage]));
        addMessageInServer(respMessage);
      }
      setResponseMessage({ content: '', reasoning_content: '' });
    });
  }

  const deleteMessage = (index: number) => {
    deleteMessageInServer(messageList[index].id as number);
    setMessageList(messageList.filter((_, i) => i !== index));
  }

  const clearHistory = () => {
    clearMessageInServer(chatId).then(() => {
      setMessageList([])
    });
  }

  const addBreak = async () => {
    if (messageList.length > 0 && messageList.at(-1)?.type === 'break') {
      return;
    }
    const toAddMessage = {
      role: "system",
      chatId: chatId,
      content: '上下文已清除',
      providerId: currentModel.provider.id,
      model: currentModel.id,
      type: 'break' as 'break',
      createdAt: new Date()
    };
    addMessageInServer(toAddMessage);
    setMessageList((m) => ([...m, toAddMessage]));
  }

  const prepareMessage = useCallback((newMessage: RequestMessage): RequestMessage[] => {
    let messages: RequestMessage[] = [];
    let tmpMessages = [];

    const validMessageType = ['text', 'image'];
    const breakIndex = messageList.findLastIndex(item => item.type === 'break');

    if (breakIndex > -1) {
      tmpMessages = messageList.slice(breakIndex)
    } else {
      tmpMessages = messageList;
    }
    messages = tmpMessages.filter((item) => validMessageType.includes(item.type))
      .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));

    if (historyType === 'all') {
      messages = messages;
    }
    if (historyType === 'none') {
      messages = [];
    }
    if (historyType === 'count') {
      if (historyCount > messages.length) {
        messages = messages;
      } else {
        messages = messages.slice(-historyCount);
      }
    }
    messages.push(newMessage);
    if (chat?.prompt) {
      messages.unshift({ role: 'system', content: chat?.prompt })
    }
    return messages;
  }, [
    chat,
    historyCount,
    historyType,
    messageList
  ]);

  const handleSubmit = useCallback(async (message: MessageContent) => {
    if (responseStatus === 'pending') {
      return;
    }
    setResponseStatus("pending");
    setIsUserScrolling(false);

    const currentMessage = {
      role: "user",
      chatId: chatId,
      content: message,
      providerId: currentModel.provider.id,
      model: currentModel.id,
      type: 'text' as const,
      createdAt: new Date()
    };

    setInput('');
    const messages = prepareMessage({
      role: "user",
      content: message,
    })
    setUserSendCount(userSendCount + 1);
    sendMessage(messages);
    addMessageInServer(currentMessage);
    setMessageList((m) => ([...m, currentMessage]));
  }, [
    chatId,
    responseStatus,
    currentModel,
    userSendCount,
    prepareMessage,
    sendMessage,
  ]);

  const prepareMessageFromIndex = (index: number): RequestMessage[] => {
    let messages: RequestMessage[] = [];
    if (historyType === 'all') {
      messages = messageList
        .slice(0, index)
        .filter((item) => item.type !== 'error')
        .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));
    }
    if (historyType === 'none') {
      // 从 index 向前寻找一天正常的用户发送的消息。
      const tmpMessage = messageList.slice(index, index + 1);
      if (tmpMessage[0]?.type === 'error') {
        messages = messageList
          .slice(index - 1, index)
          .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));
      } else {
        messages = tmpMessage
          .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));
      }
    }
    if (historyType === 'count') {
      if (historyCount > messageList.length) {
        messages = messageList
          .slice(0, index)
          .filter((item) => item.type !== 'error')
          .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));
      } else {
        messages = messageList
          .slice(0, index)
          .filter((item) => item.type !== 'error')
          .slice(-historyCount)
          .map(({ content, role }) => ({ content, role: role as 'assistant' | 'user' | 'system' }));
      }
    }
    return messages;
  }
  const retryMessage = (index: number, addNew: boolean = true) => {
    if (addNew) {
      const message = messageList[index];
      handleSubmit(message.content);
    } else {
      // 单独处理重试的逻辑
      setResponseStatus("pending");
      setIsUserScrolling(false);
      setInput('');
      const messages: RequestMessage[] = prepareMessageFromIndex(index);
      sendMessage(messages);
      shouldSetNewTitle(messages)
    }
  }

  return {
    input,
    chat,
    messageList,
    responseStatus,
    responseMessage,
    historyType,
    historyCount,
    isUserScrolling,
    currentModel,
    isPending,
    handleInputChange,
    handleSubmit,
    sendMessage,
    shouldSetNewTitle,
    deleteMessage,
    clearHistory,
    stopChat,
    retryMessage,
    addBreak,
    setIsUserScrolling,
  };
};

export default useChat;