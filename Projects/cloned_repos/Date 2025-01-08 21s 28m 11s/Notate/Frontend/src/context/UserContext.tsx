import React, {
  createContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
interface UserContextType {
  title: string | null;
  setTitle: React.Dispatch<React.SetStateAction<string | null>>;
  activeUser: User | null;
  setActiveUser: React.Dispatch<React.SetStateAction<User | null>>;
  apiKeys: ApiKey[];
  setApiKeys: React.Dispatch<React.SetStateAction<ApiKey[]>>;
  activeConversation: number | null;
  setActiveConversation: React.Dispatch<React.SetStateAction<number | null>>;
  conversations: Conversation[];
  setConversations: React.Dispatch<React.SetStateAction<Conversation[]>>;
  prompts: Prompt[];
  setPrompts: React.Dispatch<React.SetStateAction<Prompt[]>>;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  streamingMessage: string;
  setStreamingMessage: React.Dispatch<React.SetStateAction<string>>;
  filteredConversations: Conversation[];
  setFilteredConversations: React.Dispatch<
    React.SetStateAction<Conversation[]>
  >;
  isSearchOpen: boolean;
  setIsSearchOpen: React.Dispatch<React.SetStateAction<boolean>>;
  searchTerm: string;
  setSearchTerm: React.Dispatch<React.SetStateAction<string>>;
  searchRef: React.RefObject<HTMLDivElement>;
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  newConversation: boolean;
  setNewConversation: React.Dispatch<React.SetStateAction<boolean>>;
  handleResetChat: () => void;
  devAPIKeys: Keys[];
  setDevAPIKeys: React.Dispatch<React.SetStateAction<Keys[]>>;
  fetchDevAPIKeys: () => Promise<void>;
  getUserConversations: () => Promise<void>;
  alertForUser: boolean;
  setAlertForUser: React.Dispatch<React.SetStateAction<boolean>>;
  fetchApiKey: () => Promise<void>;
  fetchPrompts: () => Promise<void>;
  error: string | null;
  setError: React.Dispatch<React.SetStateAction<string | null>>;
  currentRequestId: number | null;
  setCurrentRequestId: React.Dispatch<React.SetStateAction<number | null>>;
  handleChatRequest: (
    collectionId: number | undefined,
    suggestion?: string
  ) => Promise<void>;
  cancelRequest: () => void;
  fetchMessages: () => Promise<void>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const UserProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [activeUser, setActiveUser] = useState<User | null>(null);

  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [activeConversation, setActiveConversation] = useState<number | null>(
    null
  );
  const [filteredConversations, setFilteredConversations] = useState<
    Conversation[]
  >([]);
  const [title, setTitle] = useState<string | null>(null);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [streamingMessage, setStreamingMessage] = useState<string>("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const searchRef = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newConversation, setNewConversation] = useState<boolean>(true);
  const [devAPIKeys, setDevAPIKeys] = useState<Keys[]>([]);
  const [alertForUser, setAlertForUser] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentRequestId, setCurrentRequestId] = useState<number | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        searchRef.current &&
        !searchRef.current.contains(event.target as Node)
      ) {
        setIsSearchOpen(false);
        setSearchTerm("");
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const cancelRequest = useCallback(() => {
    return new Promise<void>((resolve) => {
      if (currentRequestId) {
        window.electron.abortChatRequest(currentRequestId);
        // Give a small delay to ensure the cancellation is processed
        setTimeout(() => {
          setStreamingMessage("");
          resolve();
        }, 100);
      } else {
        resolve();
      }
    });
  }, [currentRequestId]);

  const handleResetChat = useCallback(async () => {
    await cancelRequest();
    setMessages([]);
    setInput("");
    setIsLoading(false);
    setStreamingMessage("");
    setActiveConversation(null);
  }, [cancelRequest]);

  const getUserConversations = useCallback(async () => {
    if (activeUser) {
      const conversations = await window.electron.getUserConversations(
        activeUser.id
      );
      setConversations(conversations.conversations);
    }
  }, [activeUser]);

  const fetchDevAPIKeys = useCallback(async () => {
    if (activeUser) {
      const keys = await window.electron.getDevAPIKeys(activeUser.id);
      setDevAPIKeys(keys.keys);
    }
  }, [activeUser]);

  const fetchApiKey = useCallback(async () => {
    if (activeUser) {
      const apiKeys = await window.electron.getUserApiKeys(activeUser.id);
      const settings = await window.electron.getUserSettings(activeUser.id);
      if (apiKeys.apiKeys.length === 0 && settings.provider !== "local") {
        setAlertForUser(true);
        return;
      }
      setApiKeys(apiKeys.apiKeys as ApiKey[]);
    }
  }, [activeUser]);

  const fetchPrompts = useCallback(async () => {
    if (activeUser) {
      const fetchedPrompts = await window.electron.getUserPrompts(
        activeUser.id
      );
      setPrompts(fetchedPrompts.prompts as Prompt[]);
    }
  }, [activeUser]);

  const handleChatRequest = useCallback(
    async (collectionId: number | undefined, suggestion?: string) => {
      setIsLoading(true);
      const requestId = Date.now();
      setCurrentRequestId(requestId);

      setError(null);
      const newUserMessage = {
        role: "user",
        content: suggestion || input,
        timestamp: new Date(),
      } as Message;
      setMessages((prev) => [...prev, newUserMessage]);
      setInput("");

      try {
        if (!activeUser) {
          throw new Error("Active user not found");
        }
        const result = (await window.electron.chatRequest(
          [...messages, newUserMessage],
          activeUser,
          Number(activeConversation),
          collectionId || undefined,
          undefined,
          requestId
        )) as {
          id: bigint | number;
          messages: Message[];
          title: string;
          error?: string;
        };
        setTitle(result.title);
        if (result.error) {
          setError(result.error);
          setIsLoading(false);
          console.error("Error in chat:", result.error);
        } else {
          const getMessageLength = result.messages.length;
          setMessages((prev) => [
            ...prev,
            result.messages[getMessageLength - 1],
          ]);
          setActiveConversation(Number(result.id));
          if (result.id !== Number(activeConversation)) {
            const latestMessage = result.messages[getMessageLength - 1];
            const newConversation = {
              id: Number(result.id),
              title: result.title,
              userId: activeUser.id,
              created_at: new Date(),
              latestMessageTime: latestMessage?.timestamp
                ? new Date(latestMessage.timestamp).getTime()
                : Date.now(),
            };
            fetchMessages();
            setConversations((prev) => [newConversation, ...prev]);
            setFilteredConversations((prev) => [newConversation, ...prev]);
          } else {
            setConversations((prev) =>
              prev.map((conv) => {
                if (conv.id === Number(result.id)) {
                  const latestMessage = result.messages[getMessageLength - 1];
                  return {
                    ...conv,
                    latestMessageTime: latestMessage?.timestamp
                      ? new Date(latestMessage.timestamp).getTime()
                      : Date.now(),
                  };
                }
                return conv;
              })
            );
            setFilteredConversations((prev) =>
              prev.map((conv) => {
                if (conv.id === Number(result.id)) {
                  const latestMessage = result.messages[getMessageLength - 1];
                  return {
                    ...conv,
                    latestMessageTime: latestMessage?.timestamp
                      ? new Date(latestMessage.timestamp).getTime()
                      : Date.now(),
                  };
                }
                return conv;
              })
            );
          }
        }
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          setError("Request was cancelled");
        } else {
          console.error("Error in chat:", error);
        }
      } finally {
        setIsLoading(false);
        setCurrentRequestId(null);
      }
    },
    [activeUser, activeConversation, input, messages]
  );

  const fetchMessages = useCallback(async () => {
    if (activeConversation) {
      const conversation = conversations.find(
        (conv: Conversation) => conv.id === activeConversation
      );
      if (conversation && activeUser) {
        const newMessages =
          await window.electron.getConversationMessagesWithData(
            activeUser.id,
            conversation.id
          );
        setMessages(newMessages.messages);
      }
    }
  }, [activeConversation, conversations, activeUser]);

  const contextValue = useMemo(
    () => ({
      activeUser,
      setActiveUser,
      apiKeys,
      setApiKeys,
      activeConversation,
      setActiveConversation,
      conversations,
      setConversations,
      prompts,
      setPrompts,
      filteredConversations,
      setFilteredConversations,
      isSearchOpen,
      setIsSearchOpen,
      searchTerm,
      setSearchTerm,
      searchRef,
      messages,
      setMessages,
      newConversation,
      setNewConversation,
      title,
      setTitle,
      input,
      setInput,
      isLoading,
      setIsLoading,
      streamingMessage,
      setStreamingMessage,
      handleResetChat,
      devAPIKeys,
      setDevAPIKeys,
      fetchDevAPIKeys,
      getUserConversations,
      alertForUser,
      setAlertForUser,
      fetchApiKey,
      fetchPrompts,
      fetchMessages,
      error,
      setError,
      currentRequestId,
      setCurrentRequestId,
      handleChatRequest,
      cancelRequest,
    }),
    [
      activeUser,
      apiKeys,
      activeConversation,
      conversations,
      prompts,
      filteredConversations,
      isSearchOpen,
      searchTerm,
      messages,
      newConversation,
      title,
      input,
      isLoading,
      streamingMessage,
      handleResetChat,
      devAPIKeys,
      fetchDevAPIKeys,
      getUserConversations,
      alertForUser,
      fetchApiKey,
      fetchPrompts,
      fetchMessages,
      error,
      currentRequestId,
      handleChatRequest,
      cancelRequest,
    ]
  );

  return (
    <UserContext.Provider value={contextValue}>{children}</UserContext.Provider>
  );
};

export { UserProvider, UserContext };
