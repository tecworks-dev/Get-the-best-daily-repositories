import React, { useState, useEffect } from 'react';
import Image from "next/image";
import Link from 'next/link';
import clsx from 'clsx';
import { Modal, Input, Skeleton } from 'antd';
import { ChatType } from '@/app/db/schema';
import { EditOutlined, DeleteOutlined, StarOutlined, StarFilled, PlusOutlined } from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { usePathname, useRouter } from 'next/navigation';
import { Dropdown, message } from 'antd';
import { More } from '@icon-park/react';
import MessageIcon from "@/app/images/message.svg";
import TopIcon from "@/app/images/top.svg";
import CancelTopIcon from "@/app/images/cancelTop.svg";
import DeleteIcon from "@/app/images/delete.svg";
import ThumbtackIcon from "@/app/images/thumbtack.svg";
import PlusIcon from "@/app/images/plus.svg";
import Spark from "@/app/images/spark.svg";
import useChatListStore from '@/app/store/chatList';
import MenuSection from '@/app/components/SidebarMenuSection';
import { getChatListInServer, deleteChatInServer, updateChatInServer } from '@/app/chat/actions/chat';
import { useTranslations } from 'next-intl';
import useChatStore from '@/app/store/chat';
import { useSession } from 'next-auth/react';

const ChatList = () => {
  const t = useTranslations('Chat');
  const [messageApi, contextHolder] = message.useMessage();
  const pathname = usePathname();
  const router = useRouter();
  const currentChatId = pathname.split("/").pop() || '';
  const [highlightedChat, setHighlightedChat] = useState('');
  const [newChatName, setNewChatName] = useState('');
  const [renameChatId, setRenameChatId] = useState('');
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [chatListStatus, setChatListStatus] = useState('init');

  const { chatList, setChatList, updateChat, setNewTitle } = useChatListStore();
  const [top8ChatsWithoutBot, setTop8ChatsWithoutBot] = useState<ChatType[]>([]);
  const [isOver8, setIsOver8] = useState(false);
  const { chat, setChat } = useChatStore();
  const { data: session, status } = useSession();

  const handleOpenChange = (isOpen: boolean, chatId: string) => {
    if (isOpen) {
      setHighlightedChat(chatId);
    } else {
      setHighlightedChat('');
    }
  };

  useEffect(() => {
    let top8Chats = chatList.filter(chat => !chat.isWithBot);
    if (top8Chats.length > 8) {
      top8Chats = top8Chats.slice(0, 8);
      setIsOver8(true);
    } else {
      setIsOver8(false);
    }
    setTop8ChatsWithoutBot(top8Chats)
  }, [chatList]);

  useEffect(() => {
    const fetchAllChats = async () => {
      const result = await getChatListInServer();
      if (result.status === 'success') {
        setChatList(result.data as ChatType[]);
      } else {
        setChatList([]);
      }
      setChatListStatus('done')
    };
    fetchAllChats();
  }, [status, setChatList]);

  const deleteChat = async (chat_id: string) => {
    const result = await deleteChatInServer(chat_id);

    if (result.status === 'success') {
      messageApi.success(t('deleteSuccess'));
      const chatListresult = await getChatListInServer();
      setChatList(chatListresult.data as ChatType[]);

      if (currentChatId === chat_id) {
        if (chatListresult.data && chatListresult.data?.length > 0) {
          router.push(`/chat/${chatListresult.data[0]['id']}`)
        } else {
          router.push(`/chat/`)
        }
      }
    } else {
      messageApi.success(t('deleteFail'));
    }
  };

  const toggleStar = async (chat_id: string, is_star: boolean) => {
    await updateChatInServer(chat_id, { isStar: is_star, starAt: new Date() })
    if (currentChatId === chat_id && chat) {
      setChat({ ...chat, isStar: !Boolean(chat.isStar), starAt: new Date() });
    }
    updateChat(chat_id, { isStar: is_star, starAt: new Date() });
  }

  const showEditModal = () => {
    setIsEditModalOpen(true);
  };

  const handleSaveTitle = () => {
    setNewTitle(renameChatId, newChatName);
    setIsEditModalOpen(false);
  };

  const handleCancel = () => {
    setIsEditModalOpen(false);
  };
  const getItems = (isStar: boolean): MenuProps['items'] => {
    let icon
    if (isStar) {
      icon = <StarFilled style={{ color: '#f7b83c' }} />
    } else {
      icon = <StarOutlined />
    }
    return [
      {
        label: t('rename'),
        icon: <EditOutlined />,
        key: 'edit',
      },
      {
        type: 'divider',
      },
      {
        label: t('delete'),
        key: 'delete',
        icon: <DeleteOutlined />,
      },
    ];
  };

  const getBotActionItems = (isTop: boolean): MenuProps['items'] => {
    let pinItem;
    if (isTop) {
      pinItem = {
        label: t('unPin'),
        icon: <CancelTopIcon width={18} height={18} />,
        key: 'top',
      }
    } else {
      pinItem = {
        label: t('pin'),
        icon: <TopIcon width={20} height={20} />,
        key: 'top',
      }
    }
    return [
      pinItem,
      {
        type: 'divider',
      },
      {
        label: t('delete'),
        key: 'delete',
        icon: <DeleteIcon width={20} height={20} />,
      },
    ];
  };
  return (
    <>
      {contextHolder}
      <div className="flex flex-col box-border pt-2 pl-0 pr-4">
        <Link href='/chat'>
          <div className="w-full border rounded-xl text-center p-2 text-sm new-chat-button whitespace-nowrap">
            <PlusOutlined className='mr-2 ' style={{ color: '#0057ff' }} />{t('newChat')}</div>
        </Link>
      </div>
      <div className="rounded-xl overflow-hidden mt-4 overflow-y-auto  scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
        <MenuSection
          title={t('recentChat')}
          icon={<MessageIcon width='20' height='20' />}
          defaultExpanded={true}>

          {chatListStatus !== 'done' ?
            <div className='pl-8 py-4 pr-6'>
              <Skeleton title={false} paragraph={{ rows: 6, width: ['60%', '70%', '70%', '60%', '70%', '70%'] }} active />
            </div> :
            <>
              {chatList.filter(chat => !chat.isWithBot).length === 0 && <div className='flex flex-col'>
                <span className='my-2 text-xs text-gray-400 ml-8'>{t('historyNotice')}</span>
              </div>
              }
              <ul className="pr-4">
                {top8ChatsWithoutBot.map((chat) => (
                  <Link key={chat.id} href={`/chat/${chat.id}`}>
                    <li key={chat.id}
                      style={{ fontSize: '13px' }}
                      className={clsx({ "bg-white hover:bg-white font-medium text-gray-800": chat.id === currentChatId }, { "bg-gray-200": highlightedChat === chat.id }, "pr-2 ml-5 pl-3 py-1.5 rounded-xl text-gray-500 relative group mt-1 hover:bg-gray-200")}>
                      <div className="flex items-center justify-between w-full grow">
                        <div className="whitespace-nowrap w-0 grow overflow-hidden text-ellipsis"
                        >{chat.title}</div>
                        <Dropdown
                          menu={{
                            items: getItems(chat.isStar || false), onClick: (e) => {
                              e.domEvent.preventDefault();
                              if (e.key === 'delete') {
                                deleteChat(chat.id);
                              }
                              if (e.key === 'edit') {
                                setNewChatName(chat.title || '');
                                setRenameChatId(chat.id);
                                showEditModal();
                              }
                            }
                          }}
                          onOpenChange={(isOpen) => {
                            handleOpenChange(isOpen, chat.id)
                          }}
                          trigger={['click']}>
                          <div onClick={(e) => e.preventDefault()} className={clsx({ 'bg-gray-100': highlightedChat === chat.id }, 'rounded hover:bg-gray-100')} >
                            <More
                              theme="outline"
                              size="20"
                              className={clsx({ "visible": highlightedChat === chat.id, "invisible": highlightedChat !== chat.id }, 'h-5 w-5 group-hover:visible hover:bg-gray-100')}
                              fill="#9ca3af"
                              strokeWidth={2} />
                          </div>
                        </Dropdown>
                      </div>
                    </li>
                  </Link>

                ))}
                {isOver8 && <Link href='/chat/thread/list'>
                  <li
                    style={{ fontSize: '13px' }}
                    className={clsx({ "bg-white hover:bg-white font-medium text-gray-800": '/chat/thread/list' === pathname }, "pr-2 ml-5 pl-3 py-1.5 rounded-xl text-gray-500 relative group mt-1 hover:bg-gray-200")}>
                    <div className="flex items-center justify-between w-full grow">
                      <div className="whitespace-nowrap w-0 grow overflow-hidden text-ellipsis"
                      >{t('viewAll')}...</div>
                    </div>
                  </li>
                </Link>}
              </ul>
            </>
          }
        </MenuSection>

        <MenuSection title={t('myBots')} icon={<Spark width={20} height={20} alt='spark' />} defaultExpanded={false}>
          <ul className="pr-4">
            <Link href={`/chat/bot/discover`}>
              <li
                style={{ fontSize: '13px' }}
                className={clsx({ 'bg-white font-medium text-gray-800': pathname === '/bot/discover' }, "pr-2 ml-0 pl-2 py-2 rounded-xl text-gray-500 relative group mt-1 hover:bg-gray-200")}>
                <div className="flex items-center justify-between w-full grow">
                  <div style={{ width: '20px', height: '20px' }} className="flex items-center justify-center">
                    <PlusIcon width={18} height={18} alt='add' />
                  </div>
                  <div className="ml-1 whitespace-nowrap w-0 grow overflow-hidden text-ellipsis"
                  >{t('discoverBots')}</div>
                </div>
              </li>
            </Link>
            {chatList.filter(chat => chat.isWithBot).sort((a, b) => {
              if (a?.isStar && a.starAt && b?.isStar && b.starAt) {
                return b.starAt.getTime() - a.starAt.getTime();
              }
              if (a?.isStar && !b?.isStar) {
                return -1;
              }
              if (!a?.isStar && b?.isStar) {
                return 1;
              } else {
                return a.createdAt.getTime() - b.createdAt.getTime();
              }
            }
            ).map((chat) => (
              <Link key={chat.id} href={`/chat/${chat.id}`}>
                <li key={chat.id}
                  style={{ fontSize: '13px' }}
                  className={clsx({ "bg-white hover:bg-white font-medium text-gray-800": chat.id === currentChatId }, { "bg-gray-200": highlightedChat === chat.id }, "pr-2 ml-0 pl-2 py-2 rounded-xl text-gray-500 relative group mt-1 hover:bg-gray-200")}>
                  <div className="flex items-center justify-between w-full grow">
                    <div style={{ width: '22px', height: '22px' }} className="flex items-center justify-center bg-slate-200 rounded-full">
                      {chat.avatarType === 'emoji' &&
                        <span className='text-base'>{chat.avatar}</span>}
                      {chat.avatarType === 'url' &&
                        <Image
                          src={chat.avatar as string}
                          unoptimized
                          className='rounded-full'
                          width={20}
                          height={20}
                          alt='chat bot'
                        />}

                    </div>
                    <div className="ml-1 flex flex-row whitespace-nowrap w-0 grow"
                    >
                      <span className='text-ellipsis overflow-hidden'>{chat.title}</span>
                      {chat?.isStar && <ThumbtackIcon className='ml-2' width={16} height={16} alt='thumbtack' />}</div>
                    <Dropdown
                      menu={{
                        items: getBotActionItems(Boolean(chat.isStar)), onClick: (e) => {
                          e.domEvent.preventDefault();
                          if (e.key === 'delete') {
                            deleteChat(chat.id);
                          }
                          if (e.key === 'top') {
                            toggleStar(chat.id, !chat.isStar);
                          }
                        }
                      }}
                      onOpenChange={(isOpen) => {
                        handleOpenChange(isOpen, chat.id)
                      }}
                      trigger={['click']}>
                      <div onClick={(e) => e.preventDefault()} className={clsx({ 'bg-gray-100': highlightedChat === chat.id }, 'rounded hover:bg-gray-100')} >
                        <More
                          theme="outline"
                          size="20"
                          className={clsx({ "visible": highlightedChat === chat.id, "invisible": highlightedChat !== chat.id }, 'h-5 w-5 group-hover:visible hover:bg-gray-100')}
                          fill="#9ca3af"
                          strokeWidth={2} />
                      </div>
                    </Dropdown>
                  </div>
                </li>
              </Link>
            ))}
          </ul>
        </MenuSection>
      </div>
      <Modal title={t('editChatName')}
        open={isEditModalOpen}
        onOk={handleSaveTitle}
        onCancel={handleCancel}
        cancelText={t('cancel')}
        okText={t('save')}
      >
        <Input
          value={newChatName}
          onChange={(e) => setNewChatName(e.target.value)}
          placeholder={t('chatNameplaceholder')}
          style={{ marginTop: '1em', marginBottom: '1em' }}
        />
      </Modal>
    </>
  );
};

export default ChatList;