'use client'
import React, { useState, useEffect } from 'react';
import useChatListStore from '@/app/store/chatList';
import { Button, message, Popconfirm, Modal, Input } from 'antd';
import { EditOutlined, DeleteOutlined, } from '@ant-design/icons';
import { ChatType } from '@/app/db/schema';
import InPageCollapsed from '@/app/components/InPageCollapsed';
import { getChatListInServer, deleteChatInServer } from '@/app/chat/actions/chat';
import Link from 'next/link';
import { useTranslations } from 'next-intl';

const List = () => {
  const c = useTranslations('Common');
  const t = useTranslations('Chat');
  const { chatList, setChatList, setNewTitle } = useChatListStore();
  const [todayList, setTodayList] = useState<ChatType[]>([]);
  const [weekList, setWeekList] = useState<ChatType[]>([]);
  const [monthList, setMonthList] = useState<ChatType[]>([]);
  const [otherList, setOtherList] = useState<ChatType[]>([]);
  const [newChatName, setNewChatName] = useState('');
  const [renameChatId, setRenameChatId] = useState('');
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [messageApi, contextHolder] = message.useMessage();
  useEffect(() => {
    const todayStart = new Date(new Date().setHours(0, 0, 0, 0));
    const weekStart = new Date();
    weekStart.setDate(
      new Date().getDate() - (new Date().getDay() || 7) + 1
    );
    weekStart.setHours(0, 0, 0, 0);

    const monthStart = new Date();
    monthStart.setDate(1);
    monthStart.setHours(0, 0, 0, 0);

    const normalChatList = chatList.filter(chat => !chat.isWithBot);
    const todayList = normalChatList.filter((item) => {
      return item.createdAt > todayStart;
    });

    const weekList = normalChatList.filter((item) => {
      return item.createdAt > weekStart && item.createdAt < todayStart;
    });

    const monthList = normalChatList.filter((item) => {
      return item.createdAt > monthStart && item.createdAt < weekStart;
    });

    const otherList = normalChatList.filter((item) => {
      const monthStart = new Date();
      monthStart.setDate(1);
      monthStart.setHours(0, 0, 0, 0);
      return item.createdAt < monthStart;
    });

    setTodayList(todayList);
    setWeekList(weekList);
    setMonthList(monthList);
    setOtherList(otherList);
  }, [chatList]);

  const deleteChat = async (chat_id: string) => {
    const result = await deleteChatInServer(chat_id);

    if (result.status === 'success') {
      messageApi.success(t('deleteSuccess'));
      const chatListresult = await getChatListInServer();
      setChatList(chatListresult.data as ChatType[]);
    }
  };

  const handleSaveTitle = () => {
    setNewTitle(renameChatId, newChatName);
    setIsEditModalOpen(false);
  };

  return (
    <>
      <div className='w-full h-22 p-4'>
        <InPageCollapsed />
      </div>
      <div className="container max-w-3xl mx-auto p-4">
        {contextHolder}
        <div className='w-full flex flex-row justify-between items-center border-b border-gray-200 mb-6'>
          <h1 className='text-xl font-bold mb-4 ml-3'>{t('historyChat')}</h1>
        </div>

        <div>
          {todayList.length > 0 && <h4 className='text-lg font-bold mt-4 mb-2 ml-3'>{t('today')}</h4>}
          <ul>
            {todayList.map((chat) => (
              <li key={chat.id} className='flex flex-row items-center justify-between text-sm hover:bg-gray-100 bg-gray-50 mb-3 group px-3 h-12 rounded-lg'>
                <Link key={chat.id} href={`/chat/${chat.id}`} >
                  <span>{chat.title}</span>
                </Link>
                <span className='text-gray-400 inline-block group-hover:hidden'>{chat.createdAt.toLocaleTimeString('zh-CN', { hour: "numeric", minute: "numeric" })}</span>
                <div className='hidden group-hover:block'>
                  <Button onClick={
                    () => {
                      setNewChatName(chat.title || '');
                      setRenameChatId(chat.id);
                      setIsEditModalOpen(true);
                    }
                  } size='small' type='text' icon={<EditOutlined />} />
                  <Popconfirm
                    title={t('deleteCurrentChat')}
                    description={t('deleteCurrentChatDetail')}
                    onConfirm={() => deleteChat(chat.id)}
                    okText={c('confirm')}
                    cancelText={c('cancel')}
                  >
                    <Button size='small' className='ml-1' type='text' icon={<DeleteOutlined />} />
                  </Popconfirm>
                </div>
              </li>
            ))}
          </ul>

          {weekList.length > 0 && <h4 className='text-lg font-bold mt-4 mb-2 ml-3'>{t('thisWeek')}</h4>}
          <ul >
            {weekList.map((chat) => (
              <li key={chat.id} className='flex flex-row items-center justify-between text-sm hover:bg-gray-100 bg-gray-50 mb-3 group px-3 h-12 rounded-lg'>
                <Link key={chat.id} href={`/chat/${chat.id}`} >
                  <span className=''>{chat.title}</span>
                </Link>
                <span className='text-gray-400 inline-block group-hover:hidden'>{chat.createdAt.toLocaleDateString('zh-CN', { month: "short", day: "numeric" })}</span>
                <div className='hidden group-hover:block'>
                  <Button onClick={
                    () => {
                      setNewChatName(chat.title || '');
                      setRenameChatId(chat.id);
                      setIsEditModalOpen(true);
                    }
                  } size='small' type='text' icon={<EditOutlined />} />
                  <Popconfirm
                    title={t('deleteCurrentChat')}
                    description={t('deleteCurrentChatDetail')}
                    onConfirm={() => deleteChat(chat.id)}
                    okText={c('confirm')}
                    cancelText={c('cancel')}
                  >
                    <Button size='small' className='ml-1' type='text' icon={<DeleteOutlined />} />
                  </Popconfirm>
                </div>
              </li>
            ))}
          </ul>

          {monthList.length > 0 && <h4 className='text-lg font-bold mt-4 mb-2 ml-3'>{t('thisMonth')}</h4>}
          <ul>
            {monthList.map((chat) => (
              <li key={chat.id} className='flex flex-row items-center justify-between text-sm hover:bg-gray-100 bg-gray-50 mb-3 group px-3 h-12 rounded-lg'>
                <Link key={chat.id} href={`/chat/${chat.id}`} >
                  <span className=''>{chat.title}</span>
                </Link>
                <span className='text-gray-400 inline-block group-hover:hidden'>{chat.createdAt.toLocaleDateString('zh-CN', { month: "short", day: "numeric" })}</span>
                <div className='hidden group-hover:block'>
                  <Button onClick={
                    () => {
                      setNewChatName(chat.title || '');
                      setRenameChatId(chat.id);
                      setIsEditModalOpen(true);
                    }
                  } size='small' type='text' icon={<EditOutlined />} />
                  <Popconfirm
                    title={t('deleteCurrentChat')}
                    description={t('deleteCurrentChatDetail')}
                    onConfirm={() => deleteChat(chat.id)}
                    okText={c('confirm')}
                    cancelText={c('cancel')}
                  >
                    <Button size='small' className='ml-1' type='text' icon={<DeleteOutlined />} />
                  </Popconfirm>
                </div>
              </li>
            ))}
          </ul>
          {otherList.length > 0 && <h4 className='text-lg font-bold mt-4 mb-2 ml-3'>{t('earlier')}</h4>}
          <ul>
            {otherList.map((chat) => (
              <li key={chat.id} className='flex flex-row items-center justify-between text-sm hover:bg-gray-100 bg-gray-50 mb-3 group px-3 h-12 rounded-lg'>
                <Link key={chat.id} href={`/chat/${chat.id}`} >
                  <span className=''>{chat.title}</span>
                </Link>
                <span className='text-gray-400 inline-block group-hover:hidden'>{chat.createdAt.toLocaleDateString('zh-CN', { month: "short", day: "numeric" })}</span>
                <div className='hidden group-hover:block'>
                  <Button onClick={
                    () => {
                      setNewChatName(chat.title || '');
                      setRenameChatId(chat.id);
                      setIsEditModalOpen(true);
                    }
                  } size='small' type='text' icon={<EditOutlined />} />
                  <Popconfirm
                    title={t('deleteCurrentChat')}
                    description={t('deleteCurrentChatDetail')}
                    onConfirm={() => deleteChat(chat.id)}
                    okText={c('confirm')}
                    cancelText={c('cancel')}
                  >
                    <Button size='small' className='ml-1' type='text' icon={<DeleteOutlined />} />
                  </Popconfirm>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <Modal title={t('renameChat')}
          open={isEditModalOpen}
          onOk={handleSaveTitle}
          onCancel={() => {
            setIsEditModalOpen(false);
          }}
          okText={c('confirm')}
          cancelText={c('cancel')}
        >
          <Input
            value={newChatName}
            onChange={(e) => setNewChatName(e.target.value)}
            placeholder={t('inputChatName')}
            style={{ marginTop: '1em', marginBottom: '1em' }}
          />
        </Modal>
      </div>
    </>
  )
}

export default List