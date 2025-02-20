'use client'
import React from 'react'
import { Button, message, Modal } from "antd";
import useChatListStore from '@/app/store/chatList';
import { deleteAllUserChatInServer } from '@/app/chat/actions/chat';
import { useTranslations } from 'next-intl';

const System = () => {
  const c = useTranslations('Common');
  const t = useTranslations('Settings');
  const [modal, contextHolderModal] = Modal.useModal();
  const { setChatList } = useChatListStore();

  const clearAllChats = () => {
    modal.confirm({
      title: t('deleteAllMessagesTitle'),
      content: t('deleteAllMessagesDesc'),
      okText: c('confirm'),
      cancelText: c('cancel'),
      onOk() {
        deleteAllUserChatInServer();
        setChatList([]);
        message.success(t('deleteSuccess'));
      }
    });
  }

  return (
    <div>
      {contextHolderModal}
      <div className='flex flex-row justify-between mt-6 p-6 border border-gray-200 rounded-md'>
        <div className='flex flex-col '>
          <span className='text-sm'>{t('deleteAllMessagesTitle')}</span>
          <span className='text-gray-400 text-xs'>{t('deleteAllMessagesDesc')}</span>
        </div>
        <div className='flex items-center'>
          <Button onClick={clearAllChats}>{t('deleteAction')}</Button>
        </div>
      </div>

    </div>
  )
}

export default System