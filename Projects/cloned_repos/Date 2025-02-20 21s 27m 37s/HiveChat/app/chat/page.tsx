"use client";
import React, { useEffect, useState } from 'react'
import useChatListStore from '@/app/store/chatList';
import { useRouter } from 'next/navigation';
import { Alert } from 'antd';
import useModelListStore from '@/app/store/modelList';
import ChatHeader from '@/app/components/ChatHeader';
import { MessageContent } from '@/app/adapter/interface';
import AdaptiveTextarea from '@/app/components/AdaptiveTextarea';
import { useTranslations } from 'next-intl';
import { useSession } from 'next-auth/react';
import { useLoginModal } from '@/app/contexts/loginModalContext';
import { addChatInServer } from '@/app/chat/actions/chat';
import { addMessageInServer } from '@/app/chat/actions/message';
import { ChatType } from '../db/schema';
import { localDb } from '@/app/db/localDb';

const Home = () => {
  const t = useTranslations('Chat');
  const router = useRouter();
  const { data: session, status } = useSession();
  const { visible, showLogin, hideLogin } = useLoginModal();
  const { modelList, currentModel, isPending } = useModelListStore();
  const { chatList, setChatList } = useChatListStore();
  const [greetingText, setGreetingText] = useState('');
  const [showGuideAlert, setShowGuideAlert] = useState(false);

  useEffect(() => {
    if (status === 'unauthenticated') {
      showLogin();
    }
  }, [status, showLogin]);

  useEffect(() => {
    if (!isPending && modelList.length === 0) {
      setShowGuideAlert(true);
    } else {
      setShowGuideAlert(false);
    }
  }, [isPending, modelList]);

  useEffect(() => {
    function getGreeting(): string {
      const currentHour = new Date().getHours();
      if (currentHour >= 5 && currentHour < 12) {
        return "goodMorning";
      } else if (currentHour >= 12 && currentHour < 14) {
        return "goodAfternoon";
      } else if (currentHour >= 14 && currentHour < 18) {
        return "goodEvening";
      } else {
        return "goodNight";
      }
    }
    setGreetingText(t(getGreeting()));
  }, [t]);

  const newChat = async (text: string, attachments?: Array<{ mimeType: string; data: string }>) => {
    let content: MessageContent;
    if (attachments && attachments?.length > 0) {
      const attachmentsMessages = attachments.map((attachment) => {
        return {
          "type": "image" as "image",
          "mimeType": attachment.mimeType,
          "data": attachment.data,
        };
      });
      content = [
        {
          "type": "text",
          "text": text
        },
        ...attachmentsMessages
      ]
    } else {
      content = text;
    }

    const result = await addChatInServer({ title: t('defaultChatName') });
    if (result.status === 'success') {
      const initInfo = {
        id: result.data?.id,
        title: t('defaultChatName'),
        defaultModel: 'gpt',
        createdAt: new Date(),
      };
      setChatList([initInfo as ChatType, ...chatList]);
      const toAddMessage = {
        chatId: result.data?.id!,
        content: content,
        role: 'user',
        type: 'text' as const,
        model: currentModel.id,
        providerId: currentModel.provider.id,
        createdAt: new Date(),
      };
      localDb.messages.add(toAddMessage);
      await addMessageInServer(toAddMessage);
      localStorage.setItem('f', 'home');
      router.push(`/chat/${result.data?.id}?f=home`);
    }
  };

  return (
    <>
      <ChatHeader isActionsHidden={true} />
      {showGuideAlert &&
        <div className='m-6'>
          <Alert message={t('guideAlertText')}
            type='warning'
            showIcon={true}
          />
        </div>
      }
      <div className='flex w-full grow flex-col items-center justify-center h-full'>
        <div className='container max-w-3xl mx-auto -mt-16 relative items-center justify-center'>
          <h2 className='text-3xl font-bold text-center mb-8'>{greetingText && <>👋 {greetingText}{t('welcomeNotice')}</>}&nbsp;</h2>
          <AdaptiveTextarea model={currentModel} submit={newChat} />
        </div>
      </div>
    </>
  )
}

export default Home