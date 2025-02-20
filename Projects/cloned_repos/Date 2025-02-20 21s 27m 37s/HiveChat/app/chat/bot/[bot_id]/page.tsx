'use client'
import React, { useState, useEffect } from 'react'
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { BotType } from '@/app/db/schema';
import useChatListStore from '@/app/store/chatList';
import { Button, message, Popconfirm, Divider, PopconfirmProps } from 'antd';
import { LeftOutlined } from '@ant-design/icons';
import MarkdownRender from '@/app/components/Markdown';
import { getBotInfoInServer, deleteBotInServer } from '@/app/chat/actions/bot';
import { useTranslations } from 'next-intl';

const BotInfo = ({ params }: { params: { bot_id: string } }) => {
  const t = useTranslations('Chat');
  const router = useRouter();
  const { chatList, addBot } = useChatListStore();
  const [botInfo, setBotInfo] = useState<BotType>();
  const [botChatId, setBotChatId] = useState('');
  const [isPending, setIsPending] = useState(true);
  const [isAdded, setIsAdded] = useState(false);
  useEffect(() => {
    const initBotData = async () => {
      const bot = await getBotInfoInServer(Number(params.bot_id));
      if (bot.status === 'success') {
        setBotInfo(bot.data as BotType);
        setIsPending(false);
        const botChat = chatList.filter(item => item.botId === Number(params.bot_id));
        if (botChat.length > 0) {
          setBotChatId(botChat[0].id);
          setIsAdded(true);
        } else {
          setIsAdded(false);
        }
      }
    };

    initBotData();
  }, [chatList, params.bot_id]);

  const addToChat = (botId: number) => {
    addBot(botId);
  }
  const removeCurrentBot: PopconfirmProps['onConfirm'] = async (e) => {
    if (botInfo?.id) {
      const result = await deleteBotInServer(botInfo.id);
      if (result.status === 'success') {
        message.success(t('deleteSuccess'));
        router.push(`/chat/bot/discover`);
      } else {
        message.error(t('deleteFail'));
      }
    }
  };

  return (
    <div className="container max-w-4xl mx-auto pb-8">
      <Link href='/chat/bot/discover'>
        <Button type='link' size='small' icon={<LeftOutlined />}>{t('back')}</Button>
      </Link>
      <div className='w-full flex flex-row justify-between items-center'>
        <div className="flex w-full h-28 mt-8 items-center justify-center flex-row">
          {!isPending && !botInfo && <div>{t('notExist')}</div>}
          {!isPending && botInfo && (
            <>
              {
                botInfo.avatarType === 'emoji' &&
                <div className="w-24 h-24 flex rounded-lg  bg-slate-200 text-7xl items-center justify-center overflow-hidden ">
                  {botInfo.avatar}
                </div>
              }
              {
                botInfo.avatarType === 'url' &&
                <div className="w-24 h-24 rounded-lg border border-gray-100 overflow-hidden ">
                  <Image
                    src={botInfo.avatar}
                    alt={botInfo.title}
                    width={64}
                    height={64}
                    className="w-full h-full object-cover"
                  />
                </div>
              }

              <div className="ml-6 grow w-0">
                <h3 className="text-gray-900 font-medium text-2xl mb-1 truncate">
                  {botInfo.title}
                </h3>

                <div className="text-gray-500 mb-2 text-sm line-clamp-2">
                  {botInfo.desc} {botInfo.sourceUrl && <><Divider type="vertical" />
                    <Link href={botInfo.sourceUrl} target='_blank'>
                      <Button type='text' size='small' style={{ color: '#6b7280' }}>{t('source')}</Button>
                    </Link></>}
                </div>
                {isAdded ?
                  <Link href={`/chat/${botChatId}`}><Button type='primary' shape='round'>{t('chat')}</Button></Link>
                  : <Button type='primary'
                    shape='round'
                    onClick={() => { addToChat(botInfo.id!) }}>{t('addToChats')}</Button>}
                {botInfo.creator !== 'public' &&
                  <Popconfirm
                    title="删除智能体"
                    description="确认要删除当前智能体吗？"
                    onConfirm={removeCurrentBot}
                    okText="确认"
                    cancelText="取消"
                  >
                    <Button type='text' className='ml-2' style={{ color: '#999' }} shape='round'>{t('remove')}</Button>
                  </Popconfirm>
                }

              </div>
            </>
          )}
        </div>

      </div>
      <h2 className='text-gray-800 text-lg mt-6 mb-4 font-medium'>{t('prompt')}</h2>
      <div className='text-gray-600 markdown-body border border-gray-200 p-6 rounded-lg'>
        {!isPending && botInfo?.prompt ? <MarkdownRender content={botInfo?.prompt || ''} /> :
          <span className='text-gray-400 text-sm'>{t('noPrompt')}</span>
        }
      </div>
    </div>
  )
}

export default BotInfo