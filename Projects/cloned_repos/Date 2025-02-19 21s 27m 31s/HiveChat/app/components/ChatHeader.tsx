import React from 'react'
import { useRouter } from 'next/navigation';
import ModelSelect from '@/app/components/ModelSelect';
import { Button, Popconfirm, PopconfirmProps, Tooltip } from "antd";
import { DeleteOutlined } from '@ant-design/icons';
import useSidebarCollapsedStore from '@/app/store/sidebarCollapsed';
import ToggleSidebar from "@/app/images/hideSidebar.svg";
import useChatStore from '@/app/store/chat';
import useChatListStore from '@/app/store/chatList';
import { deleteChatInServer } from '@/app/chat/actions/chat';
import { useTranslations } from 'next-intl';

const ChatHeader = (props: { isActionsHidden?: boolean }) => {
  const t = useTranslations('Chat');
  const router = useRouter();
  const { chatList, setChatList } = useChatListStore();
  const { chat } = useChatStore();
  const { isSidebarCollapsed, toggleSidebar } = useSidebarCollapsedStore();

  const deleteChat = async () => {
    if (chat && chat.id) {
      await deleteChatInServer(chat.id)
      const updatedChatList = chatList.filter(i => i.id !== chat.id);
      setChatList(updatedChatList);

      // 跳转到更新后的第一个聊天（如果存在）
      if (updatedChatList.length > 0) {
        router.push(`/chat/${updatedChatList[0].id}`);
      } else {
        router.push('/'); // 如果没有聊天记录，跳回根路径
      }
    }
  };


  const confirmDelete: PopconfirmProps['onConfirm'] = (e) => {
    deleteChat()
  };

  const cancelDelete: PopconfirmProps['onCancel'] = (e) => { };

  return (
    <div className="h-10 flex w-full bg-gray-50 shadow-sm grow-0  items-center p-2 justify-between">
      <div className='flex items-center'>
        {isSidebarCollapsed &&
          <Button
            icon={<ToggleSidebar style={{ 'fontSize': '20px', 'verticalAlign': 'middle' }} />}
            type='text'
            onClick={toggleSidebar}
          />
        }
        <ModelSelect />
      </div>
      {!props.isActionsHidden && <div className='mr-2'>
        {/* <Button type='text'
          icon={chat?.is_star ? <StarFilled style={{ color: '#f7b83c' }} /> : <StarOutlined style={{ color: 'gray' }}
          />} onClick={toggleStar} /> */}

        <Popconfirm
          title={t('deleteCurrentChat')}
          description={t('clearHistoryMessageNotice')}
          onConfirm={confirmDelete}
          onCancel={cancelDelete}
          okText={t('confirm')}
          cancelText={t('cancel')}
        >
          <Button type='text'
            icon={<DeleteOutlined style={{ color: 'gray' }} />} />
        </Popconfirm>

        {/* <Popover
          content={<AdvancedSettingsPopover />}
          title={<div className='ml-4 mt-2'>高级设置</div>}
          trigger="click"
          placement="bottomRight"
        >
          <Button type='text' icon={<SettingConfig theme="filled" size="15" fill="#808080" strokeWidth={4} />} />
        </Popover> */}
        {/* <Button type='text' icon={<More theme="filled" size="15" fill="#808080" strokeWidth={4} />} /> */}
      </div>
      }

    </div>
  )
}

export default ChatHeader
