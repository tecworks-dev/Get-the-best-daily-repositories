import Link from 'next/link';
import { Button } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import useSidebarCollapsedStore from '@/app/store/sidebarCollapsed';
import ToggleSidebar from "@/app/images/hideSidebar.svg";
import { useTranslations } from 'next-intl';

const InPageCollapsed = () => {
  const t = useTranslations('Chat');
  const { isSidebarCollapsed, toggleSidebar } = useSidebarCollapsedStore();
  return (
    <>{isSidebarCollapsed ?
      <div className='flex flex-row items-center'>
        <Button
          icon={<ToggleSidebar style={{ 'fontSize': '20px', 'verticalAlign': 'middle' }} />}
          type='text'
          onClick={toggleSidebar}
        />
        <Link href='/chat'>
          <div className="w-28 border ml-4 h-8 rounded-full text-center p-2 text-xs new-chat-button whitespace-nowrap">
            <PlusOutlined className='mr-2 ' style={{ color: '#0057ff' }} />{t('newChat')}
          </div>
        </Link>
      </div>
      : <div className='h-8'></div>
    }</>
  )
}

export default InPageCollapsed