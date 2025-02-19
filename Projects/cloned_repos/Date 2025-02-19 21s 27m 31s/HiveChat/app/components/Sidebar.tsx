"use client";
import React, { useState } from 'react';
import Image from "next/image";
import Link from 'next/link';
import { SettingOutlined, ControlOutlined, UserOutlined } from '@ant-design/icons';
import clsx from 'clsx';
import { usePathname } from 'next/navigation';
import logo from "@/app/images/logo.png";
import HivechatText from "@/app/images/hivechat.svg";
import ToggleSidebar from "@/app/images/hideSidebar.svg";
import { Button } from 'antd';
import ChatList from '@/app/components/ChatList';
import useSidebarCollapsedStore from '@/app/store/sidebarCollapsed';
import { useTranslations } from 'next-intl';
import { useSession } from 'next-auth/react';

export const Sidebar = () => {
  const t = useTranslations('Chat');
  const pathname = usePathname();
  const { data: session, status } = useSession();
  const { isSidebarCollapsed, toggleSidebar } = useSidebarCollapsedStore();
  return (
    <div
      className={clsx(
        "flex flex-col w-72 bg-gray-100 h-full p-4 pr-0 box-border transition-transform duration-300 ease-in-out",
        isSidebarCollapsed ? "-translate-x-full" : "" // 根据状态动态设置宽度
      )}
    >
      <div className="flex items-center flex-row flex-grow-0 mb-2 h-10 mr-4 justify-between">
        <Link href="/" className='flex items-center'>
          <Image src={logo} className="ml-1" alt="HiveChat logo" width={24} height={24} />
          <HivechatText width={104} height={26} />
        </Link>
        <Button
          icon={<ToggleSidebar style={{ 'color': '#999', 'fontSize': '20px', 'verticalAlign': 'middle' }} />}
          type='text'
          onClick={toggleSidebar}
        />
      </div>
      <ChatList />
      {status !== 'loading' &&
        <div className='mt-auto'>
          {session ? <div className="flex items-center flex-grow-0 h-10 mr-4 border-t border-gray-200">
            <Link className='w-full text-sm' href={"/chat/settings/account"}>
              <div className={clsx('flex items-center pl-3 mt-2 hover:bg-gray-200 h-9 w-full rounded', pathname.startsWith('/chat/settings') ? 'bg-gray-200' : '')}>
                <SettingOutlined /><span className='ml-2 whitespace-nowrap'>{t('settings')}</span>
              </div>
            </Link>
          </div>
            :
            <div className="flex items-center flex-grow-0 h-10 mr-4 border-t border-gray-200">
              <Link className='w-full text-sm' href={"/login"}>
                <div className={clsx('flex items-center pl-3 mt-2 hover:bg-gray-200 h-9 w-full rounded', pathname.startsWith('/chat/settings') ? 'bg-gray-200' : '')}>
                  <><UserOutlined /><span className='ml-2 whitespace-nowrap'>{t('login')}</span></>
                </div>
              </Link>
            </div>}

          {session?.user.isAdmin && <div className="flex items-center flex-grow-0 h-10 mr-4">
            <Link className='w-full text-sm' href={"/admin/llm"}>
              <div className='flex items-center pl-3 mt-2 hover:bg-gray-200 h-9 w-full rounded'>
                <ControlOutlined />
                <span className='ml-2 whitespace-nowrap'>{t('adminPanel')}</span>
              </div>
            </Link>
          </div>}
        </div>}
    </div>
  )
}

export default Sidebar