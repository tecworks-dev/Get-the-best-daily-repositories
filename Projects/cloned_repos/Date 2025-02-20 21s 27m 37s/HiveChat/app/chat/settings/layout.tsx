"use client"
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { Menu, MenuProps } from 'antd';
import { usePathname, useRouter } from 'next/navigation';
import { useSession } from 'next-auth/react';
import InPageCollapsed from '@/app/components/InPageCollapsed';
import { useTranslations } from 'next-intl';

type MenuItem = Required<MenuProps>['items'][number];


const Settings = ({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) => {
  const t = useTranslations('Settings');
  const [current, setCurrent] = useState('');
  const pathname = usePathname();
  const router = useRouter();
  const { data: session, status } = useSession();

  const items: MenuItem[] = [
    {
      label: <Link href="/chat/settings/account">{t('accountSettings')}</Link>,
      key: 'account',
    },
    {
      label: <Link href="/chat/settings/system">{t('systemSettings')}</Link>,
      key: 'system',
    },
  ];

  useEffect(() => {
    if (pathname === '/chat/settings/account') {
      setCurrent('account');
    } else if (pathname === '/chat/settings/system') {
      setCurrent('system');
    }
  }, [pathname]);
  const onClick: MenuProps['onClick'] = (e) => {
    setCurrent(e.key);
  };

  if (status === 'loading') {
    return null;
  }

  if (!session) {
    router.push('/login'); // 如果用户未登录，跳转到登录页
    return null;
  }
  return (
    <>
      <div className='w-full h-22 p-4'>
        <InPageCollapsed />
      </div>
      <div className='container max-w-3xl mb-6 px-4 mx-auto md:px-0'>
        <Menu onClick={onClick} selectedKeys={[current]} mode="horizontal" items={items} />
        {children}
      </div>
    </>
  )
}

export default Settings