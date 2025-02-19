'use client';
import React, { useState, useEffect } from 'react';
import { Sidebar } from "@/app/components/Sidebar";
import { LoginModalProvider } from '@/app/contexts/loginModalContext';
import LoginModal from '@/app/components/loginModal';
import useSidebarCollapsedStore from '@/app/store/sidebarCollapsed';
import SpinLoading from '@/app/components/loading/SpinLoading';
import clsx from 'clsx';

const App: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [hasInstalled, setHasInstalled] = useState(false);
  const { isSidebarCollapsed, setIsSidebarCollapsed } = useSidebarCollapsedStore();

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) { // 768 是常见的移动端断点宽度
        setIsSidebarCollapsed(true);
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [setIsSidebarCollapsed]);

  useEffect(() => {
    async function initData() {
      localStorage.setItem('hive_chat_app_status', 'installed');
      localStorage.setItem('hive_chat_app_version', '1.0.0');
    }
    const status = localStorage.getItem('hive_chat_app_status');
    if (status !== 'installed') {
      initData().then(() => {
        setHasInstalled(true);
      });
    } else {
      setHasInstalled(true);
    }
  }, []);

  if (!hasInstalled) {
    return (
      <main className="h-full flex justify-center items-center">
        <SpinLoading />
        <span className='ml-2 text-gray-600'>Loading ...</span>
      </main>
    )
  }
  return (
    <LoginModalProvider>
      <div className="flex h-dvh w-screen overflow-hidden">
        <Sidebar />
        <div
          className={clsx(
            "h-full w-0 grow flex flex-col transition-all duration-300 ease-in-out overflow-y-auto",
            { "w-full -ml-72": isSidebarCollapsed }
          )}
        >
          {children}
        </div>
      </div>
      <LoginModal />
    </LoginModalProvider>
  )
}

export default App