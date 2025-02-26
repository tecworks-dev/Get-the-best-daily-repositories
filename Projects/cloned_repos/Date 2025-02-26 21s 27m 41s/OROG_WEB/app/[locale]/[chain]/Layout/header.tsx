// 导入必要的组件和模块
'use client'
import Image from 'next/image';
import logo from '@/public/logo.png'
import { useTranslations } from 'next-intl';
import { Link, usePathname, useRouter } from '@/i18n/routing';

import OROGW from '@/public/OROGW.png'
import OROGD from '@/public/OROGD.png'
import { ChainsType } from '@/i18n/routing';
import { useThemeStore } from '@/store/theme';
import { memo, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const Searching = dynamic(() => import('./components/header/Searching'));
const SetChain = dynamic(() => import('./components/header/SetChain'));
const Connect = dynamic(() => import('./components/header/Connect'));
const MobileSearch = dynamic(() => import('./components/header/MobileSearch'));



/**
 * Header组件，用于显示网站的头部信息
 * @param {ChainsType} chain 当前的区块链类型
 * @param {'dark' | 'light'} themeSer 当前的主题模式
 */
const Header = ({ chain, themeSer }: { chain: ChainsType, themeSer: 'dark' | 'light' }) => {
    // 初始化翻译钩子
    const t = useTranslations('Header');
    // 获取当前的主题模式
    const { theme } = useThemeStore()
    // 获取路由实例
    const router = useRouter()
    // 控制主题模式的显示
    const [themeMode, setThemeMode] = useState<'dark' | 'light'>(themeSer);
    // 获取当前路径名
    const pathName = usePathname()
    // 定义路由数据
    const routerData = [
        { title: t('meme'), path: '/' },
        { title: t('trending'), path: '/trending' },
        // { title: t('Xtop'), path: '/xtop' },
        { title: t('follow'), path: '/follow' },
    ]
    // 更新主题模式
    useEffect(() => {
        setThemeMode(theme)
    }, [theme])
    return (
        <>
            <div className="h-17.5 flex items-center justify-between px-7 dark:bg-black1F bg-de custom830:px-2">
                <div className='flex items-center '>
                    <div className='flex items-center  mr-7.75 cursor-pointer' onClick={() => { router.push('/') }}>
                        <Image src={logo} alt="logo" className='w-13 mr-3.25   ' />
                        <Image priority src={themeMode === 'dark' ? OROGW : OROGD} alt='' className='w-29.25' />
                    </div>
                    <div className='flex items-center font-normal  text-xl dark:text-82 text-6d custom830:hidden'>
                        {routerData.map((item, index) => (
                            <Link passHref href={item.path} key={index} className={`mr-5 dark:hover:text-white hover:text-28 last:mr-0 ${pathName === item.path && 'dark:text-white text-28'}`}> {item.title}</Link>
                        ))}
                    </div>
                </div>
                <Searching chain={chain} />
                <div className='flex items-center'>
                    <MobileSearch chain={chain} />
                    <SetChain chain={chain} />
                    <Connect chain={chain} />
                </div>
            </div>
            <div className='px-4 py-4 items-center font-normal  text-xl dark:bg-82 bg-white  dark:text-black text-82 hidden custom830:flex'>
                {routerData.map((item, index) => (
                    <Link href={item.path} passHref key={index} className={`mr-5 dark:hover:text-white hover:text-28 last:mr-0 ${pathName === item.path && 'dark:text-white text-28'}`}> {item.title}</Link>
                ))}
            </div>
        </>
    );
}
export default memo(Header);