'use client'
import Image from 'next/image';
import { Dropdown, DropdownProps } from 'antd';
import { memo, useState } from 'react';
import { ChainsType } from '@/i18n/routing';
import useSetChain from '@/hook/useSetChainLocale';
import { chainLogo } from '@/app/chainLogo';
import dynamic from 'next/dynamic';
const SvgIcon = dynamic(() => import('@/component/SvgIcon')); 

// 定义链的数组，用于在Dropdown中显示
const itemsChian = [
    { key: 'sol', label: 'Solana' }
]

/**
 * SetChain组件用于让用户选择区块链网络
 * @param {ChainsType} chain 当前选中的区块链类型
 */
const SetChain = ({ chain }: { chain: ChainsType }) => {
    const [open, setOpen] = useState(false);
    // 切换链的hooks
    const { setChainAndCookie } = useSetChain()

    /**
     * 当用户选择一个链时调用此函数
     * @param {ChainsType} key 用户选择的链的键值
     */
    const setChainFun = (key: ChainsType) => {
        setOpen(false)
        setChainAndCookie(key)
    }

    /**
     * 处理Dropdown的打开状态变化
     * @param {boolean} nextOpen 下一个打开状态
     * @param {Object} info 事件信息
     */
    const handleOpenChange: DropdownProps['onOpenChange'] = (nextOpen, info) => {
        if (info.source === 'trigger' || nextOpen) {
            setOpen(nextOpen);
        }
    };

    return (
        <Dropdown menu={{
            items: itemsChian
        }} placement='bottom' trigger={['click']}
            onOpenChange={handleOpenChange}
            open={open}
            dropdownRender={() => (
                <div className='w-36 mt-2  dark:bg-black bg-white dark:text-white  text-black px-2 py-3  rounded-xl box-border'>
                    {itemsChian.map(({ key, label }) =>
                        <div
                            onClick={() => { key !== chain && setChainFun(key as ChainsType) }}
                            key={key}
                            className={`px-2 py-2 mb-2 last:mb-0 flex items-center cursor-pointer dark:hover:bg-black1F hover:bg-d9 rounded-lg ${key === chain ? 'dark:bg-black1F bg-d9' : ''}`}>
                            <Image src={chainLogo[key]} className='w-6 mr-2' alt="" />
                            <div className='flex items-center text-base font-medium'>{label}</div>
                        </div>
                    )}
                </div>
            )}
        >
            <div className="cursor-pointer dark:bg-black bg-white dark:border-2a border-d3 border box-border pl-5.75 pr-4.25 py-2.5 h-10 rounded-2.5xl flex items-center">
                <Image src={chainLogo[chain]} className='w-6 mr-4.25' alt="" />
                <SvgIcon value="arrowsBottom" className={`w-3.1825 transition-[0.3s] text-d4 ${open ? 'rotate-180' : 'rotate-0'}`} />
            </div>
        </Dropdown >
    )
}

export default memo(SetChain)