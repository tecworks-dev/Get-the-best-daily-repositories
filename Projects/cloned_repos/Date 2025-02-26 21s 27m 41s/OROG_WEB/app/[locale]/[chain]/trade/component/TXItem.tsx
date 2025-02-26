// 定义一个客户端组件
'use client'
// 导入所需的模块和组件
import { chainLogo } from '@/app/chainLogo'
import { timeData } from '@/component/TimeSelection'
import { useHistory } from '@/i18n/routing'
import { ListParams } from '@/interface'
import { formatLargeNumber } from '@/utils'
import dynamic from 'next/dynamic'
import Image from 'next/image'
import { memo, useMemo } from 'react'
const TablePercentage = dynamic(() => import('@/component/TablePercentage'));
const ImageUrl = dynamic(() => import('@/component/ImageUrl'));
// 定义时间周期类型
type Period = '5m' | '30m' | '1h' | '6h' | '1d';

// 定义 TXItem 组件的属性接口
interface TXItemProps {
    item: ListParams;
    period: Period;
}

/**
 * TXItem 组件用于显示交易项的详细信息
 * @param {TXItemProps} props - 组件属性，包括交易项数据和选定的时间周期
 * @returns {JSX.Element} - 交易项的详细信息视图
 */
const TXItem = ({ item, period }: TXItemProps) => {
    const timeDataPeriod = useMemo(() => timeData.find((item) => item.key === period), [period, timeData]);
    // 确保 priceChangePercent 是 number 类型
    const priceChangePercent = useMemo(() => {
        const key: string = `price_change_percent${timeDataPeriod?.text}`;
        return (item as Record<string, any>)[key] ?? 0;
    }, [period, item]);
    const history = useHistory()
    // 渲染交易项的详细信息
    return (
        <div onClick={() => {
            history(item.address)
        }} className='font-semibold flex items-center justify-between cursor-pointer py-2 px-3 rounded-10 mx-2 dark:hover:bg-333 hover:bg-gray-200'>
            <div className='flex items-center'>
                <ImageUrl logo={item.logo} symbol={item.symbol} className='w-10 h-10 rounded-full mr-1.75' />
                <div>
                    <div className='flex items-center'>
                        <span className='dark:text-white text-black mr-2'>{item.symbol}</span>
                        <Image src={chainLogo[item.chain]} className="w-3" alt="" />
                    </div>
                    <div className='text-82 font-medium text-13 flex items-center'>
                        {formatLargeNumber(item.swaps)}({formatLargeNumber(item.buys)}/{formatLargeNumber(item.sells)})
                    </div>
                </div>
            </div>
            <div className='text-right'>
                <div className='text-15 dark:text-white text-black'>${formatLargeNumber(item.price)}</div>
                <TablePercentage number1={priceChangePercent} />
            </div>
        </div>
    )
}

// 导出 TXItem 组件
export default memo(TXItem)