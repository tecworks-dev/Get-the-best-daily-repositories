import { chainLogo } from '@/app/chainLogo'
import { timeData } from '@/component/TimeSelection'
import { useHistory } from '@/i18n/routing'
import { EarningsListType, FollowListParams } from '@/interface'
import { formatLargeNumber } from '@/utils'
import dynamic from 'next/dynamic'
import Image from 'next/image'
import { memo, useMemo } from 'react'
const ImageUrl = dynamic(() => import('@/component/ImageUrl'));
const TablePercentage = dynamic(() => import('@/component/TablePercentage'));
const PriceCom = dynamic(() => import('@/component/PriceCom'));
// 定义时间周期类型
type Period = '5m' | '30m' | '1h' | '6h' | '1d';
/**
 * 渲染 FollowList 或 EarningsList 中的每个项目
 * @param {Object} props - 组件属性
 * @param {FollowListParams | EarningsListType} props.item - 要渲染的项目数据
 * @returns {JSX.Element} - 项目组件
 */
const FHItem = ({ item, period }: { item: FollowListParams | EarningsListType, period: Period }) => {
    // 获取路由以支持页面跳转
    const history = useHistory()
    // 解析并处理24小时价格变化百分比
    const timeDataPeriod = useMemo(() => timeData.find((item) => item.key === period), [period, timeData]);
    // 确保 priceChangePercent 是 number 类型
    const priceChangePercent = useMemo(() => {
        const key: string = `price_change_percent${timeDataPeriod?.text}`;
        return (item as Record<string, any>)[key] ?? 0;
    }, [item, timeDataPeriod]);
    // 计算涨跌价格
    const priceChange = useMemo(() => {
        if (priceChangePercent > 0) {
            const data = item.price / (1 + priceChangePercent)
            return item.price - data
        } else if (priceChangePercent < 0) {
            const data = item.price / (1 - priceChangePercent)
            return data - item.price
        } else {
            return 0
        }
    }, [item, priceChangePercent]);
    // 返回一个可点击的项目组件，点击时跳转到交易页面
    return (
        <div onClick={() => {
            history(item.address)
        }} className='font-semibold flex items-center justify-between cursor-pointer py-2 px-3 rounded-10 mx-2 dark:hover:bg-333 hover:bg-gray-200'>
            <div className='flex items-center'>
                <ImageUrl className="w-10 h-10 rounded-full mr-1.75" logo={item.logo} symbol={item.symbol} />
                <div>
                    <div className='flex items-center'>
                        <span className='dark:text-white text-black mr-2'>{item.symbol}</span>
                        <Image src={chainLogo[item.chain]} className="w-3" alt="" />
                    </div>
                    <div className='text-82 font-medium text-13'>{formatLargeNumber(item.price)}</div>
                </div>
            </div>
            <div className={` text-right`}>
                <PriceCom isf className='text-right' num={priceChangePercent > 0 ? priceChange : -priceChange} />
                <TablePercentage number1={priceChangePercent} />
            </div>
        </div>
    )
}

export default memo(FHItem)