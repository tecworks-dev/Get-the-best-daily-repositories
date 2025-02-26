'use client'
import {
  memo,
  useState,
} from 'react';

import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';

import { ChainsType } from '@/i18n/routing';
import {
  useEffectFiltrateStore,
  useFiltrateStore,
} from '@/store/filtrate';
import { useTokenStore } from '@/store/tokne';
import useConnectWallet from '@/store/wallet/useConnectWallet';

import FollowOrHolding from './FollowOrHolding';
import TrendingComponent from './TrendingOrXtop';

const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const TimeSelection = dynamic(() => import('@/component/TimeSelection'));
const LinkComponent = dynamic(() => import('@/component/LinkComponent'));
/**
 * Token信息组件
 * @param {Object} props 
 * @param {ChainsType} props.chain 链类型
 * @returns {JSX.Element} Token信息组件
 */
const TokenListPage = ({ chain }: { chain: ChainsType }) => {
    const t = useTranslations('Trade')
    const [followOrHolding, setFollowOrHolding] = useState('')
    const [trendingOrXtop, setTrendingOrXtop] = useState('')
    const [trendingPeriod, setTrendingPeriod] = useState(''); // 时间选择
    const [xtopPeriod, setXtopPeriod] = useState(''); // 时间选择
    const [followPeriod, setFollowPeriod] = useState(''); // 时间选择
    const { token } = useTokenStore()
    const { address } = useConnectWallet(chain)
    const [switchTop, setSwitchTop] = useState(true)
    const { setFiltrate, filtrate } = useFiltrateStore(); // 筛选条件的状态管理
    // 定义关注和持有的标题数据
    const titleData = [
        {
            title: t('follow'),
            value: '1',
        },
        // {
        //     title: t('holding'),
        //     value: '2',
        // }
    ]

    // 定义趋势和顶级的标题数据
    const titleData2 = [
        {
            title: t('trending'),
            value: '1',
        },
        // {
        //     title: t('xtop'),
        //     value: '2',
        // }
    ]
    // 初始化筛选条件，避免服务端和客户端渲染结果不一致
    useEffectFiltrateStore(({ trade }) => {
        setTrendingPeriod(trade.trendingPeriod)
        setXtopPeriod(trade.xtopPeriod)
        setFollowPeriod(trade.followPeriod)
        setTrendingOrXtop(trade.trendingOrXtop)
        setFollowOrHolding(trade.followOrHolding)
    });
    // 获取时间周期
    const getFollowPeriod = (value: string) => {
        setFollowPeriod(value)
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                followPeriod: value
            },
        });
    }
    // 获取时间周期
    const getTrendingOrXtopPeriod = (value: string) => {
        if (trendingOrXtop === '1') {
            setTrendingPeriod(value)
            setFiltrate({
                ...filtrate,
                trade: {
                    ...filtrate.trade,
                    trendingPeriod: value
                },
            });
        } else {
            setXtopPeriod(value)
            setFiltrate({
                ...filtrate,
                trade: {
                    ...filtrate.trade,
                    xtopPeriod: value
                },
            });
        }

    }

    // 改变关注或持有状态
    const changeFollowOrHolding = (value: string) => {
        setFollowOrHolding(value)
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                followOrHolding: value
            },
        });
    }

    // 改变趋势或顶级状态
    const changeTrendingOrXtop = (value: string) => {
        setTrendingOrXtop(value)
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                trendingOrXtop: value
            },
        });
    }
    return (
        <div className={` flex flex-col justify-between h-full  ${switchTop ? 'w-80 ml-2 mr-2.5' : 'w-0 ml-0 mr-10'}  relative transition-all duration-500 custom830:h-118.75`}>
            <div onClick={() => setSwitchTop(!switchTop)} className={`duration-500 absolute ${switchTop ? '' : 'translate-x-full rotate-180'}  right-0 top-2.25 cursor-pointer rounded-s-2xl w-8.25 h-7.5 dark:bg-3c bg-gray-300 flex items-center justify-center`}>
                <SvgIcon stopPropagation={false} value="arrowsBottom" className=" dark:text-d4 text-1f w-3 rotate-90" />
            </div>
            <div className="dark:bg-black1F bg-fb h-[45%]  w-full overflow-hidden rounded-10">
                <div className=" ml-3 mr-2.25">
                    <div className="text-82 pt-3.5 pb-4   font-semibold border-b dark:border-3f border-d6">
                        <div className="flex">
                            {titleData.map((item, index) =>
                                <div onClick={() => { changeFollowOrHolding(item.value) }} className={`cursor-pointer mr-2.75 last:mr-0 ${followOrHolding === item.value ? 'dark:text-white text-black' : ''}`} key={index}>{item.title}</div>
                            )}</div>
                        <TimeSelection className="mt-2" value={followPeriod} onClick={(time: string) => getFollowPeriod(time)} />
                    </div>
                </div>
                <div className="overflow-y-auto h-calcFH pt-2 scrollbar-none">
                    {
                        !(address && token) ? <LinkComponent /> :
                            <>
                                <div className={`${followOrHolding === '1' ? 'block' : 'hidden'}`}><FollowOrHolding period={followPeriod} chain={chain} funType={0} keys={~~followOrHolding} /></div>
                                <div className={`${followOrHolding === '2' ? 'block' : 'hidden'}`}><FollowOrHolding period={followPeriod} chain={chain} funType={1} keys={~~followOrHolding} /></div>
                            </>
                    }
                </div>
            </div>

            <div className="dark:bg-black1F bg-fb h-[54%] w-full overflow-hidden rounded-10">
                <div className="pt-3.5 pb-4.25 ml-3 mr-2.25  font-semibold border-b dark:border-3f border-d6">
                    <div className="text-82 flex items-center pb-2">
                        {titleData2.map((item, index) =>
                            <div onClick={() => { changeTrendingOrXtop(item.value) }} className={`cursor-pointer mr-2.75 last:mr-0 ${trendingOrXtop === item.value ? 'dark:text-white text-black' : ''}`} key={index}>{item.title}</div>
                        )}
                    </div>
                    <TimeSelection type={trendingOrXtop === '1' ? 0 : 1} value={trendingOrXtop === '1' ? trendingPeriod : xtopPeriod} onClick={(time: string) => getTrendingOrXtopPeriod(time)} />
                </div>
                <div className="h-calcFH    mt-1.75 overflow-y-auto  scrollbar-none ">
                    <div className={`${trendingOrXtop === '1' ? 'block' : 'hidden'}`}><TrendingComponent period={trendingPeriod} chain={chain} funType={0} keys={~~trendingOrXtop} /></div>
                    <div className={`${trendingOrXtop === '2' ? 'block' : 'hidden'}`}><TrendingComponent period={xtopPeriod} chain={chain} funType={1} keys={~~trendingOrXtop} /></div>
                </div>
            </div>
        </div>
    )
}
export default memo(TokenListPage)