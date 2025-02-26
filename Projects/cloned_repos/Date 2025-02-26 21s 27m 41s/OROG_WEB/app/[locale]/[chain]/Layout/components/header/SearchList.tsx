import { chainLogo } from '@/app/chainLogo'
import { ChainsType, useHistory } from '@/i18n/routing'
import { SearchListType, SearchTokenType } from '@/interface'
import { SearchHistoryType, useSearchHistoryState } from '@/store/searchHistory'
import { mobileHidden, formatLargeNumber } from '@/utils'
import React, { memo, useCallback, useEffect, useState } from 'react'
import Image from 'next/image'
import { useTranslations } from 'next-intl'
import { searchHttp } from '@/request'
import dynamic from 'next/dynamic'
const ImageUrl = dynamic(() => import('@/component/ImageUrl'));
const Follow = dynamic(() => import('@/component/Follow'));

// 搜索列表组件
// 该组件用于显示搜索结果列表，允许用户点击结果跳转到交易页面
const SearchList = ({ chain, fun, setLoading, search, getSearchData, className }: { chain: ChainsType, fun?: Function, setLoading: Function, search: string, getSearchData: Function, className?: string }) => {
    // 路由
    // 用于导航到不同页面
    const history = useHistory()
    // 翻译
    // 用于国际化，获取 Header 组件相关的翻译文本
    const t = useTranslations('Header')
    // 搜索的数据
    // 存储搜索结果的数据状态
    const [searchData, setSearchData] = useState<SearchTokenType[]>([])
    // 搜索历史记录相关操作
    const { setSearchHistoryLocal } = useSearchHistoryState()
    // 去trade页面
    // 当用户点击搜索结果时，导航到交易页面并保存搜索历史记录
    const goTrade = (token: SearchHistoryType) => {
        history(token.address, '/trade')
        setSearchHistoryLocal(chain, { address: token.address, symbol: token.symbol, quote_mint_address: token.quote_mint_address, logo: token.logo })
        fun && fun()
    }
    // 获取搜索数据
    // 根据搜索关键词获取搜索结果，并更新组件状态
    const getSearch = useCallback(async () => {
        // if (!search) return
        setLoading(true)
        try {
            const { data } = await searchHttp<SearchListType>(chain, { q: search })
            setSearchData(data.list)
            getSearchData(data.list)
        } catch (e) {
            console.log('getSearch', e);
        }
        setLoading(false)
    }, [search])
    // 使用效果钩子来自动触发搜索
    useEffect(() => {
        getSearch()
    }, [getSearch])
    return (
        <div className={`${className}`}>
            <p className={`dark:text-d9 text-black text-xm pl-3 pb-3`}>{t('token')}</p>
            {searchData.map((item) => (
                <div key={item.address} onClick={() => { goTrade(item) }} className='dark:hover:bg-black1F hover:bg-d9   cursor-pointer flex items-center justify-between dark:text-white text-51 mx-3 px-3 py-3 rounded-2xl '>
                    <div className='flex items-center  '>
                        <Follow className='w-3.5 mr-2' chain={chain} follow={item.follow} quote_mint_address={item.quote_mint_address} />
                        <ImageUrl className='w-11.25 h-11.25 rounded-full mr-3.75' logo={item.logo} symbol={item.symbol} />
                        <div>
                            <div className='flex items-center'>
                                <span className='text-xl font-semibold pr-1'>{item.symbol}</span>
                                <Image src={chainLogo[item.chain]} className='w-3 ml-2' alt='' />
                            </div>
                            <p className='text-13 text-6b font-semibold'>{mobileHidden(item.quote_mint_address, 4, 4)}</p>
                        </div>
                    </div>
                    <div className='font-semibold text-right'>
                        <div className='text-xl '>{t('liq')} {formatLargeNumber(item.liquidity)}</div>
                        <div className='text-13 text-82 text-right'>{t('vol')}{formatLargeNumber(item.volume_24h)} 24{t('h')}</div>
                    </div>
                </div>
            ))}
        </div>
    )
}

export default memo(SearchList)