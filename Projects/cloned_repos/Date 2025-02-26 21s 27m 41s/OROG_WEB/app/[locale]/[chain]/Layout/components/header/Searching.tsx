'use client'
import { Dropdown, Input } from 'antd';
import { memo, useState } from 'react';
import { LoadingOutlined } from '@ant-design/icons';
import { ChainsType, useHistory } from '@/i18n/routing';
import { SearchListType, SearchTokenType } from '@/interface';
import { SearchHistoryType, useSearchHistoryState } from '@/store/searchHistory';
import dynamic from 'next/dynamic';
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const SearchList = dynamic(() => import('./SearchList'));
const SearchHistory = dynamic(() => import('./SearchHistory'));
const SearchDef = dynamic(() => import('./SearchDef'));

// Searching组件，用于在界面上实现搜索功能
// 参数:
// - chain: ChainsType 类型的对象，表示当前的链信息
function Searching({ chain }: { chain: ChainsType }) {
    // 路由
    const history = useHistory()
    // 本地的历史纪录
    const { searchHistory, setSearchHistoryLocal } = useSearchHistoryState()
    // 搜索框的值
    const [search, setSearch] = useState('')
    // loading
    const [loading, setLoading] = useState(false)
    // 是否展开Dropdown
    const [open, setOpen] = useState(false)
    // 搜索的数据
    const [searchData, setSearchData] = useState<SearchTokenType[]>([])

    // 获取输入框的值
    const onChange = async ({ target }: { target: { value: string } }) => {
        // 设置搜索框的值
        setSearch(target.value || '')
    }

    // 用户点击回车
    const onPressEnter = async () => {
        if (!search) return
        if (!loading && searchData.length) {
            const tokenData = searchData[0]
            goTrade(tokenData)
        }
    }

    // 去trade页面
    const goTrade = (token: SearchHistoryType) => {
        history(token.address, '/trade')
        setSearchHistoryLocal(chain, { address: token.address, symbol: token.symbol, quote_mint_address: token.quote_mint_address, logo: token.logo })
        setOpen(false)
    }

    // 聚焦时将 `open` 设置为 true
    const handleFocus = () => {
        setOpen(true);
    };

    // 失焦时将 `open` 设置为 false Dropdown
    const handleBlur = (event: React.FocusEvent<HTMLDivElement>) => {
        const nextFocus = event.relatedTarget
        if (nextFocus?.id !== 'FCY') {
            setOpen(false);
        }
    };
    return (
        <Dropdown
            menu={{ items: [] }} placement="bottom" open={open} trigger={['click']}
            dropdownRender={() => {
                return <div
                    tabIndex={0}
                    id='FCY'
                    onFocus={handleFocus}
                    onBlur={handleBlur}
                    className='dark:bg-black  bg-white w-125 h-118.75 rounded-2.5xl shadow	mt-3 box-border py-6.75 '>
                    <div className='overflow-y-auto h-full scrollbar-none'>
                        <SearchHistory className={`${!!searchHistory[chain].length ? 'block' : 'hidden'}`} chain={chain} fun={() => { setOpen(false) }} />
                        {/* <SearchDef className={`${!loading && !search ? 'block' : 'hidden'}`} setLoading={setLoading} /> */}
                        {/* <SearchList className={`${!loading && search ? 'block' : 'hidden'}`} getSearchData={(data: SearchTokenType[]) => setSearchData(data)} search={search} chain={chain} setLoading={setLoading} fun={() => setOpen(false)} /> */}
                        <SearchList className={`${!loading ? 'block' : 'hidden'}`} getSearchData={(data: SearchTokenType[]) => setSearchData(data)} search={search} chain={chain} setLoading={setLoading} fun={() => setOpen(false)} />
                        <div className={`flex justify-center items-center h-full text-3xl ${loading ? 'block' : 'hidden'}`}>
                            <LoadingOutlined className='dark:text-white' />
                        </div>
                    </div>
                </div>
            }}
        >
            <Input value={search}
                tabIndex={0}
                id='FCY'
                onFocus={handleFocus}
                onBlur={handleBlur}
                onPressEnter={onPressEnter} onChange={onChange}
                prefix={
                    <SvgIcon className="w-5 dark:text-white text-c5" value="searchIcon" />
                }
                className='box-border w-105 h-11.25 rounded-5 shadow-none 
                 dark:bg-black dark:hover:bg-black dark:focus-within:bg-black dark:border-black dark:hover:border-white dark:focus-within:border-white dark:text-white
                 bg-white hover:bg-white focus-within:bg-white border-white hover:border-black focus-within:border-black text-black
                 custom830:hidden
                 '
                placeholder="" />

        </Dropdown>
    )
}
export default memo(Searching)