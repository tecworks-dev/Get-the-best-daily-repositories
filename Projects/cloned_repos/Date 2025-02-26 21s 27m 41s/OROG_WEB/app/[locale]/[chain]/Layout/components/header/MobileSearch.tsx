'use client'
import { ChainsType, useHistory } from '@/i18n/routing'
import { SearchHistoryType, useSearchHistoryState } from '@/store/searchHistory'
import { LoadingOutlined } from '@ant-design/icons'
import { Drawer, Input } from 'antd'
import { useTranslations } from 'next-intl'
import { memo, useState } from 'react'
import { SearchListType, SearchTokenType } from '@/interface'
import dynamic from 'next/dynamic'
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const SearchList = dynamic(() => import('./SearchList'));
const SearchHistory = dynamic(() => import('./SearchHistory'));
const SearchDef = dynamic(() => import('./SearchDef'));/**
 * MobileSearch组件用于在移动设备上实现搜索功能
 * @param {Object} props - 组件的props
 * @param {ChainsType} props.chain - 当前的链信息
 */
const MobileSearch = ({ chain }: { chain: ChainsType }) => {
    // Drawer控制开关
    const [open, setOpen] = useState(false);
    // 路由
    const history = useHistory()
    // 搜索框的值
    const [search, setSearch] = useState('')
    // 翻译
    const t = useTranslations('Header')
    // 本地的历史纪录
    const { searchHistory, setSearchHistoryLocal } = useSearchHistoryState()
    // loading
    const [loading, setLoading] = useState(false)
    // 搜索的数据
    const [searchData, setSearchData] = useState<SearchTokenType[]>([])

    /**
     * 获取输入框的值
     * @param {Object} event - 输入事件对象
     * @param {Object} event.target - 输入框对象
     * @param {string} event.target.value - 输入框的值
     */
    const onChange = async ({ target }: { target: { value: string } }) => {
        // 设置搜索框的值
        setSearch(target.value || '')
    }

    /**
     * 去trade页面
     * @param {SearchHistoryType} token - 选中的代币信息
     */
    const goTrade = (token: SearchHistoryType) => {
        // 跳转到交易页面
        history(token.address, '/trade')
        // 更新本地搜索历史
        setSearchHistoryLocal(chain, { address: token.address, symbol: token.symbol, quote_mint_address: token.quote_mint_address, logo: token.logo })
        // 关闭Drawer
        setOpen(false)
    }

    /**
     * 用户点击回车
     */
    const onPressEnter = async () => {
        // 如果搜索框为空，则不执行任何操作
        if (!search) return
        // 如果没有加载中且有搜索数据，则跳转到第一个搜索结果
        if (!loading && searchData.length) {
            const tokenData = searchData[0]
            goTrade(tokenData)
        }
    }

    // 开启Drawer
    const showDrawer = () => {
        setOpen(true);
    };

    // 关闭Drawer
    const onClose = () => {
        setOpen(false);
        // 清空搜索框的值
        setSearch('')
    };
    return (
        <div className='hidden custom830:block mr-2 font-semibold'>
            <SvgIcon className="w-5 dark:text-white text-c5" onClick={showDrawer} value="searchIcon" />
            <Drawer className='dark:bg-black bg-white' closable={false} placement='top' width="100%" height="100%" onClose={onClose} open={open}>
                <div className='flex justify-between items-center mb-4 text-2xl'>
                    <div>{t('search')}</div>
                    <SvgIcon stopPropagation={false} className="w-6 dark:text-white text-black" value="errorCopy" onClick={onClose} />
                </div>
                <Input value={search}
                    onPressEnter={onPressEnter} onChange={onChange}
                    prefix={
                        <SvgIcon className="w-5 text-white" value="searchIcon" />
                    }
                    className='box-border w-full h-16 rounded-5 text-lg shadow-none
                 dark:bg-black dark:hover:bg-black dark:focus-within:bg-black  dark:hover:border-white dark:focus-within:border-white dark:text-white
                 bg-white hover:bg-white focus-within:bg-white  hover:border-black focus-within:border-black text-black
                 '
                    placeholder="" />
                <div
                    className='dark:bg-black  bg-white  rounded-2.5xl dark:shadow-slate-800 shadow	mt-3 box-border py-6.75 '>
                    <div className='overflow-y-auto h-full scrollbar-none'>
                        <SearchHistory className={`${!!searchHistory[chain].length ? 'block' : 'hidden'}`} chain={chain} fun={() => { setOpen(false) }} />
                        {/* <SearchDef className={`${!loading && !search ? 'block' : 'hidden'}`} setLoading={setLoading} /> */}
                        {/* <SearchList className={`${!loading && search ? 'block' : 'hidden'}`} getSearchData={(data: SearchTokenType[]) => setSearchData(data)} search={search} chain={chain} setLoading={setLoading} fun={() => setOpen(false)} /> */}
                        <SearchList className={`${!loading ? 'block' : 'hidden'}`} getSearchData={(data: SearchTokenType[]) => setSearchData(data)} search={search} chain={chain} setLoading={setLoading} fun={() => setOpen(false)} />
                        <div className={`flex justify-center items-center h-full text-3xl ${loading ? 'block' : 'hidden'}`}>
                            <LoadingOutlined />
                        </div>
                    </div>

                </div>
            </Drawer>
        </div>
    )
}

export default memo(MobileSearch)