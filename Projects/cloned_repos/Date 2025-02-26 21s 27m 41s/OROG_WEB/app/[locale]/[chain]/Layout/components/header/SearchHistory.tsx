import { ChainsType, useHistory } from "@/i18n/routing"
import { SearchHistoryType, useSearchHistoryState } from "@/store/searchHistory"
import { useTranslations } from "next-intl"
import dynamic from "next/dynamic"
import { memo } from "react"
const ImageUrl = dynamic(() => import('@/component/ImageUrl')); 

/**
 * 搜索历史组件
 * 
 * 该组件用于显示和管理用户的搜索历史记录它接收一个包含链信息、一个可选的函数和一个可选的类名作为属性
 * 当用户点击搜索历史中的某个项时，它会导航到交易页面，并更新搜索历史记录
 * 
 * @param {ChainsType} chain - 链信息，用于识别当前链
 * @param {Function} fun - 可选的回调函数，当用户点击搜索历史中的项后会调用
 * @param {string} className - 可选的类名，用于自定义组件样式
 */
const SearchHistory = ({ chain, fun, className }: { chain: ChainsType, fun?: Function, className?: string }) => {
    // 翻译
    const t = useTranslations('Header')
    // 路由
    const history = useHistory()
    // 使用自定义钩子管理搜索历史状态
    const { searchHistory, setSearchHistoryLocal, removeSearchHistoryLocal } = useSearchHistoryState()

    /**
     * 导航到交易页面
     * 
     * 此函数接收一个搜索历史项作为参数，导航到该交易页面，并更新搜索历史记录如果提供了回调函数，则调用该函数
     * 
     * @param {SearchHistoryType} token - 搜索历史项，包含代币的地址、符号、报价地址和标志
     */
    const goTrade = (token: SearchHistoryType) => {
        history(token.address, '/trade')
        setSearchHistoryLocal(chain, { address: token.address, symbol: token.symbol, quote_mint_address: token.quote_mint_address, logo: token.logo })
        fun && fun()
    }
    return (
        <div className={`dark:text-white first-letter:text-base font-semibold px-5 pb-5 ${className}`}>
            <div className='flex items-center justify-between mb-2.5'>
                <span className='text-b1'>{t('recent')}</span>
                <span onClick={() => { removeSearchHistoryLocal(chain) }} className='text-4b cursor-pointer'>{t('clear')}</span>
            </div>
            <div className='flex flex-wrap'>
                {searchHistory[chain].map((item, index) =>
                    <div onClick={() => goTrade(item)} key={index} className='mr-5 mb-3.75 cursor-pointer flex items-center bg-f1 dark:bg-26 py-0.25 pl-0.5 pr-3 rounded-30 '>
                        <ImageUrl logo={item.logo} symbol={item.symbol} className='w-9.5 h-9.5 rounded-full mr-2.75' />
                        <span className='text-83 text-xl'>{item.symbol}</span>
                    </div>
                )}
            </div>
        </div>
    )
}

export default memo(SearchHistory)