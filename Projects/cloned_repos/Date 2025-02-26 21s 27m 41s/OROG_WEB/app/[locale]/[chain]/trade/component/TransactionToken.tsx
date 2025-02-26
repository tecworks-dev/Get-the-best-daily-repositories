'use client'
import {
  memo,
  ReactNode,
  useState,
} from 'react';

import { Modal } from 'antd';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import Image from 'next/image';

import { chainLogo } from '@/app/chainLogo';
import { ChainsType } from '@/i18n/routing';
import { TokenInfoType } from '@/interface';
import {
  formatDate,
  formatDuration,
  formatLargeNumber,
  mobileHidden,
  multiplyAndTruncate,
} from '@/utils';

const PriceCom = dynamic(() => import('@/component/PriceCom'));
const TablePercentage = dynamic(() => import('@/component/TablePercentage'));
const Follow = dynamic(() => import('@/component/Follow'));
const Copy = dynamic(() => import('@/component/Copy'));
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const ImageUrl = dynamic(() => import('@/component/ImageUrl'));
const defaultToken: TokenInfoType = {
    address: "",
    biggest_pool_address: "",
    circulating_supply: 0,
    creation_timestamp: 0,
    decimals: 0,
    base_price: 0,
    dev: {
        address: "",
        creator_address: "",
        creator_token_balance: 0,
        creator_token_status: false,
        telegram: "",
        twitter_username: "",
        website: "",
    },
    follow: false,
    holder_count: 0,
    liquidity: 0,
    logo: "",
    max_supply: 0,
    name: "",
    open_timestamp: 0,
    pool: {
        address: "",
        base_address: "",
        base_reserve: 0,
        base_reserve_value: 0,
        base_vault_address: "",
        creation_timestamp: 0,
        creator: "",
        exchange: "",
        initial_base_reserve: 0,
        initial_liquidity: 0,
        initial_quote_reserve: 0,
        liquidity: 0,
        quote_address: "",
        quote_mint_address: "",
        quote_reserve: 0,
        quote_reserve_value: 0,
        quote_symbol: "",
        quote_vault_address: "",
        token0_address: "",
        token1_address: "",
    },
    price: {
        address: "",
        buy_volume_1h: 0,
        buy_volume_1m: 0,
        buy_volume_5m: 0,
        buy_volume_6h: 0,
        buy_volume_24h: 0,
        buys: 0,
        buys_1h: 0,
        buys_1m: 0,
        buys_5m: 0,
        buys_6h: 0,
        buys_24h: 0,
        market_cap: 0,
        price: 0,
        price_1h: 0,
        price_1m: 0,
        price_5m: 0,
        price_6h: 0,
        price_24h: 0,
        sell_volume_1h: 0,
        sell_volume_1m: 0,
        sell_volume_5m: 0,
        sell_volume_6h: 0,
        sell_volume_24h: 0,
        sells: 0,
        sells_1h: 0,
        sells_1m: 0,
        sells_5m: 0,
        sells_6h: 0,
        sells_24h: 0,
        swaps: 0,
        volume: 0,
        volume_1h: 0,
        volume_1m: 0,
        volume_5m: 0,
        volume_6h: 0,
        volume_24h: 0,
    },
    symbol: "",
    total_supply: 0,
};
/**
 * Transaction组件用于显示交易相关信息，包括代币信息、价格动态、流动性等
 * @param {Object} props 组件属性
 * @param {ChainsType} props.chain 链类型，用于确定交易所属的区块链
 * @param {string} props.address 交易地址，用于获取交易信息
 */
const Transaction = ({ chain, info = defaultToken }: { chain: ChainsType, info: TokenInfoType | undefined }) => {
    // 使用useTranslations钩子获取翻译函数，用于界面文本的多语言支持
    const t = useTranslations('Trade')
    // 定义模态框状态，控制模态框的显示与隐藏
    const [isModalOpen, setIsModalOpen] = useState(false);
    // 定义开关状态，用于控制额外信息的展开与收起
    const [switchTop, setSwitchTop] = useState(false)


    return (
        <div className=" font-semibold custom830:overflow-hidden ">
            <div className="rounded-11 dark:bg-26 bg-fb relative box-border pt-4.5 transition-all ">
                <div onClick={() => setSwitchTop(!switchTop)} className={`rounded-t-10 duration-500 absolute bottom-0 left-1/2 -translate-x-1/2 cursor-pointer w-13 h-5.5 dark:bg-3c bg-gray-300 flex items-center justify-center`}>
                    <SvgIcon stopPropagation={false} value="arrowsBottom" className={`  duration-500 dark:text-d4 text-1f w-3 ${switchTop ? 'rotate-180' : ''}`} />
                </div>
                <div className='flex justify-between items-start mb-4.5 px-2.25'>
                    <div className='flex'>
                        <ImageUrl className='w-15.5 min-w-15.5 h-15.5 mr-2.75 rounded-full' logo={info.logo} symbol={info.symbol} />
                        <div >
                            <div className='flex items-center'>
                                <span className='text-xl dark:text-white text-black mr-2.25'>{info.symbol}</span>
                                <Image src={chainLogo[chain]} className='w-4.75' alt="" />

                            </div>
                            <div className='dark:text-4b text-af text-13'>{info.name}</div>
                            <div className='text-6b text-xs flex items-center'>
                                {mobileHidden(info.pool.quote_mint_address, 4, 4)}
                                <Copy className='ml-1.75 w-3 ' text={info.pool.quote_mint_address} />
                            </div>
                        </div>
                    </div>
                    <div className='flex flex-col items-end justify-between h-full'>
                        <div className='flex items-center'>
                            <SvgIcon onClick={() => setIsModalOpen(true)} className='w-4.75 text-6b mr-3.25 cursor-pointer' value='share' />
                            <Follow chain={chain} className='w-4' follow={info.follow} quote_mint_address={info.pool.quote_mint_address} />
                        </div>
                        <div className='flex items-center text-6b mt-7.75'>
                            {info.dev.website && <SvgIcon onClick={() => window.open(info.dev.website)} className='w-3  mr-1.5 cursor-pointer ' value='website' />}
                            {info.dev.telegram && <SvgIcon onClick={() => window.open(`https://t.me/${info.dev.telegram}`)} className={`w-3.25 cursor-pointer mr-1.5 `} value='telegram' />}
                            {info.dev.twitter_username && <SvgIcon onClick={() => window.open(`https://x.com/${info.dev.twitter_username}`)} className={`w-3.25 cursor-pointer  `} value='x' />}
                        </div>
                    </div>
                </div>
                <div className='text-center px-2.5 mb-3.25'>
                    <div className='flex items-end justify-between pb-1.5'>
                        <div className='flex-[2] text-left'>
                            <PriceCom className='pl-2 text-2xl pb-0.25' num={info.price.price} />
                            <div className={`pl-2 text-lg flex items-center`}><span className='text-6b pr-1'>24h</span><TablePercentage className=' ' number1={info.price.price_24h} />
                            </div>
                        </div>
                        <ItemT className='flex-[1]' title={t('mc')} showPrefix text={info.price.market_cap} />
                        <ItemT className='flex-[1]' title={t('liq')} text={info.liquidity} />
                    </div>
                    <div className='flex items-end justify-between  pb-1.5'>
                        <ItemT className='flex-[1]' title={t('age')} text={info.open_timestamp} funType={2} />
                        <ItemT className='flex-[1]' title={t('holders')} text={info.holder_count} />
                        <ItemT className='flex-[1]' title={`24h ${t('vol')}`} showPrefix text={info.price.volume_24h} />
                        <ItemT className='flex-[1]' title={t('txns')} text={info.price.swaps} />
                    </div>
                    <div className='flex items-end justify-between'>
                        <ItemT className='flex-[1]' title={t('vol')} showPrefix text={info.price.volume} />
                        <ItemT className='flex-[1]' title={t('buys')} showPrefix text={info.price.buy_volume_24h} textBool={true} type={3} />
                        <ItemT className='flex-[1]' title={t('sells')} showPrefix text={info.price.sell_volume_24h} textBool={false} type={3} />
                        <ItemT className='flex-[1]' title={t('netBuy')} showPrefix text={info.price.buy_volume_24h - info.price.sell_volume_24h} showMatheSymbo textBool={true} type={3} />
                    </div>
                </div>
                <div className='text-center border border-4b rounded-10 flex flex-wrap  gap-2 justify-between p-1.25 mx-2.25 '>
                    <ItemT className='min-w-16' type={2} funType={3} title='5m' showBfb text={info.price.price_5m} />
                    <ItemT className='min-w-16' type={2} funType={3} title='1h' showBfb text={info.price.price_1h} />
                    <ItemT className='min-w-16' type={2} funType={3} title='6h' showBfb text={info.price.price_6h} />
                    <ItemT className='min-w-16' type={2} funType={3} title='24h' showBfb text={info.price.price_24h} />
                </div>
                <div className={` px-5 overflow-hidden  duration-500 pt-4 ${switchTop ? '  pb-6.55 h-48' : 'h-0 p-0'}`}>
                    <ItemY className=' text-base dark:text-98 text-6b' title={t('marketInfo')} text={<SvgIcon className='w-4.75 ' value='solana' />} />
                    <ItemY title={t('totalLiq')}
                        // text={'$789.23K(2309 SOL)'} 
                        text={`$${formatLargeNumber(info.pool.liquidity)}(${formatLargeNumber((info.pool.liquidity / info.base_price))}SOL)`}
                    />
                    <ItemY title={t('totalSupply')} text={info.total_supply} />
                    <ItemY title={t('pair')}
                        text={<div className='flex items-center'>
                            {mobileHidden(info.address, 4, 4)}
                            <Copy className='ml-1.25 w-3' text={info.address} />
                        </div>} />
                    <ItemY title={t('tokenCreator')} text={<div className='flex items-center'>{mobileHidden(info.dev.creator_address, 4, 4)}{`(${formatLargeNumber(info.dev.creator_token_balance / info.base_price)}SOL)`} <Copy className='ml-1.25 w-3' text={info.dev.creator_address} /></div>} />
                    <ItemY title={t('poolCreated')} text={formatDate(info.creation_timestamp)} />
                </div>
            </div>
            <Modal open={isModalOpen} closable={false} onOk={() => setIsModalOpen(false)} onCancel={() => setIsModalOpen(false)}
                footer={null}
            >
                <div className='dark:bg-28 bg-fb w-full rounded-10 box-border p-5'>
                    <div className='dark:text-white text-black text-2xl mb-2'>Share</div>
                    <div className='break-words dark:bg-black bg-f1 overflow-hidden px-3  py-3 rounded-10 dark:text-white text-black'>
                        😢😢RACCOON 24h内下跌-77.97%，现价$0.0₅3348, 用GMGN更快发现，更快买卖！#RACCOON #GMGN https://gmgn.ai/sol/token/BVhB4kmteu9M1wV7RzqepExfhF6MjHeXjdJeE2VRpump
                    </div>
                    <div className='flex justify-center items-center mt-5 text-77'>
                        <div className='flex items-center justify-center cursor-pointer dark:hover:border-white dark:hover:text-white hover:border-black hover:text-black border-77 w-10 h-10 rounded-full border'>
                            <SvgIcon className='w-5' value='x' />
                        </div>
                        <div className='flex items-center justify-center cursor-pointer dark:hover:border-white dark:hover:text-white hover:border-black hover:text-black border-77 w-10  h-10 mx-10 rounded-full border'>
                            <SvgIcon className='w-5' value='telegram' />
                        </div>
                        <div className='flex items-center justify-center cursor-pointer dark:hover:border-white dark:hover:text-white hover:border-black hover:text-black border-77 w-10 h-10 rounded-full border'>
                            <SvgIcon className='w-4' value='copy' />
                        </div>
                    </div>
                </div>
            </Modal >
        </div >
    )
}

export default memo(Transaction)

/**
 * 渲染一个带有标题和文本的项，根据提供的类型和格式化类型对文本进行格式化
 * @param {Object} props - 组件属性
 * @param {string} props.title - 项的标题
 * @param {number} props.text - 需要格式化的文本内容
 * @param {number} [props.type=1] - 项的类型，决定文本的显示方式
 * @param {number} [props.funType=1] - 格式化函数的类型，决定文本的格式化方式
 * @param {string} [props.className=''] - 项的CSS类名
 * @param {string} [props.textClassName=''] - 文本的CSS类名
 * @param {boolean} [props.textBool] - 是否显示文本前缀的布尔值
 * @param {boolean} [props.showPrefix=false] - 是否显示前缀（如货币符号）
 * @param {boolean} [props.showBfb=false] - 是否显示百分比符号（%）
 */
const ItemT = ({ title, text, type = 1, funType = 1, className = '', textClassName = '', textBool, showPrefix = false, showMatheSymbo = false, showBfb = false }: { showPrefix?: boolean, type?: number, funType?: number, title: string, text: number, textClassName?: string, className?: string, textBool?: boolean, showMatheSymbo?: boolean, showBfb?: boolean }) => {
    // 定义一个函数数组，根据funType选择相应的格式化函数
    const fun = [formatLargeNumber, (value: number) => formatDuration(value, true), (value: number) => multiplyAndTruncate(value, true)]
    // 根据type值选择不同的渲染方式
    return (
        <div className={`${className} text-13`}>
            <div className='dark:text-6b text-af'>{title}</div>
            {type === 1 &&
                <div className={`dark:text-white text-black ${textClassName}`}>
                    {showPrefix && '$'}{fun[funType - 1](text)}{showBfb && '%'}
                </div>
            }
            {
                type === 2 &&
                <div className={`${text === 0 ? 'dark:text-white text-black' : text > 0 ? "dark:dark:text-6f text-2bc " : "text-ff"} ${textClassName}`}>
                    {showMatheSymbo || text !== 0 ? text >= 0 ? '+' : '-' : ''}{showPrefix && '$'}{fun[funType - 1](text)}{showBfb && '%'}
                </div>
            }
            {
                type === 3 &&
                <div className={`${textBool ? "dark:dark:text-6f text-2bc " : "text-ff"} ${textClassName}`}>
                    {showMatheSymbo ? textBool ? '+' : '-' : ''}{showPrefix && '$'}{fun[funType - 1](text)}{showBfb && '%'}
                </div>
            }
        </div>

    )
}

/**
 * 渲染一个简单的带有标题和文本的项，主要用于显示简短的信息
 * @param {Object} props - 组件属性
 * @param {string} props.title - 项的标题
 * @param {ReactNode | string} props.text - 项的文本内容，可以是字符串或React节点
 * @param {string} [props.className=''] - 项的CSS类名
 * @param {string} [props.textClassName=''] - 文本的CSS类名
 */
const ItemY = ({ title, text, className = 'dark:text-6b text-af', textClassName = '' }: { title: string, text: ReactNode | string, textClassName?: string, className?: string }) => {
    return <div className={` mb-1.75 flex items-center justify-between text-13 ${className}`}>
        <div className=''>{title}</div>
        <div className={` ${textClassName}`}>{text}</div>
    </div>
}