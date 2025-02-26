import { useRouter } from "@/i18n/routing"
import { XtopPlateType } from "@/interface"
import { formatLargeNumber } from "@/utils"
import { useTranslations } from "next-intl"
import { memo, useEffect, useState } from "react"
const onwIcon = '👑 '
const seleDefClass = [
    { tr: 'text-2xl text-fc', td: 'text-2.75xl', className: "  pb-8 pt-3.75" },
    { tr: 'text-2xl dark:text-fd text-f6', td: 'text-2.75xl ', className: "pb-8.5" },
    { tr: 'text-2xl dark:text-fe text-da', td: 'text-2.75xl ', className: " pb-5.75" }
]
const xtopData: any = [
    {
        plate: "DeFi",
        plate_trend: true,
        plate_market_cap: 500000000,
        plate_swaps: 2000,
        data: [
            {
                logo: "https://example.com/defi_logo.png",
                symbol: "UNI",
                chain: "Ethereum",
                address: "0x1234567890abcdef1234567890abcdef12345678",
                pool_address: "0xabcdef1234567890abcdef1234567890abcdef12",
                market_cap: 200000000,
                volume: 10000000,
                price: 5,
                initial_liquidity: 5000000,
                liquidity: 8000000,
                trend: true,
                telegram: "https://t.me/defi_uni",
                twitter_username: "defi_uni",
                website: "https://uniswap.org",
            },
        ],
    },
    {
        plate: "NFT",
        plate_trend: false,
        plate_market_cap: 300000000,
        plate_swaps: 1200,
        data: [
            {
                logo: "https://example.com/nft_logo.png",
                symbol: "MANA",
                chain: "Ethereum",
                address: "0x9876543210abcdef9876543210abcdef98765432",
                pool_address: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                market_cap: 120000000,
                volume: 5000000,
                price: 1.2,
                initial_liquidity: 3000000,
                liquidity: 4000000,
                trend: false,
                telegram: "https://t.me/mana_group",
                twitter_username: "mana_official",
                website: "https://decentraland.org",
            },
        ],
    },
    {
        plate: "Gaming",
        plate_trend: true,
        plate_market_cap: 700000000,
        plate_swaps: 1500,
        data: [
            {
                logo: "https://example.com/gaming_logo.png",
                symbol: "AXS",
                chain: "Ethereum",
                address: "0x123456789abcdef123456789abcdef123456789a",
                pool_address: "0xabcdefabcdefabcdefabcdefabcdefabcdef5678",
                market_cap: 400000000,
                volume: 15000000,
                price: 10,
                initial_liquidity: 7000000,
                liquidity: 9000000,
                trend: true,
                telegram: "https://t.me/axs_token",
                twitter_username: "axs_game",
                website: "https://axieinfinity.com",
            },
        ],
    },
    {
        plate: "Metaverse",
        plate_trend: true,
        plate_market_cap: 800000000,
        plate_swaps: 1300,
        data: [
            {
                logo: "https://example.com/metaverse_logo.png",
                symbol: "ENJ",
                chain: "Ethereum",
                address: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                pool_address: "0x3333333333333333333333333333333333333333",
                market_cap: 500000000,
                volume: 20000000,
                price: 2.5,
                initial_liquidity: 5000000,
                liquidity: 6000000,
                trend: true,
                telegram: "https://t.me/enj_group",
                twitter_username: "enj_project",
                website: "https://enjin.io",
            },
        ],
    },
    {
        plate: "Stablecoins",
        plate_trend: false,
        plate_market_cap: 1200000000,
        plate_swaps: 500,
        data: [
            {
                logo: "https://example.com/stablecoin_logo.png",
                symbol: "USDT",
                chain: "Ethereum",
                address: "0x9999999999999999999999999999999999999999",
                pool_address: "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                market_cap: 600000000,
                volume: 2000000,
                price: 1,
                initial_liquidity: 5000000,
                liquidity: 6000000,
                trend: false,
                telegram: "https://t.me/usdt_group",
                twitter_username: "usdt_token",
                website: "https://tether.to",
            },
        ],
    },
    {
        plate: "Layer2",
        plate_trend: true,
        plate_market_cap: 250000000,
        plate_swaps: 800,
        data: [
            {
                logo: "https://example.com/layer2_logo.png",
                symbol: "MATIC",
                chain: "Polygon",
                address: "0x8888888888888888888888888888888888888888",
                pool_address: "0x7777777777777777777777777777777777777777",
                market_cap: 250000000,
                volume: 6000000,
                price: 1.25,
                initial_liquidity: 4000000,
                liquidity: 4500000,
                trend: true,
                telegram: "https://t.me/matic_network",
                twitter_username: "matic_polygon",
                website: "https://polygon.technology",
            },
        ],
    },
    {
        plate: "DEX",
        plate_trend: true,
        plate_market_cap: 600000000,
        plate_swaps: 1400,
        data: [
            {
                logo: "https://example.com/dex_logo.png",
                symbol: "CAKE",
                chain: "Binance Smart Chain",
                address: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                pool_address: "0x1234567890abcdef1234567890abcdef12345678",
                market_cap: 300000000,
                volume: 12000000,
                price: 2.5,
                initial_liquidity: 8000000,
                liquidity: 9000000,
                trend: true,
                telegram: "https://t.me/cake_swap",
                twitter_username: "pancakeswap",
                website: "https://pancakeswap.finance",
            },
        ],
    },
    {
        plate: "Privacy",
        plate_trend: false,
        plate_market_cap: 150000000,
        plate_swaps: 700,
        data: [
            {
                logo: "https://example.com/privacy_logo.png",
                symbol: "XMR",
                chain: "Monero",
                address: "0x2222222222222222222222222222222222222222",
                pool_address: "0x4444444444444444444444444444444444444444",
                market_cap: 150000000,
                volume: 3000000,
                price: 150,
                initial_liquidity: 1000000,
                liquidity: 1500000,
                trend: false,
                telegram: "https://t.me/xmr_channel",
                twitter_username: "monero_official",
                website: "https://getmonero.org",
            },
        ],
    },
    {
        plate: "Storage",
        plate_trend: true,
        plate_market_cap: 180000000,
        plate_swaps: 600,
        data: [
            {
                logo: "https://example.com/storage_logo.png",
                symbol: "FIL",
                chain: "Filecoin",
                address: "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                pool_address: "0x5555555555555555555555555555555555555555",
                market_cap: 180000000,
                volume: 4000000,
                price: 4.5,
                initial_liquidity: 3000000,
                liquidity: 3500000,
                trend: true,
                telegram: "https://t.me/filcoin_network",
                twitter_username: "filecoin",
                website: "https://filecoin.io",
            },
        ],
    },
];
/**
 * SearchDef 组件用于搜索默认数据和导航到 xtop 页面
 * @param {Object} props 组件属性
 * @param {Function} props.setLoading 设置加载状态的函数
 * @param {string} props.className 可选的类名字符串
 */
const SearchDef = ({ setLoading, className }: { setLoading: Function, className?: string }) => {
    // 路由
    const router = useRouter()
    // 翻译
    const t = useTranslations('Header')
    // 默认的数据
    const [defaultData, setDefaultDataData] = useState<XtopPlateType>(xtopData || [])
    // 用户点击xtop板块去xtop页面,并传递板块的index
    const goXtop = (index: number) => {
        router.push({
            pathname: '/xtop',
            query: { plate: index + 1 }
        })
    }
    // 获取默认数据
    const getSearchDefault = async () => {
        setLoading(true)
        try {
            // setDefaultDataData()
        } catch (e) {
            console.log('', e);

        }
        setLoading(false)
    }
    // 在组件挂载时获取默认数据
    useEffect(() => {
        getSearchDefault()
    }, [])
    return (
        <div className={`box-border px-6 dark:text-white text-black ${className}`}>
            <div className='flex items-baseline  font-semibold pb-3  '><p className='text-4xl  pr-4'>Xtop</p><span className='text-base'>{t('inputDefTitle')}</span></div>
            <table className='table-auto w-full text-center'>
                <thead className='text-77 text-13 font-semibold'>
                    <tr >
                        <th className='text-left p-0'>{t('growp')}</th>
                        <th className="p-0">{t('mc')}</th>
                        <th className="p-0">{t('totalToken')}</th>
                        <th className="p-0">{t('txs')}</th>
                    </tr>
                </thead>
                <tbody>
                    {defaultData.map((item, index) => (
                        <tr onClick={() => { goXtop(index) }} key={index} className={`font-semibold cursor-pointer  ${seleDefClass[index] ? seleDefClass[index].tr : 'text-98  text-xl'}`}>
                            <td className={`p-0 text-left   ${seleDefClass[index] ? seleDefClass[index].className : 'pb-6.25 '} ${seleDefClass[index] && seleDefClass[index].td}`}>
                                {!!!index && onwIcon}
                                {item.plate}
                            </td>
                            <td className={`p-0 flex items-center ${seleDefClass[index] ? seleDefClass[index].className : 'pb-6.25 '}`}>
                                {formatLargeNumber(item.plate_market_cap)}
                            </td>
                            <td className={`p-0  ${seleDefClass[index] ? seleDefClass[index].className : 'pb-6.25 '}`}>
                                {formatLargeNumber(item.data.length)}
                            </td>
                            <td className={`p-0  ${seleDefClass[index] ? seleDefClass[index].className : 'pb-6.25 '}`}>{formatLargeNumber(item.plate_swaps)}</td>
                        </tr>
                    ))}

                </tbody>
            </table>
        </div>
    )
}
export default memo(SearchDef)