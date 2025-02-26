'use client'
import { memo, useCallback, useEffect, useMemo, useState } from "react"
// import SvgIcon from "@/component/SvgIcon"
// import XTop from "./component/XTop"
import { ChainsType } from "@/i18n/routing"
import { useParams, useSearchParams } from "next/navigation"
import { ListParams, XtopListType, XtopPlateType } from "@/interface"
import { useTranslations } from "next-intl"
// import TableLiqInitial from "@/component/TableLiqInitial"
// import TableMc from "@/component/TableMc"
// import TableSortIcon from "@/component/TableSortIcon"
// import TableTokenInfo from "@/component/TableTokenInfo"
// import TableTxns from "@/component/TableTxns"
import { formatLargeNumber } from "@/utils"
import { useEffectFiltrateStore, useFiltrateStore } from "@/store/filtrate"
// import Table from "@/component/Table"
import { xTopListHttp, xTopPlateHttp } from "@/request"
import { orderBy } from "@/app/chainLogo"
// import TableTooltip from "@/component/TableTooltip"
// import TableAge from "@/component/TableAge"
import dynamic from "next/dynamic"

const Table = dynamic(() => import('@/component/Table'));
const TableSortIcon = dynamic(() => import('@/component/TableSortIcon'));
const TableLiqInitial = dynamic(() => import('@/component/TableLiqInitial'));
const TableMc = dynamic(() => import('@/component/TableMc'));
const TableTxns = dynamic(() => import('@/component/TableTxns'));
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const TableTooltip = dynamic(() => import('@/component/TableTooltip'));
const TableAge = dynamic(() => import('@/component/TableAge'));
const TableTokenInfo = dynamic(() => import('@/component/TableTokenInfo'));
const XTop = dynamic(() => import('./component/XTop'));

type periodType = { key: string, text: string, value: string }

const xtopDatas: XtopPlateType = [
    {
        plate: "DeFi",
        plate_trend: true,
        plate_market_cap: 120000000,
        plate_swaps: 450,
        data: [
            {
                logo: "https://example.com/logos/token1.png",
                symbol: "SOL-DFI",
                chain: "sol",
                address: "SoLDFI1address123",
                quote_mint_address: "QuoteMintAddress123",
                market_cap: 50000000,
                volume: 100000,
                price: 5,
                initial_liquidity: 200000,
                liquidity: 300000,
                trend: true,
                telegram: "https://t.me/sol_dfi",
                twitter_username: "@sol_dfi",
                website: "https://sol-dfi.com",
            },
            {
                logo: "https://example.com/logos/token2.png",
                symbol: "SOL-DFI2",
                chain: "sol",
                address: "SoLDFI2address123",
                quote_mint_address: "QuoteMintAddress456",
                market_cap: 30000000,
                volume: 80000,
                price: 3,
                initial_liquidity: 150000,
                liquidity: 250000,
                trend: false,
                telegram: "https://t.me/sol_dfi2",
                twitter_username: "@sol_dfi2",
                website: "https://sol-dfi2.com",
            },
        ],
    },
    {
        plate: "NFT",
        plate_trend: false,
        plate_market_cap: 90000000,
        plate_swaps: 320,
        data: [
            {
                logo: "https://example.com/logos/nft1.png",
                symbol: "SOL-NFT",
                chain: "sol",
                address: "SoLNFT1address123",
                quote_mint_address: "QuoteMintAddress789",
                market_cap: 40000000,
                volume: 120000,
                price: 10,
                initial_liquidity: 500000,
                liquidity: 600000,
                trend: true,
                telegram: "https://t.me/sol_nft",
                twitter_username: "@sol_nft",
                website: "https://sol-nft.com",
            },
            {
                logo: "https://example.com/logos/nft2.png",
                symbol: "SOL-NFT2",
                chain: "sol",
                address: "SoLNFT2address123",
                quote_mint_address: "QuoteMintAddress101",
                market_cap: 25000000,
                volume: 60000,
                price: 8,
                initial_liquidity: 300000,
                liquidity: 400000,
                trend: false,
                telegram: "https://t.me/sol_nft2",
                twitter_username: "@sol_nft2",
                website: "https://sol-nft2.com",
            },
        ],
    },
    {
        plate: "GameFi",
        plate_trend: true,
        plate_market_cap: 110000000,
        plate_swaps: 290,
        data: [
            {
                logo: "https://example.com/logos/gamefi1.png",
                symbol: "SOL-GFI",
                chain: "sol",
                address: "SoLGFI1address123",
                quote_mint_address: "QuoteMintAddress202",
                market_cap: 60000000,
                volume: 200000,
                price: 12,
                initial_liquidity: 700000,
                liquidity: 900000,
                trend: true,
                telegram: "https://t.me/sol_gfi",
                twitter_username: "@sol_gfi",
                website: "https://sol-gfi.com",
            },
            {
                logo: "https://example.com/logos/gamefi2.png",
                symbol: "SOL-GFI2",
                chain: "sol",
                address: "SoLGFI2address123",
                quote_mint_address: "QuoteMintAddress303",
                market_cap: 50000000,
                volume: 180000,
                price: 15,
                initial_liquidity: 600000,
                liquidity: 850000,
                trend: true,
                telegram: "https://t.me/sol_gfi2",
                twitter_username: "@sol_gfi2",
                website: "https://sol-gfi2.com",
            },
        ],
    },
];
const XtopPage = () => {
    const t = useTranslations('Xtop')
    // 获取路由参数chain
    const { chain }: { chain: ChainsType } = useParams();
    // 获取query参数 plate
    const searchParams = useSearchParams();
    const plate = searchParams.get('plate');
    // XtopPlate 数据
    const [xtopData, setXtopData] = useState<XtopPlateType>(xtopDatas)
    // 选择的天数 24h 7d 30d
    const [period, setPeriod] = useState<string>(''); // 时间选择
    const [dataTable, setDataTable] = useState<ListParams[]>([]); // 表格数据
    const [condition, setCondition] = useState<string>(''); // 排序条件
    const [direction, setDirection] = useState<'desc' | 'asc' | ''>(''); // 排序方向
    const [shouldPoll, setShouldPoll] = useState(false); // 是否已初始化轮询
    const [stopPolling, setStopPolling] = useState(true); // 是否停止轮询
    const [loading, setLoading] = useState(true); // 是否加载中
    const { setFiltrate, filtrate } = useFiltrateStore(); // 筛选条件的状态管理
    const { xtop } = filtrate
    useEffectFiltrateStore(({ xtop }) => {
        setPeriod(xtop.period)
        setDirection(xtop.direction)
        setCondition(xtop.condition)
    })
    // 切换天数,改变数据
    const changePeriod = async (value: string) => {
        setPeriod(value)
        setFiltrate({
            ...filtrate,
            xtop: {
                ...filtrate.xtop,
                period: value
            },
        });
    }
    // 点击排序时的处理函数：更新排序条件和方向
    const getSelectedFilter = (direction: 'desc' | 'asc', key: string) => {
        setDirection(direction); // 更新排序方向
        setCondition(key); // 更新排序字段
        setFiltrate({
            ...filtrate,
            xtop: {
                ...filtrate.xtop,
                direction: direction, // 更新排序方向
                condition: key, // 更新排序字段
            },
        });
    };
    // 获取板块数据
    const getXtopPlate = useCallback(async () => {
        try {
            const { data } = await xTopPlateHttp<XtopPlateType>(chain, xtop.period, { page: 1, size: 9 })
            // setXtopData(data.)
            console.log('data', data);

        } catch (e) {
            console.log(getXtopPlate, e)
        }
    }, [xtop.period, chain])
    // 异步获取数据的方法,注意这里使用store里面存放的数据
    const XtopListHttpFun = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true); // 如果需要显示加载状态
        isStopPolling && setStopPolling(true); // 如果需要停止轮询
        try {
            // 调用数据请求方法
            const { data } = await xTopListHttp<XtopListType>(
                chain, // 当前链
                xtop.period, // 时间范围
                { page: 1, size: 100, order_by: xtop.condition, direction: xtop.direction, filters: [] } // 请求参数
            );
            setDataTable(data.list); // 更新表格数据
        } catch (e) {
            console.log('XtopListHttpFun', e); // 错误处理
        }
        isLoading && setLoading(false); // 关闭加载状态
        isStopPolling && setStopPolling(false); // 重置轮询状态
    }, [chain, xtop]);
    const periodData: periodType[] = [
        {
            key: "1d",
            text: t('24h'),
            value: "24h"
        },
        {
            key: "7d",
            text: t('7d'),
            value: "7d"
        },
        {
            key: "30d",
            text: t('30d'),
            value: "30d"
        }
    ]
    // 筛选出periodData的text传入xtop组件
    const getPeriodDataText = useMemo(() => {
        const data = periodData.filter(item => {
            return item.key === period
        })
        return data[0]?.value || ''
    }, [period, periodData])
    // 定义表格列配置
    const columns = [
        // 配对信息列
        {
            title: <div className="text-xl pl-5.25">{t('xtopTotal')}</div>, // 列标题，显示为“Pair Info”
            dataIndex: "address", // 数据字段对应的键名
            width: "14.5rem", // 列宽度
            align: "left", // 左对齐
            fixed: 'left', // 固定在表格的左侧
            render: (_value: string, data: ListParams) => {
                // 自定义渲染，使用 TableTokenInfo 组件显示配对信息
                return <TableTokenInfo
                    className="pl-5.25"
                    logo={data.logo}
                    symbol={data.symbol}
                    chain={data.chain}
                    address={data.quote_mint_address}
                    twitter_username={data.twitter_username}
                    website={data.website}
                    telegram={data.telegram}
                />;
            }
        },
        // 年龄列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                lightKey={orderBy.created_timestamp}
                checkedKey={condition}
                direction={direction}
                value={t('age')}
            />, // 带排序功能的列标题
            dataIndex: "pool_creation_timestamp", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "8rem", // 列宽度
            className: "font-semibold text-sm", // 自定义样式
            render: (value: number) => <TableAge value={value} />, // 格式化年龄数据
        },
        // 流动性/初始列
        {
            title: <div className="flex items-center justify-center">
                <TableSortIcon
                    onClick={getSelectedFilter}
                    checkedKey={condition}
                    lightKey={orderBy.liquidity}
                    direction={direction}
                    value={t('liq')} />
                <div className="mx-1">/</div>
                <TableSortIcon
                    onClick={getSelectedFilter}
                    checkedKey={condition}
                    lightKey={orderBy.init_liquidity}
                    direction={direction}
                    value={t('initial')} />
            </div>, // 列标题，包含两个排序图标
            dataIndex: "liquidity", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "11rem", // 列宽度
            render: (_value: number, data: ListParams) => (
                <TableLiqInitial liquidity={data.liquidity} initialLiquidity={data.initial_liquidity} />
            ), // 自定义渲染，使用 TableLiqInitial 显示流动性数据
        },
        // 市值列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                checkedKey={condition}
                lightKey={orderBy.market_cap}
                direction={direction}
                value={t('mc')} />, // 带排序功能的列标题
            dataIndex: "market_cap", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7.85rem", // 列宽度
            render: (value: number) => (
                <TableMc marketCap={value} />
            ), // 自定义渲染，使用 TableMc 格式化市值
        },
        // 交易次数列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                checkedKey={condition}
                lightKey={orderBy.swaps}
                direction={direction}
                value={`${getPeriodDataText} ${t('txns')}`} />, // 带排序功能的列标题
            dataIndex: "swaps", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "11rem", // 列宽度
            render: (_value: number, data: ListParams) => (
                <TableTxns swaps={data.swaps} buys={data.buys} sells={data.sells} />
            ), // 自定义渲染，使用 TableTxns 显示交易次数数据
        },
        // 交易量列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                checkedKey={condition}
                lightKey={orderBy.volume}
                direction={direction}
                value={`${getPeriodDataText} ${t('volume')}`} />, // 带排序功能的列标题
            dataIndex: "volume", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7rem", // 列宽度
            className: "font-semibold text-sm", // 自定义样式
            render: (value: number) => formatLargeNumber(value), // 格式化交易量数据
        },
        // 价格列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                checkedKey={condition}
                lightKey={orderBy.quote_price}
                direction={direction}
                value={t('price')} />, // 带排序功能的列标题
            dataIndex: "price", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7rem", // 列宽度
            className: "font-semibold text-sm", // 自定义样式
            render: (value: number) => (`$${formatLargeNumber(value)}`), // 格式化价格数据并加上货币符号
        },
        // 布尔类型列：Mint Auth Disabled
        {
            title: t('mintAuthDisabled'), // 列标题
            dataIndex: 'renounced_mint', // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7rem", // 列宽度
            render: (value: number) => (
                value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
            ), // 根据布尔值显示正确或错误图标
        },
        // 布尔类型列：Freeze Auth Disabled
        {
            title: t('freezeAuthDisabled'), // 列标题
            dataIndex: 'renounced_freeze_account', // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "8rem", // 列宽度
            render: (value: number) => (
                value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
            ), // 根据布尔值显示正确或错误图标
        },
        // 布尔类型列：LP Burned
        {
            title: t('lpBurned'), // 列标题
            dataIndex: 'audit_lp_burned', // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "8rem", // 列宽度
            render: (value: number) => (
                value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />
            ), // 根据布尔值显示正确或错误图标
        },
        // 布尔类型列：Top 10 Holders
        {
            title: t('top10Holders'), // 列标题
            dataIndex: 'top_10_holder_rate', // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7rem", // 列宽度
            render: (value: number) => (
                <TableTooltip>
                    {value < 0.3 ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />}
                </TableTooltip>
            ), // 根据布尔值显示正确或错误图标
        },
    ];
    // 进入页面获取数据
    useEffect(() => {
        getXtopPlate()
    }, [getXtopPlate])
    // 页面初始化时或时间范围变化时获取数据
    useEffect(() => {
        if (!shouldPoll) setShouldPoll(true); // 初始化轮询状态
        shouldPoll ? XtopListHttpFun(false, true) : XtopListHttpFun(true, true); // 获取数据
    }, [XtopListHttpFun]);
    // 轮询逻辑
    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (!stopPolling) {
            timer = setInterval(() => {
                XtopListHttpFun(false, true);
            }, 3000);
        }
        return () => clearInterval(timer); // 清除定时器
    }, [stopPolling, XtopListHttpFun]);
    return (
        <div className="px-5  h-[calc(100vh-8.125rem)]  overflow-y-auto scrollbar-none custom830:px-0 ">
            <div className="pl-7.75 pt-3 ">
                <div className="text-xl dark:text-d9 text-28 font-semibold">{t('twitterHotTopics')}</div>
                <div className="flex items-end justify-between mt-2">
                    <div className="flex items-center text-6b text-3.25xl font-semibold">
                        {periodData.map((item, index) =>
                            <div key={index}
                                onClick={() => { changePeriod(item.key) }}
                                className={`mr-4 cursor-pointer ${item.key === period ? 'dark:text-white text-28' : ''}`}>
                                {item.text}
                            </div>)}
                    </div>
                    <div className="flex items-center text-82  custom830:mr-2  ">
                        <span className="mr-3.5 text-xl">{t('aboutXtop')}</span>
                        <TableTooltip title='prompt text' placement='left'>
                            <SvgIcon stopPropagation={false} value="about" className="w-7 cursor-pointer hover:text-28 dark:hover:text-white" />
                        </TableTooltip>
                    </div>
                </div>
            </div>
            <XTop plate={plate} period={getPeriodDataText} xtopData={xtopData} chain={chain} />
            <div className="custom830:pb-13">
                <Table mh="13rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
            </div>
        </div>
    )
}
export default memo(XtopPage)