'use client'
import { memo, use, useCallback, useEffect, useState } from "react";
import { EarningsListType, MyWalletListType } from "@/interface";
import Share from "@/component/Share";
import { ChainsType } from "@/i18n/routing";
import { useEffectFiltrateStore, useFiltrateStore } from "@/store/filtrate";
import { useTranslations } from "next-intl";
import { orderBy } from "@/app/chainLogo";
import dynamic from "next/dynamic";


const Table = dynamic(() => import('@/component/Table')); 
const TableSortIcon = dynamic(() => import('@/component/TableSortIcon')); 
const TableEarnings = dynamic(() => import('@/component/TableEarnings')); 
const TableBalance = dynamic(() => import('@/component/TableBalance')); 
const TableTxns = dynamic(() => import('@/component/TableTxns')); 
const SvgIcon = dynamic(() => import('@/component/SvgIcon')); 
const UserInfo = dynamic(() => import('../component/UserInfo')); 
const TableAge = dynamic(() => import('@/component/TableAge')); 
const TableTokenInfo = dynamic(() => import('@/component/TableTokenInfo')); 

const MyWalletPage = ({ params }: { params: Promise<{ chain: ChainsType, address: string }> }) => {
    // 从参数中解构链和地址
    const { chain, address } = use(params);
    // 获取国际化翻译函数
    const t = useTranslations('MyWallet');

    // 控制分享模态框的打开状态
    const [isModalOpen, setIsModalOpen] = useState(false);
    // 用于存储表格数据
    const [dataTable, setDataTable] = useState<EarningsListType[]>([]);
    // 存储当前排序条件
    const [condition, setCondition] = useState<string>('');
    // 存储当前排序方向（升序、降序或无）
    const [direction, setDirection] = useState<'desc' | 'asc' | ''>('');
    // 控制是否已初始化轮询
    const [shouldPoll, setShouldPoll] = useState(false);
    // 控制是否停止轮询
    const [stopPolling, setStopPolling] = useState(true);
    // 存储加载状态
    const [loading, setLoading] = useState(true);
    // 存储代币信息的状态
    const [tokenInfo, setTokenInfo] = useState({
        logo: '', // 代币 logo
        symbol: '', // 代币符号
        main_price: 0, // 主网币价
        total_profit_percent: 0, // 总盈亏百分比
        total_profit: 0, // 总盈亏
        sells_num_usd: 0, // 卖入数量（美元）
        buys_num_usd: 0, // 买入数量（美元）
        token_price_usd: 0, // 余额（USDT）
    });

    // 使用状态管理钩子获取筛选条件和设置函数
    const { setFiltrate, filtrate } = useFiltrateStore();
    const { mywallet } = filtrate; // 从状态中解构出 mywallet 筛选条件

    // 监听筛选条件变化并更新排序状态
    useEffectFiltrateStore(({ mywallet }) => {
        setDirection(mywallet.direction); // 更新排序方向
        setCondition(mywallet.condition); // 更新排序字段
    });

    // 点击排序时的处理函数：更新排序条件和方向
    const getSelectedFilter = (direction: 'desc' | 'asc', key: string) => {
        setDirection(direction); // 更新排序方向
        setCondition(key); // 更新排序字段
        setFiltrate({ // 更新全局筛选条件
            ...filtrate,
            mywallet: {
                ...filtrate.mywallet,
                direction: direction, // 更新排序方向
                condition: key, // 更新排序字段
            },
        });
    };

    // 获取数据的异步函数
    const myWalletListHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        if (isLoading) setLoading(true); // 如果需要加载，设置加载状态为 true
        if (isStopPolling) setStopPolling(true); // 如果需要停止轮询，设置停止状态为 true
        try {
            // 设置表格数据为模拟数据
            // setDataTable(mockData);
        } catch (e) {
            console.log('trendListHttp', e); // 错误处理
        }
        if (isLoading) setLoading(false); // 加载结束，更新加载状态为 false
        if (isStopPolling) setStopPolling(false); // 恢复轮询状态为 false
    }, [chain, mywallet.condition, mywallet.direction]); // 依赖项
    // 初始化和时间选择变化时获取数据
    const columns = [
        // 配对信息列
        {
            title: <div className="text-xl pl-5.25">{t('token')}</div>,
            dataIndex: "address",
            width: "12.5rem",
            align: "left",
            fixed: 'left',
            render: (_value: string, data: EarningsListType) => {
                return <TableTokenInfo
                    className="pl-5.25"
                    logo={data.logo}
                    symbol={data.symbol}
                    chain={data.chain}
                    address={data.quote_mint_address}
                    twitter_username={data.twitter_username}
                    website={data.website} telegram={data.telegram} />;
            }
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('unrealized')} lightKey="unrealized_profits" checkedKey={condition} direction={direction} />,
            dataIndex: "unrealized_profits",
            align: "center",
            width: "10.5rem",
            render: (_value: number, data: EarningsListType) => (
                <TableEarnings money={data.unrealized_profits} moneyf={data.unrealized_profits_percent} />
            ),
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('totalProfit')} lightKey="total_profit" checkedKey={condition} direction={direction} />,
            dataIndex: "total_profit",
            align: "center",
            width: "8.5rem",
            render: (_value: number, data: EarningsListType) => (
                <TableEarnings money={data.total_profit} moneyf={data.total_profit_percent} />
            ),
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('balance')} lightKey="token_price_usd" checkedKey={condition} direction={direction} />,
            dataIndex: "token_price_usd",
            width: "6.5rem",
            align: "center",
            render: (_value: number, data: EarningsListType) => (
                <TableBalance num={`$${data.token_price_usd}`} num2={`${data.token_price_usd}`} />
            ),
        },
        {
            title: t('positionPercent'),
            dataIndex: "position_percent",
            align: "center",
            className: "font-semibold text-sm",
            width: "8.5rem",
            render: (value: number) => (
                <div>{value}%</div>
            ),
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('boughtAvg')} lightKey="bought_avg_price" checkedKey={condition} direction={direction} />,
            dataIndex: "bought_avg_price",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (_value: number, data: EarningsListType) => <TableBalance num={`${data.buys_num}`} num2={`$${data.bought_avg_price}`} />
            ,
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('soldAvg')} lightKey="sold_avg_price" checkedKey={condition} direction={direction} />,
            dataIndex: "sold_avg_price",
            align: "center",
            width: "11.5rem",
            className: "font-semibold text-sm",
            render: (_value: number, data: EarningsListType) => <TableBalance num={`${data.sells_num}`} num2={`$${data.sold_avg_price}`} />
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={`30D ${t('txs')}`} lightKey={orderBy.swaps} checkedKey={condition} direction={direction} />,
            dataIndex: "swaps",
            align: "center",
            width: "10.5rem",
            className: "font-semibold text-sm",
            render: (_value: number, data: EarningsListType) => <TableTxns swaps={data.swaps} buys={data.buys} sells={data.buys} />,
        }, {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('age')} lightKey="enter_address" checkedKey={condition} direction={direction} />,
            dataIndex: "enter_address",
            align: "center",
            width: "8.5rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TableAge value={value} />, // 格式化年龄数据

        }, {
            title: '',
            dataIndex: "",
            align: "center",
            width: "6.5rem",
            className: "font-semibold text-sm",
            render: (_value: any, data: EarningsListType) => <SvgIcon onClick={() => {
                setIsModalOpen(true); setTokenInfo({
                    logo: data.logo,
                    symbol: data.symbol,
                    main_price: data.main_price,
                    total_profit_percent: data.total_profit_percent,
                    total_profit: data.total_profit,
                    sells_num_usd: data.sells_num_usd,
                    buys_num_usd: data.buys_num_usd,
                    token_price_usd: data.token_price_usd
                })
            }} value="share" className=" w-6.75 text-6b cursor-pointer" />

            ,
        },
    ];
    // 在组件更新时执行副作用
    useEffect(() => {
        // 如果 shouldPoll 为 false，则将其设置为 true
        if (!shouldPoll) setShouldPoll(true);
        // 根据 shouldPoll 的值决定是否加载数据
        shouldPoll ? myWalletListHttp(false, true) : myWalletListHttp(true, true);
    }, [myWalletListHttp]); // 依赖于 myWalletListHttp，当其变化时重新执行该副作用

    // 轮询逻辑
    useEffect(() => {
        let timer: NodeJS.Timeout; // 定义定时器
        // 如果 stopPolling 为 false，则开始定时器
        if (!stopPolling) {
            // 每 3000 毫秒（3 秒）调用 myWalletListHttp 函数
            timer = setInterval(() => {
                myWalletListHttp(false, true);
            }, 3000);
        }
        // 清理函数：组件卸载时清除定时器
        return () => clearInterval(timer);
    }, [stopPolling, myWalletListHttp]); // 依赖于 stopPolling 和 myWalletListHttp，当它们变化时重新执行该副作用
    return (
        <div className="px-6.25 pt-8.5">
            <UserInfo address={address} />
            <Table skeletonClassname="mt-4" mh="38rem" skeletonMh="36.5rem" keys="id" columns={columns} data={dataTable} loading={loading} />
            <Share isOr={false} isModalOpen={isModalOpen} handleOk={() => setIsModalOpen(false)} tokenInfo={tokenInfo} address={address} />
        </div >
    )
}

export default memo(MyWalletPage)