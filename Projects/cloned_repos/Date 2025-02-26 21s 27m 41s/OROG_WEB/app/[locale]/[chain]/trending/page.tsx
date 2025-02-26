'use client'
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import { useTranslations } from 'use-intl';

import {
  chainMain,
  orderBy,
} from '@/app/chainLogo';
import { timeData } from '@/component/TimeSelection';
import { ChainsType } from '@/i18n/routing';
import {
  ListParams,
  TrendListType,
} from '@/interface';
import { trendListHttp } from '@/request';
import {
  useEffectFiltrateStore,
  useFiltrateStore,
} from '@/store/filtrate';
import { formatLargeNumber } from '@/utils';

const TimeSelection = dynamic(() => import('@/component/TimeSelection'));
const Table = dynamic(() => import('@/component/Table'));
const TableTokenInfo = dynamic(() => import('@/component/TableTokenInfo'));
const TableSortIcon = dynamic(() => import('@/component/TableSortIcon'));
const TableLiqInitial = dynamic(() => import('@/component/TableLiqInitial'));
const TableMc = dynamic(() => import('@/component/TableMc'));
const TableTxns = dynamic(() => import('@/component/TableTxns'));
const TableAge = dynamic(() => import('@/component/TableAge'));
const TableTooltip = dynamic(() => import('@/component/TableTooltip'));
const TablePriceTransition = dynamic(() => import('@/component/TablePriceTransition'));
const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const TablePercentage = dynamic(() => import('@/component/TablePercentage'));

const TrendingPage = () => {
    const t = useTranslations('Trending')
    const { chain }: { chain: ChainsType } = useParams();
    const [shouldPoll, setShouldPoll] = useState(false); // 是否初始化过
    const [stopPolling, setStopPolling] = useState(true); // 是否停止轮询
    const [loading, setLoading] = useState(true); // 是否加载中
    const [condition, setCondition] = useState<string>(''); // 排序条件
    const [direction, setDirection] = useState<'desc' | 'asc' | ''>(''); // 排序方向
    const [dataTable, setDataTable] = useState<ListParams[]>([]); // 数据
    const [period, setPeriod] = useState(''); // 时间选择
    const { setFiltrate, filtrate } = useFiltrateStore(); // 筛选条件的状态管理
    const { trend } = filtrate
    // 初始化筛选条件，避免服务端和客户端渲染结果不一致
    useEffectFiltrateStore(({ trend }) => {
        setPeriod(trend.period)
        setCondition(trend.condition); // 设置排序条件
        setDirection(trend.direction as 'desc' | 'asc'); // 设置排序方向
    });
    // 时间
    const getPeriod = (value: string) => {
        setPeriod(value)
        setFiltrate({
            ...filtrate,
            trend: {
                ...filtrate.trend,
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
            trend: {
                ...filtrate.trend,
                direction: direction, // 更新排序方向
                condition: key, // 更新排序字段
            },
        });
    };
    // 获取数据的异步函数
    const trendListFunHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true);
        isStopPolling && setStopPolling(true);
        try {
            const { data } = await trendListHttp<TrendListType>(
                chain,
                trend.period,
                { page: 1, size: 100, order_by: trend.condition, direction: trend.direction, filters: [] }
            );
            setDataTable(data.list);
        } catch (e) {
            console.log('trendListHttp', e);
        }
        isLoading && setLoading(false);
        isStopPolling && setStopPolling(false);
    }, [chain, trend]);
    // 根据当前 period 状态返回对应的时间显示文本
    const getTimeDataText = useMemo(() => {
        const data = timeData.filter(item => item.key === period); // 查找匹配的时间选项
        return data[0]?.text || ''; // 返回匹配的文本，如果未找到匹配项则返回空字符串
    }, [period, timeData]);
    // 定义表格的列配置
    const columns = [
        // 配对信息列
        {
            title: <div className="text-xl pl-5.25">{t('pairInfo')}</div>,
            dataIndex: "address",
            width: "12.5rem",
            align: "left",
            fixed: 'left',
            render: (_value: string, data: ListParams) => {
                return <TableTokenInfo
                    className="pl-5.25"
                    logo={data.logo}
                    symbol={data.symbol}
                    chain={data.chain}
                    address={data.quote_mint_address}
                    twitter_username={data.twitter_username}
                    website={data.website}
                    telegram={data.telegram} />;
            }
        },
        // 年龄列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('age')}
                lightKey={orderBy.created_timestamp}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "pool_creation_timestamp",
            align: "center",
            width: "10rem",
            className: "font-semibold text-sm",
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
            </div>,
            dataIndex: "initial_liquidity",
            align: "center",
            width: "11rem",
            render: (_value: number, data: ListParams) => (
                <TableLiqInitial isShow={true} liquidity={data.liquidity} initialLiquidity={data.initial_liquidity} />
            ),
        },
        {
            title: t('liq'),
            dataIndex: "liquidity",
            align: "center",
            width: "11rem",
            render: (_value: number, data: ListParams) => (
                <TablePriceTransition mainTokenName={chainMain[data.chain]} price={data.base_price} num={data.liquidity} />
            ),
        },
        // 市值列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('mc')}
                lightKey={orderBy.market_cap}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "market_cap",
            align: "center",
            width: "7.85rem",
            render: (value: number) => (
                <TableMc marketCap={value} />
            ),
        },
        // 交易次数列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={`${getTimeDataText} ${t('txns')}`}
                lightKey={orderBy.swaps}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "swaps",
            align: "center",
            width: "8rem",
            render: (_value: number, data: ListParams) => (
                <TableTxns swaps={data.swaps} buys={data.buys} sells={data.sells} />
            ),
        },
        // 交易量列
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={`${getTimeDataText} ${t('volume')}`}
                lightKey={orderBy.volume}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "volume",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => formatLargeNumber(value),
        },
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('priceChange1m')}
                lightKey={`${orderBy.price_change_}1m`}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "price_change_percent1m",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />
        },
        {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('priceChange5m')}
                lightKey={`${orderBy.price_change_}5m`}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "price_change_percent5m",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />
        }, {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('priceChange1h')}
                lightKey={`${orderBy.price_change_}1h`}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "price_change_percent1h",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />
        }, {
            title: <TableSortIcon
                onClick={getSelectedFilter}
                value={t('priceChange6h')}
                lightKey={`${orderBy.price_change_}6h`}
                checkedKey={condition}
                direction={direction} />,
            dataIndex: "price_change_percent6h",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />
        },
        // {
        //     title: t('mintAuthDisabled'),
        //     dataIndex: 'renounced_mint',
        //     align: "center",
        //     width: "10rem",
        //     render: (value: number) => (value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />),
        // },
        // {
        //     title: t('freezeAuthDisabled'),
        //     dataIndex: 'renounced_freeze_account',
        //     align: "center",
        //     width: "7rem",
        //     render: (value: number) => (value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />),
        // },
        // {
        //     title: t('lpBurned'),
        //     dataIndex: 'audit_lp_burned',
        //     align: "center",
        //     width: "7rem",
        //     render: (value: number) => (value ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />),
        // },
        // {
        //     title: t('top10Holders'),
        //     dataIndex: 'top_10_holder_rate',
        //     align: "center",
        //     width: "7rem",
        //     render: (value: number) =>
        //         <TableTooltip>
        //             {value < 0.3 ? <SvgIcon className="w-6 mx-auto" value="correct" /> : <SvgIcon className="w-6 mx-auto" value="error" />}
        //         </TableTooltip>
        //     ,
        // },
    ];
    // 初始化和时间选择变化时获取数据
    useEffect(() => {
        if (!shouldPoll) setShouldPoll(true);
        shouldPoll ? trendListFunHttp(false, true) : trendListFunHttp(true, true);
    }, [trendListFunHttp]);

    // 轮询逻辑
    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (!stopPolling) {
            timer = setInterval(() => {
                trendListFunHttp(false, true);
            }, 3000);
        }
        return () => clearInterval(timer); // 清除定时器
    }, [stopPolling, trendListFunHttp]);

    return (
        <div className="px-5.75 custom830:px-0">
            <TimeSelection className="mt-3.5 ml-9.75 mb-1 text-be font-semibold text-xl custom830:ml-2.5 " value={period} onClick={(time: string) => getPeriod(time)} />
            <div className="custom830:hidden">
                <Table mh="13.5rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
            </div>
            <div className="hidden custom830:block">
                <Table mh="17.5rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
            </div>
        </div>
    )
}
export default memo(TrendingPage)