'use client'
import { useCallback, useEffect, useState } from "react";
import { timeData } from "@/component/TimeSelection"
import { FollowListType, FollowListParams } from "@/interface";
import { formatLargeNumber } from "@/utils";
import { ChainsType } from "@/i18n/routing";
import { useParams } from "next/navigation";
import { chainMain, orderBy } from "@/app/chainLogo";
import { useEffectFiltrateStore, useFiltrateStore } from "@/store/filtrate";
import { useTranslations } from "next-intl";
import { useTokenStore } from "@/store/tokne";
import useConnectWallet from "@/store/wallet/useConnectWallet";
import useLogin from "@/hook/useLogin";
import { getFollowListHttp } from "@/request";
import dynamic from "next/dynamic";


const Table = dynamic(() => import('@/component/Table'));
const TableSortIcon = dynamic(() => import('@/component/TableSortIcon'));
const TimeSelection = dynamic(() => import('@/component/TimeSelection'));
const TableMc = dynamic(() => import('@/component/TableMc'));
const TableTxns = dynamic(() => import('@/component/TableTxns'));
const TablePriceTransition = dynamic(() => import('@/component/TablePriceTransition'));
const TableAge = dynamic(() => import('@/component/TableAge'));
const TableTokenInfo = dynamic(() => import('@/component/TableTokenInfo'));
const LinkComponent = dynamic(() => import('@/component/LinkComponent'));
const TablePercentage = dynamic(() => import('@/component/TablePercentage'));

const FollowPage = () => {
    // 导入必要的工具和函数
    const t = useTranslations('Follow'); // 用于翻译 'Follow' 命名空间的内容
    const tToast = useTranslations('Toast'); // 用于翻译 'Toast' 命名空间的内容
    const { token } = useTokenStore(); // 获取全局 token 状态，用于判断用户是否已登录
    const { chain }: { chain: ChainsType } = useParams(); // 获取当前 URL 参数中的 chain 值
    const { address } = useConnectWallet(chain)
    // 获取检测执行函数
    const { checkExecution } = useLogin(chain)
    // 定义组件的本地状态
    const [period, setPeriod] = useState<string>(''); // 表示选中的时间范围
    const [loading, setLoading] = useState(true); // 控制加载状态的布尔值
    const [dataTable, setDataTable] = useState<FollowListParams[]>([]); // 表格中显示的数据
    const [condition, setCondition] = useState<string>(''); // 当前选中的排序条件
    const [direction, setDirection] = useState<'desc' | 'asc' | ''>(''); // 当前选中的排序方向
    const [shouldPoll, setShouldPoll] = useState(false); // 标识是否需要进行轮询
    const [stopPolling, setStopPolling] = useState(true); // 控制是否停止轮询
    const { setFiltrate, filtrate } = useFiltrateStore(); // 获取和设置全局的筛选条件状态
    const { follow } = filtrate; // 从全局状态中提取筛选条件 follow
    // 监听 filtrate 状态的变化，并根据其更新本地状态
    useEffectFiltrateStore(({ follow }) => {
        setPeriod(follow.period); // 同步筛选条件的时间范围
        setCondition(follow.condition); // 同步排序条件
        setDirection(follow.direction); // 同步排序方向
    });

    // 更新时间选择并同步筛选条件
    const getPeriod = (key: string) => {
        setPeriod(key); // 更新本地状态
        setFiltrate({
            ...filtrate,
            follow: {
                ...filtrate.follow,
                period: key, // 同步更新全局筛选条件
            },
        });
    };

    // 定义异步函数，用于获取数据并更新状态
    const followListHttpFun = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true); // 如果需要加载，则设置加载状态
        isStopPolling && setStopPolling(true); // 如果需要停止轮询，则设置停止状态
        try {
            const { data } = await getFollowListHttp<FollowListType>(
                chain,
                follow.period,
                { page: 1, size: 100, order_by: follow.condition, direction: follow.direction, filters: [] }
            );
            setDataTable(data.list); // 更新表格数据
        } catch (e) {
            console.log('trendListHttp', e); // 捕获并打印错误
        }
        isLoading && setLoading(false); // 加载结束后，重置加载状态
        isStopPolling && setStopPolling(false); // 恢复轮询状态
    }, [chain, follow]);

    // 用于处理收藏和取消收藏的操作
    const removeFollowFun = async (id: number) => {
        checkExecution(() => {
            follow ? console.log('取消收藏') : console.log('收藏')
        })
    };

    // 点击排序条件时触发的操作
    const getSelectedFilter = (direction: 'desc' | 'asc', key: string) => {
        setDirection(direction); // 更新排序方向
        setCondition(key); // 更新排序条件
        setFiltrate({
            ...filtrate,
            follow: {
                ...filtrate.follow,
                direction, // 同步排序方向到全局状态
                condition: key, // 同步排序条件到全局状态
            },
        });
    };

    // 根据当前 period 状态返回对应的时间显示文本
    const getTimeDataText = () => {
        const data = timeData.filter(item => item.key === period); // 查找匹配的时间选项
        return data[0]?.text || ''; // 返回匹配的文本，如果未找到匹配项则返回空字符串
    };

    // 定义表格的列配置
    const columns = [
        // 配对信息列
        {
            title: <div className="text-xl pl-5.25">{t('token')}</div>,
            dataIndex: "address",
            width: "16rem",
            align: "left",
            fixed: 'left',
            render: (_value: string, data: FollowListParams) => {
                return <TableTokenInfo
                    isFollowShow={true}
                    className="pl-5.25"
                    logo={data.logo}
                    symbol={data.symbol}
                    chain={data.chain}
                    address={data.quote_mint_address}
                    twitter_username={data.twitter_username}
                    website={data.website}
                    telegram={data.telegram}
                    follow={true}
                    quote_mint_address={data.quote_mint_address} />;
            }
        },
        // 交易次数列
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={`${getTimeDataText()} ${t('txs')}`} lightKey={orderBy.swaps} checkedKey={condition} direction={direction} />,
            dataIndex: "swaps",
            align: "center",
            width: "14rem",
            render: (_value: number, data: FollowListParams) => (
                <TableTxns swaps={data.swaps} buys={data.buys} sells={data.sells} />
            ),
        },

        // 交易量列
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={`${getTimeDataText()} ${t('volume')}`} lightKey={orderBy.volume} checkedKey={condition} direction={direction} />,
            dataIndex: "volume",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => formatLargeNumber(value),
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value="1m%" lightKey="price_change_percent1m" checkedKey={condition} direction={direction} />,
            dataIndex: "price_change_percent1m",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />,
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value="5m%" lightKey="price_change_percent5m" checkedKey={condition} direction={direction} />,
            dataIndex: "price_change_percent5m",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />,
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value="1h%" lightKey="price_change_percent1h" checkedKey={condition} direction={direction} />,
            dataIndex: "price_change_percent1h",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />,
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value="6h%" lightKey="price_change_percent6h" checkedKey={condition} direction={direction} />,
            dataIndex: "price_change_percent6h",
            align: "center",
            width: "9rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TablePercentage number1={value} />,
        },
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('liq')} lightKey={orderBy.liquidity} checkedKey={condition} direction={direction} />,
            dataIndex: "liquidity",
            align: "center",
            width: "11rem",
            render: (_value: number, data: FollowListParams) => (
                <TablePriceTransition mainTokenName={chainMain[data.chain]} price={data.base_price} num={data.liquidity} />
            ),
        },
        // 市值列
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('mc')} lightKey={orderBy.market_cap} checkedKey={condition} direction={direction} />,
            dataIndex: "market_cap",
            align: "center",
            width: "7.85rem",
            render: (value: number) => (
                <TableMc marketCap={value} />
            ),
        },
        // 价格列
        {
            title: <TableSortIcon onClick={getSelectedFilter} lightKey="price" checkedKey={condition} direction={direction} value={t('price')} />, // 带排序功能的列标题
            dataIndex: "price", // 数据字段对应的键名
            align: "center", // 居中对齐
            width: "7rem", // 列宽度
            className: "font-semibold text-sm", // 自定义样式
            render: (value: number) => (`$${formatLargeNumber(value)}`), // 格式化价格数据并加上货币符号
        },
        // 年龄列
        {
            title: <TableSortIcon onClick={getSelectedFilter} value={t('age')} lightKey={orderBy.created_timestamp} checkedKey={condition} direction={direction} />,
            dataIndex: "pool_creation_timestamp",
            align: "center",
            width: "10rem",
            className: "font-semibold text-sm",
            render: (value: number) => <TableAge value={value} />, // 格式化年龄数据

        },
    ];

    // 初始化数据获取逻辑，监听时间选择变化
    useEffect(() => {
        if (!shouldPoll) setShouldPoll(true); // 如果未启用轮询，则启用
        shouldPoll ? followListHttpFun(false, true) : followListHttpFun(true, true); // 根据轮询状态调用数据获取函数
    }, [followListHttpFun]);

    // 轮询逻辑的实现
    useEffect(() => {
        let timer: NodeJS.Timeout; // 定义计时器变量
        if (!stopPolling) {
            // 如果轮询未停止，则设置计时器
            timer = setInterval(() => {
                followListHttpFun(false, true); // 每 3 秒获取一次数据
            }, 3000);
        }
        return () => clearInterval(timer); // 在组件卸载时清除计时器
    }, [stopPolling, followListHttpFun]);
    return (
        <div className="px-6.25 custom830:px-0">
            {
                !(address && token) ? <LinkComponent /> : <>
                    <TimeSelection className="mt-3.5 ml-4.25 text-be font-semibold mb-1 text-xl " value={period} onClick={getPeriod} />
                    <div className="custom830:hidden">
                        <Table mh="13.5rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
                    </div>
                    <div className="hidden custom830:block">
                        <Table mh="17.5rem" skeletonMh="14rem" keys="id" columns={columns} data={dataTable} loading={loading} />
                    </div>
                </>
            }
        </div>
    )
}
export default FollowPage
