'use client'
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import InfiniteScroll from 'react-infinite-scroll-component';

import useUpdateEffect from '@/hook/useUpdateEffect';
import { ChainsType } from '@/i18n/routing';
import {
  transactionListType,
  transactionType,
} from '@/interface';
import { transactionListHttp } from '@/request';
import {
  formatLargeNumber,
  mobileHidden,
} from '@/utils';
import { LoadingOutlined } from '@ant-design/icons';

const Empty = dynamic(() => import('@/component/Empty'));
const TableAge = dynamic(() => import('@/component/TableAge'));
/**
 * 交易信息组件
 * 该组件用于显示特定地址的交易历史信息
 * @param {ChainsType} chain 链类型，用于区分不同的区块链
 * @param {string} address 地址，用户钱包地址
 * @param {transactionType[]} marketKline 交易数据
 */
const TransactionInfo = ({ chain, address, marketKline }: { chain: ChainsType, address: string, marketKline: transactionType[] }) => {
    // 国际化翻译钩子
    const t = useTranslations('Trade')
    // 条件选择 Pump 或 Raydium
    const [pumpOrRaydium, setPumpOrRaydium] = useState('1');
    // 是否加载中
    const [loading, setLoading] = useState(true);
    // 数据表格状态，包含总记录数和交易列表
    const [dataTable, setDataTable] = useState<transactionListType>({
        total: 0,
        list: []
    });
    // 当前页码
    const [page, setPage] = useState(1);
    // 标题数据，用于显示切换选项
    const titleData = [
        { label: t('activity'), value: '1' },
        { label: 'Traders', value: '2' },
    ];
    // 定义表格的列配置
    const columns = [
        t('age'), t('type'), t('totalUSD'), t('amount'), t('price'), t('maker'), t('hash')
    ];

    /**
     * 加载下一页数据
     * 当数据正在加载时，该函数将返回，以防止重复加载
     */
    const nextPage = () => {
        console.log('nextPage');

        if (loading) return
        setPage(page + 1)
    }

    /**
     * 获取交易列表数据
     * 根据链类型、地址和分页信息请求交易数据
     */
    const transactionListHttpFun = useCallback(async () => {
        setLoading(true)
        try {
            const { data } = await transactionListHttp<transactionListType>(chain, address, {
                page: page,
                size: 100,
                order_by: 'created_timestamp',
                direction: 'desc'
            })
            setDataTable({ list: [...dataTable.list, ...data.list], total: data.total })
        } catch (e) {
            console.log('transactionListHttp', e)
        }
        setLoading(false)
    }, [chain, address, page])
    const typeData = useMemo(() => [t('created'), t('buy'), t('sell'), t('addLiquidity'), t('removeLiquidity')], [t])
    // 初始化时加载交易数据
    useEffect(() => {
        transactionListHttpFun()
    }, [transactionListHttpFun])
    // 将订阅的数据添加到数据表格中
    useEffect(() => {
        setDataTable({ list: [...marketKline.reverse(), ...dataTable.list], total: dataTable.total })
    }, [marketKline])
    // 地址发送变化时，清空数据表格
    useUpdateEffect(() => {
        setDataTable({ list: [], total: 0 })
        setLoading(true)
    }, [address])
    return (
        <div className="mt-2 mr-1.5 font-semibold">
            <div className="flex items-center ml-1.5  ">
                {titleData.map((item, index) => (
                    <div
                        key={index}
                        onClick={() => setPumpOrRaydium('1')}
                        className={`text-6b text-base  ${pumpOrRaydium === item.value && 'dark:text-white text-black'} mr-3 cursor-pointer`}
                    >
                        {item.label}
                    </div>
                ))}
            </div>
            <div className="custom830:overflow-x-auto custom830:scrollbar-none">
                <div className="flex items-center justify-between  mt-4 mb-1.5">
                    {columns.map((item, index) => (
                        <div key={index} className={`text-4b text-13   flex-[1] ${index === 0 ? 'text-left  pl-6' : 'text-center'} custom830:min-w-36`}>
                            {item}
                        </div>
                    ))}
                </div>

                <InfiniteScroll
                    dataLength={dataTable.list.length}
                    next={nextPage}
                    className="rounded-10   border dark:border-1f border-d6  text-center box-border pt-4  overflow-y-auto scrollbar-none custom830:min-w-[63rem]"
                    hasMore={loading || dataTable.total > dataTable.list.length}
                    height={'29.6875rem'}
                    loader={<LoadingOutlined className="text-3xl dark:text-f1 text-b1" />
                    }
                >
                    {dataTable.list.map((item, index) => {
                        const { timestamp, swap_type, volume, quote_amount, quote_price, maker_address, tx_hash } = item
                        const isType = swap_type === 0 || swap_type === 1 || swap_type === 3
                        const className = isType ? 'dark:text-b6 text-2bc' : 'dark:text-ffb text-ff'
                        const borderColor = isType ? 'dark:border-b6 border-2bc' : 'dark:border-ffb border-ff'
                        return (
                            <div className="flex items-center justify-between text-13 py-1 rounded-5 mx-1 dark:hover:bg-333 hover:bg-zinc-300" key={index}>
                                <div className="flex-[1] custom830:min-w-36 text-left dark:text-4b text-77  pl-6">
                                    <TableAge value={timestamp} />
                                </div>
                                <div className={`flex-[1] custom830:min-w-36 ${className}`}>
                                    {typeData[swap_type]}
                                </div>
                                <div className={`flex-[1] custom830:min-w-36 ${className}`}>
                                    {formatLargeNumber(volume)}
                                </div>
                                <div className={`flex-[1] custom830:min-w-36 ${className}`}>
                                    {formatLargeNumber(quote_amount)}
                                </div>
                                <div className={`flex-[1] custom830:min-w-36 ${className}`}>
                                    {formatLargeNumber(quote_price)}
                                </div>
                                <a href={`https://solscan.io/account/${maker_address}`} target="_blank" className="flex-[1] custom830:min-w-36 dark:text-white text-4b">
                                    <span className={`border-b  ${borderColor}`}>{mobileHidden(maker_address, 4, 4)}</span>
                                </a>
                                <a href={`https://solscan.io/tx/${tx_hash}`} target="_blank" className="flex-[1] custom830:min-w-36 dark:text-white text-4b">
                                    <span className={`border-b  ${borderColor}`}>
                                        {mobileHidden(tx_hash, 4, 4)}
                                    </span>
                                </a>
                            </div>
                        )
                    })}
                    {
                        !loading && !dataTable.list.length && <Empty />
                    }
                    <p className={`dark:text-white text-slate-300 text-sm ${dataTable.list.length >= dataTable.total && !loading ? 'block' : 'hidden'}`} style={{ textAlign: 'center' }}>
                        <b>{t('overList')}</b>
                    </p>
                </InfiniteScroll>
            </div>
        </div>
    )
}

export default memo(TransactionInfo)