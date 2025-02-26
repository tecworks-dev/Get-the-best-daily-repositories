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

import useWebSocket from '@/hook/useWebSocket';
import {
    ChainsType,
    usePathname,
} from '@/i18n/routing';
import {
    historicalTickerCandleType,
    TokenInfoType,
    transactionType,
} from '@/interface';
import { transactionInfoHttp } from '@/request';

const TokenList = dynamic(() => import('../component/TokenList'));
const TransactionToken = dynamic(() => import('../component/TransactionToken'));
const Transaction = dynamic(() => import('../component/Transaction'));
const TransactionInfo = dynamic(() => import('../component/TransactionInfo'));
// const TradingView = dynamic(() => import('@/component/TradingView/TradingView'), { ssr: false });

const Trade = () => {
    // 获取链信息
    const { chain }: { chain: ChainsType } = useParams();
    const pathname = usePathname();
    // 定义代币信息状态，存储代币的详细信息
    const [info, setInfo] = useState<TokenInfoType>()
    // 获取当前页面地址的最后一部分作为 address
    const address = useMemo(() => {
        const parts = pathname.split('/');
        return parts[parts.length - 1];
    }, [pathname]);

    // 维护当前地址历史状态，存储当前和上一个地址 用于取消订阅
    const [addressArr, setAddressArr] = useState<string[]>([address, '']);
    useEffect(() => {
        setAddressArr(prevState => {
            return [address, prevState[0]];
        });
    }, [address]);
    // 定义获取交易信息的函数，根据链类型和地址获取代币信息
    const transactionInfoHttpFun = useCallback(async () => {
        try {
            const { data } = await transactionInfoHttp<TokenInfoType>(chain, address)
            setInfo(data)
        } catch (error) {
            console.log('transactionInfoHttpFun', error)
        }
    }, [chain, address])
    // 使用useEffect钩子在组件挂载时调用获取交易信息的函数
    useEffect(() => {
        transactionInfoHttpFun()
    }, [transactionInfoHttpFun])
    // 维护交易时间间隔状态，存储当前和上一个时间间隔
    const [intervalState, setIntervals] = useState<string[]>(['', '']);

    // 交易蜡烛图数据
    const [marketKline, setMarketKline] = useState<historicalTickerCandleType>();

    // 交易数据
    const [marketTxActivity, setMarketTxActivity] = useState<transactionType[]>([]);

    // WebSocket 相关
    const { socket, webSocketInit, resubscribe, unsubscribe, isSocketOpen } = useWebSocket([
        setMarketTxActivity,
        setMarketKline
    ]);

    // 实时回调函数
    const [onRealtimeCallbackFun, setOnRealtimeCallbackFun] = useState<Function>();

    /**
     * 更新交易时间间隔
     * @param interval - 新的时间间隔
     * @param onRealtimeCallback - 回调函数
     */
    const getInterval = (interval: string, onRealtimeCallback: Function) => {
        setIntervals(prevState => [interval, prevState[0]]);
        if (!onRealtimeCallbackFun) {
            setOnRealtimeCallbackFun(() => onRealtimeCallback);
        }
    };

    // 初始化 WebSocket 连接
    useEffect(() => {
        webSocketInit(['market_tx_activity'], [{ chain, address: addressArr[0] }]);
    }, []);

    // 监听 intervalState[0] 变化，重新订阅 k线数据
    useEffect(() => {
        if (intervalState[0] && socket) {
            resubscribe(['market_kline'], [{ chain, address: addressArr[0], interval: intervalState[0] }]);
        }
    }, [intervalState[0]]);

    // 监听 intervalState[1] 变化，取消订阅旧的 k线数据
    useEffect(() => {
        if (intervalState[1] && socket) {
            unsubscribe(['market_kline'], [{ chain, address: addressArr[0], interval: intervalState[1] }]);
        }
    }, [intervalState[1]]);

    // 监听 addressArr[1] 变化，取消订阅旧的交易数据
    useEffect(() => {
        if (addressArr[1] && socket) {
            unsubscribe(
                ['market_tx_activity', 'market_kline'],
                [
                    { chain, address: addressArr[1] },
                    { chain, address: addressArr[1], interval: intervalState[0] }
                ]
            );
        }
    }, [addressArr[1]]);

    // 监听 WebSocket 状态，重新初始化订阅
    useEffect(() => {
        if (isSocketOpen === 2) {
            webSocketInit(
                ['market_tx_activity', 'market_kline'],
                [
                    { chain, address: addressArr[0] },
                    { chain, address: addressArr[0], interval: intervalState[0] }
                ]
            );
        }
    }, [isSocketOpen]);

    // 监听 marketKline 变化，执行回调
    useEffect(() => {
        if (onRealtimeCallbackFun && marketKline) {
            onRealtimeCallbackFun({
                ...marketKline,
                time: marketKline.time * 1000, // 时间转换为毫秒
            });
        }
    }, [marketKline]);
    return (
        <div className="pt-2 flex h-[calc(100vh-8.125rem)] custom830:flex-wrap custom830:h-auto custom830:pb-20">
            {/* 代币列表 */}
            <div className="custom830:hidden block">
                <TokenList chain={chain} />
            </div>
            {/* 交易视图和交易信息 */}
            <div className="flex-grow overflow-y-auto h-full scrollbar-none">
                <div className="h-[60vh]">
                    {/* <TradingView getInterval={getInterval} address={address} /> */}
                </div>
                <TransactionInfo chain={chain} address={address} marketKline={marketTxActivity} />
            </div>

            {/* 交易代币和交易组件 */}
            <div className="w-80 ml-2 overflow-y-auto scrollbar-none custom830:hidden">
                <TransactionToken chain={chain} info={info} />
                <Transaction decimals={info?.decimals || 0} chain={chain} quoteMintAddress={info?.pool.quote_mint_address || ''} quoteSymbol={info?.pool.quote_symbol || ''} />
            </div>
        </div>
    );
};

export default memo(Trade);