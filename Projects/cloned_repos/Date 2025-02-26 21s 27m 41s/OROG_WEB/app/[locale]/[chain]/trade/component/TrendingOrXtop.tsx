'use client'
import { ChainsType } from "@/i18n/routing";
import { ListParams, TrendListType, XtopListType } from "@/interface";
import { trendListHttp, xTopListHttp } from "@/request";
import { useFiltrateStore } from "@/store/filtrate";
import { useState, useCallback, useEffect, useMemo, memo } from "react";
import TXItem from "./TXItem";
import SkeletonCom from "./Skeleton";
/**
 * TrendingOrXtop 组件用于显示趋势或最热门的交易列表
 * @param {ChainsType} chain 链类型
 * @param {string} period 时间周期
 * @returns {funType} 接口选择
 */
const TrendingOrXtop = ({ chain, period, funType, keys }: { chain: ChainsType, period: string, funType: number, keys: number }) => {
    const [loading, setLoading] = useState(true); // 是否加载中
    const [stopPolling, setStopPolling] = useState(true); // 是否停止轮询
    const [shouldPoll, setShouldPoll] = useState(false); // 是否初始化过
    const { filtrate } = useFiltrateStore(); // 筛选条件的状态管理
    const [dataTable, setDataTable] = useState<ListParams[]>([]); // 表格中显示的数据
    const { trade } = filtrate

    // 获取数据的异步函数
    const trendListFunHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true);
        isStopPolling && setStopPolling(true);
        try {
            const { data } = await trendListHttp<TrendListType>(
                chain,
                trade.trendingPeriod,
                { page: 1, size: 100, order_by: 'creation_timestamp', direction: "desc", filters: [] }
            );
            setDataTable(data.list);
        } catch (e) {
            console.log('trendListHttp', e);
        }
        isLoading && setLoading(false);
        isStopPolling && setStopPolling(false);
    }, [chain, trade.trendingPeriod]);

    // 另一个获取数据的异步函数，用于获取最热门的交易列表
    const xTopListFunHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true);
        isStopPolling && setStopPolling(true);
        try {
            const { data } = await xTopListHttp<XtopListType>(
                chain,
                trade.xtopPeriod,
                { page: 1, size: 100, order_by: 'creation_timestamp', direction: "desc", filters: [] }
            );
            setDataTable(data.list);
        } catch (e) {
            console.log('xTopListFunHttp', e);
        }
        isLoading && setLoading(false);
        isStopPolling && setStopPolling(false);
    }, [chain, trade.xtopPeriod]);

    // 将两个请求函数组合成数组，便于切换
    const funData = useMemo(() => {
        return [trendListFunHttp, xTopListFunHttp];
    }, [trendListFunHttp, xTopListFunHttp]);
    // 初始化和时间选择变化时获取数据
    useEffect(() => {
        if (!shouldPoll) setShouldPoll(true);
        shouldPoll ? funData[funType](false, true) : funData[funType](true, true);
    }, [funData[funType]]);

    // 轮询逻辑
    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (!stopPolling && keys - 1 === funType) {
            timer = setInterval(() => {
                funData[funType](false, true);
            }, 3000);
        }
        return () => clearInterval(timer); // 清除定时器
    }, [stopPolling, funData[funType]]);

    // 渲染数据列表或加载中的骨架屏
    return <>
        {(loading && shouldPoll) ?
            <div className=" py-2 px-3 rounded-10 mx-2">
                <SkeletonCom />
            </div>
            :
            dataTable.map((item, index) => <div key={index} > <TXItem period={period as any} item={item} /></div>)
        }
    </>
}

export default memo(TrendingOrXtop);