import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from 'react';

import { ChainsType } from '@/i18n/routing';
import {
  EarningsListType,
  FollowListParams,
  FollowListType,
} from '@/interface';
import { getFollowListHttp } from '@/request';
import { useFiltrateStore } from '@/store/filtrate';

import FHItem from './FHItem';
import SkeletonCom from './Skeleton';

/**
 * 关注或持仓组件，展示关注列表或收益列表
 * @param {ChainsType} chain 链类型
 * @param {number} funType 功能类型，0为关注列表，1为收益列表
 */
const FollowOrHolding = ({ chain, funType, period, keys }: { chain: ChainsType, period: string, funType: number, keys: number }) => {
    const [loading, setLoading] = useState(true); // 是否加载中
    const [stopPolling, setStopPolling] = useState(true); // 是否停止轮询
    const [shouldPoll, setShouldPoll] = useState(false); // 是否初始化过
    const [dataTable, setDataTable] = useState<FollowListParams[] | EarningsListType[]>([]); // 表格中显示的数据
    const { filtrate } = useFiltrateStore(); // 筛选条件的状态管理
    const { trade } = filtrate

    // 获取数据的异步函数
    const followFunHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true);
        isStopPolling && setStopPolling(true);
        try {
            const { data } = await getFollowListHttp<FollowListType>(
                chain,
                trade.followPeriod,
                { page: 1, size: 100, order_by: 'pool_creation_timestamp', direction: "desc", filters: [] }
            );
            setDataTable(data.list);
        } catch (e) {
            console.log('followFunHttp', e);
        }
        isLoading && setLoading(false);
        isStopPolling && setStopPolling(false);
    }, [chain, trade.followPeriod]);

    // 另一个获取数据的异步函数，用于获取最热门的交易列表
    const earningFunHttp = useCallback(async (isLoading: boolean, isStopPolling: boolean) => {
        isLoading && setLoading(true);
        isStopPolling && setStopPolling(true);
        try {
            // const { data } = await trendListHttp<EarningsListType>(
            //     chain,
            //     trade.period,
            //     { page: 1, size: 100, order_by: 'pool_creation_timestamp', direction: "desc", filters: [] }
            // );
            // setDataTable(data.list);
        } catch (e) {
            console.log('earningFunHttp', e);
        }
        isLoading && setLoading(false);
        isStopPolling && setStopPolling(false);
    }, [chain]);

    // 将两个请求函数组合成数组，便于切换
    const funData = useMemo(() => {
        return [followFunHttp, earningFunHttp];
    }, [followFunHttp, earningFunHttp]);

    // 初始化和时间选择变化时获取数据,refresh变化的时候会重新调用
    useEffect(() => {
        if (!shouldPoll) setShouldPoll(true);
        shouldPoll ? funData[funType](false, true) : funData[funType](true, true);
    }, [funData[funType]]);

    // 轮询逻辑
    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (!stopPolling && keys - 1) {
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
            dataTable.map((item, index) => <div key={index}> <FHItem period={period as any} item={item} /></div>)
        }
    </>
}

export default memo(FollowOrHolding)