import { useEffect } from 'react';

import { create } from 'zustand';

// 定义筛选条件的类型
type MemeFiltrate = {
    pumpOrRaydium: string; // 用于选择 Pump 或 Raydium 的字符串
    check: string; // 筛选的数组字符串
    direction: "desc" | "asc"; // 排序方向，支持升序或降序
    condition: string; // 筛选的条件，如年龄
};
type TrendFiltrate = {
    period: string; // 时间
    direction: "desc" | "asc"; // 排序方向，支持升序或降序
    condition: string; // 筛选的条件，如年龄
};
type XtopFiltrate = {
    period: string; // 时间
    direction: "desc" | "asc"; // 排序方向，支持升序或降序
    condition: string; // 筛选的条件，如年龄
};
type FollowFiltrate = {
    period: string; // 时间
    direction: "desc" | "asc"; // 排序方向，支持升序或降序
    condition: string; // 筛选的条件，如年龄
};
type MyWalletType = {
    period: string; // 时间
    direction: "desc" | "asc"; // 排序方向，支持升序或降序
    condition: string; // 筛选的条件，如年龄
}
type TradeType = {
    trendingPeriod: string;  // trending时间
    xtopPeriod: string;  // xtop时间
    followPeriod: string; // follow时间
    transaction: number; // 交易选择
    trendingOrXtop: string, // xtop 和 trending 选择
    followOrHolding: string // 跟随和持有选择
    antiPinch: boolean, // 防夹
    auto: boolean, // 自动
    pointSlip: string, // 滑点
    autoNum: number, // 自定义滑点
    gasSelected: number, // gas 选择
}
// 定义的筛选类型
type FiltrateType = {
    meme: MemeFiltrate; // Meme 筛选条件
    trend: TrendFiltrate;
    xtop: XtopFiltrate;
    follow: FollowFiltrate;
    mywallet: MyWalletType;
    trade: TradeType
};

// 定义筛选接口
interface FiltrateStore {
    filtrate: FiltrateType; // 当前的筛选条件
    initFiltrate: (filtrate: FiltrateType) => void; // 初始化筛选条件的方法
    setFiltrate: (filtrate: FiltrateType) => void; // 设置筛选条件的方法
}

// 默认的初始值
const defaultFiltrate: FiltrateType = {
    meme: {
        pumpOrRaydium: "Raydium", // 默认选择的 Pump/Raydium
        check: "new_pair", // 默认筛选数组
        direction: "desc", // 默认排序方向为降序
        condition: "created_timestamp", // 默认筛选条件为年龄
    },
    trend: {
        period: '1h', // 默认筛选周期为 1 小时
        direction: "desc", // 默认排序方向为降序
        condition: "created_timestamp", // 默认筛选条件为年龄
    },
    xtop: {
        period: "1d", // 默认筛选周期为 1 天
        direction: "desc", // 默认排序方向为降序
        condition: "created_timestamp", // 默认筛选条件为年龄
    },
    follow: {
        period: "1h", // 默认筛选周期为 1 天
        direction: "desc", // 默认排序方向为降序
        condition: "created_timestamp", // 默认筛选条件为年龄
    },
    mywallet: {
        period: "7d", // 默认筛选周期为 7 天
        direction: "desc", // 默认排序方向为降序
        condition: "enter_address", // 默认筛选条件为年龄
    },
    trade: {
        trendingPeriod: "1h", //  默认筛选周期为 1 小时
        xtopPeriod: "1d",
        followPeriod: "1h",
        transaction: 0, // 交易选择
        trendingOrXtop: "1", // xtop和trending选择
        followOrHolding: "1", // 跟随和持有选择
        antiPinch: true, // 防抖
        auto: true,
        pointSlip: '10',
        autoNum: 0,
        gasSelected: 0,
    }
};

// 创建 Zustand 状态管理的 store
export const useFiltrateStore = create<FiltrateStore>((set) => {
    let filtrate: FiltrateType = defaultFiltrate
    if (typeof window !== 'undefined') {
        filtrate = JSON.parse(localStorage.getItem('filtrateStore') || JSON.stringify(filtrate))
        // const { meme, trend, xtop, follow, mywallet, trade } = filtrate
        // const { trade: intTrade, meme: intMeme, trend: intTrend, xtop: intXtop, follow: intFollow, mywallet: intMyWallet } = defaultFiltrate
        // if (!meme) filtrate = { ...filtrate, meme: intMeme }
        // if (!trend) filtrate = { ...filtrate, trend: intTrend }
        // if (!xtop) filtrate = { ...filtrate, xtop: intXtop }
        // if (!follow) filtrate = { ...filtrate, follow: intFollow }
        // if (!mywallet) filtrate = { ...filtrate, mywallet: intMyWallet }
        // if (!trade) filtrate = { ...filtrate, trade: intTrade }
        const mergeWithDefaults = <T extends object>(defaults: T, stored: Partial<T>): T => {
            return Object.keys(defaults).reduce((acc, key) => {
                const defaultValue = defaults[key as keyof T];
                const storedValue = stored[key as keyof T];

                // 递归合并对象
                if (typeof defaultValue === "object" && defaultValue !== null && !Array.isArray(defaultValue)) {
                    acc[key as keyof T] = mergeWithDefaults(defaultValue, storedValue || {});
                } else {
                    acc[key as keyof T] = storedValue !== undefined ? storedValue : defaultValue;
                }

                return acc;
            }, {} as T);
        };
        filtrate = mergeWithDefaults(defaultFiltrate, filtrate);
    }

    return {
        filtrate: filtrate, // 加载当前的筛选条件
        initFiltrate: (filtrate) => set({ filtrate }), // 初始化筛选条件
        setFiltrate: (filtrate) => {
            set({ filtrate }); // 更新筛选条件
            // 确保在浏览器环境中运行
            if (typeof window !== "undefined") {
                localStorage.setItem("filtrateStore", JSON.stringify(filtrate)); // 将筛选条件保存到 localStorage
            }
        },
    };
});
export const useEffectFiltrateStore = (fun: (filtrateLocal: FiltrateType) => void) => {
    const { filtrate } = useFiltrateStore()
    useEffect(() => {
        filtrate && fun(filtrate)
    }, [])
}