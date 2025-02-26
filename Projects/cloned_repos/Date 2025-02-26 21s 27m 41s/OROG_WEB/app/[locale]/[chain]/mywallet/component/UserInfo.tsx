'use client'
import { Input } from "antd";
import Chart from "./Chart";
import Image from "next/image";
import snake from "@/public/snake.jpg";
import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { MyWalletInfoType } from "@/interface";
import { formatLargeNumber, mobileHidden, multiplyAndTruncate } from "@/utils";
import { useEffectFiltrateStore, useFiltrateStore } from "@/store/filtrate";
import { useTranslations } from "next-intl";
import { getUserInfoHttp } from "@/request";
import { useTokenStore } from "@/store/tokne";
import dynamic from "next/dynamic";
const SvgIcon = dynamic(() => import('@/component/SvgIcon')); 
const Copy = dynamic(() => import('@/component/Copy')); 
const Share = dynamic(() => import('@/component/Share')); 

// 时间筛选选项
const timeSelect = [
    { key: '7d', text: "7D", length: 7 },
    { key: '30d', text: "30D", length: 30 }
];

// 用户信息组件
const UserInfo = ({ address }: { address: string }) => {
    const t = useTranslations('MyWallet'); // 获取国际化翻译方法
    const { token } = useTokenStore()
    const [isModalOpen, setIsModalOpen] = useState(false); // 控制分享模态框状态
    const [period, setPeriod] = useState(''); // 当前选中的时间段
    const [isName, setISName] = useState(false); // 控制是否显示昵称输入框
    const [name, setName] = useState(''); // 用户输入的新昵称
    const [loading, setLoading] = useState(false); // 控制刷新按钮的加载状态
    const { filtrate, setFiltrate } = useFiltrateStore(); // 筛选状态和方法
    const { mywallet } = filtrate; // 从筛选状态中获取钱包信息
    // 更新选中时间段时的处理
    useEffectFiltrateStore(({ mywallet }) => {
        setPeriod(mywallet.period);
    });

    // 用户信息状态
    const [useInfo, setUserInfo] = useState<MyWalletInfoType>({
        nickname: "", // 用户昵称
        initial_funding: 0, // 初始资金
        funding: 0, // 当前资金
        money_per_day: [], // 每日资金变化
        unrealized_profits: 0, // 未实现盈亏
        total_profit: 0, // 总盈亏
        buy: 0, // 买入次数
        sell: 0, // 卖出次数
    });

    // 处理昵称输入回车事件
    // const onPressEnter = async () => {
    //     setISName(false);
    //     try {
    //         await updateNickNameHttp({ name: name })
    //         getUserInfo()
    //     } catch (e) {
    //         console.log('updateNickNameHttp', e);
    //     }
    //     setName('')
    // };

    // 更新时间段
    const getPeriod = (value: string) => {
        setPeriod(value);
        setFiltrate({
            ...filtrate,
            mywallet: {
                ...filtrate.mywallet,
                period: value,
            },
        });
    };

    // 获取用户信息
    const getUserInfo = useCallback(async (isLoading = false) => {
        if (!address) return;
        isLoading && setLoading(true);
        try {
            const { data } = await getUserInfoHttp<MyWalletInfoType>(address, mywallet.period);
            setUserInfo(data)
        } catch (e) {
            console.log('getUserInfo', e);
        }
        isLoading && setTimeout(() => {
            setLoading(false)
        }, 1000)
    }, [address, mywallet.period]);

    // 获取当前选中时间段的文本
    const getTime = useMemo(() => {
        const data = timeSelect.find((item) => item.key === period);
        return data?.text || '';
    }, [period]);

    // 计算盈利数据
    const profit = useMemo(() => {
        const funding = useInfo?.funding || 0;
        const initial_funding = useInfo?.initial_funding || 0;
        const bl = funding - initial_funding; // 盈亏值
        const data1 = formatLargeNumber(bl); // 格式化盈亏值
        const data2 = multiplyAndTruncate((bl / initial_funding) || 0, true); // 盈亏百分比
        return {
            profitNum: data1,
            profitPercent: data2,
            profitBool: bl >= 0, // 盈亏状态
        };
    }, [useInfo.funding, useInfo.initial_funding]);

    // 初始化时获取用户信息
    useEffect(() => {
        getUserInfo();
    }, [getUserInfo]);

    return (
        <div className="pt-9.25 pl-9.5 pr-8.25 pb-8.75 dark:bg-21 bg-white rounded-30 ">
            <div className="flex items-start justify-between">
                {/* 用户信息部分 */}
                <div className="flex items-center">
                    <Image src={snake} className="w-30.25 mr-7 rounded-3xl" alt="用户头像" />
                    <div className="font-semibold">
                        <div className="flex items-end mb-1">
                            {isName ? (
                                <Input
                                    value={name}
                                    // onPressEnter={onPressEnter}
                                    // onChange={(e) => setName(e.target.value)}
                                    className="box-border w-56 h-11.25 rounded-5
                                    dark:bg-black dark:hover:bg-black dark:focus-within:bg-black dark:border-black dark:hover:border-white dark:focus-within:border-white dark:text-white
                                    bg-f1 hover:bg-f1 focus-within:bg-f1 border-f1 hover:border-black focus-within:border-black text-black shadow-none
                                    "
                                />
                            ) : (
                                <div className="dark:text-white text-black text-4xl">
                                    {mobileHidden(useInfo?.nickname, 5, 0)}
                                </div>
                            )}
                            {/* <SvgIcon
                                onClick={() => setISName(!isName)}
                                value="pencil"
                                className="w-11.25 ml-3 text-67 mr-2.5 cursor-pointer"
                            /> */}
                            <SvgIcon
                                onClick={() => getUserInfo(true)}
                                value="refresh"
                                className={`w-6.75 ml-3 text-67 cursor-pointer ${loading ? 'animate-spin' : ''}`}
                            />
                        </div>
                        <div className="flex items-center mb-3.75">
                            <span className="mr-0.75 text-base text-90 ">
                                <span className="custom830:hidden">{address}</span>
                                <span className="hidden custom830:block">{mobileHidden(address)}</span>
                            </span>
                            <Copy className="ml-2.5 cursor-pointer w-4.25 text-90" text={address} />
                        </div>
                        <div className="flex items-center justify-center text-82 font-normal w-32 text-base box-border py-2 rounded-10 dark:bg-gray4b bg-f1 custom830:hidden">
                            {timeSelect.map((item, index) => (
                                <div
                                    key={index}
                                    onClick={() => getPeriod(item.key)}
                                    className={`flex-1 text-center border-r-2 border-82 last:border-none cursor-pointer ${period === item.key ? 'dark:text-white text-black' : ''}`}
                                >
                                    {item.key}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* 分享图标 */}
                <div className="flex items-center">
                    <SvgIcon
                        value="share"
                        onClick={() => setIsModalOpen(true)}
                        className="w-9 text-6b cursor-pointer mr-9"
                    />
                </div>
            </div>

            {/* 盈亏和图表 */}
            <div className="flex items-start justify-between text-center mt-5.25 custom830:flex-col custom830:items-center">
                <div className="w-full items-start justify-between text-left custom830:flex  ">
                    <div>
                        <div className="text-6f6f text-2xl font-semibold">
                            {t('last')} {getTime} {t('pnl')}
                        </div>
                        <div className={`${profit.profitBool ? 'text-2bc dark:text-6f' : 'text-ff'} text-64`}>
                            {profit.profitBool ? '＋' : '-'}{`${profit.profitPercent}%`}
                        </div>
                        <div className={`${profit.profitBool ? 'text-2bc dark:text-6f' : 'text-ff'}`}>
                            {profit.profitBool ? '＋' : '-'}{profit.profitNum} USDT
                        </div>
                    </div>
                    <div className=" items-center justify-center text-82 font-normal w-32 text-base box-border py-2 rounded-10 dark:bg-gray4b bg-f1 hidden custom830:flex">
                        {timeSelect.map((item, index) => (
                            <div
                                key={index}
                                onClick={() => getPeriod(item.key)}
                                className={`flex-1 text-center border-r-2 border-82 last:border-none cursor-pointer ${period === item.key ? 'dark:text-white text-black' : ''}`}
                            >
                                {item.key}
                            </div>
                        ))}
                    </div>
                </div>
                <div className="">
                    <Chart dataValues={useInfo.money_per_day || []} />
                </div>
            </div>

            {/* 分享模态框 */}
            <Share
                address={address}
                isOr={true}
                period={getTime}
                useInfo={useInfo}
                isModalOpen={isModalOpen}
                handleOk={() => setIsModalOpen(false)}
            />
        </div>
    );
};

export default memo(UserInfo);