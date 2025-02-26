'use client'
import { formatLargeNumber, mobileHidden } from "@/utils"
import SvgIcon from "@/component/SvgIcon"
import { memo, useEffect, useState } from "react"
import Copy from "@/component/Copy"
import { ChainsType, Link } from "@/i18n/routing"
import { chainMain } from "@/app/chainLogo"
import { XtopPlateTokenType, XtopPlateType } from "@/interface"
import { useTranslations } from "next-intl"

// 定义一个名为XTop的组件，用于展示XTop数据
// 参数:
// - xtopData: XtopPlate类型的数据，包含需要展示的XTop信息
// - chain: ChainsType类型，表示链的信息
// - plate: 可能为null的字符串，表示板块信息
// - period: 字符串，表示周期信息
const XTop = ({ xtopData, chain, plate, period }: { xtopData: XtopPlateType, chain: ChainsType, plate: string | null, period: string }) => {
    // 定义三个状态变量，用于控制三个不同板块的显示与隐藏
    const [one, setOne] = useState(false)
    const [two, setTwo] = useState(false)
    const [three, setThree] = useState(false)

    // 根据plate参数的值，决定显示哪个板块的信息
    useEffect(() => {
        const plateNum = Number(plate)
        if (!plateNum) return
        if (plateNum < 4) { setOne(true); setTwo(false); setThree(false); }
        else if (plateNum > 3 && plateNum < 7) { setTwo(true); setOne(false); setThree(false) }
        else if (plateNum > 6) { setThree(true); setOne(false); setTwo(false) }
    }, [plate])
    return (
        <div className="mt-3.75 font-semibold transition-all">
            <div className="flex items-center custom830:flex-col ">
                {xtopData.slice(0, 3).map((item, index) => (
                    <div key={index}
                        className={`w-1/3 cursor-pointer  rounded-t-10  box-border  dark:bg-3b bg-f5 custom830:w-full`}>
                        <div onClick={() => setOne(!one)} className="h-20 flex items-center   justify-between px-5">
                            <div className={`text-4xl ${(index === 0) && 'text-fb'} ${(index === 1) && 'text-fc'} ${(index === 2) && 'text-fd'}`}
                            >
                                {!index && '👑 '}{item.plate}
                            </div>
                            <div className="flex items-center">
                                <SvgIcon className={`w-7.5 ${item.plate_trend ? 'dark:text-6f text-2bc' : 'text-ff rotate-180'} `} value='top' />
                            </div>
                        </div>
                        <div className={`w-full duration-500 border-r border-6b ${one ? 'h-118.75' : 'h-0'} dark:bg-26 bg-white overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-gray-400 `}>
                            {item.data.map((value, index) => (
                                <Item plate={item.plate} period={period} tokenData={value} chain={chain} key={index} index={index} />
                            ))}
                        </div>
                    </div>
                ))}
            </div>
            <div className="flex items-center custom830:flex-col">
                {xtopData.slice(3, 6).map((item, index) => (
                    <div key={index}
                        className={`w-1/3 cursor-pointer  rounded-t-10  box-border bg-fb  dark:bg-51 custom830:w-full`}>
                        <div onClick={() => setTwo(!two)} className="h-20 flex items-center   justify-between px-5">
                            <div className={`text-4xl text-ffb`}
                            >
                                {item.plate}
                            </div>
                            <div className="flex items-center">
                                <SvgIcon className={`w-7.5 ${item.plate_trend ? 'dark:text-6f text-2bc' : 'text-ff rotate-180'} `} value='top' />
                            </div>
                        </div>
                        <div className={`w-full duration-500 border-r border-6b ${two ? 'h-118.75' : 'h-0'} dark:bg-26 bg-white overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-gray-400 `}>
                            {item.data.map((value, index) => (
                                <Item plate={item.plate} period={period} tokenData={value} chain={chain} key={index} index={index} />
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div className="flex items-center custom830:flex-col">
                {xtopData.slice(6, 9).map((item, index) => (
                    <div key={index}
                        className={`w-1/3 cursor-pointer  rounded-t-10  box-border  bg-white dark:bg-28 custom830:w-full`}>
                        <div onClick={() => setThree(!three)} className="h-20 flex items-center   justify-between px-5">
                            <div className={`text-4xl text-bf`}
                            >
                                {item.plate}
                            </div>
                            <div className="flex items-center">
                                <SvgIcon className={`w-7.5 ${item.plate_trend ? 'dark:text-6f text-2bc' : 'text-ff rotate-180'} `} value='top' />
                            </div>
                        </div>
                        <div className={`w-full duration-500 border-r border-6b ${three ? 'h-118.75' : 'h-0'} dark:bg-26 bg-white overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-gray-400 `}>
                            {item.data.map((value, index) => (
                                <Item plate={item.plate} period={period} tokenData={value} chain={chain} key={index} index={index} />
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}

export default memo(XTop)

// 定义一个名为Item的组件，用于展示单个XTop项目的信息
// 参数:
// - index: 数字，表示项目的索引
// - chain: ChainsType类型，表示链的信息
// - period: 字符串，表示周期信息
// - plate: 字符串，表示板块信息
// - tokenData: XtopPlateTokenType类型，包含单个XTop项目的信息
const Item = ({ index, chain, period, plate, tokenData }: { index: number, plate: string, chain: ChainsType, period: string, tokenData: XtopPlateTokenType }) => {
    const t = useTranslations('Xtop')
    return <div className="box px-4.25 py-3.25 pb-2  border-b border-6b">
        <Link href={`/trade/${tokenData.address}`} className="flex items-center justify-between mb-1  ">
            <div className="flex items-center">
                <img src={tokenData.logo} alt="" className="w-20  mr-3.75 rounded-full overflow-hidden" />
                <div>
                    <div><span className="text-2.75xl text-black dark:text-white pr-2">{tokenData.symbol}</span><span className="text-lg text-6b">{plate}</span></div>
                    <div className="flex items-center text-6b text-15 leading-none">
                        <span>{mobileHidden(tokenData.quote_mint_address, 4, 4)}</span>
                        <Copy className="w-3" text={tokenData.quote_mint_address} />
                        {tokenData.website && <SvgIcon onClick={() => window.open(tokenData.website)} className="w-3 ml-1.75 " value="website" />}
                        {tokenData.twitter_username && <SvgIcon className="w-3 ml-1.75" onClick={() => window.open(`https://x.com/${tokenData.twitter_username}`)} value="x" />}
                        {tokenData.telegram && <SvgIcon className="w-3.25 ml-1.75 " onClick={() => window.open(`https://t.me/${tokenData.telegram}`)} value="telegram" />}
                    </div>
                </div>
            </div>
            <div className="flex items-center">
                <SvgIcon className={`w-4.75 ${index % 2 ? 'dark:text-6f text-2bc' : ' text-ff rotate-180'}  mr-2`} value="rise" />
                <span className="text-black dark:text-white text-4xl">ON.{index + 1}</span>
            </div>
        </Link>
        <div className="text-13 text-6b flex items-center">
            <div className="mr-3.75">{period}</div>
            <div className="flex items-center mr-4.25">
                <SvgIcon className="w-3.25 mr-2" value="crew" />
                <span className="text-98" >{formatLargeNumber(tokenData.price)}</span>
            </div>
            <div className="mr-4.25">{t('mc')} <span className="text-fb">${formatLargeNumber(tokenData.market_cap)}</span></div>
            <div className="mr-4.75">{t('vol')} <span className="text-fb">{formatLargeNumber(tokenData.volume)}</span></div>
            <div>{t('liq')}/{t('initial')} <span className="dark:text-6f text-2bc">{formatLargeNumber(tokenData.liquidity)}/{formatLargeNumber(tokenData.initial_liquidity)}{chainMain[chain]}</span></div>
        </div>
    </div >
}