// 导入antd库中的Modal组件
import { Modal } from "antd"
// 导入next/image库中的Image组件
import Image from "next/image"
// 导入工具函数
import { formatLargeNumber, mobileHidden, multiplyAndTruncate } from "@/utils"
// 导入SvgIcon组件
import SvgIcon from "./SvgIcon"
// 导入React相关钩子
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
// 导入MyWalletInfoType类型
import { MyWalletInfoType } from "@/interface"
// 导入图片资源
import OROGW from '@/public/OROGW.png'
import OROGD from '@/public/OROGD.png'
import snake from '@/public/snake.jpg'
// 导入主题存储
import { useThemeStore } from "@/store/theme"
// 导入国际化翻译钩子
import { useTranslations } from "next-intl"
// 导入html-to-image库中的toPng函数
import { toPng } from 'html-to-image';
// 导入ImageUrl组件
import ImageUrl from "./ImageUrl"

// 定义TokenInfoType类型
type TokenInfoType = {
    logo: string,
    symbol: string,
    main_price: number,
    total_profit_percent: number,
    total_profit: number,
    sells_num_usd: number; // 卖入数量（美元）
    buys_num_usd: number; // 买入数量（美元）
    token_price_usd: number; // 余额（USDT）
}

/**
 * Share组件用于显示分享modal框
 * @param {boolean} isModalOpen - 模态框是否可见
 * @param {Function} handleOk - 确认按钮点击事件处理函数
 * @param {boolean} isOr - 是否为特定条件，用于决定渲染哪种收益内容
 * @param {MyWalletInfoType} useInfo - 用户钱包信息
 * @param {string} period - 时间周期
 * @param {string} address - 地址
 * @param {TokenInfoType} tokenInfo - 代币信息
 * @returns {JSX.Element} - 渲染分享modal框
 */
type Props = {
    isModalOpen: boolean,
    handleOk: () => void,
    isOr: boolean,
    useInfo?: MyWalletInfoType,
    tokenInfo?: TokenInfoType,
    address: string,
    period?: string
}

const Share = ({ isModalOpen, handleOk, isOr, useInfo, period, address, tokenInfo }: Props) => {
    // 获取主题信息
    const { theme } = useThemeStore()

    // 根据条件渲染不同的收益内容
    return (
        <Modal centered={true} closable={false} width={760} open={isModalOpen} footer={null} onOk={handleOk} onCancel={handleOk}>
            <div className="dark:bg-black1F bg-fb   rounded-3xl">
                {isOr ? <AllEarnings useInfo={useInfo as any} period={period || ''} address={address} theme={theme} /> :
                    <TokenEarnings tokenInfo={tokenInfo as any} address={address} theme={theme} />}
            </div>
        </Modal>
    )
}

// 导出Share组件
export default memo(Share)
/**
 * ShareBig 组件用于在页面上渲染分享和操作的图标按钮
 * 它接受一个对象参数，该对象包含一个名为 captureRef 的 HTMLDivElement 属性
 * 主要功能包括下载 OroG.png 图片和显示可交互的操作图标
 * 
 * @param {Object} props - 组件的属性对象
 * @param {HTMLDivElement} props.captureRef - 用于捕获屏幕内容的 HTML 元素引用
 */
const ShareBig = ({ captureRef }: { captureRef: HTMLDivElement }) => {
    // 使用 useTranslations 钩子获取翻译函数，用于界面文本的国际化
    const t = useTranslations('MyWallet')

    /**
     * handleDownload 函数用于处理下载 OroG.png 的逻辑
     * 它使用 toPng 函数将 captureRef 捕获的内容转换为图片，并提供下载
     */
    const handleDownload = useCallback(async () => {
        // 检查 captureRef 是否存在，如果不存在则提前返回
        if (!(captureRef as any).current) return;
        try {
            // 将 captureRef 捕获的内容转换为 PNG 格式的 Data URL
            toPng((captureRef as any).current)
                .then((dataUrl) => {
                    // 创建一个临时的 <a> 元素来触发下载
                    const link = document.createElement('a');
                    link.download = 'OROG.png';
                    link.href = dataUrl;
                    link.click();
                })
                .catch((error) => {
                    // 处理生成图片时的错误
                    console.error('Error generating image:', error);
                });
        } catch (error) {
            // 处理生成或下载图片时的错误
            console.log('Error generating or downloading the image:', error);
        }
    }, [])

    // 定义操作数据，包含操作的文本、图标和对应的处理函数
    const data = [
        { text: 'X', img: 'x2', fun: () => { } },
        { text: 'Telegram', img: 'telegram', fun: () => { } },
        { text: t('copy'), img: 'copy', fun: () => { } },
        { text: t('download'), img: 'download', fun: handleDownload },
    ]

    // 返回一个 flex 布局的 div，内部包含根据 data 数组渲染的操作按钮
    return <div className="flex items-center justify-between">
        {data.map((item, index) => {
            // 对每个操作项渲染一个可点击的图标和文本
            return <div className="flex flex-col items-center font-semibold cursor-pointer" key={index}>
                <div onClick={() => item.fun()}
                    className="dark:text-black text-b1 dark:bg-82 bg-f1 dark:hover:bg-white dark:hover:text-black
                     hover:bg-333 hover:text-white
                 w-15 h-15  rounded-2xl bb-2  flex items-center justify-center">
                    <SvgIcon stopPropagation={false} value={item.img as any} className="w-6  cursor-pointer" />
                </div>
                <div className="text-b1 text-2xl">{item.text}</div>
            </div>
        })}
    </div>
}
/**
 * AllEarnings 组件用于显示用户在特定时期内的所有收益信息。
 * 它根据用户的资金变化计算利润，并展示利润的数值和百分比。
 * 此外，还展示了未实现的利润、总利润，以及交易次数。
 * 
 * @param {Object} props 
 * @param {MyWalletInfoType} props.useInfo - 用户钱包信息，包括资金、未实现利润、总利润等。
 * @param {string} props.period - 显示的时期信息，例如“本周”。
 * @param {"light" | "dark"} props.theme - 当前的主题，可以是“light”或“dark”。
 * @param {string} props.address - 用户的地址，用于显示部分隐藏的地址信息。
 */
const AllEarnings = ({ useInfo, period, theme, address }: { useInfo: MyWalletInfoType, period: string, theme: "light" | "dark", address: string }) => {
    // 用于获取特定元素引用的钩子
    const captureRef = useRef<HTMLDivElement>(null);
    // 使用 useTranslations 钩子获取翻译函数
    const t = useTranslations('MyWallet')

    // 使用 useMemo 钩子计算利润信息，仅当 funding 或 initial_funding 变化时重新计算
    const profit = useMemo(() => {
        const funding = useInfo?.funding || 0
        const initial_funding = useInfo?.initial_funding || 0
        const bl = funding - initial_funding
        const data1 = formatLargeNumber(funding - initial_funding || 0)
        const data2 = multiplyAndTruncate((funding - initial_funding) / initial_funding || 0, true)
        return {
            profitNum: data1,
            profitPercent: data2,
            profitBool: bl >= 0 ? true : false
        }
    }, [useInfo.funding, useInfo.initial_funding])

    // 返回 AllEarnings 组件的 JSX 结构
    return (
        <div className="pb-6 box-border ">
            <div ref={captureRef} className=" dark:bg-black1F bg-fb   rounded-10 flex  pt-22.25 pl-6.25 pr-9.75  justify-between font-semibold pb-12.75">
                <div>
                    <div className="dark:text-e7 text-51 text-3.25xl leading-none">{period}{t('realizedProfit')}</div>
                    <div className={`${profit.profitBool ? "dark:text-6f text-2bc " : "text-ff"}`}>
                        <span className="text-64 pr-1">{profit.profitBool ? '+' : '-'}${profit.profitNum}</span>
                        <span className="text-3.25xl">{profit.profitBool ? '+' : '-'}{profit.profitPercent}%</span>
                    </div>
                    <div className=" text-2xl flex items-center justify-between text-center pt-43">
                        <div className="mr-9">
                            <div className="text-90">{t('unrealizedProfits')}:</div>
                            <div className={`${useInfo.unrealized_profits >= 0 ? "dark:text-6f text-2bc " : "text-ff"}`}>{useInfo.unrealized_profits >= 0 ? '+' : '-'}${useInfo.unrealized_profits}</div>
                        </div>
                        <div className="mr-9">
                            <div className="text-90">{t('totalProfit')}:</div>
                            <div className={`${useInfo.total_profit >= 0 ? "dark:text-6f text-2bc " : "text-ff"}`}>{useInfo.unrealized_profits >= 0 ? '+' : '-'}${useInfo.total_profit}</div>
                        </div>
                        <div className="text-90 mr-9">
                            <div>{period}{t('txs')}:</div>
                            <div >
                                <span className="dark:text-6f text-2bc ">{useInfo.buy}</span>
                                /
                                <span className="text-ff">{useInfo.sell}</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="flex flex-col items-end justify-between  ">
                    <div>
                        {
                            theme === 'dark' ?
                                <Image src={OROGW} alt="" className="w-30.25" />
                                : <Image src={OROGD} alt="" className="w-30.25" />
                        }                        <div className="dark:text-white text-black text-3.25xl text-right">orog.ai</div>
                    </div>
                    <div className="flex flex-col items-center">
                        <Image src={snake} className="w-30.25 mt-5 mb-1 rounded-3xl" alt="" />
                        <div className="text-base text-90 ">{mobileHidden(address, 4, 4)}</div>
                        <div className="dark:text-white text-black text-3.25xl ">{mobileHidden(useInfo.nickname, 5, 0)}</div>

                    </div>
                </div>
            </div >
            <div className="pl-14 pr-9.75"><ShareBig captureRef={captureRef as any} /></div>
        </div>
    )
}
/**
 * TokenEarnings 组件用于显示代币收益相关信息.
 * 
 * @param {Object} props - 组件的props.
 * @param {TokenInfoType} props.tokenInfo - 代币信息对象，包含代币的各种数据.
 * @param {"light" | "dark"} props.theme - 主题模式，可以是"light"或"dark".
 * @param {string} props.address - 用户钱包地址.
 */
const TokenEarnings = ({ tokenInfo, theme, address }: { tokenInfo: TokenInfoType, theme: "light" | "dark", address: string }) => {
    // 用于获取代币logo的HTML元素引用
    const captureRef = useRef<HTMLDivElement>(null);
    // 国际化翻译钩子
    const t = useTranslations('MyWallet')
    // 用于存储logo URL的状态
    const [logoUrl, setLogoUrl] = useState('')

    /**
     * 获取代币logo URL.
     */
    const getLogoUrl = async () => {
        try {
            const res = await fetch(tokenInfo.logo)
            if (!res.ok) {
                throw new Error(`HTTP error! Status: ${res.status}`);
            }
            const data = await res.json()
            data.image && setLogoUrl(data.image)
        } catch (e) {
            console.log(e)
        }
    }

    // 在组件挂载时获取logo URL
    useEffect(() => {
        getLogoUrl()
    }, [])

    // 渲染组件
    return <div className="pb-6">
        <div ref={captureRef} className="rounded-10  dark:bg-black1F bg-fb flex justify-between font-semibold pt-13 pl-13.25 pr-10.75 pb-11.5">
            <div>
                <div className="flex items-center">
                    <ImageUrl className="w-15.25 mr-7.5 h-15.25" logo={tokenInfo.logo} symbol={tokenInfo.symbol} />
                    <div>
                        <div className="dark:text-white text-black text-40">${tokenInfo.symbol}</div>
                        {/* <div className="dark:text-82 text-51 text-xl">WANGmingming</div> */}
                    </div>
                </div>
                <div className={`${tokenInfo.total_profit_percent >= 0 ? "dark:text-6f text-2bc" : "text-ff"} pb-7.5`}>
                    <div className="text-64 pr-1">{tokenInfo.total_profit_percent >= 0 ? '+' : '-'}{Math.abs(tokenInfo.total_profit_percent)}%</div>
                    <div className="text-2xl">{tokenInfo.total_profit_percent >= 0 ? '+' : '-'}{(tokenInfo.total_profit) / tokenInfo.main_price}SOL({tokenInfo.total_profit_percent >= 0 ? '+' : '-'}${tokenInfo.total_profit})</div>
                </div>
                <div className="dark:text-82 text-51 text-2xl">
                    <div className="pb-3.75">{t('hold')}:  ${tokenInfo.token_price_usd}</div>
                    <div className="pb-3.75">{t('sold')}:  ${tokenInfo.sells_num_usd}</div>
                    <div className="pb-3.75">{t('bought')}:  ${tokenInfo.buys_num_usd}</div>
                </div>
            </div>
            <div className="flex flex-col items-center justify-between">
                <div>
                    {
                        theme === 'dark' ?
                            <Image src={OROGW} alt="" className="w-30.25" />
                            : <Image src={OROGD} alt="" className="w-30.25" />
                    }
                    <div className="dark:text-white text-black text-3.25xl ">orog.ai</div>
                </div>
                <div className="flex flex-col items-end">
                    <Image src={snake} className="w-15.25 mt-5 mb-1 rounded-xl" alt="" />
                    <div className="text-base text-90 ">{mobileHidden(address, 4, 4)}</div>
                </div>
            </div>
        </div>
        <div className="pl-13.25 pr-10.75"><ShareBig captureRef={captureRef as any} /></div>
    </div>
}