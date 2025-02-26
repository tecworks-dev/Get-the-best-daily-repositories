import { ResolutionString, widget } from '@/public/charting_library'
import { Fragment, memo, useEffect, useRef, useState } from 'react';
import getDatafeed, { timeArr } from './datafeed'
import { disabled_features, enabled_features } from './features';
import { formatLargeNumber } from '@/utils';
import { historicalTickerCandleHttp } from "@/request/index";
import { historicalTickerCandleListType } from '@/interface';
import { useThemeStore } from '@/store/theme';
import useUpdateEffect from '@/hook/useUpdateEffect';
import { LoadingOutlined } from '@ant-design/icons'

    ;
import { useParams } from 'next/navigation';
import { LocalesType } from '@/i18n/routing';
function Tradingview({ address, getInterval }: { address: string, getInterval: Function }) {
    const { theme } = useThemeStore()
    const { locale }: { locale: LocalesType } = useParams()
    const containerRef = useRef<HTMLDivElement>(null); // 引用图表容器
    // 获取历史数据
    const historicalTickerCandleHttpFun = async (requireData: any) => {
        const { data } = await historicalTickerCandleHttp<historicalTickerCandleListType>(address, requireData)
        setLoading(false)
        return data.items
    }
    const rs = useRef<any>(null)
    // 获取k线图时间
    const socketFun = (interval: string, onRealtimeCallback: Function) => {
        getInterval(interval, onRealtimeCallback)
    }
    const [loading, setLoading] = useState(true)
    useEffect(() => {
        if (containerRef.current) {
            setLoading(true)
            rs.current = new widget({
                container: containerRef.current, // 容器
                locale: locale, // 语言
                library_path: '/charting_library/', // 库的路径
                datafeed: getDatafeed(historicalTickerCandleHttpFun, socketFun) as any,
                symbol: 'OROG', // 初始符号
                interval: (localStorage.getItem('tradingview.chart.lastUsedTimeBasedResolution') || '5') as ResolutionString, // 初始的间隔时间
                fullscreen: false,  // 是否全屏
                // debug: true,  // debug
                theme: theme, // 颜色模式
                timezone: "Asia/Shanghai",
                overrides: {
                    "volumePaneSize": "medium",
                },
                settings_overrides: {
                    "paneProperties.background": theme === 'dark' ? "rgba(0,0,0,0.5)" : "rgba(255,255,255,0.5)",
                    'paneProperties.backgroundType': "solid",
                    "paneProperties.vertGridProperties.color": theme !== 'dark' ? "rgba(0,0,0,0.1)" : "rgba(255,255,255,0.1)",
                    "paneProperties.horzGridProperties.color": theme !== 'dark' ? "rgba(0,0,0,0.1)" : "rgba(255,255,255,0.1)",
                    "scalesProperties.lineColor": "rgba(19, 23, 34,0.5)",

                },
                auto_save_delay: 0.1,
                custom_css_url: "./night.css",
                width: containerRef.current.getBoundingClientRect().width, // 宽
                height: containerRef.current.getBoundingClientRect().height, // 高
                autosize: true, // 是否自适应宽高
                disabled_features: disabled_features, // 禁用
                enabled_features: enabled_features,// 显示
                favorites: {   // 用于指定哪些功能（如时间周期、指标、绘图工具、图表类型）应该默认标记为“收藏夹”。收藏夹中的功能会在用户界面中优先显示，便于快速访问。
                    intervals: timeArr.map(item => item.label) as ResolutionString[],
                    indicators: ["EMA Cross", "Relative Strength Index"],
                },
                charts_storage_api_version: "1.1",

                custom_formatters: {
                    // 自定义价格格式化工厂函数
                    priceFormatterFactory: (e: any, t: any) => ({
                        // 定义格式化方法，将价格值格式化为自定义样式
                        format: (value: any) => formatLargeNumber(value, undefined, 6), // 使用前导零格式化价格
                    }),
                    // 自定义交易量格式化工厂函数
                    studyFormatterFactory: (e: any, t: any) => {
                        return {
                            // 定义格式化方法，将交易量值格式化为自定义样式
                            format: (value: any) => {
                                return formatLargeNumber(value) // 使用前导零格式化交易量
                            }
                        }
                    },
                } as any
            })

        }
        return () => {
            if (rs.current) {
                rs.current.remove()
            }
        }
    }, [address])
    useUpdateEffect(() => {
        if (rs.current) {
            rs.current.changeTheme(theme)
        }
    }, [theme])
    return (
        <Fragment>
            <div className=' flex items-center js w-full h-full min-w-full min-h-full relative'>
                <div id="tv_chart_container" className={` w-full h-full overflow-hidden`} ref={containerRef}></div>
                <div className={`absolute w-full h-full z-50 flex items-center justify-center dark:bg-black bg-white ${loading ? 'block' : 'hidden'}`}>
                    <LoadingOutlined className={`text-3xl dark:text-2bc text-2bc ${loading ? 'block' : 'hidden'}`} />
                </div>
            </div>
        </Fragment>
    )
}

export default memo(Tradingview)

