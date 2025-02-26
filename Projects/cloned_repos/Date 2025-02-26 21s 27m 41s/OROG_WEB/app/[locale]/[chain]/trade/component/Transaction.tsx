import 'rc-slider/assets/index.css';

import {
  memo,
  useEffect,
  useMemo,
  useState,
} from 'react';

import {
  Button,
  ConfigProvider,
  InputNumber,
  InputNumberProps,
  Segmented,
  Switch,
} from 'antd';
import {
  formatUnits,
  parseUnits,
} from 'ethers';
import { useTranslations } from 'next-intl';
import dynamic from 'next/dynamic';
import Slider from 'rc-slider';

import {
  chainMain,
  precision,
} from '@/app/chainLogo';
import useGetBalance from '@/hook/useGetBlance/useGetBalance';
import useGetTokenBalance from '@/hook/useGetTokenBlance/useGetTokenBalance';
import useLogin from '@/hook/useLogin';
import useTransaction from '@/hook/useTransaction/useTransaction';
import { ChainsType } from '@/i18n/routing';
import {
  useEffectFiltrateStore,
  useFiltrateStore,
} from '@/store/filtrate';
import useConnectWallet from '@/store/wallet/useConnectWallet';
import {
  divideDecimal,
  multiplyDecimal,
} from '@/utils';

const SvgIcon = dynamic(() => import('@/component/SvgIcon'));
const TableTooltip = dynamic(() => import('@/component/TableTooltip'));
const FormatUnits = dynamic(() => import('@/component/FormatUnits'));
// 定义滑动条的标记，表示滑动条的刻度值，键为滑动位置（百分比），值为显示的标签
const marks = {
    0: '0%',
    25: '25%',
    50: '50%',
    75: '75%',
    100: '100%'
};
const wSol = "So11111111111111111111111111111111111111112"
/**
 * 交易组件
 * @param {Object} props - 组件的属性
 * @param {ChainsType} props.chain - 当前链的类型
 */
const Transaction = ({ chain, quoteMintAddress, decimals, quoteSymbol }: { chain: ChainsType, quoteMintAddress: string, decimals: number, quoteSymbol: string }) => {
    // 国际化翻译钩子，用于获取多语言翻译的函数
    const t = useTranslations('Trade');
    // 获取余额
    const { balance, getBalanceLod, getBalance } = useGetBalance(chain)
    // address
    const { address } = useConnectWallet(chain)
    // 获取token余额
    const { tokenBalance, getTokenBalanceLod, getTokenBalance } = useGetTokenBalance(chain, quoteMintAddress)
    // 连接钱包提示
    const { checkExecution } = useLogin(chain)
    // 交易类型的索引状态，表示当前选中的交易类型
    const [transaction, setTransaction] = useState(0);

    // 输入框的数量值状态，表示用户输入的交易数量
    const [value, setValue] = useState('');

    // 滑动条的值状态，表示滑动条当前所选的百分比值
    const [value2, setValue2] = useState(0);

    // 设置面板开关状态，表示是否显示设置面板
    const [openSetting, setOpenSetting] = useState(false);

    // 防夹开关状态，表示防夹功能是否开启
    const [antiPinch, setAntiPinch] = useState(false);

    // 自动滑点开关状态，表示滑点是否处于自动模式
    const [auto, setAuto] = useState(false);

    // 手动滑点值状态，表示手动设置的滑点百分比
    const [pointSlip, setPointSlip] = useState('10');

    // 自动滑点数值状态，表示在自动模式下的滑点值
    const [autoNum, setAutoNum] = useState(0);

    // Gas费用值状态，表示当前的基础Gas费用
    const [gas, setGas] = useState('0.0000001');

    // 使用自定义的筛选状态管理器，获取当前的筛选条件和更新函数
    const { filtrate, setFiltrate } = useFiltrateStore();

    // 选中的Gas费用索引状态，表示当前选中的Gas费用选项
    const [gasSelected, setGasSelected] = useState(0);
    // 
    const { transactionFun, transactionLoading } = useTransaction(chain, () => {
        getBalance()
        getTokenBalance()
    }, () => {
        getBalance()
        getTokenBalance()
    })
    // 初始化筛选条件时，根据store中的值更新组件状态
    useEffectFiltrateStore(({ trade }) => {
        setTransaction(trade.transaction);
        setAntiPinch(trade.antiPinch);
        setAuto(trade.auto);
        setAutoNum(trade.autoNum);
        setPointSlip(trade.pointSlip);
        setGasSelected(trade.gasSelected);
    });
    // 交易函数
    const transactionFunAll = () => {
        // transaction 0 是买 1 是卖
        const data = transaction ? { quoteMintAddress: wSol, baseMintAddress: quoteMintAddress, rawAmountIn: Number(value), slippage: Number(pointSlip) } :
            { quoteMintAddress, baseMintAddress: wSol, rawAmountIn: Number(value), slippage: Number(pointSlip) }
        transactionFun(data)
    }
    /**
     * 处理数量输入框变化的函数
     * @param {number} value - 用户输入的交易数量值
     */
    const onChange: InputNumberProps['onChange'] = (value) => {
        const va = value as string || ''
        setValue(va);
        // 算出滚动条的百分比
        const int = transaction ? decimals : precision[chain]
        const num = formatUnits(transaction ? tokenBalance : balance, int)
        const data = divideDecimal(va || 0, num, false)
        setValue2(Number(data) * 100)
    };

    /**
     * 处理滑动条值变化的函数
     * @param {number | number[]} values - 滑动条当前的值（单值或范围）
     */
    const sliderChange = (values: number | number[]) => {
        setValue2(values as number); // 更新滑动条值
        const int = transaction ? decimals : precision[chain]
        const num = (transaction ? tokenBalance : balance)
        const vf = parseUnits((values as number / 100) + '', int)
        const ov = (num * vf) || 0
        const data = formatUnits(ov, int * 2)
        setValue(data); // 同步更新输入框的值
    };

    /**
     * 处理防夹开关变化的函数
     * @param {boolean} val - 防夹开关的新状态
     */
    const changeAntiPinch = (val: boolean) => {
        setAntiPinch(val); // 更新防夹开关的状态
        // 更新筛选条件中的防夹开关状态
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                antiPinch: val,
            },
        });
    };

    /**
     * 处理自动滑点开关变化的函数
     * @param {boolean} val - 自动滑点开关的新状态
     */
    const changeAuto = (val: boolean) => {
        setAuto(val); // 更新自动滑点开关的状态
        // 如果自动滑点开启，则设置滑点值为默认值10
        if (val) {
            setPointSlip('10');
            setFiltrate({
                ...filtrate,
                trade: {
                    ...filtrate.trade,
                    auto: val,
                    pointSlip: '10',
                },
            });
        } else {
            setFiltrate({
                ...filtrate,
                trade: {
                    ...filtrate.trade,
                    auto: val,
                },
            });
        }
    };

    /**
     * 处理手动滑点值变化的函数
     * @param {string} value - 手动设置的滑点值
     */
    const changePointSlip = (value: string) => {
        setPointSlip(value); // 更新手动滑点值
        if (auto) {
            setAuto(false); // 如果当前是自动滑点模式，则关闭自动模式
        }
        setAutoNum(0); // 将自动滑点数值重置为0
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                pointSlip: value,
                auto: false,
                autoNum: 0,
            },
        });
    };

    /**
     * 处理自动滑点数值变化的函数
     * @param {number} value - 新的自动滑点数值
     */
    const changeAutoNum: InputNumberProps['onChange'] = (value) => {
        const num = value as number || 0;
        setAutoNum(num); // 更新自动滑点数值
        if (pointSlip !== 'auto') {
            setPointSlip('auto'); // 如果当前滑点模式不是自动，则切换到自动模式
        }
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                autoNum: num,
                pointSlip: 'auto',
            },
        });
    };

    /**
     * 处理自动滑点输入框的聚焦事件
     * 切换到自动滑点模式
     */
    const autoNumInputFocus = () => {
        if (pointSlip !== 'auto') {
            changePointSlip('auto');
        }
    };

    /**
     * 处理选中Gas费用选项的函数
     * @param {number} index - 选中的Gas费用项索引
     */
    const changeGasSelected = (index: number) => {
        setGasSelected(index); // 更新选中的Gas费用索引
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                gasSelected: index,
            },
        });
    };
    const setTransactionFun = (num: number) => {
        setTransaction(num)
        setFiltrate({
            ...filtrate,
            trade: {
                ...filtrate.trade,
                transaction: num,
            },
        });
    }
    // 定义滑点设置的选项列表
    const options = [
        { label: '3%', value: '3' },
        { label: '5%', value: '5' },
        { label: '10%', value: '10' },
        { label: '15%', value: '15' },
        {
            label: (
                <InputNumber
                    controls={false}
                    value={autoNum || null}
                    className="text-center w-full"
                    placeholder={t('custom')}
                    onChange={changeAutoNum}
                    onFocus={autoNumInputFocus}
                />
            ),
            value: 'auto',
            disabled: true,
        },
    ];

    // 根据当前的Gas基础值计算Gas费用选项数据
    const gasData = useMemo(() => {
        return [
            { label: t('normal'), value: multiplyDecimal(gas, 1) },
            { label: t('fast'), value: multiplyDecimal(gas, 1.5) },
            { label: t('rapid'), value: multiplyDecimal(gas, 2) },
        ];
    }, [gas]);
    // 判断button展示什么
    const buttonText = useMemo(() => {
        const values = value || '0'
        if (transaction) {
            if (!Number(value)) {
                return { text: t('sell'), isd: true }
            }
            else if (formatUnits(tokenBalance, decimals) >= values) {
                return { text: t('sell'), isd: false }
            } else {
                return { text: t('noBal'), isd: true }
            }
        } else {
            if (!Number(value)) {
                return { text: t('buy'), isd: true }
            }
            else if (formatUnits(balance, precision[chain]) >= values) {
                return {
                    text: t('buy'), isd: false
                }
            } else {
                return { text: t('noBal'), isd: true }
            }
        }
    }, [transaction, t, value, tokenBalance, balance])
    // 监听交易类型的变化，重置输入值和滑动条值
    useEffect(() => {
        setValue2(0); // 重置滑动条值
        setValue(''); // 重置输入值
        getBalance()
        getTokenBalance()
    }, [transaction]);

    return (
        <ConfigProvider
            theme={{
                components: {
                    InputNumber: {
                        colorText: "var(--text-color)", // 输入框文字颜色
                        activeBg: "var(--trade-buy-numInput-bg)", // 输入框激活状态时背景颜色
                        addonBg: "var(--trade-buy-numInput-bg)", // 输入框前后缀背景颜色
                        hoverBg: "var(--trade-buy-numInput-bg)", // 输入框悬停状态时背景颜色
                        colorBgContainer: "var(--trade-buy-numInput-bg)", // 输入框容器背景颜色
                        lineWidth: 0, // 输入框容器边框宽度
                        controlHeight: 48, // 控制按钮高度
                        activeShadow: "none", // 输入框激活状态时阴影
                    },
                    Switch: {
                        trackHeight: 16, // 开关高度
                        handleSize: 12, // 开关把手大小
                        trackMinWidth: 33, // 开关最小宽度
                    }
                },
            }}
        >
            <div className="font-semibold dark:bg-2b  bg-fb rounded-10 pb-3 ">
                <div className="mt-1.5  text-13 cursor-pointer flex justify-between items-center">
                    <div onClick={() => setTransactionFun(0)} className={` ${transaction === 0 ? 'rounded-t-10 bg-b6 text-black' : 'rounded-ss-10 dark:text-6f text-2bc dark:bg-black1F bg-fb '}  flex-[1]  flex items-center justify-center h-12.5`}>{t('buy')}</div>
                    <div onClick={() => setTransactionFun(1)} className={`${transaction === 1 ? 'rounded-t-10 bg-ffb text-black' : 'rounded-se-10 text-ff dark:bg-black1F bg-fb'}  flex-[1]  flex items-center justify-center h-12.5`}>{t('sell')}</div>
                    {/* <div onClick={() => setSelected(3)} className={` ${selected === 3 ? 'rounded-t-10 bg-fe text-b1' : 'rounded-se-10 text-fe bg-black1F '} flex-[1]  flex items-center justify-center h-12.5`}>AUTO</div> */}
                </div>
                <div className="px-2 pt-1.75 pb-1.25 ">
                    <div className="text-13 text-af mb-1.5 flex items-center justify-end">
                        <span>{t('bal')}：</span>
                        <div className={`${transaction === 0 ? 'block' : 'hidden'} flex items-center`}>
                            <FormatUnits loading={getBalanceLod} value={balance} />${chainMain[chain]}
                        </div>
                        <div className={`${transaction === 1 ? 'block' : 'hidden'}`}>
                            <FormatUnits loading={getTokenBalanceLod} value={tokenBalance} decimals={decimals} />
                        </div>
                    </div>
                    <div className="border-77 rounded-10 border overflow-hidden ">
                        <InputNumber controls={false}
                            value={value || null}
                            onChange={onChange}
                            className="w-full "
                            addonBefore={<div className="text-77">{t('amount')}</div>}
                            addonAfter={<div className="text-77">{transaction ? quoteSymbol : `${chainMain[chain]}`}
                            </div>}
                        />
                    </div>
                    <div className="h-20 flex items-center px-2">
                        <Slider
                            min={0}
                            max={100}
                            value={value2}
                            onChange={sliderChange}
                            // 自定义滑动条样式
                            styles={{
                                // 滑动轨道样式
                                track: {
                                    background: transaction ? '#FF4E4E' : '#6FFF89', // 轨道背景色
                                    height: '0.125rem' // 轨道高度
                                },
                                // 滑动柄样式
                                handle: {
                                    borderColor: transaction ? '#FF4E4E' : '#6FFF89', // 滑动柄边框色
                                    backgroundColor: transaction ? '#FF4E4E' : '#6FFF89', // 滑动柄背景色
                                    boxShadow: 'none' // 移除滑动柄阴影
                                },
                                // 导轨样式
                                rail: {
                                    background: 'var(--slider-bg-color)', // 导轨背景色
                                    height: '0.125rem' // 导轨高度
                                }
                            }}
                            // 自定义点的样式
                            dotStyle={{
                                borderColor: transaction ? '#FF4E4E' : '#6FFF89', // 点的边框色
                                backgroundColor: transaction ? '#FF4E4E' : '#6FFF89', // 点的背景色
                                top: '-0.2rem' // 点的垂直位置调整
                            }}
                            // 自定义活动状态下的点的样式
                            activeDotStyle={{
                                borderColor: transaction ? '#FF4E4E' : '#6FFF89', // 活动状态下点的边框色
                                backgroundColor: transaction ? '#FF4E4E' : '#6FFF89' // 活动状态下点的背景色
                            }}
                            // 滑动条标记
                            marks={marks}
                        />
                    </div>
                    {/* <div className="text-sm text-fe mb-2.25">58%的用户购入了0.5SOL</div> */}
                    <Button onClick={transactionFunAll} disabled={buttonText.isd} loading={transactionLoading || getBalanceLod || getTokenBalanceLod} className={`${transaction ? 'bg-ffb border-ffb' : 'bg-b6 border-b6'} rounded-2.5xl  text-black font-semibold h-10 leading-7 mb-2.5`} block>
                        {buttonText.text}
                    </Button>
                    <div onClick={() => setOpenSetting(!openSetting)} className=" transition-all  flex items-center text-af text-sm justify-between  cursor-pointer">
                        <div className="flex items-center">
                            <SvgIcon stopPropagation={false} value="setting" className="w-5 mr-1" />
                            <span>{t('advancedSettings')}</span>
                        </div>
                        <SvgIcon stopPropagation={false} value="arrowsBottom" className={`w-3 duration-500 ${openSetting ? 'rotate-180' : ''}`} />
                    </div>
                    <div className={`pt-1 px-1 dark:text-white text-4b overflow-hidden transition-all duration-500 ${openSetting ? 'h-0' : 'h-72'}`}>
                        <div className="box-border mt-1 dark:bg-black1F bg-f1 py-3 px-3 rounded-10 text-13">
                            <div className="flex items-center justify-between mb-2">
                                <span className="flex items-center ">
                                    {t('antiPinchSwitch')}
                                    <TableTooltip title={t('privatePoolTransaction')}>
                                        <SvgIcon value="about" className="w-3 ml-1 cursor-pointer" />
                                    </TableTooltip>
                                </span>
                                <Switch className={`${antiPinch ? 'bg-green-400' : 'bg-82'}`} value={antiPinch} onChange={changeAntiPinch} />
                            </div>
                            <div className="flex items-center mb-2">
                                <span >
                                    {t('slippage')}
                                </span>
                                <TableTooltip title={t('highSlippageAutoWarning')}>
                                    <SvgIcon stopPropagation={false} value="about" className="w-3 ml-1 cursor-pointer" />
                                </TableTooltip>
                            </div>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-a0">
                                    {t('auto')}
                                </span>
                                <Switch className={auto ? "bg-green-400" : 'bg-82'} value={auto} onChange={changeAuto} />
                            </div>
                            <ConfigProvider
                                theme={{
                                    components: {
                                        InputNumber: {
                                            colorText: "var(--text-color)", // 输入框文字颜色
                                            activeBg: "var(--trade-buy-numInput-bg)", // 输入框激活状态时背景颜色
                                            addonBg: "var(--trade-buy-numInput-bg)", // 输入框前后缀背景颜色
                                            hoverBg: "var(--trade-buy-numInput-bg)", // 输入框悬停状态时背景颜色
                                            colorBgContainer: "var(--trade-buy-numInput-bg)", // 输入框容器背景颜色
                                            lineWidth: 0, // 输入框容器边框宽度
                                            controlHeight: 10, // 控制按钮高度
                                            activeShadow: "none", // 输入框激活状态时阴影
                                            paddingInline: 5, // 输入框内边距
                                            colorTextPlaceholder: "var(--placeholder-color)" // 输入框提示文字颜色
                                        },
                                        Segmented: {
                                            trackBg: "var(--trade-buy-numInput-bg)", // Segmented 控件容器背景色
                                            itemActiveBg: "var(--trade-buy-numInput-bg)", // 选项激活态背景颜色
                                            itemSelectedBg: "var(--text-color)", // 选项选中时背景颜色
                                            itemColor: "var(--header-copy-border)", // 选项文本颜色
                                            itemSelectedColor: "var(--trade-buy-numInput-bg)", // 选项选中时文字颜色
                                            itemHoverColor: "var(--header-copy-border)", // 选项hover时文字颜色
                                            itemHoverBg: "var(--trade-buy-numInput-bg)", // 选项hover时背景颜色
                                        }
                                    },
                                }}
                            >
                                <Segmented
                                    value={pointSlip}
                                    onChange={changePointSlip}
                                    className="mb-2"
                                    options={options}
                                />
                            </ConfigProvider>
                            <div className="mb-2">
                                <p className="pb-1">{t('gasFee')}</p>
                                <p className="text-a0 text-xs">{chainMain[chain]}</p>
                            </div>
                            <div className="box-border flex items-center text-center overflow-x-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-gray-400">
                                {gasData.map((item, index) => (
                                    <div onClick={() => { changeGasSelected(index) }} className={`cursor-pointer p-2 dark:bg-51 bg-white rounded-10 mr-2 border  last:mr-0 ${gasSelected === index ? 'dark:border-white border-black' : 'border-transparent'}`} key={index}>
                                        <p>{item.label}</p>
                                        <p>{`(${item.value + chainMain[chain]})`} ≈ $1.5</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </ConfigProvider >
    )
}

export default memo(Transaction)