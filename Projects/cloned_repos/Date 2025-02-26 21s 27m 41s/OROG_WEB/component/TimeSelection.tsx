import { memo } from "react"

/**
 * 定义时间选择类型，包含展示文本和对应的关键字
 */
export type TimeSelectionType = {
    text: string,
    key: string
}

/**
 * 定义一组时间选择数据，用于短期时间选择
 */
export const timeData = [
    { text: '5m', key: '5m' },
    { text: '30m', key: '30m' },
    { text: '1h', key: '1h' },
    { text: '6h', key: '6h' },
    { text: '24h', key: '1d' },
]

/**
 * 定义另一组时间选择数据，用于中期和长期时间选择
 */
export const timeData2 = [
    { text: '24h', key: '1d' },
    { text: '7d', key: '7d' },
    { text: '30d', key: '30d' },
]

/**
 * 时间选择组件，根据类型展示不同的时间选项，并在点击时触发回调
 * @param {string} value 当前选中的时间关键字
 * @param {Function} onClick 点击时间选项时的回调函数
 * @param {0 | 1} type 时间数据类型，0 使用 timeData，1 使用 timeData2
 * @param {string} className 额外的类名
 */
const TimeSelection = ({ value, onClick, type = 0, className = '' }: { value: string, type?: 0 | 1, onClick?: Function, className?: string, hover?: string }) => {
    return (
        <div className={`text-15  dark:text-77 text-6d font-normal flex items-center ${className}`}>
            {[timeData, timeData2][type].map((item, index) => (
                <div onClick={() => { onClick && onClick(item.key) }} className={`mr-5 cursor-pointer ${value === item.key ? 'dark:text-white text-28' : ''}`} key={index}>{item.text}</div>
            ))}
        </div>
    )
}

export default memo(TimeSelection)