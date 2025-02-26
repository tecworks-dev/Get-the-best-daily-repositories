import { formatLargeNumber } from "@/utils"
import TablePercentage from "./TablePercentage"
import { memo } from "react"

/**
 * TableEarnings组件用于显示表格中的收益信息
 * 该组件根据收益的正负来改变文本颜色和符号
 * 
 * @param {Object} props - 组件的属性对象
 * @param {string} props.money - 收益的金额，以字符串形式表示
 * @param {string} props.moneyf - 收益的金额，格式化后的字符串
 * 
 * 此组件不直接与外部交互，通过props接收数据，并根据数据状态调整显示样式
 */
const TableEarnings = ({ money, moneyf }: { money: number, moneyf: number }) => {
    return (
        <div className={`${money === 0 ? '' : (money < 0 ? 'text-ff' : 'dark:text-6f text-2bc')} font-semibold text-center`}>
            <div className="text-sm">{formatLargeNumber(money)}</div>
            <TablePercentage number1={moneyf} />
        </div>
    )
}

export default memo(TableEarnings)