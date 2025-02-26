import { formatLargeNumber } from "@/utils"
import { memo } from "react"

const TablePriceTransition = ({ className, price, mainTokenName, num }: { className?: string, price: number, mainTokenName: string, num: number }) => {
    const number = num && price ? (num / price) : 0
    return (
        <div className={`${className ?? ''} text-sm font-semibold dark:text-b1 text-9b`}>
            {/* 格式化并显示当前流动性 */}
            <span className="dark:text-white  text-51"> {formatLargeNumber(number)} {mainTokenName}</span>
            {/* 显示分隔符 */}
            <span className="px-1 ">/</span>
            {/* 格式化并显示初始流动性 */}
            <span className="">
                {formatLargeNumber(num)}
            </span>
        </div>
    )
}

export default memo(TablePriceTransition)