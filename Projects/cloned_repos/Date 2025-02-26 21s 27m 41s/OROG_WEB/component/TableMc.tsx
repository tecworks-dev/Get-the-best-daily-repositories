// 引入工具函数，用于格式化大数字（例如：将数值以简洁的格式显示）
import { formatLargeNumber } from "@/utils"
import { memo } from "react"

// 定义 TableMc 组件，显示市值的同时根据值的范围动态调整颜色
const TableMc = ({ marketCap, className }: { marketCap: number, className?: string }) => {

    // 定义一个函数，用于返回动态的颜色类名
    const colorFun = () => {
        // 如果市值小于 100，返回浅色的类名
        if (marketCap < 100) return 'dark:text-fff4 text-da'
        // 如果市值在 100 和 1000 之间，返回对应的类名
        else if (marketCap >= 100 && marketCap < 1000) return 'dark:text-fe text-f6 '
        // 如果市值在 1000 和 10000 之间，返回对应的类名
        else if (marketCap >= 1000 && marketCap < 10000) return 'dark:text-fd text-fc'
        // 如果市值大于等于 10000，返回默认的深色类名
        else return 'text-fb'
    }

    // 返回组件的渲染内容
    return (
        // 使用动态生成的类名和可选的外部类名，确保样式可以灵活调整
        <div className={`font-semibold ${colorFun()} text-sm  ${className}`}>
            {
                // 调用格式化工具函数将市值格式化为简洁的显示格式
                <span>{formatLargeNumber(marketCap)}</span>
            }
        </div>
    )
}

// 导出组件供其他模块使用
export default memo(TableMc)