import { formatLargeNumber } from "@/utils"
import { useMemo } from "react"

/**
 * 价格组件
 * 
 * 此组件用于显示价格信息，根据价格的正负零值动态调整样式
 * 它接受一个类名和一个数值作为属性，并返回一个带有适当样式和格式化价格的div
 * 
 * @param {Object} props - 组件的属性对象
 * @param {string} props.className - 额外的CSS类名，默认为'pl-2 text-2xl pb-0.25'
 * @param {number} props.num - 要显示的价格数值
 * @param {boolean} props.isf - 是否显示正负号，默认为false
 * @returns {JSX.Element} - 带有价格信息和动态样式的div元素
 */
const PriceCom = ({ className = '', num, isf = false }: { className?: string, num: number, isf?: boolean }) => {
    // 根据价格数值的正负零值，动态选择CSS类名
    // 使用模板字符串拼接最终的className
    const fuhao = useMemo(() => {
        if (isf) {
            return num === 0 ? '' : num > 0 ? '+' : '-'
        }
        return ''
    }, [isf, num]) // 添加依赖数组

    return (
        <div className={`${className} ${num === 0 ? 'dark:text-c0 text-80' : (num > 0 ? "dark:dark:text-6f text-2bc " : "text-ff")}`}>
            {fuhao}
            ${formatLargeNumber(num)}
        </div>
    )
}

export default PriceCom