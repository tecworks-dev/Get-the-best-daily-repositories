// 引入工具函数，用于格式化大数字（例如：将数值以简洁的格式显示）
import { formatLargeNumber, multiplyAndTruncate } from "@/utils";
import TablePercentage from "./TablePercentage";
import { memo } from "react";

// 定义组件的接口，描述传入的属性
interface TableLiqInitialProps {
    liquidity: number; // 当前流动性
    initialLiquidity: number; // 初始流动性
    className?: string; // 可选的自定义类名
    isShow?: boolean; // 是否显示流动性百分比变化，默认值为 false
}

// 定义 TableLiqInitial 组件
const TableLiqInitial = ({
    liquidity, // 当前流动性
    initialLiquidity, // 初始流动性
    className, // 可选的自定义类名
    isShow = false // 是否显示百分比变化，默认值为 false
}: TableLiqInitialProps) => {
    // 渲染组件内容
    return (
        <>
            {/* 显示当前流动性和初始流动性 */}
            <div className={`${className ?? ''} text-sm font-semibold`}>
                {/* 格式化并显示当前流动性 */}
                {formatLargeNumber(liquidity)}
                {/* 显示分隔符 */}
                <span className="px-1 dark:text-b1 text-9b">/</span>
                {/* 格式化并显示初始流动性 */}
                <span className="dark:text-b1 text-9b">
                    {formatLargeNumber(initialLiquidity)}
                </span>
            </div>
            {/* 如果 isShow 为 true，显示流动性变化的百分比 */}
            {isShow && (
                <TablePercentage
                    number1={initialLiquidity} number2={liquidity} />
            )}
        </>
    );
};

// 导出组件供其他模块使用
export default memo(TableLiqInitial);