import { multiplyAndTruncate } from "@/utils";
import { memo } from "react";

/**
 * TablePercentage 组件用于计算并显示百分比变化
 * @param {number} number1 - 初始值
 * @param {number} number2 - 变化后的值（可选，默认为 0）
 * @param {string} className - 可选的自定义类名
 * @returns {JSX.Element} - 返回一个带有百分比变化的文本组件
 */
const TablePercentage = ({ number1, number2 = 0, className = ' text-13 font-semibold' }: { number1: number; number2?: number, className?: string }) => {
    /**
     * 计算百分比变化
     * - 如果 `number2` 存在且 `number1` 不为 0，则计算变化率 `(number2 - number1) / number1`
     * - 如果 `number1` 为 0：
     *   - `number2` 也为 0，返回 `-100`
     *   - `number2` 不为 0，返回 `100`
     * - 如果 `number2` 未提供，则直接使用 `number1` 作为结果
     */
    const liquidity_ = number2
        ? number1 !== 0
            ? multiplyAndTruncate((number2 - number1) / number1) // 计算百分比变化，并使用 `multiplyAndTruncate` 进行格式化
            : number2 === 0
                ? -100 // 初始值为 0，变化值也为 0，则显示 `-100`
                : 100 // 初始值为 0，变化值不为 0，则显示 `100`
        : multiplyAndTruncate(number1);

    /**
     * 根据计算出的百分比值设置文本颜色
     * - `liquidity_ > 0` 时，使用 `text-6f`（正数颜色）
     * - `liquidity_ < 0` 时，使用 `text-ff`（负数颜色）
     * - `liquidity_ === 0` 时，不设置颜色
     */
    const percentageClass =
        Number(liquidity_) > 0
            ? "dark:text-6f text-2bc"
            : Number(liquidity_) < 0
                ? "text-ff"
                : "dark:text-c0 text-80";

    return (
        <div className={`${percentageClass} ${className}`}>
            {Number(liquidity_) ? (Number(liquidity_) >= 0 ? '+' : '-') : ''}
            {/* 显示计算后的百分比值，带 % 符号 */}
            {Math.abs(Number(liquidity_))}%
        </div>
    );
};

export default memo(TablePercentage);