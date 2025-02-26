import { formatLargeNumber } from "@/utils";
import { memo } from "react";

// 定义 TableTxns 组件，接受交易数据作为参数
const TableTxns = ({
    swaps, // 总交易次数
    buys,  // 买入交易次数
    sells  // 卖出交易次数
}: {
    swaps: number, // 类型注解，表示 swaps 是一个数字
    buys: number,  // 类型注解，表示 buys 是一个数字
    sells: number  // 类型注解，表示 sells 是一个数字
}) => {
    // 返回组件的渲染内容
    return (
        <div className="text-sm font-semibold">
            {/* 显示总交易次数 */}
            <div>{formatLargeNumber(swaps)}</div>
            {/* 显示卖出和买入交易次数，带有分隔符 */}
            <div className="flex items-center justify-center">
                {/* 卖出次数，使用特定颜色 */}
                <div className="dark:text-6f text-2bc">{formatLargeNumber(sells)}</div>
                {/* 分隔符，使用另一种颜色 */}
                <p className="dark:text-82 text-6d">/</p>
                {/* 买入次数，使用特定颜色 */}
                <div className="text-ff">{formatLargeNumber(buys)}</div>
            </div>
        </div>
    );
};

// 导出组件以供其他模块使用
export default memo(TableTxns);