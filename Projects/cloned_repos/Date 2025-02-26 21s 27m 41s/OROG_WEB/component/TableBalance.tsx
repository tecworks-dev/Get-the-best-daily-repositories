import { memo } from "react";

/**
 * 表格平衡组件
 * 
 * 该组件用于在表格中以平衡的方式显示两个数字这些数字可能代表某种数据对比或平衡关系
 * 组件接受两个字符串类型的数字作为输入，并将它们格式化后显示在不同的样式层级上
 * 
 * @param {string} num - 要显示的第一个数字，通常代表主要或起始数值
 * @param {string} num2 - 要显示的第二个数字，通常代表次要或对比数值
 * @returns {JSX.Element} - 返回一个包含两个格式化数字的div组件
 */
const TableBalance = ({ num, num2 }: { num: string, num2: string }) => {
  return <div className="font-semibold text-center">
    <div className="text-sm dark:text-white text-51">{num}</div>
    <div className="text-13 dark:text-6b text-82">{num2}</div>
  </div>;
};

export default memo(TableBalance);