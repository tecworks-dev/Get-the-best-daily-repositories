// 允许在客户端使用 React Hooks
'use client'

// 导入格式化时长的工具函数
import { formatDuration } from "@/utils"
// 导入 React 的 Hooks
import { memo, useEffect, useState } from "react"

/**
 * TableAge 组件用于展示一个基于给定值计算的时长
 * 它接收一个数值作为属性，并将其转换为易于阅读的时长格式
 * 
 * @param {Object} props - 组件的属性对象
 * @param {number} props.value - 需要格式化为时长的数值
 * @returns {JSX.Element} - 格式化后的时长
 */
const TableAge = ({ value }: { value: number }) => {
    // 初始化状态值，用于内部计时
    const [_valueTime, setValueTime] = useState(0)

    // 轮询逻辑
    useEffect(() => {
        // 设置一个定时器，每秒执行一次
        const timer = setInterval(() => {
            // 更新_valueTime状态，如果超过10则重置为0，否则递增
            setValueTime((prevValue) => {
                if (prevValue > 10) return 0
                return prevValue + 1
            })
        }, 1000);

        // 当组件卸载时，清除定时器
        return () => clearInterval(timer);
    }, [value]);

    // 渲染组件，展示格式化后的时长
    return (
        <>{formatDuration(value)}</>
    )
}

// 导出 TableAge 组件
export default memo(TableAge)