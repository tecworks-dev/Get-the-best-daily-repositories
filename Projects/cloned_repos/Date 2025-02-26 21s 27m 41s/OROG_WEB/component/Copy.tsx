/**
 * 导入复制功能的工具函数和React相关库
 */
import { handleCopyClick } from "@/utils"
import { memo, ReactNode, useState } from "react"
import SvgIcon from "./SvgIcon"

/**
 * 定义复制组件的属性类型
 */
type CopyType = {
    text: string | number, // 需要复制的文本或数字
    className?: string, // 图标类名
    copyIcon?: ReactNode, // 自定义复制图标
    successIcon?: ReactNode, // 自定义复制成功图标
    errorIcon?: ReactNode, // 自定义复制失败图标
}

/**
 * 复制组件，用于复制指定的文本或数字
 * @param {CopyType} props 组件属性
 */
const Copy = ({ text, copyIcon, className, successIcon, errorIcon, }: CopyType) => {
    // 状态管理显示的图标，1为复制图标，2为成功图标，3为失败图标
    const [isShow, setIsShow] = useState(1)

    /**
     * 复制功能的事件处理函数
     * @param {any} e 事件对象
     */
    const copyFun = (e: any) => {
        e.stopPropagation(); // 阻止事件冒泡
        // 调用复制工具函数，并根据复制结果更新显示的图标
        const flag = handleCopyClick(text)
        flag ? setIsShow(2) : setIsShow(3)
        // 2秒后恢复为复制图标
        setTimeout(() => {
            setIsShow(1)
        }, 2000)
    }

    // 根据状态渲染对应的图标
    return (
        <>
            {isShow === 1 && <div className="cursor-pointer" onClick={copyFun}>{copyIcon || <SvgIcon stopPropagation={false} className={`${className}`} value="copy" />}</div>}
            {isShow === 2 && <div onClick={(e) => { e.stopPropagation(); }}>{successIcon || <SvgIcon stopPropagation={false} className={`${className}`} value="rightCopy" />}</div>}
            {isShow === 3 && <div onClick={(e) => { e.stopPropagation(); }}>{errorIcon || <SvgIcon stopPropagation={false} className={`${className}`} value="errorCopy" />}</div>}
        </>
    )
}

// 导出复制组件
export default memo(Copy)