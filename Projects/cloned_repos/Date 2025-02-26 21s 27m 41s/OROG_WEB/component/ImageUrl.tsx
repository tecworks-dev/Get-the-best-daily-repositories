import { useState, useEffect, memo } from "react"
import DefImage from "./DefImage"

// 定义一个React组件，用于在界面上显示图像或默认图像
const ImageUrl = ({ logo, symbol, className }: { logo: string, symbol: string, className?: string }) => {
    // 初始化状态，用于控制是否显示图像
    const [isShow, setIsShow] = useState(true)
    // 初始化状态，用于控制是否显示默认图像
    const [timeOver, setTimeOver] = useState(false)

    // 使用useEffect来设置一个定时器，如果在指定时间内图像未加载完成，则隐藏图像
    useEffect(() => {
        setIsShow(true)
        let timeOut: any
        if (!timeOver) {
            timeOut = setTimeout(() => {
                setIsShow(false)
            }, 10000)
        }
        return () => {
            clearTimeout(timeOut)
        }
    }, [timeOver, logo])

    // 根据图像URL和状态，决定显示图像还是默认图像
    return (
        <>
            {logo && isShow ?
                <img loading="lazy"
                    onLoad={() => { setTimeOver(true) }}
                    onError={(e) => { setIsShow(false) }}
                    src={logo} className={`${className}`} alt="" />
                :
                <DefImage value={symbol[0]} className={`${className}`} />}
        </>
    )
}

export default memo(ImageUrl)