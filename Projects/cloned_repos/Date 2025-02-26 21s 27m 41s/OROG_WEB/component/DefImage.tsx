import { memo } from "react"

/**
 * DefImage组件用于显示一个带有值的默认图像样式div.
 * 
 * 该组件接受一个必填的value属性和一个可选的className属性.
 * Value属性用于指定要在div中显示的文本内容，
 * 而className属性允许用户将额外的CSS类名应用于div，以便于样式定制.
 * 
 * @param {Object} props - 组件的props对象.
 * @param {string} props.value - 要在组件中显示的文本值.
 * @param {string} [props.className] - 可选的CSS类名字符串，用于定制组件的样式.
 * @returns {JSX.Element} - 返回一个应用了指定样式和类名的div元素，其中包含文本值.
 */
const DefImage = ({ value, className }: { value: string, className?: string }) => {
    return (
        <div className={`  flex items-center justify-center bg-black text-white  text-2xl font-semibold ${className}`}>
            {value}
        </div>
    )
}

export default memo(DefImage)  