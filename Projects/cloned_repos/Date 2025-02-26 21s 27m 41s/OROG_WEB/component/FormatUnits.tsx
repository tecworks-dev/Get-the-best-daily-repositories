import { precision } from "@/app/chainLogo"
import { ChainsType } from "@/i18n/routing"
import { LoadingOutlined } from "@ant-design/icons"
import { formatUnits } from "ethers"
import { useParams } from "next/navigation"

/**
 * 根据区块链类型格式化数字单位
 * 此组件用于接收一个数字和区块链类型，然后根据区块链的精度要求格式化这个数字
 * 主要用于在用户界面中以更易读的方式展示数字
 * 
 * @param {Object} props - 组件的属性对象
 * @param {boolean} props.loading - 表示是否处于加载状态的布尔值
 * @param {bigint | number} props.value - 需要格式化的数字
 * @param {ChainsType} props.chainMain - 可选参数，指定区块链类型，如果未提供，则使用URL参数中的区块链类型
 * @param {string} props.className - 可选参数，要应用到组件外部div的CSS类名
 * @returns {JSX.Element} - 返回一个包含格式化后数字的div组件，或者在加载时显示加载图标
 */
const FormatUnits = ({ loading, value, chainMain, className, decimals }: { loading: boolean, value: bigint | number, chainMain?: ChainsType, className?: string, decimals?: number }) => {
    // 从URL参数中解析出区块链类型
    const { chain }: { chain: ChainsType } = useParams()
    // 根据是否处于加载状态，决定是否显示加载图标或格式化后的数字
    return (<div className={`${className}`}>
        <LoadingOutlined className={`${loading ? 'block' : 'hidden'}`} />
        <span className={`${!loading ? 'block' : 'hidden'}`}>{formatUnits(value, decimals ? decimals : precision[chainMain ? chainMain : chain])}</span>
    </div>)
}
export default FormatUnits