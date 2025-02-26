import { followHttp, removeFollowHttp } from "@/request"
import SvgIcon from "./SvgIcon"
import useLogin from "@/hook/useLogin"
import { ChainsType } from "@/i18n/routing"
import { memo, useEffect, useState } from "react"
import { useRemoveFollow } from "@/store/followList"
import useUpdateEffect from "@/hook/useUpdateEffect"

/**
 * 关注组件
 * 
 * 该组件用于显示关注或未关注状态，并允许用户进行关注或取消关注的操作
 * 它根据当前的follow状态来决定显示的关注或未关注图标，并在用户点击时触发相应的操作
 * 
 * @param {Object} props - 组件属性
 * @param {ChainsType} props.chain - 链类型，用于确定请求的链ID
 * @param {boolean} props.follow - 当前的关注状态，true表示已关注，false表示未关注
 * @param {string} props.quote_mint_address - 被关注对象的地址
 * @param {string} [props.className] - 可选的CSS类名，默认为'w-3.5 mr-2'
 */
const Follow = ({ follow, quote_mint_address, chain, className = 'w-3.5 mr-2' }: { chain: ChainsType, follow: boolean, quote_mint_address: string, className?: string }) => {
    // 获取检测执行函数
    const { checkExecution } = useLogin(chain)
    // 状态管理：是否已关注
    const [isFollow, setIsFollow] = useState(follow)
    const { removeRemoveFollow, addRemoveFollow, follow: followArr } = useRemoveFollow()
    // 收藏或取消收藏的操作函数
    const followFun = () => {
        checkExecution(async () => {
            try {
                // 根据当前关注状态执行相应的HTTP请求
                if (isFollow) {
                    await removeFollowHttp({ chain_id: chain, token: quote_mint_address })
                    addRemoveFollow(quote_mint_address)
                } else {
                    await followHttp({ chain_id: chain, token: quote_mint_address })
                    removeRemoveFollow(quote_mint_address)
                }
                // 更新关注状态
                // setIsFollow(!isFollow)
            } catch (e) {
                console.log('followFun', e);
            }
        })
    }
    // console.log('addFollow', addFollow, 'removeFollow', removeFollow);

    // 确保组件的关注状态与传入的follow参数保持一致
    useEffect(() => {
        setIsFollow(follow)
    }, [follow])
    useUpdateEffect(() => {
        console.log('只想');
        const data1 = followArr.removeFollow.find((item) => item === quote_mint_address)
        const data2 = followArr.addFollow.find((item) => item === quote_mint_address)
        if (data1 && !data2) {
            setIsFollow(false)
        } else if (!data1 && data2) {
            setIsFollow(true)
        }
    }, [followArr])
    // 渲染SvgIcon组件，根据关注状态改变图标和颜色
    return (
        <SvgIcon
            onClick={followFun}
            className={` ${className} cursor-pointer ${isFollow ? 'text-ffe' : 'text-77'}`}
            value="follow" />
    )
}

export default memo(Follow)