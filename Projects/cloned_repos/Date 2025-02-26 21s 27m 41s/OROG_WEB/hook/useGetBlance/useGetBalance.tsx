import {
  useEffect,
  useState,
} from 'react';

import { ChainsType } from '@/i18n/routing';
import { ChainsTypeKeyFun } from '@/interface';
import useConnectWallet from '@/store/wallet/useConnectWallet';

import getBalanceSol from './getBalanceSol';

/**
 * 钩子函数，用于获取指定区块链网络上的账户余额
 * @param chain 指定的区块链类型，用于确定使用哪个区块链网络
 * @param successFun 可选参数，成功获取余额后的回调函数
 * @param errorFun 可选参数，获取余额失败后的回调函数
 */
const useGetBalance = (chain: ChainsType, successFun?: Function, errorFun?: Function) => {
    // 使用状态管理Solana区块链的余额和加载状态
    const [balanceSol, setBalanceSol] = useState(BigInt(0))
    // 从连接钱包的存储中获取地址和连接对象
    const { address, connection } = useConnectWallet(chain)
    // 管理获取余额的加载状态
    const [getBalanceLod, setLoading] = useState(false)

    // 构建余额和地址的对象，以便根据链的类型访问
    const balance = {
        sol: balanceSol,
    }
    // 定义获取余额的函数对象，根据不同的链类型调用相应的函数
    const getBalanceFun: ChainsTypeKeyFun = {
        sol: () => getBalanceSol(connection, address)
    }

    // 获取余额的异步函数
    const getBalance = async () => {
        setLoading(true)
        try {
            // 根据链类型调用相应的获取余额函数，并更新状态
            const balance = await getBalanceFun[chain]()
            setBalanceSol(BigInt(balance))
            // 如果提供了成功回调函数，则调用它
            successFun && successFun()
        } catch (e) {
            // 打印错误信息，并调用错误回调函数（如果有）
            console.log("useGetBalance", e)
            errorFun && errorFun()
        }
        setLoading(false)
    }

    // 使用Effect监听链类型或地址的变化，当它们变化时重新获取余额
    useEffect(() => {
        getBalance()
    }, [chain, address])

    // 返回当前链的余额和加载状态
    return { balance: balance[chain], getBalanceLod, getBalance }
}

export default useGetBalance