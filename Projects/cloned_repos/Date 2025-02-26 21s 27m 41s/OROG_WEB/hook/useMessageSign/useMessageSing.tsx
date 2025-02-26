import { useState } from 'react';

import { ChainsType } from '@/i18n/routing';
import { ChainsTypeKeyFun } from '@/interface';
import useConnectWallet from '@/store/wallet/useConnectWallet';

import messageSingSol from './messageSingSol';

/**
 * 自定义钩子，用于处理消息签名
 * @param chain 区块链类型，例如'sol'
 * @param successFun 成功回调函数，可选参数
 * @param errorFun 错误回调函数，可选参数
 * @returns 返回消息签名函数和加载状态
 */
const useMessageSing = (chain: ChainsType, successFun?: Function, errorFun?: Function) => {

    // 使用useConnectWallet钩子获取Solana链的钱包提供者
    const { provider } = useConnectWallet(chain)
    // 管理消息签名的加载状态
    const [messageSingLod, setLoading] = useState(false)

    // 定义一个函数映射，根据链类型调用对应的签名函数
    const singFun: ChainsTypeKeyFun = {
        sol: () => messageSingSol(provider)
    }
    /**
     * 执行消息签名的异步函数
     * @returns 返回签名数据，如果没有成功签名则返回空字符串
     */
    const messageSingFun = async () => {
        // 检查是否已经连接到Solana链的钱包提供者
        if (provider) {
            // 开始签名流程前设置加载状态为true
            setLoading(true)
            try {
                // 根据链类型调用对应的签名函数，并传递成功回调函数
                const data = await singFun[chain]();
                successFun && successFun(data);
                return data;
            } catch (e) {
                // 捕获签名过程中的错误，并调用错误回调函数（如果提供）
                console.log("messageSingSol", e);
                if (errorFun) {
                    errorFun(e);
                }
            }
            // 签名流程结束后设置加载状态为false
            setLoading(false)
        }
        // 如果没有连接到钱包提供者，返回空字符串
        return '';
    }

    // 返回消息签名函数和加载状态
    return { messageSingFun, messageSingLod }
}

// 导出自定义钩子，以便在其他地方使用
export default useMessageSing