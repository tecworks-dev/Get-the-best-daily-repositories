import { useState } from 'react';

import { useTranslations } from 'next-intl';
import { toast } from 'react-toastify';

import { Scan } from '@/app/chainLogo';
import { ChainsType } from '@/i18n/routing';
import useConnectWallet from '@/store/wallet/useConnectWallet';

import useCheckTransactionStatus
  from '../useCheckTransactionStatus/useCheckTransactionStatus';
import transactionSol from './transactionSol';

/**
 * 初始化并执行特定区块链网络的交易
 * 
 * @param chain 区块链类型，用于指定操作的区块链网络
 * @param successFun 可选的回调函数，交易成功时调用
 * @param errorFun 可选的回调函数，交易失败时调用
 */
const useTransaction = (chain: ChainsType, successFun?: Function, errorFun?: Function) => {
    const tToast = useTranslations('Toast')
    // 使用useConnectWallet钩子获取Solana链的钱包提供者
    const { provider, connection, address } = useConnectWallet(chain)
    // loading
    const [transactionLoading, setLoading] = useState(false)
    // 使用useCheckTransactionStatus钩子来检查交易状态
    const checkTransactionStatus = useCheckTransactionStatus(chain)
    const transactionSolFun = async ({ quoteMintAddress, baseMintAddress, rawAmountIn, slippage }: { quoteMintAddress: string, baseMintAddress: string, rawAmountIn: number, slippage: number }) => {
        // 确保已连接到Solana链并获取到必要的提供者和地址信息
        if (connection && provider && address) {
            setLoading(true)
            try {
                // 执行Solana链的交易并获取交易签名
                const signature = await transactionSol(connection, provider, address, quoteMintAddress, baseMintAddress, rawAmountIn, slippage)
                // 检查交易状态并输出结果
                const flag = await checkTransactionStatus(signature)
                if (flag) {
                    successFun && successFun()
                    toast.success(tToast('transactionSuccess'), { onClick: () => window.open(`${Scan[chain]}${signature}`, '_blank') })
                } else {
                    errorFun && errorFun()
                    toast.error(tToast('transactionFailed'), { onClick: () => window.open(`${Scan[chain]}${signature}`, '_blank') })
                }
                console.log('flag', flag);
            } catch (e) {
                console.log('useTransaction', e);

            }
            setLoading(false)

        }
    }
    // 定义针对不同区块链的交易函数对象
    const transactionFun = {
        // Solana区块链的交易函数
        sol: transactionSolFun
    }
    return { transactionFun: transactionFun[chain], transactionLoading }
}

export default useTransaction