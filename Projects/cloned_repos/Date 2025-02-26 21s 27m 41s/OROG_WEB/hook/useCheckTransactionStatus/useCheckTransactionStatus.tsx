import { ChainsType } from '@/i18n/routing';
import {
  useConnectWalletSolStore,
} from '@/store/wallet/connectWallet/useWalletSol';

import checkTransactionStatus from './checkTransactionStatusSol';

/**
 * 钩子函数，用于检查交易状态
 * @param chain 区块链类型，决定了使用哪种方式来检查交易状态
 * @returns 返回一个函数，该函数接受一个签名参数，并检查对应区块链的交易状态
 */
const useCheckTransactionStatus = (chain: ChainsType) => {
  // 使用useConnectWalletSolStore钩子获取钱包连接状态
  const { connection } = useConnectWalletSolStore()

  // 定义一个对象，包含不同区块链的交易状态检查函数
  const fun = {
    // Solana区块链的交易状态检查函数
    sol: async (signature: string) => {
      if (!connection) return
      // 如果有连接，就使用checkTransactionStatus函数检查交易状态
      const data = await checkTransactionStatus(connection, signature)
      return data
    }
  }

  // 根据传入的区块链类型，返回对应的交易状态检查函数
  return fun[chain]
}

export default useCheckTransactionStatus