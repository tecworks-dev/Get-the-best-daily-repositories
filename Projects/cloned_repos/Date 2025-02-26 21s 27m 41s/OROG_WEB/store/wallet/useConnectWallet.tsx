import { useConnectWalletSolStore } from './connectWallet/useWalletSol'
import { ChainsTypeKeyFun, ChainsTypeKeyNum, ChainsTypeKeyStr } from '@/interface'
import { ChainsType } from '@/i18n/routing'

/**
 * 钱包连接钩子
 * 根据指定的链类型，返回与该链相关的钱包连接信息和操作函数
 * 
 * @param chain 链类型，用于指定需要连接的钱包链
 * @returns 返回一个对象，包含指定链的钱包地址、连接钱包函数、断开钱包函数、provider、钱包名称和连接状态
 */
const useConnectWallet = (chain: ChainsType) => {
    // 使用Solana钱包状态管理
    const { addressSol, connectWalletSolStore, disconnectWalletSolStore, providerSol, walletNameSol, connection } = useConnectWalletSolStore()
    // 定义一个对象，用于根据链类型获取钱包地址
    const addressData: ChainsTypeKeyStr = {
        sol: addressSol
    }

    // 定义一个对象，用于根据链类型获取连接钱包的函数
    const connectWalletStore: ChainsTypeKeyFun = {
        sol: connectWalletSolStore,
    }

    // 定义一个对象，用于根据链类型获取断开钱包连接的函数
    const disconnectWalletStore: ChainsTypeKeyFun = {
        sol: disconnectWalletSolStore,
    }

    // 定义一个对象，用于根据链类型获取provider
    const providerData = {
        sol: providerSol,
    }

    // 定义一个对象，用于根据链类型获取钱包名称
    const walletNameData = {
        sol: walletNameSol,
    }

    // 返回与指定链相关的钱包连接信息和操作函数
    return {
        address: addressData[chain],
        connectWalletStore: connectWalletStore[chain],
        disconnectWalletStore: disconnectWalletStore[chain],
        provider: providerData[chain],
        walletName: walletNameData[chain],
        connection: connection  // sol 专用
    }
}

export default useConnectWallet