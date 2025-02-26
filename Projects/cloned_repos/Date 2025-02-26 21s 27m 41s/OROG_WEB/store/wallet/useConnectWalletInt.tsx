import { ChainsType } from "@/i18n/routing";
import { useConnectWalletSolStore } from "./connectWallet/useWalletSol";
import { useEffect } from "react";
import { getLocal } from "./tools/strage";

/**
 * 自定义 Hook，用于在组件中初始化 Solana 钱包连接
 * @param chain 当前区块链类型
 */
export const useConnectWalletSol = (chain: ChainsType, removeLocalToken?: Function) => {
    const { connectWalletSolStore } = useConnectWalletSolStore();
    useEffect(() => {
        // 如果当前链是 Solana，则尝试从本地存储中加载钱包信息并自动连接
        if (chain === "sol") {
            const walletNameSolLocal = getLocal("walletSol");
            if (walletNameSolLocal) {
                connectWalletSolStore(walletNameSolLocal, removeLocalToken);
            }
        }
    }, []);
};
