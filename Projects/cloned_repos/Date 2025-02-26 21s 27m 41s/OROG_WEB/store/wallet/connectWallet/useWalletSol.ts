import { create } from 'zustand';

import {
  clusterApiUrl,
  Connection,
} from '@solana/web3.js';

import { connectedWalletSol } from '../tools/connected/connectedWalletSol';
import {
  removeLocal,
  setLocal,
} from '../tools/strage';
import { watchSetAddressSol } from '../tools/watch/watchSol';

/**
 * 定义一个接口，用于描述 Solana 钱包连接状态的结构
 */
interface ConnectWalletSolState {
  connectWalletSolStore: (
    walletName: string,
    watchChangeFun?: Function
  ) => Promise<void>; // 连接 Solana 钱包的函数,第二个参数是当钱包地址发生变化时，需要执行的函数
  disconnectWalletSolStore: () => void; // 断开 Solana 钱包连接的函数
  walletNameSol: string | null; // 当前连接的钱包名称
  addressSol: string; // 当前连接的钱包地址
  providerSol: any | null; // Solana 钱包的提供者对象
  connection: Connection | null; // Solana 网络
}

/**
 * 使用 Zustand 创建一个管理 Solana 钱包连接状态的 store
 * 该 store 提供了钱包的连接、断开和状态管理功能
 */
export const useConnectWalletSolStore = create<ConnectWalletSolState>(
  (set, get) => {
    // 初始化钱包名称变量
    let walletNameSol: string | null = "";
    /**
     * 连接指定名称的 Solana 钱包
     * @param walletName 钱包名称
     */
    const connectWalletSolStore = async (
      walletName: string,
      watchChangeFun?: Function
    ) => {
      try {
        // 调用工具函数连接钱包，获取地址和提供者
        const result = await connectedWalletSol(walletName);
        if (!result) {
          throw new Error("connectedWalletSol returned undefined");
        }
        const { addressSol, providerSol } = result;
        const network = clusterApiUrl("mainnet-beta");
        const connection = new Connection(network, "confirmed");
        // 设置地址切换监听器，确保地址变化时能够自动重新连接
        await watchSetAddressSol(walletName, async () => {
          // 用户地址出现变化后，需要重新进行签名，但是用户的地址是一直存在的，只是发生变化，而token是会清空的，如果token先被清空的话，需要重新进行签名，然后地址发生更新，又进行签名，会出现两次
          // 所以需要先改变地址，让后进行删除token，这样才不会出现两次签名
          await connectWalletSolStore(walletName, watchChangeFun);
          watchChangeFun && watchChangeFun();
        });
        // 将钱包名称存储到本地存储中
        setLocal("walletSol", walletName);

        // 更新 Zustand 的状态
        set({ addressSol, providerSol, walletNameSol: walletName, connection });
      } catch (e) {
        console.log("error-connectWalletSolStore", e);
      }
    };

    /**
     * 断开当前连接的 Solana 钱包
     */
    const disconnectWalletSolStore = () => {
      try {
        // 获取当前提供者并调用断开连接的方法
        const providerSol = get().providerSol;
        providerSol.disconnect();

        // 移除本地存储中的钱包名称
        removeLocal("walletSol");

        // 重置 Zustand 的状态
        set({
          addressSol: "",
          providerSol: null,
          walletNameSol: "",
          connection: null,
        });
      } catch (e) {
        console.log("error-disconnectWalletSolStore", e);
      }
    };

    return {
      connectWalletSolStore, // 连接钱包方法
      addressSol: "", // 钱包地址，初始为空
      walletNameSol: walletNameSol, // 钱包名称，初始为空
      disconnectWalletSolStore, // 断开钱包方法
      providerSol: null, // 钱包提供者，初始为 null
      connection: null, // Solana 网络 连接，初始为 null
    };
  }
);
