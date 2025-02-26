import { WindowWithWallets } from "@/interface";
import { walletList } from "../walletInfo/walletList";

/**
 * 监听钱包切换地址事件
 *
 * 此函数用于监听指定钱包的地址切换事件，当检测到账户变化时，会调用传入的回调函数
 * 主要用于在前端应用中实时响应钱包的地址变化，确保应用的正常运行
 *
 * @param walletName 钱包名称，用于从walletList中获取对应的钱包配置信息
 * @param fun 可选参数，当检测到账户变化时要执行的回调函数
 */
export const watchSetAddressSol = async (walletName: string, fun: Function) => {
  try {
    // 获取指定钱包的配置信息
    const walletInfo = walletList[walletName];
    // 检查钱包信息是否存在且钱包已在窗口对象中定义
    if (walletInfo && walletInfo.walletIs in window) {
      // 从窗口对象中获取指定钱包的Solana提供者
      const providerSol = (window as WindowWithWallets)[walletInfo.walletIs][
        walletInfo.solana
      ];
      // 监听账户变化事件，当账户变化时调用传入的回调函数
      providerSol.once("accountChanged", (publicKey: any) => {
        if (publicKey) {
          fun && fun();
        }
      });
    }
  } catch (e) {
    // 错误处理：打印监听账户变化过程中的错误
    console.log("watchSetChain", e);
  }
};
