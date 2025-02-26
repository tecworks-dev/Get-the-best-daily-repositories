import { WindowWithWallets } from "@/interface";
import { walletList } from "../walletInfo/walletList";

/**
 * 连接到指定的钱包
 *
 * 此函数尝试连接到用户指定的钱包，如果钱包已安装在浏览器中，则初始化provider和signer对象
 * 如果钱包未安装，则重定向用户到钱包的下载页面
 *
 * @param walletName 需要连接的钱包名称，用于索引钱包列表
 * @returns 返回一个包含地址、签名者、提供者和链ID的对象如果连接失败或钱包不在窗口中，则不返回任何内容
 */
export const connectedWalletSol = async (walletName: string) => {
  try {
    // 获取指定钱包的配置信息
    const walletInfo = walletList[walletName];

    // 检查钱包信息是否存在且钱包已在窗口对象中定义
    if (walletInfo && walletInfo.walletIs in window) {
      // 从窗口对象中获取指定钱包的Solana提供者
      const providerSol = (window as WindowWithWallets)[walletInfo.walletIs][
        walletInfo.solana
      ];
      // 连接到钱包并获取响应
      const resp = await providerSol.connect();

      // 获取钱包的公共地址
      const addressSol = resp.publicKey.toString();
      // 返回包含地址、签名者和提供者的对象
      return {
        addressSol,
        providerSol,
      };
    } else {
      // 如果钱包未安装，打开钱包的下载页面或默认下载页面
      window.open(walletInfo?.download);
    }
  } catch (e) {
    // 如果连接过程中发生错误，打印错误信息
    console.log(`EVM Error connecting to wallet ${walletName}:`, e);
  }
};
