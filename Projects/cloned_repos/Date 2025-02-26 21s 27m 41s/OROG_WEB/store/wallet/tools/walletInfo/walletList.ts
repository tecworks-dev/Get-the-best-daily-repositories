// 可支持的钱包的列表
export const walletList: {
  [key: string]: {
    walletIs: string; // 检查是否有钱包 tp钱包检查是看ethereum.isTokenPocket 而连接仍是ethereum
    solana: string; // 初始化的sol
    download: string; // 下载链接
  };
} = {
  OkxWallet: {
    walletIs: "okxwallet",
    solana: "solana",
    download: "https://www.okx.com/download",
  },
  BitgetWallet: {
    walletIs: "bitkeep",
    solana: "solana",
    download: "https://web3.bitget.com/wallet-download",
  },
  Phantom: {
    walletIs: "phantom",
    solana: "solana",
    download: "https://phantom.com/download",
  },
};
