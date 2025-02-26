import { PublicKey } from '@solana/web3.js';

/**
 * 获取Solana区块链中指定地址的余额
 *
 * 此函数通过连接Solana区块链网络来查询指定地址的余额它首先检查是否存在一个有效的连接，
 * 然后尝试获取余额如果获取过程中没有遇到错误，则返回余额值；如果遇到错误，则打印错误信息并返回0
 *
 * @param connection 与Solana区块链的连接对象，用于查询余额
 * @param address 需要查询余额的Solana钱包地址
 * @returns 返回查询到的余额值如果连接无效或查询过程中遇到错误，则返回0
 */
const getBalanceSol = async (connection: any, address: string) => {
  // 检查是否存在一个有效的连接对象
  if (connection) {
    try {
      // 将字符串形式的地址转换为PublicKey对象，以便Solana网络可以识别和处理
      const walletAddress = new PublicKey(address);
      // 通过连接对象异步获取指定地址的余额
      const balance = await connection.getBalance(walletAddress);
      // 如果成功获取到余额，返回该余额值
      return balance;
    } catch (e) {
      // 如果在获取余额的过程中遇到错误，打印错误信息并返回0
      BigInt(0);
      throw new Error(`getBalanceSol error  ${e}`);
    }
  }
  // 如果没有有效的连接对象，返回0
  return BigInt(0);
};

export default getBalanceSol;
