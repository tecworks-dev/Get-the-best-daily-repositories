/**
 * 设置钱包名称
 * @param walletName 钱包名称
 */
export const setLocal = (key: string, walletName: string) => {
  localStorage.setItem(key, walletName);
};

/**
 * 获取钱包名称
 * @returns 钱包名称
 */
export const getLocal = (key: string) => {
  return localStorage.getItem(key);
};

/**
 * 移除钱包名称
 */
export const removeLocal = (key: string) => {
  localStorage.removeItem(key);
};
