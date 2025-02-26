import { create } from "zustand";

/**
 * 定义了应用中用于管理token状态的接口
 */
interface TokenState {
  token: string | null; // 当前的token值，可能为null
  setToken: (newToken: string) => void; // 更新token的函数
  setLocalToken: (newToken: string) => void; // 将token更新并持久化的函数
  removeLocalToken: () => void; // 删除本地存储的token的函数
}

/**
 * 创建一个用于管理主题状态的store
 * 使用了zustand库来管理状态，提供了token的更新和持久化方法
 *
 * @returns {TokenState} 包含了token状态和更新状态方法的对象
 */
export const useTokenStore = create<TokenState>((set) => {
  // 初始化 token，尝试从本地存储中获取
  let initialToken: string | null = null;
  if (typeof window !== "undefined") {
    initialToken = localStorage.getItem("token");
  }
  return {
    token: initialToken,
    setToken: (newToken) => set({ token: newToken }), // 更新token状态的函数
    setLocalToken: (newToken) => {
      // 当在浏览器环境中时，将token持久化到localStorage并更新状态
      if (typeof window !== "undefined") {
        localStorage.setItem("token", newToken);
        set({ token: newToken });
      }
    },
    removeLocalToken: () => {
      // 当在浏览器环境中时，移除localStorage中的token并更新状态
      if (typeof window !== "undefined") {
        localStorage.removeItem("token");
        set({ token: null });
      }
    },
  };
});
