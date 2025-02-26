import { create } from "zustand";
import Cookies from "js-cookie"; // 引入 js-cookie 库

// 定义主题状态接口
interface ThemeState {
  theme: "light" | "dark"; // 当前主题类型
  setThemeAndCookie: (newTheme: "light" | "dark") => void; // 切换主题函数
}
// 创建并导出主题状态管理器
export const useThemeStore = create<ThemeState>((set, get) => {
  return {
    theme: (Cookies.get("theme") as "light" | "dark") || "dark", // 设置初始值
    setThemeAndCookie: (newThemeValue) => {
      if (typeof window !== "undefined") {
        const currentTheme = get().theme; // 获取当前主题
        if (newThemeValue === currentTheme) return;
        // 更新 HTML 类名
        document.documentElement.className = newThemeValue;
        // 更新 Cookie，设置一年过期
        Cookies.set("theme", newThemeValue, { path: "/", expires: 365 });
        localStorage.setItem("theme", newThemeValue);
        // 更新 Zustand 状态
        set({ theme: newThemeValue });
      }
    },
  };
});
