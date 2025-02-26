import { getRequestConfig } from "next-intl/server"; // 从 next-intl/server 模块导入 getRequestConfig 函数
import { routing } from "./routing"; // 从当前目录下的 routing 文件导入 routing 对象

// 导出默认的请求配置，使用 getRequestConfig 包装一个异步函数
export default getRequestConfig(async ({ requestLocale }) => {
  // 这通常对应于 URL 中的 [locale] 段
  let locale = await requestLocale; // 获取请求的语言环境（locale）

  // 确保使用有效的语言环境
  if (!locale || !routing.locales.includes(locale as "en" | "zh" | "vi")) {
    locale = routing.defaultLocale; // 如果语言环境无效，则使用默认语言环境
  }

  return {
    locale, // 返回语言环境
    messages: (await import(`../messages/${locale}.json`)).default, // 动态导入对应语言环境的消息文件，并返回其默认导出
  };
});
