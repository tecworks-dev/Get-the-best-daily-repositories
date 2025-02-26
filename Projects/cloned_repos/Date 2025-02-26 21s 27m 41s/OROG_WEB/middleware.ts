import createMiddleware from "next-intl/middleware";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { LocalesType, routing } from "./i18n/routing";

// 创建国际化中间件
const intlMiddleware = createMiddleware(routing);

export default function middleware(request: NextRequest) {
  const url = new URL(request.url);

  // 忽略静态文件和 API 路径
  if (
    url.pathname.startsWith("/_next/") ||
    url.pathname.startsWith("/api") ||
    url.pathname.startsWith("/public") ||
    url.pathname.startsWith("/charting_library") ||
    url.pathname === "/favicon.ico" // 静态资源扩展名
  ) {
    return NextResponse.next(); // 不处理静态文件和 API 请求
  }

  // 处理国际化逻辑
  const response = intlMiddleware(request);

  // 从 URL 中解析路径
  const segments = url.pathname.split("/").filter(Boolean); // 移除空段
  const validChains = ["sol"]; // 支持的链
  const localeFromUrl = segments[0]; // 路径的第一段是语言环境
  const chainFromUrl = segments[1]; // 路径的第二段是链

  // 从 Cookies 中获取链参数
  const chainFromCookie = request.cookies.get("chain")?.value;

  // 验证链是否有效
  const chain = validChains.includes(chainFromUrl)
    ? chainFromUrl
    : validChains.includes(chainFromCookie || "")
    ? chainFromCookie!
    : "sol"; // 默认链

  // 如果 URL 中没有语言环境或链参数，重定向到默认路径
  if (
    !routing.locales.includes(localeFromUrl as LocalesType) ||
    !validChains.includes(chainFromUrl)
  ) {
    const newPathname = `/${routing.defaultLocale}/${chain}/${segments
      .slice(localeFromUrl ? 2 : 0)
      .join("/")}`;
    url.pathname = newPathname;

    const redirectResponse = NextResponse.redirect(url);
    redirectResponse.cookies.set("chain", chain, {
      maxAge: 60 * 60 * 24 * 365,
    }); // 存储一年
    return redirectResponse;
  }

  // 同步链参数到 Cookies，并设置存储时间为一年
  if (chain !== chainFromCookie) {
    response.cookies.set("chain", chain, { maxAge: 60 * 60 * 24 * 365 }); // 存储一年
  }
  // 处理主题逻辑
  const theme = request.cookies.get("theme")?.value || "dark"; // 默认主题
  // 设置主题到响应头，供前端读取
  response.headers.set("X-Theme", theme);

  return response;
}

// 配置国际化中间件的匹配器
export const config = {
  matcher: [
    "/", // 根路径
    "/:locale", // 语言环境路径
    "/:locale/:chain/:path*", // 带链的路径
  ],
};
