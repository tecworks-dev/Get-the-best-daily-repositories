import { defineRouting } from "next-intl/routing";
import { createNavigation } from "next-intl/navigation";

// 定义语言和链类型
export type LocalesType = "en" | "zh" | "vi";
export type ChainsType = "sol"

/**
 * 定义国际化路由配置
 */
export const routing = defineRouting({
  locales: ["en", "zh", "vi"],
  defaultLocale: "en",
});

/**
 * 从 URL 中解析 `chain` 参数
 */
const getChainFromUrl = (): ChainsType => {
  if (typeof window !== "undefined") {
    const urlSegments = window.location.pathname.split("/");
    const chainFromPath = urlSegments[2]; // 假设 chain 是 URL 的第二部分
    const validChains: ChainsType[] = ["sol"];
    if (validChains.includes(chainFromPath as ChainsType)) {
      return chainFromPath as ChainsType;
    }
  }
  return "sol"; // 如果未找到，则使用默认链 "sol"
};

/**
 * 包装路由工具，使其自动注入 `chain` 参数
 */
const {
  Link: OriginalLink,
  redirect: OriginalRedirect,
  useRouter: OriginalUseRouter,
  usePathname: OriginalUsePathname,
  getPathname: OriginalGetPathname,
} = createNavigation(routing);

// 手动定义 Link 的 Props 类型
interface OriginalLinkProps extends React.ComponentProps<typeof OriginalLink> { }

// 包装 Link 组件
const Link = (props: OriginalLinkProps) => {
  const chain = getChainFromUrl();
  const href =
    typeof props.href === "string"
      ? `/${chain}${props.href}` // 自动在路径前添加 chain
      : {
        ...props.href,
        pathname: `/${chain}${props.href.pathname}`, // 自动在对象路径前添加 chain
      };
  return <OriginalLink {...props} href={href} />;
};

// 包装 redirect 函数
const redirect = (
  url: string | { pathname: string; query?: Record<string, any> },
  ...args: any[]
) => {
  const chain = getChainFromUrl();

  // 确保 `url` 是符合 `OriginalRedirect` 类型的结构
  const newUrl =
    typeof url === "string"
      ? `/${chain}${url}` // 自动在路径前添加 chain
      : { ...url, pathname: `/${chain}${url.pathname}` };

  // 调用 `OriginalRedirect`，并传递符合类型的参数
  return OriginalRedirect(newUrl as any, ...args);
};

// 包装 useRouter 函数
const useRouter = () => {
  const originalRouter = OriginalUseRouter(); // 获取原始的 useRouter 实例
  const chain = getChainFromUrl(); // 动态获取当前链参数

  return {
    ...originalRouter,
    push: (
      url: string | { pathname: string; query?: Record<string, any> },
      ...args: any[]
    ) => {
      const newUrl =
        typeof url === "string"
          ? `/${chain}${url}` // 自动添加链参数到路径
          : { ...url, pathname: `/${chain}${url.pathname}` };

      return originalRouter.push(newUrl, ...args);
    },
    replace: (
      url: string | { pathname: string; query?: Record<string, any> },
      ...args: any[]
    ) => {
      const newUrl =
        typeof url === "string"
          ? `/${chain}${url}` // 自动添加链参数到路径
          : { ...url, pathname: `/${chain}${url.pathname}` };

      return originalRouter.replace(newUrl, ...args);
    },
  };
};
// 包装 usePathname 函数
const usePathname = () => {
  const originalPathname = OriginalUsePathname();
  const chain = getChainFromUrl();

  // 去掉路径中的 `chain` 前缀，返回开发者需要的实际路径
  return originalPathname.replace(`/${chain}`, "") || "/";
};
// 包装 getPathname 函数
const getPathname = (args: Parameters<typeof OriginalGetPathname>[0]) => {
  const originalPathname = OriginalGetPathname(args);
  const chain = getChainFromUrl();
  // 去掉路径中的 `chain` 前缀，返回开发者需要的实际路径
  return originalPathname.replace(`/${chain}`, "") || "/";
};
const useHistory = () => {
  const pathname = usePathname(); // 获取当前路径
  const router = useRouter(); // Next.js 路由

  return (str: string, page?: string) => {

    if (page) {
      if (pathname.includes(page)) {
        // 只更新 URL 参数，不跳转页面
        window.history.pushState(null, "", str);
      } else {
        console.log(new URLSearchParams(str).toString());

        // 这里改为正确的拼接方式
        router.push(`${page}/${str}`);
      }
    } else {
      window.history.pushState(null, "", str);
    }
  };
}

/**
 * 导出包装后的路由工具
 */
export { Link, redirect, useRouter, usePathname, getPathname, useHistory };
