import { ChainsType } from '@/i18n/routing';
import Cookies from 'js-cookie';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';

/**
 * 自定义钩子用于设置区块链类型和语言本地化
 */
const useSetChainLocale = () => {
    // 用于导航的路由器
    const router = useRouter();
    // 当前路径名
    const pathname = usePathname();
    // 查询参数
    const searchParams = useSearchParams();

    /**
     * 改变区块链类型并更新Cookie和URL
     * @param {ChainsType} newChain - 新选择的区块链类型
     */
    const setChainAndCookie = (newChain: ChainsType) => {
        // 将新的区块链类型保存到本地存储
        localStorage.setItem("chain", newChain);
        // 将新的区块链类型保存到 Cookie，设置过期时间为 365 天
        Cookies.set("chain", newChain, { expires: 365 });
        // 构建新的URL
        const params = searchParams.toString();
        const url = new URL(window.location.href);
        // 替换或插入新的 chain
        const updatedPathname = `/${newChain}, '')}`;
        // 保留查询参数
        if (params) url.search = params;
        // 更新路径
        url.pathname = updatedPathname;
        // 推送新的 URL 到路由器
        router.push(url.toString());
    }

    /**
     * 切换语言
     * @param {string} newLocale - 新选择的语言键
     */
    const switchLocale = (newLocale: string) => {
        // 构建新的URL
        const params = searchParams.toString();
        const url = new URL(window.location.href);
        url.pathname = `/${newLocale}${pathname.replace(/^\/[a-zA-Z-]+/, '')}`;
        // 保留查询参数
        if (params) url.search = params;
        // 推送新的 URL 到路由器
        router.push(url.toString());
    };

    // 返回改变区块链类型和切换语言的函数
    return { setChainAndCookie, switchLocale }
}

// 导出自定义钩子
export default useSetChainLocale