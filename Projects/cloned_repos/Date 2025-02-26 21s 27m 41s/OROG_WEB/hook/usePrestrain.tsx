import { useRouter } from '@/i18n/routing'
import { useEffect } from 'react'

/**
 * 使用预取路由功能的自定义钩子
 * 该钩子用于在组件初始化时预取指定路由的数据，以优化性能和用户体验
 * 
 * @param href 需要预取的路由路径
 */
const usePrestrain = (href: string[]) => {
    // 获取Next.js路由器实例
    const router = useRouter()

    // 在组件挂载时执行路由预取操作
    useEffect(() => {
        href.forEach((item) => {
            router.prefetch(item)
        })
    }, []) // 仅在组件首次挂载时执行预取，不关注后续依赖变化
}
export default usePrestrain