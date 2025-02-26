// 使用 Next.js 的 'use client' 指令，表明这个组件需要在客户端渲染
'use client'
// 导入自定义的登录钩子
import useLogin from '@/hook/useLogin'
// 导入 ChainsType 类型，用于表示链的类型
import { ChainsType } from '@/i18n/routing'
// 导入 Next.js 的国际化钩子
import { useTranslations } from 'next-intl'
// 导入 Next.js 的路由参数钩子
import { useParams } from 'next/navigation'
// 导入 React
import React, { memo } from 'react'

/**
 * LinkComponent 组件用于渲染一个登录链接
 * 该组件根据当前链的类型动态渲染，并提供一个登录操作
 * 
 * @returns 返回一个登录链接组件
 */
const LinkComponent = () => {
    // 使用国际化钩子获取翻译函数
    const t = useTranslations('Toast')
    // 使用路由参数钩子获取当前链的类型
    const { chain }: { chain: ChainsType } = useParams()
    // 使用登录钩子获取登录检查函数
    const { checkExecution } = useLogin(chain)

    // 渲染一个登录链接，点击时触发登录检查
    return (
        <div className=' w-full flex items-center justify-center mt-11' >
            <div onClick={() => checkExecution()} className='text-bf px-4 py-1 border-2 border-77 rounded-full cursor-pointer'>{t('pleaseLogin')}</div>
        </div>
    )
}

// 导出 LinkComponent 组件
export default memo(LinkComponent)