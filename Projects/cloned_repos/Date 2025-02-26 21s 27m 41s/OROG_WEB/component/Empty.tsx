import React, { memo } from 'react'
import SvgIcon from './SvgIcon'
import { useTranslations } from 'next-intl'

/**
 * Empty组件用于在表格中没有数据时显示一个空状态图标和提示信息
 * 它通过使用SvgIcon组件来展示图标，并利用next-intl库进行国际化处理
 * @returns 返回一个包含空状态图标和提示信息的React组件
 */
const Empty = () => {
    // 使用next-intl的useTranslations钩子获取翻译函数
    // 'Table'表示当前翻译的命名空间
    const t = useTranslations('Table')

    // 渲染空状态界面
    return (
        // 使用 Tailwind CSS 类进行样式和布局设置
        <div className="mt-10 flex  flex-col items-center justify-center">
            <SvgIcon value="empty" className="w-32 text-2bc" />
            <div className="text-2bc">{t('noData')}</div>
        </div>
    )
}

// 将Empty组件导出，以便在其他地方使用
export default memo(Empty)