// 导入必要的库和组件
'use client'
import {
    memo,
    useState,
} from 'react';

import {
    Dropdown,
    DropdownProps,
} from 'antd';
import { useLocale } from 'next-intl';
import dynamic from 'next/dynamic';

import useSetChainLocale from '@/hook/useSetChainLocale';
import useUpdateEffect from '@/hook/useUpdateEffect';
import { useThemeStore } from '@/store/theme';

const SvgIcon = dynamic(() => import('@/component/SvgIcon'));

// 定义支持的语言列表
const localeData = [
    {
        label: 'English',
        key: 'en'
    },
    {
        label: '中文',
        key: 'zh'
    },
    {
        label: 'Tiếng Việt',
        key: 'vi'
    }
];

// 定义媒体和反馈链接列表
const media = [
    {
        text: <SvgIcon className="w-5" value={'gitHub'} />,
        href: 'javascript:;'
    },
    {
        text: <SvgIcon className="w-5" value={'x'} />,
        href: 'javascript:;'
    },
    {
        text: <SvgIcon className="w-5" value={'telegram'} />,
        href: "javascript:;"
    }
];

/**
 * 底部栏组件，包含语言切换、主题切换和媒体链接
 * @param {Object} props - 组件属性
 * @param {('dark' | 'light')} props.themeSer - 服务端设定的主题模式
 */
const Foot = ({ themeSer }: { themeSer: 'dark' | 'light' }) => {
    const [isThrottle, setIsThrottle] = useState(false);
    // 获取当前选择的语言
    const locale = useLocale();
    // 控制语言菜单的展开状态
    const [open, setOpen] = useState(false);
    // 控制主题模式的显示
    const [themeMode, setThemeMode] = useState<'dark' | 'light'>(themeSer);
    // 从主题存储中获取设置主题和cookie的方法
    const { setThemeAndCookie, theme } = useThemeStore();
    const { switchLocale } = useSetChainLocale()
    /**
     * 处理语言菜单展开状态变化
     * @param {boolean} nextOpen - 下一个展开状态
     * @param {Object} info - 附加信息
     */
    const handleOpenChange: DropdownProps['onOpenChange'] = (nextOpen, info) => {
        if (info.source === 'trigger' || nextOpen) {
            setOpen(nextOpen);
        }
    };
    // 同步主题模式显示与存储中的主题
    useUpdateEffect(() => {
        document.body.classList.add("no-animation");
        setThemeMode(theme);
        setTimeout(() => {
            document.body.classList.remove("no-animation");
        }, 500);
    }, [theme]);
    // 根据语言键获取当前语言的标签
    const currentLocaleLabel = localeData.find(item => item.key === locale)?.label;

    // 渲染底部栏组件
    return (
        <div className="fixed bottom-0 left-0  flex items-center justify-between box-border pl-12.25 pr-6.25 w-full h-15 dark:bg-black bg-e6 z-40">
            <div className="flex items-center transition-all">
                <SvgIcon onClick={() => !isThrottle && setThemeAndCookie('light')} value="light" className={`w-6 duration-500 ${themeMode === 'light' ? 'dark:text-d3 text-6d' : 'dark:text-6b text-d3'} mr-8 cursor-pointer`} />
                <SvgIcon onClick={() => !isThrottle && setThemeAndCookie('dark')} value="dark" className={`w-6 duration-500 ${themeMode === 'dark' ? 'dark:text-d3 text-6d' : 'dark:text-6b text-d3'} cursor-pointer mr-18.75`} />
                <Dropdown menu={{ items: localeData }}
                    trigger={['click']}
                    open={open}
                    placement="bottom"
                    onOpenChange={handleOpenChange}
                    dropdownRender={() => (
                        <div className='w-36 dark:bg-black bg-white dark:text-white text-black px-2 py-3 rounded-xl box-border'>
                            {localeData.map(({ key, label }) => (
                                <div
                                    onClick={(e) => switchLocale(key)}
                                    key={key}
                                    className={`px-2 py-2 mb-2 last:mb-0 flex items-center cursor-pointer dark:hover:bg-black1F hover:bg-f1 rounded-lg ${key === locale ? 'dark:bg-black1F bg-f1' : ''}`}
                                >
                                    {label}
                                </div>
                            ))}
                        </div>
                    )}
                >
                    <div className="flex items-center justify-center text-82 text-sm w-26.25 h-9 rounded-2.5xl dark:bg-333 bg-d3 cursor-pointer">
                        {currentLocaleLabel}
                        <SvgIcon value="arrowsBottom" className={`ml-2 text-82 w-3 ${open ? 'rotate-0' : 'rotate-180'}`} />
                    </div>
                </Dropdown>
            </div>
            <div className="text-82  flex items-center">
                {media.map(({ text, href }, index) => (
                    <a href={href} className="mr-7.5 dark:hover:text-white hover:text-black" target="_blank" key={index}>
                        {text}
                    </a>
                ))}
            </div>
        </div>
    );
};

// 导出底部栏组件
export default memo(Foot);