// 定义一个用于客户端的组件，该组件将使用样式提供者和状态管理来渲染其子组件
'use client'
import { useThemeStore } from '@/store/theme';
// 从@ant-design/cssinjs中导入px2rem转换器和样式提供者
import { px2remTransformer, StyleProvider } from '@ant-design/cssinjs';
import { CheckSquareOutlined, CloseSquareOutlined, InfoCircleOutlined, WarningOutlined } from '@ant-design/icons';
import { ConfigProvider } from 'antd';
// 导入提示
import { ToastContainer } from 'react-toastify';

import { useCallback, useEffect } from "react";
import NProgress from "nprogress";
import "@/app/nprogress.css";
NProgress.configure({ showSpinner: false });
// 导入提示样式
import 'react-toastify/dist/ReactToastify.css';
// 定义一个px到rem的转换函数，假设16px等于1rem
const px1rem = px2remTransformer({
    rootValue: 16, // 16px = 1rem; @default 16
})
// 定义一个样式提供者组件，它将使用状态管理和样式转换来包装其子组件
const StyleProviderCom = ({ children }: { children: React.ReactNode }) => {
    const { theme } = useThemeStore();
    const handleStart = useCallback(() => NProgress.start(), []);
    const handleStop = useCallback(() => NProgress.done(), []);
    useEffect(() => {
        const originalFetch = window.fetch;
        let activeRequests = 0;

        window.fetch = async (...args) => {
            const [url, _options] = args;
            // 确保 URL 是字符串类型
            const urlString = typeof url === "string" ? url : url?.toString();
            // 仅拦截页面请求
            if (urlString.includes('/api/')) {
                // 这是 API 请求，跳过加载指示器
                return originalFetch(...args);
            }

            // 其他请求（如页面请求）显示加载进度条
            activeRequests++;
            handleStart()

            try {
                const response = await originalFetch(...args);
                return response;
            } finally {
                activeRequests--;
                if (activeRequests === 0) {
                    handleStop();
                }
            }
        };

        return () => {
            window.fetch = originalFetch;
        };
    }, []);

    return (
        <StyleProvider transformers={[px1rem]}>
            <ToastContainer stacked theme={theme}
                icon={({ type }) => {
                    // theme is not used in this example but you could
                    switch (type) {
                        case 'info':
                            return <InfoCircleOutlined className="dark:text-6f text-2bc" />;
                        case 'error':
                            return <CloseSquareOutlined className="stroke-red-500" />;
                        case 'success':
                            return <CheckSquareOutlined className="dark:text-6f text-2bc" />;
                        case 'warning':
                            return <WarningOutlined className="stroke-yellow-500" />;
                        default:
                            return null;
                    }
                }}
            />
            <ConfigProvider
                theme={{
                    components: {
                        Modal: {
                            /* 这里是你的组件 token */
                            contentBg: "transparent",
                        },
                        Tooltip: {
                            colorBgSpotlight: "var(--tooltip-bg-color)",
                            colorTextLightSolid: "var(--tooltip-text-color)",
                            paddingXS: 20,
                            paddingSM: 30
                        }
                    },
                }}
            >
                {children}
            </ConfigProvider>
        </StyleProvider>
    )
}

// 导出样式提供者组件作为默认导出
export default StyleProviderCom