'use client'
import React, { memo, useEffect, useRef } from "react";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    BarController, // 需要注册的模块
    Title,
    Tooltip,
    Legend,
} from "chart.js";
import { useThemeStore } from "@/store/theme";

// 注册 Chart.js 模块
ChartJS.register(BarController, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);
const Chart = ({ dataValues }: { dataValues: number[] }) => {

    const { theme } = useThemeStore()
    const chartRef = useRef<HTMLCanvasElement>(null);
    useEffect(() => {
        if (!chartRef.current || !dataValues.length) return;
        const ctx = chartRef.current.getContext("2d");
        chartRef.current.style.width = `${(dataValues.length * 4) / 4}rem`
        chartRef.current.style.maxWidth = 375 + "px"
        if (!ctx) return;
        // 根据数据值动态设置柱子的颜色
        const backgroundColors = dataValues.map(() => theme === 'dark' ? "#6fff89" : '#2bc347');

        // 初始化 Chart.js 实例
        const chartInstance = new ChartJS(ctx, {
            type: "bar", // 柱状图类型
            data: {
                labels: dataValues.map(() => ""), // 空标签
                datasets: [
                    {
                        data: dataValues,
                        backgroundColor: backgroundColors,
                        borderWidth: 0, // 去掉柱子的边框
                        barPercentage: 0.6, // 控制柱子的宽度
                        categoryPercentage: 0.8, // 控制类别的宽度
                        borderRadius: 10, // 设置柱子的圆角
                    },
                ],
            },
            options: {
                plugins: {
                    legend: { display: false }, // 隐藏图例
                    tooltip: { enabled: false }, // 禁用工具提示
                },
                scales: {
                    x: {
                        ticks: { display: false },
                        display: false, // 隐藏 Y 轴

                    },
                    y: {
                        ticks: { display: false },
                        display: false, // 隐藏 Y 轴
                    },
                },
                responsive: true,
                maintainAspectRatio: false, // 允许调整宽高
            },

        });

        // 清除实例以防止重复渲染
        return () => {
            chartInstance.destroy();
        };
    }, [theme, dataValues]);

    return <canvas ref={chartRef} className="h-32" />;
};

export default memo(Chart);