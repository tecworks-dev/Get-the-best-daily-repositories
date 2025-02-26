import { useEffect, useRef } from "react";

/**
 * 自定义 Hook：useUpdateEffect
 * 作用：类似 useEffect，但 **不会在组件首次渲染时执行**，仅在依赖更新时触发。
 *
 * @param effect - 副作用函数，依赖项更新时触发
 * @param deps - 依赖项数组，依赖变化时执行 effect（与 useEffect 一致）
 */
const useUpdateEffect = (effect: React.EffectCallback, deps?: React.DependencyList) => {
    // 通过 useRef 记录是否是首次渲染
    const isFirstRender = useRef(true);

    useEffect(() => {
        // 如果是首次渲染，则跳过 effect 执行，并标记已渲染
        if (isFirstRender.current) {
            isFirstRender.current = false;
            return;
        }

        // 非首次渲染时执行 effect
        return effect();
    }, deps);
};

export default useUpdateEffect;