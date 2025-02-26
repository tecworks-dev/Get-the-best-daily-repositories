import { create } from "zustand";

/**
 * 定义移除关注状态的接口
 */
interface RemoveFollowState {
    // 存储待移除关注的地址列表
    follow: {
        removeFollow: string[],
        addFollow: string[];
    },
    // 添加待移除关注的地址
    addRemoveFollow: (followAddress: string) => void;
    // 移除待移除关注的地址
    removeRemoveFollow: (followAddress: string) => void;
}

// 创建并导出移除关注状态的管理器
export const useRemoveFollow = create<RemoveFollowState>((set, get) => {
    return {
        // 初始化移除关注的列表为空数组
        follow: {
            removeFollow: [],
            addFollow: [],
        },
        // 添加待移除关注的地址到列表中
        addRemoveFollow: (followAddress) => {
            // 获取当前的移除关注列表，如果未初始化，则设为空数组
            const currentRemoveFollow = get().follow || { removeFollow: [], addFollow: [] };
            // 更新状态，将新的关注地址添加到列表中
            set({ follow: { removeFollow: [...currentRemoveFollow.removeFollow, followAddress], addFollow: currentRemoveFollow.addFollow.filter((item) => item !== followAddress) } })
        },
        // 从待移除关注列表中移除指定的地址
        removeRemoveFollow: (followAddress) => {
            // 获取当前的移除关注列表，如果未初始化，则设为空数组
            const currentRemoveFollow = get().follow || { removeFollow: [], addFollow: [] };
            // 更新状态，过滤掉指定的关注地址
            set({ follow: { removeFollow: currentRemoveFollow.removeFollow.filter((item) => item !== followAddress), addFollow: [...currentRemoveFollow.addFollow, followAddress] } })
        },
    };
});