import { ChainsTypeKeyArr } from "@/interface";
import { create } from "zustand";

// 定义搜索历史记录的数据结构类型
export type SearchHistoryType = {
  logo: string; // 代币的 logo
  symbol: string; // 代币的符号
  address: string; // 市场地址
  quote_mint_address: string; // 代币地址
};

// 定义状态管理的接口
interface SearchHistoryState {
  searchHistory: ChainsTypeKeyArr<SearchHistoryType>; // 存储按链分类的搜索历史
  setSearchHistoryLocal: (
    chain: keyof ChainsTypeKeyArr<SearchHistoryType>,
    value: SearchHistoryType
  ) => void; // 更新指定链的搜索历史
  removeSearchHistoryLocal: (
    chain: keyof ChainsTypeKeyArr<SearchHistoryType>
  ) => void; // 移除指定链或所有链的搜索历史
}

// 创建并导出搜索历史状态管理器
export const useSearchHistoryState = create<SearchHistoryState>((set, get) => {
  // 初始化搜索历史，尝试从 localStorage 中获取已有数据
  let searchHistory: ChainsTypeKeyArr<SearchHistoryType> = { sol: [] };

  if (typeof window !== "undefined") {
    searchHistory = JSON.parse(
      localStorage.getItem("searchHistory") || JSON.stringify(searchHistory)
    );
  }

  return {
    // 初始化 Zustand 的搜索历史状态
    searchHistory,

    // 更新指定链的搜索历史
    setSearchHistoryLocal: (chain, value) => {
      if (typeof window !== "undefined") {
        const currentSearchHistory = get().searchHistory;

        // 确保链存在于搜索历史中
        if (!currentSearchHistory[chain]) {
          currentSearchHistory[chain] = [];
        }

        // 更新链的搜索历史
        const chainHistory = currentSearchHistory[chain];
        let newChainHistory;

        // 检查记录是否已经存在
        const existingIndex = chainHistory.findIndex(
          (item) => item.address === value.address
        );

        if (existingIndex !== -1) {
          // 如果记录已存在，将其移到第一位
          const [existingRecord] = chainHistory.splice(existingIndex, 1);
          newChainHistory = [existingRecord, ...chainHistory];
        } else {
          // 如果记录不存在，添加到第一位
          if (chainHistory.length >= 10) {
            // 如果链的搜索历史超过 9 项，移除最早的记录
            const newArr = chainHistory.slice(0, -1); // 去掉最后一个元素
            newChainHistory = [value, ...newArr];
          } else {
            newChainHistory = [value, ...chainHistory];
          }
        }

        // 更新状态
        const updatedHistory = {
          ...currentSearchHistory,
          [chain]: newChainHistory,
        };

        set({ searchHistory: updatedHistory });

        // 保存到 localStorage
        localStorage.setItem("searchHistory", JSON.stringify(updatedHistory));
      }
    },

    // 移除指定链或所有链的搜索历史
    removeSearchHistoryLocal: (chain) => {
      if (typeof window !== "undefined") {
        let updatedHistory: ChainsTypeKeyArr<SearchHistoryType>;
        updatedHistory = { ...get().searchHistory, [chain]: [] };
        set({ searchHistory: updatedHistory });

        // 更新 localStorage
        localStorage.setItem("searchHistory", JSON.stringify(updatedHistory));
      }
    },
  };
});
