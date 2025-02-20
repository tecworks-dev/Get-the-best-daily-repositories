import { create } from 'zustand';

interface ISidebarCollapsedStore {
  isSidebarCollapsed: boolean;
  setIsSidebarCollapsed: (value: boolean) => void;
  toggleSidebar: () => void;
}

const useSidebarCollapsedStore = create<ISidebarCollapsedStore>((set) => ({
  isSidebarCollapsed: false,
  setIsSidebarCollapsed: (value: boolean) => {
    set({ isSidebarCollapsed: value });
  },
  toggleSidebar: () => {
    set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed }));
  },
}))

export default useSidebarCollapsedStore
