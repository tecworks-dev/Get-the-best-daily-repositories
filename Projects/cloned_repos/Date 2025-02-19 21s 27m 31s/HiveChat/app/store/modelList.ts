import { create } from 'zustand';
import { LLMModel, LLMModelProvider } from '@/app/adapter/interface';
import { llmModelType, llmModelTypeWithAllInfo } from '@/app/db/schema';

interface IModelListStore {
  currentModel: LLMModel;
  providerList: LLMModelProvider[];
  providerListByKey: { [key: string]: LLMModelProvider } | null;
  allProviderListByKey: { [key: string]: LLMModelProvider } | null;
  allProviderList: LLMModelProvider[];
  modelList: LLMModel[];
  isPending: Boolean;
  initModelList: (initModels: llmModelTypeWithAllInfo[]) => Promise<void>;
  initAllProviderList: (initModels: LLMModelProvider[]) => Promise<void>;
  toggleProvider: (providerId: string, selected: boolean) => Promise<void>;
  changeSelect: (modelId: string, selected: boolean) => Promise<void>;
  addCustomModel: (model: LLMModel) => Promise<void>;
  updateCustomModel: (modelId: string, model: LLMModel) => Promise<void>;
  deleteCustomModel: (modelId: string) => Promise<void>;
  setCurrentModel: (model: string) => void
}

const useModelListStore = create<IModelListStore>((set, get) => ({
  currentModel: {
    id: 'gpt-4o',
    displayName: 'GPT 4o',
    supportVision: true,
    maxTokens: 131072,
    selected: true,
    provider: {
      id: 'openai',
      providerName: 'Open AI',
    }
  },
  providerList: [],
  providerListByKey: null,
  allProviderListByKey: null,
  allProviderList: [],
  modelList: [],
  isPending: true,
  initModelList: async (initModels: llmModelTypeWithAllInfo[]) => {
    const newData = initModels.map((model) => ({
      id: model.name,
      displayName: model.displayName,
      maxTokens: model.maxTokens || undefined,
      supportVision: model.supportVision || undefined,
      selected: model.selected || false,
      type: model.type ?? 'default',
      provider: {
        id: model.providerId,
        providerName: model.providerName,
      }
    }));

    const providerList = Array.from(
      new Map(
        initModels.map((model) => [
          model.providerId,
          {
            id: model.providerId,
            providerName: model.providerName,
            providerLogo: model.providerLogo,
            status: true,
          }
        ])
      ).values()
    );

    set((state) => ({
      ...state,
      providerList,
      modelList: newData,
      isPending: false,
    }));

  },
  initAllProviderList: async (providers: LLMModelProvider[]) => {
    const providerByKey = providers.reduce<{ [key: string]: LLMModelProvider }>((result, provider) => {
      result[provider.id] = provider;
      return result;
    }, {});

    set((state) => ({
      ...state,
      allProviderList: providers,
      allProviderListByKey: providerByKey,
    }));
  },
  setCurrentModel: (modelId: string) => {
    set((state) => {
      // 检查新模型是否与当前模型相同
      if (state.currentModel?.id !== modelId) {
        const modelInfo = state.modelList.find(m => m.id === modelId);
        if (modelInfo) {
          localStorage.setItem('lastSelectedModel', modelInfo.id);
          return {
            ...state,
            currentModel: modelInfo,
          };
        } else {
          return state;
        }
      }
      return state; // 如果相同，则返回当前状态
    });
  },

  toggleProvider: async (providerId: string, selected: boolean) => {
    set((state) => ({
      ...state,
      allProviderList: state.allProviderList.map((item) =>
        item.id === providerId ? { ...item, status: selected } : item
      ),
    }));
  },

  changeSelect: async (modelId: string, selected: boolean) => {
    set((state) => ({
      ...state,
      modelList: state.modelList.map((model) =>
        model.id === modelId ? { ...model, selected } : model
      ),
    }));
  },
  addCustomModel: async (model: LLMModel) => {
    // 更新状态中的 modelList
    set((state) => ({
      ...state,
      modelList: [...state.modelList, model],
    }));
  },
  updateCustomModel: async (modelId: string, model: LLMModel) => {
    // 更新状态中的 modelList
    set((state) => ({
      ...state,
      modelList: state.modelList.map((existingModel) =>
        existingModel.id === modelId ? { ...existingModel, ...model } : existingModel
      ),
    }));
  },

  deleteCustomModel: async (modelId: string) => {
    set((state) => ({
      ...state,
      modelList: state.modelList.filter((model) => model.id !== modelId),
    }));
  }
}));

export default useModelListStore;