'use client';
import React, { useEffect } from 'react'
import App from "@/app/components/App";
import useModelListStore from '@/app/store/modelList';
import { fetchAvailableLlmModels, fetchAllProviders } from '@/app/adapter/actions';

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { initModelList, setCurrentModel, initAllProviderList } = useModelListStore();

  useEffect(() => {
    const initializeModelList = async () => {
      try {
        const remoteModelList = await fetchAvailableLlmModels();
        const modelNames = remoteModelList.map(model => model.name);
        await initModelList(remoteModelList);
        // -----
        const allProviderSettings = await fetchAllProviders();
        const processedList = allProviderSettings.map(item => ({
          id: item.provider,
          providerName: item.providerName,
          providerLogo: item.logo || '',
          status: item.isActive || false,
        }));
        initAllProviderList(processedList)
        // -----
        const lastSelectedModel = localStorage.getItem('lastSelectedModel');
        if (lastSelectedModel && modelNames.includes(lastSelectedModel)) {
          setCurrentModel(lastSelectedModel);
        }
        else {
          if (remoteModelList.length > 0) {
            setCurrentModel(remoteModelList[0].name);
          }
        }
      } catch (error) {
        console.error('Error initializing model list:', error);
      }
    };

    initializeModelList();
  }, [initModelList, setCurrentModel, initAllProviderList]);
  return (
    <div className="flex flex-col h-dvh">
      <App>{children}</App>
    </div>
  )
}