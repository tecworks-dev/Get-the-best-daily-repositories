'use client'
import React from 'react'
import { Select, ConfigProvider } from "antd";
import { Avatar } from "antd";
import useModelListStore from '@/app/store/modelList';
import { useTranslations } from 'next-intl';

const ModelSelect = () => {
  const { modelList, currentModel, allProviderListByKey, providerList, isPending, setCurrentModel } = useModelListStore();
  const t = useTranslations('Chat');
  const handleChangeModel = (value: string) => {
    setCurrentModel(value);
  };

  if (isPending) {
    return (
      <div className='ml-2 flex flex-row items-center'>
        <span className='ml-2 text-sm text-gray-500'>Loading</span>
      </div>
    );
  }

  if (providerList.length === 0) {
    return (
      <div className='ml-2 flex flex-row items-center'>
        <Avatar
          size={20}
          src='/images/providers/openai.svg'
        />
        <span className='ml-2 text-sm text-gray-500'>{t('modelNotConfigured')}</span>
      </div>
    );
  }
  const options = providerList.map((provider) => {
    return {
      label: <span>{provider.providerName}</span>,
      title: provider.providerName,
      options: modelList.filter((model) => model.provider.id === provider.id && model.selected).map((model) => ({
        label: (<div className='flex flex-row items-center'>
          <Avatar
            size={20}
            src={allProviderListByKey && allProviderListByKey[provider.id]?.providerLogo || ''}
          />
          <span className='ml-1'>{model.displayName}</span>
        </div>),
        value: model.id,
      }))
    }
  });

  return (
    <ConfigProvider
      theme={{
        components: {
          Select: {
            selectorBg: 'transparent',
            activeBorderColor: 'transparent',
            activeOutlineColor: 'transparent',
            hoverBorderColor: 'transparent',
            colorBorder: 'transparent',
            multipleSelectorBgDisabled: 'transparent',
          },
        },
      }}
    >
      <Select
        value={currentModel?.id}
        style={{ width: 230, border: 'none', backgroundColor: 'transparent' }}
        onChange={handleChangeModel}
        listHeight={320}
        options={options}
      />
    </ConfigProvider>
  );
}

export default ModelSelect