'use client';
import React, { useState, useEffect } from 'react'
import { fetchAllLlmSettings } from '@/app/adapter/actions';
import { llmSettingsType } from '@/app/db/schema';
import { useTranslations } from 'next-intl';

const LLMSettings = () => {
  const t = useTranslations('Admin.Models');
  const [fetchStatus, setFetchStatus] = useState(true);
  const [llmSettingList, setLlmSettingList] = useState<llmSettingsType[]>([]);

  useEffect(() => {
    const fetchLlmList = async (): Promise<void> => {
      const result = await fetchAllLlmSettings();
      setLlmSettingList(result);
      setFetchStatus(false)
    };
    fetchLlmList();
  }, []);

  return (
    <div className='container flex flex-row max-w-3xl h-full items-center justify-center mb-6'>
      <h2 className="text-gray-500 mb-4">{t('modelSettingsNotice')}</h2>
    </div>
  )
}

export default LLMSettings