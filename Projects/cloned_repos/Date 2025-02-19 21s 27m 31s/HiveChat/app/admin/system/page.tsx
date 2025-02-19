'use client';
import React, { useState, useEffect } from 'react'
import { Switch, message } from "antd";
import { fetchAppSettings, adminAndSetAppSettings } from "./actions";
import { useTranslations } from 'next-intl';

const Userpage = () => {
  const t = useTranslations('Admin.System');
  const [isRegistrationOpen, setIsRegistrationOpen] = useState(false);
  useEffect(() => {
    const fetchSettings = async () => {
      const resultValue = await fetchAppSettings('isRegistrationOpen');
      if (resultValue !== 'true') {
        setIsRegistrationOpen(false);
      } else {
        setIsRegistrationOpen(true);
      }
    }
    fetchSettings();
  }, []);

  const onChange = async (checked: boolean) => {
    setIsRegistrationOpen(checked);
    const result = await adminAndSetAppSettings('isRegistrationOpen', String(checked))
    if (result.status === 'success') {
      message.success(t('saveSuccess'));
    } else {
      message.error(t('saveFail') + result.message);
    }
  };

  return (
    <div className='container max-w-3xl mb-6 px-4 md:px-0 pt-4 pb-8 h-auto'>
      <div className='h-4 w-full mb-10'>
        <h2 className="text-xl font-bold mb-4 mt-6">{t('system')}</h2>
      </div>
      <div className='flex flex-row justify-between mt-6 p-6 border border-gray-200 rounded-md'>
        <div className='flex flex-col '>
          <span className='text-sm'>{t('isRegistrationOpen')}</span>
          <span className='text-gray-400 text-xs'>{t('isRegistrationOpenDesc')}</span>
        </div>
        <div className='flex items-center'>
          <Switch
            onChange={onChange}
            value={isRegistrationOpen}
          />
        </div>
      </div>
    </div>
  )
}

export default Userpage