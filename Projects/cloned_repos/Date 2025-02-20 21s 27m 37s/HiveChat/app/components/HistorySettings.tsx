import React, { useState } from 'react';
import { Button, Segmented, InputNumber } from "antd";
import chatHistoryConfig from '@/app/store/chatHistoryConfig';
import { useTranslations } from 'next-intl';

const HistorySettings = (props: { chat_id: string, changeVisible: (open: boolean) => void }) => {
  const t = useTranslations('Chat');
  const { historyType, historyCount, setHistoryType, setHistoryCount } = chatHistoryConfig();
  const [value, setValue] = useState<string>(historyType);
  const [localCount, setLocalCount] = useState(historyCount);
  const saveConfig = () => {
    setHistoryType(props.chat_id, value as 'all' | 'none' | 'count');
    setHistoryCount(props.chat_id, localCount);
    props.changeVisible(false);
  }

  return (
    <div className='flex flex-col'>
      <Segmented options={[{
        label: t('historyMessageCountAll'),
        value: 'all'
      },
      {
        label: t('historyMessageCountNone'),
        value: 'none'
      },
      {
        label: t('historyMessageCountSpecify'),
        value: 'count'
      }
      ]} value={value} onChange={setValue} />
      {value === 'all' && (
        <p className='text-sm text-gray-500 mt-2 w-60 p-2'>{t('historyMessageCountAllNotice')}</p>
      )}
      {value === 'none' && (
        <p className='text-sm text-gray-500 mt-2 w-60 p-2'>{t('historyMessageCountNoneNotice')}</p>
      )}
      {value === 'count' && (
        <>
          <p className='text-sm text-gray-500 mt-2 w-60 p-2'>{t('historyMessageCountSpecifyNotice')}</p>
          <InputNumber style={{ marginLeft: '0.5rem' }} min={1} max={30} onChange={(value) => setLocalCount(value || 1)} defaultValue={localCount} />
        </>
      )}
      <div className='flex justify-end'>
        <Button type='default' size='small' className='mt-2' onClick={() => props.changeVisible(false)}>{t('cancel')}</Button>
        <Button type='primary' size='small' className='mt-2 ml-2' onClick={saveConfig}>{t('save')}</Button>
      </div>

    </div>
  )
}

export default HistorySettings
