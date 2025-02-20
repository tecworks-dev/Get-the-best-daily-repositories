import React from 'react'
import { Avatar } from "antd";
import useModelListStore from '@/app/store/modelList';

const ProviderItem = (props: {
  className: string,
  data: {
    id: string;
    providerName: string;
    status?: boolean;
  }
}) => {
  const { allProviderListByKey } = useModelListStore();
  return (
    <div className={`flex flex-row items-center h-12 px-2 justify-between hover:bg-gray-200 cursor-pointer rounded-md ${props.className || ''}`}>
      <div className='flex flex-row items-center'>
        <Avatar
              style={{ border:'1px solid #ddd', padding: '0.2rem' }}
              src={allProviderListByKey && allProviderListByKey[props.data.id]?.providerLogo || ''}
            />
        <span className='ml-2'>{props.data?.providerName}</span>
      </div>
      {
        props.data?.status ?
          <div className='w-2 h-2 bg-green-500 rounded m-2'></div>
          :
          <div className='w-2 h-2 bg-gray-400 rounded m-2'></div>
      }

    </div>
  )
}

export default ProviderItem