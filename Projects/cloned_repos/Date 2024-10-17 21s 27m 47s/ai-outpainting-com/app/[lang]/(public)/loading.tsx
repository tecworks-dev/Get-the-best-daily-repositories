'use client'
import { Spin } from 'antd'
import { t } from '@lingui/macro'

export default function Loading() {
  return (
    <div className="w-full h-screen flex flex-col items-center justify-center bg-transparent">
      <Spin tip={t`Loading...`} size={'large'} fullscreen>
        <div className="content" />
      </Spin>
    </div>
  )
}
