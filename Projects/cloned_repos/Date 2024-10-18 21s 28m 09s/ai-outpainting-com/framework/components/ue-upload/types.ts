import type { UploadFile, UploadProps } from 'antd'
import type { UploadRef } from 'antd/es/upload/Upload'

export type UeUploadProviderType = 's3' | 'cos'

export type UeUploadRef = UploadRef<any> & {
  clearFileList: () => void
}




export interface UeUploadProps extends Partial<Omit<UploadProps, 'value' | 'onChange'>> {
  draggable?: boolean
  dir?: string
  withTimestamp?: boolean
  keepOriginName?: boolean
  fallbackUrl?: string
  onChangeValue?: (key: string, data: { key: string; url: string }) => void
  onUploadFinish?: (file: UploadFile) => void
  provider?: UeUploadProviderType
  onBeforeUpload?: UploadProps['beforeUpload']
  children?: React.ReactNode
}
