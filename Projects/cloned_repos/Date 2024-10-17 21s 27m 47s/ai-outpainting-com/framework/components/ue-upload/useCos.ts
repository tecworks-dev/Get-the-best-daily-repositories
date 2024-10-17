import { useState } from 'react'
import type { UploadProps } from 'antd'
import { useCosCredentials } from '@/framework/hooks'
import { createUploadFileKey } from '@/framework/components/ue-upload/utils'
import buildCosAuth from '@/framework/components/ue-upload/cos-auth'
import { UeUploadProps } from '@/framework/components'

export default function useCos(props: UeUploadProps): Partial<UploadProps> {
  const { value: cosOptions } = useCosCredentials()

  async function handleBeforeUpload(file: any, files: any[]) {
    const key = createUploadFileKey(file!.name, props)
    const url = `${cosOptions!['publicPath']}${key}`
    setUploadConfig({
      ...uploadConfig,
      action: url,
      method: 'PUT',
      headers: {
        'Content-Type': file?.type,
        'x-cos-security-token': cosOptions!.credentials['sessionToken'],
        Authorization: buildCosAuth({
          SecretId: cosOptions!.credentials['tmpSecretId'],
          SecretKey: cosOptions!.credentials['tmpSecretKey'],
          Method: 'PUT',
          Pathname: key
        })
      }

    })
    file.extra = {
      key,
      url
    }
    if (props.onBeforeUpload) {
      return props.onBeforeUpload(file, files)
    }
    return file
  }

  const [uploadConfig, setUploadConfig] = useState<UploadProps>({
    method: 'POST',
    name: 'file',
    multiple: false,
    beforeUpload: handleBeforeUpload
  })


  return uploadConfig
}