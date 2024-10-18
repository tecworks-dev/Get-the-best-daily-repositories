'use client'
import { Upload, type UploadFile } from 'antd'
import { UeUploadProps, UeUploadRef } from '@/framework/components'
import { forwardRef, useImperativeHandle, useState } from 'react'
import { Button } from '@nextui-org/react'
import { t } from '@lingui/macro'
import { FaCloudArrowUp } from 'react-icons/fa6'
import useCos from '@/framework/components/ue-upload/useCos'
import useS34R2 from '@/framework/components/ue-upload/useS34R2'
import { UploadChangeParam } from 'antd/es/upload/interface'

const UeButtonUpload = forwardRef<UeUploadRef, UeUploadProps>((props, ref) => {
  const { provider = 's34r2' } = props
  const useProviderHook = 'cos' === provider ? useCos : useS34R2
  const uploadConfig = useProviderHook(props)
  const [isLoading, setIsLoading] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  useImperativeHandle(
    ref,
    () =>
      ({
        clearFileList: () => setFileList([])
      }) as any
  )

  function handleChange({ file, fileList }: UploadChangeParam) {
    if ('done' === file.status && props?.onUploadFinish) {
      props.onUploadFinish(file)
      // @ts-ignore
      props?.onChangeValue && props?.onChangeValue(file.extra.key, file.extra)
    }
    setFileList(fileList.slice())
  }

  return (
    <Upload
      {...props}
      {...uploadConfig}
      fileList={fileList}
      showUploadList={false}
      onChange={handleChange}
    >
      {props.children ?? (
        <Button isLoading={isLoading} size="lg" color="primary" startContent={<FaCloudArrowUp />}>{t`Upload`}</Button>
      )}
    </Upload>
  )
})
UeButtonUpload.displayName = 'UeButtonUpload'

export default UeButtonUpload