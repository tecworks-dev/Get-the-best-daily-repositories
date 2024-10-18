import { forwardRef, useImperativeHandle, useState } from 'react'
import { UeUploadProps, UeUploadRef } from './types'
import type { UploadFile } from 'antd'
import { Upload } from 'antd'
import { InboxOutlined } from '@ant-design/icons'
import { UploadChangeParam } from 'antd/es/upload/interface'
import useCos from './useCos'
import useS34R2 from './useS34R2'

const { Dragger } = Upload


export const UeDropzoneUpload = forwardRef<
  UeUploadRef,
  UeUploadProps
>((props, ref) => {
  const { provider = 's34r2' } = props
  const useProviderHook = 'cos' === provider ? useCos : useS34R2
  const uploadConfig = useProviderHook(props)
  const [fileList, setFileList] = useState<UploadFile[]>([])

  useImperativeHandle(
    ref,
    () =>
      ({
        clearFileList: () => setFileList(()=>{return []})
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
    <Dragger
      ref={ref as any}
      maxCount={1}
      {...props}
      {...uploadConfig}
      fileList={fileList}
      onChange={handleChange}
    >
      {props?.children ?? <InboxOutlined />}
    </Dragger>
  )
})
UeDropzoneUpload.displayName = 'UeDropzoneUpload'
export default UeDropzoneUpload
