'use client'
import { forwardRef, useImperativeHandle, useMemo, useState } from 'react'
import { UeUploadProps, UeUploadRef } from '@/framework/components'
import { Image, Spin, Upload, type UploadFile } from 'antd'
import useCos from '@/framework/components/ue-upload/useCos'
import useS34R2 from '@/framework/components/ue-upload/useS34R2'
import { UploadChangeParam } from 'antd/es/upload/interface'
import { InboxOutlined } from '@ant-design/icons'
import { t } from '@lingui/macro'

const Dragger = Upload.Dragger

const UePictureUpload = forwardRef<UeUploadRef, UeUploadProps>((props, ref) => {
  const { provider = 's34r2' } = props
  const useProviderHook = 'cos' === provider ? useCos : useS34R2
  const uploadConfig = useProviderHook(props)
  const [isLoading, setIsLoading] = useState(false)
  const [isDone, setIsDone] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const imageUrl = useMemo(() => {
    return fileList && fileList.length > 0 ? fileList[0].url : ''
  }, [])
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
      setIsLoading(false)
      // @ts-ignore
      props?.onChangeValue && props?.onChangeValue(file.extra.key, file.extra)
    }
    if (file.status === 'uploading') {
      setIsLoading(true)
      return
    }
    setFileList(fileList.slice())
  }

  const RenderedElement = props.draggable ? Dragger : Upload
  const renderChildren = () => {
    if (isDone) {
      return <Image src={imageUrl} alt="picture upload image." />
    }
    return props.children ?? (
      <div>
        <InboxOutlined />
      </div>
    )
  }


  return (
    <Spin spinning={isLoading} tip={t`Uploading`} className="bg-mask-200">
      <RenderedElement
        {...props}
        {...uploadConfig}
        listType="picture-card"
        showUploadList={false}
        fileList={fileList}
        onChange={handleChange}
      >
        {renderChildren()}
      </RenderedElement>
    </Spin>
  )
})

UePictureUpload.displayName = 'UePictureUpload'
export default UePictureUpload