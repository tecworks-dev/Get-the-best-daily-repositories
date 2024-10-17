'use client'
import { Button, Select, Selection, SelectItem } from '@nextui-org/react'
import { t } from '@lingui/macro'
import { UeDropzoneUpload, UeUploadRef } from '@/framework/components'
import React, { useMemo, useRef, useState } from 'react'
import { AiOutlineCloudUpload } from 'react-icons/ai'
import { message, Spin } from 'antd'
import { useRouter } from 'next/navigation'
import { fetchPost } from '@/framework/utils'
import { OrderSubmit } from '@/types'
import { FaArrowsTurnToDots } from 'react-icons/fa6'
import GoogleLogin from '@/framework/components/login/GoogleLogin'
import { GoogleLoginRef } from '@/framework/components/login/types'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { PiArrowsOutLineVerticalDuotone } from "react-icons/pi";
import { PiArrowsInLineHorizontalFill } from "react-icons/pi";


export default function PaintingUploader() {
  const router = useRouter()
  const [direction, setDirection] = useState<Selection>(new Set(['Four']))
  const [msg, msgHolder] = message.useMessage()
  const uploadRef = useRef<UeUploadRef>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [order, setOrder] = useState<OrderSubmit>()
  const loginRef = useRef<GoogleLoginRef>(null)

  const handleBeforeUpload = (file: File) => {
    const authenticated = loginRef.current?.checkAuthenticated()
    // 未登录不可上传
    if (!authenticated) {
      uploadRef.current?.clearFileList()
      loginRef.current?.open()
      return false
    }
    // 限制图片上传大小
    // 校验文件大小是否超过10M，超过后提示不可上传，并阻止继续上传
    if (file.size > 10485760) {
      // 此处你可以添加你自己的提示代码
      msg.error(t`File size exceeds 10MB. Please select a smaller file.`)
      return false
    }
    return true
  }


  function handleUploadFinish(e: any) {
    setOrder({ path: e.extra.key })
    msg.success(t`Upload successfully`)
  }

  function handleUploadRemove() {
    setOrder({ path: '' })
  }

  const isDisabled = useMemo(() => !order?.path, [order])

  async function handleSubmit() {
    const authenticated = loginRef.current?.checkAuthenticated()
    // 未登录不点击提交
    if (!authenticated) {
      uploadRef.current?.clearFileList()
      loginRef.current?.open()
      return false
    }
    setIsLoading(true)
    const data = {
      ...order,
      options: {
        direction: String([...direction][0]).toLowerCase()
      }
    }
    try {
      const res = await fetchPost<any>('/api/order/submit', data)
      uploadRef.current?.clearFileList()
     
    } catch (err: any) {
      msg.error(err ? err : t`System Error,Please retry or contract website admin`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="w-full rounded-xl relative">
      <GoogleLogin ref={loginRef} />
      {msgHolder}
      <Spin spinning={isLoading} tip={t`Uploading...`} className="bg-mask-200 rounded-xl">
        <div
          className="relative md:py-5 border-primary border-dashed border-2 rounded-xl p-5">

          <UeDropzoneUpload
            ref={uploadRef}
            accept={'image/*'}
            listType="picture"
            dir={'/input/origin'}
            withTimestamp={true}
            keepOriginName={true}
            onBeforeUpload={handleBeforeUpload}
            onUploadFinish={handleUploadFinish}
            onRemove={handleUploadRemove}
          >
            <div className="md:py-10">
              <div className="flex justify-center text-6xl text-primary"><AiOutlineCloudUpload /></div>
              <div className="text-gray-600">{t`Please click or drag the image to upload it.`}</div>
            </div>
          </UeDropzoneUpload>
          {/* <div className="pt-5">
            <h6 className="mb-2 text-sm text-gray-600">{t`Expansion direction`}</h6>
            <Select selectedKeys={direction}
                    color="primary"
                    onSelectionChange={setDirection}
                    placeholder={t`Please select`}
                    size="lg"
                    aria-label="Expansion direction">
              <SelectItem key='Four' value={'Four'} aria-label={t`Four Sides`} startContent={<PiArrowsOutLineVerticalDuotone/>}>{t`Four Sides`}</SelectItem>
              <SelectItem key='Vertical' value={'Vertical'} aria-label={t`Vertical`} startContent={<PiArrowsOutLineVerticalDuotone/>}>{t`Vertical`}</SelectItem>
              <SelectItem key='Horizontal' value={'Horizontal'} aria-label={t`Horizontal`} startContent={<PiArrowsInLineHorizontalFill/>}>{t`Horizontal`}</SelectItem>
            </Select>
          </div> */}
        </div>
      </Spin>
      {/* <div className="w-full pt-5 flex items-center justify-between gap-5">
        <div className="text-xs text-gray-600">{t`Expand up to 200 pixels each time`}</div>
        <Button
          size="lg"
          color="primary"
          startContent={<FaArrowsTurnToDots />}
          isLoading={isLoading}
          aria-label="Generate"
          onClick={handleSubmit}>{t`Generate`}</Button>
      </div> */}
    </div>
  )
}