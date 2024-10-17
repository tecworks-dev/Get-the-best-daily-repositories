'use client'
import { t } from '@lingui/macro'
import {
  Button,
  Image as NextUIImage,
  Link,
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  useDisclosure
} from '@nextui-org/react'
import { Image, message, Spin } from 'antd'
import React, { useMemo, useState } from 'react'
import { FaArrowRightArrowLeft, FaArrowRotateRight, FaArrowsUpDown, FaArrowUpRightFromSquare } from 'react-icons/fa6'
import PaintingUploader from '@/components/wegic/painting-uploader'
import { fetchGet, fetchPost } from '@/framework/utils'
import { $Enums, Order } from '@prisma/client'
import clsx from 'clsx'
import { useMount } from 'react-use'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { useSession } from 'next-auth/react'
import { FaDownload } from 'react-icons/fa'
import { siteConfig } from '@/config/site'

type OrderResult = Order & { inputImageUrl: string, outputImageUrl: string }
type OrderHistory = { orderNo: string, inputImageUrl: string, outputImageUrl: string, hidden?: boolean }


export function CrossSection({ order, history, params }: {
  order: any,
  history: any,
  params: { lang: AVAILABLE_LOCALES }
}) {
  const [data, setData] = useState<OrderResult>(order)

  function onUpdateData(data: any) {
    data.inputImagePublicPath = siteConfig.r2BaseUrl+"/"+data.inputImagePath
    if(data.outputImagePath){
      data.outputImagePublicPath = siteConfig.r2BaseUrl+"/"+data.outputImagePath
    }
    setData(data)
  }

  return <>
    <GenerationSection data={data} onUpdateData={onUpdateData} params={params} />
    <HistorySection history={history} onUpdateData={onUpdateData} />
  </>
}


export function GenerationSection({ data, onUpdateData, params }: {
  data: any,
  onUpdateData: (data: any) => void,
  params?: { lang: AVAILABLE_LOCALES }
}) {
  const { update: updateSession } = useSession()
  const [msg, msgHolder] = message.useMessage()
  const { isOpen, onOpen, onOpenChange } = useDisclosure()
  const [isError, setIsError] = useState(false)
  const [isRegenerate, setIsRegenerate] = useState(false)

  const isLoading = useMemo(() => {
    return data && ($Enums.OrderStatus.PENDING === data?.status || $Enums.OrderStatus.PROCESSING === data?.status)
  }, [data])


  async function doUpdateOrderWithSession(order: any) {
    onUpdateData(order)
    await updateSession()
  }

  function queryOrderResult(orderNo: string, timeout = 5000) {
    const doQuery = () => fetchGet<OrderResult>(`/api/order/get?orderNo=${orderNo}`).then(async (res) => {
      if ($Enums.OrderStatus.SUCCEED === res.status) {
         // 调用接口更新订单耗时
         fetchPost('/api/order/updateCostTime', {
          orderNo,
        })
        await doUpdateOrderWithSession(res)
      } else if ($Enums.OrderStatus.FAILED === res.status) {
         // 调用接口更新订单耗时
         fetchPost('/api/order/updateCostTime', {
          orderNo,
        })
        await doUpdateOrderWithSession(res)
        setIsError(true)
      } else {
        await doUpdateOrderWithSession(res)
        setTimeout(doQuery, timeout)
      }
    })
    setTimeout(doQuery, timeout)
  }

  async function handleRegeneration() {
    setIsError(false)
    setIsRegenerate(true)
    try {
      const res = await fetchPost<any>('/api/order/submit', {
        path: data.inputImagePath,
        options: JSON.parse(data.outputOptions)
      })
      await doUpdateOrderWithSession(res)
      return queryOrderResult(res.orderNo)
    } catch (err: any) {
      msg.warning(err)
    } finally {
      setIsRegenerate(false)
    }
  }

  async function handleExpansion(direction: string) {
    setIsError(false)
    setIsRegenerate(true)
    try {
      const res = await fetchPost<any>('/api/order/submit', {
        path: data.outputImagePath,
        options: {
          direction
        }
      })
      await doUpdateOrderWithSession(res)
      return queryOrderResult(res.orderNo)
    } catch (err: any) {
      msg.warning(err)
    } finally {
      setIsRegenerate(false)
    }
  }

  function downloadImage(url:string){
    const link = document.createElement('a');
    link.href = url;
    link.target="_blank"
    const pathname = url.split('?')[0];
    // Extract the filename by taking the last segment after the last '/'
    link.download = pathname.substring(pathname.lastIndexOf('/') + 1);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  useMount(() => {
    queryOrderResult(data.orderNo)
  })

  return (
    <section className="relative md:py-10 px-5 md:w-3/4 mx-auto">
      {msgHolder}
      <Modal isOpen={isOpen}
             onOpenChange={onOpenChange}
             isDismissable={false}
             size="2xl"
             backdrop="blur">
        <ModalContent>
          <ModalHeader>{t`Upload New Image`}</ModalHeader>
          <ModalBody>
            <PaintingUploader/>
          </ModalBody>
          <ModalFooter></ModalFooter>
        </ModalContent>
      </Modal>
      <div className="w-full flex items-center justify-between pb-5">
        <h1 className="text-xl md:text-2xl">{t`AI Outpainting Image`}</h1>
        <Button color="primary"
                startContent={<FaArrowUpRightFromSquare />}
                size="sm"
                onClick={onOpen}>{t`Upload New Image`}</Button>
      </div>
      <div className="w-full flex flex-col items-center">
        <div className="w-full md:py-5 py-2 text-center">
          <Spin spinning={isLoading}>
            <div className="flex flex-col sm:flex-row">
              <div className="flex flex-col flex-1">
                <div className="pb-3">
                  <div className="mb-2 font-medium text-xl flex gap-1 items-center">
                    {t`Input`}
                  </div>
                  <div className="text-left text-base">{t`Before Outpainting`}</div>
                </div>
                <div className="bg-[#f5f5f5] flex justify-center items-center rounded-2xl h-96 w-full">
                  <Image
                    alt="Origin Image"
                    height={320} // 高度调整为 240
                    src={data.inputImagePublicPath}
                    style={{ objectFit: 'contain', maxHeight: '100%', maxWidth: '100%' }} // 确保按比例显示
                    placeholder={
                      <Image
                        preview={false}
                        src={data.inputImagePublicPath}
                        className=""
                        alt="Origin Image Placeholder"
                      />
                    }
                  />
                </div>
                {
                  !isError && (
                    <div className="md:flex mt-4 grid grid-cols-1 md:gap-5 gap-2 items-end justify-center">
                      <Button
                        color="primary"
                        isLoading={isRegenerate}
                        isDisabled={isLoading}
                        onClick={handleRegeneration}
                      >
                        {t`Regenerate`}
                      </Button>
                    </div>
                  )
                }
              </div>
              <div className="sm:pl-7 flex flex-col flex-1 sm:w-2/6">
                <div className="pb-3">
                  <div className="mb-2 font-medium text-xl flex gap-1 items-center">
                    {t`Output`}
                  </div>
                  <div className="text-left text-base">{t`After Outpainting`}</div>
                </div>
                <div className="bg-[#f5f5f5] flex justify-center items-center rounded-2xl h-96 w-full">
                  <Image
                    alt="AI Outpainting Image"
                    height={360} // 高度调整为 288
                    src={data.outputImagePublicPath}
                    style={{ objectFit: 'contain', maxHeight: '100%', maxWidth: '100%' }} // 确保按比例显示
                    fallback={data.outputImageUrl}
                    placeholder={
                      <Image
                        preview={false}
                        src={data.outputImageUrl}
                        alt="Genereation Image Placeholder"
                      />
                    }
                  />
                </div>
                {!isError && (
                  <div className="md:flex mt-4 grid grid-cols-1 md:gap-5 gap-2 items-end justify-center">
                    <Button
                      color="primary"
                      isLoading={isRegenerate}
                      isDisabled={isLoading}
                      startContent={<FaArrowsUpDown />}
                      onClick={() => handleExpansion('vertical')}
                    >
                      {t`Vertical Expansion`}
                    </Button>
                    <Button
                      color="primary"
                      isLoading={isRegenerate}
                      isDisabled={isLoading}
                      startContent={<FaArrowRightArrowLeft />}
                      onClick={() => handleExpansion('horizontal')}
                    >
                      {t`Horizontal Expansion`}
                    </Button>
                    <Button
                      color="primary"
                      isLoading={isRegenerate}
                      isDisabled={isLoading}
                      startContent={<FaDownload />}
                      onClick={() => downloadImage(data.outputImageUrl)}
                    >
                      {t`Download Image`}
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </Spin>
          {isError && <div className="">
            <p
              className="text-sm text-danger-500 py-5">{t`There was an error processing the image. Please try again later.`}</p>
            <Button
              color="primary"
              isLoading={isRegenerate}
              isDisabled={isLoading}
              startContent={<FaArrowRotateRight />}
              onClick={handleRegeneration}
            >
              {t`Retry`}
            </Button>
          </div>}
        </div>
      </div>


    </section>
  )
}


export function HistorySection({ history, onUpdateData }: {
  history?: any,
  onUpdateData: (data: any) => void
}) {
  const [selection, setSelection] = useState<OrderHistory>()
  const [orders, setOrders] = useState<Partial<OrderHistory>[]>((history ?? []).map((it: any) => ({
    ...it,
    hidden: false
  })))

  function handleClick(item: any) {
    if (selection) {
      selection.hidden = false
    }
    item.history = false
    setSelection(item)
    onUpdateData(item)
  }

  return <section className="relative md:py-10 py-5 px-5">
    {
      orders.length > 0 && (
        <h2 className="text-md font-bold text-center pb-10"> {t`Own By User`}</h2>
      )
    }
    <div className="md:grid md:grid-cols-4 gap-5 flex flex-col items-center">
      {orders.map((item, index) => (
        <NextUIImage
          key={index}
          alt="Own By User Image"
          src={item.outputImageUrl}
          className={clsx({ 'hidden': item.hidden })}
          onClick={() => handleClick(item)}
        />
      ))}
    </div>
  </section>
}

export function EmptyTips() {
  return <section className="w-full flex justify-center items-center">
    <p className="md:py-40 py-10 text-gray-800 text-4xl">{t`No valid resource information was found.`}</p>
  </section>
}