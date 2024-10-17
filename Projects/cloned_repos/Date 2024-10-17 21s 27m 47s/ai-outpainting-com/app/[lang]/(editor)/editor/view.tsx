'use client'
import { Dropdown, DropdownItem, Skeleton, User } from '@nextui-org/react'
import { Dropdown as AntDropDown } from 'antd'
import { UeDropzoneUpload, UeUploadRef } from '@/framework/components'
import GoogleLogin from '@/framework/components/login/GoogleLogin'
import { GoogleLoginRef } from '@/framework/components/login/types'
import Payment from '@/framework/components/paypal/Payment'
import { PaymentRef } from '@/framework/components/paypal/types'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { SessionUser } from '@/framework/types/sessionUser'
import { fetchPost } from '@/utils'
import { t } from '@lingui/macro'
import { signIn, useSession, signOut } from 'next-auth/react'
import {
  Avatar,
  Button,
  ButtonGroup,
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
  Navbar,
  Spinner,
  useDisclosure,
} from '@nextui-org/react'
import { $Enums } from '@prisma/client'
import { message, Spin } from 'antd'
import { FcGoogle } from 'react-icons/fc'
import { FaAngleRight, FaArrowUpFromBracket } from 'react-icons/fa6'
import {
  ArrowLeft,
  CreditCard,
  Eye,
  RefreshCcw,
  Trash2,
  Download,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { AiOutlineCloudUpload } from 'react-icons/ai'
import { Image as KonvaImage, Layer, Stage } from 'react-konva'
import runpodSdk from 'runpod-sdk'
import { siteConfig } from '@/config/site'
import { motion } from 'framer-motion'
import {
  RiAspectRatioLine,
  RiLayoutTopLine,
  RiLayoutBottomLine,
  RiLayoutLeftLine,
  RiLayoutRightLine,
  RiLayoutTopFill,
  RiLayoutBottomFill,
  RiLayoutLeftFill,
  RiLayoutRightFill,
} from 'react-icons/ri'

const EditorView: React.FC<{ params: { lang: AVAILABLE_LOCALES } }> = ({
  params,
}) => {
  const router = useRouter()
  const [msg, msgHolder] = message.useMessage()
  const paymentRef = useRef<PaymentRef>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [imageProps, setImageProps] = useState({
    width: 0,
    height: 0,
    x: 0,
    y: 0,
  })
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const stageRef = useRef<any>(null)
  const loginRef = useRef<GoogleLoginRef>(null)
  const [selectedExpandDirection, setSelectedExpandDirection] = useState<
    '16:9' | '9:16' | '1:1'
  >('16:9')
  const [selectedPosition, setSelectedPosition] = useState<
    'Middle' | 'Left' | 'Right' | 'Top' | 'Bottom'
  >('Middle')

  const [isLoading, setIsLoading] = useState(false)
  const [generatedImages, setGeneratedImages] = useState<string[]>([])
  const [originalImage, setOriginalImage] = useState<HTMLImageElement | null>(
    null,
  )
  const [isComparing, setIsComparing] = useState(false)
  const { isOpen, onOpen, onClose } = useDisclosure()
  const [topOptionsHeight, setTopOptionsHeight] = useState(0)
  const [bottomControlsHeight, setBottomControlsHeight] = useState(0)
  const topOptionsRef = useRef<HTMLDivElement>(null)
  const bottomControlsRef = useRef<HTMLDivElement>(null)
  const [mobileTopOptionsHeight, setMobileTopOptionsHeight] = useState(0)
  const mobileTopOptionsRef = useRef<HTMLDivElement>(null)
  const uploadRef = useRef<UeUploadRef>(null)
  // 原始图片的key
  const [originKey, setOriginKey] = useState<string | null>(null)
  // 添加一个新的 state 来存储 Stage 的样式
  const [stageStyle, setStageStyle] = useState({})
  // 获取当前用户信息
  const { data: session, status, update: updateSession } = useSession()
  const user = session?.user as SessionUser
  const isUnauthenticated = useMemo(
    () => 'unauthenticated' === status,
    [status],
  )
  const isAuthenticated = useMemo(() => 'authenticated' === status, [status])
  // 当前画布显示的图片对应的pathkey
  const [currentImagePath, setCurrentImagePath] = useState<string | null>(null)
  const [hasGeneratedImage, setHasGeneratedImage] = useState(false)
  const [showOriginalModal, setShowOriginalModal] = useState(false)
  const [historyImages, setHistoryImages] = useState<string[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [isImageLoading, setIsImageLoading] = useState(false)
  const [originalFileName, setOriginalFileName] = useState<string | null>(null)

  useEffect(() => {
    const updateCanvasSize = () => {
      const navbarHeight = 64 // Navbar 的高度
      const isMobile = window.innerWidth < 768
      const mobileTopOptionsHeight = isMobile
        ? mobileTopOptionsRef.current?.offsetHeight || 120
        : 0
      const bottomControlsHeight = bottomControlsRef.current?.offsetHeight || 80

      const availableHeight =
        window.innerHeight -
        navbarHeight -
        mobileTopOptionsHeight -
        bottomControlsHeight
      const width = window.innerWidth

      setCanvasSize({ width, height: availableHeight })
      setMobileTopOptionsHeight(mobileTopOptionsHeight)
      setBottomControlsHeight(bottomControlsHeight)
    }

    updateCanvasSize()
    window.addEventListener('resize', updateCanvasSize)
    return () => window.removeEventListener('resize', updateCanvasSize)
  }, [])

  // 更新这个 useEffect 来处理 Stage 样式的变化
  useEffect(() => {
    const isMobile = window.innerWidth < 768
    if (isMobile) {
      setStageStyle({ marginTop: `${mobileTopOptionsHeight}px` })
    } else {
      setStageStyle({})
    }
  }, [mobileTopOptionsHeight])

  const loadAndScaleImage = (imgSrc: string | File) => {
    return new Promise<void>((resolve) => {
      setIsImageLoading(true)
      const img = new Image()
      img.crossOrigin = 'anonymous'

      if (typeof imgSrc === 'string') {
        setCurrentImagePath(imgSrc)
        img.src = `${process.env.UE_S3_PUBLIC_PATH}/${imgSrc}`
      } else {
        img.src = URL.createObjectURL(imgSrc)
      }

      img.onload = () => {
        const scale = Math.min(
          (canvasSize.width - 72) / img.width,
          (canvasSize.height - 72) / img.height,
        )
        const width = img.width * scale
        const height = img.height * scale
        const x = (canvasSize.width - width) / 2
        const y = (canvasSize.height - height) / 2
        setImageProps({ width, height, x, y })
        setImage(img)
        setOriginalImage(img)
        setIsImageLoading(false)
        resolve()
      }
    })
  }

  const handleBeforeUpload = (file: File) => {
    const authenticated = loginRef.current?.checkAuthenticated()
    // 未登录不可上传
    if (!authenticated) {
      uploadRef.current?.clearFileList()
      loginRef.current?.open()
      return false
    }
    if (file.size > 10485760) {
      msg.error(t`File size exceeds 10MB. Please select a smaller file.`)
      return false
    }
    setOriginalFileName(file.name) // 保存原始文件名
    return true
  }

  function handleUploadFinish(e: any) {
    setOriginKey(e.extra.key)
    loadAndScaleImage(e.extra.key)
    setHistoryImages([]) // 清空历史图片列表，而不是加原始图片
  }

  function handleUploadRemove() {
    setOriginKey(null)
    setImage(null)
    setOriginalImage(null)
  }

  const ExpandDirectionIcon = ({
    direction,
    isSelected,
  }: {
    direction: '16:9' | '9:16' | '1:1'
    isSelected: boolean
  }) => (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {direction === '16:9' ? (
        <>
          <rect
            x="4"
            y="6"
            width="16"
            height="12"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M2 12H4M20 12H22"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M8 10L6 12L8 14M16 10L18 12L16 14"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
        </>
      ) : direction === '9:16' ? (
        <>
          <rect
            x="6"
            y="4"
            width="12"
            height="16"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M12 2V4M12 20V22"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M10 8L12 6L14 8M10 16L12 18L14 16"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
        </>
      ) : (
        <>
          <rect
            x="6"
            y="6"
            width="12"
            height="12"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M2 12H6M18 12H22M12 2V6M12 18V22"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
          <path
            d="M8 8L6 6M16 8L18 6M8 16L6 18M16 16L18 18"
            stroke={isSelected ? 'currentColor' : '#71717A'}
            strokeWidth="2"
          />
        </>
      )}
    </svg>
  )

  const PositionIcon = ({
    position,
    isSelected,
    aspectRatio,
  }: {
    position: 'Middle' | 'Left' | 'Right' | 'Top' | 'Bottom'
    isSelected: boolean
    aspectRatio: '16:9' | '9:16' | '1:1'
  }) => {
    const getOuterPath = () => {
      switch (aspectRatio) {
        case '16:9':
          return 'M2 6h20v12H2V6z'
        case '9:16':
          return 'M6 2h12v20H6V2z'
        case '1:1':
          return 'M4 4h16v16H4V4z'
      }
    }

    const getInnerRect = () => {
      switch (position) {
        case 'Middle':
          return { x: 8, y: 8, width: 8, height: 8 }
        case 'Left':
          return { x: 4, y: 8, width: 8, height: 8 }
        case 'Right':
          return { x: 12, y: 8, width: 8, height: 8 }
        case 'Top':
          return { x: 8, y: 4, width: 8, height: 8 }
        case 'Bottom':
          return { x: 8, y: 12, width: 8, height: 8 }
      }
    }

    const strokeColor = isSelected ? 'currentColor' : '#71717A'
    const fillColor = isSelected ? 'currentColor' : '#A1A1AA' // 更改为更深的灰色

    return (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d={getOuterPath()}
          stroke={strokeColor}
          strokeWidth="2"
          fill="none"
        />
        <rect {...getInnerRect()} fill={fillColor} />
      </svg>
    )
  }

  const getAvailablePositions = (expandDirection: '16:9' | '9:16' | '1:1') => {
    switch (expandDirection) {
      case '16:9':
        return ['Middle', 'Left', 'Right']
      case '9:16':
        return ['Middle', 'Top', 'Bottom']
      case '1:1':
        return ['Middle']
    }
  }

  useEffect(() => {
    const availablePositions = getAvailablePositions(selectedExpandDirection)
    if (!availablePositions.includes(selectedPosition)) {
      setSelectedPosition('Middle')
    }
  }, [selectedExpandDirection, selectedPosition])

  const handleGenerate = async (regenerate: boolean = false) => {
    setIsLoading(true)
    setIsProcessing(true)
    let orderNo = ''
    try {
      // 创建数据库订单数据
      const orderRequest = {
        path: regenerate ? originKey : currentImagePath,
        options: {
          aspectRatio: selectedExpandDirection,
          alignment: selectedPosition,
        },
      }

      // 提交订单
      const orderResponse = (await fetchPost(
        '/api/order/submit',
        orderRequest,
      )) as any
      console.log('orderResponse', orderResponse)
      orderNo = orderResponse?.orderNo
      if (orderResponse?.error) {
        if (orderResponse?.error === 'Please Sign In To Continue ') {
          msg.error(t`Please Sign In To Continue `)
        } else {
          msg.error(
            t`Your credit points are insufficient, please recharge first.`,
          )
          paymentRef.current?.open()
        }
        return
      }

      // 重新生成使用最原始图，否则使用当前画布的图片
      const req = {
        data: {
          orderNo,
          r2_key: regenerate ? originKey : currentImagePath,
          aspect_ratio: selectedExpandDirection,
          alignment: selectedPosition,
          callback_url: `${process.env.UE_WEB_API_URL}/api/order/update`,
        },
        source: 'ai-outpainting-web',
        priority: 1,
        queue_name: 'aioutpaint',
      }
      // 提交mq队列发起订单处理
      const mqResponse = (await fetchPost(
        `${process.env.UE_MQ_API_URL}/order`,
        req,
      )) as any
      console.log("###mqResponse",mqResponse)
      if (mqResponse?.error) {
        msg.error(t`System Error, Please retry or contact website admin`)
        return
      }
      const clientId = mqResponse.client_id
      // 设置SSE超时时间
      const timeout = 60000; // 60秒超时
      const timeoutId = setTimeout(() => {
        eventSource.close();
        setIsLoading(false);
        setIsProcessing(false);
        msg.error(t`Request timed out. Please try again later.`);
        updateOrderStatus(orderNo,'FAILED',0,'','Request timed out')
      }, timeout);
      // 使用SSE监听结果
      const eventSource = new EventSource(`${process.env.UE_MQ_API_URL}/sse/${clientId}`)

      eventSource.onmessage = (event) => {
        if (event.data !== 'keepalive') {
          const result = JSON.parse(event.data)
          console.log('Received SSE message:', result)
          if (result.data.status === 'SUCCEED') {
            const outputImagePath = result.data.outputImagePath
            setGeneratedImages((prev) => [...prev, outputImagePath])
            setHistoryImages((prev) => [...prev, outputImagePath])
            loadAndScaleImage(outputImagePath)
            setHasGeneratedImage(true)
          } else if (result.data.status === 'FAILED') {
            msg.error(t`System Error, Please retry or contact website admin`)
          }
          updateOrderStatus(orderNo,result.data.status,result.data.costTime,result.data.outputImagePath,result.data.reason)
          window.clearTimeout(timeoutId);
          eventSource.close()
          setIsLoading(false)
          setIsProcessing(false)
        }
      }

      eventSource.onerror = (error) => {
        console.error('SSE Error:', error)
        msg.error(t`Error occurred while processing the image`)
        eventSource.close()
        setIsLoading(false)
        setIsProcessing(false)
        updateOrderStatus(orderNo,'FAILED',0,'',JSON.stringify(error))
      }
    } catch (error) {
      console.error('Error in handleGenerate:', error)
      msg.error(t`An unexpected error occurred`)
      if(orderNo){
        updateOrderStatus(orderNo,'FAILED',0,'',JSON.stringify(error))
      }
      setIsLoading(false)
      setIsProcessing(false)
    }
  }

  const updateOrderStatus = (orderNo:string,status:string,costTime:number,outputImagePath:string,reason:string) => {
    const order = {
      orderNo,
      status,
      costTime,
      outputImagePath,
      reason,
    }
    fetchPost(`/api/order/update`,order)
  } 

  const handleCompareOriginal = (isComparing: boolean) => {
    setIsComparing(isComparing)
    if (isComparing && originalImage) {
      setImage(originalImage)
    } else if (!isComparing && image) {
      setImage(image)
    }
  }

  const handleClearCanvas = () => {
    onOpen() // 打开
  }

  const confirmClearCanvas = () => {
    setImage(null)
    setOriginalImage(null)
    setImageProps({ width: 0, height: 0, x: 0, y: 0 })
    setGeneratedImages([])
    onClose() // 关闭确认对话框
  }

  const handleDownload = () => {
    if (image) {
      const link = document.createElement('a')
      link.href = image.src
      link.target = '_blank'

      // 使用原始文件名，如果没有则使用默认名称
      const fileName = originalFileName || 'expanded_image.png'

      // 确保文件扩展名正确
      const fileExtension = fileName.split('.').pop()?.toLowerCase()
      const downloadFileName =
        fileExtension === 'png' ? fileName : `${fileName}.png`

      link.download = downloadFileName
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const AspectRatioButton = ({
    ratio,
    label,
  }: {
    ratio: '16:9' | '9:16' | '1:1'
    label: string
  }) => (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`flex items-center justify-center p-2 rounded-lg transition-colors ${
        selectedExpandDirection === ratio
          ? 'bg-primary text-white'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
      onClick={() => setSelectedExpandDirection(ratio)}
    >
      <ExpandDirectionIcon
        direction={ratio}
        isSelected={selectedExpandDirection === ratio}
      />
      <span className="ml-2 text-sm">{label}</span>
    </motion.button>
  )

  const PositionButton = ({
    position,
    isMobile = false,
  }: {
    position: 'Middle' | 'Left' | 'Right' | 'Top' | 'Bottom'
    isMobile?: boolean
  }) => (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`flex items-center justify-center p-2 rounded-lg transition-colors ${
        selectedPosition === position
          ? 'bg-primary text-white'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      } ${isMobile ? 'flex-1' : ''}`}
      onClick={() => setSelectedPosition(position)}
    >
      <PositionIcon
        position={position}
        isSelected={selectedPosition === position}
        aspectRatio={selectedExpandDirection}
      />
      <span className={`ml-2 text-sm ${isMobile ? 'text-xs' : ''}`}>
        {position === 'Middle'
          ? t`Middle`
          : position === 'Left'
            ? t`Left`
            : position === 'Right'
              ? t`Right`
              : position === 'Top'
                ? t`Top`
                : t`Bottom`}
      </span>
    </motion.button>
  )

  return (
    <Spin spinning={isProcessing} tip={t`Processing...`} size="large">
      <div className="h-screen flex flex-col">
        <GoogleLogin ref={loginRef} />
        {msgHolder}
        <Payment ref={paymentRef} locale={params?.lang} />
        <Navbar isBordered className="h-16" maxWidth="full">
          <div className="flex items-center">
            <Button
              as={Link}
              href={`/${params.lang}/`}
              variant="light"
              startContent={<ArrowLeft />}
            >
              {t`Back`}
            </Button>
          </div>

          <div className="flex-grow justify-center items-center space-x-4 hidden md:flex">
            <div className="flex items-center space-x-2">
              <span className="text-sm whitespace-nowrap">{t`Expansion Direction:`}</span>
              <div className="flex space-x-2">
                <AspectRatioButton ratio="16:9" label={t`Horizontal`} />
                <AspectRatioButton ratio="9:16" label={t`Vertical`} />
                <AspectRatioButton ratio="1:1" label={t`Square`} />
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <span className="text-sm whitespace-nowrap">{t`Original Image Position:`}</span>
              <div className="flex space-x-2">
                {getAvailablePositions(selectedExpandDirection).map((pos) => (
                  <PositionButton key={pos} position={pos as any} />
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {isUnauthenticated && siteConfig.showLogin ? (
              <>
                <div className="hidden sm:block">
                  <Button
                    color={'primary'}
                    variant="flat"
                    startContent={<FcGoogle size="1em" color="white" />}
                    onClick={() => signIn('google')}
                  >{t`Sign In With Google`}</Button>
                </div>
                <div className="sm:hidden">
                  <Button
                    color={'primary'}
                    variant="flat"
                    startContent={<FcGoogle size="1em" color="white" />}
                    onClick={() => signIn('google')}
                  >{t`Sign In`}</Button>
                </div>
              </>
            ) : (
              <AntDropDown
                menu={{
                  items: [
                    {
                      key: 'profile',
                      className: 'h-14 gap-2',
                      disabled: true,
                      label: (
                        <>
                          <p className="font-semibold">{user?.email ?? ''}</p>
                          <p className="font-semibold">{t`Credit:${user?.credit ?? 0}`}</p>
                        </>
                      ),
                    },
                    {
                      key: 'logout',
                      itemIcon: <FaArrowUpFromBracket />,
                      label: t`Log Out`,
                      onClick: () => signOut(),
                    },
                  ],
                }}
              >
                <Skeleton isLoaded={isAuthenticated} className="rounded-lg">
                  {user?.image && (
                    <User
                      name={user?.name ?? ''}
                      description={t`Credit:${user?.credit ?? 0}`}
                      className="cursor-pointer"
                      avatarProps={{
                        lang: params.lang,
                        src: user?.image ?? '#',
                      }}
                    ></User>
                  )}
                </Skeleton>
              </AntDropDown>
            )}

            <Button
              size="sm"
              color="primary"
              startContent={<CreditCard />}
              onClick={() => {
                paymentRef.current?.open()
              }}
            >
              {t`Purchase`}
            </Button>
          </div>
        </Navbar>

        <div className="flex-grow flex flex-col relative">
          <div
            ref={mobileTopOptionsRef}
            className="md:hidden sticky top-0 z-10 bg-white bg-opacity-90 p-4 shadow-md"
          >
            <div className="flex flex-col space-y-4">
              <div>
                <p className="text-sm font-medium mb-2">{t`Expansion Direction`}</p>
                <div className="flex justify-between">
                  <AspectRatioButton ratio="16:9" label={t`Horizontal`} />
                  <AspectRatioButton ratio="9:16" label={t`Vertical`} />
                  <AspectRatioButton ratio="1:1" label={t`Square`} />
                </div>
              </div>
              <div>
                <p className="text-sm font-medium mb-2">{t`Original Image Position`}</p>
                <div className="flex justify-between space-x-2">
                  {getAvailablePositions(selectedExpandDirection).map((pos) => (
                    <PositionButton
                      key={pos}
                      position={pos as any}
                      isMobile={true}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="flex-grow relative">
            <Stage
              width={canvasSize.width}
              height={canvasSize.height}
              ref={stageRef}
            >
              <Layer>
                {(isComparing ? originalImage : image) && (
                  <KonvaImage
                    image={
                      (isComparing ? originalImage : image) as CanvasImageSource
                    }
                    {...imageProps}
                    opacity={isComparing ? 0.5 : 1}
                    listening={false}
                  />
                )}
              </Layer>
            </Stage>

            {!image && (
              <div className="absolute inset-0 max-w-xl mx-auto flex items-center justify-center">
                <Spin
                  spinning={isLoading || isImageLoading}
                  tip={isLoading ? t`Uploading...` : t`Loading image...`}
                  className="bg-mask-200 rounded-xl"
                >
                  <div className="relative md:py-5 border-primary border-dashed border-2 rounded-xl p-5">
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
                        <div className="flex justify-center text-6xl text-primary">
                          <AiOutlineCloudUpload />
                        </div>
                        <div className="text-gray-600">{t`Please click or drag the image to upload it.`}</div>
                      </div>
                    </UeDropzoneUpload>
                  </div>
                </Spin>
              </div>
            )}
          </div>

          <div
            ref={bottomControlsRef}
            className="bg-gray-100 p-2 m-h-14 z-10 sticky bottom-0"
          >
            <div className="max-w-5xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-4">
              <div className="flex flex-wrap justify-center items-center gap-2 w-full sm:w-auto">
                <Button
                  size="sm"
                  color="default"
                  onClick={handleClearCanvas}
                  isDisabled={!image || isProcessing}
                >
                  <Trash2 size={20} />
                  <span className="ml-1 hidden sm:inline">{t`Clear Canvas`}</span>
                </Button>
                {hasGeneratedImage && (
                  <Button
                    size="sm"
                    color="secondary"
                    onClick={() => handleGenerate(true)}
                    isLoading={isLoading}
                    isDisabled={!image || isProcessing}
                  >
                    <RefreshCcw size={20} />
                    <span className="ml-1">{t`Regenerate`}</span>
                  </Button>
                )}
                <Button
                  size="sm"
                  color="primary"
                  onClick={() => handleGenerate(false)}
                  isLoading={isLoading}
                  isDisabled={!image || isProcessing}
                >
                  {isLoading ? <Spinner size="sm" /> : null}
                  <span className="ml-1 flex items-center">
                    {isLoading
                      ? t`Expanding`
                      : hasGeneratedImage
                        ? t`Continue Generating`
                        : t`Start Expanding`}
                  </span>
                </Button>
                <span className="ml-1 text-sm">
                  {t`1 Credit/use, ${user?.credit} left`}
                </span>
                <Button
                  size="sm"
                  color="default"
                  onClick={handleDownload}
                  isDisabled={!image || isProcessing}
                >
                  <Download size={20} />
                  <span className="ml-1 hidden sm:inline">{t`Download`}</span>
                </Button>
              </div>

              <div className="flex justify-center gap-2 mt-2 sm:mt-0 overflow-x-auto w-full sm:w-auto">
                {
                  <div
                    className={`min-w-14 w-14 h-14 rounded border bg-cover bg-center cursor-pointer ${
                      image &&
                      image.src ===
                        `${process.env.UE_S3_PUBLIC_PATH}/${originKey}`
                        ? 'border-blue-500 border-2'
                        : 'border-gray-300'
                    }`}
                    style={{
                      backgroundImage: `url(${process.env.UE_S3_PUBLIC_PATH}/${originKey})`,
                    }}
                    onClick={() => setShowOriginalModal(true)}
                  />
                }
                {historyImages.map((imgPath, index) => (
                  <div
                    key={index}
                    className={`min-w-14 w-14 h-14 rounded border bg-cover bg-center cursor-pointer ${
                      image &&
                      image.src ===
                        `${process.env.UE_S3_PUBLIC_PATH}/${imgPath}`
                        ? 'border-blue-500 border-2'
                        : 'border-gray-300'
                    }`}
                    style={{
                      backgroundImage: `url(${process.env.UE_S3_PUBLIC_PATH}/${imgPath})`,
                    }}
                    onClick={() => {
                      const img = new Image()
                      img.src = `${process.env.UE_S3_PUBLIC_PATH}/${imgPath}`
                      img.onload = () => {
                        setImage(img)
                        const scale = Math.min(
                          (canvasSize.width - 40) / img.width,
                          (canvasSize.height - 40) / img.height,
                        )
                        const width = img.width * scale
                        const height = img.height * scale
                        const x = (canvasSize.width - width) / 2
                        const y = (canvasSize.height - height) / 2
                        setImageProps({ width, height, x, y })
                      }
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
        <Modal isOpen={isOpen} onClose={onClose}>
          <ModalContent>
            <ModalHeader>{t`Confirm Clear Canvas`}</ModalHeader>
            <ModalBody>
              {t`Are you sure you want to clear the current canvas? This will delete all generated images and the current editing state.`}
            </ModalBody>
            <ModalFooter>
              <Button color="default" onClick={onClose}>
                {t`Cancel`}
              </Button>
              <Button color="danger" onClick={confirmClearCanvas}>
                {t`Confirm Clear`}
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>

        {/* 原始图片查看模态框 */}
        <Modal
          isOpen={showOriginalModal}
          onClose={() => setShowOriginalModal(false)}
          size="xl"
        >
          <ModalContent>
            <ModalHeader>{t`Original Image`}</ModalHeader>
            <ModalBody>
              <img
                src={`${process.env.UE_S3_PUBLIC_PATH}/${originKey}`}
                alt="Original"
                className="w-full h-auto"
              />
            </ModalBody>
            <ModalFooter>
              <Button
                onClick={() => setShowOriginalModal(false)}
              >{t`Close`}</Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      </div>
    </Spin>
  )
}

export default EditorView
