'use client'
import PaintingUploader from '@/components/wegic/painting-uploader'
import { GoogleLoginRef } from '@/framework/components/login/types'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { t } from '@lingui/macro'
import { Button } from '@nextui-org/react'
import { useRef } from 'react'
import { FaArrowRightLong } from 'react-icons/fa6'
export default function IndexUploader({ params }: { params: { lang: AVAILABLE_LOCALES }}){
  const loginRef = useRef<GoogleLoginRef>(null)

  const handleGetStarted = () => {
    const authenticated = loginRef.current?.checkAuthenticated()
    // 未登录不可上传
    if (!authenticated) {
      loginRef.current?.open()
      return false
    }else{
 // 用户已登录，直接跳转到编辑器页面
      window.location.href = `/${params.lang}/editor`
    }
   
  }
  return (
    <div className="w-full max-w-7xl mx-auto  items-center py-10">
    <div className="container px-4 mx-auto ">
    <div className="relative max-w-xl mx-auto sm:text-center pb-5">
    <h2 className="text-3xl font-semibold sm:text-4xl">
      {t`Try It Free`}
      </h2>
      <div className="mt-3 max-w-xl">
  <p className="text-gray-600">
    {t`After logging in, you can have three two trials expand your image with AI`}
    </p>
    </div>
    </div>
    <div className="w-full flex justify-center p-4">
               <Button 
                 type="button" 
                 color="primary" 
                 onClick={handleGetStarted}
                 className="px-8 py-3 text-lg font-bold rounded-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 transform hover:scale-105 transition duration-300 ease-in-out shadow-lg hover:shadow-xl"
               >
                 <span className="mr-2">{t`Get Started`}</span>
                 <FaArrowRightLong className="inline-block" />
               </Button>
            </div>
  </div>
  </div>
  )
}