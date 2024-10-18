'use client'
import { Button, Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, useDisclosure } from '@nextui-org/react'
import { t } from '@lingui/macro'
import { signIn, useSession } from 'next-auth/react'
import { forwardRef, useEffect, useImperativeHandle } from 'react'
import { GoogleLoginRef } from '@/framework/components/login/types'

const GoogleLogin = forwardRef<GoogleLoginRef, any>((props, ref) => {
  const { isOpen, onOpen, onOpenChange } = useDisclosure()
  const { status } = useSession()
  useEffect(() => {
    // 将status状态写入sessionStorage
    sessionStorage.setItem('userStatus', status)
  }, [status])
  const checkAuthenticated = () => {
    const userStatus = sessionStorage.getItem('userStatus')
    return 'authenticated' === userStatus
  }
  useImperativeHandle(
    ref,
    () =>
      ({
        // 检查是否已经登录，返回boolean类型
        checkAuthenticated: checkAuthenticated,
        // 弹出登录窗口
        open: () => onOpen()
      }) as any
  )
  return (
    <>
      <Modal isOpen={isOpen} onOpenChange={onOpenChange}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader className="flex flex-col gap-1">AI Outpainting Image</ModalHeader>
              <ModalBody>
                <div className="flex flex-col items-center">
                  <h2 className="md:text-4xl text-2xl font-bold text-center py-6">{t`Please Sign In To Continue`}</h2>
                  <p className="text-red-600 mb-6">{t`New users who log in will receive 2 credits`}</p>
                  <div
                    className="w-11/12 bg-white border border-blue-500 rounded-lg shadow-lg py-1.5 cursor-pointer  hover:text-white hover:bg-opacity-50 hover:bg-blue-300 flex items-center justify-center"
                    onClick={() => signIn('google')}
                  >
      <span className="text-blue-500 mr-2">
            <img
              src="https://public-image.fafafa.ai/fa-image/2024/06/76a66e7970156e8747f457790575f740.png"
              className="w-10 h-10"
            />
          </span>
                    <span className="text-lg font-medium ">{t`Sign In With Google`}</span>
                  </div>
                </div>
              </ModalBody>
              <ModalFooter>
                <Button color="danger" variant="light" onPress={onClose}>
                  {t`Close`}
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  )
})
GoogleLogin.displayName = 'GoogleLogin'
export default GoogleLogin