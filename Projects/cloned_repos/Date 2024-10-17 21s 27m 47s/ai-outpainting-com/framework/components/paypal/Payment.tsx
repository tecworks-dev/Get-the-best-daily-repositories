'use client'
import { Modal, ModalBody, ModalContent, ModalFooter, ModalHeader, useDisclosure } from '@nextui-org/react'
import { forwardRef, useImperativeHandle } from 'react'
import { PaymentRef } from '@/framework/components/paypal/types'
import Pricing from '@/components/wegic/pricing'
import { t } from '@lingui/macro'

const Payment = forwardRef<PaymentRef, any>((props, ref) => {
  const { isOpen, onOpen, onOpenChange } = useDisclosure()
  useImperativeHandle(
    ref,
    () => ({
      open: () => onOpen(),
      locale: props.locale
    })
  )     

  return (
    <>
      <Modal 
        isOpen={isOpen} 
        onOpenChange={onOpenChange}
        size="5xl"
      >
        <ModalContent>
          {(onClose) => (
            <>
              <ModalBody>
                <Pricing locale={props.locale} />
              </ModalBody>
            </>
          )}
        </ModalContent>
      </Modal>
    </>
  )
})

Payment.displayName = 'Payment'
export default Payment
