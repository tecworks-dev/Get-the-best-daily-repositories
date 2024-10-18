import {
  cn,
  Divider,
  Modal,
  ModalBody,
  ModalContent,
  RadioProps,
  useDisclosure,
  useRadio,
  VisuallyHidden
} from '@nextui-org/react'
import { RechargePlan } from '@/framework/types'
import PaymentPaypalClient from '@/components/payment/paypal'
import { t } from '@lingui/macro'
import { forwardRef, useImperativeHandle } from 'react'

export type PaymentModalProps = {
  plan: RechargePlan
  locale?: string
}
export type PaymentRef = {
  open:()=>void
}

export const CustomRadio = (props: RadioProps) => {
  const {
    Component,
    children,
    isSelected,
    description,
    getBaseProps,
    getWrapperProps,
    getInputProps,
    getLabelProps,
    getLabelWrapperProps,
    getControlProps
  } = useRadio(props)

  return (
    <Component
      {...getBaseProps()}
      className={cn(
        'flex-1 group inline-flex items-center justify-between hover:bg-content2 flex-row-reverse',
        'cursor-pointer border-2 border-default rounded-lg gap-4 p-4',
        'data-[selected=true]:border-primary'
      )}
    >
      <VisuallyHidden>
        <input {...getInputProps()} />
      </VisuallyHidden>
      <span {...getWrapperProps()}>
        <span {...getControlProps()} />
      </span>
      <div {...getLabelWrapperProps()}>
        {children && <span {...getLabelProps()}>{children}</span>}
        {description && (
          <span className="text-small text-foreground opacity-70">
            {description}
          </span>
        )}
      </div>
    </Component>
  )
}

const PaymentModal = forwardRef<PaymentRef,PaymentModalProps>((props,ref)=>{
  const { isOpen, onOpenChange, onOpen,onClose } = useDisclosure()
  const {plan,locale} = props
  useImperativeHandle(
    ref,
    () =>
      ({
        // 弹出窗口
        open:()=>onOpen()
      }) as any
  )
  const PlanSection = () => (
    <section className="h-[30vh] p-5 md:h-[60vh] md:p-20 rounded-xl md:shadow-2xl ">
      <h1 className="font-bold text-3xl">US${plan.price}</h1>
      <div className="flex gap-4 justify-between mt-10">
        <div className="flex-grow w-full">
          <div className="w-full flex justify-between text-xl">
            <span>{plan.name}</span>
            <span>US${plan.price}</span>
          </div>
          <div className="text-sm text-gray-500 text-nowrap mb-3">{plan?.desc}</div>
          <Divider />
          <div className="mt-3">
            <div className="font-bold text-nowrap">
              <span className="pr-5">{t`Total Payment`}</span>
              <span>US${plan.price}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )


  const PaymentSection = () => (
    <section className="w-full p-5 md:h-full md:p-20">
      {/*<RadioGroup*/}
      {/*  label="Payments"*/}
      {/*  orientation="horizontal"*/}
      {/*  value={method}*/}
      {/*  onValueChange={(value) => setMethod(value)}*/}
      {/*  classNames={{ wrapper: 'justify-between' }}*/}
      {/*>*/}
      {/*  <CustomRadio value="stripe">*/}
      {/*    <span>*/}
      {/*      <IoCard />*/}
      {/*    </span>*/}
      {/*    Card*/}
      {/*  </CustomRadio>*/}
      {/*  <CustomRadio value="paypal">*/}
      {/*    <span>*/}
      {/*      <IoLogoPaypal />*/}
      {/*    </span>*/}
      {/*    Paypal*/}
      {/*  </CustomRadio>*/}
      {/*</RadioGroup>*/}
      <div className="mt-10">
        {/*{method === 'stripe' && (*/}
        {/*  <PaymentStripeClient*/}
        {/*    locale={lang}*/}
        {/*    clientSecret={stripeSecret!}*/}
        {/*  />*/}
        {/*)}*/}
        {/*{method === 'paypal' && (*/}
        {/*  <PaymentPaypalClient code={plan.code} locale={lang} />*/}
        {/*)}*/}
        <PaymentPaypalClient code={plan.code} locale={locale} closePaypal={closePaypal}/>
      </div>
    </section>
  )

  const closePaypal = ()=>{
    onClose()
  }

  return (
    <Modal
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      isDismissable={false}
      disableAnimation
      size="4xl"
      classNames={{
        body: 'p-0'
      }}
    >
      <ModalContent>
        <ModalBody className="mt-3">
          <div className="w-full md:flex grid-cols-1 md:justify-between">
            <PlanSection />
            <PaymentSection />
          </div>
        </ModalBody>
      </ModalContent>
    </Modal>
  )
})

PaymentModal.displayName="PaymentModal"
export default PaymentModal