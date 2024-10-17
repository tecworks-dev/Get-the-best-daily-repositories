import { PayPalButtons, usePayPalScriptReducer } from '@paypal/react-paypal-js'
import { fetchPost, fetchPut } from '@/framework/utils'
import {
  CreateOrderActions,
  CreateOrderData,
  OnApproveActions,
  OnApproveData
} from '@paypal/paypal-js/types/components/buttons'
import { message, Spin } from 'antd'
import { t } from '@lingui/macro'
import { Button } from '@nextui-org/react'
import { FaArrowCircleLeft } from 'react-icons/fa'
import { useSession } from 'next-auth/react'

export default function PaymentPaypalClient({ code, locale,closePaypal }: { code: string, locale?: string,closePaypal:()=>void }) {
  const [{ isPending, isResolved, isRejected, isInitial }] = usePayPalScriptReducer()
  const [msg, msgHolder] = message.useMessage()
  const { update } = useSession()

  async function createOrder(
    data: CreateOrderData,
    actions: CreateOrderActions
  ) {
    return fetchPost<{ id: string }>('/api/trade', {
      code: code,
      channel: 'paypal'
    }).then((res) => {
      console.log('trade order')
      return res.id
    })
  }

  async function onApprove(data: OnApproveData, actions: OnApproveActions) {
    try{
      await fetchPut('/api/trade/paypal', {
        id: data.orderID,
        tradeNo: data.payerID
      })
      // 通知更新header 页面 session信息
      await update()
      // 关闭弹窗
      closePaypal()
    }catch (e){
      console.error("paypal callback error",e)
      msg.error(e as string)
    }
  }

  function handleClick() {
    window.location.reload()
  }

  return (
    <Spin spinning={isPending} tip={t`Loading...`}>
      <PayPalButtons createOrder={createOrder} onApprove={onApprove} />
      {isRejected && (
        <div className="text-center text-gray-500">
          <p className="text-2xl ">{t`Please try again later.`}</p>
          <p
            className="text-sm ">{t`Initialization of PayPal services encountered an error. `}</p>
          <div className="mt-10 ">
            <Button variant="flat" color="primary" size="lg"
                    startContent={<FaArrowCircleLeft />} onClick={handleClick}>{t`Try it out`}</Button>
          </div>
        </div>
      )}
    </Spin>
  )
}
