import { RechargePlan } from '@/framework/types'
import { NextRequest, NextResponse } from 'next/server'
import services from '@/framework/services/TradeService'
import {  R, toQueryParams } from '@/framework/utils'
import { capturePaypalOrder, createPaypalOrder } from '@/framework/apis/PaypalApi'
import { $Enums } from '@prisma/client'
import { createStripeOrder } from '@/framework/apis/StripeApi'

const CHANNEL_PAYPAL = 'paypal'
const CHANNEL_STRIPE = 'stripe'

export function wrapGetRouter<T extends RechargePlan[] = RechargePlan[]>(
  config: T
) {
  return async function(
    request: NextRequest,
    { params: { tradeNo } }: { params: { tradeNo: string } }
  ) {
    const trade = await services.getByTradeNo(tradeNo)
    return NextResponse.json(trade)
  }
}

export function wrapPostRouter<T extends RechargePlan[] = RechargePlan[]>(
  config: T
) {
  function findPlan(code: string) {
    return (config || []).find((it) => {
      return it.code === code
    })
  }

  return async function(request: NextRequest) {
    const body = (await request.json()) as any
    const { code, channel, returnUrl } = body
    const plan = findPlan(code)
    if (!plan) {
      console.error('未知充值套餐信息：' + code)
      return R.bad('Unknown recharge package information：' + code)
    }
    const trade = await services.createTrade(plan, channel)
    try {
      let order = null as any
      if (CHANNEL_PAYPAL === channel) {
        order = await createPaypalOrder(trade,plan.buyName)
      } else if (CHANNEL_STRIPE === channel) {
        if (!returnUrl) {
          console.error('returnUrl必填' )
          return R.bad('returnUrl is null')
        }
        order = await createStripeOrder(trade, returnUrl)
      }
      if(order==null){
        return R.bad('order is null')
      }
      await services.updateById({
        ...trade,
        outTradeNo: order.id,
        outTradeStatus: order.status
      } as any)
      return R.ok(order)
    } catch (e: any) {
      console.error('创建交易订单异常：%s', e)
      await services.updateById({
        ...trade,
        status: $Enums.TradeStatus.PAY_FAILED,
        reason: e?.raw?.message ?? String(e)
      } as any)
    }
    return R.error('CreateTradeError')
  }
}

export async function PUT(request: NextRequest) {
  const { id } = toQueryParams(request)
  // 此处的id保存到数据库的outTradeNo当中
  const trade = await services.getByOutTradeNo(id)
  try {
    await capturePaypalOrder(id)
    console.info(`用户支付成功,outTradeNo:${id},Amount:${trade?.amount},userEmail:${trade?.userEmail},credit:${trade?.credit}`)
    await services.paySuccessCallback({
      ...trade,
      status: $Enums.TradeStatus.PAY_SUCCEED,
      outTradeStatus: 'COMPLETED'
    } as any)
    return R.ok()
  } catch (e) {
    console.error('捕获交易订单支付异常：', e)
    await services.updateById({
      ...trade,
      status: $Enums.TradeStatus.PAY_FAILED,
      reason: e
    } as any)
  }
  return R.error('CaptureTradeError')
}
