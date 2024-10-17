import { NextRequest } from 'next/server'
import { auth, config, CurrencyCodes, ItemsBuilder, order, PurchaseUnitBuilder } from 'paypal-rest'
import { Trade } from '@prisma/client'
import { R } from '@/framework/utils'

const clientId = process.env['UE_PAYPAL_CLIENT_ID'] ?? ''
const clientSecret = process.env['UE_PAYPAL_CLIENT_SECRET'] ?? ''

config({
  mode: process.env.NODE_ENV==='development'?'sandbox':'live',
  client_id: clientId,
  client_secret: clientSecret,
  auto_renew: false
})


export function GET(request: NextRequest) {
  return R.ok({ clientId })
}

export async function createPaypalOrder(trade: Trade,buyName:string) {
  await auth().catch((e) => {
    console.error('paypal auth error', e)
  })
  const item = new ItemsBuilder()
    .setName(buyName)
    .setQuantity(1)
    .setUnitAmount({
      currency_code: CurrencyCodes.UnitedStatesDollar,
      value: Number(trade.amount)
    })

  const purchaseUnit = new PurchaseUnitBuilder()
    .setCurrency('USD')
    .setPrice(Number(trade.amount))
    .setDescription(`buy ${trade.planName}`)
    .addItems(item)

  try {
    return await order.create({
      purchase_units: [purchaseUnit]
    })
  } catch (e) {
    console.error('paypal order create error', e)
  }
}

export async function capturePaypalOrder(id: string) {
  return order.capture(id)
}
