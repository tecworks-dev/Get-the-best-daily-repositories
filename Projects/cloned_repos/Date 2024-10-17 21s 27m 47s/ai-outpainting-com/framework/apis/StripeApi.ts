import { NextRequest } from 'next/server'
import { R } from '@/framework/utils'
import Stripe from 'stripe'
import { Trade } from '@prisma/client'

const publicKey = process.env['UE_STRIPE_PK'] ?? ''
const secretKey = process.env['UE_STRIPE_SK'] ?? ''

const stripe = new Stripe(secretKey)

export async function GET(request: NextRequest) {
  return R.ok({
    clientId: publicKey
  })
}

export async function createStripeOrder(trade: Trade, returnUrl: string) {
  return await stripe.paymentIntents.create({
    amount: 50,
    currency: 'USD',
    description: `buy a ${trade.planName}`
  })
}

export async function listenWebhook(request: NextRequest) {
  const sig = request.headers.get('stripe-signature')!

  let event
  try {
    event = stripe.webhooks.constructEvent(await request.json(), sig, '')
  } catch (e) {
    return R.bad('webhook init error')
  }

  switch (event.type) {
    case 'payment_intent.canceled':
      const paymentIntentCanceled = event.data.object
      // Then define and call a function to handle the event payment_intent.canceled
      break
    case 'payment_intent.created':
      const paymentIntentCreated = event.data.object
      // Then define and call a function to handle the event payment_intent.created
      break
    case 'payment_intent.payment_failed':
      const paymentIntentPaymentFailed = event.data.object
      // Then define and call a function to handle the event payment_intent.payment_failed
      break
    case 'payment_intent.succeeded':
      const paymentIntentSucceeded = event.data.object
      // Then define and call a function to handle the event payment_intent.succeeded
      break
    // ... handle other event types
    default:
      console.log(`Unhandled event type ${event.type}`)
  }
  return R.ok()
}
