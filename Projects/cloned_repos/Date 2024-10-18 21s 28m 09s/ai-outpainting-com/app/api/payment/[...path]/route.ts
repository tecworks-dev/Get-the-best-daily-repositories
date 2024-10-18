import { GET as paypalGET } from '@/framework/apis/PaypalApi'
import {
  createStripeOrder,
  GET as stripeGET,
  listenWebhook
} from '@/framework/apis/StripeApi'
import { wrapPathRoutes } from '@/framework/utils'

export const GET = wrapPathRoutes([
  ['paypal', paypalGET],
  ['stripe', stripeGET]
])

export const POST = wrapPathRoutes([['stripe/webhook', listenWebhook]])
