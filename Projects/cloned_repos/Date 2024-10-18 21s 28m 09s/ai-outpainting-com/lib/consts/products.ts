import { Product, RechargePlan } from '@/framework/types'
import {t} from '@lingui/macro'
// 免费 5 次
export const FREE_GIVE_CREDITS = 5
export const PRODUCTS: Product[] = [
  {
    code: 'AI-OPI',
    name: 'AI-Outpainting',
    credit: 1
  }
]

export const RECHARGES: RechargePlan[] = [
  {
    code: 'trial',
    price: '4.9',
    credit: 60,
    name: `Basic`,
    buyName:`Buy AI-Outpainting Trial Package`,
    desc: `Get started with your business`,
    isMostPop: false,
    features: [
      `Upload your photo`,
      `Permanently valid`,
      `Fast generation`,
      `High quality images`,
    ]
  },
  {
    code: 'standard',
    price: '12.9',
    credit: 150,
    name: `Pro`,
    buyName:`Buy AI-Outpainting Standard Package`,
    desc: `Extends support for your business`,
    isMostPop: true,
    features: [
      `Upload your photo`,
      `Permanently valid`,
      `Fast generation`,
      `High quality images`,
    ]
  },
  {
    code: 'business',
    credit: 800,
    name: `Ultimate`,
    buyName:`Buy AI-Outpainting Ultimate Package`,
    desc: `The ultimate bundle for your business`,
    price: '36.9',
    isMostPop: false,
    features: [
      `Upload your photo`,
      `Permanently valid`,
      `Fast generation`,
      `High quality images`,
    ]
  },
  {
    code: 'free',
    credit: 5,
    name: `Free`,
    buyName:`Buy AI-Outpainting Ultimate Package`,
    desc: `New users who log in will receive 5 credits`,
    price: '0.0',
    isMostPop: false,
    features: [
      `Upload your photo`,
      `Permanently valid`,
      `Fast generation`,
      `High quality images`,
    ]
  }
]
