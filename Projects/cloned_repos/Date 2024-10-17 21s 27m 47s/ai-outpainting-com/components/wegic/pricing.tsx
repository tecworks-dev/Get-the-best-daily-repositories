'use client'
import { RechargePlan } from '@/framework/types'
import { Button, Tooltip } from '@nextui-org/react'
import React, { useRef, useState } from 'react'
import PaymentModal, { PaymentRef } from '@/components/payment/payment-modal'
import { t } from '@lingui/macro'
import { RECHARGES } from '@/lib/consts/products'
import { WrappedPaypalProvider } from '@/components/payment/provider'
import { signIn, useSession } from 'next-auth/react'
import GoogleLogin from '@/framework/components/login/GoogleLogin'
import { GoogleLoginRef } from '@/framework/components/login/types'
import { FcGoogle } from 'react-icons/fc'

type PricingPlan = {
  desc: string
  isMostPop: boolean
  features: string[]
} & RechargePlan

const PricingSection = () => {
  const [plan, setPlan] = useState(RECHARGES[0])
  const loginRef = useRef<GoogleLoginRef>(null)
  const paymentRef = useRef<PaymentRef>(null)
  const {status} = useSession()
  const features =  [
    t`Upload your photo`,
    t`Permanently valid`,
    t`Fast generation`,
    t`High quality images`,
  ]
  const translateRechargePlans = (plans: RechargePlan[]) => {
    return plans.map(plan => ({
      ...plan,
      name: t`${plan.name}`,
      buyName: t`${plan.buyName}`,
      desc: t`${plan.desc}`,
      features: features
    }))
  }
  const TRANS_RECHARGES = translateRechargePlans(RECHARGES)
  async function handleClick(plan: PricingPlan) {
    setPlan(plan)
    const authenticated = loginRef.current?.checkAuthenticated()
    // 未登录不可支付
    if (!authenticated) {
      loginRef.current?.open()
      return false;
    }else{
      paymentRef.current?.open()
    }
  }


  return (
    <>
      <section className="relative px-2 py-4 md:px-8 md:py-12 max-h-[80vh] overflow-y-auto">
        <GoogleLogin ref={loginRef}/>

        <PaymentModal
          ref={paymentRef}
          plan={plan}
        />
        <div className="w-full mx-auto items-center gap-4 md:gap-6">
          <div className="relative mx-auto text-center mb-4 md:mb-8">
            <h1 className="text-xl font-semibold sm:text-2xl md:text-3xl lg:text-4xl">
              {t`Pricing Plans`}
            </h1>
            <div className="mt-1 md:mt-3">
              <p className="text-gray-600 text-xs sm:text-sm md:text-base">
                {t`Purchase on demand, no subscription fee, lifetime validity`}
              </p>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 md:gap-6">
            {TRANS_RECHARGES.map((item, idx) => (
              <div
                key={idx}
                className={`bg-white relative flex flex-col rounded-xl border-2 transition-all duration-300 hover:shadow-lg ${item.isMostPop ? 'border-slate-600 md:scale-105' : 'md:hover:scale-105'}`}
              >
                {item.isMostPop && (
                  <p className="absolute right-2 top-0 -translate-y-1/2 font-heading px-2 py-0.5 text-xs max-w-max text-white tracking-[1px] rounded-full bg-primary md:text-sm md:px-3 md:py-1">
                    {t`Recommended`}
                  </p>
                )}

                <div className="p-3 md:p-5 space-y-1 border-b text-center">
                  <span className="text-slate-600 font-medium text-xs md:text-sm">
                    {item.name}
                  </span>
                  <div className="text-gray-800 text-xl font-semibold md:text-2xl lg:text-3xl">
                    ${item.price}{' '}
                    <span className="text-xs text-gray-600 font-normal md:text-sm">
                      / {item.credit}{' '}{t`credit`}
                    </span>
                  </div>
                  <p className="text-xs md:text-sm">{item.desc}</p>
                </div>
                <div className="p-3 md:p-5">
                  <ul className="space-y-1 text-xs hidden sm:block md:text-sm md:space-y-2">
                    {item.features.map((featureItem, idx) => (
                      <li key={idx} className="flex items-center gap-1 md:gap-2">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-3 w-3 text-slate-600 flex-shrink-0 md:h-4 md:w-4"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          ></path>
                        </svg>
                        {featureItem}
                      </li>
                    ))}
                  </ul>
                  <div className="pt-3 md:pt-5">
                    {item.code === 'free' ? (
                      <Tooltip showArrow={true} color="danger" content={t`New users who log in will receive 5 credits`}>
                        <Button 
                          onClick={() => signIn('google')} 
                          color={'primary'} 
                          disabled={status === 'authenticated'}
                          startContent={<FcGoogle size="1.2em" color="white" />}
                          className="w-full text-xs md:text-sm md:py-2"
                        >
                          {status === 'authenticated' ? t`Has logged in` : t`Sign In With Google`}
                        </Button>
                      </Tooltip>
                    ) : (
                      <button
                        className={`py-1.5 md:py-2 w-full rounded-full text-white text-xs md:text-sm ring-offset-2 focus:ring transition-all duration-300 ${
                          item.isMostPop
                            ? 'bg-slate-600 hover:bg-slate-500 focus:bg-slate-700 ring-slate-600'
                            : 'bg-gray-800 hover:bg-gray-700 ring-gray-800'
                        }`}
                        onClick={() => handleClick(item)}
                      >
                        {t`Get Started`}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}


export default function Pricing({ locale }: { locale?: string }) {
  return (
    <WrappedPaypalProvider locale={locale}>
      <PricingSection/>
    </WrappedPaypalProvider>
  )
}