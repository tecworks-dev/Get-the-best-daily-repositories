import orderService from '@/framework/services/OrderService'
import { CrossSection, EmptyTips } from './views'
import { first, isString } from 'radash'
import { activateLocale, AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { siteConfig } from '@/config/site'

import { t } from '@lingui/macro'
import { Metadata } from 'next'

// 动态生成metadata
export async function generateMetadata({
                                         params
                                       }: {
                                         params: { slug: string, lang: AVAILABLE_LOCALES }
                                       }
): Promise<Metadata> {
  await activateLocale(params.lang)

  const title = t`AI Outpainting Result`  +` | ${siteConfig.name}`
  return {
    title,
  }
}

export default async function Playground({ params: { slug,lang } }: { params: { slug: string | string[] ,lang: AVAILABLE_LOCALES} }) {
  const orderNo = isString(slug) ? slug : first(slug as string[])
  const order = orderNo ? await orderService.getByOrderNo(orderNo) : undefined
  const history = await orderService.getOrderHistory()
  return (
    <>
    <section className=" bg-gray-200 py-10">
      <section className="container mx-auto bg-white  rounded-2xl">
        {
          order ? (
            <CrossSection order={order} history={history} params={{ lang }}></CrossSection>
          ) : <EmptyTips />
        }
      </section>
    </section>
    </>
  )
}