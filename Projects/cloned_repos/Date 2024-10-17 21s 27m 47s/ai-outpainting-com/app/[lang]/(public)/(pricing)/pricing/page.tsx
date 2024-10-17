import { activateLocale, AVAILABLE_LOCALES, metadataLanguages } from '@/framework/locale/locale'
import FAQs from '@/components/wegic/faqs'
import Pricing from '@/components/wegic/pricing'
import { siteConfig } from '@/config/site'
import { Metadata } from 'next'
import { t } from '@lingui/macro'


export async function generateMetadata({
                                         params
                                       }: {
                                         params: { slug: string, lang: AVAILABLE_LOCALES}
                                       }
): Promise<Metadata> {
  await activateLocale(params.lang)
  return {
    title:t`Pricing`+` | ${siteConfig.name}`,
    alternates: {
      languages:metadataLanguages('/pricing')
    },
  }
}
export default function PricingPage({
                                            params
                                          }: {
  params?: { lang: AVAILABLE_LOCALES }
}) {
  return (
    <div className='container mx-auto'  >
      <Pricing locale={params?.lang} />
      <FAQs />
    </div>
  )
}
