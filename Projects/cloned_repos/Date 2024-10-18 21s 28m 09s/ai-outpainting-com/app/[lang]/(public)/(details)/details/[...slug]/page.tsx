import { AVAILABLE_LOCALES, metadataLanguages } from '@/framework/locale/locale'
import React from 'react'
import services from '@/framework/services/OrderService'
import { siteConfig } from '@/config/site'
import {t} from '@lingui/macro'
import { Metadata } from 'next'
export const metadata: Metadata = {
  title: {
    default: siteConfig.name,
    template: t`Pricing | ${siteConfig.name}`
  },
  alternates: {
    languages:metadataLanguages('/details')
  },
}

export default async function DetailsPage({
                                            params
                                          }: {
  params: { lang: AVAILABLE_LOCALES; slug: string[] }
}) {
  const {
    slug: [orderNo]
  } = params
  const order = orderNo ? await services.getByOrderNo(orderNo) : undefined


  return (
    <>

    </>
  )
}
