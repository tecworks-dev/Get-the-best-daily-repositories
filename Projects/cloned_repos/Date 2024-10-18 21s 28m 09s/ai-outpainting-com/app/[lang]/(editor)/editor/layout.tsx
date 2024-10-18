import '../../../globals.css'
import { Providers } from '../../(public)/providers'
import React, { ReactNode } from 'react'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import clsx from 'clsx'
import { serverSideTranslations } from '@/framework/locale/serverSideTranslations'
import { AntdRegistry } from '@ant-design/nextjs-registry'


export default async function LocaleLayout({
                                             children,
                                             params
                                           }: {
  children: ReactNode
  params?: { lang: AVAILABLE_LOCALES }
}) {
  const i18n = await serverSideTranslations(
    params?.lang ?? AVAILABLE_LOCALES.en
  )
  return (
    <html lang={params?.lang} suppressHydrationWarning>
    <head />
    <body
      suppressHydrationWarning
      className={clsx(
        `min-h-screen font-sans antialiased`
      )}
    >
    <Providers params={{ i18n }}>
      <AntdRegistry>
        {children}
      </AntdRegistry>
    </Providers>
    </body>
    </html>
  )
}
