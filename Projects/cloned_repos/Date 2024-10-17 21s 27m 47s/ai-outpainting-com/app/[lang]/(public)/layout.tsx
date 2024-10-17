import Footer from '@/components/wegic/footer'
import Header from '@/components/wegic/header'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { serverSideTranslations } from '@/framework/locale/serverSideTranslations'
import { AntdRegistry } from '@ant-design/nextjs-registry'
import clsx from 'clsx'
import { ReactNode } from 'react'
import '../../globals.css'
import { Providers } from './providers'

export default async function LocaleLayout({
  children,
  params,
}: {
  children: ReactNode
  params?: { lang: AVAILABLE_LOCALES }
}) {
  const i18n = await serverSideTranslations(
    params?.lang ?? AVAILABLE_LOCALES.en,
  )
  return (
    <html lang={params?.lang} suppressHydrationWarning>
      <head />
      <body
        suppressHydrationWarning
        className={clsx(`min-h-screen font-sans antialiased`)}
      >
        <Providers params={{ i18n }}>
          <AntdRegistry>
            <Header lang={params?.lang} />
            {children}
            <Footer params={params} />
          </AntdRegistry>
        </Providers>
      </body>
    </html>
  )
}
