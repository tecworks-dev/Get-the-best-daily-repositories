import { type Metadata } from 'next'

import { Providers } from '@/app/providers'
import { Layout } from '@/components/layout/Layout'
import { Analytics } from "@/components/analytics/analytics";
import { name, headline, introduction } from '@/config/infoConfig'
import '@/styles/tailwind.css'

export const metadata: Metadata = {
  title: {
    template: `%s - ${name}`,
    default:
      `${name} - ${headline}`,
  },
  description:
    `${introduction}`,
  alternates: {
    types: {
      'application/rss+xml': `${process.env.NEXT_PUBLIC_SITE_URL}/feed`,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <body className="flex h-full">
        <Providers>
          <div className="flex w-full">
            <Layout>{children}</Layout>
          </div>
          <Analytics />
        </Providers>
      </body>
    </html>
  )
}
