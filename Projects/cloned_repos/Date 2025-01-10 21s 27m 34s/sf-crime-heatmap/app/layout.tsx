import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { MotherDuckClientProvider } from '@/lib/motherduck/context/motherduckClientContext'
import { TimeProvider } from '@/contexts/TimeContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SF Crime Heatmap',
  description: 'Interactive heatmap of crime incidents in San Francisco',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <MotherDuckClientProvider>
          <TimeProvider>
            {children}
          </TimeProvider>
        </MotherDuckClientProvider>
      </body>
    </html>
  )
}
