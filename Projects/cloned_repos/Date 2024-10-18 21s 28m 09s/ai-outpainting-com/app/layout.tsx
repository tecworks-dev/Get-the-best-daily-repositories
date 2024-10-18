import React, { ReactNode } from 'react'
import clsx from 'clsx'
import './globals.css'
import { GoogleAnalytics } from '@next/third-parties/google'
import MicrosoftClarity from '@/framework/components/MicrosoftClarity'


export default async function AppLayout({ children }: { children: ReactNode }) {
  const isDev = process.env.NODE_ENV === 'development'

  return (
    <html suppressHydrationWarning>
    <head />
    <body
      suppressHydrationWarning
      className={clsx('min-h-screen Roboto,ui-sans-serif, system-ui, -apple-system, blinkmacsystemfont, "Segoe UI", roboto, "Helvetica Neue", arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"')}
    >
    {children}
    {
      /*G-SJ0Z7VPH67 AI-outpainting专属*/
      /*mxys3i7q8w  AI-outpainting专属*/
      !isDev &&
      (
        <>
          <GoogleAnalytics gaId="G-SJ0Z7VPH67" />
          <MicrosoftClarity clarityId="mxys3i7q8w"/>
        </>
      )
    }
    </body>
    </html>
  )
}
