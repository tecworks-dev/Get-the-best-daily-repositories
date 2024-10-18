import { Metadata } from 'next'
import { t } from '@lingui/macro'
import '../globals.css'
import { ReactNode } from 'react'

export const metadata: Metadata = {
  title: t`博客列表`,
}
export default async function BlogsLayout({children,}: {children: ReactNode }) {
  return (
    {children}
  )
}
