import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'DevDocs',
  description: 'Discover and extract documentation for quicker development',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
