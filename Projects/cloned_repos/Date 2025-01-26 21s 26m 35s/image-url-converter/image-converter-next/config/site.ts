import { SiteConfig } from "@/types/siteConfig";

export const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || "https://starter.weijunext.com";

const baseSiteConfig = {
  name: "Image URL Converter",
  description: "A powerful and free tool to convert any image URL into a permanent CDN link. Built with Cloudflare R2 and Next.js, offering global CDN acceleration and permanent storage.",
  url: BASE_URL,
  metadataBase: BASE_URL,
  keywords: [],
  authors: [
    {
      name: "weijunext",
      url: "https://weijunext.com",
      twitter: 'https://x.com/weijunext',
    }
  ],
  creator: '@weijunext',
  themeColors: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
  defaultNextTheme: 'system', // next-theme option: system | dark | light
  icons: {
    icon: "/favicon.ico",
    shortcut: "/logo.png",
    apple: "/logo.png", // apple-touch-icon.png
  },
}

export const siteConfig: SiteConfig = {
  ...baseSiteConfig,
  openGraph: {
    type: "website",
    locale: "en-US",
    url: baseSiteConfig.url,
    title: baseSiteConfig.name,
    description: baseSiteConfig.description,
    siteName: baseSiteConfig.name,
    images: [`${baseSiteConfig.url}/og.png`],
  },
  twitter: {
    card: "summary_large_image",
    title: baseSiteConfig.name,
    site: baseSiteConfig.url,
    description: baseSiteConfig.description,
    images: [`${baseSiteConfig.url}/og.png`],
    creator: baseSiteConfig.creator,
  },
}
