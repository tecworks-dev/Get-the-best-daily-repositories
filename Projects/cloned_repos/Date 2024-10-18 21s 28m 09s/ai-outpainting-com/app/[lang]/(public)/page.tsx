import React from 'react'
import { activateLocale, AVAILABLE_LOCALES, metadataLanguages } from '@/framework/locale/locale'
import Hero from '@/components/wegic/hero'
import Features from '@/components/wegic/features'
import Gallery from '@/components/wegic/gallery'
import Blogs from '@/components/wegic/blogs'
import FAQs from '@/components/wegic/faqs'
import { Divider } from 'antd'
import { getBlogPosts } from '@/framework/blogs/blogs'
import Pricing from '@/components/wegic/pricing'
import { t } from '@lingui/macro'
import IndexUploader from '@/components/wegic/index-uploader'
import { Metadata } from 'next'
import { siteConfig } from '@/config/site'

export const dynamic = 'force-static'


export async function generateMetadata({
                                         params
                                       }: {
                                         params: { slug: string, lang: AVAILABLE_LOCALES}
                                       }
): Promise<Metadata> {
  // 必须主动激活一下当前语言，否则t函数不生效
  await activateLocale(params.lang)
  const title = t`Free Online Tool For AI Image Expanding `+`-${siteConfig.name}`
  return {
    title,
    description:t`Explore AI Outpainting Image, A Free Online Tool For Expanding and Enhancing Images Effortlessly`,
    alternates: {
      languages:metadataLanguages('/')
    },
    icons: {
      icon: siteConfig.icon,
    }
  }
}

export async function generateStaticParams() {
  // 构建时生成静态页面
  const allLang = []
  for (const langDir of Object.values(AVAILABLE_LOCALES)) {
    allLang.push({ lang: langDir })
  }
  return allLang
}

export default async function Page({
                                     params
                                   }: {
  params?: { lang: AVAILABLE_LOCALES }
}) {
  // 编译期间获取blog数据，避免运行时去调用getBlogPosts方法，导致无法读取blogs目录下的文件
  let blogs = getBlogPosts().filter((post) => post.lang===params?.lang)
  // 最新的5篇博客
  blogs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
  const newBlogs = blogs.slice(0, 5)
  return (
    <>
      <Hero params={params!}/>
      <Features />
      <Gallery />
      <FAQs />
      <Divider className="bg-gray" />
    {/*  <Pricing />*/}
      <Blogs params={{ lang: params!.lang }} blogs={newBlogs} />
      <IndexUploader params={params!}/>
    </>
  )
}
