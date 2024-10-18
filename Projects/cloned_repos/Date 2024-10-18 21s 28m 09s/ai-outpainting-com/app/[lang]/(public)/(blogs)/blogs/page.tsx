import { activateLocale, AVAILABLE_LOCALES, metadataLanguages } from '@/framework/locale/locale'
import BlogsPageClient from '@/app/[lang]/(public)/(blogs)/blogs/blogs-page-client'
import { getBlogPosts } from '@/framework/blogs/blogs'
import { Metadata } from 'next'
import { siteConfig } from '@/config/site'
import { t } from '@lingui/macro'

export async function generateMetadata({
                                         params
                                       }: {
                                         params: { slug: string, lang: AVAILABLE_LOCALES}
                                       }
): Promise<Metadata> {
  await activateLocale(params.lang)
  return {
    title:t`Blogs`+` | ${siteConfig.name}`,
    alternates: {
      languages:metadataLanguages('/blogs')
    },
  }
}
export async function generateStaticParams() {
  // 构建时生成静态页面
  const allLang = []
  for (const langDir of Object.values(AVAILABLE_LOCALES)) {
    allLang.push({lang: langDir})
  }
  console.log('allLang', allLang)
  return allLang
}
export default async function BlogsPage({
  params,
}: {
  params: { lang: AVAILABLE_LOCALES }
}) {
  const blogs = getBlogPosts().filter((blog)=>blog.lang===params.lang)
  return (
    <BlogsPageClient params={params} blogs={blogs} />
  )
}
