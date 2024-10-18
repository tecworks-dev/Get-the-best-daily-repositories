import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { getBlogPosts } from '@/framework/blogs/blogs'
import { notFound } from 'next/navigation'
import { CustomMDX } from '@/framework/blogs/mdx'
import Link from 'next/link'
import { t } from '@lingui/macro'
import { Metadata } from 'next'

export async function generateStaticParams() {
  let posts = getBlogPosts()

  return posts.map((post) => ({
    slug: post.slug,
    lang: post.lang
  }))
}
// 动态生成metadata
export async function generateMetadata({
                                         params
                                       }: {
                                         params: { slug: string, lang: AVAILABLE_LOCALES }
                                       }
): Promise<Metadata> {
  let slug = decodeURIComponent(params.slug)
  let post = getBlogPosts().find((post) => post.lang === params.lang && post.slug === slug)
  return {
    title: post?.title,
    description:post?.description
  }
}

export default async function Page({
                                     params
                                   }: {
  params: { slug: string, lang: AVAILABLE_LOCALES }
}) {
  // console.log('访问blog详情页面', params)
  let slug = decodeURIComponent(params.slug)
  let post = getBlogPosts().find((post) => post.lang === params.lang && post.slug === slug)
  if (!post) {
   return  notFound()
  }
  return (
    <>
      <article
        className="prose prose-sm md:prose-base lg:prose-lg  rounded-2xl max-w-3xl mx-auto py-10 px-4">
        <Link href={`/${params.lang}/blogs`} className="mb-2 text-gray-500">{t`<< Return Blogs List`}</Link>
        <h1>{post.title}</h1>
        <CustomMDX source={post.content} />
      </article>
    </>

  )
}
