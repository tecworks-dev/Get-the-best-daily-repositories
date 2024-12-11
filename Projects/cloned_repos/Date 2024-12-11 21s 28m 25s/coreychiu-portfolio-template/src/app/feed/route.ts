import assert from 'assert'
import * as cheerio from 'cheerio'
import { Feed } from 'feed'
import { name, email } from '@/config/infoConfig'
import { getBlogBySlug } from '@/lib/blogs'
import { promises as fs } from 'fs'
import path from 'path'
import matter from 'gray-matter'

export async function GET(req: Request) {
  let siteUrl = process.env.NEXT_PUBLIC_SITE_URL

  if (!siteUrl) {
    throw Error('Missing NEXT_PUBLIC_SITE_URL environment variable')
  }

  let author = {
    name: name,
    email: email,
  }

  let feed = new Feed({
    title: author.name,
    description: name + '\'s blog',
    author,
    id: siteUrl,
    link: siteUrl,
    image: `${siteUrl}/favicon.ico`,
    favicon: `${siteUrl}/favicon.ico`,
    copyright: `All rights reserved ${name} ${new Date().getFullYear()}`,
    feedLinks: {
      rss2: `${siteUrl}/feed`,
    },
  })

  // 直接读取博客文件
  const blogFiles = await fs.readdir(path.join(process.cwd(), 'src/content/blog'))
  const mdxFiles = blogFiles.filter(file => file.endsWith('.mdx'))

  for (let file of mdxFiles) {
    const slug = file.replace(/\.mdx$/, '')
    const filePath = path.join(process.cwd(), 'src/content/blog', file)
    const source = await fs.readFile(filePath, 'utf-8')
    const { data, content } = matter(source)

    feed.addItem({
      title: data.title,
      id: `${siteUrl}/blogs/${slug}`,
      link: `${siteUrl}/blogs/${slug}`,
      description: data.description,
      content: content,
      author: [author],
      date: new Date(data.date),
    })
  }

  return new Response(feed.rss2(), {
    status: 200,
    headers: {
      'Content-Type': 'application/xml',
      'cache-control': 's-maxage=86400', // one day cache
    },
  })
}
