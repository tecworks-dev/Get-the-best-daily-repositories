import type { MetadataRoute } from 'next'
import { keepDays } from '@/config'
import { getPastDays } from '@/lib/utils'
import { getCloudflareContext } from '@opennextjs/cloudflare'
import { headers } from 'next/headers'

export const revalidate = 86400

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const headersList = await headers()
  const host = headersList.get('host') || ''
  const { env } = getCloudflareContext()
  const runEnv = env.NEXTJS_ENV
  const pastDays = getPastDays(keepDays)
  const posts = (await Promise.all(
    pastDays.map(async (day) => {
      const post = await env.HACKER_NEWS_KV.get(`content:${runEnv}:hacker-news:${day}`, 'json')
      return post as unknown as Article
    }),
  )).filter(Boolean)

  return [
    {
      url: `https://${host}`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    ...posts.map(post => ({
      url: `https://${host}/post/${post.date}`,
      lastModified: new Date(post.updatedAt || post.date),
      changeFrequency: 'weekly' as const,
      priority: 0.8,
    })),
  ]
}
