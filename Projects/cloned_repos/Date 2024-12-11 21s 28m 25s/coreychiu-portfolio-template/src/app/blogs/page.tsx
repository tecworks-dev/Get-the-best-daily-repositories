import { type Metadata } from 'next'

import { Card } from '@/components/shared/Card'
import { SimpleLayout } from '@/components/layout/SimpleLayout'
import { type BlogType, getAllBlogs } from '@/lib/blogs'
import { formatDate } from '@/lib/formatDate'
import { blogHeadLine, blogIntro } from '@/config/infoConfig'

export const runtime = process.env.NEXT_RUNTIME === 'edge' ? 'edge' : 'nodejs'

function Blog({ blog }: { blog: BlogType }) {
  return (
    <article className="md:grid md:grid-cols-4 md:items-baseline">
      <Card className="md:col-span-3">
        <Card.Title href={`/blogs/${blog.slug}`}>
          {blog.title}
        </Card.Title>
        <Card.Eyebrow
          as="time"
          dateTime={blog.date}
          className="md:hidden"
          decorate
        >
          {formatDate(blog.date)}
        </Card.Eyebrow>
        <Card.Description>{blog.description}</Card.Description>
        <Card.Cta>Read blog</Card.Cta>
      </Card>
      <Card.Eyebrow
        as="time"
        dateTime={blog.date}
        className="mt-1 hidden md:block"
      >
        {formatDate(blog.date)}
      </Card.Eyebrow>
    </article>
  )
}

export const metadata: Metadata = {
  title: 'Blogs',
  description:
    blogIntro
}

export default async function BlogsIndex() {
  let blogs = await getAllBlogs()

  return (
    <SimpleLayout
      title={blogHeadLine}
      intro={blogIntro}
    >
      <div className="md:border-l md:border-zinc-100 md:pl-6 md:dark:border-zinc-700/40">
        <div className="flex max-w-3xl flex-col space-y-16">
          {blogs.map((blog: BlogType) => (
            <Blog key={blog.slug} blog={blog} />
          ))}
        </div>
      </div>
    </SimpleLayout>
  )
}
