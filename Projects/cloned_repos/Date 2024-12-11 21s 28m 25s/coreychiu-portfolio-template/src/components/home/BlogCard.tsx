import { Card } from '@/components/shared/Card'
import { formatDate } from '@/lib/formatDate'
import { type BlogType } from '@/lib/blogs'

export function BlogCard({ blog, titleAs }: { blog: BlogType, titleAs?: keyof JSX.IntrinsicElements }) {
  const as = titleAs ?? 'h2'
  return (
    <Card as="article">
      <Card.Title as={as} href={`/blogs/${blog.slug}`}>
        {blog.title}
      </Card.Title>
      <Card.Eyebrow as="time" dateTime={blog.date} decorate>
        {formatDate(blog.date)}
      </Card.Eyebrow>
      <Card.Description>{blog.description}</Card.Description>
      <Card.Cta>Read blog</Card.Cta>
    </Card>
  )
}
