'use client'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { t } from '@lingui/macro'
import Link from 'next/link'
import { Blog } from '@/framework/blogs/blogs'

export default  function BlogsPage({
  params,
  blogs
}: {
  blogs:Blog[]
  params: { lang: AVAILABLE_LOCALES }
}) {
  // 只取最近的5篇文章
  // 7) Sort the blogs by createdAt in descending order

  return (
    <section className="relative px-6 bg-gray-100 py-20 md:px-8 md:py-10">
      <div className="w-full max-w-7xl mx-auto  items-center gap-16 md:grid-cols-2 md:gap-24">
        <h2 className="text-3xl font-bold text-center">{t`Blogs`}</h2>
        <section className="py-10">
          <div className="py-2">
            {blogs.map((blog: any) => (
              <Link
                href={"/" + params.lang + '/blogs/' + blog.slug}
                passHref
                key={blog.slug}
              >
                <div className="py-2 flex justify-between align-middle gap-2">
                  <div className="flex-1 mx-2">
                    <h3 className="text-lg font-bold hover:underline">{blog.title}</h3>
                    <p className="text-gray-500 hover:underline">{blog.description}</p>
                    <p  className="hidden md:block  w-64  text-gray-300 text-sm mr-2">{t`Publish Date:`}{blog.createdAt}</p>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      </div>
    </section>
  )
}
