'use client'
import { t } from '@lingui/macro'
import Link from 'next/link'
import { AVAILABLE_LOCALES } from '@/framework/locale/locale'
import { Blog } from '@/framework/blogs/blogs'



export default function BlogsPageClient({blogs,params}:{params: { lang: AVAILABLE_LOCALES },blogs: Blog[] }) {
  const BlogItem = ({ blog }:{blog: Blog}) => {
    const { title, description, createdAt, image } = blog;

    return (
      <article className="rounded-lg shadow-lg bg-white dark:shadow-none overflow-hidden p-3 md:p-6 h-full">
        <Link href={"/"+params.lang + '/blogs/' + blog.slug} passHref key={blog.slug}>
          {/*<img src={image} alt="" className="h-auto w-full rounded-lg" />*/}
          <div className="mt-6 mb-3">
            <h4 className="font-medium text-2xl mb-2  hover:underline">{title}</h4>
            <p className="mb-2">
            </p>
            <p className="opacity-60 mt-3 mb-6  hover:underline">{description}</p>
          </div>
        </Link>
      </article>
    );
  };
  return (
    <section
      className="w-full max-w-7xl mx-auto rounded-2xl mt-8 mb-8 py-14 md:py-24 text-stone-800 bg-slate-50 dark:bg-slate-900 dark:text-white overflow-hidden">
      <div className="container px-8 md:px-24">
        <div className="grid grid-cols-12 justify-center">
          <div className="col-span-12 lg:col-span-8 lg:col-start-3 lg:col-end-11 text-center">
            <h2 className="text-[32px] lg:text-[45px] leading-none font-bold mb-4">
              {t`AI Outpainting Image Blogs Post`}
            </h2>
            <p className="text-lg font-medium opacity-80 lg:px-12 mb-9">
              {t`"Discover the Magic of Expanding Your Photos with AI"`}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-6 mt-3 md:mt-12 text-center gap-x-6">
          {blogs.map((blog, i) => (
            <div
              className="col-span-6 md:col-span-3 lg:col-span-2 mt-6"
              key={i}
            >
              <BlogItem blog={blog} />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}