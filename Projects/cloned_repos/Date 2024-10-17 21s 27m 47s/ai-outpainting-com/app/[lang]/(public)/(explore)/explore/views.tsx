'use client'

import React, { useMemo, useState } from 'react'
import { t } from '@lingui/macro'
import { Pagination } from '@nextui-org/react'
import { usePathname, useSearchParams } from 'next/navigation'
import { getTotalPages } from '@/framework/utils'
import { Spin } from 'antd'


export interface ExploreStuffListProps {
  page: number
  size: number
  total: number
  stuffs: any[]
}


export function ExploreStuffList({ stuffs, total, page, size }: ExploreStuffListProps) {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const totalPages = useMemo(() => getTotalPages(total, size), [total, size])
  const [isLoading, setIsLoading] = useState(false)

  function handlePageChange(page: number) {
    setIsLoading(true)
    const params = new URLSearchParams(searchParams)
    params.set('page', page.toString())
    window.location.href = pathname + '?' + params.toString()
  }


  return (
    <>
      <section className="custom-screen mt-10 text-white">
        <div className="container px-8 md:px-24">
          <div className="grid grid-cols-12 justify-center">
            <div className="col-span-12 lg:col-span-8 lg:col-start-3 lg:col-end-11 text-center">
              <h2 className="text-[32px] lg:text-[45px] leading-none font-bold mb-4">
                {t`探索人声分离`}
              </h2>
              <p className="text-lg font-medium opacity-80 lg:px-12 mb-9 text-gray-500">
                {t`轻松掌握我们的AI人声分离技术，助您在生活和工作中提高效率。`}
              </p>
            </div>
          </div>
        </div>
      </section>
      <section
        className="custom-screen w-full py-14 md:py-24 dark:bg-[#0b1727] dark:text-white overflow-hidden">
        <Spin spinning={isLoading} className="rounded-xl">
          <div className="grid grid-cols-1 md:grid-cols-4 mt-3 md:mt-12 text-center gap-x-8 gap-y-8">
            {
              stuffs.map((it, index) => (
                <a key={index}
                   className="flex flex-col items-center bg-gray rounded-xl px-2 py-8 transition  hover:scale-105"
                   href={`/details/${it.order.resPagePath}`}>
                  <img
                    src="/images/detail-cover.jpg"
                    className="w-48 bg-gray-50 shadow-xl rounded-xl object-cover"
                    alt="Music cover"
                  />
                  <span className="text-gray-800 font-bold text-sm mt-2 px-5 break-all">{it.originName}</span>
                </a>
              ))
            }
          </div>
        </Spin>
        <div className="flex items-center justify-center md:justify-end mt-5">
          <Pagination loop showControls total={totalPages}
                      boundaries={5}
                      initialPage={1}
                      page={page}
                      size="lg"
                      isDisabled={isLoading}
                      onChange={handlePageChange} />
        </div>
      </section>
    </>
  )
}