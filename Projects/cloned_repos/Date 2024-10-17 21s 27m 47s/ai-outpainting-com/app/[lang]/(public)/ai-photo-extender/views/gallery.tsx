'use client'
import React from 'react'
import GalleryItem from '@/components/wegic/gallery-item'
import {t} from '@lingui/macro'

export default function Gallery() {

  const data: { name: string, images: string[] }[] = [
    {
      name: 'Arts',
      images: [
        'https://public-image.fafafa.ai/fa-image/2024/06/dc94c298dbb6498d38ed5a8fc3fb9293.jpeg',
        'https://public-image.fafafa.ai/fa-image/2024/06/28cf8159d8a7aeb370296993ec380797.png'
      ]
    },
  /*  {
      name: 'Characters',
      images: [
        'https://public-image.fafafa.ai/fa-image/2024/06/0f2cda07684b4eaa3cddde2845983502.jpg',
        'https://public-image.fafafa.ai/fa-image/2024/06/995adacd2006539edbe3359c8b5b1581.png'
      ]
    },
    {
      name: 'Scenery',
      images: [
        'https://public-image.fafafa.ai/fa-image/2024/06/1d0dd4e6b2f4eaddc368330f51064657.jpg',
        'https://public-image.fafafa.ai/fa-image/2024/06/098c96d00543c747402132144e298732.png'
      ]
    }*/
  ]


  return (
    <section className="bg-white dark:bg-slate-800 pt-10">
      <div>
        <div className="max-w-7xl mx-auto py-10 px-4">
          <div className="max-w-lg mx-auto text-center lg:text-left lg:max-w-none lg:mx-0">
            <h2 className="TITLE-PRIMARY text-5xl font-semibold text-slate-900 dark:text-slate-50">
              <div className="_editable_jwu41_1 undefined"
                   data-link="link=&amp;target=_blank&amp;text=AI%20Photo%20Extender%20Gallery">{t`AI Photo Extender Gallery`}
              </div>
            </h2>
            <div className="mt-4 lg:mt-6 lg:grid lg:grid-cols-2 xl:grid-cols-3 lg:gap-8">
              <p className="DESC text-base font-normal text-slate-700 dark:text-slate-300 mt-4 lg:mt-0">
                {t`Browse
                through stunning examples of AI photo extender, showcasing the limitless possibilities of extending
                artworks.`}
              </p>
              <p className="DESC text-base font-normal text-slate-700 dark:text-slate-300 mt-4 lg:mt-0">
                {t`Each
                piece in our gallery demonstrates the power of AI in transforming and enhancing creative projects.`}
              </p>
            </div>
          </div>
          <div className="mt-12 sm:mt-16 lg:mt-16 flex justify-center">
            <GalleryItem data={data} />
          </div>
        </div>
      </div>
    </section>
  )
}