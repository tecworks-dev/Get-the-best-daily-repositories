'use client'
import { Tab, Tabs } from '@nextui-org/react'
import React, { memo } from 'react'
import BeforeAfterSlider from '@/components/before-after'

export type GalleryItemData = {
  name: string,
  images: string[]
}
const GalleryItem = memo(({ data }: { data: GalleryItemData[] }) => {

  return (
    <Tabs color={'primary'} placement="top" radius="lg" aria-label="Gallery Image Tabs">
      {
        data.map(({ name, images: [before, after] }, index) => (
          <Tab key={name} title={name}>
            <div className="max-w-2xl">
              <BeforeAfterSlider key={index}
                                 firstImage={{imageUrl:before,alt:'outpainting before image'}}
                                 secondImage={{imageUrl:after,alt:'outpainting after image'}}
            />
            </div>

          </Tab>
        ))
      }
    </Tabs>
  )
})
GalleryItem.displayName = 'GalleryItem'
export default GalleryItem