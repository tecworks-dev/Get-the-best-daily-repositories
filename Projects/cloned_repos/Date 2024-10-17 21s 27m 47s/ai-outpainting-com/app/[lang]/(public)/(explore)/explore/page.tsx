import React from 'react'
import { ExploreStuffList } from './views'

export default async function Page({ searchParams }: { searchParams?: { page: string } }) {
  const page = Number(searchParams?.page ?? '1')
  const size = 20
  // const [total, stuffs] = await stuffService.getStuffsOfPage(page, size)
/*  return (<ExploreStuffList
    page={page}
    size={size}
    total={(total ?? 0) as number}
    stuffs={stuffs as any[]} />)*/
  return (
    <></>
  )
}