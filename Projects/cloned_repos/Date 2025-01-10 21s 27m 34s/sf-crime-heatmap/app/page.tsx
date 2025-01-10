'use client'
import dynamic from 'next/dynamic'
import { Suspense, useEffect, useState } from 'react'
import { Skeleton } from "@/components/ui/skeleton"
import Sidebar from '@/components/Sidebar'
import TimeSlider from '@/components/TimeSlider'
import { useSidebarStats } from '@/hooks/useSidebarStats'
import { useIncidents } from '@/hooks/useIncidents'
import { useIsMobile } from '@/components/ui/use-mobile'

const Map = dynamic(() => import('@/components/Map'), {
  loading: () => <Skeleton className="h-[calc(100vh-8rem)] w-full" />,
  ssr: false
})

const MapComponent = () => {
  const isMobile = useIsMobile()

  if (isMobile) {
    return null;
  }

  return (
    <Suspense fallback={<Skeleton className="h-full w-full" />}>
      <Map />
    </Suspense>
  )
}

export default function Home() {
  const sidebar = useSidebarStats()
  const inc = useIncidents()
  if(inc.error || sidebar.error){
    return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background">
      <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
      <div className="mt-4 text-lg text-muted-foreground max-w-md text-center">
        <p className="font-semibold text-red-500 mb-2">Error loading data:</p>
        {inc.error && <p className="mb-2">{inc.error.message}</p>}
        {sidebar.error && <p>{sidebar.error.message}</p>}
      </div>
    </div>
    )
  }
  if(inc.isLoading || sidebar.isLoading){
    return (
      <div className="min-h-screen w-full flex flex-col items-center justify-center bg-background">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        <p className="mt-4 text-lg text-muted-foreground">Loading data, this should only take a minute...</p>
      </div>
    )
  }
  return (
      <main className="flex min-h-screen">
        <Sidebar />
        <div className="hidden md:flex flex-1  flex-col p-4 h-screen overflow-hidden">
          <h1 className="text-2xl font-bold mb-4 text-center">Police Department Incident Reports (2018-2025)</h1>
          <div className="relative flex-1 mb-4">
            <MapComponent />
          </div>
          <div className="w-full px-4 pb-4">
            <TimeSlider />
          </div>
        </div>
      </main>
  )
}
