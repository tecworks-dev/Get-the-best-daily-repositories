import { NextResponse } from 'next/server'
import { discoverSubdomains } from '@/lib/crawl-service'

export async function POST(request: Request) {
  try {
    const { url } = await request.json()

    if (!url) {
      return NextResponse.json(
        { error: 'URL is required' },
        { status: 400 }
      )
    }

    console.log('Making discover request for URL:', url)
    const pages = await discoverSubdomains(url)
    console.log('Received pages from backend:', pages)

    // Even if we get an empty array, we should still return it with a 200 status
    return NextResponse.json({ 
      pages,
      message: pages.length === 0 ? 'No pages discovered' : `Found ${pages.length} pages`
    })
    
  } catch (error) {
    console.error('Error in discover route:', error)
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to discover pages',
        details: error instanceof Error ? error.stack : undefined
      },
      { status: 500 }
    )
  }
}