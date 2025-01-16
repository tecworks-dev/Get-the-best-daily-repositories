import { NextResponse } from 'next/server'
import { DiscoveredPage } from '@/lib/types'

export async function POST(request: Request) {
  try {
    const { pages } = await request.json()

    if (!Array.isArray(pages)) {
      return NextResponse.json(
        { error: 'Pages array is required' },
        { status: 400 }
      )
    }

    // TODO: Replace with actual Crawl4AI Python backend call
    // For now, return mock markdown data for testing the UI
    const mockMarkdown = `# Documentation
${pages.map((page: DiscoveredPage) => `
## ${page.title || 'Untitled Page'}
URL: ${page.url}

This is mock content for ${page.title || 'this page'}. 
It will be replaced with actual crawled content from the Crawl4AI backend.

---`).join('\n')}
`

    // Simulate network delay and processing time
    await new Promise(resolve => setTimeout(resolve, 2000))

    return NextResponse.json({ 
      markdown: mockMarkdown,
      stats: {
        pagesCrawled: pages.length,
        totalWords: mockMarkdown.split(/\s+/).length,
        dataSize: `${Math.round(mockMarkdown.length / 1024)} KB`
      }
    })
  } catch (error) {
    console.error('Error in crawl route:', error)
    return NextResponse.json(
      { error: 'Failed to crawl pages' },
      { status: 500 }
    )
  }
}