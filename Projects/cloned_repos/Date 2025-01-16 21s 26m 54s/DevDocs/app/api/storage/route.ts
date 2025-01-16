import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { URL } from 'url'

const STORAGE_DIR = path.join(process.cwd(), 'storage', 'markdown')

// Ensure storage directory exists
if (!fs.existsSync(STORAGE_DIR)) {
  fs.mkdirSync(STORAGE_DIR, { recursive: true })
}

function getFilenameFromUrl(url: string): string {
  try {
    const parsedUrl = new URL(url)
    // Use hostname and pathname to create a unique filename
    const filename = `${parsedUrl.hostname}${parsedUrl.pathname.replace(/\//g, '_')}`
      .replace(/[^a-zA-Z0-9-_]/g, '_') // Replace invalid chars with underscore
      .replace(/_+/g, '_') // Replace multiple underscores with single
      .toLowerCase()
    return `${filename}.json`
  } catch (error) {
    // Fallback for invalid URLs
    return `${url.replace(/[^a-zA-Z0-9-_]/g, '_')}.json`
  }
}

// POST /api/storage - Save markdown
export async function POST(request: NextRequest) {
  try {
    const { url, content, stats } = await request.json()
    const filename = getFilenameFromUrl(url)
    const filepath = path.join(STORAGE_DIR, filename)

    const data = {
      url,
      content,
      timestamp: new Date().toISOString(),
      stats
    }

    // Save JSON file
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2))
    
    // Save markdown file
    const markdownPath = filepath.replace('.json', '.md')
    fs.writeFileSync(markdownPath, content)

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error saving markdown:', error)
    return NextResponse.json(
      { error: 'Failed to save markdown' },
      { status: 500 }
    )
  }
}

// GET /api/storage - List files
// GET /api/storage?url=... - Load specific file
export async function GET(request: NextRequest) {
  try {
    const url = request.nextUrl.searchParams.get('url')
    
    // If no URL provided, list all files
    if (!url) {
      const files = fs.readdirSync(STORAGE_DIR)
        .filter(file => file.endsWith('.json'))
        .map(file => {
          const name = file.replace('.json', '')
          const jsonPath = `storage/markdown/${file}`
          const markdownPath = `storage/markdown/${name}.md`
          
          // Only include if both files exist
          if (fs.existsSync(path.join(process.cwd(), jsonPath)) &&
              fs.existsSync(path.join(process.cwd(), markdownPath))) {
            // Get file stats
            const stats = fs.statSync(path.join(process.cwd(), jsonPath))
            const content = JSON.parse(fs.readFileSync(path.join(process.cwd(), jsonPath), 'utf-8'))
            
            // Clean up the name by removing common prefixes and file extensions
            const cleanName = name
              .replace(/^docs[._]/, '')  // Remove leading docs prefix
              .replace(/\.json$/, '')    // Remove .json extension
              .replace(/\.md$/, '')      // Remove .md extension

            return {
              name: cleanName,           // Use cleaned name
              jsonPath,
              markdownPath,
              timestamp: stats.mtime,  // Keep as Date for sorting
              size: stats.size,
              wordCount: content.stats?.wordCount || 0,
              charCount: content.stats?.charCount || 0
            }
          }
          return null
        })
        .filter((file): file is NonNullable<typeof file> => file !== null) // Type-safe filter
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()) // Sort by newest first
      
      return NextResponse.json({ files })
    }

    // Load specific file
    const filename = getFilenameFromUrl(url)
    const filepath = path.join(STORAGE_DIR, filename)

    if (!fs.existsSync(filepath)) {
      return NextResponse.json(
        { error: 'No stored content found' },
        { status: 404 }
      )
    }

    const content = fs.readFileSync(filepath, 'utf-8')
    return NextResponse.json(JSON.parse(content))
  } catch (error) {
    console.error('Error loading markdown:', error)
    return NextResponse.json(
      { error: 'Failed to load markdown' },
      { status: 500 }
    )
  }
}