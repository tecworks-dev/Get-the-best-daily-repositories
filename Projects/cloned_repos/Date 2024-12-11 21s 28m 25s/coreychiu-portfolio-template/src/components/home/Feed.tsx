"use client"

import { Button } from '@/components/ui/button'
import { Rss, Copy, ClipboardText } from '@phosphor-icons/react'
import { Check } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

export default function Feed() {
  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL
  const feedUrl = `${siteUrl}/feed`
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(feedUrl)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <div className="rounded-2xl border p-6 border-muted shadow-sm">
      <h2 className="flex text-sm font-semibold">
        <Rss size={26} weight="duotone" />
        <span className="ml-3">Subscribe my blogs</span>
      </h2>
      <p className="mt-4 ml-1 text-sm text-muted-foreground">
        Welcome to Subscribe my blogs
      </p>
      <div className="mt-4 flex gap-4">
        <div className="relative flex-auto">
          <div 
          className={cn("min-w-0 flex-auto rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background",
            copied ? 'text-primary': ''
          )}
          >
            {feedUrl}
          </div>
          <Button
            size="icon"
            variant="ghost"
            onClick={handleCopy}
            className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
          >
            {copied ? (
              <Check className="h-4 w-4 text-primary" />
            ) : (
              <ClipboardText size={32} weight="duotone" />
            )}
            <span className="sr-only">
              {copied ? 'Copied!' : 'Copy'}
            </span>
          </Button>
        </div>
        <Button 
          onClick={() => window.open(feedUrl, '_blank')} 
          className="hidden md:block flex-none bg-primary text-primary-foreground"
        >
          Subscribe
        </Button>
      </div>
    </div>
  )
}