'use client'

import { useState } from 'react'
import { Button, ScrollArea } from "@/components/ui"
import { Download, Copy, Check, FileText } from 'lucide-react'

interface MarkdownOutputProps {
  markdown: string
  isVisible: boolean
}

export default function MarkdownOutput({ markdown, isVisible }: MarkdownOutputProps) {
  const [copied, setCopied] = useState(false)

  if (!isVisible) return null

  const handleDownload = () => {
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'crawled-content.md'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleCopy = async () => {
    await navigator.clipboard.writeText(markdown)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const wordCount = markdown.trim().split(/\s+/).length
  const charCount = markdown.length

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex justify-between items-center bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
        <div className="flex items-center gap-3">
          <FileText className="w-5 h-5 text-green-400" />
          <h2 className="text-xl font-semibold text-green-400">Generated Content</h2>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleCopy}
            variant="outline"
            className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 text-green-400" />
                <span className="text-green-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4 text-white" />
                <span className="text-white">Copy</span>
              </>
            )}
          </Button>
          <Button
            onClick={handleDownload}
            className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
          >
            <Download className="w-4 h-4" />
            <span>Download</span>
          </Button>
        </div>
      </div>

      {/* Content Stats */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-800/30 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <span className="text-sm text-gray-400">Words</span>
          <p className="text-xl font-bold text-green-400">{wordCount.toLocaleString()}</p>
        </div>
        <div className="bg-gray-800/30 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
          <span className="text-sm text-gray-400">Characters</span>
          <p className="text-xl font-bold text-green-400">{charCount.toLocaleString()}</p>
        </div>
      </div>

      {/* Content Area */}
      <ScrollArea className="h-[500px] w-full rounded-xl border border-gray-700 bg-gray-900/50 backdrop-blur-sm">
        <div className="p-4">
          <pre className="font-mono text-sm whitespace-pre-wrap text-gray-300">
            {markdown}
          </pre>
        </div>
      </ScrollArea>
    </div>
  )
}