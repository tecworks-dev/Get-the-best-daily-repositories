import { Button, ScrollArea } from "@/components/ui"
import { DiscoveredPage } from "@/lib/types"
import { Globe, Loader2, CheckCircle2, AlertCircle, Link as LinkIcon } from 'lucide-react'
import { useState, useEffect } from 'react'

interface SubdomainListProps {
  subdomains: DiscoveredPage[]
  onCrawlSelected: (selectedUrls: string[]) => void
  isProcessing: boolean
}

export default function SubdomainList({ subdomains, onCrawlSelected, isProcessing }: SubdomainListProps) {
  const [selectedPages, setSelectedPages] = useState<Set<string>>(new Set())

  // Reset selection when subdomains change
  useEffect(() => {
    setSelectedPages(new Set())
  }, [subdomains])

  const togglePage = (url: string) => {
    const newSelected = new Set(selectedPages)
    if (newSelected.has(url)) {
      newSelected.delete(url)
    } else {
      newSelected.add(url)
    }
    setSelectedPages(newSelected)
  }

  const toggleAll = () => {
    // If any pages are not selected, select all. Otherwise, unselect all.
    const allSelected = selectedPages.size === subdomains.length
    setSelectedPages(allSelected ? new Set() : new Set(subdomains.map(page => page.url)))
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'crawled':
        return <CheckCircle2 className="w-4 h-4 text-green-400" />
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-400" />
      default:
        return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
    }
  }

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'crawled':
        return 'bg-green-500/10 text-green-400 border-green-500/20'
      case 'error':
        return 'bg-red-500/10 text-red-400 border-red-500/20'
      default:
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    }
  }

  return (
    <div className="space-y-4 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex justify-between items-center bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700">
        <div className="flex items-center gap-3">
          <Globe className="w-5 h-5 text-purple-400" />
          <h2 className="text-xl font-semibold text-purple-400">Discovered Pages</h2>
          <div className="flex items-center gap-3">
            <span className="px-2 py-1 rounded-lg bg-purple-500/10 text-purple-400 text-sm">
              {subdomains.length} pages
            </span>
            {selectedPages.size > 0 && selectedPages.size !== subdomains.length && (
              <span className="px-2 py-1 rounded-lg bg-purple-600/20 text-purple-400 text-sm font-medium">
                {selectedPages.size} selected
              </span>
            )}
          </div>
        </div>
        <Button
          onClick={() => onCrawlSelected(Array.from(selectedPages))}
          disabled={isProcessing || selectedPages.size === 0}
          className={`
            flex items-center gap-2 transition-all duration-300
            ${isProcessing ? 'bg-purple-500/50' : 'bg-purple-500 hover:bg-purple-600'}
          `}
        >
          {isProcessing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Processing...</span>
            </>
          ) : (
            <>
              <Globe className="w-4 h-4" />
              <span>
                {selectedPages.size === subdomains.length
                  ? `Crawl All Pages (${subdomains.length})`
                  : `Crawl Selected (${selectedPages.size}/${subdomains.length})`}
              </span>
            </>
          )}
        </Button>
      </div>
      
      {/* Table Container */}
      <div className="rounded-xl border border-gray-700 overflow-hidden bg-gray-900/50 backdrop-blur-sm">
        <ScrollArea className="h-[400px]">
          <table className="w-full">
            <thead className="bg-gray-800/50 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left text-gray-400 font-medium w-[140px]">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      className="rounded border-gray-600 bg-gray-800 text-purple-500 focus:ring-purple-500"
                      checked={selectedPages.size === subdomains.length && subdomains.length > 0}
                      onChange={toggleAll}
                    />
                    <span className="text-sm">
                      {selectedPages.size === subdomains.length
                        ? "Unselect All"
                        : "Select All"}
                    </span>
                  </div>
                </th>
                <th className="px-4 py-3 text-left text-gray-400 font-medium">URL</th>
                <th className="px-4 py-3 text-left text-gray-400 font-medium">Title</th>
                <th className="px-4 py-3 text-left text-gray-400 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700/50">
              {subdomains.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-gray-400">
                    <div className="flex flex-col items-center gap-2">
                      <Globe className="w-8 h-8 text-gray-500" />
                      <p>No pages discovered yet. Enter a URL to start.</p>
                    </div>
                  </td>
                </tr>
              ) : (
                subdomains.map((page, index) => (
                  <tr
                    key={page.url}
                    className="transition-colors hover:bg-gray-800/30"
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          className="rounded border-gray-600 bg-gray-800 text-purple-500 focus:ring-purple-500"
                          checked={selectedPages.has(page.url)}
                          onChange={() => togglePage(page.url)}
                          aria-label={`Select ${page.title || 'page'}`}
                        />
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <LinkIcon className="w-4 h-4 text-gray-500" />
                        <span className="font-mono text-sm text-gray-300 truncate max-w-[300px]">
                          {page.url}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {page.title || 'Untitled'}
                    </td>
                    <td className="px-4 py-3">
                      <div className={`
                        inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium
                        border ${getStatusStyle(page.status)}
                      `}>
                        {getStatusIcon(page.status)}
                        <span>{page.status}</span>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </ScrollArea>
      </div>
    </div>
  )
}