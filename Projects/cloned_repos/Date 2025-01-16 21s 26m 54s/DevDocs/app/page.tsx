'use client'

import { useState, useEffect } from 'react'
import UrlInput from '@/components/UrlInput'
import ProcessingBlock from '@/components/ProcessingBlock'
import SubdomainList from '@/components/SubdomainList'
import MarkdownOutput from '@/components/MarkdownOutput'
import StoredFiles from '@/components/StoredFiles'
import { discoverSubdomains, crawlPages, validateUrl, formatBytes } from '@/lib/crawl-service'
import { saveMarkdown, loadMarkdown } from '@/lib/storage'
import { useToast } from "@/components/ui/use-toast"
import { DiscoveredPage } from '@/lib/types'

export default function Home() {
  const [url, setUrl] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [discoveredPages, setDiscoveredPages] = useState<DiscoveredPage[]>([])
  const [isCrawling, setIsCrawling] = useState(false)
  const [markdown, setMarkdown] = useState('')
  const [stats, setStats] = useState({
    subdomainsParsed: 0,
    pagesCrawled: 0,
    dataExtracted: '0 KB',
    errorsEncountered: 0
  })
  const { toast } = useToast()

  const handleSubmit = async (submittedUrl: string) => {
    if (!validateUrl(submittedUrl)) {
      toast({
        title: "Invalid URL",
        description: "Please enter a valid URL including the protocol (http:// or https://)",
        variant: "destructive"
      })
      return
    }

    setUrl(submittedUrl)
    setIsProcessing(true)
    setMarkdown('')
    setDiscoveredPages([])
    
    try {
      console.log('Discovering pages for:', submittedUrl)
      const pages = await discoverSubdomains(submittedUrl)
      console.log('Discovered pages:', pages)
      
      setDiscoveredPages(pages)
      setStats(prev => ({
        ...prev,
        subdomainsParsed: pages.length
      }))
      
      toast({
        title: "Pages Discovered",
        description: `Found ${pages.length} related pages`
      })
    } catch (error) {
      console.error('Error discovering pages:', error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to discover pages",
        variant: "destructive"
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const handleCrawlSelected = async (selectedUrls: string[]) => {
    setIsCrawling(true)
    try {
      // Filter pages to only selected ones
      const selectedPages = discoveredPages.filter(page => selectedUrls.includes(page.url))
      console.log('Starting crawl for selected pages:', selectedPages)
      
      const result = await crawlPages(selectedPages)
      console.log('Crawl result:', result)
      
      if (result.error) {
        throw new Error(result.error)
      }
      
      try {
        // Save markdown to storage
        await saveMarkdown(url, result.markdown)
        console.log('Saved content for:', url)
        
        setMarkdown(result.markdown)
        setStats(prev => ({
          ...prev,
          pagesCrawled: selectedPages.length,
          dataExtracted: formatBytes(result.markdown.length)
        }))

        toast({
          title: "Content Saved",
          description: `Crawled content has been saved and can be loaded again later`
        })
      } catch (error) {
        console.error('Error saving content:', error)
        toast({
          title: "Error",
          description: "Failed to save content for later use",
          variant: "destructive"
        })
      }
      
      // Update only selected page statuses to crawled
      setDiscoveredPages(pages =>
        pages.map(page => ({
          ...page,
          status: selectedUrls.includes(page.url) ? 'crawled' : page.status
        }))
      )
      
      toast({
        title: "Crawling Complete",
        description: "All pages have been crawled and processed"
      })
    } catch (error) {
      console.error('Error crawling pages:', error)
      setStats(prev => ({
        ...prev,
        errorsEncountered: prev.errorsEncountered + 1
      }))
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to crawl pages",
        variant: "destructive"
      })
    } finally {
      setIsCrawling(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900">
      {/* Hero Header */}
      <header className="w-full py-12 bg-gradient-to-r from-gray-900/50 to-gray-800/50 backdrop-blur-sm border-b border-gray-700">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500 mb-4">
            DevDocs
          </h1>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Discover and extract documentation for quicker development
          </p>
          <p className="text-sm text-gray-400 mt-2">by CyberAGI Inc</p>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Input and Processing */}
          <div className="space-y-8">
            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
              <h2 className="text-2xl font-semibold mb-6 text-blue-400">Start Exploration</h2>
              <UrlInput onSubmit={handleSubmit} />
            </section>

            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
              <h2 className="text-2xl font-semibold mb-6 text-purple-400">Processing Status</h2>
              <ProcessingBlock
                isProcessing={isProcessing || isCrawling}
                stats={stats}
              />
            </section>

            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
              <h2 className="text-2xl font-semibold mb-6 text-blue-400">Stored Files</h2>
              <StoredFiles />
            </section>
          </div>

          {/* Right Column - Results */}
          <div className="space-y-8">
            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
              <h2 className="text-2xl font-semibold mb-6 text-green-400">Discovered Pages</h2>
              <SubdomainList
                subdomains={discoveredPages}
                onCrawlSelected={handleCrawlSelected}
                isProcessing={isCrawling}
              />
            </section>

            <section className="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-6 border border-gray-700 shadow-xl">
              <h2 className="text-2xl font-semibold mb-6 text-yellow-400">Extracted Content</h2>
              <MarkdownOutput
                markdown={markdown}
                isVisible={markdown !== ''}
              />
            </section>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16">
          <div className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-gray-700 shadow-lg">
            <h3 className="text-xl font-semibold text-blue-400 mb-3">Advanced Extraction</h3>
            <p className="text-gray-300">
              Smart content targeting with specific selectors and quality filtering
            </p>
          </div>
          <div className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-gray-700 shadow-lg">
            <h3 className="text-xl font-semibold text-purple-400 mb-3">Bypass Restrictions</h3>
            <p className="text-gray-300">
              Handle SSL certificates and security policies with ease
            </p>
          </div>
          <div className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-gray-700 shadow-lg">
            <h3 className="text-xl font-semibold text-green-400 mb-3">Error Recovery</h3>
            <p className="text-gray-300">
              Automatic retries and enhanced session management
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
