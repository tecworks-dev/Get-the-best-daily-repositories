'use client';

import { cn } from '@/lib/utils';
import { ExternalLinkIcon } from './icons';
import { ChevronRight, FileText, Loader2 } from 'lucide-react';
import { useState } from 'react';

interface ScrapeResultsProps {
  url: string;
  data: string;
  title?: string;
  isLoading?: boolean;
}

export function ScrapeResults({
  url,
  data,
  title = 'Scraped Content...',
  isLoading = false,
}: ScrapeResultsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsOpen(!isOpen);
  };

  if (isLoading) {
    return (
      <div className="w-full">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm font-medium">
            Using Firecrawl to scrape content...
          </span>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 size={16} className="animate-spin" />
          <span>Scraping content...</span>
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className="grid gap-3">
        <div
          className={cn('flex flex-col rounded-lg bg-muted/40 overflow-hidden')}
        >
          <button
            onClick={handleToggle}
            className="flex items-center justify-between w-full p-4 hover:bg-muted/60 transition-colors"
          >
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center size-5 shrink-0 rounded-sm bg-background ring-1 ring-border text-[10px] font-medium">
                <FileText size={12} />
              </div>
              <span className="text-sm font-medium hover:underline flex items-center gap-1">
                {new URL(url).hostname}
                <ExternalLinkIcon size={12} className="text-muted-foreground" />
              </span>
            </div>
            <ChevronRight
              size={16}
              className={cn(
                'text-muted-foreground transition-transform',
                isOpen && 'rotate-90',
              )}
            />
          </button>
          <div
            className={cn(
              'grid transition-all',
              isOpen
                ? 'grid-rows-[1fr] opacity-100'
                : 'grid-rows-[0fr] opacity-0',
            )}
          >
            <div className="overflow-hidden">
              <div className="p-4 pt-0 grid gap-1.5 min-w-0">
                <div className="text-sm whitespace-pre-wrap break-words bg-muted/50 rounded-md p-2">
                  {data}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
