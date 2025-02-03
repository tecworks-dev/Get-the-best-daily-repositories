'use client';

import { cn } from '@/lib/utils';
import { ExternalLinkIcon } from './icons';
import { Globe } from 'lucide-react';

interface SearchResult {
  title: string;
  url: string;
  description?: string;
  source?: string;
}

interface SearchResultsProps {
  results: SearchResult[];
  title?: string;
}

export function SearchResults({
  results,
  title = 'Search Results...',
}: SearchResultsProps) {
  if (!results.length) return null;

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-2">
        {/* <div className="size-4 flex items-center rounded-full justify-center ring-1 shrink-0 ring-border bg-background">
          <div className="size-2 rounded-full bg-foreground" />
        </div> */}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className="grid gap-2">
        {results.map((result, i) => (
          <a
            key={i}
            href={result.url}
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'flex flex-col gap-1 p-3 rounded-lg bg-muted/40 hover:bg-muted transition-colors',
              'group cursor-pointer',
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                {result.source && (
                  <div className="flex items-center justify-center size-5 shrink-0 rounded-sm bg-background ring-1 ring-border text-[10px] font-medium uppercase">
                    <Globe size={12} />
                  </div>
                )}
                <span className="text-sm font-medium line-clamp-1">
                  {result.title}
                </span>
              </div>
              <ExternalLinkIcon
                size={14}
                className="shrink-0 text-muted-foreground group-hover:text-foreground transition-colors"
              />
            </div>
            {result.description && (
              <p className="text-sm text-muted-foreground line-clamp-2">
                {result.description}
              </p>
            )}
          </a>
        ))}
      </div>
    </div>
  );
}
