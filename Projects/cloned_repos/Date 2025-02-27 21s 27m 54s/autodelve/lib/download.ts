import * as cheerio from 'cheerio';
import path from 'path';
import { URL } from 'url';
import { convertHTMLToMarkdown } from './convertHTML';

function downloadWebsite(url: string, maxDepth: number = 3): Promise<Map<string, string>> {
  const visited = new Map<string, string>();
  
  async function crawl(currentUrl: string, depth: number): Promise<void> {
    if (depth > maxDepth || visited.has(currentUrl)) {
      return;
    }
    
    try {
      const response = await fetch(currentUrl);
      const html = await response.text();
      
      // Store the HTML content
      visited.set(currentUrl, html);
      
      // If we've reached max depth, don't extract more links
      if (depth === maxDepth) {
        return;
      }
      
      // Parse HTML and extract links
      const $ = cheerio.load(html);
      const links = new Set<string>();
      
      $('a').each((index: number, element) => {
        const href = $(element).attr('href');
        if (href) {
          // Resolve relative URLs
          const resolvedUrl = new URL(href, currentUrl).toString();
          
          // Only follow links from the same base URL
          const baseUrl = new URL(url).hostname;
          const resolvedUrlObj = new URL(resolvedUrl);
          
          if (resolvedUrlObj.hostname === baseUrl) {
            links.add(resolvedUrl);
          }
        }
      });
      
      // Recursively crawl all extracted links
      for (const link of links) {
        await crawl(link, depth + 1);
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        console.error(`Error crawling ${currentUrl}:`, error.message);
      } else {
        console.error(`Unknown error crawling ${currentUrl}`);
      }
    }
  }
  
  // Start crawling from the initial URL
  return crawl(url, 1).then(() => visited);
}


export async function download(url: string) {
  const websiteContent = await downloadWebsite(url);
  
  // Convert the Map to an array of objects with url and content fields
  const contentArray = Array.from(websiteContent.entries()).map(([pageUrl, html]) => {
    return {
      url: pageUrl,
      content: convertHTMLToMarkdown(html) // Convert HTML to Markdown
    };
  });
  
  // Wait for all conversions to complete
  const resolvedContentArray = await Promise.all(
    contentArray.map(async (item) => {
      return {
        url: item.url,
        content: await item.content
      };
    })
  );
  
  // Save all content to a single JSON file
  const filePath = path.join('content', 'website_content.json');
  await Bun.write(filePath, JSON.stringify(resolvedContentArray, null, 2));
  
  console.log(`Website content saved to ${filePath}`);
}

export async function readMarkdownFiles(): Promise<Array<{ url: string, content: string }>> {
  const filePath = path.join('content', 'website_content.json');
  
  try {
    const fileContent = await Bun.file(filePath).text();
    const contentArray = JSON.parse(fileContent);
    
    return contentArray;
  } catch (error) {
    console.error('Failed to read website content:', error);
    return [];
  }
}