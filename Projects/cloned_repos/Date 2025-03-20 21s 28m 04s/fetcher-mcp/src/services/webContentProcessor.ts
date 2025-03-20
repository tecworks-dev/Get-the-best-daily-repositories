import { JSDOM } from "jsdom";
import { Readability } from "@mozilla/readability";
import TurndownService from "turndown";
import { FetchOptions, FetchResult } from "../types/index.js";

export class WebContentProcessor {
  private options: FetchOptions;
  private logPrefix: string;

  constructor(options: FetchOptions, logPrefix: string = '') {
    this.options = options;
    this.logPrefix = logPrefix;
  }

  async processPageContent(page: any, url: string): Promise<FetchResult> {
    try {
      // Set timeout
      page.setDefaultTimeout(this.options.timeout);

      // Navigate to URL
      console.error(`${this.logPrefix} Navigating to URL: ${url}`);
      await page.goto(url, {
        timeout: this.options.timeout,
        waitUntil: this.options.waitUntil,
      });

      // Get page title
      const pageTitle = await page.title();
      console.error(`${this.logPrefix} Page title: ${pageTitle}`);

      // Get HTML content
      const html = await page.content();

      if (!html) {
        console.error(`${this.logPrefix} Browser returned empty content`);
        return {
          success: false,
          content: `Title: Error\nURL: ${url}\nContent:\n\n<error>Failed to retrieve web page content: Browser returned empty content</error>`,
          error: "Browser returned empty content"
        };
      }

      console.error(`${this.logPrefix} Successfully retrieved web page content, length: ${html.length}`);

      const processedContent = await this.processContent(html, url);
      
      // Format the response
      const formattedContent = `Title: ${pageTitle}\nURL: ${url}\nContent:\n\n${processedContent}`;

      return {
        success: true,
        content: formattedContent
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      console.error(`${this.logPrefix} Error: ${errorMessage}`);
      
      return {
        success: false,
        content: `Title: Error\nURL: ${url}\nContent:\n\n<error>Failed to retrieve web page content: ${errorMessage}</error>`,
        error: errorMessage
      };
    }
  }

  private async processContent(html: string, url: string): Promise<string> {
    let contentToProcess = html;
    
    // Extract main content if needed
    if (this.options.extractContent) {
      console.error(`${this.logPrefix} Extracting main content`);
      const dom = new JSDOM(html, { url });
      const reader = new Readability(dom.window.document);
      const article = reader.parse();

      if (!article) {
        console.error(`${this.logPrefix} Could not extract main content, will use full HTML`);
      } else {
        contentToProcess = article.content;
        console.error(`${this.logPrefix} Successfully extracted main content, length: ${contentToProcess.length}`);
      }
    }

    // Convert to markdown if needed
    let processedContent = contentToProcess;
    if (!this.options.returnHtml) {
      console.error(`${this.logPrefix} Converting to Markdown`);
      const turndownService = new TurndownService();
      processedContent = turndownService.turndown(contentToProcess);
      console.error(`${this.logPrefix} Successfully converted to Markdown, length: ${processedContent.length}`);
    }

    // Truncate if needed
    if (this.options.maxLength > 0 && processedContent.length > this.options.maxLength) {
      console.error(`${this.logPrefix} Content exceeds maximum length, will truncate to ${this.options.maxLength} characters`);
      processedContent = processedContent.substring(0, this.options.maxLength);
    }

    return processedContent;
  }
}