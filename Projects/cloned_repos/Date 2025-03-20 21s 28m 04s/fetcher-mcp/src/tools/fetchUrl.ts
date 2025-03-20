import { chromium } from "playwright";
import { WebContentProcessor } from "../services/webContentProcessor.js";
import { FetchOptions } from "../types/index.js";

// Parse command line arguments, check for debug flag
const isDebugMode = process.argv.includes("--debug");

/**
 * Tool definition for fetch_url
 */
export const fetchUrlTool = {
  name: "fetch_url",
  description: "Retrieve web page content from a specified URL",
  inputSchema: {
    type: "object",
    properties: {
      url: {
        type: "string",
        description: "URL to fetch",
      },
      timeout: {
        type: "number",
        description:
          "Page loading timeout in milliseconds, default is 30000 (30 seconds)",
      },
      waitUntil: {
        type: "string",
        description:
          "Specifies when navigation is considered complete, options: 'load', 'domcontentloaded', 'networkidle', 'commit', default is 'load'",
      },
      extractContent: {
        type: "boolean",
        description:
          "Whether to intelligently extract the main content, default is true",
      },
      maxLength: {
        type: "number",
        description:
          "Maximum length of returned content (in characters), default is no limit",
      },
      returnHtml: {
        type: "boolean",
        description:
          "Whether to return HTML content instead of Markdown, default is false",
      },
    },
    required: ["url"],
  }
};

/**
 * Implementation of the fetch_url tool
 */
export async function fetchUrl(args: any) {
  const url = String(args?.url || "");
  if (!url) {
    console.error(`[Error] URL parameter missing`);
    throw new Error("URL parameter is required");
  }

  const options: FetchOptions = {
    timeout: Number(args?.timeout) || 30000,
    waitUntil: String(args?.waitUntil || "load") as 'load' | 'domcontentloaded' | 'networkidle' | 'commit',
    extractContent: args?.extractContent !== false,
    maxLength: Number(args?.maxLength) || 0,
    returnHtml: args?.returnHtml === true
  };

  const processor = new WebContentProcessor(options, '[FetchURL]');
  let browser = null;
  let page = null;

  try {
    browser = await chromium.launch({ headless: !isDebugMode });
    const context = await browser.newContext({
      javaScriptEnabled: true,
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    });

    await context.route('**/*', async (route) => {
      const resourceType = route.request().resourceType();
      if (['image', 'stylesheet', 'font', 'media'].includes(resourceType)) {
        await route.abort();
      } else {
        await route.continue();
      }
    });

    page = await context.newPage();
    
    const result = await processor.processPageContent(page, url);
    
    return {
      content: [{ type: "text", text: result.content }]
    };
  } finally {
    if (page) await page.close().catch(e => console.error(`[Error] Failed to close page: ${e.message}`));
    if (browser) await browser.close().catch(e => console.error(`[Error] Failed to close browser: ${e.message}`));
  }
}