import LinkifyIt from "linkify-it";
import tlds from "tlds";
import UserAgent from "user-agents";
import * as cheerio from "cheerio";

const linkifyParser = new LinkifyIt().tlds(tlds).set({ fuzzyLink: true, fuzzyIP: true, fuzzyEmail: false });

export function findUrlsInText(inputText: string): string[] {
  const matches = linkifyParser.match(inputText);
  if (!matches) return [];
  return matches.map((match) => match.url);
}

export function formatPageInfoAsXml(pageData: PageData[]): string {
  return pageData
    .map(
      (page) =>
        `<externalLink><url>${page.originalUrl}</url>` +
        `${page.finalUrl !== page.originalUrl ? `<finalUrl>${page.finalUrl}</finalUrl>` : ""}` +
        `<title>${page.pageTitle}</title><description>${page.pageDescription}</description>` +
        `<bodySnippet>${page.bodySnippet}</bodySnippet></externalLink>`,
    )
    .join("");
}

export interface PageData {
  originalUrl: string;
  finalUrl: string;
  pageTitle: string;
  pageDescription: string;
  bodySnippet: string;
}

export async function fetchPageData(requestUrl: string): Promise<PageData | { error: string }> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const userAgent = new UserAgent();
    const pageResponse = await fetch(requestUrl, {
      headers: { "User-Agent": userAgent.toString() },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!pageResponse.ok) {
      return {
        error: `HTTP error: ${pageResponse.status} ${pageResponse.statusText}`,
      };
    }

    const contentType = pageResponse.headers.get("content-type") || "";
    if (!contentType.includes("text/html")) {
      return {
        error: "Response is not HTML",
      };
    }

    const pageHtml = await pageResponse.text();

    const $ = cheerio.load(pageHtml);
    $("script, style").remove();
    const pageTitle = $("title").first().text().trim();
    const pageDescription = $('meta[name="description"]').attr("content") || "";
    const bodySnippet = $("body").first().text().replace(/\s+/g, " ").slice(0, 480).trim();

    return {
      originalUrl: requestUrl,
      finalUrl: pageResponse.url,
      pageTitle,
      pageDescription,
      bodySnippet,
    };
  } catch (e) {
    return {
      error: e instanceof Error ? e.message : "An error occurred",
    };
  }
}
