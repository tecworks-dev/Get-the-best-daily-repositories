from typing import List, Optional
import logging
from datetime import datetime
from pydantic import BaseModel
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Configure logging
logger = logging.getLogger(__name__)

class DiscoveredPage(BaseModel):
    url: str
    title: Optional[str] = None
    status: str = "pending"

class CrawlStats(BaseModel):
    subdomains_parsed: int = 0
    pages_crawled: int = 0
    data_extracted: str = "0 KB"
    errors_encountered: int = 0

class CrawlResult(BaseModel):
    markdown: str
    stats: CrawlStats

def get_browser_config() -> BrowserConfig:
    """Get browser configuration that launches a local instance"""
    return BrowserConfig(
        browser_type="chromium",
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        verbose=True,
        text_mode=True,
        light_mode=True,
        extra_args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu"
        ]
    )

def get_crawler_config(session_id: str = None) -> CrawlerRunConfig:
    """Get crawler configuration for content extraction"""
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.2,  # Lower threshold to keep more content
            threshold_type="dynamic",
            min_word_threshold=5  # Lower word threshold
        ),
        options={
            "body_width": 80,
            "ignore_images": True,
            "escape_html": True
        }
    )
    
    return CrawlerRunConfig(
        word_count_threshold=5,  # Lower threshold to match content filter
        cache_mode=CacheMode.ENABLED,
        verbose=True,
        wait_until='networkidle',
        screenshot=False,
        pdf=False,
        magic=True,  # Enable magic mode for better bot detection avoidance
        scan_full_page=True,
        markdown_generator=markdown_generator  # Use custom markdown generator
    )

async def discover_pages(url: str) -> List[DiscoveredPage]:
    """
    Discover all related pages under a given URL using Crawl4AI's link analysis.
    """
    discovered_pages = []
    seen_urls = set()  # Keep track of URLs we've seen
    logger.info(f"Starting discovery for URL: {url}")
    
    try:
        browser_config = get_browser_config()
        crawler_config = get_crawler_config()
        seen_urls.add(url)  # Add the primary URL to seen set
        logger.info("Initializing crawler with browser config: %s", browser_config)
        logger.info("Using crawler config: %s", crawler_config)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            logger.info(f"Crawl completed for {url}. Success: {result.success}")
            
            # Extract title from content
            title = "Untitled Page"
            if result.markdown_v2 and result.markdown_v2.fit_markdown:
                content_lines = result.markdown_v2.fit_markdown.split('\n')
                if content_lines:
                    potential_title = content_lines[0].strip('# ').strip()
                    if potential_title:
                        title = potential_title

            primary_page = DiscoveredPage(
                url=url,
                title=title,
                status="crawled" if result.success else "pending"
            )

            # Process internal links first
            if hasattr(result, 'links') and isinstance(result.links, dict):
                internal_links = result.links.get("internal", [])
                logger.info(f"Found {len(internal_links)} internal links")
                
                for link in internal_links:
                    try:
                        href = link.get("href", "")
                        if not href:
                            logger.debug(f"Skipping empty URL")
                            continue
                        
                        # Ensure the URL is absolute
                        if not href.startswith(('http://', 'https://')):
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                            logger.debug(f"Converted to absolute URL: {href}")
                        
                        # Skip restricted paths
                        if any(excluded in href.lower() for excluded in [
                            "login", "signup", "register", "logout",
                            "account", "profile", "admin"
                        ]):
                            logger.debug(f"Skipping restricted URL: {href}")
                            continue

                        # Ensure the URL is from the same domain
                        from urllib.parse import urlparse
                        base_domain = urlparse(url).netloc
                        link_domain = urlparse(href).netloc
                        if base_domain != link_domain:
                            logger.debug(f"Skipping external URL: {href}")
                            continue

                        link_title = link.get("text", "").strip() or "Untitled Page"
                        logger.info(f"Adding discovered page: {href} ({link_title})")
                        discovered_pages.append(
                            DiscoveredPage(
                                url=href,
                                title=link_title,
                                status="pending"
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error processing link {link}: {str(e)}")
                        continue

            logger.info(f"Discovered {len(discovered_pages)} valid pages from {url}")
            return discovered_pages

    except Exception as e:
        logger.error(f"Error discovering pages: {str(e)}")
        return [DiscoveredPage(url=url, title="Main Page", status="pending")]

async def crawl_pages(pages: List[DiscoveredPage]) -> CrawlResult:
    """
    Crawl multiple pages and combine their content into a single markdown document.
    """
    all_markdown = []
    total_size = 0
    errors = 0
    
    try:
        browser_config = get_browser_config()
        crawler_config = get_crawler_config()
        logger.info("Initializing crawler with browser config: %s", browser_config)
        logger.info("Using crawler config: %s", crawler_config)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for page in pages:
                try:
                    logger.info(f"Crawling page: {page.url}")
                    result = await crawler.arun(url=page.url, config=crawler_config)
                    
                    if result and hasattr(result, 'markdown_v2') and result.markdown_v2:
                        page_markdown = f"# {page.title or 'Untitled Page'}\n"
                        page_markdown += f"URL: {page.url}\n\n"
                        
                        # Try fit_markdown first, fall back to raw_markdown
                        content = None
                        if hasattr(result.markdown_v2, 'fit_markdown') and result.markdown_v2.fit_markdown:
                            content = result.markdown_v2.fit_markdown
                            logger.info(f"Using fit_markdown for {page.url}")
                        elif hasattr(result.markdown_v2, 'raw_markdown') and result.markdown_v2.raw_markdown:
                            content = result.markdown_v2.raw_markdown
                            logger.info(f"Falling back to raw_markdown for {page.url}")
                        
                        if content:
                            # Filter out navigation instructions
                            filtered_lines = []
                            skip_next = False
                            for line in content.split('\n'):
                                if skip_next:
                                    skip_next = False
                                    continue
                                    
                                # Skip navigation instructions and their following lines
                                if 'To navigate the symbols, press' in line:
                                    skip_next = True
                                    continue
                                    
                                # Skip other navigation elements
                                if any(x in line for x in [
                                    'Skip Navigation',
                                    'Search...',
                                    '⌘K',
                                    'symbols inside <root>'
                                ]):
                                    continue
                                    
                                filtered_lines.append(line)
                            
                            filtered_content = '\n'.join(filtered_lines).strip()
                            if filtered_content:
                                page_markdown += filtered_content
                                page_markdown += "\n\n---\n\n"
                                all_markdown.append(page_markdown)
                                total_size += len(page_markdown.encode('utf-8'))
                                logger.info(f"Successfully extracted content from {page.url}")
                            else:
                                logger.warning(f"Skipping {page.url} - filtered content was empty")
                                errors += 1
                        else:
                            logger.warning(f"Skipping {page.url} - no markdown content available")
                            errors += 1
                    else:
                        logger.warning(f"Skipping {page.url} - no valid result")
                        errors += 1
                        
                except Exception as e:
                    logger.error(f"Error crawling page {page.url}: {str(e)}")
                    errors += 1

            combined_markdown = "".join(all_markdown)
            
            # Calculate human-readable size
            size_str = f"{total_size} B"
            if total_size > 1024:
                size_str = f"{total_size/1024:.2f} KB"
            if total_size > 1024*1024:
                size_str = f"{total_size/(1024*1024):.2f} MB"
            
            stats = CrawlStats(
                subdomains_parsed=len(pages),
                pages_crawled=len(pages) - errors,
                data_extracted=size_str,
                errors_encountered=errors
            )
            
            logger.info(f"Completed crawling with stats: {stats}")
            return CrawlResult(
                markdown=combined_markdown,
                stats=stats
            )
            
    except Exception as e:
        logger.error(f"Error in crawl_pages: {str(e)}")
        return CrawlResult(
            markdown="",
            stats=CrawlStats(
                subdomains_parsed=len(pages),
                pages_crawled=0,
                data_extracted="0 KB",
                errors_encountered=1
            )
        )