from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from .crawler import discover_pages, crawl_pages, DiscoveredPage, CrawlResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crawl4AI Backend")

# Configure CORS to allow requests from our frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiscoverRequest(BaseModel):
    url: str

class CrawlRequest(BaseModel):
    pages: List[DiscoveredPage]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/api/discover")
async def discover_endpoint(request: DiscoverRequest):
    """Discover pages related to the provided URL"""
    try:
        logger.info(f"Received discover request for URL: {request.url}")
        pages = await discover_pages(request.url)
        
        # Log the results
        if pages:
            logger.info(f"Successfully discovered {len(pages)} pages")
            for page in pages:
                logger.debug(f"Discovered page: {page.url} ({page.status})")
        else:
            logger.warning("No pages discovered")
            
        # Always return a valid response, even if no pages were found
        return {
            "pages": pages or [],  # Ensure we always return an array
            "message": f"Found {len(pages)} pages" if pages else "No pages discovered",
            "success": True
        }
    except Exception as e:
        logger.error(f"Error discovering pages: {str(e)}", exc_info=True)
        # Return a structured error response
        return {
            "pages": [],
            "message": f"Error discovering pages: {str(e)}",
            "success": False,
            "error": str(e)
        }

@app.post("/api/crawl")
async def crawl_endpoint(request: CrawlRequest):
    """Crawl the provided pages and generate markdown content"""
    try:
        logger.info(f"Received crawl request for {len(request.pages)} pages")
        result = await crawl_pages(request.pages)
        
        # Log the results
        logger.info(f"Successfully crawled pages. Stats: {result.stats}")
        return {
            "markdown": result.markdown,
            "stats": result.stats.dict(),
            "success": True
        }
    except Exception as e:
        logger.error(f"Error crawling pages: {str(e)}", exc_info=True)
        # Return a structured error response
        return {
            "markdown": "",
            "stats": {
                "subdomains_parsed": 0,
                "pages_crawled": 0,
                "data_extracted": "0 KB",
                "errors_encountered": 1
            },
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=24125,
        reload=True,
        log_level="info"
    )