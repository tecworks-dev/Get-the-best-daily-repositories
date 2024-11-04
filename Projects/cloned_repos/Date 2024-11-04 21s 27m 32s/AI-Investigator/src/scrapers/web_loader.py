import aiohttp
import logging
from bs4 import BeautifulSoup
from typing import Optional, Dict
from pathlib import Path
import asyncio
import json
from src.config import RAW_DIR, MAX_RETRIES

logger = logging.getLogger(__name__)

class WebLoader:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def extract_case_study(self, url: str) -> Optional[Dict]:
        """Extract and process case study content"""
        logger.info(f"Starting extraction for {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch {url} - Status: {response.status}")
                        return None
                        
                    html_content = await response.text()
                    
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unnecessary elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Untitled Case Study"
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if not main_content:
                logger.error(f"No main content found for {url}")
                return None
            
            # Clean and structure the content
            text_content = main_content.get_text(separator='\n', strip=True)
            
            # Extract structured data
            structured_data = {
                "title": title_text,
                "main_content": text_content,
                "company_name": self._extract_company_name(soup),
                "industry": self._extract_industry(soup),
                "implementation_details": self._extract_implementation(soup),
                "results": self._extract_results(soup)
            }
            
            return {
                "url": url,
                "title": title_text,
                "content": text_content,
                "structured_data": structured_data,
                "metadata": {
                    "extraction_time": str(asyncio.get_event_loop().time()),
                    "content_length": len(text_content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name from the page"""
        # Try different methods to find company name
        company = soup.find('meta', property='og:site_name')
        if company:
            return company['content']
            
        company = soup.find('h1')
        if company:
            text = company.get_text(strip=True)
            # Extract company name from title (usually at the start)
            return text.split('|')[0].split('-')[0].strip()
            
        return "Unknown Company"
    
    def _extract_industry(self, soup: BeautifulSoup) -> str:
        """Extract industry information"""
        # Look for industry-related keywords
        industry_keywords = ['industry', 'sector', 'vertical']
        for keyword in industry_keywords:
            element = soup.find(text=lambda text: text and keyword.lower() in text.lower())
            if element:
                return element.parent.get_text(strip=True)
        return "Industry not specified"
    
    def _extract_implementation(self, soup: BeautifulSoup) -> str:
        """Extract implementation details"""
        # Look for implementation-related sections
        implementation_keywords = ['implementation', 'solution', 'approach', 'methodology']
        for keyword in implementation_keywords:
            section = soup.find(['h2', 'h3', 'h4'], text=lambda text: text and keyword.lower() in text.lower())
            if section:
                content = []
                for elem in section.find_next_siblings():
                    if elem.name in ['h2', 'h3', 'h4']:
                        break
                    content.append(elem.get_text(strip=True))
                return '\n'.join(content)
        return "Implementation details not found"
    
    def _extract_results(self, soup: BeautifulSoup) -> str:
        """Extract results and outcomes"""
        # Look for results-related sections
        results_keywords = ['results', 'outcomes', 'impact', 'benefits']
        for keyword in results_keywords:
            section = soup.find(['h2', 'h3', 'h4'], text=lambda text: text and keyword.lower() in text.lower())
            if section:
                content = []
                for elem in section.find_next_siblings():
                    if elem.name in ['h2', 'h3', 'h4']:
                        break
                    content.append(elem.get_text(strip=True))
                return '\n'.join(content)
        return "Results not found"
            
    async def save_raw_content(self, case_id: int, content: Dict) -> bool:
        """Save raw case study content to file"""
        try:
            case_dir = Path(RAW_DIR) / f"case_{case_id}"
            case_dir.mkdir(exist_ok=True)
            
            # Save structured content
            with open(case_dir / "structured_content.json", "w", encoding="utf-8") as f:
                json.dump(content["structured_data"], f, indent=2)
            
            # Save raw content
            with open(case_dir / "raw_content.txt", "w", encoding="utf-8") as f:
                f.write(f"Title: {content['title']}\n")
                f.write(f"URL: {content['url']}\n")
                f.write("\nContent:\n")
                f.write(content['content'])
            
            # Save metadata
            with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(content["metadata"], f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving content for case {case_id}: {str(e)}")
            return False
