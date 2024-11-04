import pandas as pd
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
import json
from src.config import (
    INPUT_DIR, 
    RAW_DIR, 
    LOGS_DIR,
    LOG_FORMAT,
    SECTIONS_DIR,
    REPORTS_DIR
)
from src.scrapers.web_loader import WebLoader
from src.processors.claude_processor import ClaudeProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(Path(LOGS_DIR) / "processing_log.json"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def load_case_studies():
    """Load and validate the input CSV file"""
    try:
        # Try both possible filenames
        csv_path = Path(INPUT_DIR) / "ai case studies - Sheet1.csv"
        if not csv_path.exists():
            csv_path = Path(INPUT_DIR) / "urls.csv"
            
        if not csv_path.exists():
            raise FileNotFoundError(f"No CSV file found in {INPUT_DIR}")
            
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get the first column name
        first_column = df.columns[0]
        
        # Rename the column to 'url' if it's not already named that
        if first_column != 'url':
            df = df.rename(columns={first_column: 'url'})
        
        logger.info(f"Loaded {len(df)} case studies")
        logger.info(f"URLs sample: {df['url'].head().tolist()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading case studies: {str(e)}")
        return None

async def process_case_study(web_loader: WebLoader, claude_processor: ClaudeProcessor, url: str, index: int):
    """Process a single case study"""
    print(f"\n{'='*80}")
    print(f"Processing Case Study #{index + 1}")
    print(f"URL: {url}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Extract content using web scraper
        print("\n📥 Extracting content...")
        content = await web_loader.extract_case_study(url)
        
        if not content:
            print("❌ Failed to extract content")
            return
            
        # Save raw content for reference
        print("💾 Saving raw content...")
        await web_loader.save_raw_content(index, content)
        print("✅ Raw content saved")
        
        # Step 2: Analyze if it's an enterprise AI case study
        print("\n🔍 Analyzing if this is an enterprise AI case study...")
        analysis = await claude_processor.analyze_enterprise_relevance(content['content'])
        
        if analysis.get('is_enterprise_ai'):
            print("\n✅ Qualified as Enterprise AI Case Study")
            print(f"Company: {analysis.get('company_details', {}).get('name')}")
            print(f"Industry: {analysis.get('company_details', {}).get('industry')}")
            print(f"AI Technologies: {', '.join(analysis.get('ai_implementation', {}).get('technologies', []))}")
            print(f"Confidence Score: {analysis.get('confidence_score', 0.0)}")
            
            # Step 3: Generate executive report for qualified cases
            print("\n📝 Generating executive report...")
            executive_report = await claude_processor.generate_executive_report(
                content['content'], 
                analysis
            )
            
            if executive_report:
                print("✅ Executive report generated")
                
                # Save all reports
                print("\n💾 Saving reports...")
                if await claude_processor.save_reports(index, content, analysis, executive_report):
                    print("✅ All reports saved successfully")
                    print(f"- Individual report: reports/individual/case_{index}.md")
                    print(f"- Updated cross-case analysis: reports/cross_case_analysis/cross_case_analysis.json")
                    print(f"- Updated executive dashboard: reports/executive_dashboard/executive_dashboard.json")
                else:
                    print("❌ Failed to save some reports")
            else:
                print("❌ Failed to generate executive report")
        
        else:
            print("\n⚠️ Not an Enterprise AI Case Study")
            print(f"Reason: {analysis.get('disqualification_reason')}")
            print("Skipping detailed analysis...")
        
        # Wait before next case study
        print("\n⏳ Waiting before next case study...")
        await asyncio.sleep(3)
            
    except Exception as e:
        logger.error(f"Error processing case study #{index + 1}: {str(e)}")
        print(f"❌ Error: {str(e)}")

async def main():
    """Main execution function"""
    print("\n🚀 Starting AI Enterprise Case Study Analyzer")
    
    # Load case studies
    df = await load_case_studies()
    if df is None:
        print("❌ Failed to load case studies. Exiting.")
        return
    
    # Initialize processors
    web_loader = WebLoader()
    claude_processor = ClaudeProcessor()
    
    # Process one case study at a time
    total_cases = len(df)
    for index, row in df.iterrows():
        print(f"\n📌 Processing case study {index + 1} of {total_cases}")
        url = row['url'].strip()
        if url and isinstance(url, str):
            await process_case_study(web_loader, claude_processor, url, index)
    
    print("\n✨ Processing complete!")
    print(f"Results can be found in:")
    print(f"- Raw content: {RAW_DIR}")
    print(f"- Detailed analyses: {SECTIONS_DIR}")
    print(f"- Reports: {REPORTS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
