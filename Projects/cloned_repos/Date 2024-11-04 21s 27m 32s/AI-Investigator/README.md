# AI Enterprise Case Study Analyzer

An automated system that analyzes AI case studies (you can change the use case by updating the instructions.md file and prompts) to identify and document enterprise-level AI implementations using Claude 3.5 Sonnet API.

It starts by reading URLs from a CSV file and uses web scraping (either through WebLoader or Firecrawl) to extract the content from each case study. 

The extracted content is then sent to Claude 3.5 Sonnet, which analyzes whether the case study represents a genuine enterprise AI implementation based on specific criteria like company maturity, implementation scale, and measurable business outcomes. 

For each URL, the system first saves the raw content and then performs this initial qualification analysis.

If Claude determines that a case study qualifies as an enterprise AI implementation, the system proceeds to generate a detailed analysis. 

It creates three types of reports:
- an individual case study report with sections like Executive Summary, AI Strategy Analysis, and Business Impact Assessment
- a cross-case analysis that identifies patterns and trends across multiple case studies
- and an executive dashboard summarizing key metrics and insights. 

All of these reports are saved in structured formats (markdown for individual reports, JSON for cross-case analysis and dashboard) in their respective directories. 

If a case study doesn't qualify as an enterprise AI implementation, the system logs the reason and moves on to the next URL. 

The entire process is asynchronous and provides detailed terminal feedback about its progress and decisions.



## Overview

<img width="1022" alt="Screenshot 2024-11-03 at 11 23 05 PM" src="https://github.com/user-attachments/assets/db757ca6-f389-4f01-a812-7533a9e3145c">
<img width="1140" alt="Screenshot 2024-11-03 at 11 22 59 PM" src="https://github.com/user-attachments/assets/679b61cc-62ac-4d1d-b6d1-69aad9ef08ec">
<img width="820" alt="Screenshot 2024-11-03 at 11 22 16 PM" src="https://github.com/user-attachments/assets/c23f6f73-10dc-4a41-978f-eebabc74fe4a">
<img width="810" alt="Screenshot 2024-11-03 at 11 22 06 PM" src="https://github.com/user-attachments/assets/435629f1-72cf-4300-8e64-98d88a7a4b60">

1. Extracts content from AI case study URLs
2. Analyzes them to identify enterprise AI implementations
3. Generates detailed reports and insights
4. Creates cross-case analysis and executive dashboards

## Features

### Content Extraction
- Web scraping with BeautifulSoup
- Structured data extraction
- Automatic content cleaning and organization
- Support for various page layouts

### AI Analysis
- Enterprise AI qualification check
- Confidence scoring
- Detailed multi-section analysis
- Business impact assessment

### Report Generation
1. Individual Case Study Reports
   - Executive Summary
   - AI Strategy Analysis
   - Technical Implementation Details
   - Business Impact Assessment
   - Key Success Factors
   - Lessons Learned

2. Cross-Case Analysis
   - Common patterns
   - Success factors
   - Implementation challenges
   - Technology trends
   - ROI patterns

3. Executive Dashboard
   - Company profiles
   - Technology stacks
   - Success metrics
   - Implementation scales

## Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/ai-case-study-analyzer.git
cd ai-case-study-analyzer

2. Create a virtual environment:
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
bash
pip install -r requirements.txt

4. Create a .env file with your API keys:
env
ANTHROPIC_API_KEY=your_claude_api_key

## Project Structure
project/
├── src/
│ ├── scrapers/
│ │ └── web_loader.py
│ ├── processors/
│ │ └── claude_processor.py
│ ├── config.py
│ ├── main.py
│ └── test_setup.py
├── input/
│ └── urls.csv
├── raw_content/
│ └── case_[id]/
│ ├── raw_content.txt
│ ├── structured_content.json
│ └── metadata.json
├── sections/
│ └── case_[id]/
│ ├── company_context.md
│ ├── business_challenge.md
│ └── [...].md
├── reports/
│ ├── individual/
│ │ └── case_[id].md
│ ├── cross_case_analysis/
│ │ └── cross_case_analysis.json
│ └── executive_dashboard/
│ └── executive_dashboard.json
└── logs/
├── processing_log.json
└── validation_log.json


## Usage

1. Prepare Input:
   - Place your case study URLs in `input/urls.csv`
   - Format: single column with header 'url'

2. Run Tests:
bash
python -m src.test_setup

3. Run Analysis:
bash
python -m src.main


## Analysis Workflow

1. Content Extraction
   - Web scraping of case study URLs
   - Content cleaning and structuring
   - Metadata extraction

2. AI Analysis
   - Enterprise AI qualification check
   - Detailed section analysis
   - Report generation

3. Report Generation
   - Individual case study reports
   - Cross-case analysis
   - Executive dashboard updates

## Analysis Criteria

### Enterprise AI Qualification
- Established company (not startup)
- Business AI implementation
- Enterprise-scale deployment
- Clear business outcomes

### Content Requirements
- AI/ML technology details
- Enterprise integration aspects
- Business process transformation
- ROI metrics
- Change management approach

## Output Files

### Individual Reports (reports/individual/)
- Executive Summary
- AI Strategy Analysis
- Technical Implementation Details
- Business Impact Assessment
- Key Success Factors
- Lessons Learned

### Cross-Case Analysis (reports/cross_case_analysis/)
- Common patterns
- Success factors
- Implementation challenges
- Technology trends
- ROI patterns

### Executive Dashboard (reports/executive_dashboard/)
- Company profiles
- Technology stacks
- Success metrics
- Implementation scales

## Error Handling

The system includes robust error handling for:
- Web scraping failures
- API timeouts
- Content parsing errors
- File system operations
- JSON parsing issues

## Logging

Detailed logging is provided in:
- processing_log.json: Processing status and errors
- validation_log.json: Content validation results

## Configuration

Key settings in config.py:
- API configurations
- Model parameters
- File paths
- Processing options

## Dependencies

- anthropic: Claude API client
- beautifulsoup4: Web scraping
- aiohttp: Async HTTP requests
- pandas: Data processing
- python-dotenv: Environment variables

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Claude 3.5 Sonnet by Anthropic for AI analysis
- BeautifulSoup4 for web scraping
- The open-source community for various tools and libraries

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
