# AI Enterprise Case Study Analyzer

Build a Python application using Claude 3.5 Sonnet API to analyze and generate structured reports on enterprise AI adoption case studies.

## INPUT REQUIREMENTS
- CSV file containing URLs of potential AI case studies
- The system should analyze and filter for enterprise-specific AI implementation cases
- Whenever it finds an AI case study that is related with enterprise AI, it should be saved for further processing.

## ANALYSIS CRITERIA
Enterprise Case Detection:
- Must involve established companies (not startups)
- Focus on business AI implementation
- Clear business outcomes and metrics
- Enterprise-scale deployment

Content Requirements:
- AI/ML technology implementation details
- Enterprise integration aspects
- Business process transformation
- ROI or business impact metrics
- Change management approach

## PROCESSING WORKFLOW

### Phase 1: Case Study Filtering & Collection
1. For each URL in CSV:
   - Scrape content using web_loader
   - Analyze with Claude to determine if it's an enterprise AI case study
   - Filter out non-enterprise or non-AI cases
   - Save qualified content for further processing

### Phase 2: Detailed Analysis
For each qualified case study:
1. Extract and analyze six key sections:
   - Company Context & AI Strategy
   - Business Challenge & Opportunity
   - AI Solution Architecture
   - Implementation & Integration
   - Change Management & Adoption
   - Business Impact & Lessons

2. Generate structured insights:
   - AI technologies used
   - Integration patterns
   - Success metrics
   - Implementation challenges
   - Best practices identified

### Phase 3: Report Generation
Generate three types of reports:
1. Individual Case Study Reports (PDF)
2. Cross-Case Analysis Report (PDF)
3. Executive Insights Dashboard (JSON)

## TECHNICAL SPECIFICATIONS

Claude Configuration:
- Model: claude-3-5-sonnet-20241022
- Temperature: 0.2
- Output Context Window: 8192 tokens

Processing Pipeline:
1. Web Content Extraction
2. Enterprise AI Case Validation
3. Structured Analysis
4. Report Generation
5. Cross-Case Pattern Analysis

## OUTPUT REQUIREMENTS

Each qualified case study generates:
1. Validation Report:
   - Enterprise AI qualification criteria met
   - Data quality assessment
   - Content completeness check

2. Detailed Analysis Report:
   - Executive Summary
   - AI Strategy Analysis
   - Technical Implementation Details
   - Business Impact Assessment
   - Key Success Factors
   - Lessons Learned

3. Cross-Case Insights:
   - Common patterns
   - Success factors
   - Implementation challenges
   - Technology trends
   - ROI patterns


## ERROR HANDLING & LOGGING

Track and log:
- Case study qualification decisions
- Content extraction issues
- Analysis completeness
- Pattern detection confidence
- Processing errors and retries

## VALIDATION CRITERIA

Enterprise AI Case Validation:
- Company size/maturity check
- AI implementation scope
- Business process integration
- Measurable outcomes
- Enterprise-scale considerations

Report Quality Validation:
- Content completeness
- Technical accuracy
- Business impact quantification
- Implementation detail sufficiency
- Cross-reference verification


The system should focus on creating high-quality, business-focused analysis of enterprise AI implementations, highlighting patterns, best practices, and lessons learned across cases.


1-) Firecrawl extracts the content
2-) The text is sent to Claude
3-) Claude responses and IF it's an enterprise case study, creates an executive report. If not, passes and informs the user via terminal (print)
4-) The report is saved in the reports/individual folder
5-) The report is saved in the reports/cross_case_analysis folder
6-) The report is saved in the reports/executive_dashboard.json folder

