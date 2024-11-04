import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Claude Configuration
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 8192

# Application Settings
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
RAW_DIR = os.path.join(BASE_DIR, "raw_content")
SECTIONS_DIR = os.path.join(BASE_DIR, "sections")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORTS_INDIVIDUAL_DIR = os.path.join(REPORTS_DIR, "individual")
REPORTS_CROSS_CASE_DIR = os.path.join(REPORTS_DIR, "cross_case_analysis")
REPORTS_EXECUTIVE_DIR = os.path.join(REPORTS_DIR, "executive_dashboard")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [
    INPUT_DIR, 
    RAW_DIR, 
    SECTIONS_DIR, 
    REPORTS_DIR,
    REPORTS_INDIVIDUAL_DIR,
    REPORTS_CROSS_CASE_DIR,
    REPORTS_EXECUTIVE_DIR,
    LOGS_DIR
]:
    os.makedirs(directory, exist_ok=True)

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, "processing_log.json")
VALIDATION_LOG = os.path.join(LOGS_DIR, "validation_log.json") 