# config.py
"""
Configuration file for project constants and settings.
"""

import os

# API keys (should be set via environment variables for security)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
TRANSLATE_API_KEY = "not_use"

# OpenAI API configuration
BASE_URL = os.getenv('OPENAI_API_ENDPOINT', 'https://api.openai.com/v1')

MAX_TOKENS = 512

# Logging level
LOGGING_LEVEL = 'INFO'

