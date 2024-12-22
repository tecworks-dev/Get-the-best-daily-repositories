import numpy as np
from typing import List, Dict, Any
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def detect_unusual_drop(data: pd.DataFrame) -> float:
    returns = data['Close'].pct_change() * 100
    if len(returns.dropna()) < 10:
        return None
    recent_return = returns.iloc[-1]
    mu = returns.mean()
    sigma = returns.std()
    threshold = mu - 2 * sigma
    if recent_return < threshold:
        return float(recent_return)
    return None

def sentiment_analysis(articles: List[Dict[str, Any]]) -> float:
    if not articles:
        return 0.0
    
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        if not text.strip():
            continue
        
        sentiment_scores = sia.polarity_scores(text)
        sentiments.append(sentiment_scores['compound'])
    
    return np.mean(sentiments) if sentiments else 0.0

def classify_events(articles: List[Dict[str, Any]]) -> str:
    if not articles:
        return "no_news"
    
    # Define event keywords
    event_patterns = {
        'earnings': (
            r'\bearnings?\b|'
            r'\brevenue\b|'
            r'\bprofit\b|'
            r'\bloss(es)?\b|'
            r'\bquarterly\b|'
            r'\bfinancial\s+results?\b|'
            r'\bguidance\b|'
            r'\bEPS\b|'
            r'\bforecasts?\b|'
            r'\bestimates?\b|'
            r'\boutlook\b|'
            r'\bfiscal\b|'
            r'\bpreliminary\s+(results?|figures?)\b'
        ),
        'merger_acquisition': (
            r'\bmerger\b|'
            r'\bacquisition\b|'
            r'\btakeover\b|'
            r'\bdeal\b|'
            r'\bbuyout\b|'
            r'\bbid\b|'
            r'\bM&A\b|'
            r'\bjoint\s+venture\b|'
            r'\ball-cash\s+deal\b|'
            r'\bshare-swap\b|'
            r'\bconsolidation\b|'
            r'\bstrategic\s+(partnership|alliance)\b|'
            r'\bintegration\b'
        ),
        'management_change': (
            r'\bCEO\b|'
            r'\bCFO\b|'
            r'\bCOO\b|'
            r'\bexecutive\b|'
            r'\bmanagement\b|'
            r'\bleadership\b|'
            r'\bappointed\b|'
            r'\bresigned\b|'
            r'\bboard\s+of\s+directors?\b|'
            r'\bchairman\b|'
            r'\bchairwoman\b|'
            r'\bfired\b|'
            r'\bhired\b|'
            r'\bsuccession\b|'
            r'\bexecutive\s+change\b|'
            r'\bexecutive\s+shuffle\b|'
            r'\bstepping\s+down\b|'
            r'\bnew\s+management\b|'
            r'\binterim\s+(CEO|CFO|COO)\b'
        ),
        'product_launch': (
            r'\blaunch(ed|ing)?\b|'
            r'\brelease(d|ing)?\b|'
            r'\bannounce(d|ment)?\b|'
            r'\bnew\s+product\b|'
            r'\binnovation\b|'
            r'\bunveil(ed|ing)?\b|'
            r'\bintroduce(d|ing)?\b|'
            r'\broll(\s?out|ing\s+out)\b|'
            r'\bupdate(d|s)?\b|'
            r'\bfeature(s)?\b|'
            r'\bproduct\s+line\b|'
            r'\bproduct\s+portfolio\b|'
            r'\bR&D\b|'
            r'\bprototype\b|'
            r'\bupgrad(ed|ing)?\b'
        ),
        'legal': (
            r'\blawsuit(s)?\b|'
            r'\blegal\b|'
            r'\bcourt\b|'
            r'\bsettlement(s)?\b|'
            r'\bregulatory\b|'
            r'\bfine(s)?\b|'
            r'\binvestigation(s)?\b|'
            r'\bcomplaint(s)?\b|'
            r'\blitigation\b|'
            r'\bantitrust\b|'
            r'\bcompliance\b|'
            r'\bfraud\b|'
            r'\bSEC\b|'
            r'\bDOJ\b|'
            r'\bclass\s+action\b|'
            r'\bpatent\s+dispute\b|'
            r'\bwhistleblower\b|'
            r'\bsettle(d|ing)?\b'
        ),
        'market_movement': (
            r'\bmarket(s)?\b|'
            r'\bstock(s)?\b|'
            r'\bshare(s)?\b|'
            r'\btrading\b|'
            r'\binvestors?\b|'
            r'\bIPO\b|'
            r'\blisting\b|'
            r'\bvaluation\b|'
            r'\bmarket\s+cap\b|'
            r'\bprice\s+target\b|'
            r'\bupgrade(d)?\b|'
            r'\bdowngrade(d)?\b|'
            r'\banalyst\s+report\b|'
            r'\bdividend(s)?\b|'
            r'\bbuyback(s)?\b|'
            r'\bshort\s+selling\b|'
            r'\bvolatility\b|'
            r'\bindex\s+inclusion\b|'
            r'\bindex\s+removal\b|'
            r'\brating(s)?\b'
        )
    }

    
    event_counts = Counter()
    
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        for event_type, pattern in event_patterns.items():
            if re.search(pattern, text):
                event_counts[event_type] += 1
    
    if not event_counts:
        return "other"
    
    return event_counts.most_common(1)[0][0]

def estimate_impact(sentiment: float, event_type: str) -> float:
    # Event type weights
    event_weights = {
        'earnings': 1.0,
        'merger_acquisition': 0.8,
        'management_change': 0.6,
        'product_launch': 0.5,
        'legal': 0.7,
        'market_movement': 0.4,
        'other': 0.3,
        'no_news': 0.1
    }
    
    # Calculate impact score (-1 to 1)
    event_weight = event_weights.get(event_type, 0.3)
    impact = sentiment * event_weight
    
    # Ensure the impact is within bounds
    return max(min(impact, 1.0), -1.0)
