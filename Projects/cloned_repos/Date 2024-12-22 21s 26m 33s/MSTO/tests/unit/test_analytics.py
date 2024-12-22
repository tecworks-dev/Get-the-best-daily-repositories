"""
Unit tests for analytics module
"""
import pytest
from msto.core.analytics import detect_unusual_drop, sentiment_analysis, classify_events, estimate_impact

def test_detect_unusual_drop(mock_stock_data):
    """Test unusual drop detection."""
    # Test with normal data
    drop = detect_unusual_drop(mock_stock_data)
    assert drop is None or isinstance(drop, float)

def test_sentiment_analysis():
    """Test sentiment analysis."""
    # Test with empty articles
    assert sentiment_analysis([]) == 0.0
    
    # Test with sample articles
    articles = [
        {"title": "Great earnings report", "description": "Company exceeds expectations"},
        {"title": "New product launch", "description": "Innovative features announced"}
    ]
    sentiment = sentiment_analysis(articles)
    assert isinstance(sentiment, float)
    assert -1.0 <= sentiment <= 1.0

def test_classify_events():
    """Test event classification."""
    # Test with empty articles
    assert classify_events([]) == "no_news"
    
    # Test earnings event
    articles = [
        {"title": "Q4 Earnings Report", "description": "Company reports strong earnings"},
        {"title": "Financial Results", "description": "Revenue growth continues"}
    ]
    assert classify_events(articles) == "earnings"
    
    # Test product launch event
    articles = [
        {"title": "New Product Launch", "description": "Company announces new features"},
        {"title": "Innovation News", "description": "Revolutionary product released"}
    ]
    assert classify_events(articles) == "product_launch"

def test_estimate_impact():
    """Test impact estimation."""
    # Test with positive sentiment and earnings event
    impact = estimate_impact(0.8, "earnings")
    assert isinstance(impact, float)
    assert -1.0 <= impact <= 1.0
    
    # Test with negative sentiment and legal event
    impact = estimate_impact(-0.6, "legal")
    assert isinstance(impact, float)
    assert -1.0 <= impact <= 1.0
    
    # Test with neutral sentiment and unknown event
    impact = estimate_impact(0.0, "other")
    assert isinstance(impact, float)
    assert impact == 0.0 