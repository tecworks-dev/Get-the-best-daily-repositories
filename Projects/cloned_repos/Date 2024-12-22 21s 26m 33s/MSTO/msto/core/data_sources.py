import yfinance as yf
import requests
import datetime
import logging
import json
from config import NEWS_API_KEY

def fetch_stock_data(ticker: str, days: int):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=days)
    data = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    return data

def fetch_news(company_name: str, from_date: datetime.date, to_date: datetime.date):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': company_name,
        'from': from_date.isoformat(),
        'to': to_date.isoformat(),
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': NEWS_API_KEY,
        'pageSize': 50
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        logging.error(json.dumps({"level": "ERROR", "message": "News fetch error", "error": str(e)}))
        return []

def get_fundamental_metrics(ticker: str):
    info = yf.Ticker(ticker).info
    return {
        "pe_ratio": info.get("trailingPE", None),
        "pb_ratio": info.get("priceToBook", None)
    }
