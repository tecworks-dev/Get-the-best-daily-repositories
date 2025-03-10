import requests
from src.FUNCTION.get_env import load_variable
from typing import Union
# in , us 
def news_headlines(top:int = 10) -> Union[list[str] , None]:
    """Fetch top news headlines."""
    api_key = load_variable("News_api")
    country = load_variable("Country")
    headlines = []
    url = ('https://newsapi.org/v2/top-headlines?'
        f'country={country}&'
        f'apiKey={api_key}')
    try:
        response = requests.get(url).json()
        all_articles = response['articles']
        total_results = int(response['totalResults'])
        for i in range(min(top , total_results)):
            headline = all_articles[i]['title']
            headlines.append(headline)
        return headlines
    except Exception as e:
        print(e)
        pass
    return None 



