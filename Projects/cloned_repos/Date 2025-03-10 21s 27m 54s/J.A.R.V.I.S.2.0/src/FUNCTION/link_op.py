import webbrowser 


def search_youtube(topic:str) -> None:
    """Search YouTube for a specific topic."""
    format_topic = "+".join(topic.split())
    link = f"https://www.youtube.com/results?search_query={format_topic}"
    webbrowser.open(link)
    return None 

def open_youtube():
    """Open YouTube in a web browser."""
    link = f"https://www.youtube.com"
    webbrowser.open(link)
    return None 


def open_github():
    """Open GitHub in a web browser."""
    link = f"https://github.com"
    webbrowser.open(link)
    return None 

def yt_trending():
    """Open Youtube trending page."""
    link = f"https://www.youtube.com/feed/trending"
    webbrowser.open(link)
    return None 

def open_instagram():
    """Open Instagram in a web browser."""
    link = f"https://www.instagram.com"
    webbrowser.open(link)
    return None 


