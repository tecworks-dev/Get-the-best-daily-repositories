FUNCTION_CHAIN = {
    {
    "weather_report":["sen_to_ai"],
    "get_stock_data":["send_to_ai"],
    "new_headlines":["send_to_ai"],
    "yt_download":[],
    "send_to_ai":[],
    "app_runner":[],
    "open_github":[],
    "open_instagram":[],
    "open_youtube":[],
    "private_mode":[],
    "make_a_call":[],
    "send_email":["weather_report","news_headlines","get_stock_data"]
    }
}




# // {
# //     "tools": [
# //         {"name": "weather_report", "description": "Get the current weather for a location.", "parameters": {"location": "string", "required": ["location"]}},
# //         {"name": "get_stock_data", "description": "Fetch stock market data for an exchange.", "parameters": {"exchange": "string", "required": ["exchange"]}},
# //         {"name": "search_youtube", "description": "Search YouTube for a specific topic.", "parameters": {"topic": "string", "required": ["topic"]}},
# //         {"name": "news_headlines", "description": "Fetch top news headlines.", "parameters": {}},
# //         {"name": "yt_download", "description": "Download a YouTube video.", "parameters": {}},
# //         {"name": "send_to_ai", "description": "Handle creative prompts like jokes or stories.", "parameters": {"prompt": "string", "required": ["prompt"]}},
# //         {"name": "app_runner", "description": "Open the specified application by name. For example, you can say 'open WhatsApp' or 'run Google Chrome'.", "parameters": {"app_name": "string", "required": ["app_name"]}},
# //         {"name": "open_github", "description": "Open GitHub in a web browser.", "parameters": {}},
# //         {"name": "open_instagram", "description": "Open Instagram in a web browser.", "parameters": {}},
# //         {"name": "open_youtube", "description": "Open YouTube in a web browser.", "parameters": {}},
# //         {"name": "private_mode", "description": "Search in the incognito or private mode for specific topic","parameters":{"topic":"string" , "required":["topic"]}},
# //         {"name": "make_a_call","description":"make a phone call to provided contact name." , "parameters":{"name":"string" , "required":["name"]}},
# //         {"name":"send_email","description":"send email on gmail","parameters":{}}
# //     ]
# // }

data  = {
    "tools": [
        {
            "name": "weather_report",
            "description": "Get the current weather for a location.",
            "parameters": {
                "location": "string",
                "description": "The name of the city or location to get the weather for.",
                "required": ["location"]
            }
        },
        {
            "name": "get_stock_data",
            "description": "Fetch stock market data for an exchange.",
            "parameters": {
                "exchange": "string",
                "description": "The stock exchange symbol (e.g., NASDAQ, NYSE) to fetch data for.",
                "required": ["exchange"]
            }
        },
        {
            "name": "search_youtube",
            "description": "Search YouTube for a specific topic.",
            "parameters": {
                "topic": "string",
                "description": "The topic or keywords to search for on YouTube.",
                "required": ["topic"]
            }
        },
        {
            "name": "news_headlines",
            "description": "Fetch top news headlines.",
            "parameters": {}
        },
        {
            "name": "yt_download",
            "description": "Download a YouTube video.",
            "parameters": {}
        },
        {
            "name": "send_to_ai",
            "description": "Handle creative prompts like jokes or stories.",
            "parameters": {
                "prompt": "string",
                "description": "The creative prompt for the AI, such as a joke, story, or poem.",
                "required": ["prompt"]
            }
        },
        {
            "name": "app_runner",
            "description": "Open the specified application by name. For example, you can say 'open WhatsApp' or 'run Google Chrome'.",
            "parameters": {
                "app_name": "string",
                "description": "The name of the application to open.",
                "required": ["app_name"]
            }
        },
        {
            "name": "open_github",
            "description": "Open GitHub in a web browser.",
            "parameters": {}
        },
        {
            "name": "open_instagram",
            "description": "Open Instagram in a web browser.",
            "parameters": {}
        },
        {
            "name": "open_youtube",
            "description": "Open YouTube in a web browser.",
            "parameters": {}
        },
        {
            "name": "private_mode",
            "description": "Search in the incognito or private mode for a specific topic.",
            "parameters": {
                "topic": "string",
                "description": "The topic to search for in incognito mode.",
                "required": ["topic"]
            }
        },
        {
            "name": "make_a_call",
            "description": "Make a phone call to the provided contact name.",
            "parameters": {
                "name": "string",
                "description": "The name of the contact to call.",
                "required": ["name"]
            }
        },
        {
            "name": "send_email",
            "description": "Send an email on Gmail.",
            "parameters": {}
        }
    ]
}