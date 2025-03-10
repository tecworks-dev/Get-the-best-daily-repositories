# ALL_FUNCTIONS = {
#     "type": "function",
#     "functions": [
#     {
#     "name": "get_stock_fundamentals",
#     "description": "get_stock_fundamentals(symbol: str) -> dict - Get fundamental data for a given stock symbol using yfinance API.\n\n    Args:\n    symbol (str): The stock symbol.\n\n    Returns:\n    dict: A dictionary containing fundamental data.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#         "symbol": {
#             "type": "string",
#             "description": "The stock ticker symbol (e.g., AAPL, TSLA) for which fundamental data is requested."
#         }
#     },
#         "required": ["symbol"]
#     }
#     },
#     {
#         "name": "weather_report",
#         "description": "Get the current weather for a location.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "location": {"type": "string", "description": "The name of the city or location to get the weather for."}
#         },
#         "required": ["location"]
#     }
#     },
#     {
#         "name": "get_stock_data",
#         "description": "Fetch stock market data for an exchange.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#             "exchange": {"type": "string", "description": "The stock exchange symbol (e.g., NASDAQ, NYSE) to fetch data for."}
#         },
#         "required": ["exchange"]
#     }
#     },
#     {
#         "name": "search_youtube",
#         "description": "Search YouTube for a specific topic.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "topic": {"type": "string", "description": "The topic or keywords to search for on YouTube."}
#         },
#         "required": ["topic"]
#     }
#     },
#     {
#         "name": "news_headlines",
#         "description": "Fetch top news headlines.",
#         "parameters": {"type": "object", "properties": {}}
#     },
#     {
#         "name": "yt_download",
#         "description": "Download a YouTube video.",
#         "parameters": {"type": "object", "properties": {}}
#     },
#     {
#         "name": "send_to_ai",
#         "description": "Handle creative prompts like jokes or stories.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "prompt": {"type": "string", "description": "The creative prompt for the AI, such as a joke, story, or poem."}
#         },
#         "required": ["prompt"]
#     }
#     },
#     {
#         "name": "app_runner",
#         "description": "Open the specified application by name.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "app_name": {"type": "string", "description": "The name of the application to open."}
#         },
#         "required": ["app_name"]
#     }
#     },
#     {
#         "name": "open_github",
#         "description": "Open GitHub in a web browser.",
#         "parameters": {"type": "object", "properties": {}}
#     },
#     {
#         "name": "open_instagram",
#         "description": "Open Instagram in a web browser.",
#         "parameters": {"type": "object", "properties": {}}
#     },
#     {
#         "name": "open_youtube",
#         "description": "Open YouTube in a web browser.",
#         "parameters": {"type": "object", "properties": {}}
#     },
#     {
#         "name": "private_mode",
#         "description": "Search in the incognito or private mode for a specific topic.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "topic": {"type": "string", "description": "The topic to search for in incognito mode."}
#         },
#         "required": ["topic"]
#     }
#     },
#     {
#         "name": "make_a_call",
#         "description": "Make a phone call to the provided contact name.",
#         "parameters": {
#         "type": "object",
#         "properties": {
#             "name": {"type": "string", "description": "The name of the contact to call."}
#         },
#         "required": ["name"]
#     }
#     },
#     {
#         "name": "send_email",
#         "description": "Send an email on Gmail.",
#         "parameters": {"type": "object", "properties": {}}
#     },
    
# ]
# }


TEMP_ALL_FUNCTIONS = {
    "tools": [
        {"name": "weather_report", "description": "Get the current weather for a location.", "parameters": {"location": "string", "required": ["location"]}},
        {"name": "get_stock_data", "description": "Fetch stock market data for an exchange.", "parameters": {"exchange": "string", "required": ["exchange"]}},
        {"name": "search_youtube", "description": "Search YouTube for a specific topic.", "parameters": {"topic": "string", "required": ["topic"]}},
        {"name": "news_headlines", "description": "Fetch top news headlines.", "parameters": {}},
        {"name": "yt_download", "description": "Download a YouTube video.", "parameters": {}},
        {"name": "send_to_ai", "description": "Handle creative prompts like jokes or stories.", "parameters": {"prompt": "string", "required": ["prompt"]}},
        {"name": "app_runner", "description": "Open the specified application by name. For example, you can say 'open WhatsApp' or 'run Chrome'.", "parameters": {"app_name": "string", "required": ["app_name"]}},
        {"name": "open_github", "description": "Open GitHub in a web browser.", "parameters": {}},
        {"name": "open_instagram", "description": "Open Instagram in a web browser.", "parameters": {}},
        {"name": "open_youtube", "description": "Open YouTube in a web browser.", "parameters": {}},
        {"name": "private_mode", "description": "Search in the incognito or private mode for specific topic","parameters":{"topic":"string" , "required":["topic"]}},
        {"name": "make_a_call","description": "make a phone call to provided contact name." , "parameters":{"name":"string" , "required":["name"]}},
        {"name": "send_email","description":"send email on gmail","parameters":{}},
        {"name": "duckgo_search","description":"search provided query on internet for quick information.","parameters":{"query":"string" , "required":["query"]}},
        {"name": "chat_with_rag","description":"For Depper and insgithfull discussions using (RAG)." , "parameters":{"subject":"string" , "required":["subject"]}}
    ]
}

## Format for functions for llm using pydanctic 
# dict : "OBJECT"
# list : "ARRAY"
# str : "STRING
# int: "INTEGER"
# float : "NUMBER"  

ALL_FUNCTIONS = {
    "tools": [
        {
            "name": "weather_report",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "get_stock_data",
            "description": "Fetch stock market data for an exchange.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "exchange": {"type": "STRING"}
                },
                "required": ["exchange"]
            }
        },
        {
            "name": "search_youtube",
            "description": "Search YouTube for a specific topic.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"}
                },
                "required": ["topic"]
            }
        },
        {
            "name": "news_headlines",
            "description": "Fetch top news headlines."
        },
        {
            "name": "yt_download",
            "description": "Download a YouTube video."
        },
        {
            "name": "personal_chat_ai",
            "description": "Engage in an empathetic conversation, recalling stored personal information. Use this for questions about the user's memories, goals, feelings, or identity, such as 'What is my name?' or 'Tell me a memory.' Ensure responses are contextually relevant.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "first_query": {"type": "STRING"}
                },
                "required": ["first_query"]
            }
        },
        {
            "name": "send_to_ai",
            "description": "Handle creative prompts like jokes or stories.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "prompt": {"type": "STRING"}
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "app_runner",
            "description": "Open the specified application by name. For example, you can say 'open WhatsApp' or 'run Chrome'.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "app_name": {"type": "STRING"}
                },
                "required": ["app_name"]
            }
        },
        {
            "name": "open_github",
            "description": "Open GitHub in a web browser."
        },
        {
            "name": "open_instagram",
            "description": "Open Instagram in a web browser."
        },
        {
            "name": "open_youtube",
            "description": "Open YouTube in a web browser."
        },
        {
            "name": "private_mode",
            "description": "Search in incognito or private mode for a specific topic.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"}
                },
                "required": ["topic"]
            }
        },
        {
            "name": "make_a_call",
            "description": "Make a phone call to the provided contact name.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"}
                },
                "required": ["name"]
            }
        },
        {
            "name": "send_email",
            "description": "Send an email on Gmail."
        },
        {
            "name": "duckgo_search",
            "description": "Search the provided query on the internet for quick information.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "chat_with_rag",
            "description": "For deeper and insightful discussions on specific topic using (RAG).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"}
                },
                "required": ["subject"]
            }
        }
    ]
}

# ALL_FUNCTIONS = {
#   "functions_str": [
#     {
#       "name": "weather_report",
#       "description": "Retrieve real-time weather information for a specified location. Requires a city name.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "location": {
#             "type": "string",
#             "description": "The name of the location (e.g., San Francisco)"
#           }
#         },
#         "required": ["location"]
#       }
#     },
#     {
#       "name": "get_stock_data",
#       "description": "Fetch the latest stock market data for a specified exchange (e.g., NYSE, NASDAQ). Requires the stock exchange name.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "exchange": {
#             "type": "string",
#             "description": "The stock exchange name (e.g., NYSE, NASDAQ)"
#           }
#         },
#         "required": ["exchange"]
#       }
#     },
#     {
#       "name": "search_youtube",
#       "description": "Search YouTube for videos related to a given topic.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "topic": {
#             "type": "string",
#             "description": "The topic to search for on YouTube"
#           }
#         },
#         "required": ["topic"]
#       }
#     },
#     {
#       "name": "news_headlines",
#       "description": "Retrieve the latest top news headlines from global news sources. No parameters are required.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "yt_download",
#       "description": "Download a YouTube video.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "send_to_ai",
#       "description": "Generate creative responses, such as jokes, poems, or short stories, based on a given user prompt.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "prompt": {
#             "type": "string",
#             "description": "The creative prompt (e.g., a joke or short story)"
#           }
#         },
#         "required": ["prompt"]
#       }
#     },
#     {
#       "name": "app_runner",
#       "description": "Open a specified application installed on the user's device. Requires the exact app name (e.g., 'Chrome', 'WhatsApp').",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "app_name": {
#             "type": "string",
#             "description": "The name of the app to open (e.g., 'WhatsApp', 'Chrome')"
#           }
#         },
#         "required": ["app_name"]
#       }
#     },
#     {
#       "name": "open_github",
#       "description": "Open GitHub in the user's web browser.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "open_instagram",
#       "description": "Open Instagram in the user's web browser.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "open_youtube",
#       "description": "Open YouTube in the user's web browser.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "private_mode",
#       "description": "Perform an internet search in incognito (private browsing) mode for a specified topic.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "topic": {
#             "type": "string",
#             "description": "The topic to search in private mode"
#           }
#         },
#         "required": ["topic"]
#       }
#     },
#     {
#       "name": "make_a_call",
#       "description": "Initiate a phone call to a specified contact name.",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "name": {
#             "type": "string",
#             "description": "The contact name to call"
#           }
#         },
#         "required": ["name"]
#       }
#     },
#     {
#       "name": "send_email",
#       "description": "Send an email using Gmail.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     },
#     {
#       "name": "duckgo_search",
#       "description": "Perform a web search using DuckDuckGo for a specified query and return relevant results. e.g(who is donald trump.)",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "query": {
#             "type": "string",
#             "description": "The query to search for"
#           }
#         },
#         "required": ["query"]
#       }
#     },
#     {
#       "name": "chat_with_rag",
#       "description": "Engage in a conversation with a Retrieval-Augmented Generation (RAG) knowledge base.",
#       "parameters": {
#         "type": "object",
#         "properties": {},
#         "required": []
#       }
#     }
#   ]
# }

# '<|start_of_role|>available_tools<|end_of_role|>\n{fu}
# <|end_of_text|>\n<|start_of_role|>system<|end_of_role|>You are a helpful assistant with access to the following function calls. Your task is to produce a list of function calls necessary to generate response to the user utterance. Use the following function calls as required.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>What is the current weather in San Francisco?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>'


#{\n "type": "function",\n "function": {\n "name": "get_stock_price",\n "description": "Retrieves the lowest and highest stock prices for a given ticker and date.",\n "parameters": {\n "type": "object",\n "properties": {\n "ticker": {\n "type": "string",\n "description": "The stock ticker symbol, e.g., \\"IBM\\"."\n },\n "date": {\n "type": "string",\n "description": "The date in \\"YYYY-MM-DD\\" format for which you want to get stock prices."\n }\n },\n "required": [\n "ticker",\n "date"\n ]\n },\n "return": {\n "type": "object",\n "description": "A dictionary containing the low and high stock prices on the given date."\n }\n }\n}\n\n{\n "type": "function",\n "function": {\n "name": "get_current_weather",\n "description": "Fetches the current weather for a given location (default: San Francisco).",\n "parameters": {\n "type": "object",\n "properties": {\n "location": {\n "type": "string",\n "description": "The name of the city for which to retrieve the weather information."\n }\n },\n "required": [\n "location"\n ]\n },\n "return": {\n "type": "object",\n "description": "A dictionary containing weather information such as temperature, weather description, and humidity."\n }\n }\n}