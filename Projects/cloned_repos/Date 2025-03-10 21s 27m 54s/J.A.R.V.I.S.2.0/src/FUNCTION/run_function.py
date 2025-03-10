from src.FUNCTION.link_op import search_youtube 
from src.FUNCTION.weather import weather_report
from src.FUNCTION.news import news_headlines
from src.FUNCTION.youtube_downloader import yt_download
from src.FUNCTION.app_op import app_runner
from src.FUNCTION.link_op import search_youtube , open_github , open_instagram , open_youtube , yt_trending
from src.BRAIN.text_to_info import send_to_ai 
from src.FUNCTION.incog import private_mode 
from src.FUNCTION.Email_send import send_email
from src.FUNCTION.phone_call import make_a_call
from src.BRAIN.RAG import chat_with_rag
from src.FUNCTION.internet_search import duckgo_search
from src.BRAIN.chat_with_ai import personal_chat_ai
from typing import Union


FUNCTION_MAP = {
    'search_youtube': search_youtube,
    'weather_report': weather_report,
    'news_headlines':news_headlines,
    'yt_download':yt_download,
    'app_runner':app_runner,
    'open_github':open_github,
    'open_instagram':open_instagram,
    'open_youtube':open_youtube,
    'yt_trending':yt_trending,
    'send_to_ai':send_to_ai,
    'private_mode':private_mode,
    'send_email':send_email,
    'make_a_call':make_a_call,
    'duckgo_search':duckgo_search,
    'chat_with_rag':chat_with_rag,
    'personal_chat_ai':personal_chat_ai
}




def execute_function_call(function_call: dict) -> Union[None,dict,list]:
    """
    Execute a function based on the function call dictionary
    
    :param function_call: Dictionary with 'name' and 'arguments' keys
    """
    output = None 
    try:
        # Extract function name and arguments
        func_name = function_call.get('name')
        args = function_call.get('parameters')
        
        
        if not func_name:
            return  output
        
        func = FUNCTION_MAP.get(func_name)
        
        if not func:
            print("No functions Found..")
            return output
        
        if args:
            all_parameters = [args[k] for k in args.keys()]
            output = func(*all_parameters)
            return output
        else:
            print("[*] No parameters provided .....")
            output = func()
    except KeyError as e:
        print(f"Invalid function call format: Missing {e}")
    except Exception as e:
        print(f"Error executing function: {e}")
        
    return output