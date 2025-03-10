from subprocess import call , run 
from src.FUNCTION.get_env import check_os 
from os import system 


def open_chrome_incognito(topic:str) -> None:
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"  # Adjust the path if needed
    search_url = f"https://www.google.com/search?q={topic}"
    run([chrome_path, '--incognito', search_url])
    
def open_firefox_private(topic:str) -> None:
    firefox_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"  # Adjust the path if needed
    search_url = f"https://www.google.com/search?q={topic}"
    run([firefox_path, '-private-window', search_url])
    
def linux_firefox(topic:str) -> None:
    search_url = f"https://www.google.com/search?q={topic}"
    run (["firefox" , "--private" , search_url])
    
    
def incog_mode(topic:str) -> None:
    """"Search in the incognito or private mode for specific topic"""
    # Construct the URL for Google search with the topic
    search_url = f"https://www.google.com/search?q={topic}"
    applescript_code = f'''
    tell application "Google Chrome"
    activate
        tell (make new window with properties {{mode:"incognito"}})
            set URL of active tab to "{search_url}"
        end tell
    end tell'''
    run(['osascript', '-e', applescript_code])
    return  

def private_mode(topic:str) -> None:
    os_name = check_os()
    if os_name == "Linux":
        linux_firefox(topic)
    elif os_name == "Darwin":
        incog_mode(topic)
    elif os_name == "Windows":
        open_chrome_incognito(topic)
    