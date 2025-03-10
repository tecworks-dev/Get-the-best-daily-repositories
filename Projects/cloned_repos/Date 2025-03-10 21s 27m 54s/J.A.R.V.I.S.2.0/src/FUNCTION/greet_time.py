from datetime import datetime

def time_of_day() -> str:
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning sir!"
    elif 12 <= current_hour < 17:
        return "Good afternoon sir!"
    elif 17 <= current_hour < 21:
        return "Good evening sir!"
    else:
        return "Good night sir!"
