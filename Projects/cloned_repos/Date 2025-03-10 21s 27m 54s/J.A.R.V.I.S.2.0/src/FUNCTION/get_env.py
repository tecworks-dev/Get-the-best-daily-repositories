from dotenv import load_dotenv
from os import environ
from typing import Union 

load_dotenv()

def load_variable(variable_name:str) -> Union[str , None]:
    try:
        variable = environ.get(variable_name.strip())
        return variable
    except Exception as e:
        print(f"Error:{e}")
    return None 

import platform

def check_os():
    os_name = platform.system()
    return os_name 
