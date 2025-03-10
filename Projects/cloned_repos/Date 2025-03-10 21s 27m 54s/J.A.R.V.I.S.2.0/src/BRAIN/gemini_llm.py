from google import genai
from google.genai import types
from DATA.tools import ALL_FUNCTIONS
from src.FUNCTION.get_env import load_variable

# #from DATA.tools import ALL_FUNCTIONS
genai_key = load_variable("genai_key")
# #genai.configure(api_key=genai_key)
client = genai.Client(api_key=genai_key)

def function_call_gem(query:str) -> list:
    client = genai.Client(api_key=genai_key)
    
    # conversion to tools so model can understand it better.
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=ALL_FUNCTIONS["tools"])]
    )
    
    try:
    #config = types.GenerateContentConfig(tools=tools , automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=query,
            config = config
        )
        return response.function_calls 
    except Exception as e:
        print(f"Error: {e}")
        return []



def gem_generate_fuction_calls(user_query:str) -> dict:
    response_list = []
    function_call_response = function_call_gem(user_query)
    if function_call_response:
        for fn in function_call_response:
            temp = {
                "name":fn.name,
                "parameters":fn.args
            }
            response_list.append(temp)
            
    return response_list
