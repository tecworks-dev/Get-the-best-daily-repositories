from typing import Union 
import json 
import re
from typing import  Union
from langchain_ollama import ChatOllama
from DATA.tools import ALL_FUNCTIONS 

TOOLS = "./DATA/tools.json"

def load_tools(file_path: str) -> Union[dict, list]:
    try:
        with open(file_path, "r") as file:
            tools = json.load(file)
            return tools
    except FileNotFoundError:
        print("Error: Tools configuration file not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return []

def load_tools_message(file_path: str) -> str:
    tools = load_tools(file_path)
    return json.dumps(tools, indent=2)



# # Define system message for guiding the AI
# SYSTEM_MESSAGE = {
#     "role": "system",
#     "content": (
#         "You are a function-calling AI model. You are provided with function signatures"
#         f"within <tools> {load_tools_message(TOOLS)} </tools> XML tags. "
#         "For each user query, extract the most relevant parameters from the input and select the most appropriate functions."
#         "Return the list of tools call in this format :\n\n"
#         "[" '{"name": "<function-name>", "arguments": {<args-dict>}' "]"
#         "only provide values in parameters if functions required it."
#         "Ensure that extracted parameters match the function signature exactly."
#         "Return only the function calls in the specified format without explanations or extra content."
#     ),
# }


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are expert in function-calling . You are provided with function signatures"
        f"within <tools> {ALL_FUNCTIONS} </tools> XML tags."
        "first Understand the query.analyze the intent then think about avavailable funtions."
        "For each user query, extract the most relevant parameters from the input and select the most appropriate functions."
        "Return the list of function calls in this format :\n\n"
        "[json object]"
        "only provide values in parameters if functions required it."
        "Ensure that extracted parameters match the function signature exactly."
        "Return only the function calls in the specified format without explanations or extra content."
        "---\n\n"
        "**Examples of User Queries and Expected Function Calls (JSON ONLY Examples):**\n\n" # Slightly reworded section title

        "**Example 1: Weather Check**\n" # Reordered examples for numerical sequence
        "User Query: What's the weather in London?\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "weather_report",\n'
        '     "parameters": {\"location\": \"London\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 2: Open YouTube Website**\n"
        "User Query: Open YouTube.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "open_youtube",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 3: Joke Generation**\n" # Reordered examples for numerical sequence and corrected title
        "User Query: Tell me a joke.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "send_to_ai",\n'
        '     "parameters": {\"prompt\": \"Tell me a joke\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 5: Deep Conversation**\n" # Reordered examples for numerical sequence
        "User Query: Let's discuss the ethical implications of AI in detail.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 6: Multi-Function Call - News and Deep Conversation**\n" # Reordered examples for numerical sequence
        "User Query: Get news headlines and then let's discuss the top story using RAG.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "news_headlines",\n'
        '     "parameters": {}\n'
        "   },\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 7: Multi-Function Call - Stock Data and YouTube Search**\n" # Reordered examples for numerical sequence (Example 14 became Example 5)
        "User Query: Get stock data for NASDAQ and then search YouTube for NASDAQ analysis.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "get_stock_data",\n'
        '     "parameters": {\"exchange\": \"NASDAQ\"}\n'
        "   },\n"
        "   {\n"
        '     "name": "search_youtube",\n'
        '     "parameters": {\"topic\": \"NASDAQ analysis\"}\n'
        "   }\n"
        "]\n"
        "```\n"
        "**Example 8: Information Search (Web Search)**\n" # Example for previous regression case
        "User Query:  Who will win the FIFA World Cup in 2030?\n" # Future-oriented search
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "duckgo_search",\n'
        '     "parameters": {\"query\": \"FIFA World winners Cup in 2030?"}\n' # Formulated search query
        "   }\n"
        "]\n"
        "```\n\n"
    ),
}

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an expert in function-calling. You are provided with a list of available tools and their descriptions within <tools> XML tags.\n"
        "<tools>\n"
        f"{ALL_FUNCTIONS}\n"
        "</tools>\n\n"
        "Your task is to analyze user queries and determine the most appropriate tool(s) from the provided list to fulfill the user's request. Follow these steps:\n\n"
        "1. **Understand the User Query:** Carefully read and understand what the user is asking for.\n"
        "2. **Analyze Intent:** Identify the user's intent and the core action they want to perform.  Users may ask for single actions or a combination of actions.\n"
        "3. **Review Available Tools:** Examine the descriptions of the tools provided within the <tools> tags.\n"
        "4. **Select Relevant Tool(s):** Choose the tool(s) that are most relevant and necessary to address the user's query and intent.  **For queries involving multiple actions, select multiple tools if necessary.** Prioritize using the fewest tools possible while fulfilling all aspects of the request.\n"
        "5. **Parameter Extraction:** For each selected tool, identify if it requires parameters. If parameters are required, extract the necessary information directly from the user's query to populate these parameters. Ensure that the extracted parameter values are valid and match the function signature exactly as described in the tool definition.\n"
        "6. **Function Call Output:** Return a list of function calls in JSON format. Each function call should include the 'name' of the function and its 'parameters' (if any).  If a function has no parameters, the 'parameters' field should be an empty JSON object `{}`.\n"
        "7. **Output Format:** Return **ONLY** the JSON formatted list of function calls. Do not include any explanations, justifications, or extra text outside of the JSON output.\n\n"
        "**Example Output Format:**\n"
        "```json\n"
        "[ \n"
        "   {\n"
        '     "name": "function_name_1",\n'
        '     "parameters": {"param_name_1": "param_value_1", "param_name_2": "param_value_2"}\n'
        "   },\n"
        "   {\n"
        '     "name": "function_name_2",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n"
        "If no function is relevant to the user query, return an empty JSON array `[]`.\n"
        #"Ensure that extracted parameters strictly adhere to the function signature. Only provide values for parameters if the function explicitly requires them as per its definition.\n\n"
        "---\n\n"
        "**Examples of User Queries and Expected Function Calls (Including Multi-Function Calls):**\n\n"
        "**Example 1: Single Function - Weather Check**\n"
        "User Query: What's the weather in London?\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "weather_report",\n'
        '     "parameters": {\"location\": \"London\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 2: Single Function - Joke Generation (Corrected Example!)**\n" # Corrected Example
        "User Query: Tell me a joke.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "send_to_ai",\n'
        '     "parameters": {\"prompt\": \"Tell me a joke\"}\n' # Corrected parameter name and value
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 3: Single Function - Deep Conversation**\n"
        "User Query: Let's discuss the ethical implications of AI in detail.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 14: Multi-Function Call - News and Deep Conversation**\n"
        "User Query: Get news headlines and then let's discuss the top story using RAG.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "news_headlines",\n'
        '     "parameters": {}\n'
        "   },\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 4: Multi-Function Call - Stock Data and YouTube Search**\n"
        "User Query: Get stock data for NASDAQ and then search YouTube for NASDAQ analysis.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "get_stock_data",\n'
        '     "parameters": {\"exchange\": \"NASDAQ\"}\n'
        "   },\n"
        "   {\n"
        '     "name": "search_youtube",\n'
        '     "parameters": {\"topic\": \"NASDAQ analysis\"}\n'
        "   }\n"
        "]\n"
        "```\n"
    ),
}

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "**IMPORTANT: Your ONLY output should be a valid JSON list of function calls. Do not include any other text, explanations, or conversational elements.**\n\n" # Strongest JSON-only emphasis at the very beginning
        "You are an expert in function-calling. You are provided with a list of available tools and their descriptions within <tools> XML tags.\n"
        "<tools>\n"
        f"{ALL_FUNCTIONS}\n"
        "</tools>\n\n"
        "Your task is to analyze user queries and determine the most appropriate tool(s) from the provided list to fulfill the user's request. Follow these steps for accurate function calls:\n\n" # Slightly rephrased intro

        "1. **Understand User Query:** Carefully read to understand the user's request.\n" # Shorter instructions
        "2. **Analyze Intent:** Identify the core action(s) the user wants to perform.\n" # Shorter instructions
        "3. **Review Tools:** Examine tool descriptions in `<tools>` to find the best matches.\n" # Shorter instructions
        "4. **Tool Selection:** Choose the BEST tool(s) to precisely address the user's intent.  Use multiple tools if necessary for complex queries, but prioritize the fewest tools needed.\n" # Slightly shorter, more direct
        "5. **Parameter Extraction:** Extract *required* parameters from the user query, ensuring names and values strictly match function signatures.\n" # Shorter, stronger emphasis on matching
        "6. **Function Call Output:** Create a JSON list of function calls with 'name' and 'parameters'." # Shorter instruction
        "7. **Output Validation:** Ensure your function calls are based *only* on tool descriptions and the user query. **Do not invent functions or parameters. Avoid hallucination.**\n" # Added hallucination prevention directive
        
        "\n**CRITICAL OUTPUT FORMAT - JSON ONLY!**\n" # Stronger section heading
        "**You MUST return *ONLY* a valid JSON list of function calls.  Absolutely no surrounding text is allowed. Your output MUST be parsable as JSON.  If no function is relevant, return `[]`.**\n\n" # Stronger, repeated JSON-only instruction

        "**Example Output Format:**\n"
        "```json\n"
        "[ \n"
        "   {\n"
        '     "name": "function_name_1",\n'
        '     "parameters": {"param_name_1": "param_value_1", "param_name_2": "param_value_2"}\n'
        "   },\n"
        "   {\n"
        '     "name": "function_name_2",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n"
        #"Extracted parameters must strictly adhere to function signatures. Provide parameter values ONLY when explicitly required by the function definition.\n\n" # Slightly reworded for emphasis

        "---\n\n"
        "**Examples of User Queries and Expected Function Calls (JSON ONLY Examples):**\n\n" # Slightly reworded section title

        "**Example 1: Weather Check**\n" # Reordered examples for numerical sequence
        "User Query: What's the weather in London?\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "weather_report",\n'
        '     "parameters": {\"location\": \"London\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 2: Open YouTube Website**\n"
        "User Query: Open YouTube.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "open_youtube",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 3: Joke Generation**\n" # Reordered examples for numerical sequence and corrected title
        "User Query: Tell me a joke.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "send_to_ai",\n'
        '     "parameters": {\"prompt\": \"Tell me a joke\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 5: Deep Conversation**\n" # Reordered examples for numerical sequence
        "User Query: Let's discuss the ethical implications of AI in detail.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {\"subject\": \"AI\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 6: Multi-Function Call - News and Deep Conversation**\n" # Reordered examples for numerical sequence
        "User Query: Get news headlines and then let's discuss about life.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "news_headlines",\n'
        '     "parameters": {}\n'
        "   },\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {\"subject\": \"life\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 7: Multi-Function Call - Stock Data and YouTube Search**\n" # Reordered examples for numerical sequence (Example 14 became Example 5)
        "User Query: Get stock data for NASDAQ and then search YouTube for NASDAQ analysis.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "get_stock_data",\n'
        '     "parameters": {\"exchange\": \"NASDAQ\"}\n'
        "   },\n"
        "   {\n"
        '     "name": "search_youtube",\n'
        '     "parameters": {\"topic\": \"NASDAQ analysis\"}\n'
        "   }\n"
        "]\n"
        "```\n"
        "**Example 8: Information Search (Web Search)**\n" # Example for previous regression case
        "User Query:  Who will win the FIFA World Cup in 2030?\n" # Future-oriented search
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "duckgo_search",\n'
        '     "parameters": {\"query\": \"FIFA World winners Cup in 2030?"}\n' # Formulated search query
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 8: Multi-Function Call - Stock Data and Open Instagram**\n" # Reordered examples for numerical sequence (Example 14 became Example 5)
        "User Query: Get stock data for NASDAQ and then Open instagram web.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "get_stock_data",\n'
        '     "parameters": {\"exchange\": \"NASDAQ\"}\n'
        "   },\n"
        "   {\n"
        '     "name": "open_instagram",\n'
        '     "parameters": {}\n'
        "   }\n"
        "]\n"
        "```\n"
        "**Example 9: Deep Conversation**\n" # Reordered examples for numerical sequence
        "User Query: Discuss the philosophical implications of quantum mechanics.\n"
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "chat_with_rag",\n'
        '     "parameters": {\"subject\": \"quantum mechanics\"}\n'
        "   }\n"
        "]\n"
        "```\n\n"
        "**Example 9: Information Search (Web Search)**\n" # Example for previous regression case
        "User Query: Who is the current president of Brazil?\n" # Future-oriented search
        "Expected JSON Output:\n"
        "```json\n"
        "[\n"
        "   {\n"
        '     "name": "duckgo_search",\n'
        '     "parameters": {\"query\": \"current president of Brazil?"}\n' # Formulated search query
        "   }\n"
        "]\n"
        "```\n\n"
    ),
}


AVAILABLE_FUNCTION_NAMES_STRING = [func.get("name") for func in ALL_FUNCTIONS.get("tools")]

def parse_tool_calls(response: str) -> Union[list, None]:
    try:
        # Regex to extract content inside a list
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            list_content = match.group(0)
            return json.loads(list_content)
    except json.JSONDecodeError as e:
        print(f"Error in formatting JSON: \n{e}")
    except Exception as e:
        print(f"Unexpected error: \n{e}")
    return None

def create_function_call(user_query:str , model="granite3.1-dense:2b")-> Union[str, None]:
    messages = [SYSTEM_MESSAGE, {"role": "user", "content": user_query}]
    # Example conversation
    try:
        llm = ChatOllama(
        model=model,
        temperature=0
        )
        response = llm.invoke(messages)
        raw_func_response =  response.content
        functional_response = parse_tool_calls(raw_func_response)
        valid_functions = [func for func in functional_response if func.get("name").lower() in AVAILABLE_FUNCTION_NAMES_STRING]
        return valid_functions
    
                
    except Exception as e:
        print(f"Error creating function call: {e}")
    return None 

if __name__ == "__main__":
    query = "plase send email send email "
    response = create_function_call(query)