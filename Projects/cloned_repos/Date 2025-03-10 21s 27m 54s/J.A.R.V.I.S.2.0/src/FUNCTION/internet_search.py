from src.BRAIN.text_to_info import send_to_ai
from duckduckgo_search import DDGS

def duckgo_search(query:str) -> str:
    """"search provided query on internet for quick information."""
    results = DDGS().text(query, max_results=3)
    results_body = [info.get("body", "").strip() for info in results if info.get("body")]
    full_result = "\n".join(results_body)
    prompt = (
    "Analyze the following search results carefully and extract the most relevant information."
    "Provide a concise and accurate answer to the given query."
    "\n\n=== Search Results ===\n"
    f"{full_result}\n"
    "=====================\n"
    f"Query: {query}\n"
    "Your Response:"
    )
    answer = send_to_ai(prompt)
    return answer 

