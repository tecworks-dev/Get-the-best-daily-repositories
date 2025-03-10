


import json 
from langchain_ollama import ChatOllama

#client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")



def send_to_ai(prompt:str , max_token:int = 2000 , model:str = "granite3.1-dense:2b") -> str:
    """"Handle creative prompts like jokes or stories."""
    try:
        llm = ChatOllama(
        model=model,
        temperature=0.3,
        max_token = max_token
        )
        messages = [
            {"role": "system", "content": "You are an intelligent AI system. Understand the user Query carefully and provide the most relevant Answer."},
            {"role": "user", "content": str(prompt)}
        ]

        response = llm.invoke(messages)
        return response.content 

    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    query = input("QUERY: ")
    response  = send_to_ai(query)
    print(response)
    

