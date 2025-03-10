import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import math 

HISTORY_FILE_PATH = "./DATA/chat_history.json"
AI_MODEL = "granite3.1-dense:2b"
EMBEDDING_MODEL = "nomic-embed-text"
SCORE_THRESHOLD = 0.4  # Adjust this threshold as needed


def load_chat_history():
    if Path(HISTORY_FILE_PATH).exists():
        with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def save_chat_history(history):
    with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def ask_ai_importance(prompt: str) -> bool:
    """Ask AI if the chat message is important."""
    llm = ChatOllama(model=AI_MODEL, temperature=0, max_token=50)

    system_prompt = """
    You are an AI that determines if a chat message is important. 
    Prioritize messages containing personal information (e.g., names, addresses, contact details).
    Also, consider messages that express strong emotions, urgent requests, or critical information.
    Answer only 'yes' or 'no'.

    Examples:
    - "My address is 123 Main St." -> yes
    - "I need help urgently!" -> yes
    - "where i live" -> no
    - "what i do" -> no
    - "what is my name" -> no
    - "What's the weather?" -> no
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Is this conversation important? {prompt}"},
    ]

    response = llm.invoke(messages)
    return "yes" in response.content.strip().lower()


def store_important_chat(prompt: str, response: str):
    """Store chat in history if AI deems it important."""
    if ask_ai_importance(prompt):
        history = load_chat_history()
        history.append({"user": prompt, "assistant": response})
        save_chat_history(history)


def distance_to_similarity_exp(distance, scale=1.0):
    """Converts distance to similarity using an exponential transformation."""
    return math.exp(-distance * scale)


def semantic_search(query: str):
    history = load_chat_history()
    if not history:
        return []

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Create FAISS vector store:
    vectorstore = FAISS.from_texts(
        texts=[item["user"] for item in history if "user" in item],
        embedding=embedding,
    )

    # Perform similarity search with scores:
    results_with_scores = vectorstore.similarity_search_with_score(query, k=7)

    filtered_results = []
    for result, score in results_with_scores:
        similarity_score = distance_to_similarity_exp(score)
        if similarity_score >= SCORE_THRESHOLD:
            for item in history:
                if "user" in item and item["user"] == result.page_content:
                    filtered_results.append(item)

    return filtered_results


def message_management(query):
    messages = [
        {"role": "system", "content": "You are an AI chatbot that maintains conversation context and provides relevant responses."}
    ]

    relevant_chats = semantic_search(query)
    if relevant_chats:
        for chat in relevant_chats:
            messages.append({"role": "user", "content": chat["user"]})
            messages.append({"role": "assistant", "content": chat["assistant"]})

    messages.append({"role": "user", "content": query})
    return messages


def personal_chat_ai(first_query: str, max_token: int = 2000):
    """Chat system with persistent history and semantic retrieval."""
    try:
        query = first_query
        messages = message_management(query)

        llm = ChatOllama(model=AI_MODEL, temperature=0.3, max_token=max_token)

        while True:
            store = len(messages) < 5
            response_stream = llm.stream(messages)
            response_content = ""

            print("AI:", end=" ")
            for chunk in response_stream:
                text = chunk.content
                print(text, end="", flush=True)
                response_content += text
            print()

            if store:
                store_important_chat(query, response_content)

            query = input("YOU: ")
            if query.lower() in {"exit", "end"}:
                break

            messages = message_management(query)

    except Exception as e:
        print(f"An error occurred: {e}")
    return None 



# Initialize chat with user input
if __name__ == "__main__":
    print("AI Chat Initialized. Type 'exit' to stop.")
    first_query = input("YOU: ")
    personal_chat_ai(first_query)
    print("Chat session ended.")
