from src.llms.messages.formMessages import form_messages
from src.endpoint.models import QueryRequest
from src.llms.providers.ooba import ooba_query
from src.llms.providers.openai import openai_query
from src.llms.providers.ollama import ollama_query

from typing import Optional


def llm_query(data: QueryRequest, api_key: Optional[str] = None):
    try:

        messages = form_messages(data)

        if data.is_ooba:
            return ooba_query(data, messages)

        if data.is_ollama is None:
            return ollama_query(data, messages)

        else:
            return openai_query(data, api_key, messages)

    except Exception as e:
        print(f"Error in llm_query: {str(e)}")
        raise e
