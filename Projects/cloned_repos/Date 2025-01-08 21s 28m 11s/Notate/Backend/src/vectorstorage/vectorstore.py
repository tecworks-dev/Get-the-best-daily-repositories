
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os
import logging

home_dir = os.path.expanduser("~")
app_data_dir = os.path.join(home_dir, ".notate_data")
os.makedirs(app_data_dir, exist_ok=True)
chroma_db_path = os.path.join(app_data_dir, "chroma_db")
logger = logging.getLogger(__name__)


def get_vectorstore(api_key: str, collection_name: str, use_local_embeddings: bool = False, local_embedding_model: str = "granite-embedding:278m"):
    try:
        if use_local_embeddings or api_key is None:
            print(f"Using local embedding model: {local_embedding_model}")
            embeddings = OllamaEmbeddings(model=local_embedding_model)
        else:
            print(f"Using OpenAI embedding model")
            embeddings = OpenAIEmbeddings(api_key=api_key)

        return Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    except Exception as e:
        print(f"Error getting vectorstore: {str(e)}")
        return None
