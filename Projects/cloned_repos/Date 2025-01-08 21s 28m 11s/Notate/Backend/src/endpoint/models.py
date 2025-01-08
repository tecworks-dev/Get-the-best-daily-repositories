from pydantic import BaseModel
from typing import Optional, Dict, Any


class EmbeddingRequest(BaseModel):
    file_path: str
    api_key: Optional[str] = None
    collection: int
    collection_name: str
    user: int
    metadata: Optional[Dict[str, Any]] = None
    is_local: Optional[bool] = False
    local_embedding_model: Optional[str] = "granite-embedding:278m"


class VectorStoreQueryRequest(BaseModel):
    query: str
    collection: Optional[int] = None
    collection_name: str
    user: int
    api_key: Optional[str] = None
    top_k: int = 5
    is_local: Optional[bool] = False
    local_embedding_model: Optional[str] = "granite-embedding:278m"
    prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.5
    max_completion_tokens: Optional[int] = 2048
    top_p: Optional[float] = 1
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    is_ooba: Optional[bool] = False
    character: Optional[str] = None
    is_ollama: Optional[bool] = False


class YoutubeTranscriptRequest(BaseModel):
    url: str
    user_id: int
    collection_id: int
    username: str
    collection_name: str
    api_key: Optional[str] = None
    is_local: Optional[bool] = False
    local_embedding_model: Optional[str] = "granite-embedding:278m"


class DeleteCollectionRequest(BaseModel):
    collection_id: int
    collection_name: str
    is_local: Optional[bool] = False
    api_key: Optional[str] = None


class WebCrawlRequest(BaseModel):
    base_url: str
    max_workers: int
    collection_name: str
    collection_id: int
    user_id: int
    user_name: str
    api_key: Optional[str] = None
    is_local: Optional[bool] = False
    local_embedding_model: Optional[str] = "granite-embedding:278m"


class QueryRequest(BaseModel):
    input: str
    prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    collection_name: Optional[str] = None
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.5
    max_completion_tokens: Optional[int] = 2048
    top_p: Optional[float] = 1
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    is_local: Optional[bool] = False
    is_ooba: Optional[bool] = False
    local_embedding_model: Optional[str] = "granite-embedding:278m"
    character: Optional[str] = None
    is_ollama: Optional[bool] = False
