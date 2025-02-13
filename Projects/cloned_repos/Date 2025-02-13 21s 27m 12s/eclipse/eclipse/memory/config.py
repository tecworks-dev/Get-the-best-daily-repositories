import os
from pathlib import Path

from pydantic import BaseModel, Field

from eclipse.llm import LLMClient
from eclipse.vector_stores.base import BaseVectorStore


def _db_path():
    _db_dir = os.environ.get("Eclipse_MEMORY_DIR")
    if not _db_dir:
        return ":memory:"
    else:
        _db_dir = Path(_db_dir)
        return _db_dir / "history.db"


class MemoryConfig(BaseModel):
    vector_store: BaseVectorStore = Field(
        description="Configuration for the vector store",
        default=None,
    )

    db_path: str = Field(
        description="Path to the history database",
        default=_db_path(),
    )

    llm_client: LLMClient = Field(
        description="Configuration for the LLM",
        default=None,
    )

    class Config:
        arbitrary_types_allowed = True
