from typing import List

from pydantic import BaseModel, Field

from eclipse.utils.parsers.base import BaseParser


class PromptTemplate(BaseModel):

    # Required parameters

    template: str = Field(description="Prompt template")

    input_variables: List[str] | None = Field(
        description="the name of the model you want to use (e.g., gpt-3.5-turbo, gpt-4, gpt-3.5-turbo-16k-1106)",
        default=None,
    )

    return_type: BaseParser | None = Field(
        description="Return type base parser type or a string", default=None
    )
