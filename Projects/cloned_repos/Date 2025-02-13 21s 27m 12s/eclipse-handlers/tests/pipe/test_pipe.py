import pytest
from eclipse.agent import Agent
from eclipse.eclipsepipe import EclipsePipe
from eclipse.engine import Engine
from eclipse.io import IOConsole
from eclipse.llm import LLMClient
from eclipse.prompt import PromptTemplate

from eclipse_handlers.ecommerce.amazon import AmazonHandler
from eclipse_handlers.ecommerce.flipkart import FlipkartHandler

"""
Run Pytest:  

   1. pytest -s --log-cli-level=INFO tests/pipe/test_pipe.py::TestIOConsolePipe::test_ecom_pipe
   
"""


@pytest.fixture
def agent_client_init() -> dict:
    # llm_config = {'model': 'gpt-4-turbo-2024-04-09', 'llm_type': 'openai'}
    llm_config = {"model": "mistral.mistral-large-2402-v1:0", "llm_type": "bedrock"}
    llm_client: LLMClient = LLMClient(llm_config=llm_config)
    response = {"llm": llm_client, "llm_type": "openai"}
    return response


class TestIOConsolePipe:

    async def test_ecom_pipe(self, agent_client_init: dict):
        llm_client: LLMClient = agent_client_init.get("llm")
        amazon_ecom_handler = AmazonHandler(country="IN")
        flipkart_ecom_handler = FlipkartHandler()
        prompt_template = PromptTemplate()
        amazon_engine = Engine(
            handler=amazon_ecom_handler, llm=llm_client, prompt_template=prompt_template
        )
        flipkart_engine = Engine(
            handler=flipkart_ecom_handler,
            llm=llm_client,
            prompt_template=prompt_template,
        )
        ecom_agent = Agent(
            goal="Get me the best search results",
            role="You are the best product searcher",
            llm=llm_client,
            prompt_template=prompt_template,
            engines=[[amazon_engine, flipkart_engine]],
        )
        pipe = EclipsePipe(
            agents=[ecom_agent],
        )
        io = IOConsole(
            read_phrase="\n\n\nEnter your query here:\n\n=>",
            write_phrase="\n\n\nYour result is =>\n\n",
        )
        while True:
            input_instruction = await io.read()
            result = await pipe.flow(query_instruction=input_instruction)
            await io.write(result)
