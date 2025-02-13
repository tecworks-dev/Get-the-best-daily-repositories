# Eclipse: The Open-Source AI Framework  

## **✨ Introduction**  
Eclipse is a next-generation **open-source AI framework** built for **autonomous multi-agent applications**, designed to push the boundaries of **Artificial General Intelligence (AGI)** and **Artificial Super Intelligence (ASI)**.  

---

## **🔍 Key Features**  
- **🛠️ Open-Source AI Framework** - Tailored for **multi-agent autonomy**.  
- **⚡ Fast & Lightweight** - **Minimal dependencies** for streamlined performance.  
- **📂 Contextual Memory** - Uses **SQL & vector databases** to store & retrieve user data.  
- **🧠 AI-Driven** - Supports **LLMs, reinforcement learning, and goal-oriented agents**.  
- **💽 Extendable Handlers** - Plug into **IoT, APIs, blockchain, and more**.  
- **🌱 Scalable Multi-Agent System** - Handles **complex workflows autonomously**.  

---

## **🎉 Installation Guide**  
### **1. Install Eclipse**  
```bash  
pip install git+https://github.com/Eclipse-Hub/eclipse  
```  
Verify installation:  
```bash  
pip show eclipse  
```  
### **2. Install Eclipse Handlers**  
```bash  
pip install git+https://github.com/Eclipse-Hub/eclipse-handlers  
```  
### **3. Install Eclipse Examples**  
```bash  
pip install git+https://github.com/Eclipse-Hub/eclipse-examples  
```  

---

## **🚀 Quick Start Guide**  
### **1. Set Up Your Environment**  
```bash  
python3 -m venv venv  
source venv/bin/activate  
```  
### **2. Create an AI Agent**  
Modify `pipe.py` to define AI logic & automation.  
### **3. Run Your AI System**  
```bash  
python pipe.py  
```  
Your **Eclipse AI Agents** are now live! 

---

## **🌟 Example: Eclipse (E-Commerce AI)**  
This example uses **Amazon & Walmart handlers** to search for products based on user input.  

### **Pre-Requisites**  
```bash  
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxx  
export RAPID_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXX  
```  

### **Code Example**  
```python  
import asyncio

from rich import print as rprint

from eclipse.memory import Memory
from eclipse.agent import Agent
from eclipse.engine import Engine
from eclipse.llm import LLMClient
from eclipse.eclipsepipe import EclipsePipe
from eclipse.pipeimpl.iopipe import IOPipe
from eclipse.prompt import PromptTemplate
from eclipse_handlers.ecommerce.amazon import AmazonHandler
from eclipse_handlers.ecommerce.walmart import WalmartHandler


async def main():
    """
    Launches the e-commerce pipeline console client for processing requests and handling data.
    """

    # LLM Configuration
    llm_config = {'llm_type': 'openai'}
    llm_client: LLMClient = LLMClient(llm_config=llm_config)

    # Enable Memory
    memory = Memory(memory_config={"llm_client": llm_client})

    # Add Two Handlers (Tools) - Amazon, Walmart
    amazon_ecom_handler = AmazonHandler()
    walmart_ecom_handler = WalmartHandler()

    # Prompt Template
    prompt_template = PromptTemplate()

    # Amazon & Walmart Engine to execute handlers
    amazon_engine = Engine(
        handler=amazon_ecom_handler,
        llm=llm_client,
        prompt_template=prompt_template
    )
    walmart_engine = Engine(
        handler=walmart_ecom_handler,
        llm=llm_client,
        prompt_template=prompt_template
    )

    # Create Agent with Amazon, Walmart Engines execute in Parallel - Search Products from user prompts
    ecom_agent = Agent(
        name='Ecom Agent',
        goal="Get me the best search results",
        role="You are the best product searcher",
        llm=llm_client,
        prompt_template=prompt_template,
        engines=[[amazon_engine, walmart_engine]]
    )

    # Pipe Interface to send it to public accessible interface (Cli Console / WebSocket / Restful API)
    pipe = EclipsePipe(
        agents=[ecom_agent],
        memory=memory
    )

    # Create IO Cli Console - Interface
    io_pipe = IOPipe(
        search_name='Eclipse Ecom',
        agentx_pipe=pipe,
        read_prompt=f"\n[bold green]Enter your search here"
    )
    await io_pipe.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        rprint("\nUser canceled the [bold yellow][i]pipe[/i]!")

```

## Environment Setup
```shell
$ python3.12 -m pip install poetry
$ cd <path-to>/eclipse
$ python3.12 -m venv venv
$ source venv/bin/activate
(venv) $ poetry install
```

## [Documentation](https://docs.eclipsehub.io/introduction)

## License

Eclipse is released under the [MIT](LICENSE) License.