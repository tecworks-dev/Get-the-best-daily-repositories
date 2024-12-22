from typing import List, Type
import llm
from .consortium import ConsortiumPlugin

# Plugin metadata
__version__ = "0.1.0"
name = "llm-consortium"
description = "Consortium plugin for managing multiple LLM models with arbitration"

# Register plugin classes
registry: List[Type[llm.Plugin]] = [ConsortiumPlugin]
