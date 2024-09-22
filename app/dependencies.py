import os
from functools import lru_cache
from fastapi import Depends
from typing import Annotated
from dotenv import load_dotenv
from core.handler import BaseOpenAIHandler, AsyncOpenAIHandler

load_dotenv()

# Define singleton handler to use it as an app dependency
@lru_cache
def get_openai_handler() -> BaseOpenAIHandler:
    """Retrieve a singleton instance of the OpenAI handler.

    Loads the OpenAI API key from environment variables and initializes
    an asynchronous OpenAI handler with the specified system prompt file.

    Returns:
        BaseOpenAIHandler: An instance of the OpenAI handler.
    """

    api_key = os.environ.get("OPENAI_API_KEY")
    return AsyncOpenAIHandler(api_key=api_key, filename='core//system_prompt.txt')

# Dependency injection for FastAPI routes to use the OpenAI handler
OpenAIHandlerDependency = Annotated[BaseOpenAIHandler, Depends(get_openai_handler)]