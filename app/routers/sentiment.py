import os
from fastapi import FastAPI, Response, status, HTTPException, APIRouter, Header 
import app.schemas as schemas
from datetime import datetime
from typing import Union, List, Dict, Set, Optional, Any, AsyncGenerator
from app.utils import has_valid_keys
import json
from contextlib import asynccontextmanager
from app.dependencies import (
    get_openai_handler,
    OpenAIHandlerDependency
)

MAX_RETRIES = 3  # Maximum number of retries for AI model calls

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Manage the application's lifespan.

    Initializes resources when the application starts and cleans up when it shuts down.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        AsyncGenerator[None, Any]: An asynchronous generator for the application's lifespan.
    """

    get_openai_handler()
    try:
        yield
    finally:
        # Clean up resources if necessary
        pass

router = APIRouter(
                    prefix = "/sentiment",
                    tags = ['Sentiment']
)

@router.post("/v1", status_code=status.HTTP_201_CREATED, response_model=schemas.Output)
async def sentiment(
    input: schemas.Input, 
    handler: OpenAIHandlerDependency, 
    Authorization: str = Header()
    ):

    """Endpoint to perform sentiment analysis on chat messages.

    This function processes the input request, checks authorization, and performs sentiment analysis using an OpenAI handler. If the previous sentiment is 'High Risk' and the time gap between the current and previous analysis is less than 12 hrs then we output 'High Risk' only without invoking the AI Model.

    Args:
        input (schemas.InputRequest): The input data containing messages and previous analysis.
        handler (BaseOpenAIHandler): The OpenAI handler injected via dependency injection.
        Authorization (str): Authorization header token.

    Returns:
        dict: A dictionary containing the analysis results and metadata.

    Raises:
        HTTPException: If the user is not authorized or if the server is not ready.
    """

    if Authorization != 'x':
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authorized")
    
    # Use handler that is injected by Depends
    if not handler:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server is not ready")
    
    # Convert the input data to a dictionary
    input_dict = input.model_dump(mode='json')
    message = input_dict['message']
    retries = 0

    while retries < MAX_RETRIES:
        try:
            # Send the chat thread to the OpenAI handler for sentiment analysis
            result = await handler.send_request_to_chatgpt(message, retries)

            # Parse the AI model's response
            result_json = json.loads(result)

            # Check if keys are present in the JSON response
            has_valid_keys(result_json)

        except json.JSONDecodeError:
            retries += 1  # Increment retry count and try again
            continue

        except KeyError:
            retries += 1
            continue

        else:
            # All retries failed; handle the error appropriately
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get valid response from AI model after retries"
            )

    # Return the analysis results along with request metadata
    return {
        "message": input_dict["message"],
        "analysis": result_json
    }        

@router.get("/wd")
def get_working_directory_path():
    print("Current working directory:", os.getcwd())
    return {"working_directory":os.getcwd()}
