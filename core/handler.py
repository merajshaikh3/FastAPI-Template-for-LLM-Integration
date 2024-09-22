import openai
from abc import ABC, abstractmethod
from app import schemas
from typing import Optional


class BaseOpenAIHandler(ABC):
    """Abstract base class for OpenAI handlers."""

    @abstractmethod
    def __init__(self, api_key, filename):
        """Initialize the handler with API key and filename."""

        pass

class AsyncOpenAIHandler(BaseOpenAIHandler):
    """Asynchronous handler for interacting with OpenAI's API."""

    def __init__(self, api_key: str, filename: str):
        """Initialize the asynchronous OpenAI handler.

        Args:
            api_key (str): The API key for OpenAI.
            filename (str): The filename containing the system prompt.
        """

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.system_prompt = self.load_system_prompt(filename)

    def load_system_prompt(self, filename: str) -> str:
        """Load the system prompt from a file.

        Args:
            filename (str): The filename of the system prompt file.

        Returns:
            str: The content of the system prompt.
        """

        try:
            with open(filename, 'r', encoding='utf-8') as file:
                system_prompt = ''
                # ensure that the txt file has a blank line at the end else last message will get skipped
                for item in file:
                    if item.strip() != '':
                        if item[0] == ' ':
                            system_prompt += '\t' + item.strip() + '\n'
                        else:
                            system_prompt += item.strip() + '\n'
                    else:
                        system_prompt += '\n'

                return system_prompt
        except FileNotFoundError:
            print(f"System prompt file not found: {filename}")
            raise
        except Exception as e:
            print(f"Error loading system prompt: {e}")
            raise


    async def send_request_to_chatgpt(self, chat: str, retry_count: int = 0) -> Optional[schemas.Output]:
        """Send a sentiment analysis request to ChatGPT.

        Args:
            chat (str): The chat thread to analyze.

        Returns:
            Optional[schemas.SentimentOutput]: The sentiment analysis result, or None if an error occurs.
        """
        
        user_prompt = f"Please do a sentiment analysis on the below movie review (Note: Output should be in JSON format)`\n{chat}"

        system_prompt = self.system_prompt if retry_count == 0 else f"The earlier response structure was incorrect, therefore I'm prompting you again. Please ensure that the output structure is correct this time. Guidelines on how to do the Sentiment Analysis and what the output structure should look like are given below:\n\n" + self.system_prompt

        try:
            # completion = self.client.chat.completions
            response = await self.client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    response_format={"type": "json_object"},
                                    messages=[
                                                {
                                                "role": "system", 
                                                "content": system_prompt
                                                },
                                                {
                                                "role": "user", 
                                                "content": user_prompt
                                                }
                                            ]
                                )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None