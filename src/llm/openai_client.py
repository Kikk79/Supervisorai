import os
import json
from typing import Dict, Any
from openai import AsyncOpenAI

from .base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    """
    A client for interacting with OpenAI's models (e.g., GPT-4).
    """

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        super().__init__(api_key, model)
        if not self.api_key or self.api_key.startswith("YOUR_"):
            print("Warning: OPENAI_API_KEY is not configured. OpenAI LLM calls will be mocked.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)

    async def query(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Sends a prompt to the OpenAI API and returns the structured response.
        """
        if not self.client:
            return {
                "content": {"mock_response": "OpenAI client is not configured with an API key."},
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=max_tokens,
            )

            response_text = chat_completion.choices[0].message.content
            usage = chat_completion.usage

            final_content = {}
            try:
                final_content = json.loads(response_text)
            except (json.JSONDecodeError, TypeError):
                final_content = {"text_response": response_text}

            return {
                "content": final_content,
                "usage": {
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens
                }
            }
        except Exception as e:
            print(f"An unexpected error occurred with the OpenAI Client: {e}")
            return {"content": {"error": "OpenAI API error", "details": str(e)}, "usage": {"input_tokens": 0, "output_tokens": 0}}
