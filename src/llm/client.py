import os
import httpx
import json
from typing import Dict, Any, List

class LLMClient:
    """
    A generic client for interacting with a large language model.
    Currently configured for Anthropic's Claude.
    """

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or "YOUR_ANTHROPIC_API_KEY"
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY":
            print("Warning: ANTHROPIC_API_KEY is not configured. LLM calls will be mocked.")

    async def query(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and returns the structured JSON response.
        """
        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY":
            # Return a mock response if the API key is not available.
            return {"mock_response": "LLM client is not configured with an API key."}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, headers=headers, json=data, timeout=30.0)
                response.raise_for_status()

                response_data = response.json()

                # Extract token usage
                usage_data = response_data.get("usage", {})
                input_tokens = usage_data.get("input_tokens", 0)
                output_tokens = usage_data.get("output_tokens", 0)

                # Extract the text content from the response
                response_text = response_data["content"][0]["text"]

                # The response from the LLM is often a JSON string, so we parse it.
                # If it's not, we return it as a string inside a dict.
                final_content = {}
                try:
                    final_content = json.loads(response_text)
                except json.JSONDecodeError:
                    final_content = {"text_response": response_text}

                # Combine the parsed content with usage data
                return {
                    "content": final_content,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                }

        except httpx.HTTPStatusError as e:
            print(f"LLM Client API Error: {e.response.status_code} - {e.response.text}")
            return {"error": "API error", "details": e.response.text}
        except Exception as e:
            print(f"An unexpected error occurred with the LLM Client: {e}")
            return {"error": "Unexpected error", "details": str(e)}
