import os
import httpx
import json
import base64
import mimetypes
from typing import Dict, Any, Optional

from .base import BaseLLMClient

class AnthropicClient(BaseLLMClient):
    """
    A client for interacting with Anthropic's Claude models.
    """

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__(api_key, model)
        self.api_url = "https://api.anthropic.com/v1/messages"
        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY":
            print("Warning: ANTHROPIC_API_KEY is not configured. LLM calls will be mocked.")

    async def query(self, prompt: str, max_tokens: int = 1024, image_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a prompt (and optional image) to the Anthropic API and returns the structured response.
        """
        if not self.api_key or self.api_key == "YOUR_ANTHROPIC_API_KEY":
            return {
                "content": {"mock_response": "LLM client is not configured with an API key."},
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        messages = []
        if image_url:
            try:
                # Download the image data
                async with httpx.AsyncClient() as client:
                    image_response = await client.get(image_url, follow_redirects=True)
                    image_response.raise_for_status()

                image_data = image_response.content
                # Encode the image in base64
                base64_image = base64.b64encode(image_data).decode("utf-8")
                # Guess the media type
                media_type, _ = mimetypes.guess_type(image_url)
                if not media_type:
                    media_type = "image/jpeg" # Fallback

                # Construct the multi-modal message content
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                })
            except Exception as e:
                print(f"Error processing image URL {image_url}: {e}")
                return {"content": {"error": "Image processing error", "details": str(e)}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        else:
            # Standard text-only message
            messages.append({"role": "user", "content": prompt})


        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, headers=headers, json=data, timeout=60.0) # Increased timeout for images
                response.raise_for_status()

                response_data = response.json()
                usage_data = response_data.get("usage", {})
                input_tokens = usage_data.get("input_tokens", 0)
                output_tokens = usage_data.get("output_tokens", 0)
                response_text = response_data.get("content", [{}])[0].get("text", "")

                final_content = {}
                try:
                    final_content = json.loads(response_text)
                except (json.JSONDecodeError, TypeError):
                    final_content = {"text_response": response_text}

                return {
                    "content": final_content,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                }
        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            print(f"LLM Client API Error: {e.response.status_code} - {error_details}")
            return {"content": {"error": "API error", "details": error_details}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        except Exception as e:
            print(f"An unexpected error occurred with the LLM Client: {e}")
            return {"content": {"error": "Unexpected error", "details": str(e)}, "usage": {"input_tokens": 0, "output_tokens": 0}}
