from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    """
    An abstract base class for a generic LLM client.
    Defines the interface that all specific LLM clients must implement.
    """

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def query(self, prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and returns the structured response.

        The response dictionary should include at least:
        {
            "content": { ... parsed JSON content or text_response ... },
            "usage": {
                "input_tokens": ...,
                "output_tokens": ...
            }
        }
        """
        pass
