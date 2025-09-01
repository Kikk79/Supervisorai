import os
import json
from typing import Dict, Any, List
import importlib

from .base import BaseLLMClient

class LLMManager:
    """
    Manages the lifecycle of multiple LLM clients based on a configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._clients: Dict[str, BaseLLMClient] = {}
        self._load_clients()

    def _load_clients(self):
        """
        Instantiates LLM clients based on the provided configuration.
        """
        for name, client_config in self.config.items():
            try:
                module_path = client_config["module_path"]
                class_name = client_config["class_name"]
                api_key_env_var = client_config["api_key_env_var"]
                model = client_config["model"]

                # Dynamically import the client class
                module = importlib.import_module(module_path)
                client_class = getattr(module, class_name)

                # Get API key from environment
                api_key = os.environ.get(api_key_env_var)

                # Instantiate the client
                self._clients[name] = client_class(api_key=api_key, model=model)
                print(f"Successfully loaded LLM client: {name}")

            except (ImportError, AttributeError, KeyError) as e:
                print(f"Error loading LLM client '{name}': {e}. This client will be unavailable.")

    def get_client(self, name: str) -> BaseLLMClient:
        """
        Retrieves an instantiated LLM client by its configuration name.

        Raises:
            ValueError: If the requested client is not available or not configured.
        """
        client = self._clients.get(name)
        if not client:
            raise ValueError(f"LLM client '{name}' is not configured or failed to load.")
        return client

    def list_clients(self) -> List[str]:
        """Returns a list of successfully loaded client names."""
        return list(self._clients.keys())
