import logging
import asyncio
from typing import List, Union
from garak import _config
from garak.generators.base import Generator
import importlib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from adapters.adapter import register_adapters
from adapters.adapter_types import AdapterType
from adapters.adapter_factory import AdapterFactory

class GenezioAgent(Generator):
    """
    Generator for interacting with any genezio agent
    """

    generator_family_name = "genezio"

    COMPANY_NAME_ENV_VAR = "GENEZIO_COMPANY_NAME"
    ADAPTER_NAME_ENV_VAR = "GENEZIO_ADAPTER_NAME"
    ADAPTER_TYPE_ENV_VAR = "GENEZIO_ADAPTER_TYPE"
    KB_TOKEN_ENV_VAR = "GENEZIO_KB_API_KEY"
    KB_ENV_VAR = "GENEZIO_KB_ENDPOINT"
    KB_IDS_ENV_VAR = "GENEZIO_KB_IDS"

    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS
    supports_multiple_generations = False

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root)

        # Ensure company name is a string
        self.company_name = os.getenv(self.COMPANY_NAME_ENV_VAR, "").strip()
        if not self.company_name:
            raise ValueError(f"The {self.COMPANY_NAME_ENV_VAR} environment variable is required.")
        
        self.adapter_name = os.getenv(self.ADAPTER_NAME_ENV_VAR, "").strip()
        if not self.adapter_name:
            raise ValueError(f"The {self.ADAPTER_NAME_ENV_VAR} environment variable is required.")

        # Ensure optional adapter type is a string, defaulting to HTTP if it is not present.
        adapter_type_str = os.getenv(self.ADAPTER_TYPE_ENV_VAR, "").strip()
        try:
            self.adapter_type = AdapterType(adapter_type_str)
        except ValueError:
            self.adapter_type = AdapterType.HTTP

        # Ensure API token is a string
        self.token = os.getenv(self.KB_TOKEN_ENV_VAR, "").strip()
        if not self.token:
            raise ValueError(f"The {self.KB_TOKEN_ENV_VAR} environment variable is required.")

        # Ensure knowledge base URL is a string
        self.knowledge_base_url = os.getenv(self.KB_ENV_VAR, "").strip()
        if not self.knowledge_base_url:
            raise ValueError(f"The {self.KB_ENV_VAR} environment variable is required.")

        # Ensure knowledge base IDs are stored as a list of strings
        kb_ids_raw = os.getenv(self.KB_IDS_ENV_VAR, "").strip()
        if not kb_ids_raw:
            raise ValueError(f"The {self.KB_IDS_ENV_VAR} environment variable is required.")

        self.knowledge_base_ids = [kb_id.strip() for kb_id in kb_ids_raw.split(",") if kb_id.strip()]
        if not self.knowledge_base_ids:
            raise ValueError(f"The {self.KB_IDS_ENV_VAR} environment variable must contain at least one valid ID.")

        self.client = None

    async def __setup(self):
        if self.client == None:
            await register_adapters()

        self.client = AdapterFactory.create_adapter(self.company_name, self.adapter_name, self.adapter_type, self.token, self.knowledge_base_ids, self.knowledge_base_url)
        await self.client.setup()

    async def _call_model_async(self, prompt: str, generations_this_call: int = 1) -> List[Union[str, None]]:
            await self.__setup()
            if self.client is None:
                raise RuntimeError("MockClient is not initialized.")
            response = await self.client.send(message=prompt)
            return response if isinstance(response, list) else [response]

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[Union[str, None]]:
        try:
            return asyncio.run(self._call_model_async(prompt, generations_this_call))
        except RuntimeError as e:
            logging.error(f"Error during MockClient call: {e}")
            return [None]

class MockAgent(GenezioAgent):
    """
    Custom Generator for interacting with MockClient, a custom ChatGPT agent.
    """

    def __init__(self, name="", config_root=_config):
        super().__init__(name, config_root)

    async def __setup(self):
        await super().__setup()

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[Union[str, None]]:
        try:
            return asyncio.run(self._call_model_async(prompt, generations_this_call))
        except RuntimeError as e:
            logging.error(f"Error during MockClient call: {e}")
            return [None]


DEFAULT_CLASS = "MockAgent"
