import logging
from typing import List, Union
from garak import _config
from garak.generators.base import Generator
import importlib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from adapters.http.mock_dev import MockClient

class MockChatGPTGenerator(Generator):
    """
    Custom Generator for interacting with MockClient, a custom ChatGPT agent.
    """

    generator_family_name = "mock"

    COMPANY_NAME_ENV_VAR = "GENEZIO_COMPANY_NAME"
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
            self.client = MockClient(self.company_name, self.token, self.knowledge_base_ids, self.knowledge_base_url)
            await self.client.setup()

    def _call_model(self, prompt: str, generations_this_call: int = 1) -> List[Union[str, None]]:
        """
        Sends the prompt to MockClient and returns the response.
        """
        import asyncio  # Required for async calls

        async def get_response() -> List[Union[str, None]]:
            await self.__setup()
            if self.client is None:
                raise RuntimeError("MockClient is not initialized.")
            response = await self.client.send(message=prompt)
            return [r if r is not None else None for r in response] if isinstance(response, list) else [response]

        try:
            return asyncio.run(get_response())
        except Exception as e:
            logging.error(f"Error during MockClient call: {e}")
            return [None]

DEFAULT_CLASS = "MockChatGPTGenerator"
