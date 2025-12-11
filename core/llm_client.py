import logging
import os
import time
import requests
from typing import Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for calling models via OpenRouter's chat completion API.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing env variable: OPENROUTER_API_KEY")

    def complete(self, prompt: str) -> str:
        """
        Make a chat completion call with simple retry logic.
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                logger.info(
                    "Calling model %s (attempt %s)", self.model_name, attempt + 1
                )
                res = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
                res.raise_for_status()
                data = res.json()
                logger.debug("Model %s response received", self.model_name)
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Attempt %s failed for model %s: %s",
                    attempt + 1,
                    self.model_name,
                    exc,
                )
                last_error = exc
                time.sleep(2)

        raise RuntimeError(
            f"Failed after 3 retries for model {self.model_name}: {last_error}"
        )
