from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class OllamaError(Exception):
    """Raised when the Ollama service cannot produce a valid response."""
    pass


@dataclass
class OllamaClient:
    """Client for sending prompts to a local Ollama server.

    This client is intentionally lightweight and does not add any external
    HTTP dependencies beyond the Python standard library.
    """

    server_url: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    timeout: int = 120
    max_tokens: int = 150
    temperature: float = 0.0

    def generate(self, prompt: str, model_name: Optional[str] = None) -> str:
        """Send a prompt to Ollama and return the model response as text.

        Args:
            prompt: The full prompt to send to Ollama.
            model_name: Optional override for the model name.

        Returns:
            The text response from the Ollama model.

        Raises:
            OllamaError: If the request fails or the response is malformed.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        payload = {
            "model": model_name or self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body = json.dumps(payload).encode("utf-8")

        request = Request(
            url=f"{self.server_url.rstrip('/')}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                if response.status != 200:
                    raise OllamaError(
                        f"Ollama server returned HTTP {response.status}: {raw}"
                    )
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise OllamaError(
                f"Ollama server returned HTTP {exc.code}: {message}"
            ) from exc
        except URLError as exc:
            raise OllamaError(
                f"Unable to reach Ollama server at {self.server_url}: {exc.reason}"
            ) from exc
        except Exception as exc:
            raise OllamaError("Unexpected error while calling Ollama.") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc

        return self._extract_text(decoded)

    @staticmethod
    def _extract_text(response_data: dict) -> str:
        """Extract text from the Ollama response payload."""
        if not isinstance(response_data, dict):
            raise OllamaError("Ollama response payload is not a JSON object.")

        if "completion" in response_data and isinstance(response_data["completion"], str):
            return response_data["completion"].strip()

        if "output" in response_data and isinstance(response_data["output"], str):
            return response_data["output"].strip()

        choices = response_data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = first.get("text") or first.get("message", {}).get("content")
                if isinstance(text, str):
                    return text.strip()

        raise OllamaError("Ollama response did not contain a text completion.")
