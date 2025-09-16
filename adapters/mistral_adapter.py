import requests
from .base_adapter import BaseLLMAdapter
import config


class MistralAdapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Ollama typically runs on localhost:11434
        self.base_url = "http://localhost:11434"

    def generate(self, prompt: str, **kwargs) -> str:
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.GENERATION_CONFIG.get("temperature", 0.1),
                "top_p": config.GENERATION_CONFIG.get("top_p", 1.0),
                "max_tokens": config.GENERATION_CONFIG.get("max_tokens", 2048),
            }
        }

        # Send request to Ollama API
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60  # 60 seconds timeout
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        return result.get("response", "")
    