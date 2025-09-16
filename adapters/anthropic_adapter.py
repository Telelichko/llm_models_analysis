import anthropic
from .base_adapter import BaseLLMAdapter
import config


class AnthropicAdapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=config.GENERATION_CONFIG["max_tokens"],
            temperature=config.GENERATION_CONFIG["temperature"],
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text