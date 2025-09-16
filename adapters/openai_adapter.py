from openai import OpenAI
from .base_adapter import BaseLLMAdapter
import config


class OpenAIAdapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **{**config.GENERATION_CONFIG, **kwargs}
        )
        return response.choices[0].message.content