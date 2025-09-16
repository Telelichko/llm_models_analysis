from gigachat import GigaChat
from .base_adapter import BaseLLMAdapter
import config


class GigaChatAdapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Authorization with credentials
        self.client = GigaChat(credentials=config.GIGACHAT_CREDENTIALS, verify_ssl_certs=False)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat(prompt)
        return response.choices[0].message.content