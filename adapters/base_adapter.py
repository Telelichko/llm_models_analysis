import time
from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM adapters"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generates a response based on the prompt"""
        pass

    def run_with_metrics(self, prompt: str, **kwargs) -> dict:
        """Runs generation and collects metrics"""
        start_time = time.time()

        try:
            response = self.generate(prompt, **kwargs)
            end_time = time.time()

            return {
                "success": True,
                "response": response,
                "latency": end_time - start_time,
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response": None,
                "latency": end_time - start_time,
                "error": str(e)
            }
        