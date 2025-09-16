from transformers import pipeline
from .base_adapter import BaseLLMAdapter
import config


class GemmaAdapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Initialize the pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",  # Automatically uses GPU if available
            torch_dtype="auto"  # Automatically uses appropriate dtype
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # Generate text using the pipeline
        generation_args = {
            "max_new_tokens": config.GENERATION_CONFIG.get("max_tokens", 2048),
            "temperature": config.GENERATION_CONFIG.get("temperature", 0.1),
            "top_p": config.GENERATION_CONFIG.get("top_p", 1.0),
            "do_sample": True
        }
        generation_args.update(kwargs)

        outputs = self.pipeline(prompt, **generation_args)
        return outputs[0]['generated_text']
    