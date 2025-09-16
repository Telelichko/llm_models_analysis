from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base_adapter import BaseLLMAdapter
import config


class Phi3Adapter(BaseLLMAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate response
        generation_args = {
            "max_new_tokens": config.GENERATION_CONFIG.get("max_tokens", 2048),
            "temperature": config.GENERATION_CONFIG.get("temperature", 0.1),
            "top_p": config.GENERATION_CONFIG.get("top_p", 1.0),
            "do_sample": True
        }
        generation_args.update(kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_args)

        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
