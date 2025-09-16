from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .gigachat_adapter import GigaChatAdapter
from .llama_adapter import LlamaAdapter
from .mistral_adapter import MistralAdapter
from .gemma_adapter import GemmaAdapter
from .deepseek_adapter import DeepSeekAdapter
from .qwen_adapter import QwenAdapter
from .phi3_adapter import Phi3Adapter

def get_adapter(model_name: str):
    """Returns the appropriate adapter for the model name"""
    if model_name.startswith("gpt-"):
        return OpenAIAdapter(model_name)
    elif model_name.startswith("claude-"):
        return AnthropicAdapter(model_name)
    elif model_name.startswith("GigaChat"):
        return GigaChatAdapter(model_name)
    elif model_name.startswith("llama"):
        return LlamaAdapter(model_name)
    elif model_name.startswith("mistral"):
        return MistralAdapter(model_name)
    elif model_name.startswith("gemma"):
        return GemmaAdapter(model_name)
    elif model_name.startswith("deepseek"):
        return DeepSeekAdapter(model_name)
    elif model_name.startswith("Qwen"):
        return QwenAdapter(model_name)
    elif model_name.startswith("phi"):
        return Phi3Adapter(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
