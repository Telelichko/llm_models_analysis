import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")  # For accessing gated models

# List of models to test
MODELS_TO_TEST = [
    "gpt-4-turbo",
    "claude-3-sonnet-20240229",
    "GigaChat:latest",
    "llama3",  # Via Ollama
    "mistral",  # Via Ollama
    "google/gemma-2-9b",  # Via Hugging Face
    "deepseek-ai/deepseek-llm-67b",  # Via Hugging Face
    "Qwen/Qwen2-7B-Instruct",  # Via Hugging Face
    "microsoft/Phi-3-mini-4k-instruct"  # Via Hugging Face
]

# Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.1,    # For reproducible results
    "max_tokens": 2048,
    "top_p": 1.0,
}

# Data paths
GOLDEN_DATASET_PATH = "data/golden_dataset.jsonl"
RESULTS_PATH = "results/outputs.jsonl"
