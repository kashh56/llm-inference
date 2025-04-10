from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Provider configurations
PROVIDER_CONFIGS = {
    "Google": {
        "models": [
            "gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash",
            "gemini-1.5-pro",  # Latest Gemini 1.5 Pro model
            "gemini-1.5-pro-latest",  # Latest version with continuous updates
            "gemini-1.5-flash",  # Fast, efficient version
            "gemini-2.0-flash"  # Latest Gemini 2.0 with image generation
        ],
        "api_key": os.getenv("GOOGLE_API_KEY"),
    },
    "Cohere": {
        "models": [
            "command-a-03-2025",  # Latest and most performant model with 256K context
            "command-r7b-12-2024",  # Small, fast model for RAG and tool use with 128K context
            "command-r-plus-04-2024",  # High quality model for complex RAG workflows
            "command-r-08-2024",  # Updated Command R model from August 2024
            "command-r-03-2024",  # Base Command R model with 128K context
            "command",  # Standard command model with 4K context
            "command-light",  # Faster, smaller version with 4K context
            "command-nightly"  # Experimental version, not for production
        ],
        "api_key": os.getenv("COHERE_API_KEY"),
    },
    "Groq": {
        "models": [
            "gemma2-9b-it",  # Google's Gemma 2 model with 8K context
            "llama-3.3-70b-versatile",  # Latest Llama 3.3 with 128K context
            "llama-3.1-8b-instant",  # Fast Llama 3.1 with 128K context
            "llama-guard-3-8b",  # Content safety model
            "llama3-70b-8192",  # Llama 3 70B with 8K context
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama3-8b-8192",  # Llama 3 8B with 8K context
        ],
        "api_key": os.getenv("GROQ_API_KEY"),
    },
    "Together": {
        "models": [
            # Latest recommended models with context windows
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",      # 131072 context, FP8 quantized (recommended default)
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",  # 524288 context, FP8
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",    # 327680 context, FP16
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", # 131072 context, FP8
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", # 130815 context, FP8
            "mistralai/Mistral-Small-24B-Instruct-2501",    # 32768 context, FP16
            "mistralai/Mixtral-8x7B-Instruct-v0.1",         # 32768 context, FP16
            "deepseek-ai/DeepSeek-R1",                      # 128000 context, FP8
            "Qwen/Qwen2.5-Coder-32B-Instruct"              # 32768 context, FP16
        ],
        "api_key": os.getenv("TOGETHER_API_KEY"),
        # Default settings based on Together AI's recommended configuration
        "default_settings": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["<human>:", "<bot>:"]
        },
        # Inference configuration including prompt formatting and retry logic
        "inference_config": {
            "prompt_format": "<human>: {message}\n<bot>: ",
            "retry_on_error": True,
            "max_retries": 3,
            "retry_delay": 1,
            "streaming": True,
            "verbose": True
        }
    },
    "HuggingFace": {
        "models": [
            "gpt2",                        # Base GPT-2, most reliable
            "gpt2-medium",                 # Larger GPT-2
            "distilgpt2"                   # Distilled GPT-2, faster
        ],
        "api_key": os.getenv("HUGGINGFACE_API_KEY"),
        "inference_config": {
            "task": "text-generation",
            "model_kwargs": {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }
    },
    "Cerebras": {
        "models": [
            "llama-3.3-70b",  # Latest Llama 3.3 70B model
            "llama-3.1-8b",   # Efficient Llama 3.1 8B model
            "llama-4-scout-17b-16e-instruct"  # Llama 4 Scout with 16E context
        ],
        "api_key": os.getenv("CEREBRAS_API_KEY"),

    },
}

def validate_api_keys() -> Dict[str, bool]:
    """Validate API keys and return status for each provider."""
    validation_status = {}
    
    for provider, config in PROVIDER_CONFIGS.items():
        api_key = config.get("api_key")
        if not api_key:
            validation_status[provider] = False
        else:
            # Basic validation (non-empty and proper format)
            validation_status[provider] = bool(api_key.strip())
    
    return validation_status

def get_available_providers() -> List[str]:
    """Return list of providers with valid API keys."""
    validation_status = validate_api_keys()
    return [
        provider
        for provider, is_valid in validation_status.items()
        if is_valid
    ]

def get_models_for_provider(provider: str) -> List[str]:
    """Return available models for a given provider."""
    return PROVIDER_CONFIGS.get(provider, {}).get("models", [])

# Add helper functions for message formatting
def format_together_message(message: str, config: Dict) -> str:
    """Format message for Together API."""
    return config["inference_config"]["prompt_format"].format(message=message)

def get_provider_config(provider: str) -> Dict:
    """Get full configuration for a provider."""
    config = PROVIDER_CONFIGS.get(provider, {}).copy()  # Make a copy to avoid modifying original
    

    
    return config

def handle_together_error(error_code: int) -> str:
    """Handle Together API errors and return appropriate message."""
    error_messages = {
        400: "Invalid request format. Please check your message format.",
        401: "Invalid API key. Please check your Together API key.",
        402: "Account has reached spending limit.",
        403: "Request exceeds model context length.",
        404: "Model not found or invalid endpoint.",
        429: "Too many requests. Please try again later.",
        500: "Together server error. Please try again later.",
        503: "Service overloaded. Please try again later.",
    }
    return error_messages.get(error_code, "Unknown error occurred. Please try again.") 