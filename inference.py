import time
from typing import Tuple
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere.llms import Cohere
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEndpoint
from langchain_cerebras import ChatCerebras
from langchain.schema import HumanMessage
from providers import PROVIDER_CONFIGS

load_dotenv()

def time_inference(func):
    """Decorator to measure inference time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

class InferenceProvider:
    @time_inference
    def google_inference(prompt: str, model: str) -> Tuple[str, float]:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"))
        response = llm.invoke(prompt)
        return response.content

    @time_inference
    def cohere_inference(prompt: str, model: str) -> Tuple[str, float]:
        if model.startswith("command"):
            llm = ChatCohere(model=model, cohere_api_key=os.getenv("COHERE_API_KEY"))
            response = llm.invoke([HumanMessage(content=prompt)])
        else:
            llm = Cohere(model=model, cohere_api_key=os.getenv("COHERE_API_KEY"))
            response = llm.invoke(prompt)
        return response.content

    @time_inference
    def groq_inference(prompt: str, model: str) -> Tuple[str, float]:
        llm = ChatGroq(model=model, groq_api_key=os.getenv("GROQ_API_KEY"))
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    @time_inference
    def together_inference(prompt: str, model: str) -> Tuple[str, float]:
        config = PROVIDER_CONFIGS["Together"]["default_settings"]
        llm = ChatTogether(
            model=model,
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", None),
            timeout=None,
            max_retries=config.get("max_retries", 2)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    @time_inference
    def huggingface_inference(prompt: str, model: str) -> Tuple[str, float]:
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY must be provided")
        
        config = PROVIDER_CONFIGS["HuggingFace"]["inference_config"]
            
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{model}",
            huggingfacehub_api_token=api_key,
            task=config["task"],
            temperature=config["model_kwargs"]["temperature"],
            top_p=config["model_kwargs"]["top_p"],
            do_sample=config["model_kwargs"]["do_sample"],
            max_length=config["model_kwargs"]["max_length"]
        )
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                raise ValueError(f"Model {model} not found. Please check if the model name is correct and you have access to it.")
            elif "401" in error_msg or "403" in error_msg:
                raise ValueError("Authentication failed. Please check your Hugging Face API token.")
            else:
                raise ValueError(f"HuggingFace inference failed: {error_msg}")

    @time_inference
    def cerebras_inference(prompt: str, model: str) -> Tuple[str, float]:
        llm = ChatCerebras(model=model, cerebras_api_key=os.getenv("CEREBRAS_API_KEY"))
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

def get_inference_func(provider: str):
    """Return the appropriate inference function for a provider."""
    inference_funcs = {
        "Google": InferenceProvider.google_inference,
        "Cohere": InferenceProvider.cohere_inference,
        "Groq": InferenceProvider.groq_inference,
        "Together": InferenceProvider.together_inference,
        "HuggingFace": InferenceProvider.huggingface_inference,
        "Cerebras": InferenceProvider.cerebras_inference,
    }
    return inference_funcs.get(provider) 