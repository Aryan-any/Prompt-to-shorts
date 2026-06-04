from typing import Dict
from dotenv import load_dotenv
import os

def get_llm_config(base_config: Dict) -> Dict:
    load_dotenv()
    
    # Get LLM type from environment or config
    llm_type = base_config.get("llm", "openai")
    
    config = base_config.copy()
    config["tool"] = llm_type
    
    # Add model-specific configurations
    if llm_type == "openai":
        config["model_name"] = config.get("model_name", "gpt-4.1")
    # elif llm_type == "gemini":
    #     config["model_name"] = config.get("model_name", "gemini-2.0-flash")
    
    return config

def validate_llm_credentials():
    """Validate that necessary API keys are present"""
    llm_type = os.getenv('PREFERRED_LLM')
    
    if llm_type == "openai" and not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OpenAI API key not found in environment variables")
    elif llm_type == "gemini" and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError("Google API key not found in environment variables") 