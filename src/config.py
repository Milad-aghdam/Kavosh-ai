import os
from dotenv import load_dotenv

def get_openrouter_config():
    """
    Loads the OpenRouter API key from the .env file.
    """
    load_dotenv() 
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file or environment variables.")

    # OpenRouter requires these two values
    return {
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1"
    }