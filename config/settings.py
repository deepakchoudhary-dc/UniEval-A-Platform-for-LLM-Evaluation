"""
Configuration settings for the AI Chatbot
"""
import os
from typing import List
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = ""
    qwen_api_key: str = ""
    openrouter_api_key: str = ""
    huggingface_api_key: str = ""
    
    # Database
    database_url: str = "sqlite:///chatbot_memory.db"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/chatbot.log"
    
    # Memory Settings
    max_memory_entries: int = 10000
    memory_retention_days: int = 365
    
    # Search Settings
    search_index_path: str = "data/search_index"
    max_search_results: int = 10
    # Model Settings
    default_model: str = "gpt-3.5-turbo"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 4000
    temperature: float = 0.7

    # Model Provider: openai, openrouter, ollama
    model_provider: str = "openai"
    ollama_model: str = "qwen3:1.7b"
    
    # Explainability
    enable_lime: bool = True
    enable_shap: bool = True
    explanation_detail_level: str = "medium"  # low, medium, high
      # Fairness
    bias_threshold: float = 0.1
    fairness_check_enabled: bool = True
    
    # Evaluation Settings
    enable_opik_evaluation: bool = True
    evaluation_criteria: List[str] = ["relevance", "hallucination", "moderation", "faithfulness"]
    opik_project_name: str = "transparent-ai-chatbot"
    
      # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]  # Allow all origins for deployment
    
    def __init__(self, **kwargs):
        # Handle Render's PORT environment variable
        port_env = os.getenv("PORT")
        if port_env and port_env.isdigit():
            kwargs.setdefault("api_port", int(port_env))
        super().__init__(**kwargs)
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Validate required settings
def validate_settings():
    """Validate that required settings are configured"""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment variables")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.search_index_path, exist_ok=True)


if __name__ == "__main__":
    validate_settings()
    print("Settings validated successfully!")
