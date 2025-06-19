"""
Configuration settings for the AI Chatbot
"""
import os
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = ""
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
    
    # Explainability
    enable_lime: bool = True
    enable_shap: bool = True
    explanation_detail_level: str = "medium"  # low, medium, high
    
    # Fairness
    bias_threshold: float = 0.1
    fairness_check_enabled: bool = True
      # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", "8000"))  # Handle Render's PORT env var
    cors_origins: List[str] = ["*"]  # Allow all origins for deployment
    
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
