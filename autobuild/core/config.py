"""
Configuration management for Gitcontainer application.

This module handles application configuration using Pydantic settings.
"""

import os
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """
    Application settings management.
    
    This class handles all configuration parameters for the application,
    loading them from environment variables with appropriate defaults.
    """
    
    # API Keys
    openai_api_key: str = ""
    fastapi_analytics_key: Optional[str] = None
    inf_api_key: Optional[str] = None
    
    # API Configuration
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    available_models: str = "gpt-4o-mini|true,gpt-4o|true,o1-mini|false,o1|false"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Application Configuration
    repos_dir: str = "repos"
    max_context_chars: int = 50000
    max_iterations: int = 2
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_api_key(cls, v: str) -> str:
        """Validate that OpenAI API key is provided."""
        if not v:
            raise ValueError('OPENAI_API_KEY environment variable is required')
        return v
    
    def get_available_models(self) -> List[dict]:
        """
        Parse the AVAILABLE_MODELS environment variable.
        
        Format: model_name|stream_support,model_name|stream_support,...
        Example: gpt-4o-mini|true,gpt-4o|true,o1-mini|false,o1|false
        
        Returns:
            List[dict]: List of available models with their stream support info
        """
        if not self.available_models:
            return []
        
        models = []
        for model_entry in self.available_models.split(","):
            model_entry = model_entry.strip()
            if "|" in model_entry:
                model_name, stream_support = model_entry.split("|")
                models.append({
                    "name": model_name.strip(),
                    "stream": stream_support.strip().lower() == "true"
                })
            else:
                # For backward compatibility, if no | is present, assume stream is supported
                models.append({
                    "name": model_entry,
                    "stream": True
                })
        return models
    
    def get_model_stream_support(self, model_name: str) -> bool:
        """
        Check if a specific model supports streaming.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if model supports streaming, False otherwise
        """
        models = self.get_available_models()
        for model in models:
            if model["name"] == model_name:
                return model["stream"]
        # Default to True if model not found in the list
        return True