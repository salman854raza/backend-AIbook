import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration for the RAG Chatbot Backend"""

    # Docusaurus configuration
    docusaurus_url: str = os.getenv("DOCUSAURUS_URL", "")

    # Cohere configuration
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # OpenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # OpenRouter configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # Qdrant configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "docusaurus_chunks")

    # Ingestion pipeline configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    crawl_delay: float = float(os.getenv("CRAWL_DELAY", "1.0"))
    max_depth: int = int(os.getenv("MAX_DEPTH", "5"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of validation errors"""
        errors = []

        if not self.docusaurus_url:
            errors.append("DOCUSAURUS_URL environment variable is required")

        if not self.cohere_api_key:
            errors.append("COHERE_API_KEY environment variable is required")

        # Either OpenAI or OpenRouter API key is required
        if not self.openai_api_key and not self.openrouter_api_key:
            errors.append("Either OPENAI_API_KEY or OPENROUTER_API_KEY environment variable is required")

        if not self.qdrant_url:
            errors.append("QDRANT_URL environment variable is required")

        # Only require API key for cloud instances (those with 'cloud' in the URL)
        if not self.qdrant_api_key and 'cloud' in self.qdrant_url.lower():
            errors.append("QDRANT_API_KEY environment variable is required for cloud instances")

        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE must be a positive integer")

        if self.chunk_overlap < 0:
            errors.append("CHUNK_OVERLAP cannot be negative")

        if self.crawl_delay < 0:
            errors.append("CRAWL_DELAY cannot be negative")

        if self.max_depth <= 0:
            errors.append("MAX_DEPTH must be a positive integer")

        return errors

def get_config() -> Config:
    """Get the application configuration"""
    config = Config()
    validation_errors = config.validate()

    if validation_errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")

    return config