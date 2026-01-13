import cohere
from typing import List
import logging

logger = logging.getLogger(__name__)

class CohereService:
    """Service for interacting with Cohere API for embeddings"""

    def __init__(self, api_key: str):
        """
        Initialize Cohere service with API key
        """
        self.client = cohere.Client(api_key)
        # Using the multilingual v3 embedding model which has 1024 dimensions
        self.model = "embed-multilingual-v2.0"

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 96  # Cohere's recommended batch size
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {str(e)}")
                raise

        return all_embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model being used
        """
        return {
            "model": self.model,
            "dimensions": 768,  # Cohere multilingual v2 model has 768 dimensions
            "description": "Cohere multilingual embedding model v2.0"
        }

    def validate_text_for_embedding(self, text: str) -> bool:
        """
        Validate if text is suitable for embedding generation
        """
        if not text or len(text.strip()) == 0:
            return False

        # Check for length limits (Cohere has limits, though they're quite generous)
        # The actual limit is much higher, but we'll set a reasonable limit
        if len(text) > 4096:  # Conservative limit
            logger.warning(f"Text length ({len(text)}) exceeds recommended limit for embedding")
            return False

        return True