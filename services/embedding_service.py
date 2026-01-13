from typing import List, Dict, Any
import logging
from models.document_chunk import DocumentChunk
from clients.cohere_client import CohereService

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using Cohere
    """

    def __init__(self, cohere_service: CohereService):
        """
        Initialize the embedding service
        """
        self.cohere_service = cohere_service

    def generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for a list of document chunks
        """
        if not chunks:
            logger.info("No chunks provided for embedding generation")
            return []

        # Extract content from chunks
        texts = [chunk.content for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} text chunks")

        try:
            # Generate embeddings using Cohere
            embeddings = self.cohere_service.generate_embeddings(texts)

            # Assign embeddings back to chunks
            updated_chunks = []
            for i, chunk in enumerate(chunks):
                # Create a copy of the chunk with the embedding
                updated_chunk = DocumentChunk(
                    id=chunk.id,
                    content=chunk.content,
                    source_url=chunk.source_url,
                    document_hierarchy=chunk.document_hierarchy,
                    metadata=chunk.metadata,
                    embedding=embeddings[i] if i < len(embeddings) else None
                )
                updated_chunks.append(updated_chunk)

            logger.info(f"Successfully generated embeddings for {len(updated_chunks)} chunks")
            return updated_chunks

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding for a text
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        # Validate text for embedding
        if not self.cohere_service.validate_text_for_embedding(text):
            raise ValueError(f"Text is not suitable for embedding: length={len(text)}")

        try:
            embedding = self.cohere_service.generate_single_embedding(text)
            logger.info(f"Generated single embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            raise

    def validate_embeddings(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate the generated embeddings
        """
        stats = {
            'total_chunks': len(chunks),
            'with_embeddings': 0,
            'without_embeddings': 0,
            'avg_embedding_size': 0,
            'valid_embeddings': 0,
            'invalid_embeddings': 0
        }

        total_size = 0
        for chunk in chunks:
            if chunk.has_embedding():
                stats['with_embeddings'] += 1
                if chunk.embedding:
                    total_size += len(chunk.embedding)
                    # Basic validation: check if embedding is a list of floats
                    if isinstance(chunk.embedding, list) and all(isinstance(x, (int, float)) for x in chunk.embedding):
                        stats['valid_embeddings'] += 1
                    else:
                        stats['invalid_embeddings'] += 1
            else:
                stats['without_embeddings'] += 1

        if stats['with_embeddings'] > 0:
            stats['avg_embedding_size'] = total_size / stats['with_embeddings']

        return stats

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model being used
        """
        return self.cohere_service.get_model_info()

    def batch_process_chunks(self, chunks: List[DocumentChunk], batch_size: int = 96) -> List[DocumentChunk]:
        """
        Process chunks in batches to respect API limits
        """
        if not chunks:
            return []

        all_processed_chunks = []
        total_chunks = len(chunks)

        logger.info(f"Batch processing {total_chunks} chunks in batches of {batch_size}")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")

            processed_batch = self.generate_embeddings_for_chunks(batch)
            all_processed_chunks.extend(processed_batch)

        logger.info(f"Completed batch processing. Processed {len(all_processed_chunks)} chunks.")

        return all_processed_chunks

    def filter_valid_chunks_for_embedding(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Filter chunks that are valid for embedding generation
        """
        valid_chunks = []
        for chunk in chunks:
            # Check if the chunk content is suitable for embedding
            if (chunk.content and
                len(chunk.content.strip()) > 0 and
                self.cohere_service.validate_text_for_embedding(chunk.content)):
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk {chunk.id} - content not suitable for embedding")

        return valid_chunks

    def generate_embeddings_with_validation(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings with validation and filtering
        """
        # Filter valid chunks
        valid_chunks = self.filter_valid_chunks_for_embedding(chunks)
        logger.info(f"Filtered {len(valid_chunks)} valid chunks out of {len(chunks)} total")

        if not valid_chunks:
            logger.warning("No valid chunks found for embedding generation")
            return []

        # Generate embeddings
        embedded_chunks = self.generate_embeddings_for_chunks(valid_chunks)

        # Validate the results
        validation_stats = self.validate_embeddings(embedded_chunks)
        logger.info(f"Embedding validation: {validation_stats}")

        return embedded_chunks