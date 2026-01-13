from typing import List, Dict, Any
import logging
import hashlib
from models.document_chunk import DocumentChunk
from clients.qdrant_client import QdrantService

logger = logging.getLogger(__name__)

class VectorStorageService:
    """
    Service for storing embeddings in Qdrant vector database
    """

    def __init__(self, qdrant_service: QdrantService):
        """
        Initialize the vector storage service
        """
        self.qdrant_service = qdrant_service

    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store document chunks with embeddings in Qdrant
        """
        if not chunks:
            logger.info("No chunks provided for storage")
            return True

        # Prepare data for storage
        vector_ids = []
        vectors = []
        payloads = []

        for chunk in chunks:
            if not chunk.has_embedding():
                logger.warning(f"Chunk {chunk.id} does not have an embedding, skipping")
                continue

            # Generate a unique ID for Qdrant (using the chunk ID or creating one from content)
            vector_id = chunk.id or self._generate_vector_id(chunk)

            # Prepare the payload with metadata
            payload = {
                'source_url': chunk.source_url,
                'document_hierarchy': chunk.document_hierarchy,
                'content': chunk.content,
                'chunk_index': chunk.metadata.get('chunk_index', 0),
                'created_at': chunk.metadata.get('created_at', ''),
                'title': chunk.metadata.get('title', ''),
                'headings': chunk.metadata.get('headings', []),
                'content_length': len(chunk.content),
                'hash': chunk.metadata.get('hash', ''),
                'chunk_id': chunk.id
            }

            # Add any additional metadata
            for key, value in chunk.metadata.items():
                if key not in payload:
                    payload[key] = value

            vector_ids.append(vector_id)
            vectors.append(chunk.embedding)
            payloads.append(payload)

        if not vectors:
            logger.warning("No vectors with embeddings found to store")
            return True

        logger.info(f"Storing {len(vectors)} vectors in Qdrant collection {self.qdrant_service.collection_name}")

        try:
            # Store the vectors in Qdrant
            success = self.qdrant_service.upsert_vectors(vector_ids, vectors, payloads)
            logger.info(f"Successfully stored {len(vectors)} vectors in Qdrant")
            return success
        except Exception as e:
            logger.error(f"Error storing vectors in Qdrant: {str(e)}")
            raise

    def _generate_vector_id(self, chunk: DocumentChunk) -> str:
        """
        Generate a unique ID for a vector if not provided
        """
        # Create a unique ID based on content and source URL
        content_hash = hashlib.md5(chunk.content.encode('utf-8')).hexdigest()
        url_hash = hashlib.md5(chunk.source_url.encode('utf-8')).hexdigest()
        return f"{url_hash[:8]}_{content_hash[:16]}"

    def create_collection_if_not_exists(self, vector_size: int = 1024, distance_metric: str = "Cosine") -> bool:
        """
        Create the Qdrant collection if it doesn't exist
        """
        try:
            created = self.qdrant_service.create_collection_if_not_exists(vector_size, distance_metric)
            return created
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            raise

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the collection
        """
        try:
            count = self.qdrant_service.get_vector_count()
            logger.info(f"Current vector count in collection: {count}")
            return count
        except Exception as e:
            logger.error(f"Error getting vector count from Qdrant: {str(e)}")
            raise

    def search_similar(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        """
        try:
            results = self.qdrant_service.search_similar(query_vector, limit)
            logger.info(f"Found {len(results)} similar vectors")
            return results
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            raise

    def validate_document_exists(self, source_url: str) -> bool:
        """
        Check if a document with the given source URL already exists in the collection
        """
        try:
            exists = self.qdrant_service.check_document_exists(source_url)
            return exists
        except Exception as e:
            logger.error(f"Error checking document existence in Qdrant: {str(e)}")
            return False

    def validate_storage_results(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate the storage results
        """
        stats = {
            'total_chunks': len(chunks),
            'with_embeddings': 0,
            'stored_successfully': 0,
            'failed_to_store': 0,
            'duplicate_chunks': 0
        }

        for chunk in chunks:
            if chunk.has_embedding():
                stats['with_embeddings'] += 1

        # Get current vector count
        try:
            current_count = self.get_vector_count()
            stats['current_vector_count'] = current_count
        except:
            stats['current_vector_count'] = -1

        return stats

    def store_chunks_with_deduplication(self, chunks: List[DocumentChunk]) -> bool:
        """
        Store chunks with deduplication logic
        """
        if not chunks:
            return True

        # Filter chunks that have embeddings
        embeddable_chunks = [chunk for chunk in chunks if chunk.has_embedding()]

        if not embeddable_chunks:
            logger.warning("No chunks with embeddings to store")
            return True

        # Check for duplicates and filter unique chunks
        unique_chunks = []
        for chunk in embeddable_chunks:
            # Check if this document already exists in the database
            if not self.validate_document_exists(chunk.source_url):
                unique_chunks.append(chunk)
            else:
                logger.info(f"Document from {chunk.source_url} already exists, skipping")

        stats = {
            'total_chunks': len(chunks),
            'with_embeddings': len(embeddable_chunks),
            'unique_chunks': len(unique_chunks),
            'duplicates_skipped': len(embeddable_chunks) - len(unique_chunks)
        }

        logger.info(f"Storage stats: {stats}")

        if unique_chunks:
            return self.store_embeddings(unique_chunks)
        else:
            logger.info("No new unique chunks to store")
            return True

    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution!)
        """
        try:
            success = self.qdrant_service.delete_collection()
            logger.info("Successfully deleted Qdrant collection")
            return success
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {str(e)}")
            raise
