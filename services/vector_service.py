from typing import List, Dict, Any
import logging
from models.document_chunk import DocumentChunk
from services.embedding_service import EmbeddingService
from services.vector_storage_service import VectorStorageService

logger = logging.getLogger(__name__)

class VectorService:
    """
    Service for orchestrating the full embedding and storage workflow
    """

    def __init__(self, embedding_service: EmbeddingService, vector_storage_service: VectorStorageService):
        """
        Initialize the vector service
        """
        self.embedding_service = embedding_service
        self.vector_storage_service = vector_storage_service

    def process_and_store_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Process chunks through embedding and store in vector database
        """
        if not chunks:
            logger.info("No chunks provided for processing")
            return {
                'total_chunks': 0,
                'embedded_chunks': 0,
                'stored_chunks': 0,
                'success': True
            }

        logger.info(f"Starting vector processing for {len(chunks)} chunks")

        # Step 1: Generate embeddings
        logger.info("Step 1: Generating embeddings...")
        embedded_chunks = self.embedding_service.generate_embeddings_with_validation(chunks)

        # Step 2: Store embeddings in vector database
        logger.info("Step 2: Storing embeddings in vector database...")
        storage_success = self.vector_storage_service.store_chunks_with_deduplication(embedded_chunks)

        # Step 3: Validate results
        logger.info("Step 3: Validating results...")
        validation_stats = self.embedding_service.validate_embeddings(embedded_chunks)
        storage_stats = self.vector_storage_service.validate_storage_results(embedded_chunks)

        result = {
            'total_chunks': len(chunks),
            'embedded_chunks': len(embedded_chunks),
            'stored_chunks': len([c for c in embedded_chunks if c.has_embedding()]),
            'success': storage_success,
            'embedding_stats': validation_stats,
            'storage_stats': storage_stats
        }

        logger.info(f"Vector processing completed: {result}")
        return result

    def create_collection_if_needed(self, vector_size: int = 1024, distance_metric: str = "Cosine") -> bool:
        """
        Create the vector collection if it doesn't exist
        """
        try:
            created = self.vector_storage_service.create_collection_if_not_exists(vector_size, distance_metric)
            logger.info(f"Collection setup completed. Created: {created}")
            return created
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def search_similar_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar content using a text query
        """
        logger.info(f"Searching for content similar to: {query[:50]}...")

        # Generate embedding for the query
        query_embedding = self.embedding_service.generate_single_embedding(query)

        # Search in the vector database
        results = self.vector_storage_service.search_similar(query_embedding, limit)

        logger.info(f"Found {len(results)} similar content items")
        return results

    def get_vector_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        """
        try:
            vector_count = self.vector_storage_service.get_vector_count()
            model_info = self.embedding_service.get_model_info()

            stats = {
                'vector_count': vector_count,
                'embedding_model': model_info['model'],
                'embedding_dimensions': model_info['dimensions'],
                'updated_at': str(datetime.now())
            }

            logger.info(f"Vector database stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting vector database stats: {str(e)}")
            raise

    def process_chunks_in_batches(self, chunks: List[DocumentChunk], batch_size: int = 96) -> Dict[str, Any]:
        """
        Process chunks in batches to respect API limits
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'embedded_chunks': 0,
                'stored_chunks': 0,
                'success': True
            }

        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks in batches of {batch_size}")

        all_results = {
            'total_chunks': total_chunks,
            'embedded_chunks': 0,
            'stored_chunks': 0,
            'success': True,
            'batch_results': []
        }

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")

            batch_result = self.process_and_store_chunks(batch)
            all_results['embedded_chunks'] += batch_result['embedded_chunks']
            all_results['stored_chunks'] += batch_result['stored_chunks']
            all_results['success'] = all_results['success'] and batch_result['success']
            all_results['batch_results'].append(batch_result)

        logger.info(f"Batch processing completed: {all_results}")
        return all_results

    def validate_full_pipeline(self, test_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate the full embedding and storage pipeline
        """
        logger.info("Validating full embedding and storage pipeline")

        # Test embedding generation
        try:
            embedded_chunks = self.embedding_service.generate_embeddings_with_validation(test_chunks)
            embedding_validation = True
            embedding_error = None
        except Exception as e:
            embedding_validation = False
            embedding_error = str(e)
            embedded_chunks = []

        # Test storage
        storage_validation = False
        storage_error = None
        if embedding_validation and embedded_chunks:
            try:
                storage_success = self.vector_storage_service.store_chunks_with_deduplication(embedded_chunks)
                storage_validation = storage_success
            except Exception as e:
                storage_error = str(e)

        validation_result = {
            'embedding_validated': embedding_validation,
            'embedding_error': embedding_error,
            'storage_validated': storage_validation,
            'storage_error': storage_error,
            'test_chunks_count': len(test_chunks),
            'embedded_chunks_count': len(embedded_chunks)
        }

        logger.info(f"Pipeline validation result: {validation_result}")
        return validation_result

    def delete_all_vectors(self) -> bool:
        """
        Delete all vectors in the collection (use with extreme caution!)
        """
        logger.warning("Deleting all vectors in the collection!")
        try:
            success = self.vector_storage_service.delete_collection()
            logger.info("All vectors deleted successfully")
            return success
        except Exception as e:
            logger.error(f"Error deleting all vectors: {str(e)}")
            raise