import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
from typing import List

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from services.crawl_service import CrawlService
from services.embedding_service import EmbeddingService
from services.vector_service import VectorService
from models.document_chunk import DocumentChunk


class TestCrawlService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Config(
            docusaurus_url="https://example.com/docs",
            cohere_api_key="test-key",
            qdrant_url="https://test.qdrant.io",
            qdrant_api_key="test-key",
            qdrant_collection_name="test-collection"
        )

        # Mock the crawler, extractor, chunker, and other dependencies
        with patch('backend.services.crawl_service.WebCrawler'), \
             patch('backend.services.crawl_service.HTMLExtractor'), \
             patch('backend.services.crawl_service.TextChunker'), \
             patch('backend.services.crawl_service.URLDiscovery'):
            self.crawl_service = CrawlService(
                self.config.docusaurus_url,
                delay=0.1,  # Fast for tests
                max_depth=2,
                chunk_size=500,
                chunk_overlap=50
            )

    def test_crawl_service_initialization(self):
        """Test that the crawl service initializes correctly."""
        self.assertEqual(self.crawl_service.base_url, self.config.docusaurus_url)
        self.assertEqual(self.crawl_service.delay, 0.1)
        self.assertEqual(self.crawl_service.max_depth, 2)

    @patch('backend.services.crawl_service.URLDiscovery.discover_all_urls')
    def test_crawl_and_extract(self, mock_discover):
        """Test the crawl and extract functionality."""
        # Mock URL discovery to return test URLs
        mock_discover.return_value = ["https://example.com/docs/page1"]

        # Since we're mocking, we expect an empty result
        result = self.crawl_service.crawl_and_extract()
        # The actual crawling will fail because we're mocking the crawler,
        # but the method should still be callable
        self.assertIsInstance(result, list)


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock Cohere service
        self.mock_cohere = Mock()
        self.mock_cohere.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        self.mock_cohere.generate_single_embedding.return_value = [0.1, 0.2, 0.3]
        self.mock_cohere.validate_text_for_embedding.return_value = True

        self.embedding_service = EmbeddingService(self.mock_cohere)

    def test_generate_embeddings_for_chunks(self):
        """Test generating embeddings for document chunks."""
        # Create test chunks
        chunk = DocumentChunk(
            id="test-id",
            content="This is a test document chunk for embedding.",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={}
        )

        chunks = [chunk]

        # Test embedding generation
        result = self.embedding_service.generate_embeddings_for_chunks(chunks)

        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].embedding)
        self.assertEqual(result[0].embedding, [0.1, 0.2, 0.3])

    def test_validate_embeddings(self):
        """Test embedding validation."""
        # Create test chunks with embeddings
        chunk_with_embedding = DocumentChunk(
            id="test-id-1",
            content="Test content 1",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )

        chunk_without_embedding = DocumentChunk(
            id="test-id-2",
            content="Test content 2",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={}
        )

        chunks = [chunk_with_embedding, chunk_without_embedding]

        # Test validation
        result = self.embedding_service.validate_embeddings(chunks)

        self.assertEqual(result['total_chunks'], 2)
        self.assertEqual(result['with_embeddings'], 1)
        self.assertEqual(result['without_embeddings'], 1)
        self.assertEqual(result['valid_embeddings'], 1)


class TestVectorService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies
        self.mock_embedding_service = Mock()
        self.mock_vector_storage_service = Mock()

        # Configure mock behaviors
        self.mock_embedding_service.generate_embeddings_with_validation.return_value = []
        self.mock_vector_storage_service.store_chunks_with_deduplication.return_value = True
        self.mock_vector_storage_service.validate_storage_results.return_value = {
            'total_chunks': 0,
            'with_embeddings': 0,
            'stored_successfully': 0,
            'failed_to_store': 0,
            'duplicate_chunks': 0,
            'current_vector_count': 0
        }

        self.vector_service = VectorService(
            self.mock_embedding_service,
            self.mock_vector_storage_service
        )

    def test_process_and_store_chunks(self):
        """Test the process and store chunks functionality."""
        # Create test chunks
        chunk = DocumentChunk(
            id="test-id",
            content="Test content for vector storage.",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={}
        )

        chunks = [chunk]

        # Test processing and storing
        result = self.vector_service.process_and_store_chunks(chunks)

        # Verify the result structure
        self.assertIn('total_chunks', result)
        self.assertIn('embedded_chunks', result)
        self.assertIn('stored_chunks', result)
        self.assertIn('success', result)


class TestDocumentChunk(unittest.TestCase):
    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            id="test-id",
            content="This is test content.",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={"title": "Test Document"}
        )

        self.assertEqual(chunk.id, "test-id")
        self.assertEqual(chunk.content, "This is test content.")
        self.assertEqual(chunk.source_url, "https://example.com")
        self.assertEqual(chunk.document_hierarchy, "Test > Document")
        self.assertEqual(chunk.metadata["title"], "Test Document")

    def test_document_chunk_validation(self):
        """Test document chunk validation."""
        chunk = DocumentChunk(
            id="test-id",
            content="This is test content.",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={"title": "Test Document"}
        )

        # Should not raise an exception
        self.assertTrue(chunk.validate())

    def test_document_chunk_to_dict(self):
        """Test converting document chunk to dictionary."""
        chunk = DocumentChunk(
            id="test-id",
            content="Test content",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={"key": "value"}
        )

        chunk_dict = chunk.to_dict()

        self.assertEqual(chunk_dict["id"], "test-id")
        self.assertEqual(chunk_dict["content"], "Test content")
        self.assertEqual(chunk_dict["source_url"], "https://example.com")
        self.assertEqual(chunk_dict["document_hierarchy"], "Test > Document")
        self.assertEqual(chunk_dict["metadata"]["key"], "value")

    def test_document_chunk_has_embedding(self):
        """Test checking if a chunk has an embedding."""
        chunk_with_embedding = DocumentChunk(
            id="test-id-1",
            content="Test content",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )

        chunk_without_embedding = DocumentChunk(
            id="test-id-2",
            content="Test content",
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={}
        )

        self.assertTrue(chunk_with_embedding.has_embedding())
        self.assertFalse(chunk_without_embedding.has_embedding())


if __name__ == '__main__':
    unittest.main()