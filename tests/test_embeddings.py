import unittest
import numpy as np
from typing import List

from models.document_chunk import DocumentChunk
from validators.embedding_validator import EmbeddingValidator


class TestEmbeddingValidator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.validator = EmbeddingValidator()

    def create_test_chunk(self, content: str = "Test content", embedding: List[float] = None) -> DocumentChunk:
        """Helper method to create a test chunk."""
        if embedding is None:
            embedding = [0.1, 0.2, 0.3]

        return DocumentChunk(
            id="test-id",
            content=content,
            source_url="https://example.com",
            document_hierarchy="Test > Document",
            metadata={},
            embedding=embedding
        )

    def test_validate_embedding_dimensions(self):
        """Test validating embedding dimensions."""
        # Create chunks with correct dimension
        chunk1 = self.create_test_chunk(embedding=[0.1, 0.2, 0.3, 0.4])
        chunk2 = self.create_test_chunk(embedding=[0.5, 0.6, 0.7, 0.8])
        chunks = [chunk1, chunk2]

        result = self.validator.validate_embedding_dimensions(chunks, expected_dimension=4)
        self.assertTrue(result['valid'])
        self.assertEqual(result['valid_embeddings'], 2)
        self.assertEqual(result['dimension_issues'], 0)

        # Create chunks with incorrect dimension
        chunk3 = self.create_test_chunk(embedding=[0.1, 0.2])  # Only 2 dimensions
        chunks_with_wrong_dim = [chunk1, chunk3]

        result = self.validator.validate_embedding_dimensions(chunks_with_wrong_dim, expected_dimension=4)
        self.assertFalse(result['valid'])
        self.assertEqual(result['valid_embeddings'], 1)
        self.assertEqual(result['dimension_issues'], 1)

    def test_validate_embedding_values(self):
        """Test validating embedding values are within range."""
        # Create chunks with valid values
        chunk1 = self.create_test_chunk(embedding=[0.1, 0.2, 0.3])
        chunk2 = self.create_test_chunk(embedding=[0.5, 0.6, 0.7])
        chunks = [chunk1, chunk2]

        result = self.validator.validate_embedding_values(chunks)
        self.assertTrue(result['valid'])
        self.assertEqual(result['valid_embeddings'], 2)
        self.assertEqual(result['value_range_issues'], 0)

        # Create chunks with invalid values (outside range)
        chunk3 = self.create_test_chunk(embedding=[3.0, 0.2, 0.3])  # 3.0 is outside default range [-2.0, 2.0]
        chunks_with_invalid = [chunk1, chunk3]

        result = self.validator.validate_embedding_values(chunks_with_invalid)
        self.assertFalse(result['valid'])
        self.assertEqual(result['value_range_issues'], 1)

    def test_validate_embedding_norms(self):
        """Test validating embedding norms."""
        # Create chunks with embeddings that have reasonable norms
        chunk1 = self.create_test_chunk(embedding=[0.5, 0.5, 0.5])  # Norm is sqrt(0.75) ≈ 0.87
        chunk2 = self.create_test_chunk(embedding=[1.0, 1.0, 1.0])  # Norm is sqrt(3) ≈ 1.73
        chunks = [chunk1, chunk2]

        result = self.validator.validate_embedding_norms(chunks)
        self.assertTrue(result['valid'])
        self.assertEqual(result['valid_embeddings'], 2)

        # Create chunks with embeddings that have norms outside range
        chunk3 = self.create_test_chunk(embedding=[10.0, 10.0, 10.0])  # Norm is sqrt(300) ≈ 17.32, too large
        chunks_with_large_norm = [chunk1, chunk3]

        result = self.validator.validate_embedding_norms(chunks_with_large_norm)
        self.assertFalse(result['valid'])
        self.assertEqual(result['norm_issues'], 1)

    def test_validate_content_quality(self):
        """Test validating content quality for embedding."""
        # Create chunks with good content
        chunk1 = self.create_test_chunk(content="This is good content for embedding.")
        chunk2 = self.create_test_chunk(content="Another piece of good content.")
        chunks = [chunk1, chunk2]

        result = self.validator.validate_content_quality_for_embedding(chunks)
        self.assertTrue(result['valid'])
        self.assertEqual(result['valid_content'], 2)
        self.assertEqual(result['quality_issues'], 0)

        # Create chunks with poor content (too short)
        chunk3 = self.create_test_chunk(content="Hi")  # Too short
        chunks_with_short = [chunk1, chunk3]

        result = self.validator.validate_content_quality_for_embedding(chunks_with_short)
        self.assertFalse(result['valid'])
        self.assertEqual(result['quality_issues'], 1)

    def test_run_all_validations(self):
        """Test running all validations."""
        # Create chunks with valid data
        chunk1 = self.create_test_chunk(
            content="This is good content for embedding.",
            embedding=[0.1, 0.2, 0.3, 0.4]
        )
        chunk2 = self.create_test_chunk(
            content="Another good piece of content.",
            embedding=[0.5, 0.6, 0.7, 0.8]
        )
        chunks = [chunk1, chunk2]

        result = self.validator.run_all_validations(chunks, expected_dimension=4)
        self.assertTrue(result['all_valid'])
        self.assertEqual(result['total_issues'], 0)

        # Create chunks with various issues
        chunk3 = self.create_test_chunk(
            content="Hi",  # Too short
            embedding=[0.1, 0.2]  # Wrong dimension
        )
        chunks_with_issues = [chunk1, chunk3]

        result = self.validator.run_all_validations(chunks_with_issues, expected_dimension=4)
        self.assertFalse(result['all_valid'])
        self.assertGreater(result['total_issues'], 0)


if __name__ == '__main__':
    unittest.main()