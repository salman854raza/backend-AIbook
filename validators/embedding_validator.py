from typing import List, Dict, Any
import logging
import numpy as np
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingValidator:
    """
    Validator for checking embedding quality and consistency
    """

    def __init__(self):
        """
        Initialize the embedding validator
        """
        pass

    def validate_embedding_dimensions(self, chunks: List[DocumentChunk], expected_dimension: int = None) -> Dict[str, Any]:
        """
        Validate that all embeddings have the expected dimensions
        """
        if not chunks:
            return {
                'valid': True,
                'total_chunks': 0,
                'valid_embeddings': 0,
                'invalid_embeddings': 0,
                'dimension_issues': 0
            }

        if expected_dimension is None:
            # If no expected dimension provided, use the dimension of the first valid embedding
            for chunk in chunks:
                if chunk.has_embedding():
                    expected_dimension = len(chunk.embedding)
                    break

        if expected_dimension is None:
            # No valid embeddings found
            return {
                'valid': True,  # Technically valid since there are no invalid ones
                'total_chunks': len(chunks),
                'valid_embeddings': 0,
                'invalid_embeddings': 0,
                'dimension_issues': 0
            }

        valid_embeddings = 0
        invalid_embeddings = 0
        dimension_issues = 0

        for chunk in chunks:
            if chunk.has_embedding():
                if len(chunk.embedding) == expected_dimension:
                    valid_embeddings += 1
                else:
                    invalid_embeddings += 1
                    dimension_issues += 1
                    logger.warning(f"Chunk {chunk.id} has embedding dimension {len(chunk.embedding)}, expected {expected_dimension}")
            else:
                invalid_embeddings += 1

        is_valid = dimension_issues == 0
        result = {
            'valid': is_valid,
            'total_chunks': len(chunks),
            'valid_embeddings': valid_embeddings,
            'invalid_embeddings': invalid_embeddings,
            'dimension_issues': dimension_issues,
            'expected_dimension': expected_dimension
        }

        logger.info(f"Embedding dimension validation: {result}")
        return result

    def validate_embedding_values(self, chunks: List[DocumentChunk], min_value: float = -2.0, max_value: float = 2.0) -> Dict[str, Any]:
        """
        Validate that embedding values are within reasonable ranges
        """
        if not chunks:
            return {
                'valid': True,
                'total_chunks': 0,
                'valid_embeddings': 0,
                'invalid_embeddings': 0,
                'value_range_issues': 0
            }

        valid_embeddings = 0
        invalid_embeddings = 0
        value_range_issues = 0

        for chunk in chunks:
            if chunk.has_embedding():
                has_invalid_values = False
                for value in chunk.embedding:
                    if not isinstance(value, (int, float)) or value < min_value or value > max_value:
                        has_invalid_values = True
                        value_range_issues += 1
                        break

                if has_invalid_values:
                    invalid_embeddings += 1
                    logger.warning(f"Chunk {chunk.id} has embedding values outside the valid range [{min_value}, {max_value}]")
                else:
                    valid_embeddings += 1
            else:
                invalid_embeddings += 1

        is_valid = value_range_issues == 0
        result = {
            'valid': is_valid,
            'total_chunks': len(chunks),
            'valid_embeddings': valid_embeddings,
            'invalid_embeddings': invalid_embeddings,
            'value_range_issues': value_range_issues,
            'min_value': min_value,
            'max_value': max_value
        }

        logger.info(f"Embedding value validation: {result}")
        return result

    def validate_embedding_norms(self, chunks: List[DocumentChunk], expected_norm_range: tuple = (0.1, 2.0)) -> Dict[str, Any]:
        """
        Validate that embedding norms are within expected range
        """
        if not chunks:
            return {
                'valid': True,
                'total_chunks': 0,
                'valid_embeddings': 0,
                'invalid_embeddings': 0,
                'norm_issues': 0
            }

        valid_embeddings = 0
        invalid_embeddings = 0
        norm_issues = 0

        min_norm, max_norm = expected_norm_range

        for chunk in chunks:
            if chunk.has_embedding():
                # Calculate the L2 norm of the embedding
                embedding_array = np.array(chunk.embedding)
                norm = np.linalg.norm(embedding_array)

                if min_norm <= norm <= max_norm:
                    valid_embeddings += 1
                else:
                    invalid_embeddings += 1
                    norm_issues += 1
                    logger.warning(f"Chunk {chunk.id} has embedding norm {norm:.4f}, expected range [{min_norm}, {max_norm}]")
            else:
                invalid_embeddings += 1

        is_valid = norm_issues == 0
        result = {
            'valid': is_valid,
            'total_chunks': len(chunks),
            'valid_embeddings': valid_embeddings,
            'invalid_embeddings': invalid_embeddings,
            'norm_issues': norm_issues,
            'expected_norm_range': expected_norm_range
        }

        logger.info(f"Embedding norm validation: {result}")
        return result

    def validate_content_quality_for_embedding(self, chunks: List[DocumentChunk], min_length: int = 10) -> Dict[str, Any]:
        """
        Validate that content is of sufficient quality for embedding
        """
        if not chunks:
            return {
                'valid': True,
                'total_chunks': 0,
                'valid_content': 0,
                'invalid_content': 0,
                'quality_issues': 0
            }

        valid_content = 0
        invalid_content = 0
        quality_issues = 0

        for chunk in chunks:
            content = chunk.content.strip() if chunk.content else ""

            if len(content) < min_length:
                invalid_content += 1
                quality_issues += 1
                logger.warning(f"Chunk {chunk.id} has content length {len(content)}, below minimum {min_length}")
            else:
                valid_content += 1

        is_valid = quality_issues == 0
        result = {
            'valid': is_valid,
            'total_chunks': len(chunks),
            'valid_content': valid_content,
            'invalid_content': invalid_content,
            'quality_issues': quality_issues,
            'min_length': min_length
        }

        logger.info(f"Content quality validation: {result}")
        return result

    def run_all_validations(self, chunks: List[DocumentChunk], expected_dimension: int = 1024) -> Dict[str, Any]:
        """
        Run all validations on the embeddings
        """
        logger.info(f"Running all validations on {len(chunks)} chunks")

        # Run each validation
        dimension_result = self.validate_embedding_dimensions(chunks, expected_dimension)
        value_result = self.validate_embedding_values(chunks)
        norm_result = self.validate_embedding_norms(chunks)
        content_result = self.validate_content_quality_for_embedding(chunks)

        # Overall validation result
        all_valid = (
            dimension_result['valid'] and
            value_result['valid'] and
            norm_result['valid'] and
            content_result['valid']
        )

        total_issues = (
            dimension_result['dimension_issues'] +
            value_result['value_range_issues'] +
            norm_result['norm_issues'] +
            content_result['quality_issues']
        )

        overall_result = {
            'all_valid': all_valid,
            'total_chunks': len(chunks),
            'total_issues': total_issues,
            'dimension_validation': dimension_result,
            'value_validation': value_result,
            'norm_validation': norm_result,
            'content_validation': content_result
        }

        if all_valid:
            logger.info("All embeddings passed validation")
        else:
            logger.warning(f"Embeddings failed validation. Total issues: {total_issues}")

        return overall_result

    def validate_embedding_similarity(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Validate that embeddings are not too similar (indicating potential duplicates)
        For this validation, we'll check if any embeddings are nearly identical
        """
        if len(chunks) < 2:
            return {
                'valid': True,
                'total_comparisons': 0,
                'similar_pairs': 0,
                'similarity_issues': 0
            }

        # Filter chunks that have embeddings
        embeddable_chunks = [chunk for chunk in chunks if chunk.has_embedding()]

        if len(embeddable_chunks) < 2:
            return {
                'valid': True,
                'total_comparisons': 0,
                'similar_pairs': 0,
                'similarity_issues': 0
            }

        similar_pairs = 0
        similarity_issues = 0
        total_comparisons = 0

        # Compare embeddings using cosine similarity
        for i, chunk1 in enumerate(embeddable_chunks):
            for j, chunk2 in enumerate(embeddable_chunks):
                if i >= j:  # Avoid duplicate comparisons and self-comparison
                    continue

                total_comparisons += 1

                # Calculate cosine similarity
                emb1 = np.array(chunk1.embedding)
                emb2 = np.array(chunk2.embedding)

                # Normalize embeddings
                emb1_norm = emb1 / np.linalg.norm(emb1)
                emb2_norm = emb2 / np.linalg.norm(emb2)

                # Calculate cosine similarity
                similarity = np.dot(emb1_norm, emb2_norm)

                if similarity > similarity_threshold:
                    similar_pairs += 1
                    similarity_issues += 1
                    logger.warning(f"Chunks {chunk1.id} and {chunk2.id} have high similarity: {similarity:.4f}")

        is_valid = similarity_issues == 0
        result = {
            'valid': is_valid,
            'total_comparisons': total_comparisons,
            'similar_pairs': similar_pairs,
            'similarity_issues': similarity_issues,
            'similarity_threshold': similarity_threshold
        }

        logger.info(f"Embedding similarity validation: {result}")
        return result