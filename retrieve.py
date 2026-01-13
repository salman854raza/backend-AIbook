"""
RAG Chatbot Retrieval and Validation System

This module provides functionality to retrieve relevant document chunks from the vector database
and validate the retrieval results for a RAG chatbot system.
"""

import logging
from typing import List, Dict, Any, Optional
from clients.qdrant_client import QdrantService
from clients.cohere_client import CohereService
from config import get_config
from datetime import datetime


class RetrievalValidator:
    """
    Class for handling retrieval and validation of document chunks from Qdrant
    """

    def __init__(self):
        """
        Initialize the retrieval validator with required services
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

        # Initialize services
        self.qdrant_service = QdrantService(
            self.config.qdrant_url,
            self.config.qdrant_api_key,
            self.config.qdrant_collection_name
        )
        self.cohere_service = CohereService(self.config.cohere_api_key)

        self.logger.info("RetrievalValidator initialized successfully")

    def retrieve_chunks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks based on a query

        Args:
            query: The search query string
            limit: Maximum number of results to return (default: 5)

        Returns:
            List of dictionaries containing retrieved chunks with metadata
        """
        self.logger.info(f"Retrieving chunks for query: '{query}'")

        try:
            # Generate embedding for the query using Cohere
            query_embedding = self.cohere_service.generate_single_embedding(query)
            self.logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

            # Search in Qdrant for similar vectors
            results = self.qdrant_service.search_similar(query_embedding, limit=limit)

            self.logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
            return results

        except Exception as e:
            self.logger.error(f"Error retrieving chunks for query '{query}': {str(e)}")
            raise

    def validate_retrieval(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the retrieval results for relevance and metadata consistency

        Args:
            query: The original search query
            results: List of retrieved chunks with metadata

        Returns:
            Dictionary containing validation results and metrics
        """
        self.logger.info(f"Validating retrieval results for query: '{query}'")

        validation_report = {
            'query': query,
            'total_results': len(results),
            'validation_timestamp': datetime.now().isoformat(),
            'is_valid': True,
            'issues': [],
            'metrics': {
                'avg_similarity_score': 0.0,
                'min_similarity_score': 0.0,
                'max_similarity_score': 0.0,
                'unique_sources': 0,
                'metadata_completeness': 0.0
            },
            'validation_details': []
        }

        if not results:
            validation_report['is_valid'] = False
            validation_report['issues'].append('No results returned for the query')
            return validation_report

        # Calculate similarity score metrics
        scores = []
        source_urls = set()
        metadata_complete_count = 0

        for i, result in enumerate(results):
            score = result.get('score', 0)
            scores.append(score)

            # Check metadata completeness
            payload = result.get('payload', {})
            required_fields = ['source_url', 'content']
            missing_fields = [field for field in required_fields if not payload.get(field)]

            if not missing_fields:
                metadata_complete_count += 1
                source_urls.add(payload.get('source_url', ''))

            # Validate content relevance (basic check for non-empty content)
            content = payload.get('content', '').strip()
            if not content:
                validation_report['issues'].append(f"Result {i+1}: Empty content in payload")

            # Validate source URL
            source_url = payload.get('source_url', '').strip()
            if not source_url:
                validation_report['issues'].append(f"Result {i+1}: Missing source URL")

            validation_report['validation_details'].append({
                'result_index': i,
                'score': score,
                'source_url': source_url,
                'content_length': len(content),
                'has_required_metadata': len(missing_fields) == 0
            })

        # Calculate metrics
        if scores:
            validation_report['metrics']['avg_similarity_score'] = sum(scores) / len(scores)
            validation_report['metrics']['min_similarity_score'] = min(scores)
            validation_report['metrics']['max_similarity_score'] = max(scores)

        validation_report['metrics']['unique_sources'] = len(source_urls)
        validation_report['metrics']['metadata_completeness'] = (
            metadata_complete_count / len(results) if results else 0
        )

        # Overall validation
        if validation_report['metrics']['metadata_completeness'] < 0.8:
            validation_report['is_valid'] = False
            validation_report['issues'].append(
                f"Low metadata completeness: {validation_report['metrics']['metadata_completeness']:.2%}"
            )

        if validation_report['metrics']['avg_similarity_score'] < 0.5:
            validation_report['is_valid'] = False
            validation_report['issues'].append(
                f"Low average similarity score: {validation_report['metrics']['avg_similarity_score']:.3f}"
            )

        # Log validation summary
        self.logger.info(
            f"Validation completed for query '{query}'. "
            f"Valid: {validation_report['is_valid']}, "
            f"Results: {validation_report['total_results']}, "
            f"Avg Score: {validation_report['metrics']['avg_similarity_score']:.3f}, "
            f"Unique Sources: {validation_report['metrics']['unique_sources']}"
        )

        return validation_report

    def search_and_validate(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Perform retrieval and validation in a single operation

        Args:
            query: The search query string
            limit: Maximum number of results to return (default: 5)

        Returns:
            Dictionary containing both retrieval results and validation report
        """
        self.logger.info(f"Performing search and validation for query: '{query}'")

        try:
            # Retrieve chunks
            results = self.retrieve_chunks(query, limit)

            # Validate results
            validation_report = self.validate_retrieval(query, results)

            # Combine results
            combined_result = {
                'query': query,
                'retrieval_results': results,
                'validation_report': validation_report,
                'is_successful': True
            }

            self.logger.info(f"Search and validation completed successfully for query: '{query}'")
            return combined_result

        except Exception as e:
            self.logger.error(f"Error in search and validation for query '{query}': {str(e)}")

            return {
                'query': query,
                'retrieval_results': [],
                'validation_report': {
                    'query': query,
                    'total_results': 0,
                    'validation_timestamp': datetime.now().isoformat(),
                    'is_valid': False,
                    'issues': [f"Retrieval error: {str(e)}"],
                    'metrics': {},
                    'validation_details': []
                },
                'is_successful': False,
                'error': str(e)
            }

    def validate_source_consistency(self, results: List[Dict[str, Any]]) -> bool:
        """
        Validate that all results have consistent source metadata

        Args:
            results: List of retrieval results

        Returns:
            Boolean indicating if all sources are consistent
        """
        for result in results:
            payload = result.get('payload', {})
            source_url = payload.get('source_url', '')

            if not source_url or not source_url.startswith('http'):
                return False

        return True


def main():
    """
    Main function to demonstrate retrieval and validation functionality
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Initialize the retrieval validator
    logger.info("Initializing RetrievalValidator...")
    validator = RetrievalValidator()

    # Example queries for testing
    test_queries = [
        "AI Robotics",
        "ROS 2 fundamentals",
        "Docusaurus setup",
        "Python agents in ROS 2"
    ]

    logger.info("Starting retrieval and validation tests...")

    for query in test_queries:
        logger.info(f"\n--- Testing query: '{query}' ---")

        # Perform search and validation
        result = validator.search_and_validate(query, limit=3)

        # Print results
        print(f"\nQuery: '{query}'")
        print(f"Successful: {result['is_successful']}")
        print(f"Total results: {len(result['retrieval_results'])}")
        print(f"Validation valid: {result['validation_report']['is_valid']}")

        if result['is_successful'] and result['retrieval_results']:
            print("\nTop results:")
            for i, res in enumerate(result['retrieval_results'][:2]):  # Show first 2 results
                payload = res.get('payload', {})
                source = payload.get('source_url', 'N/A')
                score = res.get('score', 0)
                content_preview = payload.get('content', '')[:100] + '...' if len(payload.get('content', '')) > 100 else payload.get('content', '')

                print(f"  {i+1}. Score: {score:.3f}")
                print(f"     Source: {source}")
                print(f"     Content: {content_preview}")
                print()

        if not result['validation_report']['is_valid']:
            print(f"Validation issues: {result['validation_report']['issues']}")

        print("-" * 50)


if __name__ == "__main__":
    main()