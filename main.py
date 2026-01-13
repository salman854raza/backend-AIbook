#!/usr/bin/env python3
"""
RAG Chatbot Backend - Main Ingestion Pipeline

This script implements the complete pipeline for:
1. Crawling Docusaurus book URLs
2. Extracting clean text content
3. Chunking text and generating embeddings
4. Storing vectors with metadata in Qdrant
"""

import argparse
import sys
import os
from typing import List, Dict, Any
import logging
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, Config
from logging_config import setup_logging, log_progress, create_progress_callback
from crawlers.web_crawler import WebCrawler
from extractors.html_extractor import HTMLExtractor
from processors.text_chunker import TextChunker
from crawlers.url_discovery import URLDiscovery
from services.crawl_service import CrawlService
from services.metadata_service import MetadataService
from clients.cohere_client import CohereService
from clients.qdrant_client import QdrantService
from services.embedding_service import EmbeddingService
from services.vector_storage_service import VectorStorageService
from services.vector_service import VectorService
from services.duplicate_service import DuplicateService
from validators.embedding_validator import EmbeddingValidator
from services.state_service import StateService, PipelineState
from services.resume_service import ResumeService
from models.crawl_session import CrawlSession
from services.error_service import ErrorService, retry_on_failure
from services.checkpoint_service import CheckpointService
from services.metrics_service import MetricsService
from validators.input_validator import InputValidator
from crawlers.rate_limiter import CrawlRateLimiter


def create_services(config: Config) -> Dict[str, Any]:
    """
    Create and configure all services needed for the pipeline
    """
    # Initialize clients
    cohere_service = CohereService(config.cohere_api_key)
    qdrant_service = QdrantService(config.qdrant_url, config.qdrant_api_key, config.qdrant_collection_name)

    # Initialize services
    metadata_service = MetadataService()
    embedding_service = EmbeddingService(cohere_service)
    vector_storage_service = VectorStorageService(qdrant_service)
    vector_service = VectorService(embedding_service, vector_storage_service)
    duplicate_service = DuplicateService()
    embedding_validator = EmbeddingValidator()
    state_service = StateService()
    checkpoint_service = CheckpointService()
    metrics_service = MetricsService()
    error_service = ErrorService()
    rate_limiter = CrawlRateLimiter(config.crawl_delay)

    # Initialize crawl service
    crawl_service = CrawlService(
        config.docusaurus_url,
        config.crawl_delay,
        config.max_depth,
        config.chunk_size,
        config.chunk_overlap
    )

    # Initialize resume service
    resume_service = ResumeService(state_service, crawl_service, vector_service)

    return {
        'config': config,
        'cohere_service': cohere_service,
        'qdrant_service': qdrant_service,
        'metadata_service': metadata_service,
        'embedding_service': embedding_service,
        'vector_storage_service': vector_storage_service,
        'vector_service': vector_service,
        'duplicate_service': duplicate_service,
        'embedding_validator': embedding_validator,
        'state_service': state_service,
        'checkpoint_service': checkpoint_service,
        'metrics_service': metrics_service,
        'error_service': error_service,
        'crawl_service': crawl_service,
        'resume_service': resume_service,
        'rate_limiter': rate_limiter
    }


def validate_configuration(config: Config) -> bool:
    """
    Validate the configuration before starting the pipeline
    """
    logger = logging.getLogger(__name__)
    validator = InputValidator()

    config_dict = {
        'docusaurus_url': config.docusaurus_url,
        'cohere_api_key': config.cohere_api_key,
        'qdrant_url': config.qdrant_url,
        'qdrant_api_key': config.qdrant_api_key,
        'qdrant_collection_name': config.qdrant_collection_name,
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'crawl_delay': config.crawl_delay,
        'max_depth': config.max_depth
    }

    errors = validator.validate_docusaurus_config(config_dict)

    if errors:
        logger.error(f"Configuration validation failed with {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    logger.info("Configuration validation passed")
    return True


def create_collection_if_needed(qdrant_service: QdrantService, vector_size: int = 1024) -> bool:
    """
    Create the Qdrant collection if it doesn't exist
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Checking if Qdrant collection '{qdrant_service.collection_name}' exists...")

    try:
        created = qdrant_service.create_collection_if_not_exists(vector_size)
        if created:
            logger.info(f"Created new Qdrant collection: {qdrant_service.collection_name}")
        else:
            logger.info(f"Qdrant collection '{qdrant_service.collection_name}' already exists")
        return True
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {str(e)}")
        return False


@retry_on_failure(max_retries=3, delay=2.0)
def run_ingestion_pipeline(services: Dict[str, Any], resume: bool = False) -> bool:
    """
    Run the complete ingestion pipeline
    """
    logger = logging.getLogger(__name__)

    config = services['config']
    crawl_service = services['crawl_service']
    vector_service = services['vector_service']
    state_service = services['state_service']
    resume_service = services['resume_service']
    metrics_service = services['metrics_service']
    error_service = services['error_service']

    logger.info("Starting RAG Chatbot ingestion pipeline...")

    # Check if we should resume from a previous run
    if resume:
        logger.info("Checking for existing pipeline state to resume...")
        resume_result = resume_service.get_resume_recommendation()

        if resume_result['action'] in ['resume', 'check_progress']:
            logger.info("Resuming pipeline from previous state...")
            state = resume_service.resume_pipeline()
            if state:
                logger.info(f"Pipeline resumed successfully. Current status: {state.status}")
                return True
        elif resume_result['action'] == 'completed':
            logger.info("Pipeline already completed successfully.")
            return True

    # Create Qdrant collection if needed with correct vector size from embedding model
    cohere_service = services['cohere_service']
    model_info = cohere_service.get_model_info()
    vector_size = model_info['dimensions']

    if not create_collection_if_needed(services['qdrant_service'], vector_size):
        logger.error("Failed to create Qdrant collection. Exiting.")
        return False

    # Initialize metrics
    metrics_service.reset_session()

    # Start the crawl service with progress tracking
    logger.info(f"Starting crawl of: {config.docusaurus_url}")
    start_time = time.time()

    try:
        # Use the crawl service with progress tracking
        progress_callback = create_progress_callback(logger)
        chunks = crawl_service.crawl_with_progress_callback(progress_callback)

        if not chunks:
            logger.warning("No chunks were created from crawling. Pipeline completed with no data.")
            return True

        logger.info(f"Created {len(chunks)} chunks from crawling")

        # Process chunks through the vector service
        logger.info("Processing chunks through embedding and storage...")
        result = vector_service.process_and_store_chunks(chunks)

        if not result['success']:
            logger.error("Vector processing failed")
            return False

        # Log final metrics
        metrics_service.finalize_session()
        logger.info("Pipeline completed successfully")
        logger.info("\n" + metrics_service.get_summary_report())

        return True

    except Exception as e:
        error_info = error_service.log_error(e, "ingestion_pipeline", config.docusaurus_url)
        logger.error(f"Ingestion pipeline failed: {str(e)}")
        return False


def main():
    """
    Main entry point for the RAG Chatbot backend
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='RAG Chatbot Backend - Docusaurus Content Ingestion Pipeline')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run if available')
    parser.add_argument('--validate-only', action='store_true', help='Validate configuration and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config-file', type=str, help='Path to config file (not implemented in this version)')

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)

    logger.info("RAG Chatbot Backend - Starting ingestion pipeline")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_config()

        # Validate configuration
        logger.info("Validating configuration...")
        if not validate_configuration(config):
            logger.error("Configuration validation failed. Exiting.")
            sys.exit(1)

        # If only validating, exit here
        if args.validate_only:
            logger.info("Configuration validation passed. Exiting.")
            sys.exit(0)

        # Create services
        logger.info("Initializing services...")
        services = create_services(config)

        # Run the ingestion pipeline
        success = run_ingestion_pipeline(services, resume=args.resume)

        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()