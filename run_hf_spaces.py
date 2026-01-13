#!/usr/bin/env python3
"""
Hugging Face Space Startup Script

This script is designed to run the RAG Chatbot backend on Hugging Face Spaces.
It handles the initialization of the Qdrant vector store with content from the
Docusaurus book and starts the FastAPI server.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent))

from agent import app
from main import run_ingestion_pipeline, create_services, validate_configuration
from config import get_config
import uvicorn

def main():
    """
    Main function to run the RAG Chatbot backend on Hugging Face Spaces
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting RAG Chatbot Backend for Hugging Face Spaces...")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_config()

        # Validate configuration
        logger.info("Validating configuration...")
        if not validate_configuration(config):
            logger.error("Configuration validation failed")
            sys.exit(1)

        # Create services
        logger.info("Creating services...")
        services = create_services(config)

        # Run ingestion pipeline to populate vector store
        # This will crawl the Docusaurus book and store embeddings in Qdrant
        logger.info("Running ingestion pipeline to populate vector store...")
        success = run_ingestion_pipeline(services, resume=False)

        if not success:
            logger.warning("Ingestion pipeline did not complete successfully, continuing anyway...")

        # Determine the port for Hugging Face Spaces
        port = int(os.getenv("PORT", 8000))
        logger.info(f"Starting server on port {port}")

        # Start the FastAPI server
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            "agent:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )

    except Exception as e:
        logger.error(f"Error starting RAG Chatbot Backend: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()