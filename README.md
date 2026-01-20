# RAG Chatbot Backend for Hugging Face Spaces

This is the backend for a Retrieval-Augmented Generation (RAG) chatbot that integrates with Qdrant vector database and Cohere embeddings to provide contextual responses based on documentation.

## Overview

This service implements a pipeline that:
1. Crawls deployed Docusaurus book URLs
2. Extracts clean text content from pages
3. Chunks text and generates embeddings using Cohere
4. Stores vectors with metadata in Qdrant
5. Provides a FastAPI endpoint for querying the RAG system

## Features

- FastAPI-based REST API
- Vector storage with Qdrant
- Cohere embeddings for semantic search
- Integration with OpenRouter for LLM responses
- Configurable retrieval parameters

## Prerequisites

- Python 3.11+

## Deployment on Hugging Face Spaces

### Docker Configuration

The application is configured to run on port 7860 as required by Hugging Face Spaces.

### Secrets Configuration

You need to set up the following secrets in your Hugging Face Space settings:

- `COHERE_API_KEY`: Your Cohere API key for generating embeddings
- `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM access
- `QDRANT_API_KEY`: Your Qdrant API key
- `QDRANT_URL`: Your Qdrant cluster URL
- `DOCUSAURUS_URL`: URL of the documentation to index

### Environment Variables

The following environment variables can be configured in your Hugging Face Space settings:

- `QDRANT_COLLECTION_NAME`: Name of the Qdrant collection (default: humanoid_ai_book)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `CRAWL_DELAY`: Delay between crawl requests (default: 1.0)
- `MAX_DEPTH`: Maximum depth for crawling (default: 5)

## API Endpoints

- `POST /ask`: Submit a query and receive a response with sources
- `GET /health`: Health check endpoint

## Architecture

The backend follows a RAG (Retrieval Augmented Generation) pattern:

1. User submits a query
2. System retrieves relevant document chunks from Qdrant based on semantic similarity
3. Retrieved context is passed to the LLM to generate a contextual response
4. Response includes sources and confidence metrics

## Local Development

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Create environment file:
   ```bash
   cp .env.example .env
   ```

3. Add your API keys to `.env`:
   - `COHERE_API_KEY`: Your Cohere API key
   - `QDRANT_URL`: Your Qdrant cluster URL
   - `QDRANT_API_KEY`: Your Qdrant API key
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `DOCUSAURUS_URL`: URL of the Docusaurus book to crawl

4. Run the ingestion pipeline and start the server:
   ```bash
   uv run python run_hf_spaces.py
   ```