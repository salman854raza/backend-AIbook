# RAG Chatbot Backend

Backend service for ingesting Docusaurus book content, generating embeddings, and storing them in Qdrant for RAG functionality.

## Overview

This service implements a pipeline that:
1. Crawls deployed Docusaurus book URLs
2. Extracts clean text content from pages
3. Chunks text and generates embeddings using Cohere
4. Stores vectors with metadata in Qdrant

## Prerequisites

- Python 3.11+
- `uv` package manager installed

## Setup

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
   - `DOCUSAURUS_URL`: URL of the Docusaurus book to crawl

## Usage

Run the ingestion pipeline:
```bash
uv run python main.py
```

## Configuration

The pipeline can be configured via environment variables:
- `DOCUSAURUS_URL`: Base URL of the Docusaurus book to crawl (required)
- `CHUNK_SIZE`: Maximum characters per text chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: 100)
- `CRAWL_DELAY`: Delay between requests in seconds (default: 1.0)
- `MAX_DEPTH`: Maximum crawl depth (default: 5)