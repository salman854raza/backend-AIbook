"""
Hugging Face Space App Entry Point

This file serves as the entry point for the Hugging Face Space deployment.
It creates a Gradio interface for the RAG chatbot or simply runs the FastAPI backend.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent))

from agent import app, AIAgent
import uvicorn
from fastapi.staticfiles import StaticFiles
import argparse

def run_app():
    """Run the FastAPI application"""
    # Check if we're running in a Hugging Face Space environment
    is_space = os.getenv("SPACE_ID") is not None

    if is_space:
        # In Hugging Face Space, use the PORT environment variable
        port = int(os.getenv("PORT", 7860))
        host = "0.0.0.0"
    else:
        # Local development
        port = int(os.getenv("PORT", 7860))
        host = "0.0.0.0"

    print(f"Starting server on {host}:{port}")
    print(f"Space environment: {is_space}")

    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "agent:app",
        host=host,
        port=port,
        reload=False  # Disable reload in production
    )

if __name__ == "__main__":
    run_app()