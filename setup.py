"""
Setup file for RAG Chatbot Backend
Used for Hugging Face Space deployment
"""

from setuptools import setup, find_packages

setup(
    name="rag-chatbot-backend",
    version="1.0.0",
    description="RAG Chatbot Backend for AI Robotics Book",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Engineer",
    author_email="engineer@example.com",
    url="https://github.com/your-username/rag-chatbot-backend",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "cohere>=4.0.0",
        "qdrant-client>=1.7.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "httpx>=0.25.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ]
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "rag-chatbot=app:run_app",
        ],
    },
)