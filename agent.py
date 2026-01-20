"""
AI Agent with Retrieval Integration

This module implements an AI agent that integrates with Qdrant for
retrieval-augmented generation (RAG). The agent retrieves relevant
document chunks from Qdrant and generates responses grounded in
the retrieved context using the Google Gemini API.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from clients.qdrant_client import QdrantService
from clients.cohere_client import CohereService
from config import get_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentQueryRequest(BaseModel):
    """Request model for the agent query endpoint"""
    query: str
    max_chunks: int = 5
    min_score: float = 0.5


class SourceInfo(BaseModel):
    """Information about a source used in the response"""
    source_url: str
    similarity_score: float
    content: str


class RetrievalInfo(BaseModel):
    """Information about the retrieval process"""
    chunks_count: int
    avg_similarity: float
    processing_time: float


class AgentQueryResponse(BaseModel):
    """Response model for the agent query endpoint"""
    query: str
    answer: str
    sources: List[SourceInfo]
    retrieval_info: RetrievalInfo
    grounding_confidence: float
    timestamp: str


class RetrievalTool:
    """Tool for retrieving relevant chunks from Qdrant based on user queries"""

    def __init__(self, qdrant_service: QdrantService, cohere_service: CohereService):
        """
        Initialize the retrieval tool with required services

        Args:
            qdrant_service: Service for interacting with Qdrant vector database
            cohere_service: Service for generating embeddings with Cohere
        """
        self.qdrant_service = qdrant_service
        self.cohere_service = cohere_service
        self.logger = logging.getLogger(__name__)

    def retrieve_context(self, query: str, max_chunks: int = 5, min_score: float = 0.5) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks from Qdrant based on the query

        Args:
            query: The user's query string
            max_chunks: Maximum number of chunks to retrieve
            min_score: Minimum similarity score for inclusion

        Returns:
            Dictionary containing retrieved chunks and metadata
        """
        self.logger.info(f"Retrieving context for query: '{query[:50]}...'")

        start_time = datetime.now()

        try:
            # Generate embedding for the query using Cohere
            query_embedding = self.cohere_service.generate_single_embedding(query)
            self.logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

            # Search in Qdrant for similar vectors
            results = self.qdrant_service.search_similar(query_embedding, limit=max_chunks)

            # Filter results by minimum score and extract relevant information
            filtered_results = []
            total_score = 0.0

            for result in results:
                score = result.get('score', 0)

                if score >= min_score:
                    payload = result.get('payload', {})
                    filtered_results.append({
                        'id': result.get('id'),
                        'content': payload.get('content', ''),
                        'score': score,
                        'source_url': payload.get('source_url', ''),
                        'document_hierarchy': payload.get('document_hierarchy', '')
                    })
                    total_score += score

            # Calculate average similarity score
            avg_similarity = total_score / len(filtered_results) if filtered_results else 0.0

            processing_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"Retrieved {len(filtered_results)} relevant chunks for query: '{query[:30]}...'")

            return {
                'chunks': filtered_results,
                'query_embedding': query_embedding,
                'retrieval_score_threshold': min_score,
                'total_retrieved': len(filtered_results),
                'avg_similarity': avg_similarity,
                'processing_time': processing_time
            }

        except Exception as e:
            self.logger.error(f"Error retrieving context for query '{query[:30]}...': {str(e)}")
            raise


class AIAgent:
    """AI Agent with retrieval capabilities using Google Gemini and RAG"""

    def __init__(self):
        """
        Initialize the AI agent with required services
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

        # Initialize the retrieval tool
        self.retrieval_tool = RetrievalTool(self.qdrant_service, self.cohere_service)

        # Configure OpenRouter API
        self.client = OpenAI(
            api_key=self.config.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # Initialize the OpenRouter model
        self.model_name = "mistralai/mistral-7b-instruct:free"  # Using a free model from OpenRouter

        self.logger.info("AI Agent initialized successfully")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on the query and retrieved context

        Args:
            query: The user's query
            context_chunks: Retrieved context chunks to ground the response

        Returns:
            Generated answer string
        """
        self.logger.info(f"Generating answer for query: '{query[:50]}...'")

        try:
            # Construct the prompt with context
            context_text = ""
            for i, chunk in enumerate(context_chunks):
                context_text += f"\n\nContext Chunk {i+1} (Score: {chunk['score']:.3f}, Source: {chunk['source_url']}):\n{chunk['content']}\n"

            # Create a system message that emphasizes using only the provided context
            system_message = (
                "You are a helpful AI assistant that answers questions based ONLY on the provided context. "
                "Do not use any prior knowledge or information not contained in the provided context. "
                "If the context does not contain information to answer the question, say so explicitly. "
                "Always cite the sources when providing information."
            )

            # Create the messages for OpenAI API format
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based ONLY on the provided context. "
                               "Do not use any prior knowledge or information not contained in the provided context. "
                               "If the context does not contain information to answer the question, say so explicitly. "
                               "Always cite the sources when providing information."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context_text}\n\nAnswer:"
                }
            ]

            try:
                # Generate content using OpenRouter API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent responses
                    max_tokens=1000
                )

                answer = response.choices[0].message.content.strip()
            except Exception as e:
                # Handle rate limit and other API errors
                if "429" in str(e) or "rate" in str(e).lower() or "limit" in str(e).lower():
                    # If rate limited, return a response based on the context without LLM
                    self.logger.warning(f"Rate limited by OpenRouter API: {str(e)}")
                    # Create a simple answer using the retrieved context
                    if context_chunks:
                        context_preview = " ".join([chunk['content'][:200] for chunk in context_chunks[:2]])
                        answer = f"Due to API rate limits, here's a preview of the relevant information I found: {context_preview}... Please try again later for a complete response."
                    else:
                        answer = "Due to API rate limits, I cannot generate a full response. The system is working but the LLM service is temporarily unavailable due to usage limits."
                else:
                    # For other errors, raise the exception
                    raise e
            self.logger.info(f"Generated answer with {len(answer)} characters")

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer for query '{query[:30]}...': {str(e)}")
            raise

    def calculate_grounding_confidence(self, context_chunks: List[Dict[str, Any]], min_score_threshold: float = 0.7) -> float:
        """
        Calculate the confidence in how well the response is grounded in the context

        Args:
            context_chunks: Retrieved context chunks
            min_score_threshold: Minimum score to consider as high quality

        Returns:
            Grounding confidence score between 0.0 and 1.0
        """
        if not context_chunks:
            return 0.0

        # Calculate average score
        avg_score = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)

        # Calculate percentage of chunks above threshold
        high_quality_chunks = sum(1 for chunk in context_chunks if chunk['score'] >= min_score_threshold)
        high_quality_ratio = high_quality_chunks / len(context_chunks)

        # Combine metrics for overall confidence
        # Weight score average (0.6) and high quality ratio (0.4)
        grounding_confidence = (avg_score * 0.6) + (high_quality_ratio * 0.4)

        return min(1.0, grounding_confidence)  # Cap at 1.0

    def ask_question(self, query: str, max_chunks: int = 5, min_score: float = 0.5) -> Dict[str, Any]:
        """
        Process a user question through the RAG pipeline

        Args:
            query: The user's question
            max_chunks: Maximum number of chunks to retrieve
            min_score: Minimum similarity score for inclusion

        Returns:
            Dictionary containing the answer, sources, and metadata
        """
        self.logger.info(f"Processing question: '{query[:50]}...'")

        start_time = datetime.now()

        try:
            # Step 1: Retrieve relevant context
            retrieval_result = self.retrieval_tool.retrieve_context(
                query, max_chunks=max_chunks, min_score=min_score
            )

            # Step 2: Generate answer based on context
            answer = self.generate_answer(query, retrieval_result['chunks'])

            # Step 3: Calculate grounding confidence
            grounding_confidence = self.calculate_grounding_confidence(
                retrieval_result['chunks']
            )

            # Step 4: Format sources
            sources = []
            for chunk in retrieval_result['chunks']:
                sources.append(SourceInfo(
                    source_url=chunk['source_url'],
                    similarity_score=chunk['score'],
                    content=chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                ))

            # Step 5: Calculate total processing time
            total_processing_time = (datetime.now() - start_time).total_seconds()

            # Prepare response
            response = {
                'query': query,
                'answer': answer,
                'sources': sources,
                'retrieval_info': RetrievalInfo(
                    chunks_count=retrieval_result['total_retrieved'],
                    avg_similarity=retrieval_result['avg_similarity'],
                    processing_time=retrieval_result['processing_time']
                ),
                'grounding_confidence': grounding_confidence,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"Processed question successfully in {total_processing_time:.2f}s")
            return response

        except Exception as e:
            self.logger.error(f"Error processing question '{query[:30]}...': {str(e)}")
            raise


# Initialize FastAPI app
app = FastAPI(
    title="AI Agent with Retrieval Integration API",
    description="API for AI agent with retrieval-augmented generation capabilities",
    version="1.0.0"
)

# Initialize the AI agent
ai_agent = AIAgent()


@app.post("/ask", response_model=AgentQueryResponse)
async def ask_endpoint(request: AgentQueryRequest):
    """
    Endpoint to ask a question to the AI agent

    Args:
        request: Query request with parameters

    Returns:
        Response with answer and sources
    """
    logger.info(f"Received query via API: '{request.query[:50]}...'")

    try:
        # Process the query through the agent
        result = ai_agent.ask_question(
            query=request.query,
            max_chunks=request.max_chunks,
            min_score=request.min_score
        )

        # Convert to response model
        response = AgentQueryResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            retrieval_info=result['retrieval_info'],
            grounding_confidence=result['grounding_confidence'],
            timestamp=result['timestamp']
        )

        logger.info(f"API query processed successfully")
        return response

    except Exception as e:
        logger.error(f"Error processing API query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Agent with Retrieval Integration"
    }


def main():
    """
    Main function to run the API server
    """
    import uvicorn

    logger.info("Starting AI Agent API server...")

    # Run the server
    uvicorn.run(
        "agent:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )


if __name__ == "__main__":
    main()