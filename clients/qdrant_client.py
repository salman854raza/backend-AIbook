from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(self, url: str, api_key: str, collection_name: str):
        """
        Initialize Qdrant service with connection parameters
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def create_collection_if_not_exists(
        self,
        vector_size: int = 1024,  # Default size for Cohere embeddings
        distance_metric: str = "Cosine"
    ) -> bool:
        """
        Create the collection if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name
                for collection in collections.collections
            )

            if not collection_exists:
                # Create collection with specified vector size
                distance_enum = models.Distance[distance_metric.upper()]

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_enum
                    )
                )

                # Create index for source_url field to enable filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_url",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                logger.info(f"Created Qdrant collection: {self.collection_name}")
                return True
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
                return False

        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            raise

    def upsert_vectors(
        self,
        vector_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> bool:
        """
        Upsert (insert or update) vectors in the collection
        """
        try:
            # Prepare points for upsert
            points = []
            for i, (vector_id, vector, payload) in enumerate(zip(vector_ids, vectors, payloads)):
                point = models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)

            # Upsert the points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Upserted {len(points)} vectors to collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors to Qdrant: {str(e)}")
            raise

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        """
        try:
            from qdrant_client.http import models
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )

            # Extract payload data from results
            search_results = []
            for result in results.points:
                search_results.append({
                    "id": result.id,
                    "score": getattr(result, 'score', 0),  # Use getattr to handle different result formats
                    "payload": result.payload
                })

            return search_results

        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            raise

    def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the collection
        """
        try:
            count = self.client.count(
                collection_name=self.collection_name
            )
            return count.count
        except Exception as e:
            logger.error(f"Error getting vector count from Qdrant: {str(e)}")
            raise

    def check_document_exists(self, source_url: str) -> bool:
        """
        Check if a document with the given source URL already exists in the collection
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url",
                            match=models.MatchValue(value=source_url)
                        )
                    ]
                ),
                limit=1
            )
            return len(records) > 0
        except Exception as e:
            logger.error(f"Error checking document existence in Qdrant: {str(e)}")
            return False

    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution!)
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {str(e)}")
            raise