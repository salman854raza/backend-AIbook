from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class EmbeddingRecord:
    """
    A vector representation of document content stored in Qdrant
    """
    id: str
    vector: List[float]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the embedding record to a dictionary representation
        """
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingRecord':
        """
        Create an EmbeddingRecord from a dictionary
        """
        return cls(
            id=data["id"],
            vector=data["vector"],
            payload=data["payload"]
        )

    def validate(self) -> bool:
        """
        Validate the embedding record
        """
        if not self.id:
            raise ValueError("Embedding record must have an ID")

        if not self.vector or len(self.vector) == 0:
            raise ValueError("Embedding record must have a vector")

        if not isinstance(self.vector, list):
            raise ValueError("Vector must be a list of floats")

        if not all(isinstance(x, (int, float)) for x in self.vector):
            raise ValueError("Vector must contain only numeric values")

        if not self.payload:
            raise ValueError("Embedding record must have payload data")

        if not isinstance(self.payload, dict):
            raise ValueError("Payload must be a dictionary")

        return True

    def get_vector_dimension(self) -> int:
        """
        Get the dimension of the embedding vector
        """
        return len(self.vector) if self.vector else 0

    def has_valid_payload(self) -> bool:
        """
        Check if the payload contains required fields
        """
        required_fields = ['source_url', 'document_hierarchy', 'content']
        return all(field in self.payload for field in required_fields)