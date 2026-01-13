from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class DocumentChunk:
    """
    A segment of text extracted from a Docusaurus page with associated metadata
    """
    id: str
    content: str
    source_url: str
    document_hierarchy: str
    metadata: Dict[str, Any]
    embedding: Optional[list] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document chunk to a dictionary representation
        """
        return {
            "id": self.id,
            "content": self.content,
            "source_url": self.source_url,
            "document_hierarchy": self.document_hierarchy,
            "metadata": self.metadata,
            "embedding": self.embedding
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """
        Create a DocumentChunk from a dictionary
        """
        return cls(
            id=data["id"],
            content=data["content"],
            source_url=data["source_url"],
            document_hierarchy=data["document_hierarchy"],
            metadata=data["metadata"],
            embedding=data.get("embedding")
        )

    def validate(self) -> bool:
        """
        Validate the document chunk
        """
        if not self.id:
            raise ValueError("Document chunk must have an ID")

        if not self.content or len(self.content.strip()) == 0:
            raise ValueError("Document chunk content cannot be empty")

        if not self.source_url:
            raise ValueError("Document chunk must have a source URL")

        if not self.document_hierarchy:
            raise ValueError("Document chunk must have document hierarchy")

        # Check content length
        if len(self.content) > 10000:  # Reasonable limit
            raise ValueError("Document chunk content is too long")

        return True

    def get_content_length(self) -> int:
        """
        Get the length of the content
        """
        return len(self.content) if self.content else 0

    def has_embedding(self) -> bool:
        """
        Check if the chunk has an embedding
        """
        return self.embedding is not None and len(self.embedding) > 0