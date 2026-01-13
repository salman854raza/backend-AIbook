from typing import Dict, Any, List
import hashlib
import time
from datetime import datetime
from models.document_chunk import DocumentChunk

class MetadataService:
    """
    Service for preserving and managing metadata for document chunks
    """

    def __init__(self):
        """
        Initialize the metadata service
        """
        pass

    def generate_chunk_id(self, content: str, source_url: str, chunk_index: int = 0) -> str:
        """
        Generate a unique ID for a document chunk
        """
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        url_hash = hashlib.md5(source_url.encode('utf-8')).hexdigest()
        return f"{url_hash[:8]}_{content_hash[:8]}_{chunk_index}"

    def create_metadata(self,
                      source_url: str,
                      document_hierarchy: str,
                      title: str = "",
                      headings: List[Dict[str, Any]] = None,
                      additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create metadata for a document chunk
        """
        if headings is None:
            headings = []
        if additional_metadata is None:
            additional_metadata = {}

        metadata = {
            'source_url': source_url,
            'document_hierarchy': document_hierarchy,
            'title': title,
            'headings': headings,
            'created_at': datetime.now().isoformat(),
            'processed_at': time.time(),
            'content_length': len(title) if title else 0,
            'hash': hashlib.sha256((source_url + title).encode('utf-8')).hexdigest()
        }

        # Add additional metadata
        metadata.update(additional_metadata)

        return metadata

    def preserve_metadata_for_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Ensure proper metadata is preserved for a document chunk
        """
        # Generate a unique ID if not present
        if not chunk.id:
            chunk.id = self.generate_chunk_id(
                chunk.content,
                chunk.source_url,
                chunk.metadata.get('chunk_index', 0)
            )

        # Ensure required metadata fields are present
        if 'created_at' not in chunk.metadata:
            chunk.metadata['created_at'] = datetime.now().isoformat()

        if 'processed_at' not in chunk.metadata:
            chunk.metadata['processed_at'] = time.time()

        if 'content_length' not in chunk.metadata:
            chunk.metadata['content_length'] = len(chunk.content)

        if 'hash' not in chunk.metadata:
            chunk.metadata['hash'] = hashlib.sha256(
                (chunk.source_url + chunk.content).encode('utf-8')
            ).hexdigest()

        # Update the content length in metadata
        chunk.metadata['content_length'] = len(chunk.content)

        return chunk

    def validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Validate metadata and return list of validation errors
        """
        errors = []

        required_fields = ['source_url', 'document_hierarchy', 'created_at']
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")

        if 'source_url' in metadata and not metadata['source_url']:
            errors.append("source_url cannot be empty")

        if 'document_hierarchy' in metadata and not metadata['document_hierarchy']:
            errors.append("document_hierarchy cannot be empty")

        return errors

    def merge_metadata(self,
                     base_metadata: Dict[str, Any],
                     additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge additional metadata into base metadata
        """
        merged = base_metadata.copy()
        merged.update(additional_metadata)
        return merged

    def extract_hierarchy_from_url(self, url: str) -> str:
        """
        Extract document hierarchy from URL
        """
        # Remove protocol and domain, keep only the path
        if '//' in url:
            path = url.split('//', 1)[1]
            path = path.split('/', 1)[1] if '/' in path else ''
        else:
            path = url

        # Remove query parameters and fragments
        path = path.split('?')[0].split('#')[0]

        # Split path into components and create hierarchy
        components = [comp for comp in path.split('/') if comp and comp != 'index.html']

        if not components:
            return "Home"

        return " > ".join(components)

    def normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata to ensure consistency
        """
        normalized = metadata.copy()

        # Ensure source_url is normalized
        if 'source_url' in normalized:
            normalized['source_url'] = normalized['source_url'].strip().rstrip('/')

        # Ensure document_hierarchy is properly formatted
        if 'document_hierarchy' in normalized:
            normalized['document_hierarchy'] = normalized['document_hierarchy'].strip()

        # Ensure created_at is in ISO format
        if 'created_at' in normalized and not isinstance(normalized['created_at'], str):
            normalized['created_at'] = datetime.fromtimestamp(normalized['created_at']).isoformat()

        return normalized

    def create_embedding_payload(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Create a payload suitable for embedding storage in vector database
        """
        return {
            'source_url': chunk.source_url,
            'document_hierarchy': chunk.document_hierarchy,
            'content': chunk.content,
            'chunk_index': chunk.metadata.get('chunk_index', 0),
            'created_at': chunk.metadata.get('created_at', datetime.now().isoformat()),
            'title': chunk.metadata.get('title', ''),
            'headings': chunk.metadata.get('headings', []),
            'content_length': len(chunk.content),
            'hash': chunk.metadata.get('hash', ''),
            'chunk_id': chunk.id
        }

    def update_metadata_from_processing(self,
                                     chunk: DocumentChunk,
                                     processing_info: Dict[str, Any]) -> DocumentChunk:
        """
        Update chunk metadata with information from processing
        """
        chunk.metadata.update(processing_info)
        return chunk

    def get_metadata_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get a summary of metadata across all chunks
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'unique_urls': 0,
                'unique_hierarchies': 0,
                'avg_content_length': 0,
                'date_range': None
            }

        unique_urls = set(chunk.source_url for chunk in chunks)
        unique_hierarchies = set(chunk.document_hierarchy for chunk in chunks)
        total_content_length = sum(len(chunk.content) for chunk in chunks)

        # Find date range
        created_dates = []
        for chunk in chunks:
            created_at = chunk.metadata.get('created_at')
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_dates.append(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
                    else:
                        created_dates.append(datetime.fromtimestamp(created_at))
                except:
                    pass

        date_range = None
        if created_dates:
            date_range = {
                'start': min(created_dates).isoformat(),
                'end': max(created_dates).isoformat()
            }

        return {
            'total_chunks': len(chunks),
            'unique_urls': len(unique_urls),
            'unique_hierarchies': len(unique_hierarchies),
            'avg_content_length': total_content_length / len(chunks) if chunks else 0,
            'date_range': date_range
        }