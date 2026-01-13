from typing import List, Dict, Any, Set
import hashlib
import logging
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

class DuplicateService:
    """
    Service for detecting duplicates and ensuring idempotency
    """

    def __init__(self):
        """
        Initialize the duplicate detection service
        """
        pass

    def detect_duplicates_by_content(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Detect duplicate chunks based on content
        """
        seen_hashes: Set[str] = set()
        duplicates = []
        unique_chunks = []

        for chunk in chunks:
            # Create a hash of the content and source URL to identify duplicates
            content_hash = hashlib.sha256(f"{chunk.content}_{chunk.source_url}".encode('utf-8')).hexdigest()

            if content_hash in seen_hashes:
                duplicates.append({
                    'chunk_id': chunk.id,
                    'content_hash': content_hash,
                    'source_url': chunk.source_url,
                    'content_preview': chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                })
            else:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        logger.info(f"Found {len(duplicates)} duplicate chunks out of {len(chunks)} total")
        return duplicates

    def remove_duplicates_by_content(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Remove duplicate chunks based on content
        """
        seen_hashes: Set[str] = set()
        unique_chunks = []

        for chunk in chunks:
            # Create a hash of the content and source URL to identify duplicates
            content_hash = hashlib.sha256(f"{chunk.content}_{chunk.source_url}".encode('utf-8')).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        return unique_chunks

    def detect_duplicates_by_id(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Detect duplicate chunks based on ID
        """
        seen_ids: Set[str] = set()
        duplicate_ids = []

        for chunk in chunks:
            if chunk.id in seen_ids:
                duplicate_ids.append(chunk.id)
            else:
                seen_ids.add(chunk.id)

        logger.info(f"Found {len(duplicate_ids)} duplicate IDs")
        return duplicate_ids

    def generate_unique_chunk_id(self, content: str, source_url: str, index: int = 0) -> str:
        """
        Generate a unique chunk ID based on content and source URL
        """
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        url_hash = hashlib.md5(source_url.encode('utf-8')).hexdigest()
        return f"{url_hash[:8]}_{content_hash[:16]}_{index}"

    def ensure_idempotency(self, chunks: List[DocumentChunk], existing_ids: Set[str] = None) -> List[DocumentChunk]:
        """
        Ensure idempotency by filtering out chunks that already exist
        """
        if existing_ids is None:
            existing_ids = set()

        # Filter out chunks that already exist based on ID
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]

        logger.info(f"Filtered {len(chunks) - len(new_chunks)} existing chunks for idempotency")
        return new_chunks

    def get_content_fingerprint(self, content: str, source_url: str) -> str:
        """
        Get a fingerprint for content that can be used for duplicate detection
        """
        combined = f"{content}_{source_url}_len{len(content)}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def detect_similar_chunks(self, chunks: List[DocumentChunk], threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Detect similar chunks based on content similarity
        For simplicity, we'll use exact content matching here
        In practice, you might want to use more sophisticated similarity algorithms
        """
        similar_pairs = []
        processed = set()

        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks):
                if i >= j:  # Avoid duplicate comparisons and self-comparison
                    continue

                if (chunk1.id, chunk2.id) in processed or (chunk2.id, chunk1.id) in processed:
                    continue

                # For now, we'll just check exact content matches
                # In a real implementation, you might use cosine similarity on embeddings
                if chunk1.content == chunk2.content:
                    similarity = 1.0  # Exact match
                    if similarity >= threshold:
                        similar_pairs.append({
                            'chunk1_id': chunk1.id,
                            'chunk2_id': chunk2.id,
                            'similarity': similarity,
                            'source_url1': chunk1.source_url,
                            'source_url2': chunk2.source_url,
                            'content_preview': chunk1.content[:100] + "..." if len(chunk1.content) > 100 else chunk1.content
                        })
                        processed.add((chunk1.id, chunk2.id))

        logger.info(f"Found {len(similar_pairs)} similar chunk pairs with similarity >= {threshold}")
        return similar_pairs

    def validate_idempotency(self, chunks: List[DocumentChunk], existing_chunks: List[DocumentChunk] = None) -> Dict[str, Any]:
        """
        Validate idempotency by checking if processing the same chunks would result in duplicates
        """
        if existing_chunks is None:
            existing_chunks = []

        # Get content fingerprints of existing chunks
        existing_fingerprints = set()
        for chunk in existing_chunks:
            fingerprint = self.get_content_fingerprint(chunk.content, chunk.source_url)
            existing_fingerprints.add(fingerprint)

        # Check current chunks against existing
        new_fingerprints = set()
        duplicates_found = 0
        for chunk in chunks:
            fingerprint = self.get_content_fingerprint(chunk.content, chunk.source_url)
            if fingerprint in existing_fingerprints:
                duplicates_found += 1
            else:
                new_fingerprints.add(fingerprint)

        stats = {
            'total_chunks': len(chunks),
            'duplicates_with_existing': duplicates_found,
            'new_unique_chunks': len(new_fingerprints),
            'idempotency_safe': duplicates_found == 0  # If no duplicates with existing, it's safe
        }

        logger.info(f"Idempotency validation: {stats}")
        return stats

    def create_idempotency_key(self, operation_type: str, source_url: str, content_hash: str) -> str:
        """
        Create an idempotency key for tracking operations
        """
        combined = f"{operation_type}_{source_url}_{content_hash}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def filter_processed_chunks(self, chunks: List[DocumentChunk], processed_keys: Set[str]) -> List[DocumentChunk]:
        """
        Filter out chunks that have already been processed based on idempotency keys
        """
        remaining_chunks = []

        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()
            idempotency_key = self.create_idempotency_key('embedding', chunk.source_url, content_hash)

            if idempotency_key not in processed_keys:
                remaining_chunks.append(chunk)

        logger.info(f"Filtered out {len(chunks) - len(remaining_chunks)} already processed chunks")
        return remaining_chunks