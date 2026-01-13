from typing import List, Dict, Any
import re
import logging
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Text chunking module to split content into meaningful segments
    """

    def __init__(self, default_chunk_size: int = 1000, default_overlap: int = 100):
        """
        Initialize the text chunker
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    def chunk_text(
        self,
        text: str,
        chunk_size: int = None,
        overlap: int = None,
        source_url: str = "",
        document_hierarchy: str = "",
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Split text into chunks of specified size with overlap
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if overlap is None:
            overlap = self.default_overlap

        if not text or len(text.strip()) == 0:
            return []

        if metadata is None:
            metadata = {}

        chunks = []
        start = 0
        text_length = len(text)

        chunk_index = 0
        while start < text_length:
            end = start + chunk_size

            # If this is not the last chunk, try to break at a sentence or paragraph boundary
            if end < text_length:
                # Look for natural break points near the end
                search_start = max(0, end - 200)  # Look back up to 200 chars
                found_break = False

                # Look for paragraph breaks first
                for sep in ['\n\n', '\n\r\n', '. ', '! ', '? ', '; ', ': ', ' - ', ' -- ']:
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        # Include the separator in the chunk
                        end = pos + len(sep)
                        found_break = True
                        break

                # If no good break point found, just use the max position
                if not found_break:
                    # Find the last space to avoid breaking words
                    last_space = text.rfind(' ', search_start, end)
                    if last_space != -1:
                        end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                # Generate a unique integer ID by combining hash of source URL and chunk index
                # Use modulo to keep the ID within a reasonable range for Qdrant
                chunk_id = (abs(hash(source_url)) % 1000000) * 1000 + chunk_index  # Use integer ID

                # Create chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_index': chunk_index,
                    'total_chunks': 0,  # Will be updated after all chunks are created
                    'chunk_size': len(chunk_text)
                })

                chunk = DocumentChunk(
                    id=chunk_id,
                    content=chunk_text,
                    source_url=source_url,
                    document_hierarchy=document_hierarchy,
                    metadata=chunk_metadata
                )

                # Validate the chunk
                try:
                    chunk.validate()
                    chunks.append(chunk)
                except ValueError as e:
                    logger.warning(f"Skipping invalid chunk: {str(e)}")

            # Move to next chunk position with overlap
            start = end - overlap if end < text_length else end

            # If the next chunk would be too small, just finish
            if start < text_length and text_length - start < overlap:
                break

            chunk_index += 1

        # Update total chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_index'] = i

        return chunks

    def chunk_by_headings(
        self,
        text: str,
        headings: List[Dict[str, Any]],
        chunk_size: int = None,
        overlap: int = None,
        source_url: str = "",
        document_hierarchy: str = "",
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Split text based on headings, creating chunks that respect document structure
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if overlap is None:
            overlap = self.default_overlap

        if not text or len(text.strip()) == 0:
            return []

        if not headings:
            # Fall back to regular chunking if no headings
            return self.chunk_text(text, chunk_size, overlap, source_url, document_hierarchy, metadata)

        chunks = []
        chunk_index = 0

        # Create chunks based on heading sections
        for i, heading in enumerate(headings):
            start_pos = heading.get('position', 0)
            end_pos = len(text)  # Default to end of text

            # Find the next heading to determine end position
            if i + 1 < len(headings):
                next_heading = headings[i + 1]
                end_pos = next_heading.get('position', len(text))

            # Extract content for this heading section
            section_content = text[start_pos:end_pos].strip()

            if len(section_content) > chunk_size:
                # If the section is too large, split it further
                sub_chunks = self.chunk_text(
                    section_content,
                    chunk_size,
                    overlap,
                    source_url,
                    f"{document_hierarchy} > {heading['text']}",
                    metadata
                )
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update({
                        'heading_text': heading['text'],
                        'heading_level': heading['level'],
                        'chunk_index': chunk_index
                    })
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                # Create a single chunk for this heading section
                chunk_id = f"{hash(source_url)}_{chunk_index}"

                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'heading_text': heading['text'],
                    'heading_level': heading['level'],
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'chunk_index': chunk_index,
                    'total_chunks': 0,
                    'chunk_size': len(section_content)
                })

                chunk = DocumentChunk(
                    id=chunk_id,
                    content=section_content,
                    source_url=source_url,
                    document_hierarchy=f"{document_hierarchy} > {heading['text']}",
                    metadata=chunk_metadata
                )

                try:
                    chunk.validate()
                    chunks.append(chunk)
                    chunk_index += 1
                except ValueError as e:
                    logger.warning(f"Skipping invalid chunk: {str(e)}")

        # Update total chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_index'] = i

        return chunks

    def validate_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Validate a document chunk
        """
        try:
            return chunk.validate()
        except ValueError as e:
            logger.error(f"Chunk validation failed: {str(e)}")
            return False

    def merge_chunks(self, chunks: List[DocumentChunk]) -> str:
        """
        Merge chunks back into a single text (with overlap handling)
        """
        if not chunks:
            return ""

        # For simplicity, just concatenate all chunks
        # In practice, you might want to handle overlaps more intelligently
        merged_text = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged_text += chunk.content
            else:
                # Remove overlap from the beginning of the current chunk
                prev_chunk_end = chunks[i-1].content
                overlap_len = min(len(prev_chunk_end), len(chunk.content), self.default_overlap)

                # Simple approach: just concatenate (overlaps will be duplicated)
                merged_text += " " + chunk.content

        return merged_text.strip()

    def get_optimal_chunk_size(self, content: str, target_sentences: int = 3) -> int:
        """
        Estimate an optimal chunk size based on target number of sentences
        """
        # Count sentences in the content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return self.default_chunk_size

        avg_sentence_length = len(content) // len(sentences) if sentences else 100
        estimated_size = avg_sentence_length * target_sentences

        return min(estimated_size, self.default_chunk_size)