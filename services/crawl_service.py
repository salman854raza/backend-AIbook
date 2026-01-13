from typing import List, Dict, Any, Optional
import logging
import time
from crawlers.web_crawler import WebCrawler
from extractors.html_extractor import HTMLExtractor
from processors.text_chunker import TextChunker
from crawlers.url_discovery import URLDiscovery
from models.document_chunk import DocumentChunk
from utils import clean_text

logger = logging.getLogger(__name__)

class CrawlService:
    """
    Service for orchestrating the crawling and extraction workflow
    """

    def __init__(self, base_url: str, delay: float = 1.0, max_depth: int = 5, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the crawl service
        """
        self.base_url = base_url
        self.delay = delay
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self.crawler = WebCrawler(base_url, delay, max_depth)
        self.extractor = HTMLExtractor()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.url_discovery = URLDiscovery(base_url)

    def crawl_and_extract(self) -> List[Dict[str, Any]]:
        """
        Crawl the Docusaurus book and extract content
        """
        logger.info(f"Starting crawl and extraction for: {self.base_url}")

        # Discover all URLs in the book
        logger.info("Discovering URLs...")
        urls = self.url_discovery.discover_all_urls(self.base_url, self.max_depth)
        logger.info(f"Discovered {len(urls)} URLs to crawl")

        all_extracted_content = []

        # Crawl each URL
        for i, url in enumerate(urls):
            logger.info(f"Crawling ({i+1}/{len(urls)}): {url}")

            try:
                page_data = self.crawler.extract_page_content(url)

                if page_data.get('status') == 'success':
                    # Use the content extracted by the crawler directly
                    # The crawler already handles Docusaurus-specific extraction
                    extracted = {
                        'content': page_data.get('content', ''),
                        'title': page_data.get('title', ''),
                        'headings': page_data.get('headings', []),
                        'hierarchy': page_data.get('hierarchy', 'Unknown'),
                        'url': url
                    }

                    all_extracted_content.append(extracted)

                    # Add delay between requests to be respectful
                    time.sleep(self.delay)
                else:
                    logger.warning(f"Failed to crawl {url}: {page_data.get('error')}")

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                continue

        logger.info(f"Completed crawl and extraction. Processed {len(all_extracted_content)} pages.")
        return all_extracted_content

    def chunk_extracted_content(self, extracted_content: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Chunk the extracted content into DocumentChunk objects
        """
        all_chunks = []

        for content_item in extracted_content:
            content = content_item.get('content', '')
            url = content_item.get('url', '')
            title = content_item.get('title', '')
            hierarchy = content_item.get('hierarchy', 'Unknown')

            if not content or len(content.strip()) == 0:
                continue

            # Clean the content
            cleaned_content = clean_text(content)

            # Create metadata for the chunk
            metadata = {
                'title': title,
                'url': url,
                'extracted_at': time.time(),
                'source_type': 'docusaurus_page'
            }

            # Chunk the content
            chunks = self.chunker.chunk_text(
                cleaned_content,
                self.chunk_size,
                self.chunk_overlap,
                url,
                hierarchy,
                metadata
            )

            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from extracted content.")
        return all_chunks

    def crawl_extract_and_chunk(self) -> List[DocumentChunk]:
        """
        Complete workflow: crawl -> extract -> chunk
        """
        logger.info("Starting complete crawl -> extract -> chunk workflow")

        # Step 1: Crawl and extract
        extracted_content = self.crawl_and_extract()

        if not extracted_content:
            logger.warning("No content extracted from crawling")
            return []

        # Step 2: Chunk the extracted content
        chunks = self.chunk_extracted_content(extracted_content)

        logger.info(f"Workflow completed. Created {len(chunks)} document chunks.")
        return chunks

    def get_crawl_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the crawl
        """
        return self.crawler.get_crawl_stats()

    def validate_crawl_results(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate the results of the crawl
        """
        stats = {
            'total_chunks': len(chunks),
            'total_content_chars': sum(len(chunk.content) for chunk in chunks),
            'unique_urls': len(set(chunk.source_url for chunk in chunks)),
            'avg_chunk_size': sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0,
            'valid_chunks': 0,
            'invalid_chunks': 0
        }

        for chunk in chunks:
            try:
                chunk.validate()
                stats['valid_chunks'] += 1
            except ValueError:
                stats['invalid_chunks'] += 1

        return stats

    def crawl_single_page(self, url: str) -> List[DocumentChunk]:
        """
        Crawl and process a single page
        """
        logger.info(f"Crawling single page: {url}")

        page_data = self.crawler.extract_page_content(url)

        if page_data.get('status') != 'success':
            logger.error(f"Failed to crawl page {url}: {page_data.get('error')}")
            return []

        # Extract content
        extracted = self.extractor.extract_content(page_data.get('content', ''), url)
        extracted.update({
            'title': page_data.get('title', ''),
            'headings': page_data.get('headings', []),
            'url': url
        })

        # Chunk the content
        chunks = self.chunk_extracted_content([extracted])

        logger.info(f"Created {len(chunks)} chunks from single page: {url}")
        return chunks

    def crawl_with_progress_callback(self, progress_callback=None) -> List[DocumentChunk]:
        """
        Crawl with progress callback for UI updates
        """
        logger.info(f"Starting crawl with progress tracking for: {self.base_url}")

        # Discover all URLs in the book
        logger.info("Discovering URLs...")
        urls = self.url_discovery.discover_all_urls(self.base_url, self.max_depth)
        total_urls = len(urls)
        logger.info(f"Discovered {total_urls} URLs to crawl")

        all_chunks = []
        processed_count = 0

        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i, total_urls, f"Processing {url}")

            logger.info(f"Crawling ({i+1}/{total_urls}): {url}")

            try:
                page_data = self.crawler.extract_page_content(url)

                if page_data.get('status') == 'success':
                    # Use the content extracted by the crawler directly
                    # The crawler already handles Docusaurus-specific extraction
                    extracted = {
                        'content': page_data.get('content', ''),
                        'title': page_data.get('title', ''),
                        'headings': page_data.get('headings', []),
                        'hierarchy': page_data.get('hierarchy', 'Unknown'),
                        'url': url
                    }

                    # Chunk the content
                    page_chunks = self.chunk_extracted_content([extracted])
                    all_chunks.extend(page_chunks)

                    processed_count += 1

                    # Add delay between requests to be respectful
                    time.sleep(self.delay)
                else:
                    logger.warning(f"Failed to crawl {url}: {page_data.get('error')}")

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                continue

        if progress_callback:
            progress_callback(total_urls, total_urls, "Crawling completed")

        logger.info(f"Completed crawl. Processed {processed_count} out of {total_urls} URLs. Created {len(all_chunks)} chunks.")
        return all_chunks