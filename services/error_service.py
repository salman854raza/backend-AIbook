from typing import List, Dict, Any, Callable, Optional
import logging
import time
import random
from functools import wraps
from requests.exceptions import RequestException
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions that fail
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e

                    logger.warning(f"Function {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff  # Exponential backoff

            return None
        return wrapper
    return decorator

class ErrorService:
    """
    Service for handling errors and implementing retry logic
    """

    def __init__(self):
        """
        Initialize the error service
        """
        self.error_log: List[Dict[str, Any]] = []

    def log_error(self, error: Exception, context: str = "", url: str = None) -> Dict[str, Any]:
        """
        Log an error with context
        """
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'url': url,
            'traceback': str(error.__traceback__) if error.__traceback__ else None
        }

        self.error_log.append(error_info)
        logger.error(f"Error in {context}: {str(error)} - URL: {url}")

        return error_info

    def handle_crawl_error(self, error: Exception, url: str) -> Dict[str, Any]:
        """
        Handle errors during crawling
        """
        return self.log_error(error, "crawling", url)

    def handle_extraction_error(self, error: Exception, url: str) -> Dict[str, Any]:
        """
        Handle errors during content extraction
        """
        return self.log_error(error, "content_extraction", url)

    def handle_embedding_error(self, error: Exception, chunk_id: str = None) -> Dict[str, Any]:
        """
        Handle errors during embedding generation
        """
        return self.log_error(error, "embedding_generation", chunk_id)

    def handle_storage_error(self, error: Exception, chunk_id: str = None) -> Dict[str, Any]:
        """
        Handle errors during vector storage
        """
        return self.log_error(error, "vector_storage", chunk_id)

    def retry_with_backoff(self, func: Callable, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0,
                          exceptions: tuple = (Exception,), context: str = "") -> Any:
        """
        Execute a function with retry and exponential backoff
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    # Final attempt failed
                    logger.error(f"Function failed after {max_retries} retries in {context}: {str(e)}")
                    self.log_error(e, f"{context}_final_failure")
                    raise e

                logger.warning(f"Attempt {attempt + 1} failed in {context}: {str(e)}. Retrying in {delay} seconds...")
                self.log_error(e, f"{context}_attempt_{attempt + 1}")
                time.sleep(delay)
                delay *= backoff_factor
                # Add some jitter to prevent thundering herd
                delay += random.uniform(0, 0.1 * delay)

        return None

    def handle_rate_limit_error(self, error: Exception, delay: float = 60.0) -> bool:
        """
        Handle rate limit errors by waiting
        """
        logger.warning(f"Rate limit error encountered: {str(error)}. Waiting for {delay} seconds.")
        time.sleep(delay)
        return True

    def categorize_error(self, error: Exception) -> str:
        """
        Categorize an error into types for appropriate handling
        """
        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            return 'network'
        elif any(keyword in error_str for keyword in ['rate', 'limit', 'quota', 'exceeded']):
            return 'rate_limit'
        elif any(keyword in error_str for keyword in ['api', 'key', 'auth', 'unauthorized', 'forbidden']):
            return 'authentication'
        elif any(keyword in error_str for keyword in ['not found', '404', 'missing']):
            return 'not_found'
        elif any(keyword in error_str for keyword in ['server', '500', 'internal', 'error']):
            return 'server'
        elif any(keyword in error_str for keyword in ['invalid', 'malformed', 'bad request', '400']):
            return 'client'
        else:
            return 'unknown'

    def should_retry_error(self, error: Exception) -> bool:
        """
        Determine if an error should be retried
        """
        error_type = self.categorize_error(error)

        # Retry on network, server, and rate limit errors
        retryable_errors = ['network', 'server', 'rate_limit']
        return error_type in retryable_errors

    def get_retry_delay(self, error: Exception, attempt: int) -> float:
        """
        Get appropriate delay for retry based on error type and attempt number
        """
        error_type = self.categorize_error(error)

        base_delay = 1.0
        if error_type == 'rate_limit':
            # For rate limit errors, use a longer base delay
            base_delay = 60.0
        elif error_type == 'network':
            base_delay = 2.0
        elif error_type == 'server':
            base_delay = 1.5

        # Exponential backoff with jitter
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter

    def validate_chunks_after_error(self, chunks: List[DocumentChunk], error_context: str = "") -> List[DocumentChunk]:
        """
        Validate chunks after an error to ensure data integrity
        """
        valid_chunks = []
        invalid_count = 0

        for chunk in chunks:
            try:
                chunk.validate()
                valid_chunks.append(chunk)
            except ValueError as e:
                invalid_count += 1
                logger.error(f"Invalid chunk after {error_context}: {chunk.id} - {str(e)}")
                self.log_error(e, f"chunk_validation_after_{error_context}", chunk.source_url)

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid chunks out of {len(chunks)} total after {error_context}")

        return valid_chunks

    def handle_batch_operation_error(self,
                                   operation_func: Callable,
                                   items: List[Any],
                                   max_retries_per_item: int = 3,
                                   continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Handle errors in batch operations
        """
        successful_results = []
        failed_items = []
        errors = []

        for i, item in enumerate(items):
            success = False
            attempt = 0

            while not success and attempt < max_retries_per_item:
                try:
                    result = operation_func(item)
                    successful_results.append(result)
                    success = True
                except Exception as e:
                    attempt += 1
                    error_info = self.log_error(e, f"batch_operation_item_{i}_attempt_{attempt}", str(item))

                    if attempt >= max_retries_per_item:
                        logger.error(f"Failed to process item {i} after {max_retries_per_item} attempts: {str(e)}")
                        failed_items.append(item)
                        errors.append(error_info)

                        if not continue_on_error:
                            # Stop processing if we shouldn't continue on error
                            break
                    else:
                        # Wait before retry
                        delay = self.get_retry_delay(e, attempt)
                        time.sleep(delay)

        return {
            'successful_results': successful_results,
            'failed_items': failed_items,
            'errors': errors,
            'success_count': len(successful_results),
            'failure_count': len(failed_items),
            'total_processed': len(items)
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of logged errors
        """
        if not self.error_log:
            return {
                'total_errors': 0,
                'error_types': {},
                'recent_errors': []
            }

        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        recent_errors = self.error_log[-10:]  # Last 10 errors

        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': recent_errors,
            'error_rate': len(self.error_log)  # This would need context to be meaningful
        }

    def clear_error_log(self):
        """
        Clear the error log
        """
        self.error_log.clear()
        logger.info("Error log cleared")