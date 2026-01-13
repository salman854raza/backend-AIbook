from typing import List, Dict, Any, Optional
import logging
import time
from models.document_chunk import DocumentChunk
from services.state_service import StateService, PipelineState
from services.crawl_service import CrawlService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)

class ResumeService:
    """
    Service for handling pipeline failure recovery and resume functionality
    """

    def __init__(self, state_service: StateService, crawl_service: CrawlService, vector_service: VectorService):
        """
        Initialize the resume service
        """
        self.state_service = state_service
        self.crawl_service = crawl_service
        self.vector_service = vector_service

    def resume_pipeline(self, max_retries: int = 3) -> Optional[PipelineState]:
        """
        Resume the pipeline from where it left off after a failure
        """
        # Load the current state
        current_state = self.state_service.load_state()

        if not current_state:
            logger.warning("No saved state found. Cannot resume pipeline.")
            return None

        if current_state.status == 'completed':
            logger.info("Pipeline already completed. Nothing to resume.")
            return current_state

        logger.info(f"Resuming pipeline from state: {current_state.status}")

        # Update state to running
        current_state = self.state_service.update_state(current_state, status='running')

        # Determine what needs to be resumed based on the current state
        if current_state.checkpoint_data:
            initial_urls = current_state.checkpoint_data.get('initial_urls', [])
            current_url_index = current_state.checkpoint_data.get('current_url_index', 0)
            last_processed_url = current_state.checkpoint_data.get('last_processed_url')

            # Get the URLs that still need to be processed
            remaining_urls = initial_urls[current_url_index:]

            if remaining_urls:
                logger.info(f"Resuming with {len(remaining_urls)} remaining URLs to process")
                return self._process_remaining_urls(current_state, remaining_urls, max_retries)
            else:
                logger.info("No remaining URLs to process. Pipeline should be completed.")
                current_state = self.state_service.update_state(current_state, status='completed')
                self.state_service.save_state(current_state)
                return current_state
        else:
            logger.warning("No checkpoint data found in state. Cannot resume effectively.")
            return current_state

    def _process_remaining_urls(self, state: PipelineState, remaining_urls: List[str], max_retries: int = 3) -> PipelineState:
        """
        Process the remaining URLs in the pipeline
        """
        for i, url in enumerate(remaining_urls):
            logger.info(f"Processing URL {i+1}/{len(remaining_urls)}: {url}")

            # Update checkpoint data
            state = self.state_service.update_checkpoint_data(state, {
                'current_url_index': len(state.processed_urls),
                'last_processed_url': url
            })

            # Attempt to process the URL with retries
            success = False
            retry_count = 0

            while not success and retry_count < max_retries:
                try:
                    # Process the URL (crawling and chunking)
                    chunks = self.crawl_service.crawl_single_page(url)

                    if chunks:
                        # Process chunks through embedding and storage
                        result = self.vector_service.process_and_store_chunks(chunks)

                        if result['success']:
                            state = self.state_service.add_processed_url(state, url)
                            state = self.state_service.increment_chunk_count(state, len(chunks))
                            success = True
                        else:
                            logger.error(f"Failed to process chunks for URL: {url}")
                    else:
                        logger.warning(f"No chunks generated for URL: {url}")
                        state = self.state_service.add_processed_url(state, url)
                        success = True

                except Exception as e:
                    retry_count += 1
                    logger.error(f"Attempt {retry_count} failed for URL {url}: {str(e)}")

                    if retry_count < max_retries:
                        logger.info(f"Retrying in {2**retry_count} seconds...")
                        time.sleep(2**retry_count)  # Exponential backoff
                    else:
                        logger.error(f"Failed to process URL after {max_retries} attempts: {url}")
                        state = self.state_service.add_failed_url(state, url, str(e))

            # Save state after each URL
            self.state_service.save_state(state)

        # Update final state
        if state.status == 'running':
            state = self.state_service.update_state(state, status='completed')

        # Save final state
        self.state_service.save_state(state)
        return state

    def restart_pipeline_from_scratch(self, urls: List[str]) -> PipelineState:
        """
        Restart the pipeline from scratch with new URLs
        """
        logger.info(f"Restarting pipeline from scratch with {len(urls)} URLs")

        # Create new state
        state = self.state_service.create_initial_state(f"restart_{int(time.time())}", urls)
        self.state_service.save_state(state)

        # Process all URLs
        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            # Update checkpoint data
            state = self.state_service.update_checkpoint_data(state, {
                'current_url_index': i,
                'last_processed_url': url
            })

            try:
                # Process the URL (crawling and chunking)
                chunks = self.crawl_service.crawl_single_page(url)

                if chunks:
                    # Process chunks through embedding and storage
                    result = self.vector_service.process_and_store_chunks(chunks)

                    if result['success']:
                        state = self.state_service.add_processed_url(state, url)
                        state = self.state_service.increment_chunk_count(state, len(chunks))
                    else:
                        logger.error(f"Failed to process chunks for URL: {url}")
                        state = self.state_service.add_failed_url(state, url, "Chunk processing failed")
                else:
                    logger.warning(f"No chunks generated for URL: {url}")
                    state = self.state_service.add_processed_url(state, url)

            except Exception as e:
                logger.error(f"Failed to process URL {url}: {str(e)}")
                state = self.state_service.add_failed_url(state, url, str(e))

            # Save state after each URL
            self.state_service.save_state(state)

        # Update final state
        state = self.state_service.update_state(state, status='completed')
        self.state_service.save_state(state)

        return state

    def get_resume_recommendation(self) -> Dict[str, Any]:
        """
        Get a recommendation on what to do based on the current state
        """
        current_state = self.state_service.load_state()

        if not current_state:
            return {
                'action': 'start_new',
                'message': 'No existing state found. Start a new pipeline.',
                'state': None
            }

        if current_state.status == 'completed':
            return {
                'action': 'completed',
                'message': 'Pipeline already completed successfully.',
                'state': current_state
            }

        if current_state.status == 'failed':
            return {
                'action': 'resume',
                'message': 'Pipeline failed. Recommend resuming from the point of failure.',
                'state': current_state
            }

        if current_state.status == 'running':
            return {
                'action': 'check_progress',
                'message': 'Pipeline is currently running.',
                'state': current_state
            }

        return {
            'action': 'resume',
            'message': 'Pipeline is in progress. Recommend resuming.',
            'state': current_state
        }

    def reset_pipeline_state(self) -> bool:
        """
        Reset the pipeline state file
        """
        return self.state_service.clear_state_file()

    def validate_resume_state(self, state: PipelineState) -> bool:
        """
        Validate if the current state is valid for resuming
        """
        if not state:
            return False

        # Check if required checkpoint data exists
        if not state.checkpoint_data:
            logger.error("No checkpoint data available for resume")
            return False

        # Check if initial URLs are available
        initial_urls = state.checkpoint_data.get('initial_urls', [])
        if not initial_urls:
            logger.error("No initial URLs available in checkpoint data")
            return False

        # Check if we haven't already processed all URLs
        processed_count = len(state.processed_urls)
        total_count = len(initial_urls)

        if processed_count >= total_count:
            logger.info("All URLs have been processed")
            return False

        return True

    def recover_from_failure(self, error_context: Dict[str, Any] = None) -> bool:
        """
        Attempt to recover from a specific failure
        """
        logger.info("Attempting to recover from failure")

        current_state = self.state_service.load_state()
        if not current_state:
            logger.error("No state to recover from")
            return False

        # Log the error context
        if error_context:
            logger.info(f"Error context: {error_context}")
            current_state.error_message = str(error_context.get('error', 'Unknown error'))

        # Update state to reflect the failure
        current_state = self.state_service.update_state(current_state, status='failed')
        self.state_service.save_state(current_state)

        logger.info("State updated to reflect failure. Ready for resume.")
        return True

    def get_remaining_work(self) -> Dict[str, Any]:
        """
        Get information about the remaining work in the pipeline
        """
        current_state = self.state_service.load_state()

        if not current_state or not current_state.checkpoint_data:
            return {
                'remaining_urls': 0,
                'processed_urls': 0,
                'total_urls': 0,
                'progress_percentage': 0.0,
                'estimated_time_remaining': None
            }

        initial_urls = current_state.checkpoint_data.get('initial_urls', [])
        processed_count = len(current_state.processed_urls)
        remaining_count = len(initial_urls) - processed_count

        progress_percentage = (processed_count / len(initial_urls)) * 100 if initial_urls else 0.0

        # Rough estimate of time remaining (assuming 10 seconds per URL)
        estimated_time_remaining = None
        if remaining_count > 0:
            avg_time_per_url = 10  # seconds
            estimated_time_remaining = remaining_count * avg_time_per_url

        return {
            'remaining_urls': remaining_count,
            'processed_urls': processed_count,
            'total_urls': len(initial_urls),
            'progress_percentage': progress_percentage,
            'estimated_time_remaining': estimated_time_remaining
        }