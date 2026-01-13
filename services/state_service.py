import json
import os
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class PipelineState:
    """
    Data class to represent the state of the ingestion pipeline
    """
    session_id: str
    status: str  # 'running', 'completed', 'failed', 'paused'
    start_time: datetime
    end_time: Optional[datetime] = None
    processed_urls: List[str] = None
    failed_urls: List[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    checkpoint_data: Dict[str, Any] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the pipeline state to a dictionary"""
        state_dict = asdict(self)
        # Convert datetime objects to ISO format strings
        state_dict['start_time'] = self.start_time.isoformat()
        if self.end_time:
            state_dict['end_time'] = self.end_time.isoformat()
        return state_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create a PipelineState from a dictionary"""
        start_time = datetime.fromisoformat(data['start_time'])
        end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
        return cls(
            session_id=data['session_id'],
            status=data['status'],
            start_time=start_time,
            end_time=end_time,
            processed_urls=data.get('processed_urls', []),
            failed_urls=data.get('failed_urls', []),
            total_chunks=data.get('total_chunks', 0),
            processed_chunks=data.get('processed_chunks', 0),
            checkpoint_data=data.get('checkpoint_data', {}),
            error_message=data.get('error_message')
        )

class StateService:
    """
    Service for managing pipeline state and checkpoints
    """

    def __init__(self, state_file_path: str = "pipeline_state.json"):
        """
        Initialize the state service
        """
        self.state_file_path = state_file_path

    def save_state(self, state: PipelineState) -> bool:
        """
        Save the current pipeline state to a file
        """
        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.info(f"Pipeline state saved to {self.state_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving pipeline state: {str(e)}")
            return False

    def load_state(self) -> Optional[PipelineState]:
        """
        Load the pipeline state from a file
        """
        if not os.path.exists(self.state_file_path):
            logger.info(f"State file {self.state_file_path} does not exist")
            return None

        try:
            with open(self.state_file_path, 'r') as f:
                data = json.load(f)
            state = PipelineState.from_dict(data)
            logger.info(f"Pipeline state loaded from {self.state_file_path}")
            return state
        except Exception as e:
            logger.error(f"Error loading pipeline state: {str(e)}")
            return None

    def create_initial_state(self, session_id: str, initial_urls: List[str]) -> PipelineState:
        """
        Create an initial pipeline state
        """
        state = PipelineState(
            session_id=session_id,
            status='running',
            start_time=datetime.now(),
            processed_urls=[],
            failed_urls=[],
            total_chunks=0,
            processed_chunks=0,
            checkpoint_data={
                'initial_urls': initial_urls,
                'current_url_index': 0,
                'last_processed_url': None
            }
        )
        return state

    def update_state(self, state: PipelineState, **kwargs) -> PipelineState:
        """
        Update the pipeline state with new values
        """
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning(f"Unknown state attribute: {key}")

        # Update the end time if status is final
        if state.status in ['completed', 'failed']:
            if state.end_time is None:
                state.end_time = datetime.now()

        return state

    def add_processed_url(self, state: PipelineState, url: str) -> PipelineState:
        """
        Add a URL to the list of processed URLs
        """
        if url not in state.processed_urls:
            state.processed_urls.append(url)
        return state

    def add_failed_url(self, state: PipelineState, url: str, error: str = None) -> PipelineState:
        """
        Add a URL to the list of failed URLs
        """
        if url not in state.failed_urls:
            state.failed_urls.append(url)
        if error:
            logger.error(f"Failed to process URL {url}: {error}")
        return state

    def increment_chunk_count(self, state: PipelineState, count: int = 1) -> PipelineState:
        """
        Increment the chunk count
        """
        state.total_chunks += count
        return state

    def increment_processed_chunk_count(self, state: PipelineState, count: int = 1) -> PipelineState:
        """
        Increment the processed chunk count
        """
        state.processed_chunks += count
        return state

    def get_checkpoint_data(self, state: PipelineState) -> Dict[str, Any]:
        """
        Get checkpoint data from the state
        """
        return state.checkpoint_data or {}

    def update_checkpoint_data(self, state: PipelineState, data: Dict[str, Any]) -> PipelineState:
        """
        Update checkpoint data in the state
        """
        if state.checkpoint_data is None:
            state.checkpoint_data = {}
        state.checkpoint_data.update(data)
        return state

    def clear_state_file(self) -> bool:
        """
        Clear the state file
        """
        try:
            if os.path.exists(self.state_file_path):
                os.remove(self.state_file_path)
                logger.info(f"State file {self.state_file_path} cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing state file: {str(e)}")
            return False

    def is_pipeline_running(self, state: PipelineState) -> bool:
        """
        Check if the pipeline is currently running
        """
        return state.status == 'running'

    def is_pipeline_completed(self, state: PipelineState) -> bool:
        """
        Check if the pipeline has completed
        """
        return state.status == 'completed'

    def is_pipeline_failed(self, state: PipelineState) -> bool:
        """
        Check if the pipeline has failed
        """
        return state.status == 'failed'

    def calculate_progress(self, state: PipelineState, total_expected_urls: int = None) -> float:
        """
        Calculate the progress percentage of the pipeline
        """
        if total_expected_urls is None:
            total_expected_urls = len(state.checkpoint_data.get('initial_urls', [])) if state.checkpoint_data else 0

        if total_expected_urls == 0:
            return 0.0

        processed = len(state.processed_urls)
        progress = (processed / total_expected_urls) * 100
        return min(progress, 100.0)  # Cap at 100%