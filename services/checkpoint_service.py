import json
import os
import pickle
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class Checkpoint:
    """
    Data class to represent a pipeline checkpoint
    """
    checkpoint_id: str
    timestamp: datetime
    processed_urls: List[str]
    processed_chunks: List[str]  # List of chunk IDs
    current_position: Dict[str, Any]  # Current position in the pipeline
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the checkpoint to a dictionary"""
        checkpoint_dict = asdict(self)
        checkpoint_dict['timestamp'] = self.timestamp.isoformat()
        return checkpoint_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create a Checkpoint from a dictionary"""
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            checkpoint_id=data['checkpoint_id'],
            timestamp=timestamp,
            processed_urls=data['processed_urls'],
            processed_chunks=data['processed_chunks'],
            current_position=data['current_position'],
            metadata=data['metadata']
        )

class CheckpointService:
    """
    Service for managing pipeline checkpoints to enable resumption from failure points
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the checkpoint service
        """
        self.checkpoint_dir = checkpoint_dir
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        """Ensure the checkpoint directory exists"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def create_checkpoint(self,
                         checkpoint_id: str,
                         processed_urls: List[str],
                         processed_chunks: List[DocumentChunk],
                         current_position: Dict[str, Any],
                         metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        Create a new checkpoint
        """
        if metadata is None:
            metadata = {}

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            processed_urls=processed_urls,
            processed_chunks=[chunk.id for chunk in processed_chunks],
            current_position=current_position,
            metadata=metadata
        )

        # Save checkpoint to file
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"Created checkpoint: {checkpoint_id} with {len(processed_urls)} URLs and {len(processed_chunks)} chunks")
        return checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint by ID
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            logger.info(f"Checkpoint {checkpoint_id} does not exist")
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            checkpoint = Checkpoint.from_dict(data)
            logger.info(f"Loaded checkpoint: {checkpoint_id}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {str(e)}")
            return None

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints
        """
        try:
            files = os.listdir(self.checkpoint_dir)
            checkpoint_ids = [f.replace('.json', '') for f in files if f.endswith('.json')]
            logger.info(f"Found {len(checkpoint_ids)} checkpoints")
            return checkpoint_ids
        except Exception as e:
            logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint by ID
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info(f"Deleted checkpoint: {checkpoint_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting checkpoint {checkpoint_id}: {str(e)}")
                return False
        else:
            logger.warning(f"Checkpoint {checkpoint_id} does not exist")
            return False

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the most recent checkpoint
        """
        checkpoint_ids = self.list_checkpoints()
        if not checkpoint_ids:
            return None

        # Sort by timestamp (assuming checkpoint_id contains timestamp info)
        # If not, we'll load each and find the most recent
        latest_checkpoint = None
        latest_time = None

        for checkpoint_id in checkpoint_ids:
            checkpoint = self.load_checkpoint(checkpoint_id)
            if checkpoint and (latest_time is None or checkpoint.timestamp > latest_time):
                latest_checkpoint = checkpoint
                latest_time = checkpoint.timestamp

        return latest_checkpoint

    def create_url_checkpoint(self,
                            checkpoint_id: str,
                            processed_urls: List[str],
                            current_url_index: int,
                            metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        Create a checkpoint specifically for URL processing
        """
        current_position = {
            'type': 'url_processing',
            'current_url_index': current_url_index
        }

        return self.create_checkpoint(
            checkpoint_id=checkpoint_id,
            processed_urls=processed_urls,
            processed_chunks=[],
            current_position=current_position,
            metadata=metadata or {}
        )

    def create_chunk_checkpoint(self,
                              checkpoint_id: str,
                              processed_urls: List[str],
                              processed_chunk_ids: List[str],
                              current_chunk_index: int,
                              metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        Create a checkpoint specifically for chunk processing
        """
        current_position = {
            'type': 'chunk_processing',
            'current_chunk_index': current_chunk_index
        }

        return self.create_checkpoint(
            checkpoint_id=checkpoint_id,
            processed_urls=processed_urls,
            processed_chunks=processed_chunk_ids,
            current_position=current_position,
            metadata=metadata or {}
        )

    def create_embedding_checkpoint(self,
                                  checkpoint_id: str,
                                  processed_chunk_ids: List[str],
                                  current_embedding_index: int,
                                  metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        Create a checkpoint specifically for embedding processing
        """
        current_position = {
            'type': 'embedding_processing',
            'current_embedding_index': current_embedding_index
        }

        return self.create_checkpoint(
            checkpoint_id=checkpoint_id,
            processed_urls=[],  # At embedding stage, we're not tracking URLs anymore
            processed_chunks=processed_chunk_ids,
            current_position=current_position,
            metadata=metadata or {}
        )

    def create_storage_checkpoint(self,
                                checkpoint_id: str,
                                stored_chunk_ids: List[str],
                                current_storage_index: int,
                                metadata: Dict[str, Any] = None) -> Checkpoint:
        """
        Create a checkpoint specifically for storage processing
        """
        current_position = {
            'type': 'storage_processing',
            'current_storage_index': current_storage_index
        }

        return self.create_checkpoint(
            checkpoint_id=checkpoint_id,
            processed_urls=[],
            processed_chunks=stored_chunk_ids,
            current_position=current_position,
            metadata=metadata or {}
        )

    def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get information needed to resume from a checkpoint
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return {
                'success': False,
                'message': f'Checkpoint {checkpoint_id} not found',
                'resume_data': {}
            }

        resume_data = {
            'processed_urls': checkpoint.processed_urls,
            'processed_chunk_ids': checkpoint.processed_chunks,
            'current_position': checkpoint.current_position,
            'metadata': checkpoint.metadata
        }

        logger.info(f"Ready to resume from checkpoint {checkpoint_id}")
        return {
            'success': True,
            'message': f'Ready to resume from checkpoint {checkpoint_id}',
            'resume_data': resume_data
        }

    def get_checkpoint_summary(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a checkpoint
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None

        return {
            'checkpoint_id': checkpoint.checkpoint_id,
            'timestamp': checkpoint.timestamp.isoformat(),
            'processed_urls_count': len(checkpoint.processed_urls),
            'processed_chunks_count': len(checkpoint.processed_chunks),
            'current_position': checkpoint.current_position,
            'metadata': checkpoint.metadata
        }

    def validate_checkpoint_integrity(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of a checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            return {
                'valid': False,
                'error': f'Checkpoint file does not exist: {checkpoint_path}'
            }

        try:
            # Try to load and parse the checkpoint
            checkpoint = self.load_checkpoint(checkpoint_id)
            if not checkpoint:
                return {
                    'valid': False,
                    'error': 'Failed to load checkpoint data'
                }

            # Validate required fields
            issues = []
            if not checkpoint.checkpoint_id:
                issues.append('Missing checkpoint_id')
            if checkpoint.timestamp is None:
                issues.append('Missing timestamp')
            if checkpoint.current_position is None:
                issues.append('Missing current_position')

            # Validate data types
            if not isinstance(checkpoint.processed_urls, list):
                issues.append('processed_urls is not a list')
            if not isinstance(checkpoint.processed_chunks, list):
                issues.append('processed_chunks is not a list')
            if not isinstance(checkpoint.current_position, dict):
                issues.append('current_position is not a dict')

            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'checkpoint_id': checkpoint.checkpoint_id
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Exception during validation: {str(e)}'
            }

    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> bool:
        """
        Clean up old checkpoints, keeping only the last N
        """
        try:
            all_checkpoints = self.list_checkpoints()

            # Sort checkpoints by timestamp to identify oldest ones
            checkpoint_objects = []
            for cp_id in all_checkpoints:
                cp = self.load_checkpoint(cp_id)
                if cp:
                    checkpoint_objects.append(cp)

            # Sort by timestamp (oldest first)
            checkpoint_objects.sort(key=lambda x: x.timestamp)

            # Identify checkpoints to delete (keep only the last N)
            if len(checkpoint_objects) <= keep_last_n:
                logger.info(f"No checkpoints to clean up. Total: {len(checkpoint_objects)}, keeping: {keep_last_n}")
                return True

            checkpoints_to_delete = checkpoint_objects[:-keep_last_n]

            for cp in checkpoints_to_delete:
                self.delete_checkpoint(cp.checkpoint_id)

            logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints")
            return True

        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {str(e)}")
            return False