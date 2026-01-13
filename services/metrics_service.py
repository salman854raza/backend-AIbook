from typing import Dict, Any, List
import logging
import time
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class MetricsService:
    """
    Service for collecting and monitoring pipeline metrics
    """

    def __init__(self, metrics_file: str = "pipeline_metrics.json"):
        """
        Initialize the metrics service
        """
        self.metrics_file = metrics_file
        self.session_start_time = time.time()
        self.metrics: Dict[str, Any] = {
            'session_id': f"session_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_pages_crawled': 0,
            'total_chunks_processed': 0,
            'total_embeddings_generated': 0,
            'total_vectors_stored': 0,
            'failed_operations': 0,
            'successful_operations': 0,
            'processing_times': [],
            'url_stats': {
                'total_urls': 0,
                'processed_urls': 0,
                'failed_urls': 0,
                'skipped_urls': 0
            },
            'chunk_stats': {
                'total_chunks': 0,
                'valid_chunks': 0,
                'invalid_chunks': 0,
                'avg_chunk_size': 0,
                'total_content_chars': 0
            },
            'embedding_stats': {
                'total_embeddings': 0,
                'failed_embeddings': 0,
                'embedding_generation_time': 0,
                'avg_embedding_size': 0
            },
            'storage_stats': {
                'total_store_operations': 0,
                'successful_stores': 0,
                'failed_stores': 0,
                'storage_time': 0
            }
        }

    def start_crawl_operation(self) -> float:
        """
        Record start time for a crawl operation
        """
        return time.time()

    def end_crawl_operation(self, start_time: float, success: bool = True) -> float:
        """
        Record end time for a crawl operation and calculate duration
        """
        duration = time.time() - start_time
        self.metrics['processing_times'].append({
            'operation': 'crawl',
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })

        if success:
            self.metrics['successful_operations'] += 1
            self.metrics['total_pages_crawled'] += 1
        else:
            self.metrics['failed_operations'] += 1

        return duration

    def record_chunk_processing(self, num_chunks: int, avg_size: float = 0, success: bool = True):
        """
        Record metrics for chunk processing
        """
        self.metrics['total_chunks_processed'] += num_chunks
        self.metrics['chunk_stats']['total_chunks'] += num_chunks

        if success:
            self.metrics['chunk_stats']['valid_chunks'] += num_chunks
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['chunk_stats']['invalid_chunks'] += num_chunks
            self.metrics['failed_operations'] += 1

        # Update average chunk size
        if avg_size > 0:
            total_chars = self.metrics['chunk_stats']['total_content_chars']
            total_chars += avg_size * num_chunks
            self.metrics['chunk_stats']['total_content_chars'] = total_chars
            self.metrics['chunk_stats']['avg_chunk_size'] = total_chars / self.metrics['chunk_stats']['total_chunks']

    def record_embedding_generation(self, num_embeddings: int, generation_time: float = 0, success: bool = True):
        """
        Record metrics for embedding generation
        """
        self.metrics['total_embeddings_generated'] += num_embeddings
        self.metrics['embedding_stats']['total_embeddings'] += num_embeddings

        if success:
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['failed_operations'] += 1
            self.metrics['embedding_stats']['failed_embeddings'] += num_embeddings

        # Update embedding generation time
        self.metrics['embedding_stats']['embedding_generation_time'] += generation_time

    def record_storage_operation(self, num_vectors: int, storage_time: float = 0, success: bool = True):
        """
        Record metrics for storage operations
        """
        self.metrics['total_vectors_stored'] += num_vectors
        self.metrics['storage_stats']['total_store_operations'] += 1

        if success:
            self.metrics['storage_stats']['successful_stores'] += 1
            self.metrics['successful_operations'] += 1
        else:
            self.metrics['storage_stats']['failed_stores'] += 1
            self.metrics['failed_operations'] += 1

        # Update storage time
        self.metrics['storage_stats']['storage_time'] += storage_time

    def update_url_stats(self, total: int = None, processed: int = None, failed: int = None, skipped: int = None):
        """
        Update URL statistics
        """
        if total is not None:
            self.metrics['url_stats']['total_urls'] = total
        if processed is not None:
            self.metrics['url_stats']['processed_urls'] = processed
        if failed is not None:
            self.metrics['url_stats']['failed_urls'] = failed
        if skipped is not None:
            self.metrics['url_stats']['skipped_urls'] = skipped

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics
        """
        # Calculate derived metrics
        total_operations = self.metrics['successful_operations'] + self.metrics['failed_operations']
        success_rate = (
            (self.metrics['successful_operations'] / total_operations * 100)
            if total_operations > 0 else 0
        )

        current_metrics = self.metrics.copy()
        current_metrics['success_rate'] = success_rate
        current_metrics['current_duration'] = time.time() - self.session_start_time

        # Calculate averages
        if self.metrics['processing_times']:
            crawl_times = [pt['duration'] for pt in self.metrics['processing_times'] if pt['operation'] == 'crawl']
            if crawl_times:
                current_metrics['avg_crawl_time'] = sum(crawl_times) / len(crawl_times)

        return current_metrics

    def save_metrics(self) -> bool:
        """
        Save metrics to file
        """
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.get_current_metrics(), f, indent=2)
            logger.info(f"Metrics saved to {self.metrics_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False

    def load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from file
        """
        if not os.path.exists(self.metrics_file):
            logger.info(f"Metrics file {self.metrics_file} does not exist")
            return {}

        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Metrics loaded from {self.metrics_file}")
            return metrics
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            return {}

    def reset_session(self):
        """
        Reset metrics for a new session
        """
        self.session_start_time = time.time()
        self.metrics = {
            'session_id': f"session_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_pages_crawled': 0,
            'total_chunks_processed': 0,
            'total_embeddings_generated': 0,
            'total_vectors_stored': 0,
            'failed_operations': 0,
            'successful_operations': 0,
            'processing_times': [],
            'url_stats': {
                'total_urls': 0,
                'processed_urls': 0,
                'failed_urls': 0,
                'skipped_urls': 0
            },
            'chunk_stats': {
                'total_chunks': 0,
                'valid_chunks': 0,
                'invalid_chunks': 0,
                'avg_chunk_size': 0,
                'total_content_chars': 0
            },
            'embedding_stats': {
                'total_embeddings': 0,
                'failed_embeddings': 0,
                'embedding_generation_time': 0,
                'avg_embedding_size': 0
            },
            'storage_stats': {
                'total_store_operations': 0,
                'successful_stores': 0,
                'failed_stores': 0,
                'storage_time': 0
            }
        }

    def finalize_session(self):
        """
        Finalize the current session and record end time
        """
        self.metrics['end_time'] = datetime.now().isoformat()
        self.metrics['total_duration'] = time.time() - self.session_start_time
        self.save_metrics()

    def get_summary_report(self) -> str:
        """
        Generate a summary report of the metrics
        """
        metrics = self.get_current_metrics()
        report_lines = [
            "Pipeline Metrics Summary",
            "=" * 30,
            f"Session ID: {metrics['session_id']}",
            f"Start Time: {metrics['start_time']}",
            f"End Time: {metrics.get('end_time', 'In Progress')}",
            f"Total Duration: {metrics.get('total_duration', time.time() - self.session_start_time):.2f} seconds",
            "",
            "Operation Counts:",
            f"  Total Pages Crawled: {metrics['total_pages_crawled']}",
            f"  Total Chunks Processed: {metrics['total_chunks_processed']}",
            f"  Total Embeddings Generated: {metrics['total_embeddings_generated']}",
            f"  Total Vectors Stored: {metrics['total_vectors_stored']}",
            f"  Successful Operations: {metrics['successful_operations']}",
            f"  Failed Operations: {metrics['failed_operations']}",
            f"  Success Rate: {metrics['success_rate']:.2f}%",
            "",
            "URL Statistics:",
            f"  Total URLs: {metrics['url_stats']['total_urls']}",
            f"  Processed URLs: {metrics['url_stats']['processed_urls']}",
            f"  Failed URLs: {metrics['url_stats']['failed_urls']}",
            f"  Skipped URLs: {metrics['url_stats']['skipped_urls']}",
            "",
            "Chunk Statistics:",
            f"  Total Chunks: {metrics['chunk_stats']['total_chunks']}",
            f"  Valid Chunks: {metrics['chunk_stats']['valid_chunks']}",
            f"  Invalid Chunks: {metrics['chunk_stats']['invalid_chunks']}",
            f"  Average Chunk Size: {metrics['chunk_stats']['avg_chunk_size']:.2f} characters",
            "",
            "Embedding Statistics:",
            f"  Total Embeddings: {metrics['embedding_stats']['total_embeddings']}",
            f"  Failed Embeddings: {metrics['embedding_stats']['failed_embeddings']}",
            f"  Total Embedding Time: {metrics['embedding_stats']['embedding_generation_time']:.2f} seconds",
            "",
            "Storage Statistics:",
            f"  Total Store Operations: {metrics['storage_stats']['total_store_operations']}",
            f"  Successful Stores: {metrics['storage_stats']['successful_stores']}",
            f"  Failed Stores: {metrics['storage_stats']['failed_stores']}",
            f"  Total Storage Time: {metrics['storage_stats']['storage_time']:.2f} seconds"
        ]

        return "\n".join(report_lines)

    def log_progress(self, current: int, total: int, operation: str = "processing"):
        """
        Log progress for long-running operations
        """
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"{operation.title()} progress: {current}/{total} ({percentage:.1f}%)")

        # Update metrics if applicable
        if operation == "crawl":
            self.metrics['url_stats']['processed_urls'] = current
            self.metrics['url_stats']['total_urls'] = total
        elif operation == "chunks":
            self.metrics['chunk_stats']['valid_chunks'] = current
            self.metrics['chunk_stats']['total_chunks'] = total

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance-related metrics
        """
        total_duration = time.time() - self.session_start_time
        pages_per_second = self.metrics['total_pages_crawled'] / total_duration if total_duration > 0 else 0
        chunks_per_second = self.metrics['total_chunks_processed'] / total_duration if total_duration > 0 else 0
        embeddings_per_second = self.metrics['total_embeddings_generated'] / total_duration if total_duration > 0 else 0

        return {
            'pages_per_second': pages_per_second,
            'chunks_per_second': chunks_per_second,
            'embeddings_per_second': embeddings_per_second,
            'total_duration': total_duration,
            'average_crawl_time': (
                sum(pt['duration'] for pt in self.metrics['processing_times'] if pt['operation'] == 'crawl') /
                len([pt for pt in self.metrics['processing_times'] if pt['operation'] == 'crawl'])
                if self.metrics['processing_times'] else 0
            )
        }