from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class CrawlSession:
    """
    An execution instance of the ingestion pipeline that processes a set of URLs and generates embeddings
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # 'running', 'completed', 'failed'
    processed_urls: List[str] = None
    failed_urls: List[str] = None
    total_chunks: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.processed_urls is None:
            self.processed_urls = []
        if self.failed_urls is None:
            self.failed_urls = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the crawl session to a dictionary representation
        """
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "processed_urls": self.processed_urls,
            "failed_urls": self.failed_urls,
            "total_chunks": self.total_chunks,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlSession':
        """
        Create a CrawlSession from a dictionary
        """
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = None
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"])

        return cls(
            session_id=data["session_id"],
            start_time=start_time,
            end_time=end_time,
            status=data["status"],
            processed_urls=data.get("processed_urls", []),
            failed_urls=data.get("failed_urls", []),
            total_chunks=data.get("total_chunks", 0),
            metadata=data.get("metadata", {})
        )

    def validate(self) -> bool:
        """
        Validate the crawl session
        """
        if not self.session_id:
            raise ValueError("Crawl session must have a session ID")

        if not self.start_time:
            raise ValueError("Crawl session must have a start time")

        if self.status not in ["running", "completed", "failed"]:
            raise ValueError(f"Invalid status: {self.status}")

        if self.total_chunks < 0:
            raise ValueError("Total chunks cannot be negative")

        return True

    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the session in seconds
        """
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            # Session is still running
            return (datetime.now() - self.start_time).total_seconds()
        else:
            return None

    def get_success_rate(self) -> float:
        """
        Get the success rate of URL processing
        """
        total_urls = len(self.processed_urls) + len(self.failed_urls)
        if total_urls == 0:
            return 0.0
        return len(self.processed_urls) / total_urls

    def add_processed_url(self, url: str):
        """
        Add a URL to the list of processed URLs
        """
        if url not in self.processed_urls:
            self.processed_urls.append(url)

    def add_failed_url(self, url: str, error: str = None):
        """
        Add a URL to the list of failed URLs
        """
        if url not in self.failed_urls:
            self.failed_urls.append(url)

    def mark_completed(self):
        """
        Mark the session as completed
        """
        self.status = "completed"
        self.end_time = datetime.now()

    def mark_failed(self, error_message: str = None):
        """
        Mark the session as failed
        """
        self.status = "failed"
        self.end_time = datetime.now()
        if error_message:
            self.metadata["error_message"] = error_message

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the crawl session
        """
        duration = self.get_duration()
        success_rate = self.get_success_rate()

        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "processed_urls_count": len(self.processed_urls),
            "failed_urls_count": len(self.failed_urls),
            "total_urls_count": len(self.processed_urls) + len(self.failed_urls),
            "success_rate": success_rate,
            "total_chunks": self.total_chunks,
            "chunks_per_second": self.total_chunks / duration if duration and duration > 0 else 0,
            "status": self.status
        }