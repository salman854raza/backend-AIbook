import time
import threading
from typing import Dict, Optional
from urllib.parse import urlparse
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter for web crawling to respect server limits
    """

    def __init__(self, default_rate: float = 1.0, default_burst: int = 1):
        """
        Initialize the rate limiter
        :param default_rate: Requests per second (default 1 req/sec)
        :param default_burst: Burst capacity (default 1 request)
        """
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.domains: Dict[str, Dict] = defaultdict(lambda: {
            'rate': self.default_rate,
            'burst': self.default_burst,
            'timestamps': deque(),
            'lock': threading.Lock()
        })
        self.global_rate = default_rate
        self.global_burst = default_burst
        self.global_timestamps = deque()
        self.global_lock = threading.Lock()

    def set_domain_limits(self, domain: str, requests_per_second: float, burst: int = 1):
        """
        Set rate limits for a specific domain
        """
        with self.domains[domain]['lock']:
            self.domains[domain]['rate'] = requests_per_second
            self.domains[domain]['burst'] = burst

    def set_global_limits(self, requests_per_second: float, burst: int = 1):
        """
        Set global rate limits
        """
        self.global_rate = requests_per_second
        self.global_burst = burst

    def _wait_for_capacity(self, timestamps: deque, rate: float, burst: int):
        """
        Wait until there's capacity for a new request
        """
        now = time.time()

        # Remove timestamps older than the rate window
        while timestamps and timestamps[0] <= now - (1.0 / rate):
            timestamps.popleft()

        # If we've reached the burst limit, wait
        if len(timestamps) >= burst:
            sleep_time = (1.0 / rate) - (now - timestamps[0]) + 0.01  # Small buffer
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()

        # Add current timestamp
        timestamps.append(now)

    def wait(self, url: str):
        """
        Wait for permission to make a request to the given URL
        """
        domain = urlparse(url).netloc

        # Apply domain-specific rate limiting
        domain_config = self.domains[domain]
        with domain_config['lock']:
            self._wait_for_capacity(
                domain_config['timestamps'],
                domain_config['rate'],
                domain_config['burst']
            )

        # Apply global rate limiting
        with self.global_lock:
            self._wait_for_capacity(
                self.global_timestamps,
                self.global_rate,
                self.global_burst
            )

    def acquire(self, url: str, blocking: bool = True) -> bool:
        """
        Acquire permission to make a request (non-blocking option available)
        """
        domain = urlparse(url).netloc
        now = time.time()

        # Check domain-specific limits
        domain_config = self.domains[domain]
        with domain_config['lock']:
            # Remove old timestamps
            while domain_config['timestamps'] and domain_config['timestamps'][0] <= now - (1.0 / domain_config['rate']):
                domain_config['timestamps'].popleft()

            if len(domain_config['timestamps']) >= domain_config['burst']:
                if not blocking:
                    return False
                # Wait for capacity
                sleep_time = (1.0 / domain_config['rate']) - (now - domain_config['timestamps'][0]) + 0.01
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

        # Check global limits
        with self.global_lock:
            while self.global_timestamps and self.global_timestamps[0] <= now - (1.0 / self.global_rate):
                self.global_timestamps.popleft()

            if len(self.global_timestamps) >= self.global_burst:
                if not blocking:
                    return False
                # Wait for capacity
                sleep_time = (1.0 / self.global_rate) - (now - self.global_timestamps[0]) + 0.01
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()

        # Add timestamps for the request
        domain_config['timestamps'].append(now)
        self.global_timestamps.append(now)

        return True

class CrawlRateLimiter:
    """
    Specialized rate limiter for crawling with additional features
    """

    def __init__(self, default_delay: float = 1.0):
        """
        Initialize the crawl rate limiter
        """
        self.default_delay = default_delay
        self.rate_limiter = RateLimiter(default_rate=1.0/default_delay if default_delay > 0 else 1.0)
        self.domain_delays: Dict[str, float] = {}
        self.failed_requests: Dict[str, int] = defaultdict(int)
        self.last_request_time: Dict[str, float] = {}

    def set_domain_delay(self, domain: str, delay: float):
        """
        Set a specific delay for requests to a domain
        """
        self.domain_delays[domain] = delay
        # Update rate limiter as well (convert delay to rate)
        if delay > 0:
            self.rate_limiter.set_domain_limits(domain, 1.0/delay, burst=1)

    def should_delay_request(self, url: str) -> float:
        """
        Determine if a request should be delayed and return the delay amount
        """
        domain = urlparse(url).netloc

        # Use domain-specific delay if set, otherwise use default
        delay = self.domain_delays.get(domain, self.default_delay)

        # Increase delay if there have been recent failures for this domain
        failure_count = self.failed_requests.get(domain, 0)
        if failure_count > 0:
            # Exponential backoff: double the delay for each failure, max 60 seconds
            adjusted_delay = min(delay * (2 ** min(failure_count, 5)), 60.0)
            return adjusted_delay

        return delay

    def record_request(self, url: str, success: bool = True):
        """
        Record a request attempt
        """
        domain = urlparse(url).netloc
        self.last_request_time[domain] = time.time()

        if not success:
            self.failed_requests[domain] += 1
        else:
            # Reset failure count on success
            self.failed_requests[domain] = 0

    def wait_before_request(self, url: str):
        """
        Wait before making a request to respect rate limits
        """
        domain = urlparse(url).netloc

        # Check the time since the last request to this domain
        last_time = self.last_request_time.get(domain, 0)
        delay = self.should_delay_request(url)

        time_since_last = time.time() - last_time
        remaining_delay = max(0, delay - time_since_last)

        if remaining_delay > 0:
            time.sleep(remaining_delay)

        # Use the rate limiter as well for additional protection
        self.rate_limiter.wait(url)

    def can_make_request(self, url: str) -> bool:
        """
        Check if a request can be made without blocking
        """
        domain = urlparse(url).netloc

        # Check if enough time has passed since the last request
        last_time = self.last_request_time.get(domain, 0)
        delay = self.should_delay_request(url)

        time_since_last = time.time() - last_time
        return time_since_last >= delay

class AdaptiveRateLimiter:
    """
    Rate limiter that adapts based on server responses and errors
    """

    def __init__(self, initial_delay: float = 1.0, min_delay: float = 0.1, max_delay: float = 60.0):
        """
        Initialize the adaptive rate limiter
        """
        self.initial_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delays: Dict[str, float] = {}
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)

    def record_response_time(self, url: str, response_time: float):
        """
        Record the response time for a request
        """
        domain = urlparse(url).netloc
        self.response_times[domain].append(response_time)

    def record_result(self, url: str, success: bool):
        """
        Record the result of a request
        """
        domain = urlparse(url).netloc
        if success:
            self.success_counts[domain] += 1
            # Reduce delay on success (but not below minimum)
            current_delay = self.current_delays.get(domain, self.initial_delay)
            new_delay = max(self.min_delay, current_delay * 0.9)
            self.current_delays[domain] = new_delay
        else:
            self.error_counts[domain] += 1
            # Increase delay on error (but not above maximum)
            current_delay = self.current_delays.get(domain, self.initial_delay)
            new_delay = min(self.max_delay, current_delay * 1.5)
            self.current_delays[domain] = new_delay

    def get_delay(self, url: str) -> float:
        """
        Get the appropriate delay for a URL
        """
        domain = urlparse(url).netloc
        return self.current_delays.get(domain, self.initial_delay)

    def wait_before_request(self, url: str):
        """
        Wait before making a request based on adaptive delay
        """
        delay = self.get_delay(url)
        time.sleep(delay)