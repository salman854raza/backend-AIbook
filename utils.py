from urllib.parse import urljoin, urlparse
from typing import List, Set
import re

def is_valid_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing fragments and ensuring proper formatting
    """
    parsed = urlparse(url)
    # Remove fragment and query parameters for consistency
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # Ensure trailing slash for root paths
    if not parsed.path or parsed.path == "/":
        normalized = f"{normalized.rstrip('/')}/"
    return normalized

def is_same_domain(base_url: str, test_url: str) -> bool:
    """
    Check if two URLs belong to the same domain
    """
    base_domain = urlparse(base_url).netloc
    test_domain = urlparse(test_url).netloc
    return base_domain == test_domain

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and normalizing
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def get_unique_urls(urls: List[str]) -> List[str]:
    """
    Get unique URLs while preserving order
    """
    seen: Set[str] = set()
    unique_urls = []
    for url in urls:
        normalized = normalize_url(url)
        if normalized not in seen:
            seen.add(normalized)
            unique_urls.append(normalized)
    return unique_urls

def url_to_filename(url: str) -> str:
    """
    Convert a URL to a safe filename
    """
    # Remove protocol and replace special characters
    parsed = urlparse(url)
    path = parsed.path.replace('/', '_').replace('.', '_')
    filename = f"{parsed.netloc}{path}"
    # Remove any characters that might be problematic for filenames
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return filename