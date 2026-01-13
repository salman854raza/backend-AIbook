from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import requests
from typing import List, Set, Dict, Any
import logging
try:
    from backend.utils import is_valid_url, normalize_url, is_same_domain
except ImportError:
    from utils import is_valid_url, normalize_url, is_same_domain

logger = logging.getLogger(__name__)

class URLDiscovery:
    """
    Module for discovering and navigating URLs within a Docusaurus book
    """

    def __init__(self, base_url: str, session: requests.Session = None):
        """
        Initialize the URL discovery module
        """
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.session = session or requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DocusaurusBot/1.0)'
        })

    def extract_links_from_page(self, url: str) -> List[str]:
        """
        Extract all valid links from a given page
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            links = []

            # Find all anchor tags with href attributes
            for link in soup.find_all('a', href=True):
                href = link['href']

                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(url, href)

                # Normalize the URL
                normalized_url = normalize_url(absolute_url)

                # Only include links from the same domain and with proper extensions
                if (is_same_domain(self.base_url, absolute_url) and
                    not normalized_url.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.doc', '.docx')) and
                    not any(skip in normalized_url.lower() for skip in ['mailto:', 'tel:', '#', 'javascript:', 'data:'])):

                    links.append(normalized_url)

            return list(set(links))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []

    def discover_urls_breadth_first(self, start_url: str, max_depth: int = 5) -> List[str]:
        """
        Discover URLs using breadth-first search up to max_depth
        """
        visited: Set[str] = set()
        to_visit: List[Dict[str, Any]] = [{'url': normalize_url(start_url), 'depth': 0}]
        all_urls: Set[str] = {normalize_url(start_url)}

        while to_visit:
            current_item = to_visit.pop(0)
            current_url = current_item['url']
            current_depth = current_item['depth']

            if current_url in visited:
                continue

            if current_depth > max_depth:
                continue

            visited.add(current_url)

            logger.info(f"Discovering links from: {current_url} (depth: {current_depth})")

            # Extract links from the current page
            links = self.extract_links_from_page(current_url)

            for link in links:
                if link not in visited and link not in all_urls:
                    all_urls.add(link)
                    to_visit.append({'url': link, 'depth': current_depth + 1})

        return list(all_urls)

    def discover_urls_docusaurus_specific(self, start_url: str, max_depth: int = 5) -> List[str]:
        """
        Discover URLs with Docusaurus-specific logic (navigation, sidebar, etc.)
        """
        visited: Set[str] = set()
        to_visit: List[Dict[str, Any]] = [{'url': normalize_url(start_url), 'depth': 0}]
        all_urls: Set[str] = {normalize_url(start_url)}

        while to_visit:
            current_item = to_visit.pop(0)
            current_url = current_item['url']
            current_depth = current_item['depth']

            if current_url in visited:
                continue

            if current_depth > max_depth:
                continue

            visited.add(current_url)

            logger.info(f"Discovering Docusaurus-specific links from: {current_url} (depth: {current_depth})")

            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for Docusaurus-specific navigation elements
                docusaurus_selectors = [
                    'nav a[href]',  # Navigation links
                    '.navbar a[href]',  # Navbar links
                    '.sidebar a[href]',  # Sidebar links
                    '.theme-doc-sidebar-menu a[href]',  # Docusaurus sidebar menu
                    '.menu a[href]',  # Menu links
                    'header a[href]',  # Header links
                    'footer a[href]',  # Footer links
                    '.pagination-nav a[href]',  # Pagination links
                    '.theme-edit-this-page a[href]',  # Edit links
                    '.table-of-contents a[href]',  # Table of contents
                    '.sidebar-container a[href]',  # Alternative sidebar
                    '.nav-links a[href]',  # Navigation links
                ]

                links = []
                for selector in docusaurus_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        href = element.get('href')
                        if href:
                            absolute_url = urljoin(current_url, href)
                            normalized_url = normalize_url(absolute_url)

                            # Only include valid, same-domain links
                            if (is_same_domain(self.base_url, absolute_url) and
                                not normalized_url.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.doc', '.docx')) and
                                not any(skip in normalized_url.lower() for skip in ['mailto:', 'tel:', '#', 'javascript:', 'data:'])):
                                links.append(normalized_url)

                # Remove duplicates
                links = list(set(links))

                for link in links:
                    if link not in visited and link not in all_urls:
                        all_urls.add(link)
                        to_visit.append({'url': link, 'depth': current_depth + 1})

            except Exception as e:
                logger.error(f"Error discovering Docusaurus-specific links from {current_url}: {str(e)}")

        return list(all_urls)

    def validate_urls(self, urls: List[str]) -> Dict[str, bool]:
        """
        Validate a list of URLs to check if they are accessible
        """
        results = {}
        for url in urls:
            try:
                response = self.session.head(url, timeout=5)
                results[url] = response.status_code < 400
            except Exception:
                results[url] = False

        return results

    def get_sitemap_urls(self) -> List[str]:
        """
        Try to get URLs from a sitemap.xml if available
        """
        sitemap_url = urljoin(self.base_url, 'sitemap.xml')
        urls = []

        try:
            response = self.session.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                for loc in soup.find_all('loc'):
                    url = loc.text.strip()
                    if is_same_domain(self.base_url, url):
                        urls.append(normalize_url(url))
        except Exception as e:
            logger.info(f"No sitemap found or error reading sitemap: {str(e)}")

        return urls

    def discover_all_urls(self, start_url: str, max_depth: int = 5) -> List[str]:
        """
        Discover all possible URLs using multiple strategies
        """
        logger.info(f"Starting comprehensive URL discovery from: {start_url}")

        # Get URLs from sitemap if available
        sitemap_urls = self.get_sitemap_urls()
        logger.info(f"Found {len(sitemap_urls)} URLs from sitemap")

        # Get URLs using Docusaurus-specific discovery
        docusaurus_urls = self.discover_urls_docusaurus_specific(start_url, max_depth)
        logger.info(f"Found {len(docusaurus_urls)} URLs using Docusaurus-specific discovery")

        # Combine all URLs
        all_urls = list(set(sitemap_urls + docusaurus_urls))

        # Validate the URLs
        valid_urls = []
        for url in all_urls:
            if is_valid_url(url) and is_same_domain(self.base_url, url):
                valid_urls.append(url)

        logger.info(f"Discovered {len(valid_urls)} valid URLs")

        return valid_urls