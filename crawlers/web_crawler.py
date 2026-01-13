import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Set, Dict, Any
import time
import logging
from utils import is_valid_url, normalize_url, is_same_domain

logger = logging.getLogger(__name__)

class WebCrawler:
    """
    Web crawler for Docusaurus book content
    """

    def __init__(self, base_url: str, delay: float = 1.0, max_depth: int = 5):
        """
        Initialize web crawler
        """
        self.base_url = base_url
        self.delay = delay
        self.max_depth = max_depth
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DocusaurusBot/1.0)'
        })

        # Track visited URLs to avoid duplicates
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()

    def is_valid_docusaurus_page(self, url: str, soup: BeautifulSoup) -> bool:
        """
        Check if page appears to be a valid Docusaurus page
        """
        # Check if page is from same domain as base URL
        if not is_same_domain(self.base_url, url):
            return False

        # Check for common Docusaurus elements
        # Look for Docusaurus-specific classes or elements
        docusaurus_indicators = [
            'navbar',
            'main-wrapper',
            'doc-page',
            'theme-doc-sidebar',
            'doc-content',
            'container',
            'row',
            'col'
        ]

        for indicator in docusaurus_indicators:
            if soup.find(class_=indicator):
                return True

        # If no Docusaurus-specific elements found, check for general content
        # Check for main/content wrappers (used in module pages)
        main_container = soup.find('div', class_='docMainContainer')
        if main_container:
            return True  # This is a valid content page

        # Check for article (used in docs)
        article = soup.find('article')
        if article:
            return True  # This is a valid content page

        # Check for body with any content
        body = soup.find('body')
        if body and body.get_text(strip=True).strip():
            return True  # Has some content in body

        # No recognized content structure found
        return False

    def extract_page_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single page
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Check if this appears to be a valid Docusaurus page
            if not self.is_valid_docusaurus_page(url, soup):
                logger.warning(f"URL does not appear to be a valid Docusaurus page: {url}")
                return {}

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""

            # Extract main content - look for Docusaurus-specific content areas
            # Updated selectors based on actual site structure
            content_selectors = [
                'div[class*="docMainContainer"]',
                'div[class*="markdown"]',
                'div[class*="theme-layout-main"]',
                'div[class*="main-wrapper"]',
                'div[class*="theme-doc-markdown"]',
                'div[class*="docsWrapper"]',
                '[class*="main"]',
                'main',
                'div[class*="theme-layout-navbar"]',
                'div[class*="theme-layout-footer-column"]',
                'body'
            ]

            content_element = None
            for selector in content_selectors:
                try:
                    content_element = soup.select_one(selector)
                    if content_element:
                        break
                except:
                    continue

            if not content_element or content_element is None:
                # If no specific content area found, use body
                content_element = soup.find('body')

            # If still no element or body is None, just return empty
            if not content_element or content_element is None:
                content = ""
            else:
                # Remove navigation, headers, footers, and other non-content elements
                for element in content_element.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'menu', 'div[class*="theme-layout-navbar"]', 'div[class*="theme-layout-footer"]']):
                    element.decompose()

                # Get text content with proper spacing
                raw_content = content_element.get_text(separator=' ', strip=True) if content_element else ""

                # Only use content if it has meaningful text (> 50 characters to avoid placeholder divs)
                if len(raw_content) > 50:
                    content = raw_content
                else:
                    content = ""

                # Remove excessive whitespace
                content = ' '.join(content.split())

            # Extract headings for hierarchy
            headings = []
            for i in range(1, 7):  # h1 to h6
                for heading in soup.find_all(f'h{i}'):
                    headings.append({
                        'level': i,
                        'text': heading.get_text().strip()
                    })

            # Extract document hierarchy from URL
            hierarchy = self._extract_hierarchy_from_url(url)

            return {
                'url': url,
                'title': title,
                'content': content,
                'headings': headings,
                'hierarchy': hierarchy,
                'status': 'success'
            }

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            self.failed_urls.add(url)
            return {
                'url': url,
                'status': 'failed',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            self.failed_urls.add(url)
            return {
                'url': url,
                'status': 'failed',
                'error': str(e)
            }

    def _extract_hierarchy_from_url(self, url: str) -> str:
        """
        Extract document hierarchy from URL path
        """
        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split('/') if part and part != 'index.html']

        if not path_parts:
            return "Home"

        # Join path parts with ' > ' to create hierarchy
        return " > ".join(path_parts)

    def crawl_from_url(self, start_url: str, current_depth: int = 0) -> List[Dict[str, Any]]:
        """
        Crawl starting from a specific URL up to max depth
        """
        if current_depth > self.max_depth:
            logger.info(f"Max depth {self.max_depth} reached, stopping crawl from {start_url}")
            return []

        if normalize_url(start_url) in self.visited_urls:
            logger.debug(f"Already visited: {start_url}")
            return []

        # Validate URL
        if not is_valid_url(start_url):
            logger.warning(f"Invalid URL: {start_url}")
            return []

        # Add to visited URLs
        normalized_start_url = normalize_url(start_url)
        self.visited_urls.add(normalized_start_url)

        # Extract content from current page
        logger.info(f"Crawling: {start_url}")
        page_data = self.extract_page_content(start_url)

        if page_data.get('status') == 'failed':
            logger.error(f"Failed to crawl {start_url}: {page_data.get('error')}")
            return [page_data]

        results = [page_data]

        # Add delay to be respectful to server
        time.sleep(self.delay)

        # Find links on page and crawl them recursively
        if current_depth < self.max_depth:
            try:
                response = self.session.get(start_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find all links on page
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']

                    # Convert relative URLs to absolute URLs
                    absolute_url = urljoin(start_url, href)

                    # Normalize URL
                    normalized_url = normalize_url(absolute_url)

                    # Only follow links from same domain and with proper extensions
                    if (is_same_domain(self.base_url, absolute_url) and
                        normalized_url not in self.visited_urls and
                        not normalized_url.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe'))):

                        # Recursively crawl link
                        child_results = self.crawl_from_url(absolute_url, current_depth + 1)
                        results.extend(child_results)

                        # Add delay between requests to be respectful
                        time.sleep(self.delay)

            except Exception as e:
                logger.error(f"Error finding links on {start_url}: {str(e)}")

        return results

    def crawl_book(self) -> List[Dict[str, Any]]:
        """
        Crawl entire Docusaurus book starting from base URL
        """
        logger.info(f"Starting crawl of Docusaurus book: {self.base_url}")
        self.visited_urls.clear()
        self.failed_urls.clear()
        results = self.crawl_from_url(self.base_url)
        logger.info(f"Crawl completed. Processed {len(results)} pages, {len(self.failed_urls)} failed.")
        return results

    def get_crawl_stats(self) -> Dict[str, Any]:
        """
        Get statistics about crawl
        """
        return {
            'total_visited': len(self.visited_urls),
            'total_failed': len(self.failed_urls),
            'success_rate': len([r for r in self.visited_urls if r not in self.failed_urls]) / len(self.visited_urls) if self.visited_urls else 0,
            'failed_urls': list(self.failed_urls)
        }
