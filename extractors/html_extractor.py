from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class HTMLExtractor:
    """
    Extract clean text content from HTML, preserving document structure and hierarchy
    """

    def __init__(self):
        """
        Initialize the HTML extractor
        """
        pass

    def extract_content(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """
        Extract clean text content from HTML with preserved structure
        """
        # Handle None or empty HTML content
        if not html_content or not html_content.strip():
            return {
                'title': '',
                'content': '',
                'headings': [],
                'hierarchy': '',
                'links': [],
                'url': url
            }

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract document title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()

        # Extract headings and content
        headings = self._extract_headings(soup)
        content = self._extract_main_content(soup)
        links = self._extract_links(soup)

        # Extract document hierarchy from headings
        hierarchy = self._build_hierarchy(headings)

        return {
            'title': title,
            'content': content,
            'headings': headings,
            'hierarchy': hierarchy,
            'links': links,
            'url': url
        }

    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract all headings from the HTML
        """
        headings = []
        for i in range(1, 7):  # h1 to h6
            for heading in soup.find_all(f'h{i}'):
                heading_text = heading.get_text().strip()
                if heading_text:
                    headings.append({
                        'level': i,
                        'text': heading_text,
                        'id': heading.get('id', ''),
                        'class': heading.get('class', [])
                    })

        return headings

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main content from HTML, focusing on article or main content areas
        """
        # Look for main content areas in order of preference
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.doc-content',
            '.container',
            'body'
        ]

        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        if not content_element or content_element is None:
            content = ""
        else:
            # Remove navigation, headers, footers, and other non-content elements
            for element in content_element.find_all(['nav', 'header', 'footer', 'aside', 'menu']):
                element.decompose()

            # Remove elements that are likely navigation or UI elements
            for element in content_element.find_all(class_=re.compile(r'nav|menu|sidebar|toc|footer|header')):
                element.decompose()

            # Get text content with proper spacing
            content = content_element.get_text(separator=' ', strip=True)

        # Clean up excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        return content

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract all links from the HTML
        """
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().strip()
            link_url = link['href']
            links.append({
                'text': link_text,
                'url': link_url
            })

        return links

    def _build_hierarchy(self, headings: List[Dict[str, Any]]) -> str:
        """
        Build a hierarchy string from headings
        """
        if not headings:
            return "Unknown"

        # Find the most prominent heading (lowest level number)
        top_level = min([h['level'] for h in headings]) if headings else 1

        # Get headings at the top level or one level below
        hierarchy_parts = []
        for heading in headings:
            if heading['level'] == top_level:
                hierarchy_parts.append(heading['text'])
            elif heading['level'] == top_level + 1:
                hierarchy_parts.append(heading['text'])
                break  # Only include the first subheading

        if not hierarchy_parts:
            return "Unknown"

        return " > ".join(hierarchy_parts)

    def extract_text_chunks(self, html_content: str, url: str = "", chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        Extract content and split into chunks of specified size
        """
        extracted = self.extract_content(html_content, url)

        content = extracted['content']
        if not content:
            return []

        # Split content into chunks
        chunks = []
        start = 0
        content_length = len(content)

        while start < content_length:
            end = start + chunk_size

            # If this is not the last chunk, try to break at a sentence or paragraph boundary
            if end < content_length:
                # Look for sentence boundary near the end
                search_start = end - 50  # Look back up to 50 chars
                if search_start < content_length:
                    for sep in ['.', '!', '?', '\n', ';', ',']:
                        pos = content.rfind(sep, search_start, end)
                        if pos != -1:
                            end = pos + 1
                            break

            chunk_text = content[start:end].strip()

            if chunk_text:  # Only add non-empty chunks
                chunk_data = {
                    'content': chunk_text,
                    'source_url': url,
                    'title': extracted['title'],
                    'hierarchy': extracted['hierarchy'],
                    'headings': extracted['headings'],
                    'start_pos': start,
                    'end_pos': end
                }
                chunks.append(chunk_data)

            # Move to next chunk position with overlap
            start = end - overlap if end < content_length else end

            # If the next chunk would be too small, just finish
            if start < content_length and content_length - start < overlap:
                break

        return chunks

    def clean_content(self, content: str) -> str:
        """
        Clean extracted content by removing extra whitespace and normalizing
        """
        if not content:
            return ""

        # Replace multiple whitespace with single space
        content = re.sub(r'\s+', ' ', content)
        # Remove leading/trailing whitespace
        content = content.strip()
        return content

    def extract_metadata(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """
        Extract metadata from HTML content
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'author': '',
            'language': '',
            'published_date': '',
            'tags': []
        }

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Extract meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '')

        # Extract other meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')

            if name == 'author':
                metadata['author'] = content
            elif name == 'language':
                metadata['language'] = content
            elif name in ['date', 'published', 'pubdate', 'dc.date', 'dcterms.date']:
                metadata['published_date'] = content

        # Extract tags/keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            keywords = keywords_tag.get('content', '')
            if keywords:
                metadata['tags'] = [tag.strip() for tag in keywords.split(',')]

        return metadata