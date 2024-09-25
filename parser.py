import time
from typing import List, Set

import requests
from bs4 import BeautifulSoup


class SitemapParser:
    """A class to parse sitemaps and extract all page URLs from a website."""

    def __init__(self, user_agent: str = None, delay: float = 1.0):
        """Initialize the SitemapParser with a custom User-Agent and delay between each request.

        Args:
            user_agent (str, optional): Custom User-Agent string for HTTP headers. Defaults to None.
            delay (float, optional): Delay in seconds between HTTP requests. Defaults to 1.0.

        """
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent
                or (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"
                )
            }
        )
        self.delay = delay

    def fetch_content(self, url: str) -> bytes:
        """Fetch the content of a URL

        Args:
            url (str): URL to fetch

        Raises:
            ValueError: requests.RequestException if the request fails

        Returns:
            bytes: Content of the response
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.delay)
            return response.content

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            raise

    def parse_xml(self, content: bytes) -> BeautifulSoup:
        """Parse XML conntent using Beautiful Soup

        Args:
            content (bytes): XML content as bytes

        Returns:
            BeautifulSoup: BeautifulSoup Object
        """
        return BeautifulSoup(content, "xml")

    def extract_urls(self, soup: BeautifulSoup, tag: str = "loc") -> List[str]:
        """Extract URLs from a BeautifulSoup object

        Args:
            soup (BeautifulSoup): BeautifulSoup Object containg XML
            tag (str, optional): XML stag to search for URLs. Defaults to 'loc'.

        Returns:
            List[str]: List of URLs as strings
        """
        return [loc.text.strip() for loc in soup.find_all(tag) if loc.text]

    def is_sitemap(self, url: str) -> bool:
        """Check if a URL points to a sitemap based on its file extension.

        Args:
            url (str): URL to check

        Returns:
            bool: True is URL is a sitemap, False otherwise.
        """
        return url.lower().endswith(".xml")

    def is_html_page(self, url: str) -> bool:
        """Check if a URL is likely to be a HTML page.

        Args:
            url (str): URL to check

        Returns:
            bool: True is URL is likely an HTML page, False otherwise.
        """
        # Exclude common non-HTML file extensions
        non_html_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
            ".svg",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".zip",
            ".rar",
            ".gz",
            ".tar",
        )
        return not url.lower().endswith(non_html_extensions)

    def get_all_page_urls(self, sitemap_url: str) -> Set[str]:
        """Recursively fetch all page URLs from a sitemap

        Args:
            base_url (str): URL of the sitemap

        Returns:
            Set[str]: Set of all page URLs
        """
        page_urls = set()

        try:
            content = self.fetch_content(sitemap_url)
            soup = self.parse_xml(content)
            urls = self.extract_urls(soup)

            for url in urls:
                if self.is_sitemap(url=url):

                    # Recursively parse nested sitemaps
                    page_urls.update(self.get_all_page_urls(url))

                elif self.is_html_page(url):
                    page_urls.add(url)

                else:

                    print(f"Excluded non-HTML URL: {url}")

        except Exception as e:
            print(f"Failed to process sitemap {sitemap_url}: {e}")

        return page_urls
