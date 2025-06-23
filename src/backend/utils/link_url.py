# File: src/backend/utils/link_url.py (NEW FILE)
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


# --- NEW UrlFilter for link graph crawler ---
class LinkUrlFilter:
    def __init__(self, crawling_scope_path: str, base_domain: str):
        self.crawling_scope_path = crawling_scope_path
        self.base_domain = base_domain
        self.file_extension_pattern = re.compile(
            r"\.(pdf|jpg|jpeg|png|gif|zip|rar|mp3|mp4|svg|xml|css|js|webp|ico)$",
            re.IGNORECASE,
        )

    def is_valid_for_crawling(self, url: str) -> bool:
        try:
            parsed_url = urlparse(url)
            return (
                parsed_url.scheme in ("http", "https")
                and parsed_url.netloc == self.base_domain
                and parsed_url.path.startswith(self.crawling_scope_path)
                and not self.file_extension_pattern.search(parsed_url.path)
            )
        except Exception:
            return False


# --- NEW LinkExtractor for link graph crawler ---
class LinkExtractorForEdges:
    def __init__(self, url_filter: LinkUrlFilter):  # Uses LinkUrlFilter
        self.url_filter = url_filter

    def normalize_url(self, url: str) -> str:
        """Strips ALL query parameters and fragments from a URL."""
        return urlparse(url)._replace(query="", fragment="").geturl()

    def extract_links(self, base_url: str, html_content: str) -> set[str]:
        links = set()
        soup = BeautifulSoup(html_content, "lxml")
        for a_tag in soup.find_all("a", href=True):
            absolute_link = urljoin(base_url, a_tag["href"])
            normalized_link = self.normalize_url(absolute_link)
            if self.url_filter.is_valid_for_crawling(normalized_link):
                links.add(normalized_link)
        return links
