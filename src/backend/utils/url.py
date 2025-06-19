import re
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from bs4 import BeautifulSoup


class UrlFilter:
    def __init__(self, allowed_path_segment: str, base_domain: str):
        self.allowed_path_segment = allowed_path_segment
        self.base_domain = base_domain
        self.file_extension_pattern = re.compile(
            r"\.(jpg|jpeg|png|gif|pdf|doc|xls|zip|rar|mp3|mp4)$", re.I
        )

    def is_valid(self, url: str) -> bool:
        try:
            parsed_url = urlparse(url)
            return (
                parsed_url.scheme in ("http", "https")
                and parsed_url.netloc == self.base_domain
                and self.allowed_path_segment in parsed_url.path
                and not self.file_extension_pattern.search(parsed_url.path)
            )
        except Exception:
            return False


class LinkExtractor:
    def __init__(self, url_filter: UrlFilter, allowed_params: list[str]):
        self.url_filter = url_filter
        self.allowed_params = allowed_params

    def normalize_url(self, url: str) -> str:
        """Strips fragments and unwanted query parameters from a URL."""
        parsed = urlparse(url)

        # If there are query parameters, filter them
        if parsed.query:
            query_params = parse_qs(parsed.query)
            # Keep only the parameters that are in our whitelist
            filtered_params = {
                k: v for k, v in query_params.items() if k in self.allowed_params
            }
            # Re-encode the query string with only the allowed parameters
            clean_query = urlencode(filtered_params, doseq=True)
        else:
            clean_query = ""

        # Reconstruct the URL with the clean query and no fragment
        normalized = parsed._replace(query=clean_query, fragment="").geturl()
        return normalized

    def extract_links(self, base_url: str, html_content: str) -> list[str]:
        links = set()
        soup = BeautifulSoup(html_content, "lxml")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_link = urljoin(base_url, href)

            # Normalize the link before doing anything else
            normalized_link = self.normalize_url(absolute_link)

            if self.url_filter.is_valid(normalized_link):
                links.add(normalized_link)
        return list(links)
