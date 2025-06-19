import requests
import time
import random
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.backend.config.crawler_config import CrawlerConfig
from src.shared.interfaces import ILogger

class HttpClient:
    def __init__(self, config: CrawlerConfig, logger: ILogger):
        self.config = config
        self.logger = logger
        self.session = self._create_requests_session()
        self.current_delay = self.config.min_request_delay

    def _create_requests_session(self):
        retry = Retry(
            total=self.config.max_retries_request,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch(self, url: str) -> tuple[int, str, str | None]:
        headers = {"User-Agent": random.choice(self.config.user_agents)}
        try:
            time.sleep(self.current_delay)
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config.request_timeout,
                allow_redirects=False,
            )
            if 300 <= response.status_code < 400:
                redirect_url = response.headers.get("Location")
                return response.status_code, "", urljoin(url, redirect_url)
            return response.status_code, response.text, None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
            return -2, f"Request Error: {e}", None