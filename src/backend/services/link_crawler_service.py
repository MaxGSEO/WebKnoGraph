# File: src/backend/services/link_crawler_service.py
import gc
import fireducks.pandas as pd  # Using fireducks.pandas as specified
from urllib.parse import urlparse
from tqdm import tqdm

from src.backend.config.link_crawler_config import LinkCrawlerConfig  # Using new config
from src.backend.data.link_graph_repository import (
    LinkGraphStateManager,
)  # Using new state manager
from src.backend.utils.strategies import (
    CrawlingStrategy,
)  # Reusing strategies
from src.backend.utils.http import HttpClient  # Reusing existing HttpClient
from src.backend.utils.link_url import (
    LinkExtractorForEdges,
)  # Using new URL components
from src.shared.interfaces import ILogger


class EdgeCrawler:  # Renamed from EdgeCrawler
    def __init__(
        self,
        config: LinkCrawlerConfig,
        crawling_strategy: CrawlingStrategy,
        state_manager: LinkGraphStateManager,
        http_client: HttpClient,
        link_extractor: LinkExtractorForEdges,
        logger: ILogger,
    ):
        self.config = config
        self.crawling_strategy = crawling_strategy
        self.state_manager = state_manager
        self.http_client = http_client
        self.link_extractor = link_extractor
        self.logger = logger
        self.edge_buffer = []
        self.pages_processed_session = 0

    def _process_page_for_edges(self, from_url: str, num_redirects: int):
        if num_redirects >= self.config.max_redirects:
            self.logger.warning(f"Max redirects for {from_url}. Skipping.")
            return

        status, content, redirect_url = self.http_client.fetch(
            from_url
        )  # HttpClient expects a config compatible with LinkCrawlerConfig
        self.logger.info(f"Processed {from_url} [{status}]")

        if 200 <= status < 300 and content:
            linked_urls = self.link_extractor.extract_links(from_url, content)

            from_url_path = urlparse(from_url).path
            saving_scope = self.config.saving_scope_path

            if from_url_path.startswith(saving_scope):
                for to_url in linked_urls:
                    to_url_path = urlparse(to_url).path
                    if to_url_path.startswith(saving_scope):
                        self.edge_buffer.append({"FROM": from_url, "TO": to_url})

            # Add links to crawling strategy (for further crawling)
            self.crawling_strategy.add_links([(link, 0) for link in linked_urls])
            del content  # Free up memory

        elif redirect_url:
            normalized_redirect = self.link_extractor.normalize_url(redirect_url)
            # Use the link_extractor's url_filter for validation
            if self.link_extractor.url_filter.is_valid_for_crawling(
                normalized_redirect
            ):
                self.crawling_strategy.add_links(
                    [(normalized_redirect, num_redirects + 1)]
                )

    def _save_edges_to_csv(self):
        if not self.edge_buffer:
            return

        df = pd.DataFrame(self.edge_buffer)
        self.state_manager.append_edges_to_csv(df)  # Use state manager to save
        self.edge_buffer = []
        gc.collect()

    def crawl(self):
        pbar = tqdm(total=self.config.max_pages_to_crawl, desc="Processing Pages")
        while self.pages_processed_session < self.config.max_pages_to_crawl:
            if not self.crawling_strategy.has_next():
                self.logger.info("Frontier is empty. Stopping crawl.")
                break

            url_data = self.crawling_strategy.get_next()
            self._process_page_for_edges(url_data[0], url_data[1])
            self.pages_processed_session += 1
            pbar.update(1)

            if len(self.edge_buffer) >= self.config.save_interval_edges:
                self._save_edges_to_csv()
                self.state_manager.save_frontier(self.crawling_strategy.get_queue())

            yield f"Processed {self.pages_processed_session}/{self.config.max_pages_to_crawl} pages."

        pbar.close()
        self.logger.info("Crawl finished. Performing final save...")
        self._save_edges_to_csv()
        self.state_manager.save_frontier(self.crawling_strategy.get_queue())
        yield f"Crawl finished. Processed {self.pages_processed_session} pages in this session."
