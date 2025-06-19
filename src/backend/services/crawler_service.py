import os
import gc
import time
import fireducks.pandas as pd
from datetime import datetime
from tqdm import tqdm

# Updated import paths based on the new structure
from src.backend.config.crawler_config import CrawlerConfig
from src.backend.utils.strategies import CrawlingStrategy
from src.backend.data.repositories import (
    CrawlStateRepository,
)  # Renamed from StateManager
from src.backend.utils.http import HttpClient
from src.backend.utils.url import UrlFilter, LinkExtractor
from src.shared.interfaces import ILogger


class WebCrawler:
    def __init__(
        self,
        config: CrawlerConfig,
        crawling_strategy: CrawlingStrategy,
        state_repository: CrawlStateRepository,  # Updated parameter name
        http_client: HttpClient,
        url_filter: UrlFilter,
        link_extractor: LinkExtractor,
        logger: ILogger,
    ):
        self.config = config
        self.crawling_strategy = crawling_strategy
        self.state_repository = state_repository  # Updated attribute name
        self.http_client = http_client
        self.url_filter = url_filter
        self.link_extractor = link_extractor
        self.logger = logger
        self.data_buffer = []
        self.pages_crawled_session = 0

    def _process_url(self, url_info: tuple[str, int]):
        url, num_redirects = url_info
        if num_redirects >= self.config.max_redirects:
            self.data_buffer.append(
                {"URL": url, "Status_Code": 999, "Content": "Max redirects reached"}
            )
            return

        status, content, redirect_url = self.http_client.fetch(url)
        self.logger.info(f"Fetched {url} [{status}]")
        self.data_buffer.append(
            {
                "URL": url,
                "Status_Code": status,
                "Content": content if 200 <= status < 300 else "",
            }
        )

        if 200 <= status < 300:
            extracted = self.link_extractor.extract_links(url, content)
            self.crawling_strategy.add_links([(link, 0) for link in extracted])
            del content
        elif redirect_url:
            normalized_redirect = self.link_extractor.normalize_url(redirect_url)
            if self.url_filter.is_valid(normalized_redirect):
                self.crawling_strategy.add_links(
                    [(normalized_redirect, num_redirects + 1)]
                )

    def _save_buffer_to_parquet(self) -> str | None:
        if not self.data_buffer:
            return None
        num_records = len(self.data_buffer)
        df = pd.DataFrame(self.data_buffer)
        today = datetime.now().date()
        df["crawl_date"] = today
        try:
            partition_path = os.path.join(
                self.config.parquet_path, f"crawl_date={today}"
            )
            os.makedirs(partition_path, exist_ok=True)
            df.to_parquet(
                path=os.path.join(partition_path, f"{int(time.time())}.parquet"),
                engine="pyarrow",
                compression="snappy",
            )
            log_message = f"✅ Saved a batch of **{num_records}** pages to partition `{partition_path}`"
            self.logger.info(log_message)
            self.data_buffer = []
            gc.collect()
            return log_message
        except Exception as e:
            self.logger.error(f"Failed to save to Parquet: {e}")
            return None

    def crawl(self):
        pbar = tqdm(total=self.config.max_pages_to_crawl, desc="Crawling Progress")
        while self.pages_crawled_session < self.config.max_pages_to_crawl:
            if not self.crawling_strategy.has_next():
                self.logger.info("Frontier is empty. Stopping crawl.")
                break

            url_data = self.crawling_strategy.get_next()
            self._process_url(url_data)
            self.pages_crawled_session += 1
            pbar.update(1)

            save_event_msg = None
            if len(self.data_buffer) >= self.config.save_interval_pages:
                save_event_msg = self._save_buffer_to_parquet()
                current_frontier = []
                if hasattr(self.crawling_strategy, "queue"):
                    current_frontier = list(self.crawling_strategy.queue)
                elif hasattr(self.crawling_strategy, "stack"):
                    current_frontier = list(self.crawling_strategy.stack)
                self.state_repository.save_frontier(current_frontier)

            yield {
                "status": f"Crawled {self.pages_crawled_session}/{self.config.max_pages_to_crawl} pages.",
                "save_event": save_event_msg,
            }

        pbar.close()
        self.logger.info("Crawl finished. Performing final save...")
        final_save_msg = self._save_buffer_to_parquet()
        current_frontier = []
        if hasattr(self.crawling_strategy, "queue"):
            current_frontier = list(self.crawling_strategy.queue)
        elif hasattr(self.crawling_strategy, "stack"):
            current_frontier = list(self.crawling_strategy.stack)
        self.state_repository.save_frontier(current_frontier)

        yield {
            "status": f"Crawl finished. Processed {self.pages_crawled_session} pages.",
            "save_event": final_save_msg,
        }
