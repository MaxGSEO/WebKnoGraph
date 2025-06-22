import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
from datetime import datetime
from tqdm import (
    tqdm,
)  # Import tqdm if it's used in the tested code, to avoid NameError if not mocked

# Adjust import paths as necessary based on your project structure
from src.backend.services.crawler_service import WebCrawler
from src.backend.config.crawler_config import CrawlerConfig
from src.backend.utils.strategies import CrawlingStrategy
from src.backend.data.repositories import (
    CrawlStateRepository,
)  # Renamed from StateManager
from src.backend.utils.http import HttpClient
from src.backend.utils.url import UrlFilter, LinkExtractor
from src.shared.interfaces import ILogger


class TestWebCrawler(unittest.TestCase):
    def setUp(self):
        # Initialize mock objects for all dependencies
        self.mock_config = MagicMock(spec=CrawlerConfig)
        self.mock_crawling_strategy = MagicMock(spec=CrawlingStrategy)
        self.mock_state_repository = MagicMock(spec=CrawlStateRepository)
        self.mock_http_client = MagicMock(spec=HttpClient)
        self.mock_url_filter = MagicMock(spec=UrlFilter)
        self.mock_link_extractor = MagicMock(spec=LinkExtractor)
        self.mock_logger = MagicMock(spec=ILogger)

        # Configure default behaviors for mocks
        self.mock_config.max_redirects = 2
        self.mock_config.parquet_path = "/mock/path/to/parquet"
        self.mock_config.save_interval_pages = 2
        self.mock_config.max_pages_to_crawl = 5
        self.mock_url_filter.is_valid.return_value = True
        self.mock_link_extractor.normalize_url.side_effect = (
            lambda x: x
        )  # Normalize URL to return the same URL for testing simplicity

        # Instantiate WebCrawler with mocks
        self.crawler = WebCrawler(
            config=self.mock_config,
            crawling_strategy=self.mock_crawling_strategy,
            state_repository=self.mock_state_repository,
            http_client=self.mock_http_client,
            url_filter=self.mock_url_filter,
            link_extractor=self.mock_link_extractor,
            logger=self.mock_logger,
        )

        # Mock tqdm to prevent it from printing progress bars during tests
        self.mock_tqdm_patch = patch(
            "tqdm.tqdm", side_effect=lambda iterable, *args, **kwargs: iterable
        )
        self.mock_tqdm = self.mock_tqdm_patch.start()

    def tearDown(self):
        # Stop the tqdm patch after each test
        self.mock_tqdm_patch.stop()

    # --- Test Cases for __init__ ---
    def test_init(self):
        self.assertIsInstance(self.crawler.config, MagicMock)
        self.assertIsInstance(self.crawler.crawling_strategy, MagicMock)
        self.assertIsInstance(self.crawler.state_repository, MagicMock)
        self.assertIsInstance(self.crawler.http_client, MagicMock)
        self.assertIsInstance(self.crawler.url_filter, MagicMock)
        self.assertIsInstance(self.crawler.link_extractor, MagicMock)
        self.assertIsInstance(self.crawler.logger, MagicMock)
        self.assertEqual(self.crawler.data_buffer, [])
        self.assertEqual(self.crawler.pages_crawled_session, 0)

    # --- Test Cases for _process_url ---
    def test_process_url_successful_fetch(self):
        test_url = "http://example.com/page1"
        self.mock_http_client.fetch.return_value = (
            200,
            "<html><body><a href='/link1'>Link</a></body></html>",
            None,
        )
        self.mock_link_extractor.extract_links.return_value = [
            "http://example.com/link1"
        ]

        self.crawler._process_url((test_url, 0))

        self.mock_http_client.fetch.assert_called_once_with(test_url)
        self.mock_logger.info.assert_called_with(f"Fetched {test_url} [200]")
        self.mock_link_extractor.extract_links.assert_called_once()
        self.mock_crawling_strategy.add_links.assert_called_once_with(
            [("http://example.com/link1", 0)]
        )
        self.assertEqual(len(self.crawler.data_buffer), 1)
        self.assertEqual(self.crawler.data_buffer[0]["URL"], test_url)
        self.assertEqual(self.crawler.data_buffer[0]["Status_Code"], 200)
        self.assertIn("Link", self.crawler.data_buffer[0]["Content"])

    def test_process_url_max_redirects_reached(self):
        test_url = "http://example.com/redirect_loop"
        self.mock_config.max_redirects = 1  # Set max redirects for this specific test

        self.crawler._process_url((test_url, self.mock_config.max_redirects))

        self.mock_http_client.fetch.assert_not_called()  # Should not fetch if max redirects reached
        self.mock_logger.info.assert_not_called()
        self.assertEqual(len(self.crawler.data_buffer), 1)
        self.assertEqual(self.crawler.data_buffer[0]["URL"], test_url)
        self.assertEqual(self.crawler.data_buffer[0]["Status_Code"], 999)
        self.assertEqual(
            self.crawler.data_buffer[0]["Content"], "Max redirects reached"
        )

    def test_process_url_redirect(self):
        test_url = "http://example.com/old_page"
        redirect_to_url = "http://example.com/new_page"
        self.mock_http_client.fetch.return_value = (301, "", redirect_to_url)

        self.crawler._process_url((test_url, 0))

        self.mock_http_client.fetch.assert_called_once_with(test_url)
        self.mock_logger.info.assert_called_with(f"Fetched {test_url} [301]")
        self.mock_link_extractor.extract_links.assert_not_called()  # No content to extract
        self.mock_crawling_strategy.add_links.assert_called_once_with(
            [(redirect_to_url, 1)]
        )
        self.assertEqual(len(self.crawler.data_buffer), 1)
        self.assertEqual(self.crawler.data_buffer[0]["URL"], test_url)
        self.assertEqual(self.crawler.data_buffer[0]["Status_Code"], 301)
        self.assertEqual(self.crawler.data_buffer[0]["Content"], "")

    def test_process_url_fetch_error(self):
        test_url = "http://example.com/error_page"
        self.mock_http_client.fetch.return_value = (
            500,
            "",
            None,
        )  # Simulate server error

        self.crawler._process_url((test_url, 0))

        self.mock_http_client.fetch.assert_called_once_with(test_url)
        self.mock_logger.info.assert_called_with(f"Fetched {test_url} [500]")
        self.mock_link_extractor.extract_links.assert_not_called()
        self.mock_crawling_strategy.add_links.assert_not_called()
        self.assertEqual(len(self.crawler.data_buffer), 1)
        self.assertEqual(self.crawler.data_buffer[0]["URL"], test_url)
        self.assertEqual(self.crawler.data_buffer[0]["Status_Code"], 500)
        self.assertEqual(self.crawler.data_buffer[0]["Content"], "")

    # --- Test Cases for _save_buffer_to_parquet ---
    @patch("fireducks.pandas.DataFrame.to_parquet")
    @patch("os.makedirs")
    @patch("time.time", return_value=1234567890.0)  # Mock time for consistent filename
    def test_save_buffer_to_parquet_success(
        self, mock_time, mock_makedirs, mock_to_parquet
    ):
        self.crawler.data_buffer = [
            {"URL": "url1", "Status_Code": 200, "Content": "content1"},
            {"URL": "url2", "Status_Code": 200, "Content": "content2"},
        ]
        expected_log_message = f"✅ Saved a batch of **2** pages to partition `{self.mock_config.parquet_path}/crawl_date={datetime.now().date()}`"

        result = self.crawler._save_buffer_to_parquet()

        mock_makedirs.assert_called_once()
        mock_to_parquet.assert_called_once()
        self.mock_logger.info.assert_called_with(expected_log_message)
        self.assertEqual(self.crawler.data_buffer, [])  # Buffer should be cleared
        self.assertEqual(result, expected_log_message)

    def test_save_buffer_to_parquet_empty_buffer(self):
        self.crawler.data_buffer = []
        result = self.crawler._save_buffer_to_parquet()
        self.assertIsNone(result)
        self.mock_logger.info.assert_not_called()  # No save, no log message
        self.mock_logger.error.assert_not_called()

    @patch("fireducks.pandas.DataFrame.to_parquet", side_effect=Exception("Disk Full"))
    @patch("os.makedirs")
    @patch("time.time", return_value=1234567890.0)
    def test_save_buffer_to_parquet_failure(
        self, mock_time, mock_makedirs, mock_to_parquet
    ):
        self.crawler.data_buffer = [
            {"URL": "url1", "Status_Code": 200, "Content": "content1"}
        ]
        result = self.crawler._save_buffer_to_parquet()
        self.assertIsNone(result)
        self.mock_logger.error.assert_called_once_with(
            "Failed to save to Parquet: Disk Full"
        )
        self.assertEqual(
            len(self.crawler.data_buffer), 1
        )  # Buffer should not be cleared on failure

    # --- Test Cases for crawl ---
    def test_crawl_finishes_when_max_pages_reached(self):
        # Configure mocks for crawl loop
        self.mock_crawling_strategy.has_next.side_effect = [
            True,
            True,
            True,
            True,
            True,
            False,
        ]  # Allows 5 pages to be crawled
        self.mock_crawling_strategy.get_next.side_effect = [
            ("http://example.com/p1", 0),
            ("http://example.com/p2", 0),
            ("http://example.com/p3", 0),
            ("http://example.com/p4", 0),
            ("http://example.com/p5", 0),
        ]
        self.mock_http_client.fetch.return_value = (200, "content", None)
        self.mock_config.max_pages_to_crawl = 5
        self.mock_config.save_interval_pages = (
            1  # Save after every page for simplicity in test
        )

        # Mock saving to parquet to avoid file system ops
        with patch.object(
            self.crawler, "_save_buffer_to_parquet", return_value="Save message"
        ) as mock_save:
            results = list(self.crawler.crawl())

            self.assertEqual(self.crawler.pages_crawled_session, 5)
            self.assertEqual(self.mock_crawling_strategy.get_next.call_count, 5)
            self.assertEqual(
                self.mock_state_repository.save_frontier.call_count, 6
            )  # 5 intermediate saves + 1 final save
            self.assertEqual(mock_save.call_count, 6)
            # Corrected assertion: logger is called with "Crawl finished. Performing final save..."
            self.mock_logger.info.assert_any_call(
                "Crawl finished. Performing final save..."
            )
            # This assertion is now removed as the message is yielded, not logged.
            # self.mock_logger.info.assert_any_call("Crawl finished. Processed 5 pages.")

            # Verify the yielded statuses
            self.assertEqual(results[0]["status"], "Crawled 1/5 pages.")
            self.assertEqual(results[4]["status"], "Crawled 5/5 pages.")
            self.assertEqual(
                results[5]["status"], "Crawl finished. Processed 5 pages."
            )  # This correctly checks the yielded message

    def test_crawl_finishes_when_frontier_empty(self):
        self.mock_crawling_strategy.has_next.side_effect = [
            True,
            True,
            False,
        ]  # Allows 2 pages, then empty
        self.mock_crawling_strategy.get_next.side_effect = [
            ("http://example.com/a", 0),
            ("http://example.com/b", 0),
        ]
        self.mock_http_client.fetch.return_value = (200, "content", None)
        self.mock_config.max_pages_to_crawl = (
            10  # Set higher than actual crawled to test empty frontier
        )
        self.mock_config.save_interval_pages = 1

        with patch.object(
            self.crawler, "_save_buffer_to_parquet", return_value="Save message"
        ) as mock_save:
            results = list(self.crawler.crawl())

            self.assertEqual(self.crawler.pages_crawled_session, 2)
            self.assertEqual(self.mock_crawling_strategy.get_next.call_count, 2)
            self.mock_logger.info.assert_any_call("Frontier is empty. Stopping crawl.")
            self.assertEqual(self.mock_state_repository.save_frontier.call_count, 3)
            self.assertEqual(mock_save.call_count, 3)

            # Verify the yielded statuses
            self.assertEqual(results[0]["status"], "Crawled 1/10 pages.")
            self.assertEqual(results[1]["status"], "Crawled 2/10 pages.")
            self.assertEqual(results[2]["status"], "Crawl finished. Processed 2 pages.")


if __name__ == "__main__":
    unittest.main()
