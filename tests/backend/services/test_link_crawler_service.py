import unittest
from unittest.mock import MagicMock, patch
import pandas as pd  # Using standard pandas as per your global change
import numpy as np  # Keep if any mock might return numpy arrays, otherwise can remove
from datetime import datetime
from urllib.parse import urlparse  # Needed for URL path comparisons in tests

# Import the class to be tested and its dependencies
from src.backend.services.link_crawler_service import EdgeCrawler
from src.backend.config.link_crawler_config import LinkCrawlerConfig
from src.backend.data.link_graph_repository import LinkGraphStateManager
from src.backend.utils.strategies import (
    CrawlingStrategy,
    VisitedUrlManager,
)  # Need VisitedUrlManager for accurate mock setup
from src.backend.utils.http import HttpClient
from src.backend.utils.link_url import (
    LinkExtractorForEdges,
    LinkUrlFilter,
)  # Need LinkUrlFilter for accurate mock setup
from src.shared.interfaces import ILogger


class TestEdgeCrawler(unittest.TestCase):
    """
    Unit tests for the EdgeCrawler class.
    Mocks all external dependencies to test the crawler's logic in isolation.
    """

    def setUp(self):
        """
        Set up mock objects for all dependencies before each test.
        """
        # Initialize MagicMock objects for each dependency
        self.mock_config = MagicMock(spec=LinkCrawlerConfig)
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_state_manager = MagicMock(spec=LinkGraphStateManager)
        self.mock_http_client = MagicMock(spec=HttpClient)
        self.mock_link_extractor = MagicMock(spec=LinkExtractorForEdges)
        self.mock_crawling_strategy = MagicMock(spec=CrawlingStrategy)
        self.mock_url_filter = MagicMock(
            spec=LinkUrlFilter
        )  # Mock the internal url_filter of LinkExtractorForEdges

        # Configure default behaviors for mocks
        self.mock_config.max_redirects = 2
        self.mock_config.saving_scope_path = "/blog/"
        self.mock_config.max_pages_to_crawl = 5
        self.mock_config.save_interval_edges = (
            2  # This is the intended setting for the test
        )

        # LinkExtractorForEdges expects a LinkUrlFilter
        self.mock_link_extractor.url_filter = self.mock_url_filter
        self.mock_link_extractor.normalize_url.side_effect = (
            lambda x: urlparse(x)._replace(query="", fragment="").geturl()
        )
        self.mock_url_filter.is_valid_for_crawling.return_value = (
            True  # By default, all URLs are valid for crawling
        )

        # EdgeCrawler expects a get_queue method on the crawling strategy
        self.mock_crawling_strategy.get_queue.return_value = []

        # Instantiate the EdgeCrawler with the mock dependencies
        self.crawler = EdgeCrawler(
            config=self.mock_config,
            crawling_strategy=self.mock_crawling_strategy,
            state_manager=self.mock_state_manager,
            http_client=self.mock_http_client,
            link_extractor=self.mock_link_extractor,
            logger=self.mock_logger,
        )

        # Patch tqdm to prevent it from printing progress bars during tests
        self.mock_tqdm_patch = patch(
            "tqdm.tqdm", side_effect=lambda iterable, *args, **kwargs: iterable
        )
        self.mock_tqdm = self.mock_tqdm_patch.start()

    def tearDown(self):
        """
        Clean up resources after each test.
        """
        # Stop the tqdm patch
        self.mock_tqdm_patch.stop()

    # --- Test Cases for __init__ ---
    def test_init(self):
        """
        Test that the EdgeCrawler is initialized correctly with its dependencies.
        """
        self.assertIsInstance(self.crawler.config, MagicMock)
        self.assertIsInstance(self.crawler.crawling_strategy, MagicMock)
        self.assertIsInstance(self.crawler.state_manager, MagicMock)
        self.assertIsInstance(self.crawler.http_client, MagicMock)
        self.assertIsInstance(self.crawler.link_extractor, MagicMock)
        self.assertIsInstance(self.crawler.logger, MagicMock)
        self.assertEqual(self.crawler.edge_buffer, [])
        self.assertEqual(self.crawler.pages_processed_session, 0)

    # --- Test Cases for _process_page_for_edges ---
    def test_process_page_for_edges_successful_fetch_and_in_scope_links(self):
        """
        Test successful page fetch and extraction of links that are within the saving scope.
        """
        from_url = "http://example.com/blog/page1"
        html_content = """
        <html><body>
            <a href="/blog/link1">Link 1</a>
            <a href="/blog/link2">Link 2</a>
            <a href="/outside/link3">Link 3 (outside scope)</a>
        </body></html>
        """
        expected_extracted_links = [
            "http://example.com/blog/link1",
            "http://example.com/blog/link2",
            "http://example.com/outside/link3",
        ]
        self.mock_http_client.fetch.return_value = (200, html_content, None)
        self.mock_link_extractor.extract_links.return_value = set(
            expected_extracted_links
        )

        self.crawler._process_page_for_edges(from_url, 0)

        self.mock_http_client.fetch.assert_called_once_with(from_url)
        self.mock_logger.info.assert_called_with(f"Processed {from_url} [200]")
        self.mock_link_extractor.extract_links.assert_called_once_with(
            from_url, html_content
        )

        # Only links within /blog/ should be added to edge_buffer
        expected_edges = [
            {
                "FROM": "http://example.com/blog/page1",
                "TO": "http://example.com/blog/link1",
            },
            {
                "FROM": "http://example.com/blog/page1",
                "TO": "http://example.com/blog/link2",
            },
        ]
        # Use assertCountEqual for comparing lists of dictionaries where order doesn't matter
        # and there might be multiple ways the items are inserted (e.g. from set conversion)
        self.assertCountEqual(self.crawler.edge_buffer, expected_edges)

        # All extracted links should be added to the crawling strategy
        self.mock_crawling_strategy.add_links.assert_called_once()
        # Check that the links passed to add_links are as expected (order might vary from set)
        actual_add_links_args = [
            item[0] for item in self.mock_crawling_strategy.add_links.call_args[0][0]
        ]
        self.assertCountEqual(actual_add_links_args, expected_extracted_links)

    def test_process_page_for_edges_max_redirects_reached(self):
        """
        Test that processing is skipped if max redirects are reached.
        """
        from_url = "http://example.com/too_many_redirects"
        self.mock_config.max_redirects = 1

        self.crawler._process_page_for_edges(from_url, self.mock_config.max_redirects)

        self.mock_http_client.fetch.assert_not_called()
        self.mock_logger.warning.assert_called_once_with(
            f"Max redirects for {from_url}. Skipping."
        )
        self.assertEqual(self.crawler.edge_buffer, [])
        self.mock_crawling_strategy.add_links.assert_not_called()

    def test_process_page_for_edges_http_redirect(self):
        """
        Test handling of HTTP redirects.
        """
        from_url = "http://example.com/old_path"
        redirect_to_url = "http://example.com/blog/new_path"
        self.mock_http_client.fetch.return_value = (301, "", redirect_to_url)

        self.crawler._process_page_for_edges(from_url, 0)

        self.mock_http_client.fetch.assert_called_once_with(from_url)
        self.mock_logger.info.assert_called_once_with(f"Processed {from_url} [301]")
        self.mock_link_extractor.extract_links.assert_not_called()
        self.mock_link_extractor.normalize_url.assert_called_once_with(redirect_to_url)
        self.mock_url_filter.is_valid_for_crawling.assert_called_once_with(
            redirect_to_url
        )
        self.mock_crawling_strategy.add_links.assert_called_once_with(
            [(redirect_to_url, 1)]
        )
        self.assertEqual(self.crawler.edge_buffer, [])

    def test_process_page_for_edges_http_error(self):
        """
        Test handling of HTTP errors (e.g., 404, 500).
        """
        from_url = "http://example.com/non_existent"
        self.mock_http_client.fetch.return_value = (404, "", None)

        self.crawler._process_page_for_edges(from_url, 0)

        self.mock_http_client.fetch.assert_called_once_with(from_url)
        self.mock_logger.info.assert_called_once_with(f"Processed {from_url} [404]")
        self.mock_link_extractor.extract_links.assert_not_called()
        self.mock_crawling_strategy.add_links.assert_not_called()
        self.assertEqual(self.crawler.edge_buffer, [])

    # --- Test Cases for _save_edges_to_csv ---
    def test_save_edges_to_csv_success(self):
        """
        Test successful saving of edges to CSV via state manager.
        """
        self.crawler.edge_buffer = [
            {"FROM": "urlA", "TO": "urlB"},
            {"FROM": "urlC", "TO": "urlD"},
        ]

        self.crawler._save_edges_to_csv()

        expected_df = pd.DataFrame(
            [
                {"FROM": "urlA", "TO": "urlB"},
                {"FROM": "urlC", "TO": "urlD"},
            ]
        )

        # Verify append_edges_to_csv was called with the correct DataFrame
        # Use pd.testing.assert_frame_equal inside a custom assertion or after retrieving calls
        self.mock_state_manager.append_edges_to_csv.assert_called_once()
        actual_df_passed = self.mock_state_manager.append_edges_to_csv.call_args[0][0]
        pd.testing.assert_frame_equal(actual_df_passed, expected_df)

        self.assertEqual(self.crawler.edge_buffer, [])  # Buffer should be cleared

    def test_save_edges_to_csv_empty_buffer(self):
        """
        Test that save is skipped if the edge buffer is empty.
        """
        self.crawler.edge_buffer = []
        self.crawler._save_edges_to_csv()
        self.mock_state_manager.append_edges_to_csv.assert_not_called()

    # --- Test Cases for crawl ---
    def test_crawl_finishes_when_max_pages_reached(self):
        """
        Test the crawl loop ends when max_pages_to_crawl is reached.
        """
        self.mock_crawling_strategy.has_next.side_effect = [True] * 5 + [
            False
        ]  # 5 pages to process
        self.mock_crawling_strategy.get_next.side_effect = [
            ("http://example.com/blog/p1", 0),
            ("http://example.com/blog/p2", 0),
            ("http://example.com/blog/p3", 0),
            ("http://example.com/blog/p4", 0),
            ("http://example.com/blog/p5", 0),
        ]
        self.mock_http_client.fetch.return_value = (
            200,
            "<html><a href='/blog/next'></a></html>",
            None,
        )
        self.mock_link_extractor.extract_links.return_value = {
            "http://example.com/blog/next"
        }  # always yields one link for simplicity

        # Define a side effect for the mocked _save_edges_to_csv that clears the buffer
        def mock_save_edges_side_effect():
            # Mimic the actual behavior of _save_edges_to_csv in clearing the buffer
            self.crawler.edge_buffer = []

        # Ensure _save_edges_to_csv and save_frontier are called at intervals and at the end
        with patch.object(
            self.crawler, "_save_edges_to_csv", side_effect=mock_save_edges_side_effect
        ) as mock_save_edges:
            results = list(self.crawler.crawl())

            self.assertEqual(self.crawler.pages_processed_session, 5)
            self.assertEqual(self.mock_crawling_strategy.get_next.call_count, 5)
            # Based on save_interval_edges=2, and buffer clearing, this should be 3 calls:
            # after page 2, after page 4, and final save.
            self.assertEqual(mock_save_edges.call_count, 3)
            self.assertEqual(
                self.mock_state_manager.save_frontier.call_count, 3
            )  # Should align with mock_save_edges calls

            # Verify yielded statuses
            self.assertIn("Processed 1/5 pages.", results)
            self.assertIn("Processed 5/5 pages.", results)
            self.assertIn("Crawl finished. Processed 5 pages in this session.", results)

    def test_crawl_finishes_when_frontier_empty(self):
        """
        Test the crawl loop ends when the frontier becomes empty.
        """
        self.mock_crawling_strategy.has_next.side_effect = [
            True,
            True,
            False,
        ]  # Process 2 pages then empty
        self.mock_crawling_strategy.get_next.side_effect = [
            ("http://example.com/blog/a", 0),
            ("http://example.com/blog/b", 0),
        ]
        self.mock_http_client.fetch.return_value = (
            200,
            "<html><a href='/blog/test'></a></html>",
            None,
        )
        self.mock_link_extractor.extract_links.return_value = (
            set()
        )  # No new links added to frontier

        # Define a side effect for the mocked _save_edges_to_csv that clears the buffer
        def mock_save_edges_side_effect_empty_buffer():
            self.crawler.edge_buffer = []  # This will always be empty anyway in this test scenario

        with patch.object(
            self.crawler,
            "_save_edges_to_csv",
            side_effect=mock_save_edges_side_effect_empty_buffer,
        ) as mock_save_edges:
            results = list(self.crawler.crawl())

            self.assertEqual(self.crawler.pages_processed_session, 2)
            self.assertEqual(self.mock_crawling_strategy.get_next.call_count, 2)
            self.mock_logger.info.assert_any_call("Frontier is empty. Stopping crawl.")
            # If edge_buffer is always empty, _save_edges_to_csv returns immediately.
            # It's called once at the end.
            self.assertEqual(mock_save_edges.call_count, 1)
            # save_frontier is only called once at the very end as no intermediate saves occur.
            self.assertEqual(self.mock_state_manager.save_frontier.call_count, 1)

            # Verify yielded statuses
            self.assertIn("Processed 1/5 pages.", results)
            self.assertIn("Processed 2/5 pages.", results)
            self.assertIn("Crawl finished. Processed 2 pages in this session.", results)


if __name__ == "__main__":
    unittest.main()
