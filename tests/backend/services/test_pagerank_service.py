# File: tests/backend/services/test_pagerank_service.py

import unittest
import os
import pandas as pd
from unittest.mock import MagicMock, patch, ANY, call  # Import ANY and call
import tempfile
import pandas.errors  # Import for specific pandas exceptions

# Assuming the project structure allows direct import like this or needs adjustment
# Adjust import path if necessary based on actual project root
from src.backend.services.pagerank_service import CSVLoader, PageRankService
from src.shared.interfaces import ILogger
from src.backend.config.pagerank_config import PageRankConfig
from src.backend.utils.url_processing import URLProcessor
from src.backend.graph.analyzer import PageRankGraphAnalyzer, HITSGraphAnalyzer


# Mock implementations for interfaces and external classes
class MockLogger(ILogger):
    """A mock logger for testing purposes."""

    def debug(self, message: str):
        pass

    def info(self, message: str):
        pass

    def warning(self, message: str):
        pass

    def error(self, message: str):
        pass

    def critical(self, message: str):
        pass

    def exception(self, message: str):
        pass


class MockPageRankConfig(PageRankConfig):
    """A mock PageRankConfig for testing purposes."""

    def __init__(
        self,
        output_analysis_path="mock_pagerank.csv",
        input_edge_list_path="mock_link_graph.csv",
    ):
        # For testing, we only need to mock the paths it holds
        self.output_analysis_path = output_analysis_path
        self.input_edge_list_path = input_edge_list_path
        # Add any other attributes that might be accessed by PageRankService if necessary
        self.max_nodes_to_display = 100
        self.graph_display_limit = 50


class TestCSVLoader(unittest.TestCase):
    """Unit tests for the CSVLoader class."""

    def setUp(self):
        # Create a temporary directory for test CSV files
        self.test_dir = tempfile.mkdtemp()
        self.valid_csv_path = os.path.join(self.test_dir, "valid_data.csv")
        self.empty_csv_path = os.path.join(self.test_dir, "empty_data.csv")
        self.malformed_csv_path = os.path.join(self.test_dir, "malformed_data.csv")

        # Create a valid CSV file
        pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]}).to_csv(
            self.valid_csv_path, index=False
        )
        # Create an empty CSV file (truly empty, no headers)
        with open(self.empty_csv_path, "w") as f:
            pass
        # Create a malformed CSV file (e.g., a row with too many columns)
        with open(self.malformed_csv_path, "w") as f:
            f.write("col1,col2\n1,2,3")  # Too many columns in the second row

    def tearDown(self):
        # Clean up the temporary directory and files
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_load_data_success(self):
        """Test loading a valid, non-empty CSV file."""
        loader = CSVLoader(self.valid_csv_path)
        df = loader.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(list(df.columns), ["col1", "col2"])
        self.assertEqual(df.shape, (2, 2))

    def test_load_data_file_not_found(self):
        """Test loading a non-existent CSV file."""
        loader = CSVLoader("non_existent_file.csv")
        with self.assertRaises(FileNotFoundError):
            loader.load_data()

    def test_load_data_empty_file(self):
        """
        Test loading an empty CSV file.
        The original service code would catch pandas.errors.EmptyDataError
        and re-raise it as IOError, containing the original message.
        """
        loader = CSVLoader(self.empty_csv_path)
        with self.assertRaises(IOError) as cm:
            loader.load_data()
        self.assertIn(
            "No columns to parse from file", str(cm.exception)
        )  # Check for part of pandas' EmptyDataError message
        self.assertIn(
            "Error loading CSV data from", str(cm.exception)
        )  # Check for the outer IOError message

    def test_load_data_malformed_file(self):
        """
        Test loading a malformed CSV file.
        The original service code's generic exception handling wraps
        pandas parsing errors into IOError.
        """
        loader = CSVLoader(self.malformed_csv_path)
        with self.assertRaises(IOError) as cm:
            loader.load_data()
        self.assertIn("Error loading CSV data from", str(cm.exception))
        # The specific internal pandas error message might vary (e.g. 'ParserError'),
        # but the wrapper 'Error loading CSV data' should always be present.


# Patch the classes where they are imported within pagerank_service.py
@patch("src.backend.services.pagerank_service.CSVLoader")
@patch("src.backend.services.pagerank_service.PageRankGraphAnalyzer")
@patch("src.backend.services.pagerank_service.HITSGraphAnalyzer")
class TestPageRankService(unittest.TestCase):
    """Unit tests for the PageRankService class."""

    def setUp(self):
        self.mock_config = MockPageRankConfig()
        self.mock_logger = MockLogger()
        self.pagerank_service = PageRankService(self.mock_config, self.mock_logger)

        # Data for PageRank CSV
        self.pagerank_data = {
            "URL": ["http://a.com", "http://b.com", "http://c.com"],
            "Folder_Depth": [0, 1, 2],
            "PageRank": [0.3, 0.5, 0.2],
        }
        self.pagerank_df = pd.DataFrame(self.pagerank_data)

        # Data for Link Graph CSV
        self.link_graph_data = {
            "FROM": ["http://a.com", "http://b.com"],
            "TO": ["http://b.com", "http://c.com"],
        }
        self.link_graph_df = pd.DataFrame(self.link_graph_data)

        # These will hold the mock instances for CSVLoader, set by _setup_csv_loader_mock
        self.mock_pagerank_loader_instance = None
        self.mock_link_graph_loader_instance = None

    # Helper to set up CSVLoader mocks for different scenarios
    def _setup_csv_loader_mock(
        self, MockCSVLoader, pagerank_behavior="success", link_graph_behavior="success"
    ):
        """
        Configures the MockCSVLoader behavior.
        pagerank_behavior/link_graph_behavior can be:
        "success": returns self.pagerank_df / self.link_graph_df
        "not_found": raises FileNotFoundError
        "empty": simulates a CSVLoader returning an empty DataFrame with columns, triggering PageRankService's ValueError or specific empty handling.
        "malformed": simulates a parsing error wrapped in IOError from CSVLoader.
        "missing_cols_pagerank": returns a DataFrame missing required pagerank columns.
        "missing_cols_link_graph": returns a DataFrame missing required link graph columns.
        """
        self.mock_pagerank_loader_instance = MagicMock()
        self.mock_link_graph_loader_instance = MagicMock()

        # Mock for PageRank CSV loader instance
        if pagerank_behavior == "success":
            self.mock_pagerank_loader_instance.load_data.return_value = (
                self.pagerank_df.copy()
            )
        elif pagerank_behavior == "not_found":
            self.mock_pagerank_loader_instance.load_data.side_effect = (
                FileNotFoundError(self.mock_config.output_analysis_path)
            )
        elif pagerank_behavior == "empty":
            # Simulate CSVLoader raising ValueError for empty DataFrame (as per original CSVLoader code)
            self.mock_pagerank_loader_instance.load_data.side_effect = ValueError(
                f"CSV file at {self.mock_config.output_analysis_path} is empty."
            )
        elif pagerank_behavior == "malformed":
            self.mock_pagerank_loader_instance.load_data.side_effect = IOError(
                f"Error loading CSV data from {self.mock_config.output_analysis_path}: Some parsing error"
            )
        elif pagerank_behavior == "missing_cols_pagerank":
            self.mock_pagerank_loader_instance.load_data.return_value = pd.DataFrame(
                {"URL": ["a"], "PageRank": [0.1]}
            )
        else:
            self.mock_pagerank_loader_instance.load_data.side_effect = RuntimeError(
                f"Unexpected pagerank_behavior: {pagerank_behavior}"
            )

        # Mock for Link Graph CSV loader instance
        if link_graph_behavior == "success":
            self.mock_link_graph_loader_instance.load_data.return_value = (
                self.link_graph_df.copy()
            )
        elif link_graph_behavior == "not_found":
            self.mock_link_graph_loader_instance.load_data.side_effect = (
                FileNotFoundError(self.mock_config.input_edge_list_path)
            )
        elif link_graph_behavior == "empty":
            # Simulate CSVLoader returning an empty DataFrame with headers.
            # The PageRankService's own `if not self.full_link_graph_df.empty:` check will then trigger the warning.
            self.mock_link_graph_loader_instance.load_data.return_value = pd.DataFrame(
                columns=["FROM", "TO"]
            )
        elif link_graph_behavior == "malformed":
            self.mock_link_graph_loader_instance.load_data.side_effect = IOError(
                f"Error loading CSV data from {self.mock_config.input_edge_list_path}: Some parsing error"
            )
        elif link_graph_behavior == "missing_cols_link_graph":
            self.mock_link_graph_loader_instance.load_data.return_value = pd.DataFrame(
                {"FROM": ["a"]}
            )
        else:
            self.mock_link_graph_loader_instance.load_data.side_effect = RuntimeError(
                f"Unexpected link_graph_behavior: {link_graph_behavior}"
            )

        # Configure the CSVLoader class mock itself to return the specific loader instances
        MockCSVLoader.side_effect = lambda path: {
            self.mock_config.output_analysis_path: self.mock_pagerank_loader_instance,
            self.mock_config.input_edge_list_path: self.mock_link_graph_loader_instance,
        }.get(path, MagicMock(name=f"CSVLoader_unexpected_path_{path}"))

    def test_initial_data_load_success(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test successful initial data load for both PageRank and Link Graph."""
        self._setup_csv_loader_mock(MockCSVLoader, "success", "success")

        # Create specific mock instances for the analyzers that the service will receive
        mock_pr_analyzer_instance = MagicMock(spec=PageRankGraphAnalyzer)
        mock_hits_analyzer_instance = MagicMock(spec=HITSGraphAnalyzer)

        # Configure the patched analyzer *classes* to return these specific instances when called (instantiated)
        MockPageRankGraphAnalyzer.return_value = mock_pr_analyzer_instance
        MockHITSGraphAnalyzer.return_value = mock_hits_analyzer_instance

        status = self.pagerank_service.initial_data_load()

        self.assertIn(
            f"✅ PageRank data loaded from {self.mock_config.output_analysis_path}",
            status,
        )
        self.assertIn(
            f"✅ Link graph data loaded from {self.mock_config.input_edge_list_path}",
            status,
        )
        self.assertFalse(self.pagerank_service.full_pagerank_df.empty)
        self.assertFalse(self.pagerank_service.full_link_graph_df.empty)
        # Assert that the service received the *correct mock instances*
        self.assertIs(
            self.pagerank_service.pagerank_analyzer_instance, mock_pr_analyzer_instance
        )
        self.assertIs(
            self.pagerank_service.hits_analyzer_instance, mock_hits_analyzer_instance
        )

        # Assert that the analyzer classes were called to create instances with ANY DataFrame
        MockPageRankGraphAnalyzer.assert_called_once_with(ANY)
        MockHITSGraphAnalyzer.assert_called_once_with(ANY, ANY)

        # Assert that load_data was called on the mock loader instances
        self.mock_pagerank_loader_instance.load_data.assert_called_once()
        self.mock_link_graph_loader_instance.load_data.assert_called_once()

    def test_initial_data_load_pagerank_file_not_found(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when PageRank CSV is not found."""
        self._setup_csv_loader_mock(MockCSVLoader, "not_found", "success")

        # Ensure MockHITSGraphAnalyzer returns a mock instance if Link Graph is loaded
        MockHITSGraphAnalyzer.return_value = MagicMock(spec=HITSGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()

        self.assertIn(
            f"⚠️ PageRank CSV not found at {self.mock_config.output_analysis_path}",
            status,
        )
        self.assertIn(
            f"✅ Link graph data loaded from {self.mock_config.input_edge_list_path}",
            status,
        )
        self.assertTrue(self.pagerank_service.full_pagerank_df.empty)
        self.assertFalse(self.pagerank_service.full_link_graph_df.empty)
        self.assertIsNone(self.pagerank_service.pagerank_analyzer_instance)
        self.assertIsNotNone(
            self.pagerank_service.hits_analyzer_instance
        )  # HITS should still be initialized

        MockPageRankGraphAnalyzer.assert_not_called()  # PageRank analyzer should not be initialized
        MockHITSGraphAnalyzer.assert_called_once_with(
            ANY, ANY
        )  # HITS analyzer should be initialized

    def test_initial_data_load_link_graph_file_not_found(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when Link Graph CSV is not found."""
        self._setup_csv_loader_mock(MockCSVLoader, "success", "not_found")

        # Ensure MockPageRankGraphAnalyzer returns a mock instance if PageRank is loaded
        MockPageRankGraphAnalyzer.return_value = MagicMock(spec=PageRankGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()

        self.assertIn(
            f"✅ PageRank data loaded from {self.mock_config.output_analysis_path}",
            status,
        )
        self.assertIn(
            f"⚠️ Link Graph CSV not found at {self.mock_config.input_edge_list_path}. Please run link extraction to generate it.",
            status,
        )  # Full message check
        self.assertFalse(self.pagerank_service.full_pagerank_df.empty)
        self.assertTrue(self.pagerank_service.full_link_graph_df.empty)
        self.assertIsNotNone(self.pagerank_service.pagerank_analyzer_instance)
        self.assertIsNone(self.pagerank_service.hits_analyzer_instance)

        MockPageRankGraphAnalyzer.assert_called_once_with(ANY)
        MockHITSGraphAnalyzer.assert_not_called()

    def test_initial_data_load_pagerank_empty_or_malformed(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when PageRank CSV is empty or malformed."""
        self._setup_csv_loader_mock(MockCSVLoader, "empty", "success")

        MockHITSGraphAnalyzer.return_value = MagicMock(spec=HITSGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()

        self.assertIn("❌ Error loading/parsing PageRank CSV:", status)
        self.assertIn("File might be empty or malformed.", status)
        self.assertIn(
            f"✅ Link graph data loaded from {self.mock_config.input_edge_list_path}",
            status,
        )
        self.assertTrue(self.pagerank_service.full_pagerank_df.empty)
        self.assertFalse(self.pagerank_service.full_link_graph_df.empty)
        self.assertIsNone(self.pagerank_service.pagerank_analyzer_instance)
        self.assertIsNotNone(self.pagerank_service.hits_analyzer_instance)

        MockPageRankGraphAnalyzer.assert_not_called()
        MockHITSGraphAnalyzer.assert_called_once_with(ANY, ANY)

    def test_initial_data_load_link_graph_empty_or_malformed(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when Link Graph CSV is empty or malformed."""
        # For this test, we configure `_setup_csv_loader_mock` to return an empty DataFrame with columns for link_graph
        # This will trigger the service's `if dataframe.empty:` check, leading to the specific warning message.
        self._setup_csv_loader_mock(MockCSVLoader, "success", "empty")

        MockPageRankGraphAnalyzer.return_value = MagicMock(spec=PageRankGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()

        self.assertIn(
            f"✅ PageRank data loaded from {self.mock_config.output_analysis_path}",
            status,
        )
        # Assert the specific warning message about an empty link graph CSV
        self.assertIn(
            f"⚠️ Link graph CSV at {self.mock_config.input_edge_list_path} is empty. HITS analysis will not be possible.",
            status,
        )
        self.assertFalse(self.pagerank_service.full_pagerank_df.empty)
        self.assertTrue(
            self.pagerank_service.full_link_graph_df.empty
        )  # Should be an empty DataFrame
        self.assertIsNotNone(self.pagerank_service.pagerank_analyzer_instance)
        self.assertIsNone(
            self.pagerank_service.hits_analyzer_instance
        )  # HITS analyzer should be None

        MockPageRankGraphAnalyzer.assert_called_once_with(ANY)
        MockHITSGraphAnalyzer.assert_not_called()

    def test_initial_data_load_pagerank_missing_columns(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when PageRank CSV is missing required columns."""
        self._setup_csv_loader_mock(MockCSVLoader, "missing_cols_pagerank", "success")

        MockHITSGraphAnalyzer.return_value = MagicMock(spec=HITSGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()
        self.assertIn("❌ Error loading/parsing PageRank CSV", status)
        self.assertIn("PageRank data missing required columns. Found:", status)
        self.assertTrue(self.pagerank_service.full_pagerank_df.empty)
        self.assertIsNone(self.pagerank_service.pagerank_analyzer_instance)

        MockPageRankGraphAnalyzer.assert_not_called()
        MockHITSGraphAnalyzer.assert_called_once_with(ANY, ANY)

    def test_initial_data_load_link_graph_missing_columns(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test initial data load when Link Graph CSV is missing required columns."""
        self._setup_csv_loader_mock(MockCSVLoader, "success", "missing_cols_link_graph")

        MockPageRankGraphAnalyzer.return_value = MagicMock(spec=PageRankGraphAnalyzer)

        status = self.pagerank_service.initial_data_load()
        self.assertIn("❌ Error loading/parsing Link Graph CSV", status)
        self.assertIn("Link Graph data missing required columns. Found:", status)
        self.assertTrue(self.pagerank_service.full_link_graph_df.empty)
        self.assertIsNone(self.pagerank_service.hits_analyzer_instance)

        MockPageRankGraphAnalyzer.assert_called_once_with(ANY)
        MockHITSGraphAnalyzer.assert_not_called()

    def test_process_and_save_pagerank_success(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test successful processing and saving of PageRank data."""
        # Setup link graph data for processing
        self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()

        mock_pagerank_analyzer_instance = MagicMock()
        mock_pagerank_analyzer_instance.get_all_nodes.return_value = [
            "http://a.com",
            "http://b.com",
            "http://c.com",
        ]
        mock_pagerank_analyzer_instance.calculate_pagerank.return_value = {
            "http://a.com": 0.3,
            "http://b.com": 0.5,
            "http://c.com": 0.2,
        }
        MockPageRankGraphAnalyzer.return_value = mock_pagerank_analyzer_instance  # This mock instance will be returned when PageRankGraphAnalyzer is "called"

        # Ensure the directory exists for saving - patch os.makedirs for actual file write prevention
        with patch("os.makedirs", return_value=None):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                result_message = self.pagerank_service.process_and_save_pagerank()

                self.assertEqual(
                    result_message, "✅ PageRank data processed and saved."
                )
                # Assert that the PageRankGraphAnalyzer constructor was called
                MockPageRankGraphAnalyzer.assert_called_once_with(ANY)
                # Assert that methods on the *returned instance* were called
                mock_pagerank_analyzer_instance.get_all_nodes.assert_called_once()
                mock_pagerank_analyzer_instance.calculate_pagerank.assert_called_once()
                mock_to_csv.assert_called_once_with(
                    self.mock_config.output_analysis_path, index=False
                )

    def test_process_and_save_pagerank_empty_link_graph(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test processing PageRank with an empty link graph."""
        self.pagerank_service.full_link_graph_df = pd.DataFrame(columns=["FROM", "TO"])

        with self.assertRaises(ValueError) as cm:
            self.pagerank_service.process_and_save_pagerank()
        self.assertIn("Link graph data is not loaded or is empty", str(cm.exception))

    def test_process_and_save_pagerank_calculation_error(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test error handling during PageRank calculation."""
        self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()

        mock_pagerank_analyzer_instance = MagicMock()
        mock_pagerank_analyzer_instance.calculate_pagerank.side_effect = RuntimeError(
            "Calculation failed"
        )
        MockPageRankGraphAnalyzer.return_value = (
            mock_pagerank_analyzer_instance  # Ensure this mock is used
        )

        with patch("os.makedirs", return_value=None):
            with patch(
                "pandas.DataFrame.to_csv"
            ):  # Mock to_csv to prevent actual file write
                with self.assertRaises(
                    RuntimeError
                ) as cm:  # Assert that the RuntimeError is re-raised
                    self.pagerank_service.process_and_save_pagerank()
                self.assertIn("Calculation failed", str(cm.exception))

    def test_perform_analysis_pagerank_success(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test successful PageRank analysis."""
        self.pagerank_service.full_pagerank_df = self.pagerank_df.copy()
        # Mock the analyzer instance that the service would have after initial_data_load
        self.pagerank_service.pagerank_analyzer_instance = MagicMock(
            spec=PageRankGraphAnalyzer
        )

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="PageRank", depth_level=1, top_n=1
            )
        )

        self.assertFalse(results_df.empty)
        # Updated assertion message to match service's actual output
        self.assertIn("Top 1 Worst PageRank Candidates at Depth Level 1:", status_msg)
        self.assertEqual(headers, ["URL", "Folder_Depth", "PageRank"])
        self.assertEqual(datatypes, ["str", "number", "number"])
        self.assertEqual(results_df.iloc[0]["URL"], "http://b.com")
        self.assertEqual(results_df.shape, (1, 3))

    def test_perform_analysis_pagerank_no_candidates(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test PageRank analysis when no candidates are found at the specified depth."""
        self.pagerank_service.full_pagerank_df = self.pagerank_df.copy()
        self.pagerank_service.pagerank_analyzer_instance = MagicMock(
            spec=PageRankGraphAnalyzer
        )  # Set to a mock instance

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="PageRank", depth_level=99, top_n=1
            )
        )

        self.assertTrue(results_df.empty)
        self.assertIn("No PageRank candidates found at Depth Level 99", status_msg)
        self.assertEqual(headers, ["URL", "Folder_Depth", "PageRank"])
        self.assertEqual(datatypes, ["str", "number", "number"])
        self.assertEqual(results_df.shape, (0, 3))

    def test_perform_analysis_pagerank_no_analyzer_instance(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test PageRank analysis when the analyzer instance is None."""
        self.pagerank_service.full_pagerank_df = self.pagerank_df.copy()
        self.pagerank_service.pagerank_analyzer_instance = (
            None  # Explicitly set to None
        )

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="PageRank", depth_level=0, top_n=1
            )
        )

        self.assertTrue(results_df.empty)
        self.assertIn(
            "PageRank analysis not possible. Data not loaded or empty.", status_msg
        )
        self.assertEqual(headers, ["URL", "Folder_Depth", "PageRank"])
        self.assertEqual(datatypes, ["str", "number", "number"])

    @patch.object(PageRankService, "initial_data_load")
    def test_perform_analysis_pagerank_empty_dataframe_before_load(
        self,
        mock_initial_data_load,
        MockHITSGraphAnalyzer,
        MockPageRankGraphAnalyzer,
        MockCSVLoader,
    ):
        """Test PageRank analysis when full_pagerank_df is empty, triggering an initial_data_load."""
        # Start with empty dataframes in the service
        self.pagerank_service.full_pagerank_df = pd.DataFrame(
            columns=["URL", "Folder_Depth", "PageRank"]
        )
        self.pagerank_service.full_link_graph_df = pd.DataFrame(columns=["FROM", "TO"])
        self.pagerank_service.pagerank_analyzer_instance = None

        # Configure mock_initial_data_load's side effect: when called, it populates the service's DFs
        def side_effect_initial_load():
            self.pagerank_service.full_pagerank_df = self.pagerank_df.copy()
            self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()
            # Ensure the analyzers are mocked as if successfully initialized by initial_data_load
            mock_pr_analyzer = MagicMock(spec=PageRankGraphAnalyzer)
            mock_hits_analyzer = MagicMock(spec=HITSGraphAnalyzer)
            MockPageRankGraphAnalyzer.return_value = mock_pr_analyzer
            MockHITSGraphAnalyzer.return_value = mock_hits_analyzer
            self.pagerank_service.pagerank_analyzer_instance = mock_pr_analyzer
            self.pagerank_service.hits_analyzer_instance = mock_hits_analyzer

            # Set the return value for calculate_pagerank on THIS specific mock instance
            mock_pr_analyzer.calculate_pagerank.return_value = {
                "http://a.com": 0.3,
                "http://b.com": 0.5,
                "http://c.com": 0.2,
            }
            return "Data reloaded by mock"

        mock_initial_data_load.side_effect = side_effect_initial_load

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="PageRank", depth_level=1, top_n=1
            )
        )
        mock_initial_data_load.assert_called_once()
        self.assertFalse(results_df.empty)
        self.assertIn("Top 1 Worst PageRank Candidates at Depth Level 1:", status_msg)
        self.assertFalse(self.pagerank_service.full_pagerank_df.empty)

    def test_perform_analysis_hits_success(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test successful HITS analysis."""
        self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()
        self.pagerank_service.full_pagerank_df = (
            self.pagerank_df.copy()
        )  # HITS needs pagerank data too

        mock_hits_analyzer_instance = MagicMock(spec=HITSGraphAnalyzer)
        # Mock the return value of calculate_hits_scores
        mock_hits_results_df = pd.DataFrame(
            {
                "URL": ["http://b.com", "http://c.com", "http://a.com"],
                "Folder_Depth": [1, 2, 0],
                "Hub Score": [0.7, 0.5, 0.3],
                "Authority Score": [0.8, 0.6, 0.4],
            }
        )
        mock_hits_analyzer_instance.calculate_hits_scores.return_value = (
            mock_hits_results_df
        )
        self.pagerank_service.hits_analyzer_instance = mock_hits_analyzer_instance

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="HITS",
                depth_level=ANY,
                top_n=2,  # depth_level is ignored for HITS here, use ANY
            )
        )

        self.assertFalse(results_df.empty)
        self.assertIn("Top 2 HITS Authority/Hub Score Candidates:", status_msg)
        self.assertEqual(
            headers, ["URL", "Folder_Depth", "Hub Score", "Authority Score"]
        )
        self.assertEqual(datatypes, ["str", "number", "number", "number"])
        # Ensure sorting by Authority Score (descending) is correct
        self.assertEqual(results_df.iloc[0]["URL"], "http://b.com")
        self.assertEqual(results_df.iloc[1]["URL"], "http://c.com")
        self.assertEqual(results_df.shape, (2, 4))
        mock_hits_analyzer_instance.calculate_hits_scores.assert_called_once()

    def test_perform_analysis_hits_no_analyzer_instance(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test HITS analysis when the analyzer instance is None."""
        self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()
        self.pagerank_service.hits_analyzer_instance = None  # Explicitly set to None

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="HITS", depth_level=0, top_n=1
            )
        )

        self.assertTrue(results_df.empty)
        self.assertIn(
            "HITS analysis not possible. Graph data not loaded or empty.", status_msg
        )
        self.assertEqual(
            headers, ["URL", "Folder_Depth", "Hub Score", "Authority Score"]
        )
        self.assertEqual(datatypes, ["str", "number", "number", "number"])

    @patch.object(PageRankService, "initial_data_load")
    def test_perform_analysis_hits_empty_dataframe_before_load(
        self,
        mock_initial_data_load,
        MockHITSGraphAnalyzer,
        MockPageRankGraphAnalyzer,
        MockCSVLoader,
    ):
        """Test HITS analysis when full_link_graph_df is empty, triggering an initial_data_load."""
        # Start with empty dataframes
        self.pagerank_service.full_link_graph_df = pd.DataFrame(columns=["FROM", "TO"])
        self.pagerank_service.full_pagerank_df = pd.DataFrame(
            columns=["URL", "Folder_Depth", "PageRank"]
        )
        self.pagerank_service.hits_analyzer_instance = (
            None  # Ensure it's not pre-initialized
        )

        # Configure mock_initial_data_load's side effect: when called, it populates the service's DFs
        def side_effect_initial_load_hits():
            self.pagerank_service.full_link_graph_df = self.link_graph_df.copy()
            self.pagerank_service.full_pagerank_df = self.pagerank_df.copy()
            # Ensure the analyzers are mocked as if successfully initialized by initial_data_load
            mock_pr_analyzer = MagicMock(spec=PageRankGraphAnalyzer)
            mock_hits_analyzer = MagicMock(spec=HITSGraphAnalyzer)
            MockPageRankGraphAnalyzer.return_value = mock_pr_analyzer
            MockHITSGraphAnalyzer.return_value = mock_hits_analyzer
            self.pagerank_service.pagerank_analyzer_instance = mock_pr_analyzer
            self.pagerank_service.hits_analyzer_instance = mock_hits_analyzer

            # Set the return value for calculate_hits_scores on THIS specific mock instance
            mock_hits_analyzer.calculate_hits_scores.return_value = pd.DataFrame(
                {
                    "URL": ["http://b.com"],
                    "Folder_Depth": [1],
                    "Hub Score": [0.7],
                    "Authority Score": [0.8],
                }
            )
            return "Data reloaded by mock for HITS"

        mock_initial_data_load.side_effect = side_effect_initial_load_hits

        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="HITS", depth_level=0, top_n=1
            )
        )

        mock_initial_data_load.assert_called_once()
        self.assertFalse(results_df.empty)
        self.assertIn("HITS Authority/Hub Score Candidates:", status_msg)
        self.assertFalse(self.pagerank_service.full_link_graph_df.empty)

    def test_perform_analysis_invalid_type(
        self, MockHITSGraphAnalyzer, MockPageRankGraphAnalyzer, MockCSVLoader
    ):
        """Test analysis with an invalid type."""
        results_df, status_msg, headers, datatypes = (
            self.pagerank_service.perform_analysis(
                analysis_type="Invalid", depth_level=0, top_n=1
            )
        )

        self.assertTrue(results_df.empty)
        self.assertEqual(status_msg, "Invalid analysis type selected.")
        self.assertEqual(headers, ["URL", "Score1", "Score2"])
        self.assertEqual(datatypes, ["str", "number", "number"])


if __name__ == "__main__":
    unittest.main()
