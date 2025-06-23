import unittest
from unittest.mock import MagicMock, patch
import pandas as pd  # Changed back to 'import pandas as pd' for consistency with your intended global change
import numpy as np
from datetime import datetime

# Import the class to be tested and its dependencies
from src.backend.services.embeddings_service import EmbeddingPipeline
from src.backend.config.embeddings_config import EmbeddingConfig
from src.backend.data.embedding_state_manager import EmbeddingStateManager
from src.backend.data.embeddings_loader import DataLoader
from src.backend.data.embeddings_saver import DataSaver
from src.backend.utils.text_processing import TextExtractor
from src.backend.utils.embedding_generation import EmbeddingGenerator
from src.shared.interfaces import ILogger


class TestEmbeddingPipeline(unittest.TestCase):
    """
    Unit tests for the EmbeddingPipeline class.
    Mocks all external dependencies to test the pipeline's orchestration logic in isolation.
    """

    def setUp(self):
        """
        Set up mock objects for all dependencies before each test.
        """
        # Initialize MagicMock objects for each dependency
        self.mock_config = MagicMock(spec=EmbeddingConfig)
        self.mock_logger = MagicMock(spec=ILogger)
        self.mock_state_manager = MagicMock(spec=EmbeddingStateManager)
        self.mock_data_loader = MagicMock(spec=DataLoader)
        self.mock_text_extractor = MagicMock(spec=TextExtractor)
        self.mock_embedding_generator = MagicMock(spec=EmbeddingGenerator)
        self.mock_data_saver = MagicMock(spec=DataSaver)

        # Configure default behaviors for mocks
        self.mock_config.batch_size = 2  # Small batch size for easy testing
        self.mock_state_manager.get_processed_urls.return_value = (
            set()
        )  # Assume no URLs processed initially

        # Instantiate the EmbeddingPipeline with the mock dependencies
        self.pipeline = EmbeddingPipeline(
            config=self.mock_config,
            logger=self.mock_logger,
            state_manager=self.mock_state_manager,
            data_loader=self.mock_data_loader,
            text_extractor=self.mock_text_extractor,
            embedding_generator=self.mock_embedding_generator,
            data_saver=self.mock_data_saver,
        )

        # Patch tqdm to prevent it from printing progress bars during tests
        # This makes the tests run silently without interfering with stdout/stderr
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

    def test_init(self):
        """
        Test that the EmbeddingPipeline is initialized correctly with its dependencies.
        Verifies that all injected dependencies are assigned to instance attributes.
        """
        self.assertIsInstance(self.pipeline.config, MagicMock)
        self.assertIsInstance(self.pipeline.logger, MagicMock)
        self.assertIsInstance(self.pipeline.state_manager, MagicMock)
        self.assertIsInstance(self.pipeline.data_loader, MagicMock)
        self.assertIsInstance(self.pipeline.text_extractor, MagicMock)
        self.assertIsInstance(self.pipeline.embedding_generator, MagicMock)
        self.assertIsInstance(self.pipeline.data_saver, MagicMock)

    def test_run_no_new_pages(self):
        """
        Test the run method when there are no new pages to process.
        The data loader should yield an empty stream, and the pipeline should
        report that it's already up to date.
        """
        # Configure data_loader to yield an empty stream (no data to process)
        self.mock_data_loader.stream_unprocessed_data.return_value = []

        # Convert the generator output to a list to check all yielded messages
        status_messages = list(self.pipeline.run())

        # Assertions
        self.mock_logger.info.assert_called_with(
            "No new pages to process. The dataset is already up to date."
        )
        self.assertIn("Initializing...", status_messages)
        self.assertIn("Loading model and querying data...", status_messages)
        self.assertIn("Already up to date.", status_messages)
        self.mock_data_loader.stream_unprocessed_data.assert_called_once_with(
            set(), self.mock_config.batch_size
        )
        # Ensure no processing steps were called
        self.mock_text_extractor.extract.assert_not_called()
        self.mock_embedding_generator.generate.assert_not_called()
        self.mock_data_saver.save_batch.assert_not_called()

    def test_run_with_data_processing(self):
        """
        Test the run method with multiple batches of data to process.
        Verifies that text extraction, embedding generation, and data saving occur as expected.
        """
        # Sample data for two batches
        # These DataFrames will now correctly be pandas.DataFrame objects (assuming you change the service imports)
        sample_df1 = pd.DataFrame(
            {
                "URL": ["http://url1.com", "http://url2.com"],
                "Content": ["<html>long text 1</html>", "<html>long text 2</html>"],
            }
        )
        sample_df2 = pd.DataFrame(
            {"URL": ["http://url3.com"], "Content": ["<html>long text 3</html>"]}
        )

        # Mock the data loader to return these batches
        self.mock_data_loader.stream_unprocessed_data.return_value = [
            sample_df1,
            sample_df2,
        ]

        # Mock text extractor and embedding generator
        # Ensure the returned text is long enough to pass the length filter (> 100 chars)
        self.mock_text_extractor.extract.side_effect = (
            lambda x: f"Cleaned {x} " + "a" * 100
        )
        self.mock_embedding_generator.generate.side_effect = lambda texts: np.array(
            [[0.1, 0.2]] * len(texts)
        )

        # Run the pipeline
        status_messages = list(self.pipeline.run())

        # Assertions for initial steps
        self.assertIn("Initializing...", status_messages)
        self.assertIn("Loading model and querying data...", status_messages)
        self.mock_state_manager.get_processed_urls.assert_called_once()
        self.mock_data_loader.stream_unprocessed_data.assert_called_once_with(
            set(), self.mock_config.batch_size
        )

        # Assertions for text extraction (3 calls: 2 from batch1 + 1 from batch2)
        self.assertEqual(self.mock_text_extractor.extract.call_count, 3)
        # Verify the call to embedding generator for the first batch with sufficiently long text
        self.mock_embedding_generator.generate.assert_any_call(
            [
                f"Cleaned <html>long text 1</html> {'a' * 100}",
                f"Cleaned <html>long text 2</html> {'a' * 100}",
            ]
        )

        # Assertions for data saving
        # Verify that save_batch was called twice (once for each batch)
        self.assertEqual(self.mock_data_saver.save_batch.call_count, 2)

        # Retrieve the actual DataFrames passed to save_batch and compare them
        # Assertions for batch 1 processing
        self.assertIn("Processing Batch 1 (2 pages)...", status_messages)
        actual_df1, actual_batch_num1 = self.mock_data_saver.save_batch.call_args_list[
            0
        ].args
        # Now, pd.DataFrame here will create a standard pandas.DataFrame, ensuring type compatibility
        pd.testing.assert_frame_equal(
            actual_df1,
            pd.DataFrame(
                {
                    "URL": ["http://url1.com", "http://url2.com"],
                    "Embedding": [[0.1, 0.2], [0.1, 0.2]],
                }
            ),
        )
        self.assertEqual(actual_batch_num1, 1)

        # Assertions for batch 2 processing
        self.assertIn("Processing Batch 2 (1 pages)...", status_messages)
        self.mock_embedding_generator.generate.assert_any_call(
            [f"Cleaned <html>long text 3</html> {'a' * 100}"]
        )
        actual_df2, actual_batch_num2 = self.mock_data_saver.save_batch.call_args_list[
            1
        ].args
        # Now, pd.DataFrame here will create a standard pandas.DataFrame, ensuring type compatibility
        pd.testing.assert_frame_equal(
            actual_df2,
            pd.DataFrame({"URL": ["http://url3.com"], "Embedding": [[0.1, 0.2]]}),
        )
        self.assertEqual(actual_batch_num2, 2)

        # Assertions for final state
        self.mock_logger.info.assert_called_with(
            "All new batches processed successfully."
        )
        self.assertIn("Finished", status_messages)

    def test_run_batch_with_insufficient_text(self):
        """
        Test that batches with insufficient extracted text are skipped.
        """
        # Sample data where extracted text is too short (length <= 100)
        # This DataFrame will now correctly be a standard pandas.DataFrame object
        sample_df = pd.DataFrame(
            {
                "URL": ["http://url_short_text.com"],
                "Content": ["<html>short text</html>"],
            }
        )
        self.mock_data_loader.stream_unprocessed_data.return_value = [sample_df]

        # Make the extracted text shorter than 100 characters
        self.mock_text_extractor.extract.return_value = "This text is deliberately short for testing purposes, making it less than 100 characters."

        # Run the pipeline
        status_messages = list(self.pipeline.run())

        # Assertions
        self.assertIn("Processing Batch 1 (1 pages)...", status_messages)
        self.mock_text_extractor.extract.assert_called_once_with(
            "<html>short text</html>"
        )
        # Assert that both the "Batch had no pages..." and "All new batches processed..." messages were logged
        self.mock_logger.info.assert_any_call(
            "Batch had no pages with sufficient text after cleaning."
        )
        self.mock_logger.info.assert_any_call("All new batches processed successfully.")

        self.mock_embedding_generator.generate.assert_not_called()  # Should not generate embeddings
        self.mock_data_saver.save_batch.assert_not_called()  # Should not save
        self.assertIn("Finished", status_messages)

    def test_run_exception_handling(self):
        """
        Test that exceptions during the run method are caught and logged.
        """
        # Simulate an error during text extraction
        # This DataFrame will now correctly be a standard pandas.DataFrame object
        sample_df = pd.DataFrame(
            {"URL": ["http://error.com"], "Content": ["<html>error</html>"]}
        )
        self.mock_data_loader.stream_unprocessed_data.return_value = [sample_df]
        self.mock_text_extractor.extract.side_effect = Exception(
            "Text extraction failed!"
        )

        # Run the pipeline
        status_messages = list(self.pipeline.run())

        # Assertions
        self.mock_logger.exception.assert_called_once_with(
            "A critical pipeline error occurred: Text extraction failed!"
        )
        self.assertIn(
            "Error: Text extraction failed!", status_messages[-1]
        )  # Last yielded message should be the error


if __name__ == "__main__":
    unittest.main()
