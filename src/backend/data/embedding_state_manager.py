import os
import duckdb
from src.shared.interfaces import ILogger


class EmbeddingStateManager:
    """Manages the state of the embedding process, enabling resumes."""

    def __init__(self, output_path: str, logger: ILogger):
        self.output_path = output_path
        self.logger = logger

    def get_processed_urls(self) -> set:
        """Scans the output directory to find URLs that have already been embedded."""
        processed_urls = set()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.logger.info("Output directory created.")
            return processed_urls

        try:
            # Ensure output_glob_path uses forward slashes for DuckDB even on Windows
            output_glob_path = os.path.join(self.output_path, "*.parquet").replace(
                os.sep, "/"
            )
            processed_df = duckdb.query(
                f"SELECT DISTINCT URL FROM read_parquet('{output_glob_path}')"
            ).to_df()
            processed_urls = set(processed_df["URL"])
            if processed_urls:
                self.logger.info(
                    f"Found {len(processed_urls)} URLs that have already been processed. They will be skipped."
                )
        except Exception as e:  # Corrected indentation for the except block
            self.logger.info(
                f"No previously processed embeddings found or error reading existing files: {e}. Starting fresh."
            )
        return processed_urls
