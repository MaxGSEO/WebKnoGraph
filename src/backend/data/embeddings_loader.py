import os
import duckdb
import fireducks.pandas as pd  # Using fireducks.pandas as specified
from src.shared.interfaces import ILogger


class DataLoader:
    """Responsible for loading unprocessed data in batches."""

    def __init__(self, input_path: str, logger: ILogger):
        self.input_path = input_path
        self.logger = logger
        self.con = duckdb.connect()

    def stream_unprocessed_data(self, processed_urls: set, batch_size: int):
        """A generator that yields batches of new data to be processed."""
        # Ensure input_glob_path uses forward slashes for DuckDB even on Windows
        input_glob_path = os.path.join(self.input_path, "**", "*.parquet").replace(
            os.sep, "/"
        )
        base_query = f"SELECT URL, Content FROM read_parquet('{input_glob_path}') WHERE Status_Code >= 200 AND Status_Code < 300 AND Content IS NOT NULL AND Content != ''"

        if processed_urls:
            # Create a DuckDB table from the processed URLs DataFrame for the join
            self.con.register(
                "processed_urls_df_table",
                pd.DataFrame(list(processed_urls), columns=["URL"]),
            )

            final_query = f"""
                SELECT t1.URL, t1.Content
                FROM ({base_query}) AS t1
                LEFT JOIN processed_urls_df_table AS t2 ON t1.URL = t2.URL
                WHERE t2.URL IS NULL
            """
        else:
            final_query = base_query

        self.logger.info("Querying for new pages to process...")
        try:
            for batch in self.con.execute(final_query).fetch_record_batch(batch_size):
                yield batch.to_pandas()
        except Exception as e:
            self.logger.error(
                f"Could not query Parquet files. Please check the input path: {e}"
            )
            return
