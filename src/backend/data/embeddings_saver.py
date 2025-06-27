# File: src/backend/data/embeddings_saver.py
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from src.shared.interfaces import ILogger


class DataSaver:
    def __init__(self, output_path: str, logger: ILogger):
        self.output_path = output_path
        self.logger = logger
        os.makedirs(self.output_path, exist_ok=True)
        self.output_file = os.path.join(self.output_path, "url_embeddings.parquet")

    def save_embeddings_batch(self, df_batch: pd.DataFrame):
        """Appends a DataFrame of embeddings to the Parquet file."""
        if df_batch.empty:
            self.logger.warning("Attempted to save an empty batch.")
            return

        table = pa.Table.from_pandas(df_batch, preserve_index=False)

        # Check if the file exists to decide whether to write a new file or append
        if not os.path.exists(self.output_file):
            self.logger.info(f"Creating new Parquet file: {self.output_file}")
            pq.write_table(table, self.output_file)
        else:
            # For appending, we read the existing table and concatenate.
            # This is not efficient for very large files but works for batches.
            try:
                existing_table = pq.read_table(self.output_file)
                combined_table = pa.concat_tables([existing_table, table])
                pq.write_table(combined_table, self.output_file)
            except Exception as e:
                self.logger.error(f"Error appending to Parquet file: {e}")
                # Fallback to appending mode if it is supported (e.g. for a specific Parquet writer)
                # This fallback might not work with all writers, hence the preferred method above.
                # Here we just log the error and let it fail to avoid data corruption.
                raise

        self.logger.info(
            f"Appended batch of {len(df_batch)} embeddings to {self.output_file}"
        )
