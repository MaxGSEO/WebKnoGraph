import os
import time
import fireducks.pandas as pd # Using fireducks.pandas as specified
from src.shared.interfaces import ILogger

class DataSaver:
    """Saves a batch of embeddings to a Parquet file."""

    def __init__(self, output_path: str, logger: ILogger):
        self.output_path = output_path
        self.logger = logger

    def save_batch(self, df_batch: pd.DataFrame, batch_num: int):
        """Saves a DataFrame of URLs and embeddings to a uniquely named file."""
        # Ensure output_path exists
        os.makedirs(self.output_path, exist_ok=True)
        batch_filename = f"embeddings_batch_{int(time.time())}_{batch_num}.parquet"
        batch_output_path = os.path.join(self.output_path, batch_filename)
        df_batch.to_parquet(batch_output_path, index=False)
        self.logger.info(f"✅ Saved batch {batch_num} to {batch_filename}")