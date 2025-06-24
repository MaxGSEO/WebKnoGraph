import pandas as pd
from tqdm import tqdm  # For internal progress bar in generate function

from src.backend.config.embeddings_config import EmbeddingConfig
from src.backend.data.embedding_state_manager import EmbeddingStateManager
from src.backend.data.embeddings_loader import DataLoader
from src.backend.data.embeddings_saver import DataSaver
from src.backend.utils.text_processing import TextExtractor
from src.backend.utils.embedding_generation import EmbeddingGenerator
from src.shared.interfaces import ILogger


class EmbeddingPipeline:
    """Orchestrates the entire embedding generation process."""

    def __init__(
        self,
        config: EmbeddingConfig,
        logger: ILogger,
        state_manager: EmbeddingStateManager,
        data_loader: DataLoader,
        text_extractor: TextExtractor,
        embedding_generator: EmbeddingGenerator,
        data_saver: DataSaver,
    ):
        self.config = config
        self.logger = logger
        self.state_manager = state_manager
        self.data_loader = data_loader
        self.text_extractor = text_extractor
        self.embedding_generator = embedding_generator
        self.data_saver = data_saver

    def run(self):
        """A generator that executes the pipeline and yields status updates."""
        try:
            yield "Initializing..."
            processed_urls = self.state_manager.get_processed_urls()

            yield "Loading model and querying data..."
            data_stream = self.data_loader.stream_unprocessed_data(
                processed_urls, self.config.batch_size
            )

            batch_num = 1
            processed_in_this_session = False
            for df_batch in data_stream:
                processed_in_this_session = True
                status_msg = f"Processing Batch {batch_num} ({len(df_batch)} pages)..."
                self.logger.info(status_msg)
                yield status_msg

                # Extract Text
                # Use tqdm here to show progress for text extraction
                df_batch["clean_text"] = [
                    self.text_extractor.extract(html)
                    for html in tqdm(
                        df_batch["Content"], desc="Extracting Text", unit="docs"
                    )
                ]
                df_batch = df_batch[df_batch["clean_text"].str.len() > 100]

                if df_batch.empty:
                    self.logger.info(
                        "Batch had no pages with sufficient text after cleaning."
                    )
                    continue

                # Generate Embeddings
                embeddings = self.embedding_generator.generate(
                    df_batch["clean_text"].tolist()
                )

                # Save Batch
                output_df = pd.DataFrame(
                    {
                        "URL": df_batch["URL"],
                        "Embedding": [e.tolist() for e in embeddings],
                    }
                )
                self.data_saver.save_batch(output_df, batch_num)
                batch_num += 1

            if not processed_in_this_session:
                self.logger.info(
                    "No new pages to process. The dataset is already up to date."
                )
                yield "Already up to date."
            else:
                self.logger.info("All new batches processed successfully.")
                yield "Finished"

        except Exception as e:
            self.logger.exception(f"A critical pipeline error occurred: {e}")
            yield f"Error: {e}"
