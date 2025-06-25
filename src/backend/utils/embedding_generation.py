import numpy as np
from sentence_transformers import SentenceTransformer
from src.shared.interfaces import ILogger
import warnings

# Suppress a common warning from the sentence-transformers library
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)


class EmbeddingGenerator:
    """Generates embeddings for a list of texts."""

    def __init__(self, model_name: str, logger: ILogger):
        self.logger = logger
        self.logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.logger.info("Model loaded successfully.")

    def generate(self, texts: list[str]) -> np.ndarray:
        self.logger.info(f"Generating embeddings for {len(texts)} texts...")
        # tqdm progress bar is handled by the calling pipeline if needed
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
