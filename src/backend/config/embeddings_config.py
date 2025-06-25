from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Holds all configuration settings for the pipeline."""

    input_path: str = "/content/drive/My Drive/WebKnoGraph/data/crawled_data_parquet/"
    output_path: str = "/content/drive/My Drive/WebKnoGraph/data/url_embeddings/"
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"  # should be changed to other model if the content is non-English or longer than 8192 tokens
    batch_size: int = 10
