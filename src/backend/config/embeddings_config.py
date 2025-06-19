from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Holds all configuration settings for the pipeline."""

    input_path: str = "/content/drive/My Drive/WebKnoGraph/data/crawled_data_parquet/"
    output_path: str = "/content/drive/My Drive/WebKnoGraph/data/url_embeddings/"
    model_name: str = "all-MiniLM-L6-v2"  # should be changed to other/multilingual content if the content is not in English or longer
    batch_size: int = 10
