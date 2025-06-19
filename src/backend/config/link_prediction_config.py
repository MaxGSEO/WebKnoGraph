# File: src/backend/config/link_prediction_config.py
import os
from dataclasses import dataclass


@dataclass
class LinkPredictionConfig:
    """Holds all configuration for the graph prediction pipeline."""

    # Input Paths
    edge_csv_path: str = "/content/drive/My Drive/WebKnoGraph/data/link_graph_edges.csv"
    embeddings_dir_path: str = (
        "/content/drive/My Drive/WebKnoGraph/data/url_embeddings/"
    )

    # Output Artifact Paths
    output_dir: str = "/content/drive/My Drive/WebKnoGraph/data/prediction_model/"
    model_state_path: str = os.path.join(output_dir, "graphsage_link_predictor.pth")
    node_embeddings_path: str = os.path.join(output_dir, "final_node_embeddings.pt")
    node_mapping_path: str = os.path.join(
        output_dir, "model_metadata.json"
    )  # Was node_mapping_path previously
    edge_index_path: str = os.path.join(output_dir, "edge_index.pt")

    # Model Hyperparameters
    hidden_channels: int = 128
    out_channels: int = 64
    learning_rate: float = 0.01
    epochs: int = 100
