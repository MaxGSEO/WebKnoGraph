# File: src/backend/services/recommendation_engine.py
import os
import torch
import json
import pandas as pd
from urllib.parse import urlparse
from src.shared.interfaces import ILogger
from src.backend.config.link_prediction_config import LinkPredictionConfig
from src.backend.models.graph_models import GraphSAGEModel
from src.backend.utils.url_processing import URLProcessor


class RecommendationEngine:
    """Loads trained artifacts and provides link recommendations using a Top-K strategy."""

    def __init__(
        self, config: LinkPredictionConfig, logger: ILogger, url_processor: URLProcessor
    ):
        self.config = config
        self.logger = logger
        self.url_processor = url_processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.node_embeddings = None
        self.url_to_idx = None
        self.idx_to_url = None
        self.existing_edges = None

    def load_artifacts(self):
        """Loads the trained model, embeddings, and mappings into memory."""
        if self.model is not None:
            return True

        self.logger.info("Loading trained artifacts for recommendations...")
        try:
            with open(self.config.node_mapping_path, "r") as f:
                model_metadata = json.load(f)

            self.url_to_idx = model_metadata["url_to_idx"]
            in_channels = model_metadata["in_channels"]
            hidden_channels = model_metadata["hidden_channels"]
            out_channels = model_metadata["out_channels"]

            self.idx_to_url = {v: k for k, v in self.url_to_idx.items()}

            self.node_embeddings = torch.load(self.config.node_embeddings_path).to(
                self.device
            )
            edge_index = torch.load(self.config.edge_index_path)
            self.existing_edges = set(
                zip(edge_index[0].tolist(), edge_index[1].tolist())
            )

            self.model = GraphSAGEModel(in_channels, hidden_channels, out_channels)
            self.model.load_state_dict(torch.load(self.config.model_state_path))
            self.model.to(self.device)
            self.model.eval()

            self.logger.info("Artifacts loaded successfully.")
            return True
        except FileNotFoundError:
            self.logger.error(
                "Could not find trained model artifacts. Please run the training pipeline first."
            )
            return False
        except Exception as e:
            self.logger.error(f"An error occurred while loading artifacts: {e}")
            raise

    def get_recommendations(
        self,
        source_url: str,
        top_n: int = 20,
        min_folder_depth: int = 0,
        max_folder_depth: int = 10,
    ):
        if not self.load_artifacts():
            return (
                None,
                "Error: Trained model artifacts not found. Please run the training pipeline first.",
            )
        if source_url not in self.url_to_idx:
            return (
                None,
                f"Error: Source URL '{source_url}' not found in the graph's training data.",
            )

        source_idx = self.url_to_idx[source_url]
        num_nodes = len(self.url_to_idx)

        candidate_dest_indices = torch.arange(num_nodes, device=self.device)
        candidate_source_indices = torch.full_like(
            candidate_dest_indices, fill_value=source_idx
        )
        candidate_edge_index = torch.stack(
            [candidate_source_indices, candidate_dest_indices]
        )

        with torch.no_grad():
            scores = self.model.predict_link(self.node_embeddings, candidate_edge_index)

        k = min(num_nodes, top_n + 200)  # Increased buffer
        top_scores, top_indices = torch.topk(scores, k=k)

        recommendations = []
        for i in range(k):
            dest_idx = top_indices[i].item()

            # Defensive check: if for some reason dest_idx is out of bounds, skip
            if dest_idx >= num_nodes:
                self.logger.warning(
                    f"Skipping recommendation candidate with out-of-bounds index: {dest_idx} (max_valid_index={num_nodes - 1})"
                )
                continue

            recommended_url = self.idx_to_url[dest_idx]

            if len(recommendations) >= top_n and i > top_n * 2:
                break

            is_self_link = dest_idx == source_idx
            is_existing_link = (source_idx, dest_idx) in self.existing_edges or (
                dest_idx,
                source_idx,
            ) in self.existing_edges

            if not is_self_link and not is_existing_link:
                folder_depth = self.url_processor.get_folder_depth(recommended_url)
                if min_folder_depth <= folder_depth <= max_folder_depth:
                    recommendations.append(
                        {
                            "RECOMMENDED_URL": recommended_url,
                            "SCORE": torch.sigmoid(top_scores[i]).item(),
                            "FOLDER_DEPTH": folder_depth,
                        }
                    )

        final_recommendations_df = (
            pd.DataFrame(recommendations)
            .sort_values(by="SCORE", ascending=False)
            .head(top_n)
        )

        if final_recommendations_df.empty:
            return (
                None,
                "No recommendations found matching the criteria (filters, existing links, etc.). Try adjusting filters or source URL.",
            )

        return final_recommendations_df, None
