# File: src/backend/data/graph_processor.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from src.shared.interfaces import ILogger


class GraphDataProcessor:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def process(self, node_features_df: pd.DataFrame, edge_list_df: pd.DataFrame):
        self.logger.info(
            "Processing data into tensors with neighbor feature inference..."
        )
        url_to_features = node_features_df.set_index("url").features.to_dict()
        adj = defaultdict(set)
        for _, row in edge_list_df.iterrows():
            adj[row["FROM"]].add(row["TO"])
            adj[row["TO"]].add(row["FROM"])

        imputed_features = {}
        nodes_with_missing_features_count = 0

        valid_features = node_features_df["features"].dropna()
        if valid_features.empty:
            self.logger.warning(
                "No nodes with non-null features found. Assuming feature dimension 384 for imputation."
            )
            feature_dim = (
                384  # Common for 'all-MiniLM-L6-v2', assuming this is consistent
            )
        else:
            feature_dim = len(valid_features.iloc[0])
            self.logger.info(f"Detected feature dimension: {feature_dim}")

        for url, features in url_to_features.items():
            is_missing = pd.isna(features)
            if (isinstance(is_missing, bool) and is_missing) or (
                hasattr(is_missing, "any") and is_missing.any()
            ):
                nodes_with_missing_features_count += 1
                neighbors = adj.get(url, set())
                neighbor_features = [
                    np.array(url_to_features.get(n), dtype=np.float32)
                    for n in neighbors
                    if url_to_features.get(n) is not None
                    and not pd.isna(url_to_features.get(n)).any()
                ]
                if neighbor_features:
                    imputed_features[url] = np.mean(neighbor_features, axis=0)
                else:
                    imputed_features[url] = np.zeros(feature_dim, dtype=np.float32)
            else:
                imputed_features[url] = np.array(features, dtype=np.float32)
        if nodes_with_missing_features_count > 0:
            self.logger.info(
                f"Imputed features for {nodes_with_missing_features_count} nodes."
            )

        self.logger.info("Constructing final PyTorch tensors...")
        node_list = node_features_df["url"].tolist()
        url_to_idx = {url: i for i, url in enumerate(node_list)}
        final_feature_list = [imputed_features[url] for url in node_list]
        x = torch.tensor(np.array(final_feature_list), dtype=torch.float)
        source_indices = [url_to_idx.get(url) for url in edge_list_df["FROM"]]
        dest_indices = [url_to_idx.get(url) for url in edge_list_df["TO"]]
        edge_index = torch.tensor([source_indices, dest_indices], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        self.logger.info(
            f"Created PyG Data object with {data.num_nodes} nodes and {data.num_edges} edges."
        )
        return data, url_to_idx
