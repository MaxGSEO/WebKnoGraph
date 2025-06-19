# File: src/backend/data/graph_dataloader.py
import os
import duckdb
from src.shared.interfaces import ILogger


class GraphDataLoader:
    def __init__(
        self, config, logger: ILogger
    ):  # Config type will be LinkPredictionConfig
        self.config = config
        self.logger = logger

    def load_data(self):
        self.logger.info("Loading graph data using DuckDB...")
        try:
            con = duckdb.connect()
            # Ensure paths use forward slashes for DuckDB even on Windows
            edge_csv_path_clean = self.config.edge_csv_path.replace(os.sep, "/")
            embeddings_glob_path_clean = os.path.join(
                self.config.embeddings_dir_path, "*.parquet"
            ).replace(os.sep, "/")

            all_nodes_query = f"""
                (SELECT "FROM" AS url FROM read_csv_auto('{edge_csv_path_clean}', header=true))
                UNION
                (SELECT "TO" AS url FROM read_csv_auto('{edge_csv_path_clean}', header=true))
            """
            node_features_query = f"""
                WITH all_nodes AS ({all_nodes_query})
                SELECT n.url, e.Embedding AS features
                FROM all_nodes AS n
                LEFT JOIN read_parquet('{embeddings_glob_path_clean}') AS e ON n.url = e.URL
            """
            node_features_df = con.execute(
                node_features_query
            ).fetchdf()  # This is already standard pandas df
            edge_list_df = con.execute(
                f"SELECT * FROM read_csv_auto('{edge_csv_path_clean}', header=true)"
            ).fetchdf()  # This is already standard pandas df
            self.logger.info(
                f"Loaded {len(edge_list_df)} edges and {len(node_features_df)} unique nodes."
            )
            return node_features_df, edge_list_df
        except Exception as e:
            self.logger.error(f"Failed to load data for graph: {e}")
            raise
