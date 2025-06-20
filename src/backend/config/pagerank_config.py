# File: src/backend/config/pagerank_config.py
from dataclasses import dataclass


@dataclass
class PageRankConfig:
    """Holds configuration settings for the PageRank analysis pipeline."""

    input_edge_list_path: str = (
        "/content/drive/My Drive/WebKnoGraph/data/link_graph_edges.csv"
    )
    output_analysis_path: str = "/content/drive/My Drive/WebKnoGraph/data/url_analysis_results.csv"  # THIS LINE IS CRITICAL
