# File: src/backend/graph/analyzer.py
import pandas as pd
import networkx as nx

class GraphAnalyzer:
    """
    Handles graph construction and PageRank calculation.
    """
    def __init__(self, edges_df: pd.DataFrame):
        """
        Initializes the GraphAnalyzer with a DataFrame of graph edges.
        """
        if edges_df.shape[1] < 2:
            raise ValueError("Input DataFrame must have at least two columns for source and target URLs.")
        self.edges_df = edges_df
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """
        Builds a directed graph from the provided edges DataFrame.
        """
        G = nx.DiGraph()
        # Get all unique URLs from both columns to ensure all nodes are added
        all_urls = pd.concat([self.edges_df.iloc[:, 0], self.edges_df.iloc[:, 1]]).unique()
        G.add_nodes_from(all_urls)

        for _, row in self.edges_df.iterrows():
            source = row.iloc[0]
            target = row.iloc[1]
            G.add_edge(source, target)
        return G

    def calculate_pagerank(self) -> dict:
        """
        Calculates PageRank for all nodes in the graph.
        Returns a dictionary mapping URLs to their PageRank scores.
        """
        return nx.pagerank(self.graph)

    def get_all_nodes(self) -> list:
        """
        Returns a list of all unique nodes (URLs) in the graph.
        """
        return list(self.graph.nodes())