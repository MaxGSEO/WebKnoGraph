# File: src/backend/graph/analyzer.py
import pandas as pd
import networkx as nx

class PageRankGraphAnalyzer:
    """
    Handles graph construction and PageRank calculation.
    """
    def __init__(self, edges_df: pd.DataFrame):
        """
        Initializes the PageRankGraphAnalyzer with a DataFrame of graph edges.
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

class HITSGraphAnalyzer:
    """
    Performs HITS algorithm analysis on a link graph.
    """
    def __init__(self, edges_dataframe: pd.DataFrame, pagerank_dataframe: pd.DataFrame):
        """
        Initializes the HITSGraphAnalyzer with a pandas DataFrame containing 'FROM' and 'TO' columns
        and the pagerank_dataframe to merge Folder_Depth.
        """
        required_edges_columns = ['FROM', 'TO']
        if not all(col in edges_dataframe.columns for col in required_edges_columns):
            raise ValueError(
                f"Link graph data must contain 'FROM' and 'TO' columns. "
                f"Found columns: {edges_dataframe.columns.tolist()}"
            )

        self.graph = nx.DiGraph()
        for _, row in edges_dataframe.iterrows():
            self.graph.add_edge(row['FROM'], row['TO'])

        self.pagerank_dataframe = pagerank_dataframe[['URL', 'Folder_Depth']].copy()
        # --- FIX: Avoid inplace=True for future Pandas versions ---
        self.pagerank_dataframe['Folder_Depth'] = pd.to_numeric(self.pagerank_dataframe['Folder_Depth'], errors='coerce').fillna(-1).astype(int)
        # --- END FIX ---

    def calculate_hits_scores(self) -> pd.DataFrame:
        """
        Calculates Hub and Authority scores using the HITS algorithm and merges Folder_Depth.
        :return: A DataFrame with 'URL', 'Folder_Depth', 'Hub Score', and 'Authority Score'.
                 Returns an empty DataFrame if the graph is empty.
        """
        if self.graph.number_of_nodes() == 0:
            return pd.DataFrame(columns=['URL', 'Folder_Depth', 'Hub Score', 'Authority Score'])

        try:
            hubs, authorities = nx.hits(self.graph)

            hits_data = []
            for node in self.graph.nodes():
                hits_data.append({
                    'URL': node,
                    'Hub Score': hubs.get(node, 0.0),
                    'Authority Score': authorities.get(node, 0.0)
                })

            hits_df = pd.DataFrame(hits_data)

            merged_hits_df = pd.merge(
                hits_df,
                self.pagerank_dataframe,
                on='URL',
                how='left'
            )
            # --- FIX: Avoid inplace=True for future Pandas versions ---
            merged_hits_df['Folder_Depth'] = merged_hits_df['Folder_Depth'].fillna(-1)
            # --- END FIX ---
            merged_hits_df['Folder_Depth'] = merged_hits_df['Folder_Depth'].astype(int)

            merged_hits_df = merged_hits_df.sort_values(by='Authority Score', ascending=False).reset_index(drop=True)

            return merged_hits_df[['URL', 'Folder_Depth', 'Hub Score', 'Authority Score']]
        except Exception as e:
            raise RuntimeError(f"Error during HITS calculation: {e}")
