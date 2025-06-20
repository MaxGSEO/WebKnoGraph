# File: src/backend/services/pagerank_service.py
import os
import pandas as pd # Using standard pandas as pd
from src.shared.interfaces import ILogger
from src.backend.config.pagerank_config import PageRankConfig # Using new config
from src.backend.utils.url_processing import URLProcessor # Using centralized URLProcessor

class PageRankProcessor: # Renamed from LinkGraphProcessor for clarity
    """
    Orchestrates the loading of data, analysis, and saving of results.
    """
    def __init__(self, config: PageRankConfig, logger: ILogger, url_processor: URLProcessor, graph_analyzer_class):
        self.config = config
        self.logger = logger
        self.url_processor = url_processor
        self.graph_analyzer_class = graph_analyzer_class # Pass GraphAnalyzer class for DI

    def process_graph_data(self): # Removed input/output file paths, now from config
        self.logger.info(f"Loading data from {self.config.input_edge_list_path}...")
        if not os.path.exists(self.config.input_edge_list_path):
            raise FileNotFoundError(f"Input file not found: {self.config.input_edge_list_path}")

        df_edges = pd.read_csv(self.config.input_edge_list_path)
        self.logger.info("Data loaded successfully.")

        graph_analyzer = self.graph_analyzer_class(df_edges)
        all_urls = graph_analyzer.get_all_nodes()

        self.logger.info("Calculating folder depths...")
        folder_depths = {url: self.url_processor.get_folder_depth(url) for url in all_urls}
        self.logger.info("Folder depths calculated.")

        self.logger.info("Calculating PageRank scores...")
        pagerank_scores = graph_analyzer.calculate_pagerank()
        self.logger.info("PageRank scores calculated.")

        self.logger.info("Compiling results...")
        results_df = pd.DataFrame({
            'URL': all_urls,
            'Folder_Depth': [folder_depths[url] for url in all_urls],
            'PageRank': [pagerank_scores.get(url, 0.0) for url in all_urls]
        })

        self.logger.info(f"Saving results to {self.config.output_analysis_path}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.output_analysis_path), exist_ok=True)
        results_df.to_csv(self.config.output_analysis_path, index=False)
        self.logger.info("Results saved successfully.")
        self.logger.info(f"\nExample of results (first 5 rows):\n{results_df.head().to_markdown(index=False)}") # Use to_markdown for better display in logs
