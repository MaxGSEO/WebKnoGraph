# File: src/backend/services/pagerank_service.py
import os
import pandas as pd
import fireducks.pandas as fpd
from src.shared.interfaces import ILogger
from src.backend.config.pagerank_config import PageRankConfig # Ensure this is correct
from src.backend.utils.url_processing import URLProcessor
from src.backend.graph.analyzer import PageRankGraphAnalyzer, HITSGraphAnalyzer

class CSVLoader:
    """
    Handles loading data from a CSV file.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> fpd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found at: {self.file_path}")
        try:
            dataframe = fpd.read_csv(self.file_path)
            if dataframe.empty:
                raise ValueError(f"CSV file at {self.file_path} is empty.")
            return dataframe
        except Exception as e:
            raise IOError(f"Error loading CSV data from {self.file_path}: {e}")

class PageRankService:
    """
    Orchestrates the loading of data, analysis, and saving of results for PageRank and HITS.
    """
    def __init__(self, config: PageRankConfig, logger: ILogger):
        self.config = config # The config object passed during instantiation
        self.logger = logger
        self.url_processor = URLProcessor()
        self.pagerank_analyzer_instance = None
        self.hits_analyzer_instance = None
        self.full_pagerank_df = pd.DataFrame(columns=['URL', 'Folder_Depth', 'PageRank'])
        self.full_link_graph_df = pd.DataFrame(columns=['FROM', 'TO'])

    # IMPORTANT: This method now takes NO explicit path arguments. It uses self.config.
    def initial_data_load(self):
        """
        Attempts to load both PageRank and Link Graph CSVs at application startup.
        Uses paths from self.config. Returns the initial load status messages.
        """
        # --- DEBUG PRINTS INSIDE THE SERVICE METHOD ---
        self.logger.info(f"DEBUG (Service): initial_data_load called. Using pagerank_csv_path from self.config: {self.config.output_analysis_path}")
        self.logger.info(f"DEBUG (Service): initial_data_load called. Using link_graph_csv_path from self.config: {self.config.input_edge_list_path}")
        # --- END DEBUG PRINTS ---

        initial_load_status = []
        self.full_pagerank_df = pd.DataFrame(columns=['URL', 'Folder_Depth', 'PageRank'])
        self.full_link_graph_df = pd.DataFrame(columns=['FROM', 'TO'])

        # --- Load PageRank data (url_analysis_results.csv) ---
        try:
            # Use self.config.output_analysis_path directly
            self.logger.info(f"Attempting to load PageRank data from {self.config.output_analysis_path}")
            pagerank_loader = CSVLoader(self.config.output_analysis_path)
            fpd_pagerank_df = pagerank_loader.load_data()
            temp_pagerank_df = fpd_pagerank_df.to_pandas()

            required_cols_pagerank = ['URL', 'Folder_Depth', 'PageRank']
            if not all(col in temp_pagerank_df.columns for col in required_cols_pagerank):
                raise ValueError(f"PageRank data missing required columns. Found: {temp_pagerank_df.columns.tolist()}")

            self.full_pagerank_df = temp_pagerank_df.copy()
            self.full_pagerank_df['Folder_Depth'] = pd.to_numeric(self.full_pagerank_df['Folder_Depth'], errors='coerce').fillna(-1).astype(int)
            self.full_pagerank_df['PageRank'] = pd.to_numeric(self.full_pagerank_df['PageRank'], errors='coerce')
            self.full_pagerank_df.dropna(subset=['PageRank'], inplace=True)

            self.pagerank_analyzer_instance = PageRankGraphAnalyzer(self.full_pagerank_df.copy())
            initial_load_status.append(f"✅ PageRank data loaded from {self.config.output_analysis_path}")
        except FileNotFoundError:
            initial_load_status.append(f"⚠️ PageRank CSV not found at {self.config.output_analysis_path}. Please run analysis to generate it.")
            self.pagerank_analyzer_instance = None
            self.logger.warning(f"PageRank CSV not found: {self.config.output_analysis_path}")
        except (IOError, ValueError) as e:
            initial_load_status.append(f"❌ Error loading/parsing PageRank CSV: {e}. File might be empty or malformed.")
            self.pagerank_analyzer_instance = None
            self.logger.error(f"PageRank CSV load/parse failed: {e}")
        except Exception as e:
            initial_load_status.append(f"❌ Unexpected error during PageRank CSV load: {e}")
            self.logger.exception(f"Unexpected error during PageRank CSV load: {e}")


        # --- Load Link Graph data (link_graph_edges.csv) for HITS ---
        try:
            # Use self.config.input_edge_list_path directly
            self.logger.info(f"Attempting to load Link Graph data from {self.config.input_edge_list_path}")
            link_graph_loader = CSVLoader(self.config.input_edge_list_path)
            fpd_link_graph_df = link_graph_loader.load_data()
            temp_link_graph_df = fpd_link_graph_df.to_pandas()

            required_cols_link_graph = ['FROM', 'TO']
            if not all(col in temp_link_graph_df.columns for col in required_cols_link_graph):
                raise ValueError(f"Link Graph data missing required columns. Found: {temp_link_graph_df.columns.tolist()}")

            self.full_link_graph_df = temp_link_graph_df.copy()

            if not self.full_link_graph_df.empty:
                self.hits_analyzer_instance = HITSGraphAnalyzer(self.full_link_graph_df.copy(), self.full_pagerank_df.copy())
                initial_load_status.append(f"✅ Link graph data loaded from {self.config.input_edge_list_path}")
            else:
                initial_load_status.append(f"⚠️ Link graph CSV at {self.config.input_edge_list_path} is empty. HITS analysis will not be possible.")
                self.hits_analyzer_instance = None
                self.logger.warning(f"Link graph CSV is empty: {self.config.input_edge_list_path}")

        except FileNotFoundError:
            initial_load_status.append(f"⚠️ Link Graph CSV not found at {self.config.input_edge_list_path}. Please run link extraction to generate it.")
            self.hits_analyzer_instance = None
            self.logger.warning(f"Link Graph CSV not found: {self.config.input_edge_list_path}")
        except (IOError, ValueError) as e:
            initial_load_status.append(f"❌ Error loading/parsing Link Graph CSV: {e}. File might be empty or malformed.")
            self.hits_analyzer_instance = None
            self.logger.error(f"Link Graph CSV load/parse failed: {e}")
        except RuntimeError as e:
            initial_load_status.append(f"❌ Error building graph for HITS: {e}")
            self.hits_analyzer_instance = None
            self.logger.error(f"HITS graph build failed: {e}")
        except Exception as e:
            initial_load_status.append(f"❌ Unexpected error during Link Graph CSV load: {e}")
            self.logger.exception(f"Unexpected error during Link Graph CSV load: {e}")

        return "\n".join(initial_load_status)

    def process_and_save_pagerank(self):
        """Processes and saves PageRank data to CSV."""
        self.logger.info("Starting PageRank data processing and saving...")
        try:
            if self.full_link_graph_df.empty or self.full_link_graph_df.shape[1] < 2:
                raise ValueError("Link graph data is not loaded or is empty for PageRank processing.")

            pagerank_analyzer = PageRankGraphAnalyzer(self.full_link_graph_df.copy())
            all_urls = pagerank_analyzer.get_all_nodes()

            self.logger.info("Calculating folder depths...")
            folder_depths = {url: self.url_processor.get_folder_depth(url) for url in all_urls}
            self.logger.info("Folder depths calculated.")

            self.logger.info("Calculating PageRank scores...")
            pagerank_scores = pagerank_analyzer.calculate_pagerank()
            self.logger.info("PageRank scores calculated.")

            self.logger.info("Compiling results...")
            results_df = pd.DataFrame({
                'URL': all_urls,
                'Folder_Depth': [folder_depths[url] for url in all_urls],
                'PageRank': [pagerank_scores.get(url, 0.0) for url in all_urls]
            })

            self.logger.info(f"Saving results to {self.config.output_analysis_path}...")
            os.makedirs(os.path.dirname(self.config.output_analysis_path), exist_ok=True)
            results_df.to_csv(self.config.output_analysis_path, index=False)
            self.logger.info("Results saved successfully.")
            self.logger.info(f"\nExample of results (first 5 rows):\n{results_df.head().to_markdown(index=False)}")
            return "✅ PageRank data processed and saved."
        except Exception as e:
            self.logger.exception(f"Error during PageRank data processing/saving: {e}")
            raise

    def perform_analysis(self, analysis_type: str, depth_level: int, top_n: int):
        """
        Performs the selected analysis (PageRank filtering or HITS) and returns results.
        Returns a tuple: (pandas.DataFrame, status_message, list_of_headers, list_of_datatypes)
        """
        self.logger.info(f"perform_analysis called with type: {analysis_type}, depth: {depth_level}, top_n: {top_n}")

        status_msg = ""
        results_df = pd.DataFrame()
        new_headers = []
        new_datatype = []

        # Defensive reload of data within perform_analysis if DataFrames are empty
        if self.full_pagerank_df.empty and analysis_type == 'PageRank':
            self.logger.warning("PageRank DataFrame is empty, attempting to re-run initial data load for PageRank analysis.")
            self.initial_data_load() # Now called without arguments, using self.config

        if self.full_link_graph_df.empty and analysis_type == 'HITS':
             self.logger.warning("Link Graph DataFrame is empty, attempting to re-run initial data load for HITS analysis.")
             self.initial_data_load() # Now called without arguments, using self.config


        if analysis_type == 'PageRank':
            if self.pagerank_analyzer_instance is None or self.full_pagerank_df.empty:
                status_msg = "PageRank analysis not possible. Data not loaded or empty."
                self.logger.error("PageRank analyzer is not initialized or data is empty.")
                new_headers = ['URL', 'Folder_Depth', 'PageRank']
                results_df = pd.DataFrame(columns=new_headers)
            else:
                try:
                    filtered_df = self.full_pagerank_df[self.full_pagerank_df['Folder_Depth'] == depth_level].copy()
                    sorted_df = filtered_df.sort_values(by='PageRank', ascending=True)
                    results_df = sorted_df.head(top_n)

                    if results_df.empty:
                        status_msg = f"No PageRank candidates found at Depth Level {depth_level} or after filtering."
                        self.logger.info("No PageRank candidates found for filtering criteria.")
                    else:
                        status_msg = f"Top {len(results_df)} Worst PageRank Candidates at Depth Level {depth_level}:"
                        self.logger.info(f"PageRank analysis successful, {len(results_df)} results.")
                    new_headers = ['URL', 'Folder_Depth', 'PageRank']
                except Exception as e:
                    status_msg = f"An error occurred during PageRank analysis: {e}"
                    self.logger.exception(f"Error during PageRank analysis: {e}")
                    new_headers = ['URL', 'Folder_Depth', 'PageRank']
                    results_df = pd.DataFrame(columns=new_headers)

        elif analysis_type == 'HITS':
            if self.hits_analyzer_instance is None or self.full_link_graph_df.empty:
                status_msg = "HITS analysis not possible. Graph data not loaded or empty."
                self.logger.error("HITS graph analyzer is not initialized or data is empty.")
                new_headers = ['URL', 'Folder_Depth', 'Hub Score', 'Authority Score']
                results_df = pd.DataFrame(columns=new_headers)
            else:
                try:
                    results_df = self.hits_analyzer_instance.calculate_hits_scores()
                    if results_df.empty:
                        status_msg = "No HITS scores calculated (empty graph or no results after calculation)."
                        self.logger.info("No HITS scores calculated.")
                    else:
                        results_df = results_df.sort_values(by='Authority Score', ascending=False).head(top_n).reset_index(drop=True)
                        status_msg = f"Top {len(results_df)} HITS Authority/Hub Score Candidates:"
                        self.logger.info(f"HITS analysis successful, {len(results_df)} results.")
                    new_headers = ['URL', 'Folder_Depth', 'Hub Score', 'Authority Score']
                except Exception as e:
                    status_msg = f"An error occurred during HITS analysis: {e}"
                    self.logger.exception(f"Error during HITS analysis: {e}")
                    new_headers = ['URL', 'Folder_Depth', 'Hub Score', 'Authority Score']
                    results_df = pd.DataFrame(columns=new_headers)
        else:
            status_msg = "Invalid analysis type selected."
            self.logger.error(f"Invalid analysis type: {analysis_type}")
            new_headers = ['URL', 'Score1', 'Score2']
            results_df = pd.DataFrame(columns=new_headers)

        if len(new_headers) == 3:
            new_datatype = ['str', 'number', 'number']
        elif len(new_headers) == 4:
            new_datatype = ['str', 'number', 'number', 'number']
        else:
            new_datatype = ['str', 'number', 'number']

        return results_df, status_msg, new_headers, new_datatype
