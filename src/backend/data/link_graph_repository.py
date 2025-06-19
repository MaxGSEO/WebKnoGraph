# File: src/backend/data/link_graph_repository.py
import sqlite3
import os
import fireducks.pandas as pd  # Using fireducks.pandas as specified
from src.shared.interfaces import ILogger


class LinkGraphStateManager:  # Renamed from StateManager
    def __init__(self, db_path: str, edge_list_path: str, logger: ILogger):
        self.db_path = db_path
        self.edge_list_path = edge_list_path
        self.logger = logger
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Ensure directory for CSV also exists
        csv_dir = os.path.dirname(self.edge_list_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        self._execute_query(
            "CREATE TABLE IF NOT EXISTS crawl_frontier (URL TEXT UNIQUE, Redirects INTEGER)"
        )

    def _execute_query(self, query: str, params=None, fetch=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or [])
                if fetch:
                    return cursor.fetchall()
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"LinkGraphStateManager DB error: {e}")
            return None

    def save_frontier(self, frontier_urls_info: list[tuple[str, int]]):
        self._execute_query("DELETE FROM crawl_frontier")
        if not frontier_urls_info:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO crawl_frontier (URL, Redirects) VALUES (?, ?)",
                    frontier_urls_info,
                )
                conn.commit()  # Ensure commit is explicit here
        except sqlite3.Error as e:
            self.logger.error(f"Error saving frontier: {e}")

    def load_frontier(self) -> list[tuple[str, int]]:
        return (
            self._execute_query("SELECT URL, Redirects FROM crawl_frontier", fetch=True)
            or []
        )

    def load_visited_from_edges(self) -> set:
        """Loads all URLs from the existing edge list CSV to rebuild the visited set."""
        visited_urls = set()
        if os.path.exists(self.edge_list_path):
            try:
                # Use low_memory=False to prevent DtypeWarning for mixed types if columns are empty
                edge_df = pd.read_csv(self.edge_list_path, low_memory=False)
                if not edge_df.empty:
                    # Concatenate 'FROM' and 'TO' columns and get unique values
                    all_urls_in_graph = pd.concat(
                        [edge_df["FROM"], edge_df["TO"]]
                    ).unique()
                    visited_urls = set(all_urls_in_graph)
                self.logger.info(
                    f"Rebuilt visited set with {len(visited_urls)} URLs from edge list CSV."
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not rebuild visited set from edge list CSV (may be a new crawl or empty file): {e}"
                )
        return visited_urls

    def append_edges_to_csv(self, df_edges: pd.DataFrame):
        """Appends a DataFrame of edges to the CSV file."""
        write_header = not os.path.exists(self.edge_list_path)
        try:
            df_edges.to_csv(
                self.edge_list_path, mode="a", header=write_header, index=False
            )
            self.logger.info(f"Appended {len(df_edges)} edges to {self.edge_list_path}")
        except Exception as e:
            self.logger.error(f"Error appending edges to CSV: {e}")
