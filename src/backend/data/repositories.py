import sqlite3
import os
from src.shared.interfaces import ILogger


class CrawlStateRepository:  # Renamed from StateManager
    def __init__(self, db_path: str, logger: ILogger):
        self.db_path = db_path
        self.logger = logger
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.ensure_frontier_table_exists()

    def _execute_query(self, query: str, params=None, fetch=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or [])
                if fetch:
                    return cursor.fetchall()
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"CrawlStateRepository DB error: {e}")
            return None

    def ensure_frontier_table_exists(self):
        self._execute_query(
            "CREATE TABLE IF NOT EXISTS crawl_frontier (URL TEXT UNIQUE, Redirects INTEGER)"
        )
        self.logger.info("Crawl frontier table ensured to exist.")

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
                conn.commit()
            self.logger.info(f"Saved {len(frontier_urls_info)} URLs to frontier.")
        except sqlite3.Error as e:
            self.logger.error(f"Error saving frontier: {e}")

    def load_frontier(self) -> list[tuple[str, int]]:
        result = self._execute_query(
            "SELECT URL, Redirects FROM crawl_frontier", fetch=True
        )
        return result or []
