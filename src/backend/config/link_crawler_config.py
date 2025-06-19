# File: src/backend/config/link_crawler_config.py
from dataclasses import dataclass, field

@dataclass
class LinkCrawlerConfig: # Renamed to LinkCrawlerConfig to avoid conflict
    state_db_path: str = "/content/drive/My Drive/WebKnoGraph/data/crawler_state.db"
    edge_list_path: str = "/content/drive/My Drive/WebKnoGraph/data/link_graph_edges.csv"
    min_request_delay: float = 1.0
    max_pages_to_crawl: int = 1000
    save_interval_edges: int = 250
    max_retries_request: int = 3
    max_redirects: int = 2
    request_timeout: int = 15
    initial_start_url: str = 'https://example.com/'
    crawling_scope_path: str = "/" # The "playground" for the crawler. Set to '/' to explore the whole site.
    saving_scope_path: str = "/blog/" # The rule for what gets saved. Only edges within this path will be recorded.
    user_agents: list[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    ])
    base_domain: str = ""