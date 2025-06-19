# File: src/backend/utils/strategies.py
from collections import deque
from abc import ABC, abstractmethod
from src.shared.interfaces import ILogger


class VisitedUrlManager:
    def __init__(self):
        self.visited = set()

    def add(self, url: str):
        self.visited.add(url)

    def contains(self, url: str) -> bool:
        return url in self.visited

    def size(self) -> int:
        return len(self.visited)


class CrawlingStrategy(ABC):  # Updated base class
    def __init__(self, visited_manager: VisitedUrlManager, logger: ILogger):
        self.visited = visited_manager
        self.logger = logger

    # Updated: add_links now calls an abstract _add_to_collection
    def add_links(self, links_info: list[tuple[str, int]]):
        new_links = [
            link_info
            for link_info in links_info
            if not self.visited.contains(link_info[0])
        ]
        for link_url, _ in new_links:
            self.visited.add(link_url)
        self._add_to_collection(new_links)  # Calls abstract method

    @abstractmethod
    def _add_to_collection(self, links: list[tuple[str, int]]):  # NEW Abstract Method
        pass

    @abstractmethod
    def get_next(self) -> tuple[str, int]:
        pass

    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def prime_with_frontier(self, frontier_urls_info: list[tuple[str, int]]):
        pass

    @abstractmethod
    def get_queue(self) -> list[tuple[str, int]]:  # NEW Abstract Method
        pass


class BFSCrawlingStrategy(CrawlingStrategy):  # Updated implementation
    def __init__(self, visited_manager: VisitedUrlManager, logger: ILogger):
        super().__init__(visited_manager, logger)
        self.queue = deque()

    # NEW: Implementation for _add_to_collection
    def _add_to_collection(self, links: list[tuple[str, int]]):
        self.queue.extend(links)

    def get_next(self) -> tuple[str, int]:
        return self.queue.popleft()

    def has_next(self) -> bool:
        return len(self.queue) > 0

    def prime_with_frontier(self, frontier_urls_info: list[tuple[str, int]]):
        self.queue.extend(frontier_urls_info)

    # NEW: Implementation for get_queue
    def get_queue(self) -> list[tuple[str, int]]:
        return list(self.queue)


class DFSCrawlingStrategy(CrawlingStrategy):  # Updated implementation
    def __init__(self, visited_manager: VisitedUrlManager, logger: ILogger):
        super().__init__(visited_manager, logger)
        self.stack = []

    # NEW: Implementation for _add_to_collection
    def _add_to_collection(self, links: list[tuple[str, int]]):
        self.stack.extend(links)

    def get_next(self) -> tuple[str, int]:
        return self.stack.pop()

    def has_next(self) -> bool:
        return len(self.stack) > 0

    def prime_with_frontier(self, frontier_urls_info: list[tuple[str, int]]):
        self.stack.extend(frontier_urls_info)

    # NEW: Implementation for get_queue
    def get_queue(self) -> list[tuple[str, int]]:
        return list(self.stack)
