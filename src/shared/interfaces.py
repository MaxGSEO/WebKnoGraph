# File: src/shared/interfaces.py
from abc import ABC, abstractmethod


class ILogger(ABC):
    """Interface for logging messages."""

    @abstractmethod
    def debug(self, message: str):
        pass

    @abstractmethod
    def info(self, message: str):
        pass

    @abstractmethod
    def warning(self, message: str):
        pass

    @abstractmethod
    def error(self, message: str):
        pass

    @abstractmethod
    def exception(self, message: str):
        pass
