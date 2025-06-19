import logging
import io
from src.shared.interfaces import ILogger


class GradioLogHandler(logging.StreamHandler):  # Inherit from StreamHandler
    def __init__(self, log_output_stream: io.StringIO):
        super().__init__(log_output_stream)  # Pass stream directly to super
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )


class ConsoleAndGradioLogger(ILogger):
    def __init__(
        self,
        log_output_stream: io.StringIO,
        logger_name: str = "AppLogger",
        level=logging.INFO,
    ):
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(console_handler)

        gradio_handler = GradioLogHandler(
            log_output_stream
        )  # Use the specific Gradio handler
        self._logger.addHandler(gradio_handler)

    def info(self, message: str):
        self._logger.info(message)

    def error(self, message: str):
        self._logger.error(message)

    def exception(self, message: str):
        self._logger.exception(message)

    # Added debug and warning methods to match ILogger interface
    def debug(self, message: str):
        self._logger.debug(message)

    def warning(self, message: str):
        self._logger.warning(message)
