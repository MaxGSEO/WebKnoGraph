# File: src/backend/utils/url_processing.py
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)  # Changed to INFO so it shows up
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class URLProcessor:
    """
    Handles URL-related operations, specifically calculating folder depth.
    """

    @staticmethod
    def get_folder_depth(url: str) -> int:
        """
        Calculates the folder depth of a given URL.
        Example: https://kalicube.com/learning-spaces/faq-list/generative-ai/ -> 3 (Corrected example output)
        Returns: integer folder depth, or -1 if parsing fails or invalid path.
        """
        try:
            # Check if URL is indeed a string
            if not isinstance(url, str):
                logger.error(
                    f"Input URL is not a string type: {type(url)}. Value: {url}"
                )
                return -1  # Return -1 for non-string input

            parsed_url = urlparse(url)
            path = parsed_url.path

            logger.debug(f"Processing URL path for depth: '{path}' from URL: '{url}'")

            if not path or path == "/":
                logger.debug(
                    f"Path is empty or root for URL: '{url}', returning 0 depth."
                )
                return 0

            segments = [s for s in path.strip("/").split("/") if s]

            logger.debug(
                f"URL: '{url}', Segments: {segments}, Calculated Depth: {len(segments)}"
            )

            return len(segments)
        except Exception:
            logger.exception(
                f"CRITICAL ERROR calculating folder depth for URL: '{url}'. Exception details:"
            )  # Use logger.exception to print full traceback
            return -1  # Return -1 on error, which will be preserved by fillna(-1)
