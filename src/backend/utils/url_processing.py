# File: src/backend/utils/url_processing.py
from urllib.parse import urlparse

class URLProcessor:
    """
    Handles URL-related operations, specifically calculating folder depth.
    """

    @staticmethod
    def get_folder_depth(url: str) -> int:
        """
        Calculates the folder depth of a given URL.
        Example: https://kalicube.com/learning-spaces/faq-list/generative-ai/ -> 3 (corrected example output)
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        if not path or path == "/":
            return 0
        # Remove leading/trailing slashes and split by '/'
        segments = [s for s in path.strip("/").split("/") if s]
        return len(segments)
