import re
import trafilatura
import warnings

# Suppress a common warning from the sentence-transformers library
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)


class TextExtractor:
    """Extracts clean text from raw HTML."""

    def extract(self, html_content: str) -> str:
        if not html_content or not isinstance(html_content, str):
            return ""
        text = trafilatura.extract(
            html_content, include_comments=False, include_tables=False, deduplicate=True
        )
        if text:
            text = re.sub(r"\n\s*\n", "\n\n", text)
            return text.strip()
        return ""
