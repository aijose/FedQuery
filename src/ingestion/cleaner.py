"""Text cleaning for FOMC HTML documents."""

import re

from bs4 import BeautifulSoup


def clean_html_text(html: str) -> str:
    """Extract and clean text from an FOMC document HTML page.

    Extracts text from the main article div, strips HTML artifacts,
    normalizes whitespace, and gracefully handles tables and
    non-textual content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # FOMC documents typically use id="article" for the main content
    article = soup.find("div", id="article")
    if not article:
        # Fallback: try common content containers
        article = soup.find("div", class_="col-xs-12") or soup.body
        if not article:
            return ""

    # Remove script and style elements
    for tag in article.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Handle tables: extract text but mark them
    for table in article.find_all("table"):
        table_text = table.get_text(separator=" | ", strip=True)
        if table_text:
            table.replace_with(f"\n[Table: {table_text[:200]}]\n")
        else:
            table.decompose()

    # Extract text with paragraph separation
    text = article.get_text(separator="\n", strip=False)

    # Normalize whitespace
    text = _normalize_whitespace(text)

    return text.strip()


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Replace multiple blank lines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Clean up lines that are just whitespace
    text = re.sub(r"\n +\n", "\n\n", text)
    return text
