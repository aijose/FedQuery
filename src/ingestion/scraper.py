"""FOMC document scraper for federalreserve.gov.

Supports two modes:
- Web scraping: fetches documents from federalreserve.gov (default)
- Local loading: reconstructs FOMCDocument objects from previously saved
  .txt and .html files on disk, avoiding network requests entirely.
"""

import logging
import re
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.ingestion.cleaner import clean_html_text
from src.models.document import FOMCDocument
from src.models.enums import DocumentType

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"

# URL patterns for FOMC documents
STATEMENT_PATTERN = re.compile(r"/newsevents/pressreleases/monetary\d{8}a\.htm")
MINUTES_PATTERN = re.compile(r"/monetarypolicy/fomcminutes\d{8}\.htm")

# Date extraction from URL: monetary20240131a.htm or fomcminutes20240131.htm
DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")


def build_calendar_url(year: int) -> str:
    """Build the FOMC calendar page URL for a given year.

    The main fomccalendars.htm page covers recent years (2021+).
    Years before 2021 use fomchistorical{year}.htm.
    """
    if year >= 2021:
        return f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
    return f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"


def _extract_date_from_url(url: str) -> date | None:
    """Extract a date from an FOMC document URL."""
    match = DATE_PATTERN.search(url)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def parse_calendar_page(html: str, year: int) -> list[dict]:
    """Parse an FOMC calendar page to extract document URLs.

    Returns a list of dicts with keys: url, type (DocumentType), date.
    """
    soup = BeautifulSoup(html, "html.parser")
    documents = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = href if href.startswith("http") else f"{BASE_URL}{href}"

        if STATEMENT_PATTERN.search(href):
            doc_date = _extract_date_from_url(href)
            if doc_date and doc_date.year == year:
                documents.append({
                    "url": full_url,
                    "type": DocumentType.STATEMENT,
                    "date": doc_date,
                })
        elif MINUTES_PATTERN.search(href):
            doc_date = _extract_date_from_url(href)
            if doc_date and doc_date.year == year:
                documents.append({
                    "url": full_url,
                    "type": DocumentType.MINUTES,
                    "date": doc_date,
                })

    return documents


def fetch_document(
    url: str,
    doc_type: DocumentType,
    doc_date: date,
) -> FOMCDocument | None:
    """Fetch a single FOMC document and return as FOMCDocument.

    Returns None if the fetch fails.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    raw_html = resp.text
    raw_text = clean_html_text(raw_html)
    if not raw_text or not raw_text.strip():
        logger.warning("Empty content after cleaning: %s", url)
        return None

    type_label = "Statement" if doc_type == DocumentType.STATEMENT else "Minutes"
    title = f"FOMC {type_label} - {doc_date.strftime('%B %d, %Y')}"

    return FOMCDocument(
        title=title,
        date=doc_date,
        document_type=doc_type,
        source_url=url,
        raw_text=raw_text,
        raw_html=raw_html,
    )


def scrape_fomc_documents(years: list[int]) -> list[FOMCDocument]:
    """Scrape FOMC documents for the given years.

    Fetches calendar pages, extracts document URLs, downloads and parses
    each document. Returns a list of FOMCDocument objects.
    """
    documents = []

    for year in years:
        calendar_url = build_calendar_url(year)
        logger.info("Fetching calendar for %d: %s", year, calendar_url)

        try:
            resp = requests.get(calendar_url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Failed to fetch calendar for %d: %s", year, e)
            continue

        doc_entries = parse_calendar_page(resp.text, year)
        logger.info("Found %d documents for %d", len(doc_entries), year)

        for entry in doc_entries:
            doc = fetch_document(entry["url"], entry["type"], entry["date"])
            if doc:
                documents.append(doc)
            # Be polite to the server
            time.sleep(0.5)

    return documents


def _build_source_url(doc_type: DocumentType, doc_date: date) -> str:
    """Reconstruct the federalreserve.gov URL from document type and date."""
    date_str = doc_date.strftime("%Y%m%d")
    if doc_type == DocumentType.STATEMENT:
        return f"{BASE_URL}/newsevents/pressreleases/monetary{date_str}a.htm"
    return f"{BASE_URL}/monetarypolicy/fomcminutes{date_str}.htm"


def load_local_documents(
    years: list[int],
    text_dir: Path,
    html_dir: Path,
) -> list[FOMCDocument]:
    """Load FOMCDocument objects from previously saved local text and HTML files.

    Scans text_dir/{year}/ for files matching {type}_{YYYY-MM-DD}.txt,
    pairs them with corresponding HTML files from html_dir, and
    reconstructs FOMCDocument objects without any network requests.

    Returns a list of FOMCDocument objects sorted by date.
    """
    documents = []
    file_pattern = re.compile(r"^(statement|minutes)_(\d{4}-\d{2}-\d{2})\.txt$")

    for year in years:
        year_dir = text_dir / str(year)
        if not year_dir.is_dir():
            logger.warning("No local text directory for %d: %s", year, year_dir)
            continue

        for txt_path in sorted(year_dir.glob("*.txt")):
            match = file_pattern.match(txt_path.name)
            if not match:
                continue

            doc_type_str, date_str = match.groups()
            doc_type = DocumentType(doc_type_str)
            doc_date = date.fromisoformat(date_str)

            raw_text = txt_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                logger.warning("Empty text file, skipping: %s", txt_path)
                continue

            # Load HTML if available
            html_path = html_dir / str(year) / f"{doc_type_str}_{date_str}.html"
            raw_html = ""
            if html_path.exists():
                raw_html = html_path.read_text(encoding="utf-8")

            type_label = "Statement" if doc_type == DocumentType.STATEMENT else "Minutes"
            title = f"FOMC {type_label} - {doc_date.strftime('%B %d, %Y')}"
            source_url = _build_source_url(doc_type, doc_date)

            documents.append(FOMCDocument(
                title=title,
                date=doc_date,
                document_type=doc_type,
                source_url=source_url,
                raw_text=raw_text,
                raw_html=raw_html,
            ))

        logger.info("Loaded %d local documents for %d", sum(1 for d in documents if d.date.year == year), year)

    return documents
