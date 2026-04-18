import io
import re
from datetime import datetime

import pdfplumber
from dateutil import parser as date_parser

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Matches "Wednesday, December 3, 2025" or plain "December 3, 2025"
_DATE_PATTERN = re.compile(
    r'\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+)?'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
    re.IGNORECASE,
)


def extract_pdf_content(pdf_bytes: bytes) -> tuple[str, datetime | None]:
    """
    Open a PDF once and extract both the full document text and the meeting date.

    This is the preferred entry point for the ingestion pipeline — it avoids
    opening the same PDF twice (once for text, once for the date).

    Strategy:
    - Iterates every page and concatenates extracted text.
    - Scans only the first 10 non-empty lines of page 1 for a date pattern
      (e.g. "Wednesday, December 3, 2025").  Meeting dates always appear in
      the document header, so deeper scanning is unnecessary.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        A tuple of:
          - full_text (str): All extracted text joined across pages.
            Empty string if extraction fails or the PDF has no pages.
          - meeting_date (datetime | None): Parsed date, or None if not found.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                logger.warning("PDF has no pages — returning empty content.")
                return "", None

            pages_text: list[str] = []
            meeting_date: datetime | None = None

            for page_idx, page in enumerate(pdf.pages):
                page_text: str = page.extract_text() or ""
                pages_text.append(page_text)

                # Only scan the header area of the first page for the meeting date
                if page_idx == 0:
                    lines = [
                        line.strip()
                        for line in page_text.splitlines()
                        if line.strip()
                    ]
                    for line in lines[:10]:
                        match = _DATE_PATTERN.search(line)
                        if match:
                            try:
                                meeting_date = date_parser.parse(match.group())
                                logger.debug(
                                    "Extracted meeting date '%s' from line: '%s'",
                                    meeting_date.date(),
                                    line,
                                )
                            except Exception as parse_exc:
                                logger.warning(
                                    "Date pattern matched but parsing failed for '%s': %s",
                                    match.group(),
                                    parse_exc,
                                )
                            break  # stop scanning after first date match

            full_text = "\n".join(pages_text).strip()

            if not full_text:
                logger.warning("PDF yielded no extractable text.")

            if meeting_date is None:
                logger.warning("No recognisable date found in the first 10 lines of page 1.")

            return full_text, meeting_date

    except Exception as exc:
        logger.error("Failed to extract content from PDF: %s", exc)
        return "", None


def extract_meeting_date(pdf_bytes: bytes) -> datetime | None:
    """
    Extract only the meeting date from the first page of a PDF.

    Retained for standalone use cases where full-text extraction is not needed.
    For the main ingestion pipeline, prefer ``extract_pdf_content`` which
    opens the PDF once and returns both text and date together.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Parsed datetime if a date is found, None otherwise.
    """
    _, meeting_date = extract_pdf_content(pdf_bytes)
    return meeting_date
