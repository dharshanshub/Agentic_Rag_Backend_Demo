from datetime import datetime

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata extracted from a single meeting minutes document."""

    filename: str
    meeting_date: datetime | None   # None if date could not be parsed from PDF
    size_bytes: int
    last_modified: datetime
    content_type: str
    chunks_indexed: int             # Number of chunks successfully indexed for this document


class IngestionResult(BaseModel):
    """Response envelope returned by the ingestion endpoint."""

    total: int                      # Number of documents successfully processed
    total_chunks: int               # Total chunks indexed across all documents
    documents: list[DocumentMetadata]
