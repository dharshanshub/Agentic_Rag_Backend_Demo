import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.clients.ai_search_client import AISearchClient
from app.clients.azure_openai_client import AzureOpenAIClient
from app.clients.blob_client import BlobStorageClient
from app.exceptions.api_exceptions import BlobIngestionError, SearchIndexingError
from app.schemas.ingestion_schema import DocumentMetadata, IngestionResult
from app.utils.logger import get_logger
from app.utils.text_utils import extract_pdf_content

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chunking configuration
# chunk_size=800 chars balances context richness vs. embedding token limits.
# chunk_overlap=100 chars preserves continuity across split boundaries.
# separators follow natural document breaks: paragraph → line → sentence → word.
# ---------------------------------------------------------------------------
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Namespace for deterministic UUID generation.
# Using the same namespace + (filename::chunk_idx) always produces the same ID,
# so re-running the pipeline updates existing index records (upsert semantics).
_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


class IngestionService:
    """
    Orchestrates the full document ingestion pipeline:

      For each blob in Azure Blob Storage:
        1. Download raw PDF bytes.
        2. Extract full text + meeting date in a single PDF pass.
        3. Skip documents with no extractable text.
        4. Chunk the text using RecursiveCharacterTextSplitter.
        5. Embed the document title + all chunks in one batched API call.
        6. Build AI Search documents with deterministic IDs.

      After processing all blobs:
        7. Upload all search documents to Azure AI Search in batches.
        8. Return structured metadata for every successfully processed document.

    Individual document failures (download, extraction, embedding) are logged
    and skipped so the rest of the batch continues.  Pipeline-level failures
    (blob listing, final indexing) raise domain exceptions.
    """

    def __init__(self) -> None:
        self._blob_client = BlobStorageClient()
        self._openai_client = AzureOpenAIClient()
        self._search_client = AISearchClient()

    async def process_documents(self) -> IngestionResult:
        """
        Run the full ingestion pipeline and return a summary result.

        Returns:
            IngestionResult containing per-document metadata and aggregate stats.

        Raises:
            BlobIngestionError: If blob listing fails entirely.
            SearchIndexingError: If the final AI Search upload fails.
        """
        logger.info("Ingestion pipeline started.")

        blob_list = await self._blob_client.list_blobs()
        logger.info("%d blob(s) found in container.", len(blob_list))

        results: list[DocumentMetadata] = []
        all_search_docs: list[dict] = []

        for blob_meta in blob_list:
            try:
                # ----------------------------------------------------------
                # Step 1 — Download PDF
                # ----------------------------------------------------------
                content = await self._blob_client.download_blob(blob_meta.name)

                # ----------------------------------------------------------
                # Step 2 — Extract full text and meeting date (single PDF open)
                # ----------------------------------------------------------
                full_text, meeting_date = extract_pdf_content(content)

                if not full_text.strip():
                    logger.warning(
                        "Skipping '%s' — no extractable text found.", blob_meta.name
                    )
                    continue

                # ----------------------------------------------------------
                # Step 3 — Chunk text
                # ----------------------------------------------------------
                chunks: list[str] = _SPLITTER.split_text(full_text)
                if not chunks:
                    logger.warning(
                        "Skipping '%s' — text splitter produced no chunks.", blob_meta.name
                    )
                    continue

                logger.info(
                    "Split '%s' into %d chunk(s).", blob_meta.name, len(chunks)
                )

                # ----------------------------------------------------------
                # Step 4 — Embed title + all chunks in one batched API call
                # Title is prepended so we only pay for one API round-trip
                # per document regardless of chunk count.
                # ----------------------------------------------------------
                texts_to_embed: list[str] = [blob_meta.name] + chunks
                embeddings: list[list[float]] = await self._openai_client.embed_texts(
                    texts_to_embed
                )

                title_embedding: list[float] = embeddings[0]
                chunk_embeddings: list[list[float]] = embeddings[1:]

                # ----------------------------------------------------------
                # Step 5 — Build AI Search documents
                # ID is deterministic: same blob + chunk index → same UUID.
                # ----------------------------------------------------------
                date_str = (
                    meeting_date.strftime("%Y-%m-%d") if meeting_date else ""
                )

                for chunk_idx, (chunk_text, chunk_emb) in enumerate(
                    zip(chunks, chunk_embeddings)
                ):
                    doc_id = str(
                        uuid.uuid5(_UUID_NAMESPACE, f"{blob_meta.name}::{chunk_idx}")
                    )
                    all_search_docs.append(
                        {
                            "id": doc_id,
                            "title": blob_meta.name,
                            "chunk": chunk_text,
                            "chunk_vector": chunk_emb,
                            "title_vector": title_embedding,
                            "Date": date_str,
                        }
                    )

                # ----------------------------------------------------------
                # Step 6 — Collect document metadata for the API response
                # ----------------------------------------------------------
                results.append(
                    DocumentMetadata(
                        filename=blob_meta.name,
                        meeting_date=meeting_date,
                        size_bytes=blob_meta.size,
                        last_modified=blob_meta.last_modified,
                        content_type=blob_meta.content_type,
                        chunks_indexed=len(chunks),
                    )
                )

                logger.info(
                    "Processed '%s' — meeting_date=%s, chunks=%d.",
                    blob_meta.name,
                    meeting_date.date() if meeting_date else "not found",
                    len(chunks),
                )

            except Exception as exc:
                # Per-document failure: log and continue with remaining blobs.
                # This ensures one bad PDF does not abort the entire pipeline.
                logger.warning(
                    "Skipping '%s' due to error: %s", blob_meta.name, exc
                )

        # ------------------------------------------------------------------
        # Step 7 — Upload all collected search documents to Azure AI Search
        # Done once at the end to minimise round-trips and allow the service
        # to raise a single, clean error if indexing fails.
        # ------------------------------------------------------------------
        if all_search_docs:
            logger.info(
                "Uploading %d search document(s) to AI Search.", len(all_search_docs)
            )
            await self._search_client.upload_documents(all_search_docs)
        else:
            logger.warning("No search documents to upload — pipeline produced no output.")

        total_chunks = sum(doc.chunks_indexed for doc in results)
        logger.info(
            "Pipeline complete — %d document(s) processed, %d chunk(s) indexed.",
            len(results),
            total_chunks,
        )

        return IngestionResult(
            total=len(results),
            total_chunks=total_chunks,
            documents=results,
        )

    async def close(self) -> None:
        """Release all underlying client resources."""
        await self._blob_client.close()
        await self._openai_client.close()
        await self._search_client.close()
