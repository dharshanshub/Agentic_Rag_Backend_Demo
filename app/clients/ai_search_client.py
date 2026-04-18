from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery

from app.core.config import get_settings
from app.exceptions.api_exceptions import SearchIndexingError, SearchRetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Azure AI Search enforces a maximum of 1000 documents per upload batch.
_UPLOAD_BATCH_SIZE = 1000

# Number of nearest neighbours considered during vector search phase of hybrid query.
# Higher = better recall at the cost of latency.  50 is a sensible production default.
_VECTOR_K_NEAREST = 50


class AISearchClient:
    """
    Async client for Azure AI Search — indexing and retrieval.

    Wraps the azure-search-documents SDK to provide:
      - Upsert semantics (merge_or_upload) so re-processing a document
        updates existing records rather than creating duplicates.
      - Batch upload respecting the 1000-document-per-request limit.
      - Hybrid search (BM25 keyword + vector) scored via Reciprocal Rank Fusion.

    Usage:
        client = AISearchClient()
        await client.upload_documents(docs)
        results = await client.hybrid_search(query_text, query_embedding, top_k=5)
        await client.close()
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._index_name = settings.AZURE_SEARCH_INDEX_NAME

        try:
            self._client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=self._index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
            )
            logger.info(
                "AISearchClient initialised for index '%s'.",
                self._index_name,
            )
        except Exception as exc:
            logger.error("Failed to initialise AISearchClient: %s", exc)
            raise SearchIndexingError(
                f"Could not connect to Azure AI Search: {exc}"
            ) from exc

    async def upload_documents(self, documents: list[dict]) -> None:
        """
        Upsert a list of documents into the configured search index.

        Uses ``merge_or_upload_documents`` so that re-running the ingestion
        pipeline for the same blobs updates existing index entries rather
        than creating duplicates (relies on deterministic document IDs).

        Each batch result is inspected for per-document failures.
        If any document in a batch fails, a ``SearchIndexingError`` is raised
        after logging which document IDs were rejected.

        Args:
            documents: List of dicts whose keys match the index field names.
                       Each dict must include the ``id`` key field.

        Raises:
            SearchIndexingError: If any document fails to index, or if an
                                 Azure-level error occurs during upload.
        """
        if not documents:
            logger.warning("upload_documents called with an empty list — nothing to index.")
            return

        total_batches = -(-len(documents) // _UPLOAD_BATCH_SIZE)  # ceiling division

        for batch_num, batch_start in enumerate(
            range(0, len(documents), _UPLOAD_BATCH_SIZE), start=1
        ):
            batch = documents[batch_start : batch_start + _UPLOAD_BATCH_SIZE]

            try:
                results = await self._client.merge_or_upload_documents(documents=batch)

                # Inspect individual document results — a 207 response can contain
                # per-document failures even when the HTTP call itself succeeds.
                failed = [r for r in results if not r.succeeded]
                if failed:
                    for r in failed:
                        logger.error(
                            "Index rejection | id='%s' | status=%d | reason: %s",
                            r.key,
                            r.status_code,
                            r.error_message,
                        )
                    raise SearchIndexingError(
                        f"{len(failed)} document(s) failed to index in batch "
                        f"{batch_num}/{total_batches} (starting at offset {batch_start})."
                    )

                logger.info(
                    "Indexed batch %d/%d — %d document(s) uploaded to '%s'.",
                    batch_num,
                    total_batches,
                    len(batch),
                    self._index_name,
                )

            except SearchIndexingError:
                raise  # already logged above; propagate as-is

            except AzureError as exc:
                logger.error(
                    "Azure Search error on batch %d/%d: %s",
                    batch_num,
                    total_batches,
                    exc,
                )
                raise SearchIndexingError(
                    f"Azure AI Search upload failed on batch {batch_num}/{total_batches}: {exc}"
                ) from exc

        logger.info(
            "All %d document(s) indexed successfully into '%s'.",
            len(documents),
            self._index_name,
        )

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Execute a hybrid search combining BM25 keyword matching and vector
        similarity, fused via Reciprocal Rank Fusion (RRF).

        How it works:
        - ``search_text`` triggers BM25 across all searchable text fields
          (``chunk``, ``title``).
        - ``vector_queries`` runs approximate nearest-neighbour search over
          ``chunk_vector`` using the HNSW index.
        - Azure AI Search merges both ranked lists using RRF and returns the
          top ``top_k`` results.

        Args:
            query_text:      Raw query string for BM25 keyword matching.
            query_embedding: Dense embedding vector of the query for vector search.
            top_k:           Number of results to return (default 5).

        Returns:
            List of dicts with keys: ``id``, ``title``, ``chunk``, ``Date``.
            Results are ordered by RRF score (highest relevance first).

        Raises:
            SearchRetrievalError: On Azure-level query errors.
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=_VECTOR_K_NEAREST,
                fields="chunk_vector",  # search only the chunk embedding field
            )

            search_results = await self._client.search(
                search_text=query_text,       # BM25 over searchable text fields
                vector_queries=[vector_query],  # vector search over chunk_vector
                select=["id", "title", "chunk", "Date"],  # only retrieve needed fields
                top=top_k,
            )

            # Collect async results into a plain list for downstream use
            retrieved: list[dict] = []
            async for result in search_results:
                retrieved.append(
                    {
                        "id": result["id"],
                        "title": result["title"],
                        "chunk": result["chunk"],
                        "date": result.get("Date") or None,
                    }
                )

            logger.info(
                "Hybrid search returned %d result(s) for query: '%s'",
                len(retrieved),
                query_text[:80],  # truncate long queries in logs
            )
            return retrieved

        except AzureError as exc:
            logger.error("Azure Search hybrid query failed: %s", exc)
            raise SearchRetrievalError(
                f"Hybrid search query failed: {exc}"
            ) from exc

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        await self._client.close()
        logger.debug("AISearchClient closed.")
