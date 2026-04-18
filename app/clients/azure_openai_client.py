from openai import AsyncAzureOpenAI
from openai import APIError, APIConnectionError, RateLimitError

from app.core.config import get_settings
from app.exceptions.api_exceptions import EmbeddingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Conservative batch size to stay well within Azure OpenAI rate limits.
# text-embedding-ada-002 supports up to 2048 inputs per request,
# but smaller batches reduce timeout risk on large documents.
_EMBEDDING_BATCH_SIZE = 16


class AzureOpenAIClient:
    """
    Async client for Azure OpenAI embedding operations.

    Wraps the openai SDK to provide:
      - Batch embedding with configurable batch size
      - Structured error mapping to domain exceptions
      - Clean resource teardown

    Usage:
        client = AzureOpenAIClient()
        embeddings = await client.embed_texts(["hello world", "meeting notes"])
        await client.close()
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

        try:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
            )
            logger.info(
                "AzureOpenAIClient initialised with deployment '%s'.",
                self._deployment,
            )
        except Exception as exc:
            logger.error("Failed to initialise AzureOpenAIClient: %s", exc)
            raise EmbeddingError(
                f"Could not initialise Azure OpenAI client: {exc}"
            ) from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Processes in batches of ``_EMBEDDING_BATCH_SIZE`` to respect API
        rate limits. The returned list preserves the input order.

        Args:
            texts: List of plain strings to embed. Empty strings are
                   accepted by the API but produce near-zero vectors —
                   callers should filter blanks before calling this.

        Returns:
            List of embedding vectors (one float list per input text).

        Raises:
            EmbeddingError: On rate-limit, connection, or API errors.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _EMBEDDING_BATCH_SIZE]

            try:
                response = await self._client.embeddings.create(
                    input=batch,
                    model=self._deployment,
                )
                # SDK returns results sorted by index — safe to extend in order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "Embedded batch [%d:%d] — %d vector(s) returned.",
                    batch_start,
                    batch_start + len(batch),
                    len(batch_embeddings),
                )

            except RateLimitError as exc:
                logger.error(
                    "Azure OpenAI rate limit hit at batch [%d:%d]: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                raise EmbeddingError(
                    f"Rate limit exceeded during embedding (batch starting at {batch_start})."
                ) from exc

            except APIConnectionError as exc:
                logger.error(
                    "Azure OpenAI connection error at batch [%d:%d]: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                raise EmbeddingError(
                    f"Connection to Azure OpenAI failed (batch starting at {batch_start})."
                ) from exc

            except APIError as exc:
                logger.error(
                    "Azure OpenAI API error at batch [%d:%d]: %s",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                raise EmbeddingError(
                    f"Azure OpenAI API error (batch starting at {batch_start}): {exc}"
                ) from exc

        logger.info("Embedded %d text(s) across %d batch(es).", len(texts), -(-len(texts) // _EMBEDDING_BATCH_SIZE))
        return all_embeddings

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        await self._client.close()
        logger.debug("AzureOpenAIClient closed.")
