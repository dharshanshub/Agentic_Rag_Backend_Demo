class AppBaseException(Exception):
    """Base class for all application-specific exceptions."""
    pass


# ---------------------------------------------------------------------------
# Azure Blob Storage exceptions
# ---------------------------------------------------------------------------

class BlobStorageError(AppBaseException):
    """Raised when an Azure Blob Storage operation fails."""
    pass


class BlobNotFoundError(BlobStorageError):
    """Raised when a requested blob does not exist in the container."""
    pass


class ContainerNotFoundError(BlobStorageError):
    """Raised when the configured container does not exist in the storage account."""
    pass


class BlobIngestionError(BlobStorageError):
    """Raised when the ingestion pipeline encounters an unrecoverable error."""
    pass


# ---------------------------------------------------------------------------
# Azure OpenAI exceptions
# ---------------------------------------------------------------------------

class EmbeddingError(AppBaseException):
    """Raised when Azure OpenAI embedding generation fails."""
    pass


# ---------------------------------------------------------------------------
# Azure AI Search exceptions
# ---------------------------------------------------------------------------

class SearchIndexingError(AppBaseException):
    """Raised when Azure AI Search document indexing fails."""
    pass


class SearchRetrievalError(AppBaseException):
    """Raised when Azure AI Search hybrid/vector query fails."""
    pass


# ---------------------------------------------------------------------------
# Azure OpenAI LLM exceptions
# ---------------------------------------------------------------------------

class LLMError(AppBaseException):
    """Raised when Azure OpenAI chat completion fails."""
    pass
