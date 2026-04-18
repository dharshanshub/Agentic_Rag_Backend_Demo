from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient, ContainerClient

from app.core.config import get_settings
from app.exceptions.api_exceptions import (
    BlobIngestionError,
    BlobNotFoundError,
    ContainerNotFoundError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BlobMetadata:
    """Lightweight metadata snapshot for a single blob."""

    name: str
    size: int                   # bytes
    last_modified: datetime
    content_type: str


class BlobStorageClient:
    """
    Async client for Azure Blob Storage.

    Wraps the azure-storage-blob SDK to provide:
      - Listing blobs with rich metadata
      - Downloading a single blob by name
      - Batch-downloading all blobs in the container

    Usage:
        client = BlobStorageClient()
        blobs  = await client.list_blobs()
        data   = await client.download_blob("meeting_notes.pdf")
        await  client.close()
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._container_name: str = settings.AZURE_STORAGE_CONTAINER_NAME
        conn_str: str = settings.AZURE_STORAGE_CONNECTION_STRING

        # Parse account name and key from the connection string so we can
        # generate SAS tokens locally (no network call required).
        parsed = self._parse_connection_string(conn_str)
        self._account_name: str = parsed["AccountName"]
        self._account_key: str = parsed["AccountKey"]

        try:
            # BlobServiceClient is created synchronously; actual I/O is async
            self._service_client: BlobServiceClient = (
                BlobServiceClient.from_connection_string(conn_str)
            )
            self._container_client: ContainerClient = (
                self._service_client.get_container_client(self._container_name)
            )
            logger.info(
                "BlobStorageClient initialised for container '%s'.",
                self._container_name,
            )
        except Exception as exc:
            logger.error("Failed to initialise BlobStorageClient: %s", exc)
            raise BlobIngestionError(
                f"Could not connect to Azure Blob Storage: {exc}"
            ) from exc

    @staticmethod
    def _parse_connection_string(conn_str: str) -> dict[str, str]:
        """
        Extract key=value pairs from an Azure Storage connection string.
        Uses partition('=') so base64 values containing '=' are handled correctly.
        """
        result: dict[str, str] = {}
        for segment in conn_str.split(";"):
            if "=" in segment:
                key, _, value = segment.partition("=")
                result[key.strip()] = value.strip()
        return result

    def generate_sas_url(self, blob_name: str, expiry_hours: int = 1) -> str:
        """
        Generate a time-limited, read-only SAS URL for a single blob.

        The URL grants read access for ``expiry_hours`` hours and can be
        opened directly in a browser to view the PDF.  No network call is
        required — SAS generation is a local cryptographic operation.

        Args:
            blob_name:    Full blob path within the container.
            expiry_hours: How long the URL remains valid (default 1 hour).

        Returns:
            Fully-qualified HTTPS URL with embedded SAS token.
        """
        expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        sas_token: str = generate_blob_sas(
            account_name=self._account_name,
            container_name=self._container_name,
            blob_name=blob_name,
            account_key=self._account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )

        url = (
            f"https://{self._account_name}.blob.core.windows.net"
            f"/{self._container_name}/{blob_name}?{sas_token}"
        )
        logger.debug("Generated SAS URL for '%s' (expires in %dh).", blob_name, expiry_hours)
        return url

    async def list_blobs(self) -> list[BlobMetadata]:
        """
        List all blobs in the configured container.

        Returns:
            Ordered list of BlobMetadata (one per blob found).

        Raises:
            ContainerNotFoundError: If the container does not exist.
            BlobIngestionError: For any other Azure-side failure.
        """
        try:
            metadata_list: list[BlobMetadata] = []

            async for blob in self._container_client.list_blobs():
                metadata_list.append(
                    BlobMetadata(
                        name=blob.name,
                        size=blob.size,
                        last_modified=blob.last_modified,
                        content_type=(
                            blob.content_settings.content_type
                            or "application/octet-stream"
                        ),
                    )
                )

            logger.info(
                "Listed %d blob(s) in container '%s'.",
                len(metadata_list),
                self._container_name,
            )
            return metadata_list

        except ResourceNotFoundError as exc:
            logger.error("Container '%s' not found: %s", self._container_name, exc)
            raise ContainerNotFoundError(
                f"Container '{self._container_name}' does not exist."
            ) from exc
        except AzureError as exc:
            logger.error("Azure error while listing blobs: %s", exc)
            raise BlobIngestionError(f"Failed to list blobs: {exc}") from exc

    async def download_blob(self, blob_name: str) -> bytes:
        """
        Download the raw content of a single blob.

        Args:
            blob_name: Full blob path within the container.

        Returns:
            Raw bytes content of the blob.

        Raises:
            BlobNotFoundError: If the blob does not exist.
            BlobIngestionError: For any other Azure-side failure.
        """
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            downloader = await blob_client.download_blob()
            content: bytes = await downloader.readall()

            logger.info(
                "Downloaded '%s' (%s bytes).", blob_name, f"{len(content):,}"
            )
            return content

        except ResourceNotFoundError as exc:
            logger.error("Blob '%s' not found: %s", blob_name, exc)
            raise BlobNotFoundError(f"Blob '{blob_name}' does not exist.") from exc
        except AzureError as exc:
            logger.error("Azure error downloading '%s': %s", blob_name, exc)
            raise BlobIngestionError(
                f"Failed to download blob '{blob_name}': {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the SDK client and release underlying connections."""
        await self._service_client.close()
        logger.debug("BlobStorageClient closed.")
