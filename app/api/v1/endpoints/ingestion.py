from fastapi import APIRouter, HTTPException, status

from app.exceptions.api_exceptions import (
    BlobIngestionError,
    ContainerNotFoundError,
    EmbeddingError,
    SearchIndexingError,
)
from app.schemas.ingestion_schema import IngestionResult
from app.services.ingestion_service import IngestionService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/process",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Chunk, embed, and index all meeting minutes from Blob Storage",
)
async def process_documents() -> IngestionResult:
    """
    Triggers the full ingestion pipeline:

    1. Lists all PDFs in Azure Blob Storage.
    2. Extracts full text and meeting date from each document.
    3. Chunks text using RecursiveCharacterTextSplitter.
    4. Generates embeddings via Azure OpenAI.
    5. Indexes all chunks into Azure AI Search.

    Returns structured metadata for each successfully processed document,
    including the number of chunks indexed per document and in total.
    """
    service = IngestionService()
    try:
        result = await service.process_documents()
        return result

    except ContainerNotFoundError as exc:
        logger.error("Blob container not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except BlobIngestionError as exc:
        logger.error("Blob ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    except EmbeddingError as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    except SearchIndexingError as exc:
        logger.error("AI Search indexing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )

    finally:
        await service.close()
