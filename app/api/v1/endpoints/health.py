from fastapi import APIRouter, status
 
from app.core.config import get_settings
 
router = APIRouter()
 
 
@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness probe.
    """
    return {"status": "alive"}
 
 
@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness probe.
    """
    # Ensure configuration can be loaded
    get_settings()
 
    return {"status": "ready"}