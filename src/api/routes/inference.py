"""
Inference Routes - Async inference endpoints

Provides both sync and async inference:
- /infer: Synchronous (wait for result)
- /async/submit: Submit to queue (return immediately)
- /async/result: Poll for result
- /async/status: Check request status
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
import base64
import cv2
import numpy as np

from src.api.schemas import InferenceRequest, InferenceResponse
from src.inference.redis_queue import RedisInferenceQueue
from config.settings import get_settings
from src.utils.logger import get_logger


router = APIRouter(prefix="/async", tags=["Async Inference"])
queue = RedisInferenceQueue()
logger = get_logger(__name__)


@router.post("/submit")
async def submit_inference(request: InferenceRequest):
    """
    Submit inference request to queue (non-blocking).
    
    Returns immediately with request ID.
    Client polls /result/{request_id} for completion.
    """
    try:
        # Decode image to validate
        image_bytes = base64.b64decode(request.image_base64)
        
        # Submit to queue
        request_id = queue.submit_request(
            image_data=image_bytes,
            metadata={'vehicle_state': request.vehicle_state}
        )
        
        return {
            'request_id': request_id,
            'status': 'queued',
            'queue_length': queue.queue_length(),
            'message': 'Request submitted successfully'
        }
        
    except Exception as e:
        logger.error(f"Submit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{request_id}")
async def get_result(request_id: str, timeout: int = 30):
    """
    Get inference result by request ID.
    
    Polls Redis until result ready or timeout.
    """
    try:
        result = queue.get_result(request_id, timeout=timeout)
        
        if result is None:
            status = queue.get_status(request_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Request not found")
            else:
                raise HTTPException(status_code=408, detail="Request timeout")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get result error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """Get request status."""
    status = queue.get_status(request_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return {
        'request_id': request_id,
        'status': status,
        'queue_length': queue.queue_length()
    }


@router.get("/queue/stats")
async def queue_stats():
    """Get queue statistics."""
    return {
        'queue_length': queue.queue_length(),
        'message': 'Queue statistics retrieved'
    }
