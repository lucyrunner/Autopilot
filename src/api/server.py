"""
FastAPI Server - Main Application

Production-ready API server with:
- Async request handling
- CORS middleware
- Health checks
- Metrics collection
- Error handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time
import base64
import numpy as np
import cv2

from config.settings import get_settings
from src.api.schemas import InferenceRequest, InferenceResponse
from src.perception.lane_detector import ProductionLaneDetector
from src.perception.sign_detector import ONNXSignDetector
from src.control.decision_engine import BasicControlEngine
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCollector


# Initialize app
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production autopilot vision system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
lane_detector = None
sign_detector = None
control_engine = None
logger = None
metrics = None
start_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global lane_detector, sign_detector, control_engine, logger, metrics, start_time
    
    logger = get_logger(__name__)
    logger.info("Starting Autopilot Vision API")
    
    metrics = MetricsCollector()
    
    try:
        lane_detector = ProductionLaneDetector(image_shape=(720, 1280))
        sign_detector = ONNXSignDetector(
            model_path=settings.model.sign_model_path,
            conf_threshold=settings.model.sign_confidence_threshold
        )
        control_engine = BasicControlEngine()
        
        logger.info("All models loaded successfully")
        start_time = time.time()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.version,
        "uptime_seconds": time.time() - start_time if start_time else 0
    }


@app.post("/infer")
async def infer_frame(request: InferenceRequest):
    """Run inference on single frame."""
    start = time.time()
    
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Run inference
        lane_result = lane_detector.detect(frame)
        sign_detections = sign_detector.detect(frame)
        
        vehicle_state = request.vehicle_state or {"speed": 0}
        control_event = control_engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        processing_time = (time.time() - start) * 1000
        metrics.record_request(processing_time)
        
        return {
            "request_id": request.request_id,
            "lane_result": lane_result,
            "sign_detections": sign_detections,
            "control_event": control_event,
            "processing_time_ms": processing_time,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from src.utils.timing import StageTimer

@app.post("/infer")
async def infer_frame(request: InferenceRequest):
    """Run inference with per-stage timing."""
    timer = StageTimer()
    
    try:
        # Decode image
        with timer.stage("decoding"):
            image_bytes = base64.b64decode(request.image_base64)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Lane detection
        with timer.stage("lane_detection"):
            lane_result = lane_detector.detect(frame)
        
        # Sign detection
        with timer.stage("sign_detection"):
            sign_detections = sign_detector.detect(frame)
        
        # Control decision
        with timer.stage("control_decision"):
            vehicle_state = request.vehicle_state or {"speed": 0}
            control_event = control_engine.generate_control_event(
                lane_result, sign_detections, vehicle_state
            )
        
        # Get timing info
        timings = timer.get_timings()
        
        return {
            "request_id": request.request_id,
            "lane_result": lane_result,
            "sign_detections": sign_detections,
            "control_event": control_event,
            "stage_timings": timings,  # NEW: Per-stage breakdown
            "processing_time_ms": timings['total'],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
