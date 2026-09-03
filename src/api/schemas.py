"""
API Data Models using Pydantic

Pydantic provides:
- Automatic data validation
- Type checking at runtime
- JSON serialization/deserialization
- Auto-generated OpenAPI documentation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum


class BoundingBox(BaseModel):
    """Bounding box for detected objects."""
    x1: int = Field(..., ge=0, description="Top-left x")
    y1: int = Field(..., ge=0, description="Top-left y")
    x2: int = Field(..., gt=0, description="Bottom-right x")
    y2: int = Field(..., gt=0, description="Bottom-right y")


class SignDetection(BaseModel):
    """Traffic sign detection result."""
    bbox: BoundingBox
    class_name: str
    class_id: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class LaneResult(BaseModel):
    """Lane detection result."""
    left_lane: Optional[List[float]] = None
    right_lane: Optional[List[float]] = None
    confidence: float
    curvature: Optional[float] = None


class ControlEvent(BaseModel):
    """Control event from decision engine."""
    event_type: str
    priority: int = Field(..., ge=0, le=10)
    parameters: Dict
    timestamp: float


class InferenceRequest(BaseModel):
    """Request for inference."""
    image_base64: str
    vehicle_state: Optional[Dict] = None
    request_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response from inference."""
    request_id: Optional[str] = None
    lane_result: LaneResult
    sign_detections: List[SignDetection]
    control_event: Optional[ControlEvent] = None
    processing_time_ms: float
    timestamp: float
