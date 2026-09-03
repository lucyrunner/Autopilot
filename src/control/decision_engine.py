"""
Decision Engine - Rule-Based Control Logic

This module translates perception outputs into control events.
It implements a rule-based system (not ML) because:
- Rules are interpretable and auditable (critical for safety)
- Deterministic behavior (same inputs → same outputs)
- Easy to debug and modify
- Fast execution (no model inference)

Control Flow:
Perception Results → Rule Evaluation → Priority Ranking → Control Event

Event Types:
- KEEP_LANE: Maintain current trajectory
- SLOW_DOWN: Reduce speed (speed limit, hazard ahead)
- FULL_STOP: Come to complete stop (stop sign, red light)
- LANE_DEPARTURE_WARNING: Vehicle drifting out of lane
- TURN_LEFT/RIGHT: Navigate turns
"""

from typing import Dict, List, Optional
from enum import Enum
from src.utils.logger import get_logger


class ControlEventType(str, Enum):
    """
    Enumeration of possible control events.
    
    Using Enum ensures type safety and prevents typos.
    Events are ordered by typical priority (FULL_STOP most critical).
    """
    FULL_STOP = "full_stop"                           # Priority: 10 (critical)
    EMERGENCY_BRAKE = "emergency_brake"                # Priority: 10 (critical)
    LANE_DEPARTURE_WARNING = "lane_departure_warning" # Priority: 8 (warning)
    SLOW_DOWN = "slow_down"                           # Priority: 7 (advisory)
    TURN_LEFT = "turn_left"                           # Priority: 5 (navigation)
    TURN_RIGHT = "turn_right"                         # Priority: 5 (navigation)
    KEEP_LANE = "keep_lane"                           # Priority: 3 (normal)


class BasicControlEngine:
    """
    Rule-based control engine for generating driving commands.
    
    This is a simplified version suitable for highway driving.
    Production systems would include:
    - State machine for complex maneuvers
    - Trajectory planning
    - Integration with vehicle dynamics model
    - Sensor fusion (camera + radar + lidar)
    """
    
    def __init__(self):
        """
        Initialize control engine with safety thresholds.
        """
        # Lane departure threshold
        # If confidence drops below this, warn driver
        self.lane_departure_threshold = 0.3
        
        # Speed adjustment parameters
        # How much to slow down for different scenarios
        self.speed_reduction_map = {
            'speed_limit': 0.0,      # Match exactly (don't exceed)
            'curve': 0.8,            # 80% of speed limit for curves
            'poor_visibility': 0.7,  # 70% for fog/rain
            'construction': 0.6      # 60% for construction zones
        }
        
        # Current state tracking
        self.current_speed_limit = None  # Last detected speed limit
        
        # Logger for debugging control decisions
        self.logger = get_logger(__name__)
    
    def generate_control_event(
        self,
        lane_result: Dict,
        sign_detections: List[Dict],
        vehicle_state: Dict
    ) -> Optional[Dict]:
        """
        Generate control event based on perception and vehicle state.
        
        This is the main decision-making function. It evaluates all rules
        and returns the highest-priority event.
        
        Rule Priority:
        1. Safety-critical (stop sign, obstacle) → Priority 10
        2. Warnings (lane departure) → Priority 8
        3. Speed adjustments → Priority 7
        4. Normal operation → Priority 3
        
        Args:
            lane_result: Output from lane detector, contains:
                - left_lane: Polynomial coefficients or None
                - right_lane: Polynomial coefficients or None
                - confidence: Detection confidence (0.0-1.0)
                - curvature: Road curvature in meters
                
            sign_detections: List of detected signs, each with:
                - class_name: Sign type (e.g., "stop")
                - confidence: Detection confidence
                - bbox: Bounding box coordinates
                
            vehicle_state: Current vehicle information:
                - speed: Current speed (km/h or mph)
                - steering_angle: Current steering angle (degrees)
                - gear: Current gear
                
        Returns:
            Dict: Control event with:
                - event_type: Type of control command
                - priority: Event priority (0-10, higher = more urgent)
                - parameters: Event-specific parameters
                - reason: Human-readable explanation
            Or None if no action needed
            
        Example:
            engine = BasicControlEngine()
            
            event = engine.generate_control_event(
                lane_result={'confidence': 0.8, ...},
                sign_detections=[{'class_name': 'speed_limit_50', ...}],
                vehicle_state={'speed': 70}
            )
            
            if event and event['event_type'] == 'SLOW_DOWN':
                print(f"Reduce speed to {event['parameters']['target_speed']}")
        """
        # Collect all potential events
        events = []
        
        # Rule 1: Check for stop signs (highest priority)
        stop_event = self._check_stop_signs(sign_detections)
        if stop_event:
            events.append(stop_event)
            # Stop sign is critical - return immediately without checking other rules
            # (In practice, you might still check for lane departure as secondary warning)
            return stop_event
        
        # Rule 2: Check for speed limit compliance
        speed_event = self._check_speed_limits(sign_detections, vehicle_state)
        if speed_event:
            events.append(speed_event)
        
        # Rule 3: Check lane keeping status
        lane_event = self._check_lane_keeping(lane_result)
        if lane_event:
            events.append(lane_event)
        
        # Rule 4: Check for curve warnings
        curve_event = self._check_curvature(lane_result, vehicle_state)
        if curve_event:
            events.append(curve_event)
        
        # If no events generated, return None (continue current behavior)
        if not events:
            return None
        
        # Sort by priority and return highest
        # Priority is in descending order (10 = most urgent, 0 = least)
        events.sort(key=lambda e: e['priority'], reverse=True)
        
        highest_priority_event = events[0]
        
        # Log the decision for debugging
        self.logger.info(
            f"Control decision: {highest_priority_event['event_type']}",
            extra={
                'priority': highest_priority_event['priority'],
                'reason': highest_priority_event['reason']
            }
        )
        
        return highest_priority_event
    
    def _check_stop_signs(self, sign_detections: List[Dict]) -> Optional[Dict]:
        """
        Check for stop signs and yield signs.
        
        Stop signs are critical - missing one could cause an accident.
        Therefore, we use:
        - High confidence threshold (>0.8 to reduce false positives)
        - Immediate action (don't wait for multiple confirmations)
        
        Args:
            sign_detections: List of detected signs
            
        Returns:
            Dict: FULL_STOP event or None
        """
        for detection in sign_detections:
            # Check for stop sign or yield sign
            if detection['class_name'] in ['stop', 'yield']:
                # Only trigger if high confidence
                # False positive would cause unnecessary stops (annoying but safe)
                # False negative would miss stop sign (dangerous)
                if detection['confidence'] > 0.8:
                    return {
                        'event_type': ControlEventType.FULL_STOP.value,
                        'priority': 10,  # Maximum priority
                        'parameters': {
                            'sign_type': detection['class_name'],
                            'confidence': detection['confidence']
                        },
                        'reason': f"{detection['class_name']} sign detected"
                    }
        
        return None
    
    def _check_speed_limits(
        self, 
        sign_detections: List[Dict], 
        vehicle_state: Dict
    ) -> Optional[Dict]:
        """
        Check for speed limit signs and compare with current speed.
        
        Speed limit enforcement logic:
        1. Detect speed limit sign
        2. Store as current limit (persists until new sign)
        3. Compare vehicle speed with limit
        4. Generate slow down event if exceeding
        
        Args:
            sign_detections: List of detected signs
            vehicle_state: Current vehicle state
            
        Returns:
            Dict: SLOW_DOWN event or None
        """
        current_speed = vehicle_state.get('speed', 0)
        
        # Check for speed limit signs
        for detection in sign_detections:
            class_name = detection['class_name']
            
            # Speed limit signs have format: "speed_limit_50"
            if 'speed_limit' in class_name and 'end' not in class_name:
                try:
                    # Extract numeric speed value
                    # "speed_limit_50" → 50
                    speed_limit = int(class_name.split('_')[-1])
                    
                    # Update current speed limit
                    self.current_speed_limit = speed_limit
                    
                    self.logger.info(f"Speed limit updated to {speed_limit}")
                    
                except ValueError:
                    # Failed to parse speed limit number
                    self.logger.warning(f"Could not parse speed limit from {class_name}")
                    continue
        
        # Check if we're exceeding current speed limit
        if self.current_speed_limit is not None:
            if current_speed > self.current_speed_limit:
                # Calculate how much to reduce speed
                # Add 5% margin for safety (go slightly below limit)
                target_speed = self.current_speed_limit * 0.95
                
                return {
                    'event_type': ControlEventType.SLOW_DOWN.value,
                    'priority': 7,
                    'parameters': {
                        'current_speed': current_speed,
                        'target_speed': target_speed,
                        'speed_limit': self.current_speed_limit,
                        'excess_speed': current_speed - self.current_speed_limit
                    },
                    'reason': f"Exceeding speed limit ({self.current_speed_limit})"
                }
        
        return None
    
    def _check_lane_keeping(self, lane_result: Dict) -> Optional[Dict]:
        """
        Check if vehicle is maintaining lane position.
        
        Lane departure detection:
        - Low confidence means lanes are not clearly detected
        - This could indicate:
          1. Vehicle is drifting (partially on lane marking)
          2. Poor visibility (fog, rain)
          3. Worn lane markings
          4. Complex road (intersection, construction)
        
        Args:
            lane_result: Lane detection result
            
        Returns:
            Dict: Control event (LANE_DEPARTURE_WARNING or KEEP_LANE) or None
        """
        confidence = lane_result.get('confidence', 0.0)
        
        # Lane departure warning
        if confidence < self.lane_departure_threshold:
            # Low confidence - potential lane departure
            return {
                'event_type': ControlEventType.LANE_DEPARTURE_WARNING.value,
                'priority': 8,
                'parameters': {
                    'lane_confidence': confidence,
                    'threshold': self.lane_departure_threshold
                },
                'reason': f"Low lane confidence ({confidence:.2f})"
            }
        
        # Normal lane keeping
        if confidence > 0.7:
            # High confidence - lanes clearly detected
            # This is the normal driving state
            return {
                'event_type': ControlEventType.KEEP_LANE.value,
                'priority': 3,
                'parameters': {
                    'lane_confidence': confidence,
                    'curvature': lane_result.get('curvature')
                },
                'reason': "Lanes detected, maintaining trajectory"
            }
        
        return None
    
    def _check_curvature(
        self, 
        lane_result: Dict, 
        vehicle_state: Dict
    ) -> Optional[Dict]:
        """
        Check road curvature and suggest speed adjustment for curves.
        
        Physics: Lateral acceleration in curve = v² / r
        Where:
        - v: vehicle speed
        - r: curve radius
        
        Sharp curves (small radius) at high speed create high lateral
        acceleration, which is uncomfortable and can cause loss of control.
        
        Args:
            lane_result: Lane detection result with curvature
            vehicle_state: Current vehicle state
            
        Returns:
            Dict: SLOW_DOWN event if approaching sharp curve, or None
        """
        curvature = lane_result.get('curvature')
        
        if curvature is None:
            return None
        
        current_speed = vehicle_state.get('speed', 0)
        
        # Classify curve sharpness
        # Curvature radius in meters:
        # > 1000m: Straight or gentle curve (no action)
        # 500-1000m: Moderate curve (monitor)
        # 100-500m: Sharp curve (slow down)
        # < 100m: Very sharp curve (slow down significantly)
        
        if curvature < 500:
            # Sharp curve detected
            # Recommend speed reduction based on curve sharpness
            
            if curvature < 100:
                # Very sharp curve - reduce to 60% of current speed
                target_speed = current_speed * 0.6
                urgency = 8
            else:
                # Moderate curve - reduce to 80% of current speed
                target_speed = current_speed * 0.8
                urgency = 6
            
            return {
                'event_type': ControlEventType.SLOW_DOWN.value,
                'priority': urgency,
                'parameters': {
                    'current_speed': current_speed,
                    'target_speed': target_speed,
                    'curve_radius': curvature
                },
                'reason': f"Sharp curve ahead (radius: {curvature:.0f}m)"
            }
        
        return None
