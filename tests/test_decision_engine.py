"""
Test Decision Engine Module

What this teaches:
- Testing rule-based systems
- Priority ordering validation
- Safety-critical logic testing
"""

import pytest
from src.control.decision_engine import BasicControlEngine


class TestDecisionEngine:
    """Test suite for control decision making."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for tests."""
        return BasicControlEngine()
    
    def test_stop_sign_detection(self, engine):
        """
        Test: Stop sign triggers FULL_STOP event.
        
        Educational: Safety-critical rules must be tested thoroughly.
        """
        lane_result = {'confidence': 0.8, 'curvature': 1000}
        sign_detections = [
            {'class_name': 'stop', 'confidence': 0.95}
        ]
        vehicle_state = {'speed': 50}
        
        event = engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        assert event is not None, "Should generate event for stop sign"
        assert event['event_type'] == 'full_stop', "Should be full stop"
        assert event['priority'] == 10, "Stop sign is highest priority"
        
        print("✓ Stop sign correctly triggers FULL_STOP")
    
    def test_speed_limit_enforcement(self, engine):
        """
        Test: Speed limit signs are enforced.
        
        Educational: Stateful rules (speed limit persists) are tricky.
        """
        lane_result = {'confidence': 0.8, 'curvature': 1000}
        
        # First: Detect speed limit sign
        sign_detections = [
            {'class_name': 'speed_limit_50', 'confidence': 0.9}
        ]
        vehicle_state = {'speed': 70}  # Exceeding limit
        
        event = engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        assert event is not None, "Should generate event"
        assert event['event_type'] == 'slow_down', "Should slow down"
        assert event['parameters']['speed_limit'] == 50
        
        print("✓ Speed limit enforcement works")
    
    def test_lane_departure_warning(self, engine):
        """
        Test: Low lane confidence triggers warning.
        
        Educational: Confidence thresholds are key safety parameters.
        """
        lane_result = {'confidence': 0.2, 'curvature': 1000}  # Low confidence!
        sign_detections = []
        vehicle_state = {'speed': 60}
        
        event = engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        assert event is not None, "Should generate event"
        assert 'lane_departure' in event['event_type'].lower()
        assert event['priority'] >= 8, "Lane departure is high priority"
        
        print("✓ Lane departure warning triggers correctly")
    
    def test_priority_ordering(self, engine):
        """
        Test: Higher priority events take precedence.
        
        Educational: Priority systems prevent conflicting commands.
        """
        lane_result = {'confidence': 0.5, 'curvature': 800}
        sign_detections = [
            {'class_name': 'stop', 'confidence': 0.95},  # Priority 10
            {'class_name': 'speed_limit_30', 'confidence': 0.9}  # Priority 7
        ]
        vehicle_state = {'speed': 50}
        
        event = engine.generate_control_event(
            lane_result, sign_detections, vehicle_state
        )
        
        # Stop sign should win due to higher priority
        assert event['event_type'] == 'full_stop'
        assert event['priority'] == 10
        
        print("✓ Priority ordering works correctly")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
