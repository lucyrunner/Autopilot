"""
Pytest Configuration

Shared fixtures and test configuration.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_image():
    """Provide a sample image for tests."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_vehicle_state():
    """Provide sample vehicle state."""
    return {
        'speed': 60,
        'steering_angle': 0,
        'gear': 'D'
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
