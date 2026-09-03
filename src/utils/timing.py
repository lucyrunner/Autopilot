"""
Timing Utilities for Performance Analysis

What this teaches:
- How to measure performance bottlenecks
- Context managers for clean timing code
- Production performance monitoring
"""

import time
from typing import Dict, Optional
from contextlib import contextmanager


class StageTimer:
    """
    Context manager for timing code stages.
    
    Usage:
        timer = StageTimer()
        
        with timer.stage("preprocessing"):
            preprocess_image()
        
        with timer.stage("inference"):
            run_model()
        
        print(timer.get_timings())
        # {'preprocessing': 12.3, 'inference': 45.6, 'total': 57.9}
    
    Educational:
    - Learn context managers for clean resource management
    - Understand timing granularity trade-offs
    - See how to measure production performance
    """
    
    def __init__(self):
        """Initialize timer with empty stage dictionary."""
        self.timings: Dict[str, float] = {}
        self.start_time = time.time()
    
    @contextmanager
    def stage(self, name: str):
        """
        Time a specific stage.
        
        Args:
            name: Stage name (e.g., "preprocessing", "inference")
        
        Yields:
            None (context manager)
        """
        stage_start = time.time()
        try:
            yield
        finally:
            stage_duration = (time.time() - stage_start) * 1000  # Convert to ms
            self.timings[name] = stage_duration
    
    def get_timings(self) -> Dict[str, float]:
        """
        Get all stage timings.
        
        Returns:
            Dict mapping stage names to durations (ms)
        """
        total = (time.time() - self.start_time) * 1000
        return {
            **self.timings,
            'total': total
        }
    
    def get_summary(self) -> str:
        """
        Get human-readable timing summary.
        
        Returns:
            Formatted string with timings
        """
        timings = self.get_timings()
        lines = ["⏱️  Performance Breakdown:"]
        for stage, duration in timings.items():
            if stage != 'total':
                percentage = (duration / timings['total']) * 100
                lines.append(f"  • {stage}: {duration:.1f}ms ({percentage:.1f}%)")
        lines.append(f"  📊 Total: {timings['total']:.1f}ms")
        return "\n".join(lines)


def measure_fps(iterations: int = 100):
    """
    Decorator to measure FPS of a function.
    
    Educational: Learn about decorators and performance measurement.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            duration = time.time() - start
            fps = iterations / duration
            print(f"⚡ {func.__name__}: {fps:.1f} FPS (avg {duration/iterations*1000:.1f}ms)")
            return result
        return wrapper
    return decorator
