"""
Metrics Collection System

Collects and aggregates performance metrics for monitoring system health.
Metrics are stored in-memory for quick access and can be exported for
visualization tools like Grafana or Prometheus.

Key Metrics Tracked:
- Request latency (mean, percentiles)
- Throughput (requests per second)
- Error rate
- Success rate

Why track metrics?
- Detect performance degradation before users complain
- Identify bottlenecks (which operations are slow?)
- Monitor SLA compliance (are we meeting our <150ms target?)
- Enable data-driven capacity planning
"""

import time
import numpy as np
from collections import deque
from threading import Lock
from typing import Dict, List, Optional
import json


class MetricsCollector:
    """
    Thread-safe metrics collection for production monitoring.
    
    This class maintains a sliding window of recent metrics in memory.
    Thread-safe means multiple threads can call record_request() simultaneously
    without corrupting the data.
    
    Design Pattern: Sliding Window
    - Only keep recent N requests in memory (configurable)
    - Old data automatically drops out
    - Prevents unbounded memory growth
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of recent requests to keep in memory.
                        Larger = more historical data but more memory usage.
                        1000 requests = last ~30 seconds at 30 FPS.
        """
        self.window_size = window_size
        
        # Thread safety: Lock prevents race conditions when multiple threads
        # try to update metrics simultaneously
        self.lock = Lock()
        
        # Metrics storage using deque (double-ended queue)
        # deque with maxlen automatically drops oldest items when full
        # This implements our sliding window efficiently
        self.latencies = deque(maxlen=window_size)      # Request latencies in ms
        self.timestamps = deque(maxlen=window_size)     # When each request occurred
        self.errors = deque(maxlen=window_size)         # 1 = error, 0 = success
        
        # Lifetime counters (never reset)
        self.total_requests = 0    # Total requests since startup
        self.total_errors = 0      # Total errors since startup
        
        # Track when metrics collection started
        self.start_time = time.time()
    
    def record_request(self, latency_ms: float, error: bool = False):
        """
        Record a single request with its latency and success/failure status.
        
        This is the main method called after each inference request.
        Thread-safe: can be called from multiple threads simultaneously.
        
        Args:
            latency_ms: How long the request took in milliseconds
            error: True if request failed, False if successful
            
        Example:
            metrics = MetricsCollector()
            
            start = time.time()
            try:
                result = process_frame(frame)
                metrics.record_request((time.time() - start) * 1000, error=False)
            except Exception:
                metrics.record_request((time.time() - start) * 1000, error=True)
        """
        # Acquire lock to ensure thread safety
        # 'with' automatically releases lock even if exception occurs
        with self.lock:
            # Record the metrics
            self.latencies.append(latency_ms)
            self.timestamps.append(time.time())
            self.errors.append(1 if error else 0)  # Convert boolean to int
            
            # Update lifetime counters
            self.total_requests += 1
            if error:
                self.total_errors += 1
    
    def average_latency(self) -> float:
        """
        Calculate average (mean) latency over the sliding window.
        
        Average is useful for general performance but can be skewed by outliers.
        For SLA monitoring, percentiles (p95, p99) are often more meaningful.
        
        Returns:
            float: Mean latency in milliseconds, or 0.0 if no data
        """
        with self.lock:
            if not self.latencies:
                return 0.0
            return float(np.mean(self.latencies))
    
    def percentile_latency(self, percentile: int) -> float:
        """
        Calculate latency at a specific percentile.
        
        Percentiles are crucial for understanding tail latency:
        - P50 (median): Half of requests are faster, half slower
        - P95: 95% of requests are faster, 5% are slower
        - P99: 99% of requests are faster, 1% are slower
        
        Why P95/P99 matter:
        - Average can hide bad experiences (one 10s request doesn't affect average much)
        - P99 shows worst-case performance that users actually experience
        - SLAs are often written in terms of P95 or P99
        
        Args:
            percentile: Which percentile to calculate (0-100)
                       50 = median, 95 = P95, 99 = P99
                       
        Returns:
            float: Latency at that percentile in milliseconds
            
        Example:
            p95 = metrics.percentile_latency(95)
            print(f"95% of requests complete within {p95:.1f}ms")
        """
        with self.lock:
            if not self.latencies:
                return 0.0
            return float(np.percentile(self.latencies, percentile))
    
    def requests_per_second(self) -> float:
        """
        Calculate current throughput (requests per second).
        
        This measures how many requests the system is handling.
        Useful for:
        - Capacity planning (can we handle 100 RPS?)
        - Detecting traffic spikes
        - Calculating when to scale up workers
        
        Calculation: Count requests in last 10 seconds / 10
        
        Returns:
            float: Requests per second over recent window
        """
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
            
            # Look at last 10 seconds of data
            current_time = time.time()
            recent_timestamps = [ts for ts in self.timestamps if current_time - ts <= 10]
            
            if len(recent_timestamps) < 2:
                return 0.0
            
            # Calculate time span
            time_span = current_time - min(recent_timestamps)
            
            # Requests per second = count / time_span
            return len(recent_timestamps) / time_span if time_span > 0 else 0.0
    
    def error_rate(self) -> float:
        """
        Calculate error rate as a fraction (0.0 to 1.0).
        
        Error rate is a key reliability metric:
        - 0.01 = 1% of requests fail (concerning)
        - 0.001 = 0.1% of requests fail (acceptable for many systems)
        - 0.0 = No errors (ideal but rare in production)
        
        Returns:
            float: Error rate from 0.0 (no errors) to 1.0 (all errors)
            
        Example:
            error_rate = metrics.error_rate()
            if error_rate > 0.05:  # More than 5% errors
                alert_operations_team()
        """
        with self.lock:
            if self.total_requests == 0:
                return 0.0
            return self.total_errors / self.total_requests
    
    def get_summary(self) -> Dict:
        """
        Get complete metrics summary as a dictionary.
        
        This provides a snapshot of all current metrics at once.
        Useful for:
        - Exporting to monitoring dashboards
        - Logging periodic health checks
        - API endpoints that expose metrics
        
        Returns:
            dict: Dictionary containing all current metrics
            
        Example:
            summary = metrics.get_summary()
            print(json.dumps(summary, indent=2))
            
            # Output:
            # {
            #   "total_requests": 1523,
            #   "total_errors": 3,
            #   "error_rate": 0.002,
            #   "average_latency_ms": 87.3,
            #   "p50_latency_ms": 82.1,
            #   "p95_latency_ms": 143.5,
            #   "p99_latency_ms": 189.2,
            #   "requests_per_second": 32.5,
            #   "uptime_seconds": 3847.2
            # }
        """
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': self.error_rate(),
            'average_latency_ms': self.average_latency(),
            'p50_latency_ms': self.percentile_latency(50),
            'p95_latency_ms': self.percentile_latency(95),
            'p99_latency_ms': self.percentile_latency(99),
            'requests_per_second': self.requests_per_second(),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def save_to_file(self, filepath: str):
        """
        Save current metrics to JSON file.
        
        Useful for:
        - Saving metrics before shutdown
        - Periodic backups
        - Historical analysis
        
        Args:
            filepath: Where to save the JSON file
            
        Example:
            metrics.save_to_file('metrics_2024-01-15.json')
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
    
    def reset(self):
        """
        Reset all metrics (useful for testing or after maintenance).
        
        Warning: This clears all historical data!
        In production, you typically don't want to reset metrics.
        """
        with self.lock:
            self.latencies.clear()
            self.timestamps.clear()
            self.errors.clear()
            self.total_requests = 0
            self.total_errors = 0
            self.start_time = time.time()


class LatencyHistogram:
    """
    Track latency distribution in buckets for histogram visualization.
    
    While MetricsCollector tracks individual latencies, this class groups
    them into buckets for histogram charts.
    
    Example buckets: 0-50ms, 50-100ms, 100-150ms, 150-200ms, 200+ms
    This shows the distribution: are most requests fast? Where's the tail?
    """
    
    def __init__(self, buckets: List[float] = None):
        """
        Initialize latency histogram.
        
        Args:
            buckets: Bucket boundaries in milliseconds
                    Default: [50, 100, 150, 200] creates buckets:
                    [0-50ms, 50-100ms, 100-150ms, 150-200ms, 200+ms]
        """
        if buckets is None:
            # Default buckets aligned with common latency SLAs
            buckets = [50, 100, 150, 200]
        
        self.buckets = sorted(buckets)  # Ensure sorted for binary search
        self.counts = [0] * (len(buckets) + 1)  # One extra for overflow bucket
        self.lock = Lock()
    
    def record(self, latency_ms: float):
        """
        Record a latency value in the appropriate bucket.
        
        Args:
            latency_ms: Latency to record
        """
        with self.lock:
            # Find which bucket this latency belongs to
            bucket_idx = 0
            for i, boundary in enumerate(self.buckets):
                if latency_ms > boundary:
                    bucket_idx = i + 1
                else:
                    break
            
            self.counts[bucket_idx] += 1
    
    def get_distribution(self) -> Dict[str, int]:
        """
        Get latency distribution as a dictionary.
        
        Returns:
            dict: Bucket labels mapped to counts
            
        Example:
            {
                '0-50ms': 850,
                '50-100ms': 120,
                '100-150ms': 25,
                '150-200ms': 4,
                '200ms+': 1
            }
        """
        with self.lock:
            result = {}
            
            # First bucket: 0 to first boundary
            result[f'0-{self.buckets[0]}ms'] = self.counts[0]
            
            # Middle buckets
            for i in range(len(self.buckets) - 1):
                label = f'{self.buckets[i]}-{self.buckets[i+1]}ms'
                result[label] = self.counts[i + 1]
            
            # Last bucket: last boundary to infinity
            result[f'{self.buckets[-1]}ms+'] = self.counts[-1]
            
            return result


# Example usage demonstration (commented out for production)
"""
if __name__ == "__main__":
    # Create metrics collector
    metrics = MetricsCollector(window_size=100)
    
    # Simulate some requests
    import random
    for i in range(50):
        # Random latency between 50-200ms
        latency = random.uniform(50, 200)
        
        # 2% chance of error
        error = random.random() < 0.02
        
        metrics.record_request(latency, error)
        time.sleep(0.01)  # Simulate time between requests
    
    # Get metrics summary
    summary = metrics.get_summary()
    print("Metrics Summary:")
    print(json.dumps(summary, indent=2))
    
    # Create latency histogram
    histogram = LatencyHistogram()
    for latency in [45, 87, 123, 156, 89, 234, 67, 145]:
        histogram.record(latency)
    
    print("\nLatency Distribution:")
    print(json.dumps(histogram.get_distribution(), indent=2))
"""
