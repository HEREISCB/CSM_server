"""
Prometheus Metrics for CSM TTS Server

Provides detailed metrics for monitoring TTS performance.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    first_chunk_time: Optional[float] = None
    total_chunks: int = 0
    total_audio_bytes: int = 0
    text_length: int = 0
    status: str = "pending"
    error: Optional[str] = None


class MetricsCollector:
    """
    Thread-safe metrics collector for TTS server.
    
    Provides Prometheus-compatible metrics export.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: Dict[str, RequestMetrics] = {}
        self._completed_requests: List[RequestMetrics] = []
        self._max_history = 1000  # Keep last N completed requests
        
        # Counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_audio_seconds = 0.0
        self.total_text_chars = 0
        
        # Histograms (buckets)
        self.ttfb_histogram: List[float] = []
        self.latency_histogram: List[float] = []
        self.rtf_histogram: List[float] = []
        
        self.start_time = time.time()
    
    def start_request(self, request_id: str, text_length: int) -> None:
        """Record the start of a request."""
        with self._lock:
            self._requests[request_id] = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                text_length=text_length,
            )
            self.total_requests += 1
            self.total_text_chars += text_length
    
    def record_first_chunk(self, request_id: str) -> None:
        """Record when the first audio chunk is generated."""
        with self._lock:
            if request_id in self._requests:
                req = self._requests[request_id]
                req.first_chunk_time = time.time()
                ttfb = req.first_chunk_time - req.start_time
                self.ttfb_histogram.append(ttfb)
    
    def record_chunk(self, request_id: str, chunk_bytes: int) -> None:
        """Record an audio chunk."""
        with self._lock:
            if request_id in self._requests:
                req = self._requests[request_id]
                req.total_chunks += 1
                req.total_audio_bytes += chunk_bytes
    
    def complete_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Record request completion."""
        with self._lock:
            if request_id not in self._requests:
                return
            
            req = self._requests[request_id]
            req.end_time = time.time()
            req.status = "success" if success else "error"
            req.error = error
            
            if success:
                self.successful_requests += 1
                
                # Calculate metrics
                latency = req.end_time - req.start_time
                self.latency_histogram.append(latency)
                
                # Estimate audio duration (24kHz, 16-bit mono)
                audio_seconds = req.total_audio_bytes / (24000 * 2)
                self.total_audio_seconds += audio_seconds
                
                # Calculate RTF if we have valid data
                if audio_seconds > 0:
                    rtf = latency / audio_seconds
                    self.rtf_histogram.append(rtf)
            else:
                self.failed_requests += 1
            
            # Move to completed
            self._completed_requests.append(req)
            del self._requests[request_id]
            
            # Trim history
            if len(self._completed_requests) > self._max_history:
                self._completed_requests = self._completed_requests[-self._max_history:]
    
    def get_active_requests(self) -> int:
        """Get number of currently active requests."""
        with self._lock:
            return len(self._requests)
    
    def get_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        with self._lock:
            lines = []
            
            # Request counters
            lines.append("# HELP tts_requests_total Total number of TTS requests")
            lines.append("# TYPE tts_requests_total counter")
            lines.append(f'tts_requests_total {self.total_requests}')
            
            lines.append("# HELP tts_requests_success_total Successful TTS requests")
            lines.append("# TYPE tts_requests_success_total counter")
            lines.append(f'tts_requests_success_total {self.successful_requests}')
            
            lines.append("# HELP tts_requests_failed_total Failed TTS requests")
            lines.append("# TYPE tts_requests_failed_total counter")
            lines.append(f'tts_requests_failed_total {self.failed_requests}')
            
            # Active requests gauge
            lines.append("# HELP tts_requests_active Current active requests")
            lines.append("# TYPE tts_requests_active gauge")
            lines.append(f'tts_requests_active {len(self._requests)}')
            
            # Audio generated
            lines.append("# HELP tts_audio_seconds_total Total audio seconds generated")
            lines.append("# TYPE tts_audio_seconds_total counter")
            lines.append(f'tts_audio_seconds_total {self.total_audio_seconds:.2f}')
            
            # Text processed
            lines.append("# HELP tts_text_chars_total Total text characters processed")
            lines.append("# TYPE tts_text_chars_total counter")
            lines.append(f'tts_text_chars_total {self.total_text_chars}')
            
            # TTFB percentiles
            if self.ttfb_histogram:
                lines.append("# HELP tts_ttfb_seconds Time to first byte")
                lines.append("# TYPE tts_ttfb_seconds summary")
                lines.append(f'tts_ttfb_seconds{{quantile="0.5"}} {self.get_percentile(self.ttfb_histogram, 50):.4f}')
                lines.append(f'tts_ttfb_seconds{{quantile="0.9"}} {self.get_percentile(self.ttfb_histogram, 90):.4f}')
                lines.append(f'tts_ttfb_seconds{{quantile="0.99"}} {self.get_percentile(self.ttfb_histogram, 99):.4f}')
            
            # Latency percentiles
            if self.latency_histogram:
                lines.append("# HELP tts_latency_seconds Request latency")
                lines.append("# TYPE tts_latency_seconds summary")
                lines.append(f'tts_latency_seconds{{quantile="0.5"}} {self.get_percentile(self.latency_histogram, 50):.4f}')
                lines.append(f'tts_latency_seconds{{quantile="0.9"}} {self.get_percentile(self.latency_histogram, 90):.4f}')
                lines.append(f'tts_latency_seconds{{quantile="0.99"}} {self.get_percentile(self.latency_histogram, 99):.4f}')
            
            # RTF percentiles
            if self.rtf_histogram:
                lines.append("# HELP tts_rtf Real-time factor (lower is better)")
                lines.append("# TYPE tts_rtf summary")
                lines.append(f'tts_rtf{{quantile="0.5"}} {self.get_percentile(self.rtf_histogram, 50):.4f}')
                lines.append(f'tts_rtf{{quantile="0.9"}} {self.get_percentile(self.rtf_histogram, 90):.4f}')
                lines.append(f'tts_rtf{{quantile="0.99"}} {self.get_percentile(self.rtf_histogram, 99):.4f}')
            
            # Uptime
            lines.append("# HELP tts_uptime_seconds Server uptime")
            lines.append("# TYPE tts_uptime_seconds gauge")
            lines.append(f'tts_uptime_seconds {time.time() - self.start_time:.2f}')
            
            return "\n".join(lines)
    
    def get_summary(self) -> dict:
        """Get a summary of current metrics."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_requests": len(self._requests),
                "total_audio_seconds": round(self.total_audio_seconds, 2),
                "total_text_chars": self.total_text_chars,
                "ttfb_p50_ms": round(self.get_percentile(self.ttfb_histogram, 50) * 1000, 1),
                "ttfb_p95_ms": round(self.get_percentile(self.ttfb_histogram, 95) * 1000, 1),
                "latency_p50_ms": round(self.get_percentile(self.latency_histogram, 50) * 1000, 1),
                "latency_p95_ms": round(self.get_percentile(self.latency_histogram, 95) * 1000, 1),
                "rtf_p50": round(self.get_percentile(self.rtf_histogram, 50), 3),
                "rtf_p95": round(self.get_percentile(self.rtf_histogram, 95), 3),
                "uptime_seconds": round(time.time() - self.start_time, 2),
            }


# Global metrics instance
metrics = MetricsCollector()
