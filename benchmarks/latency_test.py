"""
TTS Latency Benchmark

Measures TTFB, total latency, and RTF for the CSM TTS server.

Usage:
    python benchmarks/latency_test.py --host localhost --port 8080 --requests 10
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import List, Tuple

import httpx


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self):
        self.ttfb_times: List[float] = []
        self.total_times: List[float] = []
        self.audio_bytes: List[int] = []
        self.errors: List[str] = []
    
    def add_success(self, ttfb: float, total: float, audio_size: int):
        self.ttfb_times.append(ttfb)
        self.total_times.append(total)
        self.audio_bytes.append(audio_size)
    
    def add_error(self, error: str):
        self.errors.append(error)
    
    def percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        total_requests = len(self.ttfb_times) + len(self.errors)
        success_rate = len(self.ttfb_times) / total_requests * 100 if total_requests > 0 else 0
        
        print(f"\nRequests: {total_requests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Errors: {len(self.errors)}")
        
        if self.ttfb_times:
            print("\n--- Time to First Byte (TTFB) ---")
            print(f"  Min:    {min(self.ttfb_times) * 1000:.1f}ms")
            print(f"  Max:    {max(self.ttfb_times) * 1000:.1f}ms")
            print(f"  Mean:   {statistics.mean(self.ttfb_times) * 1000:.1f}ms")
            print(f"  Median: {statistics.median(self.ttfb_times) * 1000:.1f}ms")
            print(f"  P95:    {self.percentile(self.ttfb_times, 95) * 1000:.1f}ms")
            print(f"  P99:    {self.percentile(self.ttfb_times, 99) * 1000:.1f}ms")
        
        if self.total_times:
            print("\n--- Total Latency ---")
            print(f"  Min:    {min(self.total_times) * 1000:.1f}ms")
            print(f"  Max:    {max(self.total_times) * 1000:.1f}ms")
            print(f"  Mean:   {statistics.mean(self.total_times) * 1000:.1f}ms")
            print(f"  Median: {statistics.median(self.total_times) * 1000:.1f}ms")
            print(f"  P95:    {self.percentile(self.total_times, 95) * 1000:.1f}ms")
            print(f"  P99:    {self.percentile(self.total_times, 99) * 1000:.1f}ms")
        
        if self.audio_bytes:
            total_audio_kb = sum(self.audio_bytes) / 1024
            # Estimate audio duration (24kHz, 16-bit mono = 48KB/s)
            total_audio_seconds = sum(self.audio_bytes) / (24000 * 2)
            total_generation_time = sum(self.total_times)
            
            rtf = total_generation_time / total_audio_seconds if total_audio_seconds > 0 else 0
            
            print("\n--- Audio Output ---")
            print(f"  Total Size: {total_audio_kb:.1f}KB")
            print(f"  Total Duration: {total_audio_seconds:.1f}s")
            print(f"  Real-Time Factor: {rtf:.3f}x")
        
        if self.errors:
            print("\n--- Errors ---")
            for i, error in enumerate(self.errors[:5], 1):
                print(f"  {i}. {error[:100]}")
        
        print("\n" + "=" * 60)


async def benchmark_request(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
) -> Tuple[bool, float, float, int, str]:
    """
    Make a single benchmark request.
    
    Returns: (success, ttfb, total_time, audio_bytes, error)
    """
    start_time = time.perf_counter()
    ttfb = 0.0
    audio_bytes = 0
    
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/audio/speech",
            json={"input": text, "voice": "default"},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                return (False, 0, 0, 0, f"HTTP {response.status_code}")
            
            first_chunk = True
            async for chunk in response.aiter_bytes():
                if first_chunk:
                    ttfb = time.perf_counter() - start_time
                    first_chunk = False
                audio_bytes += len(chunk)
        
        total_time = time.perf_counter() - start_time
        return (True, ttfb, total_time, audio_bytes, "")
    
    except Exception as e:
        return (False, 0, 0, 0, str(e))


async def run_benchmark(
    host: str,
    port: int,
    num_requests: int,
    warmup_requests: int,
    concurrent: int,
    text: str,
):
    """Run the benchmark."""
    base_url = f"http://{host}:{port}"
    result = BenchmarkResult()
    
    print(f"Benchmarking {base_url}")
    print(f"Text: \"{text[:50]}...\"" if len(text) > 50 else f"Text: \"{text}\"")
    print(f"Requests: {num_requests} (warmup: {warmup_requests}, concurrent: {concurrent})")
    
    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/health", timeout=5.0)
            health = response.json()
            print(f"Server status: {health.get('status', 'unknown')}")
            print(f"Model loaded: {health.get('model_loaded', False)}")
            
            if not health.get("model_loaded", False):
                print("ERROR: Model not loaded!")
                return
        except Exception as e:
            print(f"ERROR: Cannot connect to server: {e}")
            return
    
    # Warmup phase
    if warmup_requests > 0:
        print(f"\n--- Warmup ({warmup_requests} requests) ---")
        async with httpx.AsyncClient() as client:
            for i in range(warmup_requests):
                success, ttfb, total, size, error = await benchmark_request(
                    client, base_url, text
                )
                status = "OK" if success else f"FAIL: {error}"
                print(f"  Warmup {i+1}/{warmup_requests}: {status}")
    
    # Benchmark phase
    print(f"\n--- Benchmark ({num_requests} requests) ---")
    
    async with httpx.AsyncClient() as client:
        if concurrent == 1:
            # Sequential requests
            for i in range(num_requests):
                success, ttfb, total, size, error = await benchmark_request(
                    client, base_url, text
                )
                
                if success:
                    result.add_success(ttfb, total, size)
                    print(f"  Request {i+1}/{num_requests}: TTFB={ttfb*1000:.0f}ms, Total={total*1000:.0f}ms, Size={size/1024:.1f}KB")
                else:
                    result.add_error(error)
                    print(f"  Request {i+1}/{num_requests}: FAIL - {error}")
        else:
            # Concurrent requests
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_request(i: int):
                async with semaphore:
                    return await benchmark_request(client, base_url, text)
            
            tasks = [bounded_request(i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)
            
            for i, (success, ttfb, total, size, error) in enumerate(results):
                if success:
                    result.add_success(ttfb, total, size)
                else:
                    result.add_error(error)
    
    result.print_summary()


def main():
    parser = argparse.ArgumentParser(description="TTS Server Latency Benchmark")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--requests", "-n", type=int, default=10, help="Number of requests")
    parser.add_argument("--warmup", "-w", type=int, default=3, help="Warmup requests")
    parser.add_argument("--concurrent", "-c", type=int, default=1, help="Concurrent requests")
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the text to speech system. The quick brown fox jumps over the lazy dog.",
        help="Text to synthesize"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(
        host=args.host,
        port=args.port,
        num_requests=args.requests,
        warmup_requests=args.warmup,
        concurrent=args.concurrent,
        text=args.text,
    ))


if __name__ == "__main__":
    main()
