"""
Production-Grade CSM TTS Server

OpenAI-compatible TTS API with streaming support.
"""

import asyncio
import io
import json
import logging
import os
import queue
import struct
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import torch
import torchaudio
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

# Local imports
from generator import Generator, Segment, load_csm_1b, load_csm_1b_local

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tts-server")


class ServerConfig:
    """Server configuration with environment variable overrides."""

    HOST: str = os.getenv("TTS_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("TTS_PORT", "8080"))
    
    # Model settings
    MODEL_PATH: str = os.getenv("TTS_MODEL_PATH", "")  # Empty = download from HF
    DEVICE: str = os.getenv("TTS_DEVICE", "cuda")
    
    # API settings
    API_KEYS: List[str] = os.getenv("TTS_API_KEYS", "").split(",") if os.getenv("TTS_API_KEYS") else []
    REQUIRE_AUTH: bool = os.getenv("TTS_REQUIRE_AUTH", "false").lower() == "true"
    
    # Rate limiting
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("TTS_MAX_CONCURRENT", "10"))
    MAX_TEXT_LENGTH: int = int(os.getenv("TTS_MAX_TEXT_LENGTH", "4096"))
    
    # Audio settings
    DEFAULT_VOICE: str = os.getenv("TTS_DEFAULT_VOICE", "default")
    MAX_AUDIO_LENGTH_MS: int = int(os.getenv("TTS_MAX_AUDIO_MS", "120000"))


config = ServerConfig()


# ============================================================================
# Pydantic Models
# ============================================================================

class AudioFormat(str, Enum):
    WAV = "wav"
    PCM = "pcm"
    MP3 = "mp3"
    OPUS = "opus"


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = Field(default="csm-1b", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize", max_length=4096)
    voice: str = Field(default="default", description="Voice ID to use")
    response_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed multiplier")
    
    # CSM-specific extensions
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    context_audio: Optional[str] = Field(default=None, description="Base64-encoded reference audio")
    context_text: Optional[str] = Field(default=None, description="Text for reference audio")


class StreamRequest(BaseModel):
    """WebSocket streaming request."""
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default")
    temperature: float = Field(default=0.7)
    top_k: int = Field(default=50)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used: Optional[str] = None
    uptime_seconds: float
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: Optional[str] = None


# ============================================================================
# Request Queue & Worker Management
# ============================================================================

@dataclass(order=True)
class PrioritizedRequest:
    """Request wrapper for priority queue."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    request: TTSRequest = field(compare=False)
    result_queue: asyncio.Queue = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


class TTSWorker:
    """
    GPU worker that processes TTS requests.
    Runs in a dedicated thread to avoid blocking the event loop.
    """

    def __init__(self, device: str = "cuda", model_path: str = ""):
        self.device = device
        self.model_path = model_path
        self.generator: Optional[Generator] = None
        self.request_queue: queue.Queue = queue.Queue()
        self.running = threading.Event()
        self.ready = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.start_time = time.time()
        
        # Voice cache
        self.voice_cache: Dict[str, List[Segment]] = {}

    def start(self):
        """Start the worker thread."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.running.set()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("TTS worker thread started")

    def stop(self):
        """Stop the worker thread."""
        self.running.clear()
        self.request_queue.put(None)  # Sentinel to unblock
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("TTS worker thread stopped")

    def _load_model(self):
        """Load the CSM model with CUDA optimizations."""
        logger.info(f"Loading CSM model on {self.device}...")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.empty_cache()
            logger.info("CUDA optimizations enabled (TF32, cuDNN benchmark, Flash SDP)")
        
        load_start = time.time()
        if self.model_path:
            self.generator = load_csm_1b_local(self.model_path, self.device)
        else:
            self.generator = load_csm_1b(self.device)
        
        logger.info(f"CSM model loaded and warmed up in {time.time() - load_start:.1f}s")
        self.ready.set()

    def _worker_loop(self):
        """Main worker loop - runs in dedicated thread."""
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

        while self.running.is_set():
            try:
                item = self.request_queue.get(timeout=0.1)
                if item is None:
                    continue
                
                request_id, request, result_queue, loop = item
                
                if request is None:  # Cancelled
                    continue
                
                try:
                    self._process_request(request_id, request, result_queue, loop)
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put({"error": str(e)}),
                        loop
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(
                        result_queue.put(None),  # Signal completion
                        loop
                    )
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    def _process_request(
        self,
        request_id: str,
        request: TTSRequest,
        result_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        """Process a single TTS request with detailed timing."""
        logger.info(f"Processing request {request_id}: {request.input[:50]}...")
        start_time = time.time()
        
        # Get voice context
        context = self._get_voice_context(request.voice)
        
        # Preprocess text
        text = self._preprocess_text(request.input)
        preprocess_time = time.time() - start_time
        
        # Estimate max audio length
        words = len(text.split())
        estimated_duration_ms = min(
            int((words / 2.5) * 1000),  # ~150 wpm
            config.MAX_AUDIO_LENGTH_MS
        )
        
        logger.info(f"Request {request_id}: {words} words, max {estimated_duration_ms}ms, preprocess {preprocess_time*1000:.1f}ms")
        
        chunk_count = 0
        first_chunk_time = None
        total_audio_samples = 0
        gen_start = time.time()
        
        # Stream generation
        for audio_chunk in self.generator.generate_stream(
            text=text,
            speaker=0,
            context=context,
            max_audio_length_ms=estimated_duration_ms,
            temperature=request.temperature,
            topk=request.top_k,
        ):
            chunk_count += 1
            total_audio_samples += audio_chunk.shape[0]
            
            if chunk_count == 1:
                first_chunk_time = time.time() - gen_start
                logger.info(f"Request {request_id}: First chunk from generator in {first_chunk_time*1000:.1f}ms ({audio_chunk.shape[0]} samples)")
            
            # Convert to bytes based on format
            audio_bytes = self._encode_chunk(audio_chunk, request.response_format)
            
            asyncio.run_coroutine_threadsafe(
                result_queue.put({"chunk": audio_bytes, "chunk_num": chunk_count}),
                loop
            )
        
        gen_time = time.time() - gen_start
        total_time = time.time() - start_time
        audio_seconds = total_audio_samples / 24000
        rtf = gen_time / audio_seconds if audio_seconds > 0 else 0
        
        logger.info(
            f"Request {request_id} complete: {chunk_count} chunks, {audio_seconds:.1f}s audio "
            f"in {gen_time:.2f}s (RTF: {rtf:.2f}x, TTFB: {first_chunk_time*1000:.1f}ms)"
        )

    def _get_voice_context(self, voice_id: str) -> List[Segment]:
        """Get voice context segments."""
        if voice_id in self.voice_cache:
            return self.voice_cache[voice_id]
        return []

    def _preprocess_text(self, text: str) -> str:
        """Clean text for TTS."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s.,!?\']', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.,!?])(\S)', r'\1 \2', text)
        return text.strip()

    def _encode_chunk(self, chunk: torch.Tensor, fmt: AudioFormat) -> bytes:
        """Encode audio chunk to bytes."""
        audio_np = chunk.cpu().numpy()
        
        if fmt == AudioFormat.PCM:
            # Raw 16-bit PCM
            return (audio_np * 32767).astype(np.int16).tobytes()
        
        elif fmt == AudioFormat.WAV:
            # WAV chunk (just the data, header handled separately)
            return (audio_np * 32767).astype(np.int16).tobytes()
        
        elif fmt == AudioFormat.MP3:
            # Would need pydub/ffmpeg - for now return PCM
            return (audio_np * 32767).astype(np.int16).tobytes()
        
        elif fmt == AudioFormat.OPUS:
            # Would need opuslib - for now return PCM
            return (audio_np * 32767).astype(np.int16).tobytes()
        
        return (audio_np * 32767).astype(np.int16).tobytes()

    def submit_request(
        self,
        request_id: str,
        request: TTSRequest,
        result_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ):
        """Submit a request for processing."""
        self.request_queue.put((request_id, request, result_queue, loop))

    def get_stats(self) -> dict:
        """Get worker statistics."""
        stats = {
            "ready": self.ready.is_set(),
            "queue_size": self.request_queue.qsize(),
            "uptime_seconds": time.time() - self.start_time,
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
            stats["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB"
        
        return stats


# ============================================================================
# Global State
# ============================================================================

worker: Optional[TTSWorker] = None
active_requests: Dict[str, asyncio.Queue] = {}
request_semaphore: Optional[asyncio.Semaphore] = None


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global worker, request_semaphore
    
    logger.info("Starting TTS server...")
    
    # Initialize semaphore for concurrent request limiting
    request_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    
    # Start worker
    worker = TTSWorker(device=config.DEVICE, model_path=config.MODEL_PATH)
    worker.start()
    
    # Wait for model to load
    logger.info("Waiting for model to load...")
    while not worker.ready.is_set():
        await asyncio.sleep(0.5)
    logger.info("Model ready!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down TTS server...")
    if worker:
        worker.stop()


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CSM TTS API",
    description="Production-grade Text-to-Speech API powered by Sesame CSM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key if authentication is required."""
    if not config.REQUIRE_AUTH:
        return True
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    
    # Support both "Bearer <key>" and raw key
    key = authorization.replace("Bearer ", "").strip()
    
    if key not in config.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return True


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and Kubernetes."""
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
    
    return HealthResponse(
        status="healthy" if worker and worker.ready.is_set() else "starting",
        model_loaded=worker.ready.is_set() if worker else False,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used=gpu_mem,
        uptime_seconds=time.time() - worker.start_time if worker else 0,
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if not worker or not worker.ready.is_set():
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}


@app.get("/v1/models")
async def list_models(authenticated: bool = Depends(verify_api_key)):
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "csm-1b",
                "object": "model",
                "created": 1704067200,
                "owned_by": "sesame",
            }
        ]
    }


@app.post("/v1/audio/speech")
async def create_speech(
    request: TTSRequest,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Generate speech from text (OpenAI-compatible endpoint).
    
    Returns audio as a streaming response.
    """
    if not worker or not worker.ready.is_set():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if len(request.input) > config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {config.MAX_TEXT_LENGTH} characters"
        )
    
    request_id = str(uuid4())[:8]
    logger.info(f"New request {request_id}: {len(request.input)} chars")
    
    # Acquire semaphore for concurrency limiting
    async with request_semaphore:
        result_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        # Submit to worker
        worker.submit_request(request_id, request, result_queue, loop)
        
        async def generate_audio():
            """Stream audio chunks as they're generated."""
            # Write WAV header for streaming
            if request.response_format == AudioFormat.WAV:
                # Placeholder header - we'll update chunk sizes as we go
                yield create_wav_header(0, 24000)
            
            total_bytes = 0
            
            while True:
                result = await result_queue.get()
                
                if result is None:
                    break
                
                if "error" in result:
                    logger.error(f"Request {request_id} error: {result['error']}")
                    break
                
                if "chunk" in result:
                    chunk_bytes = result["chunk"]
                    total_bytes += len(chunk_bytes)
                    yield chunk_bytes
            
            logger.info(f"Request {request_id}: Streamed {total_bytes} bytes")
        
        # Determine content type
        content_type = {
            AudioFormat.WAV: "audio/wav",
            AudioFormat.PCM: "audio/pcm",
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.OPUS: "audio/opus",
        }.get(request.response_format, "audio/wav")
        
        return StreamingResponse(
            generate_audio(),
            media_type=content_type,
            headers={
                "X-Request-ID": request_id,
                "Transfer-Encoding": "chunked",
            }
        )


@app.websocket("/v1/audio/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Protocol:
    1. Client sends JSON: {"input": "text", "voice": "default", ...}
    2. Server sends binary audio chunks
    3. Server sends JSON: {"status": "complete"} when done
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            
            if "input" not in msg:
                await websocket.send_json({"error": "Missing 'input' field"})
                continue
            
            # Create TTS request
            request = TTSRequest(
                input=msg["input"],
                voice=msg.get("voice", "default"),
                temperature=msg.get("temperature", 0.7),
                top_k=msg.get("top_k", 50),
                response_format=AudioFormat.PCM,
            )
            
            request_id = str(uuid4())[:8]
            result_queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()
            
            # Submit to worker
            worker.submit_request(request_id, request, result_queue, loop)
            
            # Send status
            await websocket.send_json({"status": "generating", "request_id": request_id})
            
            # Stream chunks
            while True:
                result = await result_queue.get()
                
                if result is None:
                    await websocket.send_json({"status": "complete", "request_id": request_id})
                    break
                
                if "error" in result:
                    await websocket.send_json({"error": result["error"], "request_id": request_id})
                    break
                
                if "chunk" in result:
                    # Send binary audio data
                    await websocket.send_bytes(result["chunk"])
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@app.get("/v1/voices")
async def list_voices(authenticated: bool = Depends(verify_api_key)):
    """List available voices."""
    voices = [
        {"id": "default", "name": "Default", "description": "Default CSM voice"},
    ]
    
    # Add cached voices
    if worker:
        for voice_id in worker.voice_cache.keys():
            if voice_id != "default":
                voices.append({
                    "id": voice_id,
                    "name": voice_id.title(),
                    "description": f"Custom voice: {voice_id}",
                })
    
    return {"voices": voices}


@app.post("/v1/voices")
async def create_voice(
    voice_id: str = Query(..., description="Voice ID"),
    reference_text: str = Query(..., description="Text spoken in reference audio"),
    authenticated: bool = Depends(verify_api_key),
):
    """
    Create a custom voice from reference audio.
    
    Upload reference audio as multipart form data.
    """
    # TODO: Implement voice cloning endpoint
    raise HTTPException(status_code=501, detail="Voice creation not yet implemented")


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if not worker:
        return ""
    
    stats = worker.get_stats()
    
    metrics = []
    metrics.append(f'tts_model_ready {{}} {1 if stats["ready"] else 0}')
    metrics.append(f'tts_queue_depth {{}} {stats["queue_size"]}')
    metrics.append(f'tts_uptime_seconds {{}} {stats["uptime_seconds"]:.2f}')
    
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        metrics.append(f'tts_gpu_memory_allocated_gb {{}} {gpu_alloc:.2f}')
        metrics.append(f'tts_gpu_memory_reserved_gb {{}} {gpu_reserved:.2f}')
    
    return "\n".join(metrics)


# ============================================================================
# Utilities
# ============================================================================

def create_wav_header(data_size: int, sample_rate: int = 24000) -> bytes:
    """Create a WAV header for streaming."""
    # For streaming, we use a placeholder size (will be chunked transfer)
    header = io.BytesIO()
    
    # RIFF header
    header.write(b'RIFF')
    header.write(struct.pack('<I', data_size + 36))  # File size - 8
    header.write(b'WAVE')
    
    # fmt chunk
    header.write(b'fmt ')
    header.write(struct.pack('<I', 16))  # Chunk size
    header.write(struct.pack('<H', 1))   # PCM format
    header.write(struct.pack('<H', 1))   # Mono
    header.write(struct.pack('<I', sample_rate))
    header.write(struct.pack('<I', sample_rate * 2))  # Byte rate
    header.write(struct.pack('<H', 2))   # Block align
    header.write(struct.pack('<H', 16))  # Bits per sample
    
    # data chunk
    header.write(b'data')
    header.write(struct.pack('<I', data_size))
    
    return header.getvalue()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting CSM TTS Server on {config.HOST}:{config.PORT}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Auth required: {config.REQUIRE_AUTH}")
    logger.info(f"Max concurrent requests: {config.MAX_CONCURRENT_REQUESTS}")
    
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        workers=1,  # Single worker for GPU
        log_level="info",
    )
