# CSM TTS API - Production Server

Production-grade Text-to-Speech API powered by [Sesame CSM](https://github.com/SesameAILabs/csm).

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8080/health
```

### Option 2: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

## API Endpoints

### Generate Speech (OpenAI-compatible)

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "default"}' \
  --output output.wav
```

### WebSocket Streaming

Connect to `ws://localhost:8080/v1/audio/stream` and send:
```json
{"input": "Hello world", "voice": "default"}
```

Receive binary audio chunks in real-time.

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (streaming response) |
| `/v1/audio/stream` | WS | Real-time WebSocket streaming |
| `/v1/models` | GET | List available models |
| `/v1/voices` | GET | List available voices |
| `/health` | GET | Health check |
| `/ready` | GET | Kubernetes readiness probe |
| `/metrics` | GET | Prometheus metrics |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PORT` | 8080 | Server port |
| `TTS_DEVICE` | cuda | Device (cuda/cpu) |
| `TTS_MODEL_PATH` | "" | Local model path (empty = download) |
| `TTS_REQUIRE_AUTH` | false | Require API key |
| `TTS_API_KEYS` | "" | Comma-separated API keys |
| `TTS_MAX_CONCURRENT` | 10 | Max concurrent requests |

## Benchmarking

```bash
python benchmarks/latency_test.py --host localhost --port 8080 --requests 10
```

## Python Client

```python
import httpx
import asyncio

async def tts(text: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8080/v1/audio/speech",
            json={"input": text}
        )
        return response.content

# Usage
audio = asyncio.run(tts("Hello, world!"))
```

## Performance

On RTX 4090:
- **TTFB**: ~200-300ms
- **RTF**: ~0.28x (faster than real-time)
- **Concurrent requests**: 10+

## License

Apache 2.0 - See LICENSE file.
