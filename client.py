"""
Example Client for CSM TTS API

Shows how to use the TTS API from Python.
"""

import asyncio
import wave
import sys
from pathlib import Path

import httpx


async def text_to_speech_rest(
    text: str,
    output_file: str = "output.wav",
    host: str = "localhost",
    port: int = 8080,
    voice: str = "default",
):
    """
    Generate speech using the REST API.
    
    Streams audio chunks as they're generated.
    """
    base_url = f"http://{host}:{port}"
    
    print(f"Generating speech for: \"{text[:50]}...\"" if len(text) > 50 else f"Generating speech for: \"{text}\"")
    
    async with httpx.AsyncClient() as client:
        # Make streaming request
        async with client.stream(
            "POST",
            f"{base_url}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                return
            
            # Stream to file
            with open(output_file, "wb") as f:
                chunk_count = 0
                total_bytes = 0
                
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    f.write(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    
                    if chunk_count % 10 == 0:
                        print(f"  Received {total_bytes / 1024:.1f}KB...")
            
            print(f"Done! Saved {total_bytes / 1024:.1f}KB to {output_file}")


async def text_to_speech_websocket(
    text: str,
    output_file: str = "output_ws.wav",
    host: str = "localhost",
    port: int = 8080,
    voice: str = "default",
):
    """
    Generate speech using WebSocket streaming.
    
    Lower latency as chunks are received immediately.
    """
    import websockets
    import json
    
    ws_url = f"ws://{host}:{port}/v1/audio/stream"
    sample_rate = 24000
    
    print(f"Connecting to {ws_url}...")
    
    audio_chunks = []
    
    async with websockets.connect(ws_url) as ws:
        # Send request
        await ws.send(json.dumps({
            "input": text,
            "voice": voice,
        }))
        
        print("Request sent, waiting for audio...")
        
        while True:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                # Audio chunk
                audio_chunks.append(message)
                print(f"  Chunk {len(audio_chunks)}: {len(message)} bytes")
            else:
                # JSON status message
                data = json.loads(message)
                print(f"  Status: {data}")
                
                if data.get("status") == "complete":
                    break
                elif "error" in data:
                    print(f"Error: {data['error']}")
                    break
    
    # Save to WAV file
    if audio_chunks:
        with wave.open(output_file, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            
            for chunk in audio_chunks:
                wav.writeframes(chunk)
        
        print(f"Saved to {output_file}")


def example_curl_commands():
    """Print example cURL commands."""
    print("""
Example cURL Commands
=====================

# Generate speech (WAV)
curl -X POST http://localhost:8080/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, world!", "voice": "default"}' \\
  --output output.wav

# Generate speech with options
curl -X POST http://localhost:8080/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{
    "input": "Hello, world!",
    "voice": "default",
    "response_format": "wav",
    "temperature": 0.7,
    "top_k": 50
  }' \\
  --output output.wav

# List models
curl http://localhost:8080/v1/models

# List voices
curl http://localhost:8080/v1/voices

# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <text>")
        print("       python client.py --examples")
        sys.exit(1)
    
    if sys.argv[1] == "--examples":
        example_curl_commands()
    else:
        text = " ".join(sys.argv[1:])
        asyncio.run(text_to_speech_rest(text))
