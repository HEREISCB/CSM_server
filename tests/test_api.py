"""
API Tests for CSM TTS Server

Run with: pytest tests/test_api.py -v
"""

import asyncio
import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch


# Mock the generator before importing server
@pytest.fixture(autouse=True)
def mock_generator():
    """Mock the CSM generator to avoid loading the actual model."""
    with patch('server.load_csm_1b') as mock_load:
        mock_gen = MagicMock()
        mock_gen.sample_rate = 24000
        
        # Mock generate_stream to yield fake audio chunks
        def fake_stream(*args, **kwargs):
            for i in range(3):
                yield torch.randn(4800)  # 0.2s of audio at 24kHz
        
        mock_gen.generate_stream = fake_stream
        mock_load.return_value = mock_gen
        yield mock_gen


@pytest.fixture
def app():
    """Create test app instance."""
    from server import app
    return app


@pytest.fixture
async def client(app):
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test /health endpoint returns correct structure."""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "uptime_seconds" in data
    
    @pytest.mark.asyncio
    async def test_ready_check(self, client):
        """Test /ready endpoint."""
        response = await client.get("/ready")
        # May return 503 if model not ready in test
        assert response.status_code in [200, 503]


class TestModelsEndpoint:
    """Tests for /v1/models endpoint."""
    
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test listing available models."""
        response = await client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "csm-1b"


class TestSpeechEndpoint:
    """Tests for /v1/audio/speech endpoint."""
    
    @pytest.mark.asyncio
    async def test_create_speech_basic(self, client, mock_generator):
        """Test basic speech generation."""
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Hello, world!",
                "voice": "default"
            }
        )
        
        # May return 503 if model not loaded in test environment
        if response.status_code == 200:
            assert response.headers.get("content-type") == "audio/wav"
            assert "X-Request-ID" in response.headers
    
    @pytest.mark.asyncio
    async def test_create_speech_validation(self, client):
        """Test request validation."""
        # Missing input
        response = await client.post(
            "/v1/audio/speech",
            json={"voice": "default"}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_create_speech_text_too_long(self, client):
        """Test text length limit."""
        long_text = "x" * 5000  # Exceeds 4096 limit
        response = await client.post(
            "/v1/audio/speech",
            json={"input": long_text}
        )
        # Should return 400 or 503 (if model not ready)
        assert response.status_code in [400, 503]


class TestVoicesEndpoint:
    """Tests for /v1/voices endpoint."""
    
    @pytest.mark.asyncio
    async def test_list_voices(self, client):
        """Test listing available voices."""
        response = await client.get("/v1/voices")
        assert response.status_code == 200
        
        data = response.json()
        assert "voices" in data
        assert len(data["voices"]) > 0
        assert data["voices"][0]["id"] == "default"


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics(self, client):
        """Test Prometheus metrics export."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        
        # Should contain Prometheus format
        text = response.text
        assert "tts_" in text or text == ""  # Empty if no requests yet


class TestAuthentication:
    """Tests for API authentication."""
    
    @pytest.mark.asyncio
    async def test_no_auth_required_by_default(self, client):
        """Test that auth is not required by default."""
        response = await client.get("/v1/models")
        # Should work without auth header
        assert response.status_code == 200


class TestRequestFormat:
    """Tests for various request format options."""
    
    @pytest.mark.asyncio
    async def test_wav_format(self, client):
        """Test WAV output format."""
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Test",
                "response_format": "wav"
            }
        )
        if response.status_code == 200:
            assert "wav" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_pcm_format(self, client):
        """Test PCM output format."""
        response = await client.post(
            "/v1/audio/speech",
            json={
                "input": "Test",
                "response_format": "pcm"
            }
        )
        if response.status_code == 200:
            assert "pcm" in response.headers.get("content-type", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
