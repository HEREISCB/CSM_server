"""
Voice Manager for CSM TTS Server

Handles voice presets, reference audio, and voice cloning.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class VoicePreset:
    """A voice preset configuration."""
    id: str
    name: str
    description: str
    speaker_id: int = 0
    temperature: float = 0.7
    top_k: int = 50
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None


@dataclass 
class Segment:
    """Audio segment for voice context."""
    text: str
    speaker: int
    audio: torch.Tensor
    sample_rate: int = 24000


class VoiceManager:
    """
    Manages voice presets and reference audio for TTS.
    
    Features:
    - Built-in voice presets
    - Custom voice registration
    - Reference audio caching
    - Voice cloning support
    """
    
    def __init__(
        self,
        presets_path: str = "config/voices.yaml",
        cache_dir: str = "voice_cache",
        sample_rate: int = 24000,
    ):
        self.presets_path = Path(presets_path)
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Voice storage
        self.presets: Dict[str, VoicePreset] = {}
        self.segments_cache: Dict[str, List[Segment]] = {}
        
        # Load built-in presets
        self._load_builtin_presets()
        
        # Load custom presets if available
        if self.presets_path.exists():
            self._load_presets_file()
    
    def _load_builtin_presets(self):
        """Load built-in voice presets."""
        self.presets["default"] = VoicePreset(
            id="default",
            name="Default",
            description="Default CSM voice with balanced settings",
            speaker_id=0,
            temperature=0.7,
            top_k=50,
        )
        
        self.presets["expressive"] = VoicePreset(
            id="expressive",
            name="Expressive",
            description="More expressive with higher temperature",
            speaker_id=0,
            temperature=0.9,
            top_k=60,
        )
        
        self.presets["stable"] = VoicePreset(
            id="stable",
            name="Stable",
            description="More stable and consistent output",
            speaker_id=0,
            temperature=0.5,
            top_k=30,
        )
        
        logger.info(f"Loaded {len(self.presets)} built-in voice presets")
    
    def _load_presets_file(self):
        """Load voice presets from YAML file."""
        try:
            import yaml
            with open(self.presets_path) as f:
                data = yaml.safe_load(f)
            
            if "voices" in data:
                for voice_id, voice_data in data["voices"].items():
                    self.presets[voice_id] = VoicePreset(
                        id=voice_id,
                        name=voice_data.get("name", voice_id),
                        description=voice_data.get("description", ""),
                        speaker_id=voice_data.get("speaker_id", 0),
                        temperature=voice_data.get("temperature", 0.7),
                        top_k=voice_data.get("top_k", 50),
                        reference_audio_path=voice_data.get("reference_audio_path"),
                        reference_text=voice_data.get("reference_text"),
                    )
            
            logger.info(f"Loaded {len(self.presets)} voice presets from {self.presets_path}")
        except Exception as e:
            logger.warning(f"Could not load presets file: {e}")
    
    def get_preset(self, voice_id: str) -> VoicePreset:
        """
        Get a voice preset by ID.
        
        Falls back to default if not found.
        """
        if voice_id in self.presets:
            return self.presets[voice_id]
        
        logger.warning(f"Voice preset '{voice_id}' not found, using default")
        return self.presets["default"]
    
    def list_voices(self) -> List[Dict]:
        """List all available voices."""
        return [
            {
                "id": preset.id,
                "name": preset.name,
                "description": preset.description,
            }
            for preset in self.presets.values()
        ]
    
    def get_segments(self, voice_id: str) -> List[Segment]:
        """
        Get context segments for a voice.
        
        Returns cached segments or loads from reference audio.
        """
        # Check cache
        if voice_id in self.segments_cache:
            return self.segments_cache[voice_id]
        
        preset = self.get_preset(voice_id)
        segments = []
        
        # Load reference audio if available
        if preset.reference_audio_path and os.path.exists(preset.reference_audio_path):
            try:
                segment = self._load_reference_audio(
                    preset.reference_audio_path,
                    preset.reference_text or "",
                    preset.speaker_id,
                )
                if segment:
                    segments.append(segment)
            except Exception as e:
                logger.error(f"Failed to load reference audio: {e}")
        
        # Cache and return
        self.segments_cache[voice_id] = segments
        return segments
    
    def _load_reference_audio(
        self,
        audio_path: str,
        text: str,
        speaker_id: int,
    ) -> Optional[Segment]:
        """Load and preprocess reference audio."""
        try:
            audio, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio.squeeze(0),
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                )
            else:
                audio = audio.squeeze(0)
            
            return Segment(
                text=text,
                speaker=speaker_id,
                audio=audio,
                sample_rate=self.sample_rate,
            )
        except Exception as e:
            logger.error(f"Error loading reference audio from {audio_path}: {e}")
            return None
    
    def register_voice(
        self,
        voice_id: str,
        name: str,
        reference_audio: bytes,
        reference_text: str,
        description: str = "",
    ) -> bool:
        """
        Register a custom voice from reference audio.
        
        Args:
            voice_id: Unique voice identifier
            name: Display name for the voice
            reference_audio: Audio bytes (WAV format)
            reference_text: Text spoken in the reference audio
            description: Optional description
        
        Returns:
            True if successful
        """
        try:
            # Generate cache filename
            audio_hash = hashlib.md5(reference_audio).hexdigest()[:12]
            cache_path = self.cache_dir / f"{voice_id}_{audio_hash}.wav"
            
            # Save audio to cache
            with open(cache_path, "wb") as f:
                f.write(reference_audio)
            
            # Create preset
            self.presets[voice_id] = VoicePreset(
                id=voice_id,
                name=name,
                description=description,
                speaker_id=0,
                temperature=0.7,
                top_k=50,
                reference_audio_path=str(cache_path),
                reference_text=reference_text,
            )
            
            # Clear segments cache to force reload
            if voice_id in self.segments_cache:
                del self.segments_cache[voice_id]
            
            logger.info(f"Registered custom voice: {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register voice {voice_id}: {e}")
            return False
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a custom voice."""
        if voice_id in ["default", "expressive", "stable"]:
            logger.warning(f"Cannot delete built-in voice: {voice_id}")
            return False
        
        if voice_id in self.presets:
            preset = self.presets[voice_id]
            
            # Delete cached audio
            if preset.reference_audio_path and os.path.exists(preset.reference_audio_path):
                try:
                    os.remove(preset.reference_audio_path)
                except Exception as e:
                    logger.warning(f"Could not delete cache file: {e}")
            
            del self.presets[voice_id]
            
            if voice_id in self.segments_cache:
                del self.segments_cache[voice_id]
            
            logger.info(f"Deleted voice: {voice_id}")
            return True
        
        return False


# Global voice manager instance
voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get or create the global voice manager."""
    global voice_manager
    if voice_manager is None:
        voice_manager = VoiceManager()
    return voice_manager
