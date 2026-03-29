"""
SCHEMAS — Pydantic models for pipeline stage data contracts.
Validates data at critical boundaries to catch issues early.
"""
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator


class Segment(BaseModel):
    """A single subtitle/TTS segment."""
    id: int
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str = ""
    dubbed_text: Optional[str] = None
    speaker: str = "NARRATOR"
    mood: str = "neutral"
    tts_text: Optional[str] = None
    tts_group: Optional[int] = None

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end ({v}) must be >= start ({start})")
        return v

    @field_validator("speaker")
    @classmethod
    def speaker_not_empty(cls, v):
        if not v or not v.strip():
            return "NARRATOR"
        return v.strip()


class DirectorResult(BaseModel):
    """Output from director.analyze() — validated before entering translator."""
    content_type: str = Field(..., min_length=1)
    real_speaker_count: int = Field(..., ge=1)
    narrative_summary: str = ""
    mood_arc: Optional[List[str]] = None
    scenes: List[Any] = Field(default_factory=list)
    segment_moods: dict = Field(default_factory=dict)
    speaker_map: dict = Field(default_factory=dict)
    voice_plan: List[Any] = Field(default_factory=list)
    segments: List[Segment] = Field(..., min_length=1)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v):
        valid = {"drama", "action", "comedy", "romance", "documentary",
                 "interview", "narration", "unknown"}
        if v.lower() not in valid:
            # Soft validation — just warn
            pass
        return v


class TTSManifestEntry(BaseModel):
    """A single TTS clip entry from tts_engine.generate()."""
    id: int
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    clip_path: str = ""
    actual_dur: float = Field(default=0.0, ge=0)
    speaker: str = "NARRATOR"
    mood: str = "neutral"
    text: str = ""
    tts_group: Optional[int] = None
    engine: Optional[str] = None
    cached: bool = False


def validate_director_result(data: dict) -> DirectorResult:
    """
    Validate director.analyze() output.
    Raises ValueError/ValidationError if invalid.
    """
    # Reconstruct segments with proper models if needed
    segments_data = data.get("segments", [])
    if segments_data and not isinstance(segments_data[0], Segment):
        # Convert raw dicts to Segment models
        segments_data = [Segment(**s) if isinstance(s, dict) else s for s in segments_data]

    validated = DirectorResult(
        content_type=data.get("content_type", "unknown"),
        real_speaker_count=data.get("real_speaker_count", 1),
        narrative_summary=data.get("narrative_summary", ""),
        mood_arc=data.get("mood_arc"),
        scenes=data.get("scenes", []),
        segment_moods=data.get("segment_moods", {}),
        speaker_map=data.get("speaker_map", {}),
        voice_plan=data.get("voice_plan", []),
        segments=segments_data,
    )
    return validated


def validate_tts_manifest(manifest: List[dict]) -> List[TTSManifestEntry]:
    """Validate tts_engine manifest output."""
    return [TTSManifestEntry(**m) for m in manifest]


def validate_segments(segments: List[dict], field: str = "segments") -> List[Segment]:
    """Validate a list of segments."""
    result = []
    for i, s in enumerate(segments):
        try:
            result.append(Segment(**s) if isinstance(s, dict) else s)
        except Exception as e:
            raise ValueError(f"Invalid segment at index {i} in {field}: {e}")
    return result
