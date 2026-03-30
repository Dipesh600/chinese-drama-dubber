"""
TRANSCRIBER — Local GPU transcription using faster-whisper (CTranslate2).

Primary: faster-whisper (local GPU, no API cost)
Fallback: Groq Whisper API

faster-whisper is ~4-6x faster than original Whisper with identical accuracy.
Uses CTranslate2 for GPU-accelerated inference.
"""
import os, logging, time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Check available backends
HAS_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    logger.warning("[TRANSCRIBER] faster-whisper not installed (pip install faster-whisper)")

HAS_GROQ_WHISPER = False
try:
    from groq import Groq
    HAS_GROQ_WHISPER = True
except ImportError:
    logger.warning("[TRANSCRIBER] groq not installed for fallback transcription")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# faster-whisper model sizes — larger = more accurate but slower
# On T4 GPU: distil-large-v3 is good balance (medium/small also great for speed)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "distil-large-v3")
WHISPER_DEVICE = "cuda"  # GPU for speed
WHISPER_COMPUTE = "float16"  # float16 on GPU

# Optional: enable speaker diarization (requires pyannote.audio)
ENABLE_DIARIZATION = os.environ.get("ENABLE_DIARIZATION", "false").lower() == "true"
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline as DiaPipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    logger.warning("[TRANSCRIBER] pyannote.audio not installed — speaker diarization disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTER-WHISPER TRANSCRIPTION (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════════

_faster_model = None


def _get_faster_model():
    """Get or create singleton faster-whisper model."""
    global _faster_model
    if _faster_model is None:
        if not HAS_FASTER_WHISPER:
            raise RuntimeError("faster-whisper not installed")
        logger.info(f"[TRANSCRIBER] Loading faster-whisper {WHISPER_MODEL_SIZE} on {WHISPER_DEVICE}...")
        _faster_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        logger.info(f"[TRANSCRIBER] ✓ Model loaded!")
    return _faster_model


def _transcribe_faster(audio_path: str, language: str = None) -> Optional[Dict[str, Any]]:
    """
    Transcribe using faster-whisper (local GPU).
    Returns word-level timestamps + segments.
    """
    model = _get_faster_model()

    # Language parameter
    lang_param = language if language and language != "auto" else None

    logger.info(f"[TRANSCRIBER] Transcribing with faster-whisper ({WHISPER_MODEL_SIZE})...")
    t0 = time.time()

    # Word-level timestamps = True for precise subtitle alignment
    segments, info = model.transcribe(
        audio_path,
        language=lang_param,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=None,
    )

    # Collect results
    seg_list = []
    word_list = []

    for seg in segments:
        seg_text = seg.text.strip()
        seg_list.append({
            "id": len(seg_list),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg_text,
        })

        if seg.words:
            for w in seg.words:
                word_list.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })

    elapsed = time.time() - t0
    logger.info(
        f"[TRANSCRIBER] ✓ {len(seg_list)} segments, {len(word_list)} words "
        f"in {elapsed:.1f}s ({info.language or 'auto-detected'})"
    )

    return {"segments": seg_list, "words": word_list, "language": info.language}


# ═══════════════════════════════════════════════════════════════════════════════
# SPEAKER DIARIZATION (OPTIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

_dia_pipeline = None


def _get_diarizer():
    """Get or create pyannote diarization pipeline."""
    global _dia_pipeline
    if _dia_pipeline is None:
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError("pyannote.audio not installed")
        # Requires Hugging Face token with pyannote access
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            logger.warning("[TRANSCRIBER] HF_TOKEN not set — diarization needs pyannote.token")
            logger.warning("[TRANSCRIBER] Set: os.environ['HF_TOKEN'] = 'hf_...'")
        _dia_pipeline = DiaPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token or None,
        )
    return _dia_pipeline


def _assign_speakers(segments: List[Dict], words: List[Dict], audio_path: str) -> List[Dict]:
    """
    Assign speaker labels to segments using pyannote.audio diarization.
    Updates segment entries with 'speaker' field.
    """
    if not ENABLE_DIARIZATION:
        return segments

    try:
        import torch
        diarizer = _get_diarizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diarizer.to(torch.device(device))

        logger.info("[TRANSCRIBER] Running speaker diarization...")
        diarization = diarizer(audio_path)

        # Build speaker map: timestamp → speaker
        # For each segment, find which speaker was talking
        seg_speakers = []
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            # Find speaker active at midpoint of segment
            mid = (seg_start + seg_end) / 2
            speaker = "UNKNOWN"
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= mid <= turn.end:
                    speaker = speaker
                    break
            seg_speakers.append(speaker)

        # Assign speaker labels back to segments
        for i, seg in enumerate(segments):
            seg["speaker"] = seg_speakers[i]

        unique_speakers = set(seg_speakers)
        logger.info(f"[TRANSCRIBER] ✓ Speaker diarization: {len(unique_speakers)} speakers")
        for spk in sorted(unique_speakers):
            count = seg_speakers.count(spk)
            logger.info(f"[TRANSCRIBER]   {spk}: {count} segments")

    except Exception as e:
        logger.warning(f"[TRANSCRIBER] Diarization failed: {e} — using NARRATOR for all")

    return segments


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ WHISPER FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def _transcribe_groq(audio_path: str, language: str = None) -> Optional[Dict[str, Any]]:
    """Transcribe using Groq Whisper API (fallback)."""
    if not HAS_GROQ_WHISPER:
        raise RuntimeError("Groq not available for fallback transcription")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)
    logger.info("[TRANSCRIBER] Transcribing with Groq Whisper API...")

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language=language if language != "auto" else None,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

    segs = [{
        "id": i,
        "start": round(s.start, 3),
        "end": round(s.end, 3),
        "text": s.text.strip()
    } for i, s in enumerate(resp.segments)]

    words = []
    if hasattr(resp, 'words') and resp.words:
        for w in resp.words:
            words.append({
                "word": w.word,
                "start": w.start,
                "end": w.end,
            })

    logger.info(f"[TRANSCRIBER] ✓ {len(segs)} segments via Groq")
    return {"segments": segs, "words": words}


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED TRANSCRIBE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe(
    audio_path: str,
    language: str = None,
    use_faster: bool = True,
    enable_diarization: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Transcribe audio with automatic backend selection.

    Args:
        audio_path: Path to audio file (wav/mp3/etc)
        language: Source language code (None = auto-detect)
        use_faster: Try faster-whisper first (default: True)
        enable_diarization: Run speaker diarization (requires pyannote.audio + HF_TOKEN)

    Returns:
        Dict with segments (list) and words (list), each with start/end timestamps
    """
    global ENABLE_DIARIZATION
    ENABLE_DIARIZATION = enable_diarization

    result = None

    if use_faster and HAS_FASTER_WHISPER:
        try:
            result = _transcribe_faster(audio_path, language)
        except Exception as e:
            logger.warning(f"[TRANSCRIBER] faster-whisper failed: {e}")

    # Fallback to Groq if faster-whisper unavailable or failed
    if result is None:
        if HAS_GROQ_WHISPER:
            try:
                result = _transcribe_groq(audio_path, language)
            except Exception as e:
                logger.error(f"[TRANSCRIBER] Groq Whisper fallback also failed: {e}")
        else:
            raise RuntimeError(
                "[TRANSCRIBER] No transcription backend available. "
                "Install faster-whisper: pip install faster-whisper"
            )

    # Apply speaker diarization if enabled
    if result and enable_diarization and ENABLE_DIARIZATION:
        result["segments"] = _assign_speakers(
            result["segments"], result.get("words", []), audio_path
        )

    return result
