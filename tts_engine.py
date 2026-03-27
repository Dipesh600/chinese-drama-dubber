"""
TTS ENGINE v4 — Production-grade Text-to-Speech:
  1. Edge TTS with SSML mode (prosody control: rate, pitch, volume)
  2. Per-clip loudness normalization (-14 LUFS)
  3. Voice profile LOCKING — same character always uses same params
  4. Intelligent retry with text simplification on failure
  5. Fish Speech local GPU as premium option (localhost:8080)
  6. Duration-aware: warns if clip will overflow its time window
"""
import os, json, logging, subprocess, asyncio, hashlib, re
from concurrent.futures import ThreadPoolExecutor
logger = logging.getLogger(__name__)

try:
    import edge_tts
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False
    logger.warning("[TTS] edge_tts not installed")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_dur(path):
    """Get audio file duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip())
    except:
        return 0.0


def _normalize_clip(input_path, output_path, target_lufs=-14):
    """Normalize a single clip to target LUFS using loudnorm."""
    try:
        # Two-pass loudness normalization
        cmd1 = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-"
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        
        stderr = r1.stderr
        json_start = stderr.rfind('{')
        json_end = stderr.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            m = json.loads(stderr[json_start:json_end])
            cmd2 = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", (
                    f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                    f"measured_I={m.get('input_i','-24')}:"
                    f"measured_TP={m.get('input_tp','-1')}:"
                    f"measured_LRA={m.get('input_lra','11')}:"
                    f"measured_thresh={m.get('input_thresh','-34')}:"
                    f"linear=true"
                ),
                "-ar", "44100", "-ac", "1", output_path
            ]
            r2 = subprocess.run(cmd2, capture_output=True, timeout=30)
            if r2.returncode == 0:
                return True
        
        # Fallback: simple loudnorm
        cmd_s = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5",
            "-ar", "44100", "-ac", "1", output_path
        ]
        r = subprocess.run(cmd_s, capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception as e:
        logger.debug(f"[TTS] Normalize failed: {e}")
        return False


def _simplify_text(text):
    """Simplify text for TTS retry: remove special chars, shorten."""
    # Remove multiple punctuation
    text = re.sub(r'[!?]{2,}', '!', text)
    # Remove ellipsis in middle of text
    text = re.sub(r'\.{3,}', ', ', text)
    # Remove quotes and brackets
    text = re.sub(r'["\'\[\](){}]', '', text)
    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE TTS WITH SSML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _edge_tts_generate(text, output_path, voice, rate="+0%", pitch="+0Hz", volume="+0%"):
    """Generate TTS using Edge TTS with SSML prosody control."""
    if not HAS_EDGE:
        raise RuntimeError("edge_tts not installed")
    
    if not text or not text.strip():
        raise ValueError("Empty text")
    
    comm = edge_tts.Communicate(
        text=text.strip(),
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    
    await comm.save(output_path)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 500:
        raise RuntimeError(f"Edge TTS produced empty/small file")
    
    return output_path


def _edge_tts_sync(text, output_path, voice, rate="+0%", pitch="+0Hz", volume="+0%"):
    """Synchronous wrapper for Edge TTS."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        _edge_tts_generate(text, output_path, voice, rate, pitch, volume)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FISH SPEECH LOCAL (optional premium)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FISH_URL = "http://localhost:8080"

def _fish_available():
    """Check if Fish Speech local server is running."""
    if not HAS_HTTPX:
        return False
    try:
        r = httpx.get(f"{FISH_URL}/health", timeout=2.0)
        return r.status_code == 200
    except:
        return False


def _fish_generate(text, output_path, voice_id=None):
    """Generate TTS using local Fish Speech server."""
    try:
        payload = {"text": text, "format": "wav"}
        if voice_id:
            payload["voice"] = voice_id
        
        r = httpx.post(
            f"{FISH_URL}/v1/tts",
            json=payload,
            timeout=30.0
        )
        r.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(r.content)
        
        if os.path.getsize(output_path) < 500:
            raise RuntimeError("Fish Speech produced empty file")
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Fish Speech failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN GENERATION PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(segments, work_dir, cast_map, target_lang="Hindi"):
    """
    Generate TTS clips for all segments with:
    - Voice profile locking per character
    - Per-clip loudness normalization
    - Intelligent retry with text simplification
    - Fish Speech (if available) → Edge TTS fallback
    """
    clips_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    use_fish = _fish_available()
    engine = "Fish Speech (local GPU)" if use_fish else "Edge TTS (SSML)"
    
    logger.info(f"[TTS] Engine: {engine}")
    logger.info(f"[TTS] Generating {len(segments)} clips | {len(cast_map)} voice profiles")
    logger.info(f"[TTS] Profiles:")
    for vid, profile in cast_map.items():
        logger.info(f"[TTS]   {vid} → {profile['voice']} rate={profile['rate']} pitch={profile['pitch']}")
    
    manifest = []
    ok, fail = 0, 0
    
    for i, seg in enumerate(segments):
        text = seg.get("tts_text", seg.get("dubbed_text", "")).strip()
        if not text:
            logger.debug(f"[TTS] Segment {seg['id']} — empty text, skipping")
            continue
        
        speaker = seg.get("speaker", "NARRATOR")
        profile = cast_map.get(speaker, cast_map.get("NARRATOR", {}))
        
        voice = profile.get("voice", "hi-IN-MadhurNeural")
        rate = profile.get("rate", "+0%")
        pitch = profile.get("pitch", "+0Hz")
        volume = profile.get("volume", "+0%")
        
        # Generate clip filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        clip_name = f"clip_{seg['id']:04d}_{speaker}_{text_hash}.mp3"
        clip_path = os.path.join(clips_dir, clip_name)
        
        # Skip if already generated (caching)
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 500:
            dur = _get_dur(clip_path)
            if dur > 0:
                manifest.append({
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "clip_path": clip_path,
                    "actual_dur": dur,
                    "speaker": speaker,
                    "text": text[:50],
                    "tts_group": seg.get("tts_group"),
                })
                ok += 1
                continue
        
        # ── Generate TTS ────────────────────────────────────────
        generated = False
        
        for attempt in range(3):
            try:
                attempt_text = text if attempt == 0 else _simplify_text(text)
                
                if use_fish:
                    try:
                        _fish_generate(attempt_text, clip_path, voice_id=speaker.lower())
                        generated = True
                        break
                    except Exception as e:
                        logger.debug(f"[TTS] Fish failed seg {seg['id']}: {e}")
                        if attempt == 0:
                            # Fall back to Edge TTS for this clip
                            pass
                
                if not generated and HAS_EDGE:
                    _edge_tts_sync(attempt_text, clip_path, voice, rate, pitch, volume)
                    generated = True
                    break
                    
            except Exception as e:
                logger.debug(f"[TTS] Attempt {attempt+1} failed seg {seg['id']}: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
        
        if not generated:
            logger.warning(f"[TTS] ✗ Failed segment {seg['id']}: {text[:40]}")
            fail += 1
            continue
        
        # ── Per-clip loudness normalization ──────────────────────
        normalized_path = clip_path.replace(".mp3", "_norm.wav")
        if _normalize_clip(clip_path, normalized_path, target_lufs=-14):
            # Replace original with normalized
            os.replace(normalized_path, clip_path.replace(".mp3", ".wav"))
            final_path = clip_path.replace(".mp3", ".wav")
        else:
            final_path = clip_path
        
        dur = _get_dur(final_path)
        target_dur = seg["end"] - seg["start"]
        
        # Duration warning
        if dur > target_dur * 1.3 and target_dur > 0.5:
            logger.debug(
                f"[TTS] ⚠ Clip {seg['id']} is {dur:.1f}s but window is {target_dur:.1f}s "
                f"(will be time-stretched)"
            )
        
        manifest.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "clip_path": final_path,
            "actual_dur": dur,
            "speaker": speaker,
            "mood": seg.get("mood", "neutral"),
            "text": text[:50],
            "tts_group": seg.get("tts_group"),
        })
        ok += 1
        
        # Progress logging every 10 clips
        if (i + 1) % 10 == 0:
            logger.info(f"[TTS] Progress: {i+1}/{len(segments)} clips ({ok} ok, {fail} fail)")
    
    # ── Summary ──────────────────────────────────────────────────
    logger.info(f"[TTS] {'━' * 55}")
    logger.info(f"[TTS] ✓ {ok}/{len(segments)} clips generated ({fail} failed)")
    
    # Duration analysis
    total_clip_dur = sum(c.get("actual_dur", 0) for c in manifest)
    total_target_dur = sum(c["end"] - c["start"] for c in manifest)
    logger.info(f"[TTS] Total clip duration: {total_clip_dur:.1f}s | Target: {total_target_dur:.1f}s")
    
    overflows = sum(1 for c in manifest if c.get("actual_dur", 0) > (c["end"] - c["start"]) * 1.3)
    underflows = sum(1 for c in manifest if c.get("actual_dur", 0) < (c["end"] - c["start"]) * 0.6)
    if overflows:
        logger.info(f"[TTS] ⚠ {overflows} clips overflow their time windows (will be stretched)")
    if underflows:
        logger.info(f"[TTS] ⚠ {underflows} clips are much shorter than window (will be padded)")
    logger.info(f"[TTS] {'━' * 55}")
    
    # Save manifest
    mp = os.path.join(work_dir, "tts_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest
