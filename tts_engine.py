"""
TTS ENGINE v5 — Production-grade Text-to-Speech:
  1. Fish Speech PRIMARY (local GPU, best quality, zero cost)
  2. Edge TTS FALLBACK (cloud, always works)
  3. Parallel clip generation with ThreadPoolExecutor
  4. Per-clip loudness normalization (-14 LUFS)
  5. Voice profile LOCKING — same character always uses same params
  6. Intelligent retry with text simplification on failure
  7. Duration-aware: warns if clip will overflow its time window
  8. Content-hash caching — skip already-generated clips
"""
import os, json, logging, subprocess, asyncio, hashlib, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

logger = logging.getLogger(__name__)

try:
    import edge_tts
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False
    logger.warning("[TTS] edge_tts not installed — will use Fish Speech only")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logger.warning("[TTS] httpx not installed — Fish Speech unavailable")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FISH_URL = os.environ.get("FISH_SPEECH_URL", "http://localhost:8080")
TTS_WORKERS = int(os.environ.get("TTS_WORKERS", "5"))
MAX_RETRIES = 3
TARGET_LUFS = -14


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


def _normalize_clip(input_path, output_path, target_lufs=TARGET_LUFS):
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
    text = re.sub(r'[!?]{2,}', '!', text)
    text = re.sub(r'\.{3,}', ', ', text)
    text = re.sub(r'["\'\[\](){}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FISH SPEECH (PRIMARY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fish_available():
    """Check if Fish Speech local server is running."""
    if not HAS_HTTPX:
        return False
    try:
        r = httpx.get(f"{FISH_URL}/health", timeout=2.0)
        return r.status_code == 200
    except:
        return False


def _fish_generate(text, output_path, voice_id=None, speaker=None):
    """Generate TTS using local Fish Speech server."""
    if not HAS_HTTPX:
        raise RuntimeError("httpx not installed")

    try:
        payload = {
            "text": text,
            "format": "wav",
            "model": "fish-speech-1.4"
        }
        # Fish Speech voice reference or speaker
        if voice_id:
            payload["voice"] = voice_id
        elif speaker:
            payload["reference_id"] = speaker.lower()

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

        return True
    except Exception as e:
        raise RuntimeError(f"Fish Speech failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE TTS (FALLBACK)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _edge_tts_generate_async(text, output_path, voice, rate="+0%", pitch="+0Hz", volume="+0%"):
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

    return True


def _edge_tts_generate(text, output_path, voice, rate="+0%", pitch="+0Hz", volume="+0%"):
    """Synchronous wrapper for Edge TTS."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        _edge_tts_generate_async(text, output_path, voice, rate, pitch, volume)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TTS GENERATION WORKER (for parallel execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_clip(seg, cast_map, clips_dir, fish_available):
    """
    Generate a single TTS clip with Fish Speech PRIMARY, Edge TTS FALLBACK.
    Returns (success, clip_info_dict) tuple.
    """
    text = seg.get("tts_text", seg.get("dubbed_text", "")).strip()
    if not text:
        return False, {"id": seg["id"], "skipped": "empty_text"}

    speaker = seg.get("speaker", "NARRATOR")
    profile = cast_map.get(speaker, cast_map.get("NARRATOR", {}))
    voice = profile.get("voice", "hi-IN-MadhurNeural")
    rate = profile.get("rate", "+0%")
    pitch = profile.get("pitch", "+0Hz")
    volume = profile.get("volume", "+0%")

    # Generate clip filename
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    clip_name = f"clip_{seg['id']:04d}_{speaker}_{text_hash}.wav"
    clip_path = os.path.join(clips_dir, clip_name)

    # Check cache
    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 500:
        dur = _get_dur(clip_path)
        if dur > 0:
            return True, {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "clip_path": clip_path,
                "actual_dur": dur,
                "speaker": speaker,
                "text": text[:50],
                "tts_group": seg.get("tts_group"),
                "cached": True,
            }

    # ── Generate TTS: Fish Speech PRIMARY → Edge TTS FALLBACK ──────
    generated = False
    last_error = None
    engine_used = None

    for attempt in range(MAX_RETRIES):
        attempt_text = text if attempt == 0 else _simplify_text(text)

        # Try Fish Speech FIRST (primary)
        if fish_available:
            try:
                _fish_generate(attempt_text, clip_path, voice_id=speaker.lower(), speaker=speaker)
                generated = True
                engine_used = "fish"
                break
            except Exception as e:
                last_error = str(e)
                logger.debug(f"[TTS] Fish Speech failed seg {seg['id']} attempt {attempt+1}: {e}")

        # Try Edge TTS as FALLBACK (when Fish unavailable or failed)
        if not generated and HAS_EDGE:
            try:
                # Use .mp3 extension for Edge, will convert
                temp_path = clip_path.replace(".wav", ".mp3")
                _edge_tts_sync(attempt_text, temp_path, voice, rate, pitch, volume)
                generated = True
                engine_used = "edge"
                break
            except Exception as e:
                last_error = str(e)
                logger.debug(f"[TTS] Edge TTS failed seg {seg['id']} attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(0.5 * (attempt + 1))  # Backoff

    if not generated:
        logger.warning(f"[TTS] ✗ Failed segment {seg['id']}: {text[:40]} | Error: {last_error}")
        return False, {"id": seg["id"], "skipped": "generation_failed", "error": last_error}

    # ── Normalize to target LUFS ────────────────────────────────────
    normalized_path = clip_path.replace(".wav", "_norm.wav")
    if engine_used == "edge":
        # Convert Edge output to wav for normalization
        temp_wav = clip_path.replace(".wav", "_temp.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", clip_path.replace(".wav", ".mp3"),
            "-ar", "44100", "-ac", "1", temp_wav
        ], capture_output=True, timeout=30)
        os.remove(clip_path.replace(".wav", ".mp3"))
        clip_source = temp_wav
    else:
        clip_source = clip_path

    if _normalize_clip(clip_source if os.path.exists(clip_source) else clip_path, normalized_path):
        final_path = normalized_path
    else:
        final_path = clip_path if engine_used != "edge" else clip_path.replace(".wav", "_temp.wav")

    dur = _get_dur(final_path)
    target_dur = seg["end"] - seg["start"]

    if dur > target_dur * 1.3 and target_dur > 0.5:
        logger.debug(
            f"[TTS] ⚠ Clip {seg['id']} is {dur:.1f}s but window is {target_dur:.1f}s "
            f"(will be time-stretched)"
        )

    return True, {
        "id": seg["id"],
        "start": seg["start"],
        "end": seg["end"],
        "clip_path": final_path,
        "actual_dur": dur,
        "speaker": speaker,
        "mood": seg.get("mood", "neutral"),
        "text": text[:50],
        "tts_group": seg.get("tts_group"),
        "engine": engine_used,
        "cached": False,
    }


def _edge_tts_sync(text, output_path, voice, rate="+0%", pitch="+0Hz", volume="+0%"):
    """Synchronous wrapper for Edge TTS."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        _edge_tts_generate_async(text, output_path, voice, rate, pitch, volume)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN GENERATION PIPELINE (PARALLEL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(segments, work_dir, cast_map, target_lang="Hindi"):
    """
    Generate TTS clips for all segments with:
    - Fish Speech PRIMARY (local GPU) → Edge TTS FALLBACK
    - PARALLEL clip generation (ThreadPoolExecutor)
    - Voice profile locking per character
    - Per-clip loudness normalization
    - Content-hash caching
    """
    clips_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(clips_dir, exist_ok=True)

    fish_available = _fish_available()
    primary_engine = "Fish Speech (local GPU)" if fish_available else "Edge TTS (cloud)"
    fallback_engine = "Edge TTS" if fish_available else "N/A"

    logger.info(f"[TTS] ════════════════════════════════════════════")
    logger.info(f"[TTS] Primary:   {primary_engine}")
    logger.info(f"[TTS] Fallback:  {fallback_engine}")
    logger.info(f"[TTS] Workers:   {TTS_WORKERS} parallel threads")
    logger.info(f"[TTS] Segments:  {len(segments)} clips")
    logger.info(f"[TTS] Profiles:  {len(cast_map)} voice mappings")
    logger.info(f"[TTS] ════════════════════════════════════════════")

    if not fish_available and not HAS_EDGE:
        raise RuntimeError("[TTS] No TTS engine available! Install edge-tts or start Fish Speech server.")

    manifest = []
    stats = {"ok": 0, "fail": 0, "cached": 0}
    lock = Lock()

    t0 = time.time()

    # ── Parallel Generation ─────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=TTS_WORKERS) as executor:
        futures = {
            executor.submit(_generate_clip, seg, cast_map, clips_dir, fish_available): seg
            for seg in segments
        }

        for i, future in enumerate(as_completed(futures)):
            success, result = future.result()

            if success:
                with lock:
                    manifest.append(result)
                    if result.get("cached"):
                        stats["cached"] += 1
                    else:
                        stats["ok"] += 1
            else:
                with lock:
                    stats["fail"] += 1

            # Progress every 20 clips
            done = i + 1
            if done % 20 == 0 or done == len(segments):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[TTS] Progress: {done}/{len(segments)} | "
                    f"OK: {stats['ok']} | Cached: {stats['cached']} | Failed: {stats['fail']} | "
                    f"Rate: {rate:.1f} clips/sec"
                )

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0

    logger.info(f"[TTS] ════════════════════════════════════════════")
    logger.info(f"[TTS] ✓ Generation complete in {elapsed:.1f}s")
    logger.info(f"[TTS]   Generated:  {stats['ok']} clips")
    logger.info(f"[TTS]   Cached:     {stats['cached']} clips")
    logger.info(f"[TTS]   Failed:     {stats['fail']} clips")
    logger.info(f"[TTS]   Throughput: {len(segments)/elapsed:.1f} clips/sec")
    logger.info(f"[TTS] ════════════════════════════════════════════")

    # Duration analysis
    if manifest:
        total_clip_dur = sum(c.get("actual_dur", 0) for c in manifest)
        total_target_dur = sum(c["end"] - c["start"] for c in manifest)
        logger.info(f"[TTS] Total clip duration: {total_clip_dur:.1f}s | Target: {total_target_dur:.1f}s")

        overflows = sum(1 for c in manifest if c.get("actual_dur", 0) > (c["end"] - c["start"]) * 1.3)
        underflows = sum(1 for c in manifest if c.get("actual_dur", 0) < (c["end"] - c["start"]) * 0.6)
        if overflows:
            logger.info(f"[TTS] ⚠ {overflows} clips overflow (will be stretched)")
        if underflows:
            logger.info(f"[TTS] ⚠ {underflows} clips underflow (will be padded)")

    # Sort manifest by segment ID for consistent ordering
    manifest.sort(key=lambda c: c["id"])

    # Save manifest
    mp = os.path.join(work_dir, "tts_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
