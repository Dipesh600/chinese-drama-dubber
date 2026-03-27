"""
TTS ENGINE v5 — LOCAL Fish Speech on Colab GPU + Edge TTS fallback.
NO API KEYS NEEDED. Fish Speech runs on the T4 GPU directly.

Architecture:
1. Fish Speech local server on localhost:8080 (start in Colab notebook)
2. Edge TTS as reliable fallback (no GPU, no key)
3. Emotion-aware generation using segment mood data
4. Smart speed matching to original segment duration
"""
import os, json, logging, asyncio, subprocess, time
logger = logging.getLogger(__name__)

# Fix for Google Colab event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    try:
        subprocess.run(["pip", "install", "nest_asyncio"], capture_output=True, timeout=30)
        import nest_asyncio
        nest_asyncio.apply()
    except Exception:
        pass

def _run_async(coro):
    """Run async coroutine, compatible with Colab and normal Python."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FISH SPEECH LOCAL (runs on Colab T4 GPU — no API key!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FISH_LOCAL_URL = "http://localhost:8080"

def _check_fish_server():
    """Check if Fish Speech local server is running."""
    try:
        import httpx
        r = httpx.get(f"{FISH_LOCAL_URL}/", timeout=2.0)
        return r.status_code < 500
    except Exception:
        return False

_fish_available = None  # Cached check

def _is_fish_available():
    """Check Fish Speech availability (cached)."""
    global _fish_available
    if _fish_available is None:
        _fish_available = _check_fish_server()
    return _fish_available


def _fish_local_generate(text, output_path, emotion=None, reference_audio=None):
    """
    Generate TTS using Fish Speech running LOCALLY on Colab GPU.
    No API key needed — the model runs on the T4.

    Args:
        text: Text to speak
        output_path: Where to save the audio
        emotion: Optional mood tag (happy, sad, angry, etc.)
        reference_audio: Optional reference audio path for voice style
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import httpx
    except ImportError:
        return False

    if not _is_fish_available():
        return False

    # Add emotion/prosody hints if available
    if emotion and emotion != "neutral":
        emotion_prefix = {
            "happy": "(happily) ", "excited": "(excitedly) ",
            "sad": "(sadly) ", "emotional": "(emotionally) ",
            "angry": "(angrily) ", "tense": "(in a tense voice) ",
            "wise": "(wisely) ", "gentle": "(gently) ",
            "urgent": "(urgently) ", "fearful": "(fearfully) ",
            "romantic": "(romantically) ", "humorous": "(humorously) ",
        }
        prefix = emotion_prefix.get(emotion, "")
        text = prefix + text

    try:
        # Build request for the local Fish Speech server
        payload = {
            "text": text,
            "format": "wav",
            "streaming": False,
        }

        # If a reference audio is provided (for voice style/cloning)
        if reference_audio and os.path.exists(reference_audio):
            import base64
            with open(reference_audio, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            payload["references"] = [{"audio": audio_b64, "text": ""}]

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{FISH_LOCAL_URL}/v1/tts",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.debug(f"[TTS-FISH] Local server returned {response.status_code}")
                return False

    except Exception as e:
        logger.debug(f"[TTS-FISH] Local inference failed: {e}")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE TTS (FALLBACK — always works, no GPU needed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EDGE_ROLE_STYLES = {
    "NARRATOR":    {"rate": "+5%",  "pitch": "+0Hz"},
    "FATHER":      {"rate": "-5%",  "pitch": "-10Hz"},
    "MOTHER":      {"rate": "+0%",  "pitch": "+0Hz"},
    "OLD_MAN":     {"rate": "-12%", "pitch": "-18Hz"},
    "OLD_WOMAN":   {"rate": "-8%",  "pitch": "-5Hz"},
    "VILLAIN":     {"rate": "-8%",  "pitch": "-15Hz"},
    "HERO":        {"rate": "+5%",  "pitch": "+5Hz"},
    "HEROINE":     {"rate": "+5%",  "pitch": "+5Hz"},
    "YOUNG_MAN":   {"rate": "+10%", "pitch": "+5Hz"},
    "YOUNG_WOMAN": {"rate": "+8%",  "pitch": "+5Hz"},
    "BOY":         {"rate": "+12%", "pitch": "+12Hz"},
    "GIRL":        {"rate": "+10%", "pitch": "+8Hz"},
    "SON":         {"rate": "+8%",  "pitch": "+8Hz"},
    "DAUGHTER":    {"rate": "+5%",  "pitch": "+5Hz"},
    "CHAR_A":      {"rate": "+0%",  "pitch": "+0Hz"},
    "CHAR_B":      {"rate": "+0%",  "pitch": "+0Hz"},
    "CHAR_C":      {"rate": "-8%",  "pitch": "-8Hz"},
    "CHAR_D":      {"rate": "+8%",  "pitch": "+8Hz"},
}

async def _edge_tts_generate(text, voice, rate, pitch, path):
    """Generate TTS using Edge TTS."""
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await comm.save(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DURATION UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_dur(path):
    """Get audio duration using ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except:
        return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN GENERATE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(segments, cast_map, work_dir, target_lang="Hindi",
             voice_catalog_map=None, use_fish_audio=True):
    """
    Generate TTS clips for all segments.

    Priority:
    1. Fish Speech LOCAL (runs on Colab T4 GPU — no API key)
    2. Edge TTS fallback (cloud, always works)

    Args:
        segments: TTS segments with text, timing, speaker, mood
        cast_map: Edge TTS voice strings for fallback
        work_dir: Working directory for output
        target_lang: Target language
        voice_catalog_map: Optional voice catalog entries
        use_fish_audio: Whether to try Fish Speech local first
    """
    tts_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(tts_dir, exist_ok=True)

    manifest = []
    fish_local_ok = use_fish_audio and _is_fish_available()
    fish_used = 0
    edge_used = 0
    failed = 0

    if fish_local_ok:
        logger.info(f"[TTS] 🐟 Fish Speech LOCAL server detected — running on GPU!")
        logger.info(f"[TTS] 🐟 No API key needed. High quality mode.")
    else:
        if use_fish_audio:
            logger.info(f"[TTS] ⚠ Fish Speech server not found at {FISH_LOCAL_URL}")
            logger.info(f"[TTS] 💡 To enable: run the Fish Speech setup cell in the notebook")
        logger.info(f"[TTS] 🔊 Using Edge TTS (fallback mode)")

    for idx, seg in enumerate(segments):
        seg_id = seg.get("id", idx)
        text = seg.get("dubbed_text", seg.get("text", "")).strip()
        speaker = seg.get("speaker", "NARRATOR")
        mood = seg.get("mood", "neutral")

        if not text:
            continue

        t_start = seg["start"]
        t_end = seg["end"]
        t_dur = t_end - t_start

        # Calculate max allowed duration
        max_win = t_dur
        if idx + 1 < len(segments):
            gap = segments[idx + 1]["start"] - t_end
            max_win = t_dur + min(gap * 0.6, 1.0)
        else:
            max_win = t_dur + 2.0

        raw = os.path.join(tts_dir, f"s{seg_id:05d}_raw.mp3")
        wav = os.path.join(tts_dir, f"s{seg_id:05d}.wav")

        # ── Step 1: Try Fish Speech LOCAL ───────────────────────
        generated = False

        if fish_local_ok:
            success = _fish_local_generate(text, raw, emotion=mood)
            if success and os.path.exists(raw) and os.path.getsize(raw) > 100:
                generated = True
                fish_used += 1

        # ── Step 2: Edge TTS Fallback ───────────────────────────
        if not generated:
            # Get Edge TTS voice
            if voice_catalog_map and speaker in voice_catalog_map:
                from voice_catalog import get_edge_fallback
                voice = get_edge_fallback(speaker, target_lang)
            elif isinstance(cast_map.get(speaker), str):
                voice = cast_map.get(speaker, cast_map.get("NARRATOR", "hi-IN-MadhurNeural"))
            else:
                voice = "hi-IN-MadhurNeural"

            style = EDGE_ROLE_STYLES.get(speaker, {"rate": "+0%", "pitch": "+0Hz"})
            rate, pitch = style["rate"], style["pitch"]

            try:
                _run_async(asyncio.wait_for(
                    _edge_tts_generate(text, voice, rate, pitch, raw), timeout=15.0
                ))
                if os.path.exists(raw) and os.path.getsize(raw) > 100:
                    generated = True
                    edge_used += 1
            except Exception as e:
                logger.warning(f"[TTS] seg {seg_id} Edge TTS failed ({e}), retrying plain...")
                try:
                    import edge_tts
                    _run_async(asyncio.wait_for(
                        edge_tts.Communicate(text, voice).save(raw), timeout=15.0
                    ))
                    if os.path.exists(raw) and os.path.getsize(raw) > 100:
                        generated = True
                        edge_used += 1
                except Exception as e2:
                    logger.error(f"[TTS] seg {seg_id} completely failed: {e2}")

        if not generated:
            failed += 1
            continue

        # ── Step 3: Speed Adjust + Trim ─────────────────────────
        actual = _get_dur(raw)
        if actual <= 0:
            failed += 1
            continue

        speed = max(0.82, min(1.3, actual / max(t_dur, 0.1)))
        adj = actual / speed
        trim = f",atrim=end={max_win - 0.06:.3f}" if adj > max_win - 0.06 else ""

        cmd = [
            "ffmpeg", "-y", "-i", raw,
            "-af", f"atempo={speed}{trim},afade=t=out:st={max(0, adj-0.08):.3f}:d=0.08",
            "-ar", "44100", "-ac", "1", wav
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)

        if not os.path.exists(wav) or os.path.getsize(wav) < 100:
            failed += 1
            continue

        clip_dur = _get_dur(wav)
        manifest.append({
            "id": seg_id, "start": t_start, "end": t_end,
            "target_dur": t_dur, "actual_dur": round(clip_dur, 3),
            "speaker": speaker, "mood": mood,
            "text": text, "clip_path": wav,
            "tts_group": seg.get("tts_group"),
            "engine": "fish_speech_local" if fish_used > edge_used else "edge_tts",
        })

    # Save manifest
    mp = os.path.join(work_dir, "tts_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"[TTS] {'━' * 50}")
    logger.info(f"[TTS] ✓ Generated: {len(manifest)}/{len(segments)} clips")
    if fish_used > 0:
        logger.info(f"[TTS] 🐟 Fish Speech LOCAL: {fish_used} clips (GPU, no API key)")
    if edge_used > 0:
        logger.info(f"[TTS] 🔊 Edge TTS:          {edge_used} clips (fallback)")
    if failed > 0:
        logger.warning(f"[TTS] ❌ Failed:            {failed} clips")
    logger.info(f"[TTS] {'━' * 50}")

    return {"manifest": manifest, "manifest_path": mp,
            "fish_used": fish_used, "edge_used": edge_used, "failed": failed}
