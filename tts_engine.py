"""
TTS ENGINE v4 — Fish Audio + Edge TTS with intelligent fallback.
Features:
- Fish Audio API for high-quality, emotional TTS (2M+ voices)
- Edge TTS as reliable fallback (no API key needed)
- Emotion-aware generation using segment mood data
- Smart speed matching to original segment duration
- Colab/Jupyter compatible (nest_asyncio)
"""
import os, json, logging, asyncio, subprocess, time, hashlib
logger = logging.getLogger(__name__)

# Fix for Google Colab — already has a running event loop
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
# FISH AUDIO TTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fish_audio_generate(text, voice_ref_id, output_path, emotion=None):
    """
    Generate TTS using Fish Audio API.
    
    Args:
        text: Text to speak
        voice_ref_id: Fish Audio model/voice reference ID
        output_path: Path to save the generated audio
        emotion: Optional emotion tag (happy, sad, angry, etc.)
    
    Returns:
        True if successful, False otherwise
    """
    api_key = os.environ.get("FISH_AUDIO_API_KEY", "")
    if not api_key:
        return False
    
    try:
        import httpx
    except ImportError:
        try:
            subprocess.run(["pip", "install", "httpx"], capture_output=True, timeout=30)
            import httpx
        except Exception:
            return False
    
    # Add emotion markers if available
    if emotion and emotion != "neutral":
        emotion_map = {
            "happy": "(happily) ", "excited": "(excitedly) ",
            "sad": "(sadly) ", "emotional": "(emotionally) ",
            "angry": "(angrily) ", "tense": "(tensely) ",
            "wise": "(wisely) ", "gentle": "(gently) ",
            "urgent": "(urgently) ", "fearful": "(fearfully) ",
        }
        prefix = emotion_map.get(emotion, "")
        text = prefix + text
    
    try:
        url = "https://api.fish.audio/v1/tts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "reference_id": voice_ref_id,
            "format": "mp3",
            "latency": "normal",
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                logger.warning(f"[TTS-FISH] API error {response.status_code}: {response.text[:100]}")
                return False
    except Exception as e:
        logger.warning(f"[TTS-FISH] Request failed: {e}")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE TTS (FALLBACK)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Rate/pitch variation per role for Edge TTS voice differentiation
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
    
    Architecture:
    1. Try Fish Audio API (if API key available + voice catalog mapped)
    2. Fall back to Edge TTS (always available, no API key)
    3. Speed-adjust clips to match original timing
    4. Trim clips to prevent overlap with next segment
    
    Args:
        segments: TTS segments with text, timing, speaker, mood
        cast_map: Either voice_catalog entries (v2) or Edge TTS voice strings (v1)
        work_dir: Working directory for output
        target_lang: Target language
        voice_catalog_map: Optional mapping from voice_catalog.match_voices()
        use_fish_audio: Whether to try Fish Audio first
    
    Returns:
        dict with manifest of generated clips
    """
    tts_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(tts_dir, exist_ok=True)
    
    manifest = []
    fish_audio_available = bool(os.environ.get("FISH_AUDIO_API_KEY"))
    fish_used = 0
    edge_used = 0
    failed = 0
    
    if use_fish_audio and fish_audio_available:
        logger.info(f"[TTS] 🐟 Fish Audio API available — high quality mode")
    else:
        logger.info(f"[TTS] Using Edge TTS (fallback mode)")
    
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
        
        # Calculate max allowed duration (prevent overlap)
        max_win = t_dur
        if idx + 1 < len(segments):
            gap = segments[idx + 1]["start"] - t_end
            max_win = t_dur + min(gap * 0.6, 1.0)
        else:
            max_win = t_dur + 2.0
        
        raw = os.path.join(tts_dir, f"s{seg_id:05d}_raw.mp3")
        wav = os.path.join(tts_dir, f"s{seg_id:05d}.wav")
        
        # ── Step 1: Try Fish Audio ──────────────────────────────
        generated = False
        
        if use_fish_audio and fish_audio_available and voice_catalog_map:
            voice_entry = voice_catalog_map.get(speaker, voice_catalog_map.get("NARRATOR"))
            if voice_entry:
                from voice_catalog import get_fish_ref
                fish_ref = get_fish_ref(voice_entry, target_lang)
                
                if fish_ref:
                    success = _fish_audio_generate(text, fish_ref, raw, emotion=mood)
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
                logger.warning(f"[TTS] seg {seg_id} Edge TTS with style failed ({e}), retrying plain...")
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
            "engine": "fish_audio" if generated and fish_used > edge_used else "edge_tts",
        })
    
    # Save manifest
    mp = os.path.join(work_dir, "tts_manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[TTS] {'━' * 50}")
    logger.info(f"[TTS] ✓ Generated: {len(manifest)}/{len(segments)} clips")
    if fish_used > 0:
        logger.info(f"[TTS] 🐟 Fish Audio: {fish_used} clips (high quality)")
    if edge_used > 0:
        logger.info(f"[TTS] 🔊 Edge TTS:   {edge_used} clips (fallback)")
    if failed > 0:
        logger.warning(f"[TTS] ❌ Failed:     {failed} clips")
    logger.info(f"[TTS] {'━' * 50}")
    
    return {"manifest": manifest, "manifest_path": mp,
            "fish_used": fish_used, "edge_used": edge_used, "failed": failed}
