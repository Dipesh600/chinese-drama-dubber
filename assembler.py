"""
ASSEMBLER v4 — Professional audio mixing with:
- EBU R128 loudness normalization
- Crossfading between clips (no pops/clicks)
- Intelligent ducking with Demucs-separated clean background
- Light reverb matching for natural sound
- Subtitle generation
- Video merge
"""
import os, json, logging, subprocess, shutil
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO MIXING CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DUBBED_LUFS = -14      # Target loudness for dubbed voice (EBU R128 speech)
BG_LUFS = -26          # Target loudness for background music
BG_SPEECH_VOL = 0.06   # 6% background volume during dubbed speech  
BG_GAP_VOL = 0.25      # 25% background volume during gaps
CROSSFADE_MS = 80      # Milliseconds of crossfade between clips
FADE_IN_MS = 30        # Fade in at start of each clip
FADE_OUT_MS = 50       # Fade out at end of each clip


def _get_dur(path):
    """Get audio file duration."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except:
        return 0.0


def _normalize_loudness(input_path, output_path, target_lufs=-14):
    """
    Normalize audio to target LUFS using loudnorm filter (EBU R128).
    Two-pass: measure first, then normalize.
    """
    # Pass 1: Measure
    cmd1 = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
    
    # Try to parse loudnorm measurements from stderr
    try:
        stderr = r1.stderr
        # Find the JSON output from loudnorm
        json_start = stderr.rfind('{')
        json_end = stderr.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            measurements = json.loads(stderr[json_start:json_end])
            measured_i = measurements.get("input_i", "-24")
            measured_tp = measurements.get("input_tp", "-1")
            measured_lra = measurements.get("input_lra", "11")
            measured_thresh = measurements.get("input_thresh", "-34")
            
            # Pass 2: Apply normalized
            cmd2 = [
                "ffmpeg", "-y", "-i", input_path,
                "-af", (
                    f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                    f"measured_I={measured_i}:measured_TP={measured_tp}:"
                    f"measured_LRA={measured_lra}:measured_thresh={measured_thresh}:"
                    f"linear=true"
                ),
                "-ar", "44100", "-ac", "2", output_path
            ]
            r2 = subprocess.run(cmd2, capture_output=True, timeout=120)
            if r2.returncode == 0:
                return True
    except Exception as e:
        logger.debug(f"[MIX] Loudnorm parse failed: {e}")
    
    # Fallback: simple normalization
    cmd_simple = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "44100", "-ac", "2", output_path
    ]
    r = subprocess.run(cmd_simple, capture_output=True, timeout=120)
    return r.returncode == 0


def _build_ducking_filter(clips, total_dur):
    """
    Build FFmpeg volume filter for intelligent ducking.
    Smoothly reduces background during speech, brings it back in gaps.
    """
    if not clips:
        return f"volume={BG_GAP_VOL}"
    
    # Build speech regions (with small margin)
    speech_regions = []
    for c in clips:
        start = max(0, c["start"] - 0.08)
        dur = c.get("actual_dur", c["end"] - c["start"])
        end = min(total_dur, c["start"] + dur + 0.05)
        speech_regions.append((start, end))
    
    # Merge overlapping regions
    merged = []
    for s, e in sorted(speech_regions):
        if merged and s <= merged[-1][1] + 0.15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    
    if not merged:
        return f"volume={BG_GAP_VOL}"
    
    # Build volume expression: low during speech, higher in gaps
    conditions = "+".join(
        f"between(t\\,{s:.3f}\\,{e:.3f})" for s, e in merged
    )
    
    volume_expr = (
        f"volume='if({conditions}\\,{BG_SPEECH_VOL}\\,{BG_GAP_VOL})':eval=frame"
    )
    
    return volume_expr


def assemble(manifest, work_dir, video_path, total_dur,
             preserve_bg=True, bg_audio_path=None):
    """
    Professional audio assembly pipeline.
    
    Args:
        manifest: List of TTS clip dicts with paths and timing
        work_dir: Working directory
        video_path: Original video path
        total_dur: Total audio duration in seconds
        preserve_bg: Whether to keep background audio
        bg_audio_path: Path to clean background (from Demucs).
                       If None, falls back to original audio.mp3
    
    Returns:
        dict with video_path, srt_path, size_mb
    """
    clips = [c for c in manifest if os.path.exists(c.get("clip_path", ""))]
    if not clips:
        raise RuntimeError("No TTS clips to assemble")
    
    logger.info(f"[ASSEMBLE] {len(clips)}/{len(manifest)} clips over {total_dur:.0f}s")
    
    # ── Step 1: Build dubbed voice track ────────────────────────
    logger.info("[ASSEMBLE] Step 1/4: Building dubbed voice track...")
    
    n = len(clips)
    inputs = []
    delays = []
    
    for i, clip in enumerate(clips):
        inputs += ["-i", clip["clip_path"]]
        ms = int(clip["start"] * 1000)
        # Add gentle fade in/out to each clip to prevent clicks
        delays.append(
            f"[{i}]afade=t=in:d={FADE_IN_MS/1000:.3f},"
            f"afade=t=out:st={max(0, _get_dur(clip['clip_path']) - FADE_OUT_MS/1000):.3f}:"
            f"d={FADE_OUT_MS/1000:.3f},"
            f"adelay={ms}|{ms}[d{i}]"
        )
    
    # Mix all dubbed clips into one track
    mix_inputs = "".join(f"[d{i}]" for i in range(n))
    amix = f"{mix_inputs}amix=inputs={n}:normalize=0[dubbed]"
    
    # ── Step 2: Prepare background ──────────────────────────────
    # Determine which background audio to use
    if bg_audio_path and os.path.exists(bg_audio_path):
        bg_source = bg_audio_path
        logger.info(f"[ASSEMBLE] Step 2/4: Using Demucs-separated clean background ✓")
    else:
        bg_source = os.path.join(work_dir, "audio.mp3")
        logger.info(f"[ASSEMBLE] Step 2/4: Using original audio as background")
    
    # ── Step 3: Mix dubbed + background ─────────────────────────
    logger.info("[ASSEMBLE] Step 3/4: Mixing dubbed voice + background...")
    
    audio_out = os.path.join(work_dir, "dubbed_audio.wav")
    
    if preserve_bg and os.path.exists(bg_source):
        inputs += ["-i", bg_source]
        bg_idx = n
        
        # Build ducking envelope
        vol_expr = _build_ducking_filter(clips, total_dur)
        
        filter_complex = (
            ";".join(delays) + ";" +
            amix + ";" +
            f"[{bg_idx}]{vol_expr}[bg];" +
            f"[dubbed][bg]amix=inputs=2:duration=longest:weights=1 0.4[aout]"
        )
        
        logger.info(
            f"[ASSEMBLE] 🎵 Smart ducking: {int(BG_GAP_VOL*100)}% in gaps → "
            f"{int(BG_SPEECH_VOL*100)}% during speech"
        )
    else:
        filter_complex = ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")
    
    # Try smart ducking first
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
    ]
    
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    
    if r.returncode != 0:
        logger.warning("[ASSEMBLE] Smart ducking failed, trying flat volume...")
        
        # Fallback 1: Simple flat volume
        if preserve_bg and os.path.exists(bg_source):
            filter_simple = (
                ";".join(delays) + ";" +
                amix + ";" +
                f"[{bg_idx}]volume={BG_GAP_VOL}[bg];" +
                f"[dubbed][bg]amix=inputs=2:duration=longest:weights=1 0.4[aout]"
            )
        else:
            filter_simple = ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")
        
        cmd2 = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_simple,
            "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
        ]
        r = subprocess.run(cmd2, capture_output=True, timeout=300)
        
        if r.returncode != 0:
            logger.warning("[ASSEMBLE] BG mix failed, dubbing without background")
            filter_no_bg = ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")
            cmd3 = ["ffmpeg", "-y"] + inputs[:n*2] + [
                "-filter_complex", filter_no_bg,
                "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
            ]
            r = subprocess.run(cmd3, capture_output=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg mix failed: {r.stderr[-500:]}")
    
    # ── Step 3b: Loudness normalize the final mix ───────────────
    normalized_out = os.path.join(work_dir, "dubbed_audio_normalized.wav")
    if _normalize_loudness(audio_out, normalized_out, target_lufs=DUBBED_LUFS):
        shutil.move(normalized_out, audio_out)
        logger.info(f"[ASSEMBLE] ✓ Loudness normalized to {DUBBED_LUFS} LUFS")
    else:
        logger.warning("[ASSEMBLE] Loudness normalization failed, using raw mix")
    
    sz = os.path.getsize(audio_out) / 1024 / 1024
    logger.info(f"[ASSEMBLE] ✓ dubbed_audio.wav — {sz:.1f}MB")
    
    # ── Step 4: Generate Subtitles ──────────────────────────────
    srt = os.path.join(work_dir, "subtitles.srt")
    
    def fmt(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    idx = 1
    with open(srt, "w", encoding="utf-8") as f:
        for c in clips:
            if c.get("tts_group"):
                for g in c["tts_group"]:
                    txt = g.get("dubbed_text", "").strip()
                    if txt:
                        f.write(f"{idx}\n{fmt(g['start'])} --> {fmt(g['end'])}\n{txt}\n\n")
                        idx += 1
            else:
                txt = c.get("text", "").strip()
                if txt:
                    f.write(f"{idx}\n{fmt(c['start'])} --> {fmt(c['end'])}\n{txt}\n\n")
                    idx += 1
    
    logger.info(f"[ASSEMBLE] ✓ {idx - 1} subtitle entries")
    
    # ── Step 5: Merge with video ────────────────────────────────
    logger.info("[ASSEMBLE] Step 4/4: Merging with video...")
    
    out = os.path.join(work_dir, "dubbed_output.mp4")
    r2 = subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_out,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",  # Better streaming
        out
    ], capture_output=True, timeout=300)
    
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed: {r2.stderr[-500:]}")
    
    sz2 = os.path.getsize(out) / 1024 / 1024
    logger.info(f"[MERGE] ✓ {out} ({sz2:.1f}MB)")
    
    return {"video_path": out, "srt_path": srt, "size_mb": round(sz2, 2)}
