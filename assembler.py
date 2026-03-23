"""
ASSEMBLER v3 — Single-pass mix + INTELLIGENT audio ducking + subtitles + video merge.
BG audio: 25-30% in gaps, ducked to 8% during dubbed speech.
"""
import os, json, logging, subprocess
logger = logging.getLogger(__name__)

BG_SPEECH = 0.08   # 8% during dubbed speech
BG_GAP = 0.30      # 30% during gaps (music/ambient comes through)
FADE_MS = 150       # crossfade between duck levels

def _build_bg_envelope(clips, total_dur):
    """Build an ffmpeg volume filter that ducks BG audio during speech."""
    # Generate volume keyframes: high in gaps, low during speech
    events = []
    for c in clips:
        events.append((max(0, c["start"] - 0.1), "duck"))    # duck slightly before speech
        events.append((c["start"] + c.get("target_dur", c["end"] - c["start"]) + 0.05, "raise"))
    
    events.sort(key=lambda x: x[0])
    
    # Build ffmpeg volume expression with enable ranges
    # Use multiple volume filters for each speech region
    filters = []
    duck_ranges = []
    
    i = 0
    while i < len(events):
        if events[i][1] == "duck":
            start = events[i][0]
            # Find matching raise (or next duck)
            end = total_dur
            for j in range(i + 1, len(events)):
                if events[j][1] == "raise":
                    end = events[j][0]
                    break
            duck_ranges.append((max(0, start), min(end, total_dur)))
        i += 1
    
    # Merge overlapping ranges
    merged = []
    for s, e in sorted(duck_ranges):
        if merged and s <= merged[-1][1] + 0.2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    
    if not merged:
        return f"volume={BG_GAP}"
    
    # Build volume expression using enable syntax
    # During speech: volume=BG_SPEECH, during gaps: volume=BG_GAP
    # Use afade between transitions for smoothness
    expr_parts = []
    for s, e in merged:
        expr_parts.append(f"volume=enable='between(t,{s:.2f},{e:.2f})':volume={BG_SPEECH}")
    
    # Alternative: use a single volume expression
    # volume='if(between(t,s1,e1)+between(t,s2,e2)+..., BG_SPEECH, BG_GAP)'
    conditions = "+".join(f"between(t\\,{s:.2f}\\,{e:.2f})" for s, e in merged)
    volume_expr = f"volume='if({conditions}\\,{BG_SPEECH}\\,{BG_GAP})':eval=frame"
    
    return volume_expr

def assemble(manifest, work_dir, video_path, total_dur, preserve_bg=True):
    clips = [c for c in manifest if os.path.exists(c.get("clip_path", ""))]
    if not clips:
        raise RuntimeError("No TTS clips to assemble")
    logger.info(f"[ASSEMBLE] {len(clips)}/{len(manifest)} clips over {total_dur:.0f}s")

    audio_path = os.path.join(work_dir, "audio.mp3")
    n = len(clips)

    # Build dubbed audio track
    inputs = []
    delays = []
    for i, clip in enumerate(clips):
        inputs += ["-i", clip["clip_path"]]
        ms = int(clip["start"] * 1000)
        delays.append(f"[{i}]adelay={ms}|{ms}[d{i}]")
    
    amix = "".join(f"[d{i}]" for i in range(n)) + f"amix=inputs={n}:normalize=0[dubbed]"

    if preserve_bg and os.path.exists(audio_path):
        inputs += ["-i", audio_path]
        bg_idx = n
        
        # Build intelligent ducking envelope
        vol_expr = _build_bg_envelope(clips, total_dur)
        
        filter_complex = (
            ";".join(delays) + ";" +
            amix + ";" +
            f"[{bg_idx}]{vol_expr}[bg];" +
            f"[dubbed][bg]amix=inputs=2:duration=longest:weights=1 0.5[aout]"
        )
        logger.info(f"[ASSEMBLE] 🎵 Smart ducking: {int(BG_GAP*100)}% in gaps → {int(BG_SPEECH*100)}% during speech")
    else:
        filter_complex = ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")

    audio_out = os.path.join(work_dir, "dubbed_audio.wav")
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    
    if r.returncode != 0:
        # Fallback: flat volume
        logger.warning(f"[ASSEMBLE] Smart ducking failed, falling back to flat {int(BG_GAP*100)}%...")
        filter_simple = (
            ";".join(delays) + ";" +
            amix + ";" +
            f"[{bg_idx}]volume={BG_GAP}[bg];" +
            f"[dubbed][bg]amix=inputs=2:duration=longest:weights=1 0.5[aout]"
        ) if preserve_bg and os.path.exists(audio_path) else (
            ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")
        )
        cmd2 = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_simple,
            "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
        ]
        r = subprocess.run(cmd2, capture_output=True, timeout=300)
        if r.returncode != 0:
            # Last fallback: no BG
            logger.warning(f"[ASSEMBLE] BG mix failed entirely, dubbing without BG")
            filter_no_bg = ";".join(delays) + ";" + amix.replace("[dubbed]", "[aout]")
            cmd3 = ["ffmpeg", "-y"] + inputs[:n*2] + [
                "-filter_complex", filter_no_bg,
                "-map", "[aout]", "-ar", "44100", "-ac", "2", audio_out
            ]
            r = subprocess.run(cmd3, capture_output=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg mix failed: {r.stderr[-500:]}")

    sz = os.path.getsize(audio_out) / 1024 / 1024
    logger.info(f"[ASSEMBLE] ✓ dubbed_audio.wav — {sz:.1f}MB")

    # Subtitles
    srt = os.path.join(work_dir, "subtitles.srt")
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60)
        s = int(t % 60); ms = int((t % 1) * 1000)
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
                f.write(f"{idx}\n{fmt(c['start'])} --> {fmt(c['end'])}\n{c['text']}\n\n")
                idx += 1
    logger.info(f"[ASSEMBLE] ✓ {idx-1} subtitle entries")

    # Merge with video
    out = os.path.join(work_dir, "dubbed_output.mp4")
    r2 = subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_out,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", out
    ], capture_output=True, timeout=300)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed: {r2.stderr[-500:]}")

    sz2 = os.path.getsize(out) / 1024 / 1024
    logger.info(f"[MERGE] ✓ {out} ({sz2:.1f}MB)")
    return {"video_path": out, "srt_path": srt, "size_mb": round(sz2, 2)}
