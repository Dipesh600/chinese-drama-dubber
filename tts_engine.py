"""TTS Engine v3 — Edge TTS with sentence-level units, rate/pitch per character, overlap prevention."""
import os, json, logging, asyncio, subprocess
logger = logging.getLogger(__name__)

# Rate/pitch variation per role for voice differentiation
ROLE_STYLES = {
    "NARRATOR":    {"rate":"+5%",  "pitch":"+0Hz"},
    "FATHER":      {"rate":"-5%",  "pitch":"-10Hz"},
    "MOTHER":      {"rate":"+0%",  "pitch":"+0Hz"},
    "OLD_MAN":     {"rate":"-12%", "pitch":"-18Hz"},
    "OLD_WOMAN":   {"rate":"-8%",  "pitch":"-5Hz"},
    "VILLAIN":     {"rate":"-8%",  "pitch":"-15Hz"},
    "HERO":        {"rate":"+5%",  "pitch":"+5Hz"},
    "HEROINE":     {"rate":"+5%",  "pitch":"+5Hz"},
    "YOUNG_MAN":   {"rate":"+10%", "pitch":"+5Hz"},
    "YOUNG_WOMAN": {"rate":"+8%",  "pitch":"+5Hz"},
    "BOY":         {"rate":"+12%", "pitch":"+12Hz"},
    "GIRL":        {"rate":"+10%", "pitch":"+8Hz"},
    "SON":         {"rate":"+8%",  "pitch":"+8Hz"},
    "DAUGHTER":    {"rate":"+5%",  "pitch":"+5Hz"},
    "CHAR_A":      {"rate":"+0%",  "pitch":"+0Hz"},
    "CHAR_B":      {"rate":"+0%",  "pitch":"+0Hz"},
    "CHAR_C":      {"rate":"-8%",  "pitch":"-8Hz"},
    "CHAR_D":      {"rate":"+8%",  "pitch":"+8Hz"},
}

async def _gen_async(text, voice, rate, pitch, path):
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await comm.save(path)

def _get_dur(path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",path], capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 0.0

def generate(segments, cast_map, work_dir, target_lang="Hindi"):
    """Generate TTS for sentence-merged segments."""
    tts_dir = os.path.join(work_dir, "tts_clips")
    os.makedirs(tts_dir, exist_ok=True)
    manifest = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        # Use tts_text (sentence-merged) if available, else dubbed_text
        text = seg.get("tts_text", seg.get("dubbed_text", "")).strip()
        if not text or len(text) < 2:
            continue

        spk = seg.get("speaker", "NARRATOR")
        voice = cast_map.get(spk, cast_map.get("NARRATOR", list(cast_map.values())[0]))
        style = ROLE_STYLES.get(spk, ROLE_STYLES.get("NARRATOR", {"rate":"+0%","pitch":"+0Hz"}))
        rate, pitch = style["rate"], style["pitch"]

        seg_id = seg["id"]
        t_dur = seg.get("tts_duration", round(seg["end"] - seg["start"], 3))

        # Max window = gap until next segment
        if idx + 1 < len(segments):
            max_win = round(segments[idx+1]["start"] - seg["start"] - 0.08, 3)
            max_win = max(max_win, t_dur)
        else:
            max_win = t_dur + 2.0

        raw = os.path.join(tts_dir, f"s{seg_id:05d}_raw.mp3")
        wav = os.path.join(tts_dir, f"s{seg_id:05d}.wav")

        # Generate TTS with timeout
        try:
            asyncio.run(asyncio.wait_for(_gen_async(text, voice, rate, pitch, raw), timeout=15.0))
        except Exception as e:
            logger.warning(f"[TTS] seg {seg_id} with rate/pitch failed ({e}), retrying plain...")
            try:
                import edge_tts
                asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, voice).save(raw), timeout=15.0))
            except Exception as e2:
                logger.error(f"[TTS] seg {seg_id} failed: {e2}")
                continue

        if not os.path.exists(raw) or os.path.getsize(raw) < 100:
            continue

        actual = _get_dur(raw)
        if actual <= 0:
            continue

        # Speed adjust + hard trim to prevent overlap
        speed = max(0.82, min(1.3, actual / max(t_dur, 0.1)))
        adj = actual / speed
        trim = f",atrim=end={max_win - 0.06:.3f}" if adj > max_win - 0.06 else ""

        subprocess.run([
            "ffmpeg", "-y", "-i", raw,
            "-af", f"atempo={speed:.4f},aresample=44100{trim}",
            "-ac", "2", wav
        ], capture_output=True, timeout=20)

        if not os.path.exists(wav):
            continue

        entry = {
            "id": seg_id, "start": seg["start"], "end": seg["end"],
            "target_dur": t_dur, "max_window": max_win,
            "speaker": spk, "voice": voice, "rate": rate, "pitch": pitch,
            "speed": round(speed, 3), "text": text, "clip_path": wav,
            "tts_merged_count": seg.get("tts_merged_count", 1),
        }
        # Include sub-segment timings for subtitles
        if seg.get("tts_group"):
            entry["tts_group"] = [{
                "id": g["id"], "start": g["start"], "end": g["end"],
                "dubbed_text": g.get("dubbed_text", g.get("text", ""))
            } for g in seg["tts_group"]]
        manifest.append(entry)

        if (idx + 1) % 20 == 0 or idx < 3:
            vn = voice.split("-")[2] if "-" in voice else voice
            mc = seg.get("tts_merged_count", 1)
            merge_tag = f" [{mc}merged]" if mc > 1 else ""
            logger.info(f"[TTS] [{idx+1}/{total}] {spk}({vn} {rate}){merge_tag} {t_dur:.1f}s | {text[:45]}")

    mp = os.path.join(work_dir, "tts_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"[TTS] ✓ {len(manifest)}/{total} clips generated")
    return {"manifest": manifest, "manifest_path": mp}
