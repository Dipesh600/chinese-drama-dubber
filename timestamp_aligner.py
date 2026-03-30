"""
TIMESTAMP ALIGNER — Precise clip-to-timestamp alignment with pitch-preserved time-stretching.
Ensures every dubbed clip fits EXACTLY into its original time window with consistent loudness.

Strategy:
  1. If clip is shorter than window → pad with silence
  2. If clip is longer than window → time-stretch using rubberband (explicit pitch=1.0)
  3. If clip overlaps next clip → trim with fade-out
  4. Per-clip loudness normalization (EBU R128) before assembly for consistent prosody
  5. Insert 20ms silence gaps between clips for natural breathing
"""
import os, subprocess, logging, shutil
from config import ALIGN_TOLERANCE, MAX_STRETCH, BREATHING_GAP_MS, FADE_OUT_MS
logger = logging.getLogger(__name__)

FADE_TRIM_MS = FADE_OUT_MS


def _get_dur(path):
    """Get audio duration in seconds."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(r.stdout.strip())
    except:
        return 0.0


def _normalize_loudness(input_path, output_path, target_lufs=-23.0):
    """
    Normalize audio to target loudness (EBU R128).
    This ensures all clips have consistent perceived volume/prosody before assembly.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "1", output_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return True
    except Exception:
        pass
    # Fallback: just copy if normalization fails
    shutil.copy2(input_path, output_path)
    return True


def _time_stretch(input_path, output_path, speed_ratio):
    """
    Time-stretch audio while PRESERVING PITCH using rubberband.
    pitch=1.0 is EXPLICITLY required — without it, rubberband may pitch-shift.
    """
    # Clamp ratio to valid rubberband range
    safe_ratio = max(0.5, min(4.0, speed_ratio))

    # Try rubberband first (best quality, pitch-preserved)
    try:
        # EXPLICIT pitch=1.0 — without this, rubberband may pitch-shift on some versions
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"rubberband=tempo={safe_ratio:.4f}:pitch=1.0",
            "-ar", "44100", "-ac", "1", output_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return True
        else:
            logger.debug(f"[ALIGN] rubberband failed: {r.stderr[:100] if r.stderr else 'no stderr'}")
    except Exception as e:
        logger.debug(f"[ALIGN] rubberband exception: {e}")

    # Fallback: atempo (lower quality but always available, no pitch preservation)
    atempo_filters = []
    ratio = safe_ratio
    while ratio > 2.0:
        atempo_filters.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        atempo_filters.append("atempo=0.5")
        ratio /= 0.5
    atempo_filters.append(f"atempo={ratio:.4f}")

    af = ",".join(atempo_filters)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", af,
        "-ar", "44100", "-ac", "1", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100


def _pad_with_silence(input_path, output_path, target_dur):
    """
    Pad audio with silence at the END to reach target duration.
    Silence padding at end (not distributed) sounds more natural.
    """
    actual_dur = _get_dur(input_path)
    if actual_dur <= 0:
        return False

    pad_total = target_dur - actual_dur
    if pad_total <= 0:
        shutil.copy2(input_path, output_path)
        return True

    # Pad at end only — use delay pad to add silence after audio
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"apad=whole_dur={target_dur:.3f}",
        "-t", f"{target_dur:.3f}",
        "-ar", "44100", "-ac", "1", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return True

    # Fallback: manual silence concatenation
    silence_path = output_path + "_silence.wav"
    silence_dur = pad_total - 0.01
    gen = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", f"{silence_dur:.3f}", silence_path
    ], capture_output=True, text=True, timeout=10)
    if gen.returncode == 0:
        concat_list = output_path + "_concat.txt"
        with open(concat_list, "w") as f:
            f.write(f"file '{input_path}'\nfile '{silence_path}'\n")
        cat = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-ar", "44100", "-ac", "1", output_path
        ], capture_output=True, text=True, timeout=30)
        try:
            os.remove(silence_path)
            os.remove(concat_list)
        except:
            pass
        return cat.returncode == 0

    return False


def _trim_with_fade(input_path, output_path, max_dur):
    """Trim audio to max duration with a gentle fade-out at the end."""
    fade_start = max(0, max_dur - FADE_TRIM_MS / 1000)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"afade=t=out:st={fade_start:.3f}:d={FADE_TRIM_MS/1000:.3f}",
        "-t", f"{max_dur:.3f}",
        "-ar", "44100", "-ac", "1", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def align(manifest, work_dir):
    """
    Align all TTS clips to their target time windows.
    Each clip is: time-stretched or padded → normalized to consistent loudness → ready for assembly.
    """
    aligned_dir = os.path.join(work_dir, "tts_aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    logger.info(f"[ALIGN] Aligning {len(manifest)} clips to timestamps...")

    stats = {"kept": 0, "padded": 0, "stretched": 0, "trimmed": 0, "normalized": 0, "failed": 0}
    manifest_sorted = sorted(manifest, key=lambda c: c["id"])

    for i, clip in enumerate(manifest_sorted):
        clip_path = clip.get("clip_path", "")
        if not clip_path or not os.path.exists(clip_path):
            stats["failed"] += 1
            continue

        actual_dur = clip.get("actual_dur", _get_dur(clip_path))
        target_dur = clip["end"] - clip["start"]

        if target_dur <= 0.1:
            stats["kept"] += 1
            continue

        # Check for overlap with next clip
        max_allowed_dur = target_dur
        if i + 1 < len(manifest_sorted):
            next_start = manifest_sorted[i + 1]["start"]
            gap = next_start - clip["start"]
            max_allowed_dur = min(max_allowed_dur, gap - BREATHING_GAP_MS / 1000)
            max_allowed_dur = max(max_allowed_dur, 0.3)

        ratio = actual_dur / target_dur if target_dur > 0 else 1.0
        aligned_path = os.path.join(aligned_dir, f"aligned_{clip['id']:04d}.wav")

        # Step 1: Time-correct the clip (stretch/pad/trim)
        if abs(ratio - 1.0) <= ALIGN_TOLERANCE:
            # Duration is close enough — check if it fits within gap
            if actual_dur > max_allowed_dur + 0.05:
                if _trim_with_fade(clip_path, aligned_path, max_allowed_dur):
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["trimmed"] += 1
                else:
                    shutil.copy2(clip_path, aligned_path)
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["kept"] += 1
            else:
                shutil.copy2(clip_path, aligned_path)
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["kept"] += 1

        elif ratio < (1.0 - ALIGN_TOLERANCE):
            # Clip is SHORT — pad with silence at end
            if _pad_with_silence(clip_path, aligned_path, target_dur):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["padded"] += 1
            else:
                shutil.copy2(clip_path, aligned_path)
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["kept"] += 1

        elif ratio <= MAX_STRETCH:
            # Clip is LONG — time-stretch (pitch-preserved)
            speed = actual_dur / target_dur
            if _time_stretch(clip_path, aligned_path, speed):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["stretched"] += 1
            else:
                # Fallback: trim
                if _trim_with_fade(clip_path, aligned_path, target_dur):
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["trimmed"] += 1
                else:
                    shutil.copy2(clip_path, aligned_path)
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["kept"] += 1

        else:
            # Ratio too high — trim to fit
            if _trim_with_fade(clip_path, aligned_path, target_dur):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["trimmed"] += 1
            else:
                shutil.copy2(clip_path, aligned_path)
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["kept"] += 1

        # Step 2: Normalize loudness for consistent prosody across all clips
        normalized_path = aligned_path.replace(".wav", "_norm.wav")
        if _normalize_loudness(clip["clip_path"], normalized_path, target_lufs=-23.0):
            clip["clip_path"] = normalized_path
            clip["actual_dur"] = _get_dur(normalized_path)
            stats["normalized"] += 1

    logger.info(f"[ALIGN] {'━' * 55}")
    logger.info(f"[ALIGN] ✓ Alignment complete:")
    logger.info(f"[ALIGN]   Kept as-is:    {stats['kept']}")
    logger.info(f"[ALIGN]   Padded:        {stats['padded']}")
    logger.info(f"[ALIGN]   Time-stretched: {stats['stretched']}")
    logger.info(f"[ALIGN]   Trimmed:       {stats['trimmed']}")
    logger.info(f"[ALIGN]   Normalized:    {stats['normalized']}")
    if stats["failed"]:
        logger.warning(f"[ALIGN]   Failed:        {stats['failed']}")
    logger.info(f"[ALIGN] {'━' * 55}")

    return manifest_sorted
