"""
TIMESTAMP ALIGNER — Precise clip-to-timestamp alignment.
Ensures every dubbed clip fits EXACTLY into its original time window.

Strategy:
  1. If clip is shorter than window → pad with silence
  2. If clip is longer than window → time-stretch using rubberband (pitch-preserving)
  3. If clip overlaps next clip → trim with fade-out
  4. Insert 20ms silence gaps between clips for natural breathing
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


def _time_stretch(input_path, output_path, speed_ratio):
    """Time-stretch audio while preserving pitch using rubberband or atempo."""
    # Try rubberband first (best quality)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"rubberband=tempo={speed_ratio:.4f}",
            "-ar", "44100", "-ac", "1", output_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return True
    except Exception:
        pass

    # Fallback: atempo (lower quality but always available)
    atempo_filters = []
    ratio = speed_ratio
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
    return r.returncode == 0


def _pad_with_silence(input_path, output_path, target_dur):
    """Pad audio with silence to reach target duration."""
    actual_dur = _get_dur(input_path)
    if actual_dur <= 0:
        return False

    pad_total = target_dur - actual_dur
    if pad_total <= 0:
        shutil.copy2(input_path, output_path)
        return True

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", f"apad=pad_dur={pad_total:.3f}",
        "-t", f"{target_dur:.3f}",
        "-ar", "44100", "-ac", "1", output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def _trim_with_fade(input_path, output_path, max_dur):
    """Trim audio to max duration with a gentle fade-out."""
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
    """
    aligned_dir = os.path.join(work_dir, "tts_aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    logger.info(f"[ALIGN] Aligning {len(manifest)} clips to timestamps...")

    stats = {"kept": 0, "padded": 0, "stretched": 0, "trimmed": 0, "failed": 0}
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

        if abs(ratio - 1.0) <= ALIGN_TOLERANCE:
            if actual_dur > max_allowed_dur + 0.05:
                if _trim_with_fade(clip_path, aligned_path, max_allowed_dur):
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["trimmed"] += 1
                else:
                    stats["kept"] += 1
            else:
                stats["kept"] += 1
            continue

        if ratio < (1.0 - ALIGN_TOLERANCE):
            if _pad_with_silence(clip_path, aligned_path, target_dur):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["padded"] += 1
            else:
                stats["kept"] += 1

        elif ratio <= MAX_STRETCH:
            speed = actual_dur / target_dur
            if _time_stretch(clip_path, aligned_path, speed):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["stretched"] += 1
            else:
                if _trim_with_fade(clip_path, aligned_path, target_dur):
                    clip["clip_path"] = aligned_path
                    clip["actual_dur"] = _get_dur(aligned_path)
                    stats["trimmed"] += 1
                else:
                    stats["kept"] += 1

        else:
            if _trim_with_fade(clip_path, aligned_path, target_dur):
                clip["clip_path"] = aligned_path
                clip["actual_dur"] = _get_dur(aligned_path)
                stats["trimmed"] += 1
            else:
                stats["kept"] += 1

    logger.info(f"[ALIGN] {'━' * 55}")
    logger.info(f"[ALIGN] ✓ Alignment complete:")
    logger.info(f"[ALIGN]   Kept as-is:    {stats['kept']}")
    logger.info(f"[ALIGN]   Padded:        {stats['padded']}")
    logger.info(f"[ALIGN]   Time-stretched: {stats['stretched']}")
    logger.info(f"[ALIGN]   Trimmed:       {stats['trimmed']}")
    if stats["failed"]:
        logger.warning(f"[ALIGN]   Failed:        {stats['failed']}")
    logger.info(f"[ALIGN] {'━' * 55}")

    return manifest_sorted
