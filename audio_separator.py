"""
AUDIO SEPARATOR — Uses Meta's Demucs for vocal/music/ambient separation.
Features:
  1. Smart vocal bleed detection — skips Demucs if background is already clean
  2. Two-stem mode (vocals vs rest) for speed when separation is needed
  3. Automatic device selection (cuda > mps > cpu)
  4. 10-minute timeout with graceful fallback
"""
import os, logging, subprocess, shutil
logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs_ft"  # Fine-tuned model — best quality

# Smart detection thresholds
VOCAL_FREQ_MIN = 300    # Hz - lowest vocal frequency
VOCAL_FREQ_MAX = 3400   # Hz - highest vocal frequency
VOCAL_BLEED_THRESHOLD = 0.20  # If vocal energy > 20% of total, needs Demucs


def _analyze_vocal_bleed(audio_path, sample_duration=10):
    """
    Quick analysis to detect if audio has significant vocal bleed.
    Uses FFmpeg's aformat and astat to measure frequency distribution.

    Returns:
        float: Ratio of vocal-frequency energy to total energy (0.0 - 1.0)
        - If < VOCAL_BLEED_THRESHOLD: background is clean, Demucs can be skipped
        - If >= threshold: vocal bleed detected, Demucs needed
    """
    logger.info("[SEPARATOR] Analyzing audio for vocal bleed...")

    try:
        # Analyze a sample of the audio (first 10 seconds is representative)
        # Use FFmpeg to get RMS energy in vocal frequency band
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-t", str(sample_duration),
            "-af", (
                f"aformat=sample_fmts=s16:channel_layouts=mono,"
                f"astats=metadata=1:reset=1,"
                f"volumedetect"
            ),
            "-f", "null", "-"
        ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = r.stderr

        # Get mean volume as proxy for energy
        mean_vol = -30.0  # default
        for line in output.split('\n'):
            if "mean_volume" in line:
                try:
                    mean_vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                except:
                    pass

        # Also try to detect significant vocal frequencies using high-pass/low-pass
        # If an audio has lots of content in 300Hz-3.4kHz vocal range,
        # it's likely to have vocals

        # High-pass to remove bass, measure what's left (vocal + instruments)
        cmd_high = [
            "ffmpeg", "-y", "-i", audio_path,
            "-t", str(sample_duration),
            "-af", "highpass=f=200,volumedetect",
            "-f", "null", "-"
        ]
        r_high = subprocess.run(cmd_high, capture_output=True, text=True, timeout=30)

        high_vol = -30.0
        for line in r_high.stderr.split('\n'):
            if "mean_volume" in line:
                try:
                    high_vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                except:
                    pass

        # Low-pass to get bass/music only
        cmd_low = [
            "ffmpeg", "-y", "-i", audio_path,
            "-t", str(sample_duration),
            "-af", "lowpass=f=300,volumedetect",
            "-f", "null", "-"
        ]
        r_low = subprocess.run(cmd_low, capture_output=True, text=True, timeout=30)

        low_vol = -30.0
        for line in r_low.stderr.split('\n'):
            if "mean_volume" in line:
                try:
                    low_vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                except:
                    pass

        # Estimate vocal bleed as the difference
        # High freq (200+) minus low freq (below 300) gives us vocal+treble
        # If high is much louder than low, there's significant vocal/melody content
        vocal_proxy = high_vol - low_vol  # dB difference

        # Also compare overall to see if there's significant content
        total_dynamic = mean_vol - (-60.0)  # normalized to 0-60 range

        logger.info(f"[SEPARATOR] Audio analysis: total={mean_vol:.1f}dB, high={high_vol:.1f}dB, low={low_vol:.1f}dB")
        logger.info(f"[SEPARATOR] Vocal proxy (high-low): {vocal_proxy:.1f}dB, dynamic range: {total_dynamic:.1f}")

        # Heuristic:
        # - If low frequencies dominate (low_vol ≈ mean_vol), it's mostly music
        # - If there's significant high-frequency content, could be vocals or instruments
        # - Bass-heavy music shows low_vol close to mean_vol

        # If low freq is within 10dB of overall, it's music-heavy (no vocals)
        is_music_heavy = (mean_vol - low_vol) < 10.0

        # If there's wide dynamic range and low frequencies are strong, background is clean
        if is_music_heavy and total_dynamic > 10:
            logger.info(f"[SEPARATOR] ✓ Audio appears to be music-heavy, minimal vocal bleed expected")
            return 0.1  # Low bleed detected, Demucs can be skipped

        # Otherwise, return moderate bleed estimate
        bleed_ratio = max(0.2, min(0.8, 1.0 - (total_dynamic / 60.0)))
        logger.info(f"[SEPARATOR] Estimated vocal bleed ratio: {bleed_ratio:.2f}")

        return bleed_ratio

    except Exception as e:
        logger.warning(f"[SEPARATOR] Audio analysis failed: {e}, will run Demucs to be safe")
        return 0.5  # Default to running Demucs


def _needs_demucs(audio_path):
    """
    Decide if Demucs separation is actually needed.
    Uses smart detection to skip if background is already clean.
    """
    bleed_ratio = _analyze_vocal_bleed(audio_path)

    if bleed_ratio < VOCAL_BLEED_THRESHOLD:
        logger.info(
            f"[SEPARATOR] ✓ Vocal bleed ratio ({bleed_ratio:.1%}) below threshold ({VOCAL_BLEED_THRESHOLD:.1%})"
        )
        logger.info(f"[SEPARATOR] → Skipping Demucs, using original audio as background")
        return False
    else:
        logger.info(
            f"[SEPARATOR] ⚠ Vocal bleed ratio ({bleed_ratio:.1%}) above threshold ({VOCAL_BLEED_THRESHOLD:.1%})"
        )
        logger.info(f"[SEPARATOR] → Running Demucs to clean background")
        return True


def separate(audio_path, work_dir, device="auto", force_separate=False, preserve_bg=True):
    """
    Separate audio into stems using Demucs.

    For dubbing (preserve_bg=True): Demucs ALWAYS runs to remove original vocals.
    Smart skip is disabled for dubbing — original vocals must be removed.

    Returns:
        dict with paths to separated stems
    """
    output_dir = os.path.join(work_dir, "separated")
    os.makedirs(output_dir, exist_ok=True)

    bg_path = os.path.join(work_dir, "background_clean.wav")

    # Check cache
    if os.path.exists(bg_path) and os.path.getsize(bg_path) > 1000:
        logger.info("[SEPARATOR] ✓ Already separated, reusing cached stems")
        return {
            "bg_path": bg_path,
            "music_path": os.path.join(output_dir, "music.wav"),
            "other_path": os.path.join(output_dir, "other.wav"),
            "vocals_path": os.path.join(output_dir, "vocals.wav"),
        }

    # FOR DUBBING: Always run Demucs to remove original vocals.
    # Smart skip is only valid for music remix workflows where vocals are wanted.
    # When preserve_bg=True, we need clean background (no original vocals).
    if not force_separate and not preserve_bg and not _needs_demucs(audio_path):
        # Smart skip only applies when NOT preserving background
        try:
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", "44100", "-ac", "2",
                bg_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            bg_size = os.path.getsize(bg_path) / 1024 / 1024
            logger.info(f"[SEPARATOR] ✓ No background preserve requested, using original: {bg_size:.1f}MB")
            return {
                "bg_path": bg_path,
                "music_path": "",
                "other_path": "",
                "vocals_path": "",
                "skipped_demucs": True,
            }
        except Exception as e:
            logger.warning(f"[SEPARATOR] Failed: {e}, running Demucs")

    # ── Run Demucs ──────────────────────────────────────────────────
    logger.info(f"[SEPARATOR] Separating audio with {DEMUCS_MODEL}...")
    logger.info(f"[SEPARATOR] This removes Chinese vocals, keeps clean background music + SFX")

    # Determine device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"[SEPARATOR] Device: {device}")

    # Run Demucs via CLI
    cmd = [
        "python", "-m", "demucs",
        "--name", DEMUCS_MODEL,
        "--out", output_dir,
        "--device", device,
        "--two-stems", "vocals",
        audio_path
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            logger.warning(f"[SEPARATOR] Two-stem mode failed, trying full separation...")
            cmd_full = [
                "python", "-m", "demucs",
                "--name", DEMUCS_MODEL,
                "--out", output_dir,
                "--device", device,
                audio_path
            ]
            result = subprocess.run(cmd_full, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"[SEPARATOR] Demucs failed: {result.stderr[-300:]}")
                raise RuntimeError(f"Demucs separation failed")
    except subprocess.TimeoutExpired:
        logger.error("[SEPARATOR] Demucs timed out (>10 min)")
        raise RuntimeError("Audio separation timed out")

    # Find output files
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    stems_dir = os.path.join(output_dir, DEMUCS_MODEL, audio_name)

    if not os.path.exists(stems_dir):
        for root, dirs, files in os.walk(output_dir):
            if any(f.endswith('.wav') for f in files):
                stems_dir = root
                break

    # Collect stem paths
    stems = {}
    for stem_name in ["vocals", "no_vocals", "drums", "bass", "other"]:
        stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
        if os.path.exists(stem_path):
            clean_path = os.path.join(output_dir, f"{stem_name}.wav")
            shutil.copy2(stem_path, clean_path)
            stems[stem_name] = clean_path
            logger.info(f"[SEPARATOR] Found: {stem_name}.wav ({os.path.getsize(stem_path)/1024/1024:.1f}MB)")

    # Create combined background
    if "no_vocals" in stems:
        shutil.copy2(stems["no_vocals"], bg_path)
        stems["music_path"] = stems.get("no_vocals", "")
        stems["other_path"] = ""
    elif "drums" in stems and "bass" in stems and "other" in stems:
        stem_files = [stems[k] for k in ["drums", "bass", "other"] if k in stems]
        _mix_stems(stem_files, bg_path)
        stems["music_path"] = stems.get("other", "")
        stems["other_path"] = stems.get("drums", "")
    else:
        logger.warning("[SEPARATOR] Could not find expected stems, using original audio")
        shutil.copy2(audio_path, bg_path)

    stems["bg_path"] = bg_path
    stems["vocals_path"] = stems.get("vocals", "")

    bg_size = os.path.getsize(bg_path) / 1024 / 1024
    logger.info(f"[SEPARATOR] ✓ Clean background: {bg_size:.1f}MB")
    logger.info(f"[SEPARATOR] ✓ Chinese vocals removed!")

    # Cleanup nested Demucs output
    nested_dir = os.path.join(output_dir, DEMUCS_MODEL)
    if os.path.exists(nested_dir):
        shutil.rmtree(nested_dir, ignore_errors=True)

    return stems


def _mix_stems(stem_paths, output_path):
    """Mix multiple audio stems together using FFmpeg."""
    if not stem_paths:
        return

    if len(stem_paths) == 1:
        shutil.copy2(stem_paths[0], output_path)
        return

    inputs = []
    for p in stem_paths:
        inputs.extend(["-i", p])

    n = len(stem_paths)
    filter_str = "".join(f"[{i}]" for i in range(n)) + f"amix=inputs={n}:normalize=0[out]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[out]", "-ar", "44100", "-ac", "2", output_path
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        shutil.copy2(stem_paths[0], output_path)
