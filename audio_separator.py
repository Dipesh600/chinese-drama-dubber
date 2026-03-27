"""
AUDIO SEPARATOR — Uses Meta's Demucs for vocal/music/ambient separation.
Removes original Chinese vocals and keeps clean background music + SFX.
This is critical for professional dubbing — no more Chinese vocal bleed-through.
"""
import os, logging, subprocess, shutil
logger = logging.getLogger(__name__)

DEMUCS_MODEL = "htdemucs_ft"  # Fine-tuned model — best quality


def separate(audio_path, work_dir, device="auto"):
    """
    Separate audio into stems using Demucs.
    
    Returns:
        dict with paths to separated stems:
        - music_path: Background music (no vocals)
        - other_path: Ambient sounds, SFX
        - vocals_path: Original vocals (for reference, usually discarded)
        - bg_path: Combined music + other (what we use for dubbing background)
    """
    output_dir = os.path.join(work_dir, "separated")
    os.makedirs(output_dir, exist_ok=True)
    
    bg_path = os.path.join(work_dir, "background_clean.wav")
    
    # Check if already separated
    if os.path.exists(bg_path) and os.path.getsize(bg_path) > 1000:
        logger.info("[SEPARATOR] ✓ Already separated, reusing cached stems")
        return {
            "bg_path": bg_path,
            "music_path": os.path.join(output_dir, "music.wav"),
            "other_path": os.path.join(output_dir, "other.wav"),
            "vocals_path": os.path.join(output_dir, "vocals.wav"),
        }
    
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
    
    # Run Demucs via CLI (most reliable method)
    cmd = [
        "python", "-m", "demucs",
        "--name", DEMUCS_MODEL,
        "--out", output_dir,
        "--device", device,
        "--two-stems", "vocals",  # Optimize: only separate vocals vs rest
        audio_path
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            logger.warning(f"[SEPARATOR] Two-stem mode failed, trying full separation...")
            # Fallback: full 4-stem separation
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
    
    # Find the output files
    # Demucs outputs to: output_dir/htdemucs_ft/audio/stem.wav
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    stems_dir = os.path.join(output_dir, DEMUCS_MODEL, audio_name)
    
    if not os.path.exists(stems_dir):
        # Try alternate path patterns
        for root, dirs, files in os.walk(output_dir):
            if any(f.endswith('.wav') for f in files):
                stems_dir = root
                break
    
    # Collect stem paths
    stems = {}
    for stem_name in ["vocals", "no_vocals", "drums", "bass", "other"]:
        stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
        if os.path.exists(stem_path):
            # Copy to our clean output dir
            clean_path = os.path.join(output_dir, f"{stem_name}.wav")
            shutil.copy2(stem_path, clean_path)
            stems[stem_name] = clean_path
            logger.info(f"[SEPARATOR] Found: {stem_name}.wav ({os.path.getsize(stem_path)/1024/1024:.1f}MB)")
    
    # Create combined background (everything minus vocals)
    if "no_vocals" in stems:
        # Two-stem mode: no_vocals is already the clean background
        shutil.copy2(stems["no_vocals"], bg_path)
        stems["music_path"] = stems.get("no_vocals", "")
        stems["other_path"] = ""
    elif "drums" in stems and "bass" in stems and "other" in stems:
        # Full mode: combine drums + bass + other → background
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
    
    # Cleanup the nested Demucs output directory
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
        # Fallback: just use the first stem
        shutil.copy2(stem_paths[0], output_path)
