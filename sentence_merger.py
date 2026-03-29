"""
SENTENCE MERGER — Combines choppy short dubbed lines into natural sentence-level TTS units.
This fixes the robotic feel of generating TTS for tiny 0.5-1s segments.

Strategy: Adjacent segments from SAME speaker with combined duration < MAX_TTS_DUR
get merged into a single TTS generation, then placed at the FIRST segment's timestamp.
"""
import logging
from config import MAX_TTS_DUR, MIN_MERGE_DUR, MAX_GAP
logger = logging.getLogger(__name__)


def merge_for_tts(segments):
    """
    Merge adjacent same-speaker segments into natural TTS units.
    Each merged unit gets one TTS generation for smoother speech.
    Returns new segment list with 'tts_text' (full text for TTS) and
    'tts_group' (list of original segment timings for subtitle placement).
    """
    if not segments:
        return segments

    merged = []
    i = 0

    while i < len(segments):
        seg = dict(segments[i])
        seg["tts_text"] = seg.get("dubbed_text", seg.get("text", ""))
        seg["tts_group"] = [dict(segments[i])]

        # Try to merge forward
        while i + 1 < len(segments):
            nxt = segments[i + 1]
            cur_dur = seg["end"] - seg["start"]
            nxt_dur = nxt["end"] - nxt["start"]
            combined_dur = nxt["end"] - seg["start"]
            gap = nxt["start"] - seg["end"]

            # Merge conditions
            same_speaker = seg.get("speaker") == nxt.get("speaker")
            short_enough = cur_dur < MIN_MERGE_DUR or nxt_dur < MIN_MERGE_DUR
            fits = combined_dur <= MAX_TTS_DUR
            close = gap <= MAX_GAP

            if same_speaker and short_enough and fits and close:
                nxt_text = nxt.get("dubbed_text", nxt.get("text", ""))
                seg["tts_text"] = seg["tts_text"].rstrip() + " " + nxt_text.strip()
                seg["end"] = nxt["end"]
                seg["tts_group"].append(dict(nxt))
                i += 1
            else:
                break

        seg["tts_duration"] = round(seg["end"] - seg["start"], 3)
        seg["tts_merged_count"] = len(seg["tts_group"])
        merged.append(seg)
        i += 1

    merged_count = sum(1 for m in merged if m["tts_merged_count"] > 1)
    logger.info(f"[SENTENCE] {len(segments)} segments → {len(merged)} TTS units ({merged_count} merged groups)")
    return merged
