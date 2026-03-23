"""PRE-PROCESSOR: Merge micro-segments (<1.5s) before translation to prevent LLM hallucination."""
import logging
logger = logging.getLogger(__name__)
MIN_DUR = 1.5
MAX_DUR = 8.0

def merge_short_segments(segments):
    if not segments: return segments
    merged, i = [], 0
    while i < len(segments):
        seg = dict(segments[i])
        seg["sub_segments"] = [dict(segments[i])]
        while i + 1 < len(segments):
            dur = seg["end"] - seg["start"]
            nxt = segments[i+1]
            combined = nxt["end"] - seg["start"]
            # Merge if short AND same speaker AND combined not too long
            same_speaker = seg.get("speaker") == nxt.get("speaker", seg.get("speaker"))
            if (dur < MIN_DUR or len(seg["text"].strip()) < 6) and combined <= MAX_DUR and same_speaker:
                seg["text"] = seg["text"].rstrip() + " " + nxt["text"].strip()
                seg["end"] = nxt["end"]
                seg["mood"] = nxt.get("mood", seg.get("mood", "neutral"))  # take later mood
                seg["sub_segments"].append(dict(nxt))
                i += 1
            else:
                break
        seg["duration"] = round(seg["end"] - seg["start"], 3)
        seg["merged_count"] = len(seg["sub_segments"])
        merged.append(seg)
        i += 1
    logger.info(f"[PRE] {len(segments)} segments → {len(merged)} translation units")
    return merged

def expand_dubbed_to_subsegments(merged_segments):
    expanded = []
    for seg in merged_segments:
        subs = seg.get("sub_segments", [seg])
        dubbed = seg.get("dubbed_text", seg.get("text", ""))
        if len(subs) == 1:
            subs[0]["dubbed_text"] = dubbed
            subs[0]["speaker"] = seg.get("speaker", "NARRATOR")
            subs[0]["mood"] = seg.get("mood", "neutral")
            expanded.append(subs[0])
            continue
        total_dur = sum(s["end"] - s["start"] for s in subs)
        words = dubbed.split()
        total_words = len(words)
        word_pos = 0
        for j, sub in enumerate(subs):
            sub_dur = sub["end"] - sub["start"]
            if j == len(subs) - 1:
                sub["dubbed_text"] = " ".join(words[word_pos:]).strip()
            else:
                n = max(1, round(total_words * (sub_dur / total_dur)))
                sub["dubbed_text"] = " ".join(words[word_pos:word_pos + n]).strip()
                word_pos += n
            sub["speaker"] = seg.get("speaker", "NARRATOR")
            sub["mood"] = sub.get("mood", seg.get("mood", "neutral"))
            if not sub["dubbed_text"]:
                sub["dubbed_text"] = sub.get("text", "")
            expanded.append(sub)
    logger.info(f"[PRE] Expanded back to {len(expanded)} TTS segments")
    return expanded
