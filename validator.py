"""
QUALITY VALIDATOR — Catches translation/dialogue issues before TTS generation.
Checks: word count vs timing, Devanagari presence, empty text, hallucination markers.
"""
import re, logging
from config import get_wps
logger = logging.getLogger(__name__)

_DEV_RE = re.compile(r'[\u0900-\u097F]')

def validate(segments, target_lang="Hindi", auto_fix=True):
    """
    Validate dubbed segments before TTS. Returns issues dict.
    If auto_fix=True, attempts to fix issues in-place.
    """
    issues = {
        "devanagari": [],
        "too_long": [],
        "too_short": [],
        "empty": [],
        "hallucination": [],
    }
    fixed = 0
    
    for s in segments:
        text = s.get("dubbed_text", "").strip()
        sid = s.get("id", "?")
        dur = s.get("end", 0) - s.get("start", 0)
        max_words = max(3, int(dur * get_wps(target_lang)))
        
        # Check empty
        if not text or len(text) < 2:
            issues["empty"].append(sid)
            if auto_fix:
                s["dubbed_text"] = s.get("text", "...")
                fixed += 1
            continue
        
        # Check Devanagari
        if _DEV_RE.search(text):
            issues["devanagari"].append(sid)
            # Romanizer should have already fixed this, but flag it
        
        # Check word count vs timing
        word_count = len(text.split())
        if word_count > max_words + 3:
            issues["too_long"].append({
                "id": sid, "words": word_count, "max": max_words,
                "dur": round(dur, 2), "text": text[:50]
            })
            if auto_fix:
                # Trim to max words
                s["dubbed_text"] = " ".join(text.split()[:max_words])
                fixed += 1
        
        if dur > 2 and word_count < 2:
            issues["too_short"].append(sid)
        
        # Check for hallucination patterns
        # If dubbed text is 5x+ longer than original, likely hallucination
        orig = s.get("text", "")
        if orig and len(text) > len(orig) * 5 and len(orig) < 20:
            issues["hallucination"].append({
                "id": sid, "orig_len": len(orig), "dub_len": len(text),
                "orig": orig[:30], "dub": text[:50]
            })
    
    # Report
    total_issues = sum(len(v) for v in issues.values())
    if total_issues:
        logger.info(f"[VALIDATE] Issues found:")
        if issues["devanagari"]:
            logger.warning(f"  ⚠ Devanagari: {len(issues['devanagari'])} segments")
        if issues["too_long"]:
            logger.warning(f"  ⚠ Too long: {len(issues['too_long'])} segments (auto-trimmed)")
            for t in issues["too_long"][:3]:
                logger.warning(f"    seg {t['id']}: {t['words']} words > {t['max']} max ({t['dur']}s) | {t['text']}")
        if issues["empty"]:
            logger.warning(f"  ⚠ Empty: {len(issues['empty'])} segments")
        if issues["hallucination"]:
            logger.warning(f"  ⚠ Possible hallucination: {len(issues['hallucination'])} segments")
        if fixed:
            logger.info(f"  ✓ Auto-fixed {fixed} issues")
    else:
        logger.info(f"[VALIDATE] ✓ All {len(segments)} segments passed validation")
    
    return issues
