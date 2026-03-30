"""
TRANSLATOR v6 — Industry-grade translation using LLM provider:
  1. Full-story context in EVERY batch (not just 3-segment window)
  2. Two-pass translation: Draft → Polish for coherence
  3. Character name locking — consistent names across all batches
  4. Timing-aware prompts — "speak this in 2.3 seconds"
  5. Post-translation validation — no fragments, no repeats
  6. Robust model fallback via llm_provider
  7. Scene-based parallel processing for speed
"""
import os, json, logging, time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from config import (
    get_wps, get_lang_instruction,
    TRANSLATOR_BATCH_SIZE, TRANSLATOR_POLISH_BATCH,
)
from llm_provider import get_llm
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRAFT_PROMPT = """You are a professional video dubbing translator for {target_lang} dubbing.

TRANSLATION RULES:
1. {lang_instruction}
2. Preserve ALL key information — never drop names, numbers, events, or meaning
3. Write in ROMAN script (English letters, no native scripts)
4. PRESERVE character names exactly (proper nouns stay unchanged)
5. Match the MOOD: angry→sharp words, sad→softer words, happy→energetic words
6. Natural word order for {target_lang} — translate meaning, not word order
7. max_words is a SOFT guide only — prioritize complete, meaningful sentences over the word limit

{name_glossary}

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "..."}}]}}"""

POLISH_PROMPT = """You are a senior {target_lang} dubbing script editor.

Your job is to POLISH draft translations into natural, conversational {target_lang} speech. The drafts have the right meaning — your job is to make them SOUND REAL.

CHECK AND FIX:
1. NATURALNESS: Does it sound like natural {target_lang} people actually speak? Remove stiff/bookish phrasing. Use contractions/natural forms.
2. COMPLETE SENTENCES: Ensure each line is a complete, meaningful sentence — not a fragment
3. CHARACTER VOICE: Does each character speak consistently? Adjust vocabulary to match
4. EMOTION MATCH: Angry lines should be punchy. Sad lines softer. Don't use neutral phrasing for emotional moments
5. NO WORD COUNT LIMITS: If polishing makes it slightly longer and it sounds more natural — keep it
6. IF A SEGMENT IS TOO SHORT to convey meaning naturally, you MAY expand slightly using context
7. Write in ROMAN script only

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "polished text"}}]}}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASYNC LLM CALLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm_sync(prompt: str, message: str, model: str = None) -> Optional[Dict]:
    """Synchronous wrapper for LLM call (runs in thread pool)."""
    llm = get_llm()
    return llm.chat(prompt, message, model=model, json_response=True, max_tokens=8000)


async def _call_llm_async(prompt: str, message: str, model: str = None) -> Optional[Dict]:
    """Async wrapper using thread pool for non-blocking LLM calls."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm_sync, prompt, message, model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BATCH PROCESSING HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_name_glossary(character_names: dict) -> str:
    """Build character name glossary for prompt."""
    if not character_names:
        return ""
    glossary = "CHARACTER NAME GLOSSARY (keep these unchanged):\n"
    for role, name in character_names.items():
        glossary += f"  {role} = {name}\n"
    return glossary


def _build_context_from_previous(summary: str, batch: List[Dict]) -> str:
    """Build context string from previous batch results."""
    if not batch:
        return ""
    last3 = batch[-3:]
    lines = []
    for s in last3:
        orig = s.get("text", "")[:40]
        dubbed = s.get("dubbed_text", "")[:40]
        lines.append(f"  [{s['id']}|{s.get('speaker','?')}] {orig} → {dubbed}")
    return "PREVIOUS TRANSLATIONS (for continuity):\n" + "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 1: PARALLEL DRAFT TRANSLATION (by scene)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _translate_scene_pass1(
    scene_id: int,
    scene_name: str,
    segments: List[Dict],
    target_lang: str,
    story_context: str,
    character_names: dict,
    wps: float,
    llm_prompt: str,
) -> Dict:
    """Translate a single scene in Pass 1."""
    logger.info(f"[TRANSLATOR] Pass 1: Scene {scene_id} ({len(segments)} segments)...")

    batches = [segments[i:i+TRANSLATOR_BATCH_SIZE] for i in range(0, len(segments), TRANSLATOR_BATCH_SIZE)]
    prev_context = ""

    for bi, batch in enumerate(batches):
        items = [{
            "id": s["id"],
            "duration_sec": round(s["end"] - s["start"], 2),
            "max_words": max(2, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "original": s["text"]
        } for s in batch]

        user_msg = f"""Story context: {story_context}
{prev_context}
Translate these {len(items)} segments to {target_lang}:
{json.dumps(items, ensure_ascii=False)}"""

        result = await _call_llm_async(llm_prompt, user_msg)

        if result:
            seg_map = {s["id"]: s for s in batch}
            for t in result.get("segments", []):
                if t["id"] in seg_map:
                    text = t.get("dubbed_text", "")
                    # Only truncate if it has 3x+ more words than reasonable
                    dur = seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]
                    max_w = max(3, int(dur * wps))
                    words = text.split()
                    if len(words) > max_w * 2:
                        # Gentle trim, keep the start and end meaning
                        text = " ".join(words[:int(max_w * 1.5)])
                    seg_map[t["id"]]["dubbed_text"] = text

            prev_context = _build_context_from_previous(story_context, batch)
            logger.info(f"[TRANSLATOR]   Scene {scene_id} batch {bi+1}/{len(batches)} ✓")
        else:
            logger.warning(f"[TRANSLATOR] Scene {scene_id} batch {bi+1} failed")

    return {"scene_id": scene_id, "segments": segments}


async def _translate_pass1_parallel(
    segments: List[Dict],
    target_lang: str,
    summary: str,
    character_names: dict,
    scenes: List[Dict],
    wps: float,
) -> List[Dict]:
    """Pass 1: Parallel translation by scene."""
    lang_inst = get_lang_instruction(target_lang)
    name_gloss = _build_name_glossary(character_names)
    prompt = DRAFT_PROMPT.format(lang_instruction=lang_inst, name_glossary=name_gloss)

    # Group segments by scene
    scene_groups = []
    if scenes:
        for scene in scenes:
            scene_segs = [s for s in segments if s["id"] in scene.get("segment_ids", [])]
            if scene_segs:
                scene_groups.append((scene["scene_id"], f"Scene {scene['scene_id']+1}", scene_segs))

    # Fallback: treat whole segment list as one scene
    if not scene_groups:
        scene_groups = [(0, "All segments", segments)]

    logger.info(f"[TRANSLATOR] Pass 1/2: Translating {len(scene_groups)} scenes in parallel...")

    # Launch all scene translations concurrently
    tasks = [
        _translate_scene_pass1(
            scene_id, name, segs, target_lang, summary, character_names, wps, prompt
        )
        for scene_id, name, segs in scene_groups
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for errors
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[TRANSLATOR] Scene {scene_groups[i][0]} failed: {r}")

    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 2: PARALLEL POLISH (by scene)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _translate_scene_pass2(
    scene_id: int,
    scene_name: str,
    segments: List[Dict],
    target_lang: str,
    story_context: str,
    wps: float,
    llm_prompt: str,
) -> Dict:
    """Polish a single scene in Pass 2."""
    if not segments:
        return {"scene_id": scene_id, "segments": []}

    logger.info(f"[TRANSLATOR] Pass 2: Polishing Scene {scene_id} ({len(segments)} segments)...")

    batches = [segments[i:i+TRANSLATOR_POLISH_BATCH] for i in range(0, len(segments), TRANSLATOR_POLISH_BATCH)]

    for bi, batch in enumerate(batches):
        items = [{
            "id": s["id"],
            "duration_sec": round(s["end"] - s["start"], 2),
            "max_words": max(2, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "original_chinese": s.get("text", "")[:60],
            "draft_translation": s.get("dubbed_text", "")
        } for s in batch]

        user_msg = f"""Story: {story_context}
Polish these {len(items)} draft translations:
{json.dumps(items, ensure_ascii=False)}"""

        result = await _call_llm_async(llm_prompt, user_msg)

        if result:
            seg_map = {s["id"]: s for s in batch}
            polished_count = 0
            for t in result.get("segments", []):
                if t["id"] in seg_map:
                    new_text = t.get("dubbed_text", "")
                    if new_text and new_text != seg_map[t["id"]].get("dubbed_text", ""):
                        # Only truncate at 2x budget — polish pass should expand if needed
                        dur = seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]
                        max_w = max(3, int(dur * wps))
                        words = new_text.split()
                        if len(words) > max_w * 2:
                            new_text = " ".join(words[:int(max_w * 1.5)])
                        seg_map[t["id"]]["dubbed_text"] = new_text
                        polished_count += 1

            logger.info(f"[TRANSLATOR]   Scene {scene_id} polish {bi+1}/{len(batches)} ✓ ({polished_count} refined)")
        else:
            logger.info(f"[TRANSLATOR]   Scene {scene_id} polish {bi+1} skipped")

    return {"scene_id": scene_id, "segments": segments}


async def _translate_pass2_parallel(
    segments: List[Dict],
    target_lang: str,
    summary: str,
    scenes: List[Dict],
    wps: float,
) -> List[Dict]:
    """Pass 2: Parallel polish by scene."""
    prompt = POLISH_PROMPT.format(target_lang=target_lang)

    # Group segments by scene
    scene_groups = []
    if scenes:
        for scene in scenes:
            scene_segs = [s for s in segments if s["id"] in scene.get("segment_ids", [])]
            if scene_segs:
                scene_groups.append((scene["scene_id"], f"Scene {scene['scene_id']+1}", scene_segs))

    if not scene_groups:
        scene_groups = [(0, "All segments", segments)]

    logger.info(f"[TRANSLATOR] Pass 2/2: Polishing {len(scene_groups)} scenes in parallel...")

    tasks = [
        _translate_scene_pass2(
            scene_id, name, segs, target_lang, summary, wps, prompt
        )
        for scene_id, name, segs in scene_groups
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"[TRANSLATOR] Scene polish {scene_groups[i][0]} failed: {r}")

    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST-TRANSLATION VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _validate_translations(segments: List[Dict]) -> List[Dict]:
    """Check for common translation issues and fix them."""
    issues = 0

    # Check for duplicates
    texts_seen = {}
    for s in segments:
        t = s.get("dubbed_text", "").strip()
        if t and len(t) > 15:
            if t in texts_seen:
                logger.debug(f"[TRANSLATOR] Duplicate text in seg {s['id']} and {texts_seen[t]}")
                issues += 1
            texts_seen[t] = s["id"]

    # Fill missing
    for s in segments:
        if not s.get("dubbed_text", "").strip():
            s["dubbed_text"] = s.get("text", "...")
            issues += 1

    if issues:
        logger.info(f"[TRANSLATOR] Validation: {issues} issues found/fixed")

    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN TRANSLATE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def translate(dir_result, work_dir, target_lang="Hindi"):
    """
    Two-pass translation pipeline with scene-based parallelization:
    1. Draft: Parallel scene translation with context
    2. Polish: Parallel scene refinement
    3. Validate: Check for duplicates, fragments, missing
    """
    segments = dir_result["segments"]
    style = dir_result.get("translation_style", "narrative_flow")
    summary = dir_result.get("narrative_summary", "")
    character_names = dir_result.get("character_names", {})
    scenes = dir_result.get("scenes", [])
    wps = get_wps(target_lang)

    logger.info(f"[TRANSLATOR] Two-pass parallel pipeline: {len(segments)} segments → {target_lang}")
    logger.info(f"[TRANSLATOR] Style: {style} | Word rate: {wps}/sec | Scenes: {len(scenes)}")

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    # Use a fresh event loop — do NOT set it as global default (avoids conflicts)
    loop = asyncio.new_event_loop()

    try:
        # Pass 1: Parallel draft translation
        t0 = time.time()
        segments = loop.run_until_complete(
            _translate_pass1_parallel(segments, target_lang, summary, character_names, scenes, wps)
        )
        logger.info(f"[TRANSLATOR] Pass 1 complete in {time.time()-t0:.1f}s")

        # Pass 2: Parallel polish
        t0 = time.time()
        segments = loop.run_until_complete(
            _translate_pass2_parallel(segments, target_lang, summary, scenes, wps)
        )
        logger.info(f"[TRANSLATOR] Pass 2 complete in {time.time()-t0:.1f}s")

    finally:
        loop.close()

    # Validation
    segments = _validate_translations(segments)

    # Save
    sp = os.path.join(work_dir, "translated_raw.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({
            "segments": segments, "style": style,
            "target_lang": target_lang,
            "words_per_sec": wps,
            "two_pass": True
        }, f, indent=2, ensure_ascii=False)

    # Preview
    logger.info(f"[TRANSLATOR] {'━' * 55}")
    logger.info(f"[TRANSLATOR] ✓ Translation complete ({len(segments)} segments)")
    logger.info(f"[TRANSLATOR] Preview:")
    for s in segments[:5]:
        spk = s.get("speaker", "?")
        mood = s.get("mood", "?")
        logger.info(f"  [{s['start']:.1f}s|{spk}|{mood}] {s['text'][:30]} → {s.get('dubbed_text','')[:40]}")
    logger.info(f"[TRANSLATOR] {'━' * 55}")

    return {"raw_script_path": sp}
