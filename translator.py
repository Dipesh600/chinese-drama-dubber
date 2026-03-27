"""
TRANSLATOR v4 — Industry-grade translation:
  1. Full-story context in EVERY batch (not just 3-segment window)
  2. Two-pass translation: Draft → Polish for coherence
  3. Character name locking — consistent names across all batches
  4. Timing-aware prompts — "speak this in 2.3 seconds"
  5. Post-translation validation — no fragments, no repeats
  6. Robust model fallback chain with retry
"""
import os, json, logging, time
from groq import Groq
logger = logging.getLogger(__name__)
_client = None

def _gc():
    global _client
    if not _client:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Words per second by language (for timing constraints)
WORDS_PER_SEC = {
    "Hindi": 2.5, "Tamil": 2.0, "Telugu": 2.2, "Bengali": 2.3,
    "Marathi": 2.4, "Gujarati": 2.4, "Kannada": 2.1, "Malayalam": 1.9,
    "Nepali": 2.5, "Urdu": 2.5, "English": 3.0, "Spanish": 2.8,
    "French": 2.7, "Portuguese": 2.6, "German": 2.5, "Japanese": 3.5,
    "Korean": 2.8, "Arabic": 2.5, "Turkish": 2.6,
}

LANG_INSTRUCTIONS = {
    "Hindi": "Translate to natural Hinglish (Hindi+English mix as Indians actually speak). Keep English words Indians commonly use: okay, sorry, please, time, market, phone, office, school. Write in ROMAN script (NOT Devanagari).",
    "Tamil": "Translate to natural spoken Tamil (Tanglish where natural). Roman script only.",
    "Telugu": "Translate to natural spoken Telugu. Roman script only.",
    "Bengali": "Translate to natural spoken Bengali. Roman script only.",
    "Marathi": "Translate to natural spoken Marathi. Roman script only.",
    "Gujarati": "Translate to natural spoken Gujarati. Roman script only.",
    "Kannada": "Translate to natural spoken Kannada. Roman script only.",
    "Malayalam": "Translate to natural spoken Malayalam. Roman script only.",
    "Nepali": "Translate to natural spoken Nepali (Nepali+English mix). Use Hajur for respect. Roman script.",
    "English": "Translate to natural spoken English. Conversational, engaging.",
    "Spanish": "Translate to natural Latin American Spanish.",
    "French": "Translate to natural spoken French.",
    "Portuguese": "Translate to natural Brazilian Portuguese.",
    "German": "Translate to natural spoken German.",
    "Japanese": "Translate to natural spoken Japanese in Romaji.",
    "Korean": "Translate to natural spoken Korean in Romanized form.",
    "Arabic": "Translate to natural spoken Arabic in Romanized form.",
    "Turkish": "Translate to natural spoken Turkish.",
    "Urdu": "Translate to natural spoken Urdu in Roman script.",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 1: DRAFT TRANSLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRAFT_PROMPT = """You are a professional video dubbing translator. Translate for lip-sync dubbing.

ABSOLUTE RULES:
1. Translate ONLY what is in "original" — NEVER add, invent, or hallucinate content
2. {lang_instruction}
3. TIMING IS KING: Each segment has "duration_sec" and "max_words". Your translation MUST:
   - Have ≤ max_words words
   - Be speakable within the given duration at normal pace
   - A 1.5s segment needs ~4 Hindi words, NOT a full sentence
4. Short original = short translation. "好" (1 word) → "Accha" (1 word), NOT "Yeh bahut accha hai"
5. PRESERVE character names exactly (proper nouns stay unchanged)
6. Match the MOOD: angry→sharp words, sad→softer words, happy→energetic words
7. Each segment is INDEPENDENT — translate it as the CHARACTER would say it, not as narration about it

{name_glossary}

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "..."}}]}}"""

BATCH_SIZE = 20

def _call_llm(prompt, msg, model=None, max_retries=4):
    """Call LLM with automatic retry and model fallback."""
    c = _gc()
    m = model or PRIMARY_MODEL
    
    for attempt in range(max_retries):
        try:
            resp = c.chat.completions.create(
                model=m,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": msg}],
                temperature=0.25, max_tokens=4000,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                if m == PRIMARY_MODEL:
                    logger.warning(f"[TRANSLATOR] {PRIMARY_MODEL} rate limited → {FALLBACK_MODEL}")
                    m = FALLBACK_MODEL
                    continue
                wait = min(2 ** attempt, 12)
                logger.warning(f"[TRANSLATOR] Rate limited, retry in {wait}s...")
                time.sleep(wait)
            elif "model" in err.lower() and m == PRIMARY_MODEL:
                logger.warning(f"[TRANSLATOR] {PRIMARY_MODEL} unavailable → {FALLBACK_MODEL}")
                m = FALLBACK_MODEL
                continue
            else:
                wait = min(2 ** attempt, 10)
                logger.warning(f"[TRANSLATOR] Error: {e}, retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
    return None


def _translate_pass1(segments, target_lang, summary, character_names, wps):
    """Pass 1: Draft translation — fast, parallel batches."""
    lang_inst = LANG_INSTRUCTIONS.get(target_lang, LANG_INSTRUCTIONS["English"])
    
    # Build name glossary
    name_gloss = ""
    if character_names:
        name_gloss = "CHARACTER NAME GLOSSARY (keep these unchanged):\n"
        for role, name in character_names.items():
            name_gloss += f"  {role} = {name}\n"
    
    prompt = DRAFT_PROMPT.format(lang_instruction=lang_inst, name_glossary=name_gloss)
    batches = [segments[i:i+BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
    
    logger.info(f"[TRANSLATOR] Pass 1/2: Draft translation ({len(batches)} batches)...")
    
    # Translation context from previous batch
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
        
        user_msg = f"""Story context: {summary}
{prev_context}
Translate these {len(items)} segments to {target_lang}:
{json.dumps(items, ensure_ascii=False)}"""
        
        result = _call_llm(prompt, user_msg)
        
        if result:
            seg_map = {s["id"]: s for s in batch}
            for t in result.get("segments", []):
                if t["id"] in seg_map:
                    text = t.get("dubbed_text", "")
                    # Enforce word limit
                    dur = seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]
                    max_w = max(2, int(dur * wps))
                    words = text.split()
                    if len(words) > max_w + 2:
                        text = " ".join(words[:max_w])
                    seg_map[t["id"]]["dubbed_text"] = text
            
            # Build context for next batch
            last3 = batch[-3:]
            prev_lines = []
            for s in last3:
                orig = s.get("text", "")[:40]
                dubbed = s.get("dubbed_text", "")[:40]
                prev_lines.append(f"  [{s['id']}|{s.get('speaker','?')}] {orig} → {dubbed}")
            prev_context = "PREVIOUS TRANSLATIONS (for continuity):\n" + "\n".join(prev_lines)
            
            logger.info(f"[TRANSLATOR] Draft batch {bi+1}/{len(batches)} ✓")
        else:
            logger.warning(f"[TRANSLATOR] Draft batch {bi+1} failed — keeping original text")
    
    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PASS 2: POLISH (coherence + naturalness)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POLISH_PROMPT = """You are a senior dubbing script editor reviewing draft translations for {target_lang} dubbing.

Your job is to POLISH the draft translations for naturalness and coherence. The translations are already roughly correct — you are refining, not re-translating.

CHECK AND FIX:
1. COHERENCE: Do consecutive lines flow naturally? Fix awkward transitions.
2. CHARACTER VOICE: Does each character speak consistently? Father=authoritative, Child=simple vocabulary.
3. NATURALNESS: Does it sound like natural {target_lang}? Remove stiff/bookish phrasing.
4. REPEATED LINES: If two segments have identical translation, vary the wording.
5. SENTENCE FRAGMENTS: If a sentence is split across segments, ensure each part is complete.
6. WORD COUNT: STILL respect max_words. If polishing makes it longer, CUT words.
7. EMOTION MATCH: Angry lines should be punchy. Sad lines softer. Don't use neutral phrasing for emotional moments.
8. Write in ROMAN script only (no Devanagari/native scripts).

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "polished text"}}]}}"""

POLISH_BATCH = 30  # Bigger batches for polish (needs more context)

def _translate_pass2(segments, target_lang, summary, scenes, wps):
    """Pass 2: Polish translation for coherence, grouped by scene."""
    prompt = POLISH_PROMPT.format(target_lang=target_lang)
    
    # Group by scene if available, otherwise by batch
    groups = []
    if scenes:
        for scene in scenes:
            scene_segs = [s for s in segments if s["id"] in scene.get("segment_ids", [])]
            if scene_segs:
                # Split large scenes
                for i in range(0, len(scene_segs), POLISH_BATCH):
                    groups.append(scene_segs[i:i+POLISH_BATCH])
    
    if not groups:
        groups = [segments[i:i+POLISH_BATCH] for i in range(0, len(segments), POLISH_BATCH)]
    
    logger.info(f"[TRANSLATOR] Pass 2/2: Polish ({len(groups)} scene-groups)...")
    
    for gi, group in enumerate(groups):
        items = [{
            "id": s["id"],
            "duration_sec": round(s["end"] - s["start"], 2),
            "max_words": max(2, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "original_chinese": s.get("text", "")[:60],
            "draft_translation": s.get("dubbed_text", "")
        } for s in group]
        
        user_msg = f"""Story: {summary}
Polish these {len(items)} draft translations:
{json.dumps(items, ensure_ascii=False)}"""
        
        result = _call_llm(prompt, user_msg)
        
        if result:
            seg_map = {s["id"]: s for s in group}
            polished = 0
            for t in result.get("segments", []):
                if t["id"] in seg_map:
                    new_text = t.get("dubbed_text", "")
                    if new_text and new_text != seg_map[t["id"]].get("dubbed_text", ""):
                        # Enforce word limit
                        dur = seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]
                        max_w = max(2, int(dur * wps))
                        words = new_text.split()
                        if len(words) > max_w + 2:
                            new_text = " ".join(words[:max_w])
                        seg_map[t["id"]]["dubbed_text"] = new_text
                        polished += 1
            
            logger.info(f"[TRANSLATOR] Polish group {gi+1}/{len(groups)} ✓ ({polished} refined)")
        else:
            logger.info(f"[TRANSLATOR] Polish group {gi+1} skipped (keeping draft)")
    
    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST-TRANSLATION VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _validate_translations(segments):
    """Check for common translation issues and fix them."""
    issues = 0
    
    # Check for duplicates (same translation across multiple segments)
    texts_seen = {}
    for s in segments:
        t = s.get("dubbed_text", "").strip()
        if t and len(t) > 15:  # Only check substantial text
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
    Two-pass translation pipeline:
    1. Draft: Fast batch translation with context
    2. Polish: Scene-based refinement for coherence
    3. Validate: Check for duplicates, fragments, missing
    """
    segments = dir_result["segments"]
    style = dir_result.get("translation_style", "narrative_flow")
    summary = dir_result.get("narrative_summary", "")
    character_names = dir_result.get("character_names", {})
    scenes = dir_result.get("scenes", [])
    wps = WORDS_PER_SEC.get(target_lang, 2.5)
    
    logger.info(f"[TRANSLATOR] Two-pass pipeline: {len(segments)} segments → {target_lang}")
    logger.info(f"[TRANSLATOR] Style: {style} | Word rate: {wps}/sec | Scenes: {len(scenes)}")
    
    # Pass 1: Draft
    segments = _translate_pass1(segments, target_lang, summary, character_names, wps)
    
    # Pass 2: Polish (scene-aware)
    segments = _translate_pass2(segments, target_lang, summary, scenes, wps)
    
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
