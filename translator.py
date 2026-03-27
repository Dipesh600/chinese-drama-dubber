"""
TRANSLATOR v3 — Context-aware translation with:
- Rolling 3-segment context window for continuity
- Language-specific syllable rate for accurate word limits
- Character name detection and preservation
- Emotion-aware translation hints
- Robust model fallback chain
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

PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Language-specific speaking rates (syllables per second → words per second)
# Different languages have very different syllable densities
WORDS_PER_SEC = {
    "Hindi": 2.5,     # Hindi words average ~2.5 syllables
    "Tamil": 2.0,     # Tamil is more syllable-heavy
    "Telugu": 2.2,
    "Bengali": 2.3,
    "Marathi": 2.4,
    "Gujarati": 2.4,
    "Kannada": 2.1,
    "Malayalam": 1.9,  # Most syllable-heavy Indian language
    "Nepali": 2.5,
    "Urdu": 2.5,
    "English": 3.0,   # English words are shorter
    "Spanish": 2.8,
    "French": 2.7,
    "Portuguese": 2.6,
    "German": 2.5,
    "Japanese": 3.5,   # Very fast syllable rate
    "Korean": 2.8,
    "Arabic": 2.5,
    "Turkish": 2.6,
}

# Language-specific translation instructions
LANG_INSTRUCTIONS = {
    "Hindi": "Translate to natural Hinglish (Hindi+English mix as Indians actually speak). Keep common English words: time, market, office, phone, okay, sorry, please, agent, mission. Write in Roman script (not Devanagari).",
    "Tamil": "Translate to natural spoken Tamil (Tanglish where appropriate). Keep English words Tamils commonly use. Write in Roman script.",
    "Telugu": "Translate to natural spoken Telugu (Tenglish where appropriate). Keep English words Telugu speakers commonly use. Write in Roman script.",
    "Bengali": "Translate to natural spoken Bengali (Benglish where appropriate). Keep English words Bengali speakers commonly use. Write in Roman script.",
    "Marathi": "Translate to natural spoken Marathi. Keep English words Marathi speakers commonly use. Write in Roman script.",
    "Gujarati": "Translate to natural spoken Gujarati. Keep English words commonly used. Write in Roman script.",
    "Kannada": "Translate to natural spoken Kannada. Keep English words commonly used. Write in Roman script.",
    "Malayalam": "Translate to natural spoken Malayalam. Keep English words commonly used. Write in Roman script.",
    "Nepali": "Translate to natural spoken Nepali (Nepali+English mix as Nepalis actually speak). Keep common English words. Write in Roman script (not Devanagari).",
    "English": "Translate to natural spoken English. Keep the meaning and emotion. Write clear, engaging sentences.",
    "Spanish": "Translate to natural spoken Latin American Spanish. Keep the meaning and emotion.",
    "French": "Translate to natural spoken French. Keep the meaning and emotion.",
    "Japanese": "Translate to natural spoken Japanese in Romaji. Keep the meaning and emotion.",
    "Korean": "Translate to natural spoken Korean in Romanized form. Keep the meaning and emotion.",
    "Arabic": "Translate to natural spoken Arabic in Romanized form. Keep the meaning and emotion.",
    "Portuguese": "Translate to natural spoken Brazilian Portuguese. Keep the meaning and emotion.",
    "German": "Translate to natural spoken German. Keep the meaning and emotion.",
    "Turkish": "Translate to natural spoken Turkish. Keep the meaning and emotion.",
    "Urdu": "Translate to natural spoken Urdu in Roman script. Keep English words commonly used.",
}

STYLE_PROMPTS = {
    "narrative_flow": """You are an expert dubbing translator. Translate video narration for dubbing.
CRITICAL RULES:
1. Translate ONLY what is in "original" — NEVER add, invent, or continue from context
2. Short original (1-3 words) → short translation (similar word count)
3. {lang_instruction}
4. Match timing: the "max_words" field = MAXIMUM words allowed. Your translation MUST have ≤ max_words words.
5. Punchy, cinematic narrator style — like a popular YouTube channel
6. Use PREVIOUS CONTEXT for continuity, but translate ONLY the current segment
7. PRESERVE character names (don't translate proper nouns)
8. EMOTION hints: segment mood tells you the emotional tone — let it influence word choice
Return JSON: {{"segments": [{{"id": X, "dubbed_text": "..."}}]}}""",

    "dramatic_dialogue": """You are an expert dubbing translator for drama/romance content.
CRITICAL RULES:
1. Translate EXACTLY what is in "original" — no adding, no inventing
2. {lang_instruction}
3. Dialogue must sound SPOKEN and emotional, not written
4. RESPECT max_words limit strictly — short is better than too long
5. NEVER expand a short original into a long dubbed line
6. Use context for character voice consistency
7. PRESERVE character names
8. Match the MOOD — angry lines should sound angry, sad lines should sound sad
Return JSON: {{"segments": [{{"id": X, "dubbed_text": "..."}}]}}""",

    "conversational": """You are an expert dubbing translator for interview/talk content.
RULES:
1. Translate what is said, don't add content
2. {lang_instruction}
3. Conversational, natural flow
4. RESPECT max_words limit
5. Use context for continuity
Return JSON: {{"segments": [{{"id": X, "dubbed_text": "..."}}]}}"""
}

BATCH_SIZE = 20


def translate(dir_result, work_dir, target_lang="Hindi"):
    """
    Context-aware translation with rolling window and emotion hints.
    """
    c = _gc()
    segments = dir_result["segments"]
    style = dir_result.get("translation_style", "narrative_flow")
    summary = dir_result.get("narrative_summary", "")
    
    lang_inst = LANG_INSTRUCTIONS.get(target_lang, LANG_INSTRUCTIONS["English"])
    prompt_template = STYLE_PROMPTS.get(style, STYLE_PROMPTS["narrative_flow"])
    prompt = prompt_template.format(lang_instruction=lang_inst)
    
    wps = WORDS_PER_SEC.get(target_lang, 2.5)
    
    logger.info(f"[TRANSLATOR] {len(segments)} segments → {target_lang} | Style: {style}")
    logger.info(f"[TRANSLATOR] Word rate: {wps} words/sec | Primary: {PRIMARY_MODEL}")
    
    batches = [segments[i:i+BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
    model = PRIMARY_MODEL
    
    for bi, batch in enumerate(batches):
        # Build context from previous batch's last 3 segments
        context = ""
        if bi > 0:
            prev_batch = batches[bi - 1]
            context_segs = prev_batch[-3:]  # Last 3 from previous batch
            context_lines = []
            for cs in context_segs:
                orig = cs.get("text", "")[:50]
                dubbed = cs.get("dubbed_text", "")[:50]
                context_lines.append(f"  [{cs['id']}] {orig} → {dubbed}")
            context = "PREVIOUS CONTEXT (for continuity, do NOT re-translate):\n" + "\n".join(context_lines) + "\n\n"
        
        items = [{
            "id": s["id"],
            "duration": round(s["end"] - s["start"], 2),
            "max_words": max(3, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "original": s["text"]
        } for s in batch]
        
        user_msg = f"""Story: {summary}
{context}Translate these {len(items)} segments to {target_lang}:
{json.dumps(items, ensure_ascii=False)}"""
        
        for attempt in range(5):
            try:
                resp = c.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
                    temperature=0.3, max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                translated = json.loads(resp.choices[0].message.content)
                seg_map = {s["id"]: s for s in batch}
                
                for t in translated.get("segments", []):
                    if t["id"] in seg_map:
                        text = t.get("dubbed_text", "")
                        # Enforce word limit
                        dur = seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]
                        max_w = max(3, int(dur * wps))
                        words = text.split()
                        if len(words) > max_w + 2:
                            text = " ".join(words[:max_w])
                        seg_map[t["id"]]["dubbed_text"] = text
                
                logger.info(f"[TRANSLATOR] Batch {bi+1}/{len(batches)} ✓ ({model.split('/')[-1]})")
                break
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    if model == PRIMARY_MODEL:
                        logger.warning(f"[TRANSLATOR] Scout rate limited → {FALLBACK_MODEL}")
                        model = FALLBACK_MODEL
                        continue
                    wait = min(2 ** attempt, 10)
                    logger.warning(f"[TRANSLATOR] Rate limited, retry in {wait}s...")
                    time.sleep(wait)
                elif "model" in err_str.lower() and model == PRIMARY_MODEL:
                    logger.warning(f"[TRANSLATOR] Scout unavailable → {FALLBACK_MODEL}")
                    model = FALLBACK_MODEL
                    continue
                else:
                    wait = min(2 ** attempt, 10)
                    logger.warning(f"[TRANSLATOR] Batch {bi+1} error: {e}, retry in {wait}s")
                    time.sleep(wait)
    
    # Fill missing
    for s in segments:
        if not s.get("dubbed_text"):
            s["dubbed_text"] = s["text"]
    
    sp = os.path.join(work_dir, "translated_raw.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({
            "segments": segments, "style": style,
            "target_lang": target_lang, "model_used": model,
            "words_per_sec": wps
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[TRANSLATOR] Preview:")
    for s in segments[:5]:
        logger.info(f"  [{s['start']:.1f}s] {s['text'][:35]} → {s.get('dubbed_text', '')[:45]}")
    
    return {"raw_script_path": sp}
