"""
DIALOGUE WRITER v3 — Emotion-aware rewriting with:
- Emotion markers for Fish Audio TTS
- Character personality tracking across segments
- Better syllable-aware word budgeting
- Natural spoken dialogue polishing
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

# Language-specific word rates (should match translator.py)
WORDS_PER_SEC = {
    "Hindi": 2.5, "Tamil": 2.0, "Telugu": 2.2, "Bengali": 2.3,
    "Marathi": 2.4, "Gujarati": 2.4, "Kannada": 2.1, "Malayalam": 1.9,
    "Nepali": 2.5, "Urdu": 2.5, "English": 3.0, "Spanish": 2.8,
    "French": 2.7, "Portuguese": 2.6, "German": 2.5, "Japanese": 3.5,
    "Korean": 2.8, "Arabic": 2.5, "Turkish": 2.6,
}

WRITER_PROMPT = """You are an expert DIALOGUE WRITER for dubbed video content.
You receive raw translations and REWRITE them as polished, natural spoken {target_lang}.

YOUR JOB:
- Rewrite raw translations to sound natural and spoken
- Match MOOD of each segment
- CRITICAL: Respect the MAX WORDS limit — this is non-negotiable!

⚠️ WORD COUNT RULES (STRICT — output WILL BE REJECTED if violated):
- Each segment has a "max_words" field = the MAXIMUM words allowed
- Your dubbed_text MUST have ≤ max_words words
- If raw translation is too long, CUT IT DOWN. Remove filler, compress meaning.
- Short is better than too long. TTS will sound robotic if sped up.
- Example: max_words=8 → "Ek baar ek aadmi tha jo bahut pareshan tha" ✗ (10 words)
                        → "Ek aadmi bahut pareshan tha" ✓ (5 words)

🎭 EMOTION-AWARE DELIVERY:
- Each segment has a "mood" field. Write dialogue that SOUNDS like that emotion.
- neutral: calm, steady pacing
- happy/excited: upbeat words, energetic phrasing, exclamations
- sad/emotional: softer words, shorter phrases, pauses (use "...")
- angry/tense: sharp, clipped sentences, strong words, no filler
- wise/gentle: measured pace, warm words, slightly formal
- urgent: direct commands, no fluff, imperative tone
- fearful: hesitant, broken phrases, questioning tone

🗣️ CHARACTER CONSISTENCY:
- NARRATOR lines: smooth storytelling, NO casual fillers (no "yaar", "arey", "na")
- CHARACTER lines: can use natural fillers sparingly
- Keep each character's voice consistent throughout:
  - If FATHER was authoritative in segment 5, keep him authoritative in segment 25
  - If SON was curious/energetic, maintain that energy
  - VILLAIN: menacing, cold; HERO: brave, warm; OLD_MAN: wise, measured

✍️ SCRIPT RULES:
- Write ONLY in Roman script (English letters)
- NEVER write in Devanagari (Hindi script) or native scripts
- "Papa ne kaha" ✓ | "पापा ने कहा" ✗

{lang_specific}

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "rewritten dialogue"}}]}}"""

LANG_SPECIFIC = {
    "Hindi": """HINDI/HINGLISH:
- Natural Hinglish: "Beta, jao market se ghadi ki keemat puchh ke aao" ✓
- Keep English words Indians use: okay, sorry, please, time, market
- NEVER use formal Hindi ("aap", "kripya") unless character is elderly
- Use "Bhai", "Yaar" ONLY for young friend characters, NOT narrator""",
    "Nepali": """NEPALI:
- Natural spoken Nepali with English mix as Nepalis actually speak
- Use "Hajur" for respectful address, "Bhai" for brothers
- Keep common English words Nepalis use: office, school, phone, time
- Write in Roman script""",
    "Tamil": "Natural spoken Tamil with Tanglish. Roman script only.",
    "Telugu": "Natural spoken Telugu. Roman script only.",
    "Bengali": "Natural spoken Bengali. Roman script only.",
    "English": "Natural, engaging English. Conversational tone.",
    "Spanish": "Natural Latin American Spanish.",
    "French": "Natural spoken French.",
}

BATCH_SIZE = 20


def rewrite(segments, work_dir, target_lang="Hindi", narrative_summary="", mood_arc=None):
    """
    Emotion-aware dialogue rewriting with character consistency tracking.
    """
    c = _gc()
    wps = WORDS_PER_SEC.get(target_lang, 2.5)
    
    logger.info(f"[WRITER] Rewriting {len(segments)} segments as polished {target_lang} dialogue...")
    logger.info(f"[WRITER] Word rate: {wps} words/sec | Emotion-aware mode ✓")
    
    lang_spec = LANG_SPECIFIC.get(target_lang, LANG_SPECIFIC.get("English", ""))
    prompt = WRITER_PROMPT.format(target_lang=target_lang, lang_specific=lang_spec)
    
    batches = [segments[i:i+BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
    
    # Track character voices for consistency
    char_examples = {}  # speaker → list of rewritten examples
    
    for bi, batch in enumerate(batches):
        items = [{
            "id": s["id"],
            "duration": round(s["end"] - s["start"], 2),
            "max_words": max(3, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "raw_translation": s.get("dubbed_text", s["text"])
        } for s in batch]
        
        # Build character consistency context from previous batches
        char_context = ""
        if char_examples:
            char_context = "\n\nCHARACTER VOICE EXAMPLES (maintain same style):\n"
            for spk, examples in char_examples.items():
                recent = examples[-2:]  # Last 2 examples per character
                for ex in recent:
                    char_context += f"  {spk} ({ex['mood']}): \"{ex['text'][:60]}\"\n"
        
        user_msg = f"""Story: {narrative_summary}
Mood arc: {mood_arc or 'not specified'}
{char_context}
Rewrite these {len(items)} segments. RESPECT max_words for each!
{json.dumps(items, ensure_ascii=False)}"""
        
        for attempt in range(5):
            try:
                resp = c.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
                    temperature=0.35, max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                rewritten = json.loads(resp.choices[0].message.content)
                seg_map = {s["id"]: s for s in batch}
                
                for t in rewritten.get("segments", []):
                    if t["id"] in seg_map:
                        new_text = t.get("dubbed_text", "")
                        # Enforce word limit as safety net
                        max_w = max(3, int((seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]) * wps))
                        words = new_text.split()
                        if len(words) > max_w + 2:  # small grace margin
                            new_text = " ".join(words[:max_w])
                            logger.debug(f"[WRITER] Trimmed seg {t['id']} from {len(words)} to {max_w} words")
                        seg_map[t["id"]]["dubbed_text"] = new_text
                        
                        # Track for character consistency
                        speaker = seg_map[t["id"]].get("speaker", "NARRATOR")
                        mood = seg_map[t["id"]].get("mood", "neutral")
                        if speaker not in char_examples:
                            char_examples[speaker] = []
                        char_examples[speaker].append({
                            "text": new_text,
                            "mood": mood
                        })
                
                logger.info(f"[WRITER] Batch {bi+1}/{len(batches)} ✓")
                break
                
            except Exception as e:
                if "429" in str(e):
                    wait = min(2 ** attempt, 15)
                    logger.warning(f"[WRITER] Rate limited, retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"[WRITER] Error: {e}, attempt {attempt+1}/5")
                    if attempt >= 2:
                        # Fallback to 8b model
                        try:
                            logger.info(f"[WRITER] Falling back to llama-3.1-8b-instant...")
                            resp = c.chat.completions.create(
                                model='llama-3.1-8b-instant',
                                messages=[{'role': 'system', 'content': prompt},
                                         {'role': 'user', 'content': user_msg}],
                                temperature=0.35, max_tokens=4000,
                                response_format={'type': 'json_object'}
                            )
                            rewritten = json.loads(resp.choices[0].message.content)
                            seg_map = {s['id']: s for s in batch}
                            for t in rewritten.get('segments', []):
                                if t['id'] in seg_map:
                                    new_text = t.get('dubbed_text', '')
                                    max_w = max(3, int((seg_map[t['id']]['end'] - seg_map[t['id']]['start']) * wps))
                                    words = new_text.split()
                                    if len(words) > max_w + 2:
                                        new_text = ' '.join(words[:max_w])
                                    seg_map[t['id']]['dubbed_text'] = new_text
                            logger.info(f'[WRITER] Batch {bi+1}/{len(batches)} ✓ (fallback 8b)')
                            break
                        except Exception as e3:
                            logger.warning(f'[WRITER] Fallback also failed: {e3}')
                    time.sleep(2 ** attempt)
                    if attempt >= 4:
                        logger.warning(f"[WRITER] Keeping raw for batch {bi+1}")
                        break
    
    # Save
    sp = os.path.join(work_dir, "dubbed_script.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "target_lang": target_lang}, f, indent=2, ensure_ascii=False)
    
    # Log before/after preview
    raw_path = os.path.join(work_dir, "translated_raw.json")
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = {s["id"]: s.get("dubbed_text", "") for s in json.load(f)["segments"]}
        logger.info(f"[WRITER] ✓ Preview (before → after):")
        shown = 0
        for s in segments:
            r = raw.get(s["id"], "")
            d = s.get("dubbed_text", "")
            if r != d and shown < 5:
                spk = s.get("speaker", "?")
                mood = s.get("mood", "?")
                logger.info(f"  [{s['start']:.1f}s|{spk}|{mood}] RAW:   {r[:55]}")
                logger.info(f"                       FINAL: {d[:55]}")
                shown += 1
    
    return {"script_path": sp}
