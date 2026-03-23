"""
DIALOGUE WRITER v2 — Rewrites raw translations with:
- Mood-aware dialogue polishing
- STRICT word count limits based on duration (prevents speed-up)
- No fillers in narrator lines
- Roman script enforcement
"""
import os, json, logging, time
from groq import Groq
logger = logging.getLogger(__name__)
_client = None

def _gc():
    global _client
    if not _client: _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

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

MOOD GUIDELINES:
- neutral: calm, steady | happy/excited: energetic | sad/emotional: softer
- angry/tense: sharp, clipped | wise/gentle: measured, warm | urgent: direct

SPEAKER RULES:
- NARRATOR lines: smooth storytelling, NO casual fillers (no "yaar", "arey", "na")
- CHARACTER lines (FATHER/SON/OLD_MAN etc): can use natural fillers sparingly
- FATHER: authoritative, warm | SON/BOY: curious, energetic | OLD_MAN: measured, wise

SCRIPT RULES:
- Write ONLY in Roman script (English letters)
- NEVER write in Devanagari (Hindi script)
- "Papa ne kaha" ✓ | "पापा ने कहा" ✗

{lang_specific}

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "rewritten dialogue"}}]}}"""

LANG_SPECIFIC = {
    "Hindi": """HINDI/HINGLISH:
- Natural Hinglish: "Beta, jao market se ghadi ki keemat puchh ke aao" ✓
- Keep English words Indians use: okay, sorry, please, time, market
- NEVER use formal Hindi ("aap", "kripya") unless character is elderly""",
    "Tamil": "Natural spoken Tamil with Tanglish. Roman script only.",
    "Telugu": "Natural spoken Telugu. Roman script only.",
    "Bengali": "Natural spoken Bengali. Roman script only.",
    "English": "Natural, engaging English. Conversational tone.",
    "Spanish": "Natural Latin American Spanish.",
    "French": "Natural spoken French.",
}

BATCH_SIZE = 20
WORDS_PER_SEC = 2.8  # conservative — prevents speed-up

def rewrite(segments, work_dir, target_lang="Hindi", narrative_summary="", mood_arc=None):
    c = _gc()
    logger.info(f"[WRITER] Rewriting {len(segments)} segments as polished {target_lang} dialogue...")
    
    lang_spec = LANG_SPECIFIC.get(target_lang, LANG_SPECIFIC.get("English", ""))
    prompt = WRITER_PROMPT.format(target_lang=target_lang, lang_specific=lang_spec)

    batches = [segments[i:i+BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
    
    for bi, batch in enumerate(batches):
        items = [{
            "id": s["id"],
            "duration": round(s["end"] - s["start"], 2),
            "max_words": max(3, int((s["end"] - s["start"]) * WORDS_PER_SEC)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "raw_translation": s.get("dubbed_text", s["text"])
        } for s in batch]

        user_msg = f"""Story: {narrative_summary}
Mood arc: {mood_arc or 'not specified'}

Rewrite these {len(items)} segments. RESPECT max_words for each!
{json.dumps(items, ensure_ascii=False)}"""

        for attempt in range(5):
            try:
                resp = c.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"system","content":prompt}, {"role":"user","content":user_msg}],
                    temperature=0.35, max_tokens=3000,
                    response_format={"type":"json_object"}
                )
                rewritten = json.loads(resp.choices[0].message.content)
                seg_map = {s["id"]: s for s in batch}
                
                for t in rewritten.get("segments", []):
                    if t["id"] in seg_map:
                        new_text = t.get("dubbed_text", "")
                        # Enforce word limit as safety net
                        max_w = max(3, int((seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]) * WORDS_PER_SEC))
                        words = new_text.split()
                        if len(words) > max_w + 2:  # small grace margin
                            new_text = " ".join(words[:max_w])
                            logger.debug(f"[WRITER] Trimmed seg {t['id']} from {len(words)} to {max_w} words")
                        seg_map[t["id"]]["dubbed_text"] = new_text
                
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
                                messages=[{'role':'system','content':prompt}, {'role':'user','content':user_msg}],
                                temperature=0.35, max_tokens=3000,
                                response_format={'type':'json_object'})
                            rewritten = json.loads(resp.choices[0].message.content)
                            seg_map = {s['id']: s for s in batch}
                            for t in rewritten.get('segments', []):
                                if t['id'] in seg_map:
                                    new_text = t.get('dubbed_text', '')
                                    max_w = max(3, int((seg_map[t['id']]['end'] - seg_map[t['id']]['start']) * WORDS_PER_SEC))
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
            raw = {s["id"]: s.get("dubbed_text","") for s in json.load(f)["segments"]}
        logger.info(f"[WRITER] ✓ Preview (before → after):")
        shown = 0
        for s in segments:
            r = raw.get(s["id"], "")
            d = s.get("dubbed_text", "")
            if r != d and shown < 5:
                logger.info(f"  [{s['start']:.1f}s] RAW:   {r[:55]}")
                logger.info(f"         FINAL: {d[:55]}")
                shown += 1
    
    return {"script_path": sp}
