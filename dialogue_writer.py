"""
DIALOGUE WRITER v4 — Industry-grade script polishing:
  1. Character voice bible — defines HOW each character speaks
  2. Scene-grouped writing — writes per scene, not arbitrary batches
  3. Pronunciation hints for Edge TTS (commas, elongation, emphasis)
  4. Emotion-aware delivery markers
  5. Strict word budgeting with syllable awareness
  6. llama-3.3-70b for quality
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

WORDS_PER_SEC = {
    "Hindi": 2.5, "Tamil": 2.0, "Telugu": 2.2, "Bengali": 2.3,
    "Marathi": 2.4, "Gujarati": 2.4, "Kannada": 2.1, "Malayalam": 1.9,
    "Nepali": 2.5, "Urdu": 2.5, "English": 3.0, "Spanish": 2.8,
    "French": 2.7, "Portuguese": 2.6, "German": 2.5, "Japanese": 3.5,
    "Korean": 2.8, "Arabic": 2.5, "Turkish": 2.6,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARACTER VOICE BIBLE GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_voice_bible(voice_plan, target_lang):
    """
    Build a 'voice bible' — specific instructions for how each character speaks.
    This gets sent with EVERY writing batch for consistency.
    """
    bible = "CHARACTER VOICE BIBLE (MUST follow for every line):\n"
    
    character_rules = {
        "NARRATOR": {
            "style": "Smooth, cinematic narration. Like a professional documentary narrator.",
            "dos": "clear pacing, dramatic pauses (use commas), vivid descriptions",
            "donts": "casual slang, filler words, character-specific vocabulary"
        },
        "FATHER": {
            "style": "Authoritative, firm but caring. Short, decisive sentences.",
            "dos": "commanding tone, wisdom, protective language",
            "donts": "slang, hesitation, overly emotional outbursts"
        },
        "MOTHER": {
            "style": "Warm, nurturing, concerned. Flowing, emotional sentences.",
            "dos": "endearments (beta/beti for Hindi), concern, warmth",
            "donts": "aggressive language, cold detachment"
        },
        "HERO": {
            "style": "Brave, determined, warm. Medium-length confident sentences.",
            "dos": "decisive words, emotional depth, occasional humor",
            "donts": "weakness without purpose, monotone"
        },
        "HEROINE": {
            "style": "Strong, expressive, emotionally rich. Variable sentence length.",
            "dos": "emotional range, independent spirit, genuine reactions",
            "donts": "helpless language (unless plot requires)"
        },
        "VILLAIN": {
            "style": "Cold, calculating, menacing. Short, sharp sentences.",
            "dos": "threats, sarcasm, control language, clipped phrasing",
            "donts": "warmth, hesitation, long flowery sentences"
        },
        "OLD_MAN": {
            "style": "Wise, measured, slow-paced. Thoughtful, complete sentences.",
            "dos": "wisdom, proverbs, measured pace, respected vocabulary",
            "donts": "slang, rushed language, modern references"
        },
        "OLD_WOMAN": {
            "style": "Gentle, wise, nurturing. Warm, unhurried speech.",
            "dos": "gentle wisdom, care, traditional expressions",
            "donts": "aggressive language, modern slang"
        },
        "YOUNG_MAN": {
            "style": "Energetic, casual, direct. Short punchy sentences.",
            "dos": "casual language, energy, occasional English words (Hindi)",
            "donts": "formal/stiff language"
        },
        "YOUNG_WOMAN": {
            "style": "Bright, expressive, modern. Natural flowing speech.",
            "dos": "expressiveness, modern vocabulary, emotional reactions",
            "donts": "overly formal, stilted"
        },
        "BOY": {
            "style": "Simple vocabulary, curious, high energy. Very short sentences.",
            "dos": "simple words, questions, excitement, wide-eyed wonder",
            "donts": "complex vocabulary, adult phrasing"
        },
        "GIRL": {
            "style": "Sweet, curious, playful. Simple short sentences.",
            "dos": "simple words, playfulness, innocence",
            "donts": "adult vocabulary, complex sentences"
        },
        "SON": {
            "style": "Respectful to parents, casual with peers. Medium sentences.",
            "dos": "respect markers (Hindi: 'ji'), youth energy when with friends",
            "donts": "disrespectful tone (unless plot requires)"
        },
        "DAUGHTER": {
            "style": "Warm, sometimes emotional. Variable sentence length.",
            "dos": "affection, expressiveness, family vocabulary",
            "donts": "cold detachment (unless plot requires)"
        },
    }
    
    for v in voice_plan:
        vid = v.get("voice_id", "NARRATOR")
        rules = character_rules.get(vid, character_rules.get("NARRATOR"))
        personality = v.get("personality", "")
        
        bible += f"\n  {vid} ({v.get('gender','?')}, {v.get('age','?')}):\n"
        bible += f"    Style: {rules['style']}\n"
        if personality:
            bible += f"    Personality: {personality}\n"
        bible += f"    DO: {rules['dos']}\n"
        bible += f"    DON'T: {rules['donts']}\n"
    
    return bible


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WRITER PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WRITER_PROMPT = """You are an expert DIALOGUE WRITER for dubbed video content in {target_lang}.
You receive draft translations and REWRITE them as polished, natural SPOKEN dialogue.

YOUR JOB IS TO MAKE IT SOUND REAL — like actual people talking, not a textbook.

⚠️ WORD COUNT RULES (STRICT — violations = rejection):
- Each segment has "max_words" = MAXIMUM words allowed
- Your dubbed_text MUST have ≤ max_words words
- If draft is too long → COMPRESS meaning, cut filler
- SHORT IS ALWAYS BETTER than too long (too-long = robotic TTS)

🎭 EMOTION DELIVERY (match the mood field):
- neutral → calm, steady, clear
- happy/excited → upbeat! Exclamations! Energy!
- sad/emotional → softer... shorter phrases... pauses
- angry/tense → SHARP. Clipped. No filler. Direct.
- wise/gentle → Measured, warm, slightly formal
- urgent → NOW! Direct commands. No fluff.
- fearful → H-hesitant? Broken phrases...
- romantic → Warm, intimate, tender words
- dramatic → Bold statements. Impactful pauses.
- menacing → Cold. Calculating. Threat in every word.

📝 PUNCTUATION FOR TTS (these control how it sounds!):
- Use commas for natural pauses: "Beta, market jaao" ✓
- Use "..." for hesitation/emotion: "Main... samajh gaya" ✓
- Use "!" for emphasis/energy: "Chalo!" ✓
- Use "?" for questioning tone: "Kyun?" ✓

✍️ SCRIPT RULES:
- Write ONLY in Roman script (English letters)
- NEVER write in native scripts (no Devanagari, no Chinese)
- Keep character names unchanged
{lang_specific}

{voice_bible}

Return JSON: {{"segments": [{{"id": X, "dubbed_text": "rewritten dialogue"}}]}}"""

LANG_SPECIFIC = {
    "Hindi": """HINDI DIALOGUE:
- Natural Hinglish as Indians actually speak
- Keep English words: okay, sorry, please, time, phone, market, school
- Narrator: NO slang (no "yaar", "arey", "na")
- Characters CAN use light slang: "Arey bhai!", "Haan yaar"
- Elderly: Use more formal Hindi: "Aap", "Ji"
- Young: Casual: "Tu", "Tum", English mix""",
    "Nepali": """NEPALI DIALOGUE:
- Natural spoken Nepali with English mix
- "Hajur" for respectful address
- Keep English words Nepalis use: office, school, phone, okay
- Roman script only""",
    "Tamil": "Natural spoken Tamil with Tanglish. Roman script only.",
    "Telugu": "Natural spoken Telugu. Roman script only.",
    "Bengali": "Natural spoken Bengali. Roman script only.",
    "English": "Natural engaging English. Conversational, expressive.",
}


def _call_llm(prompt, msg, max_retries=4):
    """Call LLM with retry and model fallback."""
    c = _gc()
    models = ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"]
    
    for mi, model in enumerate(models):
        for attempt in range(max_retries if mi == 0 else 2):
            try:
                resp = c.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": msg}],
                    temperature=0.3, max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                return json.loads(resp.choices[0].message.content), model
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    if mi < len(models) - 1:
                        logger.warning(f"[WRITER] {model} rate limited → trying next model")
                        break  # Try next model
                    wait = min(2 ** attempt, 12)
                    time.sleep(wait)
                elif "model" in err.lower():
                    logger.warning(f"[WRITER] {model} unavailable → trying next")
                    break
                else:
                    wait = min(2 ** attempt, 8)
                    time.sleep(wait)
    return None, "none"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN REWRITE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rewrite(segments, work_dir, target_lang="Hindi", narrative_summary="", 
            mood_arc=None, voice_plan=None, scenes=None):
    """
    Scene-aware dialogue rewriting with character voice bible.
    """
    wps = WORDS_PER_SEC.get(target_lang, 2.5)
    
    logger.info(f"[WRITER] Rewriting {len(segments)} segments as polished {target_lang} dialogue")
    logger.info(f"[WRITER] Word rate: {wps}/sec | Voice bible mode ✓")
    
    # Build voice bible
    vp = voice_plan or [{"voice_id": "NARRATOR", "gender": "male", "age": "adult", "personality": "narrator"}]
    voice_bible = _build_voice_bible(vp, target_lang)
    
    lang_spec = LANG_SPECIFIC.get(target_lang, LANG_SPECIFIC.get("English", ""))
    prompt = WRITER_PROMPT.format(
        target_lang=target_lang, lang_specific=lang_spec, voice_bible=voice_bible
    )
    
    # Group by scene if available
    groups = []
    group_labels = []
    
    if scenes:
        logger.info(f"[WRITER] Writing by scene ({len(scenes)} scenes)")
        for scene in scenes:
            scene_segs = [s for s in segments if s["id"] in scene.get("segment_ids", [])]
            if scene_segs:
                # Split large scenes into chunks of 25
                for i in range(0, len(scene_segs), 25):
                    chunk = scene_segs[i:i+25]
                    groups.append(chunk)
                    group_labels.append(f"Scene {scene['scene_id']+1}")
    
    if not groups:
        groups = [segments[i:i+20] for i in range(0, len(segments), 20)]
        group_labels = [f"Batch {i+1}" for i in range(len(groups))]
    
    # Track character examples for consistency across groups
    char_examples = {}
    
    for gi, (group, label) in enumerate(zip(groups, group_labels)):
        items = [{
            "id": s["id"],
            "duration_sec": round(s["end"] - s["start"], 2),
            "max_words": max(2, int((s["end"] - s["start"]) * wps)),
            "speaker": s.get("speaker", "NARRATOR"),
            "mood": s.get("mood", "neutral"),
            "draft": s.get("dubbed_text", s.get("text", ""))
        } for s in group]
        
        # Character consistency context
        char_context = ""
        if char_examples:
            char_context = "\nCHARACTER VOICE EXAMPLES (maintain same style):\n"
            for spk, examples in char_examples.items():
                recent = examples[-2:]
                for ex in recent:
                    char_context += f"  {spk} ({ex['mood']}): \"{ex['text'][:50]}\"\n"
        
        user_msg = f"""Story: {narrative_summary}
Mood arc: {mood_arc or 'not specified'}
{char_context}
Rewrite these {len(items)} segments ({label}). RESPECT max_words!
{json.dumps(items, ensure_ascii=False)}"""
        
        result, model = _call_llm(prompt, user_msg)
        
        if result:
            seg_map = {s["id"]: s for s in group}
            for t in result.get("segments", []):
                if t["id"] in seg_map:
                    new_text = t.get("dubbed_text", "")
                    if new_text:
                        # Enforce word limit
                        max_w = max(2, int((seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]) * wps))
                        words = new_text.split()
                        if len(words) > max_w + 2:
                            new_text = " ".join(words[:max_w])
                        seg_map[t["id"]]["dubbed_text"] = new_text
                        
                        # Track for character consistency
                        speaker = seg_map[t["id"]].get("speaker", "NARRATOR")
                        mood = seg_map[t["id"]].get("mood", "neutral")
                        if speaker not in char_examples:
                            char_examples[speaker] = []
                        char_examples[speaker].append({"text": new_text, "mood": mood})
            
            logger.info(f"[WRITER] {label} ✓ ({model.split('/')[-1]})")
        else:
            logger.warning(f"[WRITER] {label} — keeping draft translations")
    
    # Save
    sp = os.path.join(work_dir, "dubbed_script.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "target_lang": target_lang}, f, indent=2, ensure_ascii=False)
    
    # Preview
    logger.info(f"[WRITER] {'━' * 55}")
    raw_path = os.path.join(work_dir, "translated_raw.json")
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = {s["id"]: s.get("dubbed_text", "") for s in json.load(f)["segments"]}
        shown = 0
        for s in segments:
            r = raw.get(s["id"], "")
            d = s.get("dubbed_text", "")
            if r != d and shown < 5:
                logger.info(f"  [{s['start']:.1f}s|{s.get('speaker','?')}|{s.get('mood','?')}]")
                logger.info(f"    DRAFT: {r[:55]}")
                logger.info(f"    FINAL: {d[:55]}")
                shown += 1
    logger.info(f"[WRITER] {'━' * 55}")
    
    return {"script_path": sp}
