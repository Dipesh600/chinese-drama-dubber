"""
DIALOGUE WRITER v6 — Industry-grade script polishing using LLM provider:
  1. Character voice bible — defines HOW each character speaks
  2. Scene-grouped writing — writes per scene, not arbitrary batches
  3. Pronunciation hints for Edge TTS (commas, elongation, emphasis)
  4. Emotion-aware delivery markers
  5. Strict word budgeting with syllable awareness
  6. Async scene-based parallel processing
  7. Uses config for shared settings
"""
import os, json, logging, time
import asyncio
from typing import List, Dict, Optional

from config import get_wps, get_lang_instruction, TRANSLATOR_BATCH_SIZE
from llm_provider import get_llm
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHARACTER VOICE BIBLE GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_voice_bible(voice_plan, target_lang):
    """Build a 'voice bible' — specific instructions for how each character speaks."""
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

⚠️ WORD COUNT RULES:
- Each segment has "max_words" = SOFT maximum words allowed
- If draft is within max_words+3 → keep it as-is (don't chop!)
- Only compress if significantly over budget; cut filler NOT meaning
- PRESERVE ALL KEY NOUNS and VERBS — meaning loss = rejection
- SHORT IS ONLY BETTER if the original is wordy/rambling

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
- Write COMPLETE sentences in Hindi using Roman script
- DO NOT mix English and Hindi (e.g., "Razor, ek shell" is WRONG → write "Razor ek shell tha")
- Only keep English words that Indians naturally use: okay, sorry, please, phone, school, time, office, TV
- For names/events: use phonetic Hindi ("Seizar" not "Caesar", "Reyzor" not "Razor")
- Narrator: formal Hindi, no slang
- Characters: natural spoken Hindi, light Hinglish OK
- Preserve FULL meaning — never drop nouns or verbs""",
    "Nepali": """NEPALI DIALOGUE:
- Natural spoken Nepali with English mix
- "Hajur" for respectful address
- Keep English words Nepalis use: office, school, phone, okay
- Roman script only""",
    "Tamil": "Natural spoken Tamil with Tanglish. Roman script only. Write full Tamil sentences, not English fragments.",
    "Telugu": "Natural spoken Telugu. Roman script only. Write full Telugu sentences.",
    "Bengali": "Natural spoken Bengali. Roman script only. Write full Bengali sentences.",
    "English": "Natural engaging English. Conversational, expressive.",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASYNC LLM CALLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_llm_sync(prompt: str, message: str) -> Optional[Dict]:
    """Synchronous wrapper for LLM call."""
    llm = get_llm()
    return llm.chat(prompt, message, temperature=0.3, max_tokens=4000, json_response=True)


async def _call_llm_async(prompt: str, message: str) -> Optional[Dict]:
    """Async wrapper using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm_sync, prompt, message)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARALLEL SCENE PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _rewrite_scene(
    scene_id: int,
    scene_name: str,
    segments: List[Dict],
    target_lang: str,
    narrative_summary: str,
    mood_arc: list,
    voice_bible: str,
    lang_spec: str,
    wps: float,
    char_examples: dict,
) -> Dict:
    """Rewrite a single scene's dialogue."""
    logger.info(f"[WRITER] Scene {scene_name} ({len(segments)} segments)...")

    prompt = WRITER_PROMPT.format(
        target_lang=target_lang,
        lang_specific=lang_spec,
        voice_bible=voice_bible
    )

    # Build character consistency context
    char_context = ""
    if char_examples:
        char_context = "\nCHARACTER VOICE EXAMPLES (maintain same style):\n"
        for spk, examples in char_examples.items():
            recent = examples[-2:]
            for ex in recent:
                char_context += f"  {spk} ({ex['mood']}): \"{ex['text'][:50]}\"\n"

    items = [{
        "id": s["id"],
        "duration_sec": round(s["end"] - s["start"], 2),
        "max_words": max(2, int((s["end"] - s["start"]) * wps)),
        "speaker": s.get("speaker", "NARRATOR"),
        "mood": s.get("mood", "neutral"),
        "draft": s.get("dubbed_text", s.get("text", ""))
    } for s in segments]

    user_msg = f"""Story: {narrative_summary}
Mood arc: {mood_arc or 'not specified'}
{char_context}
Rewrite these {len(items)} segments ({scene_name}). RESPECT max_words!
{json.dumps(items, ensure_ascii=False)}"""

    result = await _call_llm_async(prompt, user_msg)

    if result:
        seg_map = {s["id"]: s for s in segments}
        for t in result.get("segments", []):
            if t["id"] in seg_map:
                new_text = t.get("dubbed_text", "")
                if new_text:
                    # Allow up to max_words + 5 buffer before truncating
                    # Don't truncate if already within budget (avoids destroying meaning)
                    max_w = max(2, int((seg_map[t["id"]]["end"] - seg_map[t["id"]]["start"]) * wps))
                    words = new_text.split()
                    if len(words) > max_w + 5:
                        # Gentle truncation: keep meaning, cut trailing filler
                        new_text = " ".join(words[:max_w])
                    seg_map[t["id"]]["dubbed_text"] = new_text

                    # Track for character consistency
                    speaker = seg_map[t["id"]].get("speaker", "NARRATOR")
                    mood = seg_map[t["id"]].get("mood", "neutral")
                    if speaker not in char_examples:
                        char_examples[speaker] = []
                    char_examples[speaker].append({"text": new_text, "mood": mood})

        logger.info(f"[WRITER] Scene {scene_name} ✓")
        return {"scene_id": scene_id, "segments": segments, "char_examples": char_examples}
    else:
        logger.warning(f"[WRITER] Scene {scene_name} — keeping draft translations")
        return {"scene_id": scene_id, "segments": segments, "char_examples": char_examples}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN REWRITE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rewrite(segments, work_dir, target_lang="Hindi", narrative_summary="",
            mood_arc=None, voice_plan=None, scenes=None):
    """Scene-aware dialogue rewriting with parallel processing."""
    wps = get_wps(target_lang)

    logger.info(f"[WRITER] Rewriting {len(segments)} segments as polished {target_lang} dialogue")
    logger.info(f"[WRITER] Word rate: {wps}/sec | Voice bible mode ✓ | Parallel scenes")

    # Build voice bible
    vp = voice_plan or [{"voice_id": "NARRATOR", "gender": "male", "age": "adult", "personality": "narrator"}]
    voice_bible = _build_voice_bible(vp, target_lang)
    lang_spec = LANG_SPECIFIC.get(target_lang, LANG_SPECIFIC.get("English", ""))

    # Group by scene
    scene_groups = []
    if scenes:
        logger.info(f"[WRITER] Processing by scene ({len(scenes)} scenes)")
        for scene in scenes:
            scene_segs = [s for s in segments if s["id"] in scene.get("segment_ids", [])]
            if scene_segs:
                scene_groups.append((scene["scene_id"], f"Scene {scene['scene_id']+1}", scene_segs))

    if not scene_groups:
        # Split into pseudo-scenes for parallel processing
        chunk_size = 25
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i+chunk_size]
            scene_groups.append((i // chunk_size, f"Batch {i//chunk_size + 1}", chunk))

    # Track character examples across scenes
    char_examples = {}

    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    # Use a fresh event loop — do NOT set it as global default (avoids conflicts)
    loop = asyncio.new_event_loop()

    try:
        t0 = time.time()

        # Launch all scene rewrites in parallel
        tasks = [
            _rewrite_scene(
                scene_id, name, segs, target_lang, narrative_summary,
                mood_arc, voice_bible, lang_spec, wps, char_examples
            )
            for scene_id, name, segs in scene_groups
        ]

        results = asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

        # Update char_examples from results
        for r in results:
            if isinstance(r, dict) and r.get("char_examples"):
                for spk, exs in r["char_examples"].items():
                    if spk not in char_examples:
                        char_examples[spk] = []
                    char_examples[spk].extend(exs[-2:])

        logger.info(f"[WRITER] All scenes complete in {time.time()-t0:.1f}s")

    finally:
        loop.close()

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
