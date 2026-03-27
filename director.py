"""
DIRECTOR v3 — Enhanced content analysis with:
- Word-level timestamp analysis for precise speaker diarization
- Audio energy analysis for emotional shifts
- Better mood detection per segment
- Enhanced prompting for accurate character identification
"""
import os, json, logging, subprocess
from groq import Groq
logger = logging.getLogger(__name__)
_client = None

def _gc():
    global _client
    if not _client:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _analyze_audio_energy(audio_path, segments):
    """
    Analyze audio energy per segment using ffmpeg.
    Loud segments = emotional intensity. Quiet segments = intimate/whispered.
    Returns dict: segment_id → energy_level (low/medium/high)
    """
    energy_map = {}
    
    for seg in segments[:50]:  # Cap at 50 to avoid timeout
        sid = seg["id"]
        start = seg["start"]
        dur = seg["end"] - seg["start"]
        
        if dur < 0.3:
            energy_map[str(sid)] = "medium"
            continue
        
        try:
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                "-af", "volumedetect",
                "-f", "null", "-"
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # Parse mean_volume from stderr
            for line in r.stderr.split("\n"):
                if "mean_volume" in line:
                    vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                    if vol > -20:
                        energy_map[str(sid)] = "high"     # Shouting / loud
                    elif vol > -35:
                        energy_map[str(sid)] = "medium"   # Normal
                    else:
                        energy_map[str(sid)] = "low"      # Whisper / quiet
                    break
            else:
                energy_map[str(sid)] = "medium"
        except Exception:
            energy_map[str(sid)] = "medium"
    
    return energy_map


def _analyze_word_patterns(words, segments):
    """
    Analyze word-level timestamps for speaker change patterns.
    Long pauses between words often indicate speaker changes.
    Returns analysis string for the LLM prompt.
    """
    if not words:
        return ""
    
    # Find significant pauses (> 0.8s between words)
    pauses = []
    for i in range(1, len(words)):
        gap = words[i].get("start", 0) - words[i-1].get("end", 0)
        if gap > 0.8:
            pauses.append({
                "time": round(words[i-1].get("end", 0), 2),
                "gap": round(gap, 2),
                "before": words[i-1].get("word", ""),
                "after": words[i].get("word", ""),
            })
    
    if not pauses:
        return ""
    
    analysis = "WORD-LEVEL PAUSE ANALYSIS (≥0.8s pauses often = speaker change):\n"
    for p in pauses[:30]:  # Cap at 30
        analysis += f"  [{p['time']:.1f}s] {p['gap']:.1f}s pause: ...{p['before']} → {p['after']}...\n"
    
    return analysis


PROMPT = """You are a professional dubbing director analyzing a video transcript for dubbing.
You must determine speakers, moods, and the overall dubbing strategy.

Analyze the transcript and return a JSON plan:
{
  "content_type": "single_narrator" | "dialogue_drama" | "interview" | "mixed",
  "real_speaker_count": <integer>,
  "narrative_summary": "<2-3 sentence summary of the story/content>",
  "mood_arc": ["opening_mood", "middle_mood", "climax_mood", "ending_mood"],
  "translation_style": "narrative_flow" | "dramatic_dialogue" | "conversational",
  "notes": "<dubbing notes>",
  "voice_plan": [
    {"voice_id": "NARRATOR", "role": "story narrator", "gender": "male", "tone": "calm_narrative", "personality": "warm storyteller"}
  ],
  "speaker_map": {"0": "NARRATOR", ...},
  "segment_moods": {"0": "neutral", "5": "tense", "10": "emotional", ...}
}

DIARIZATION RULES:
- Analyze pauses between segments. Gap > 1s between segments often = speaker change
- Look at dialogue markers: quotes, "he said", "she replied" = character speech
- Narration segments are typically longer, descriptive
- Character dialogue is shorter, more emotional, often has quotes
- Use content clues: "father said" → next segment = FATHER speaking
- For single narrator videos: ALL segments → NARRATOR
- If provided, use WORD-LEVEL PAUSE ANALYSIS for additional speaker change cues
- If provided, use AUDIO ENERGY data to inform mood assignments:
  - high energy + sharp words = angry/tense
  - low energy + soft words = sad/gentle/whispered
  - high energy + positive words = happy/excited

MOOD OPTIONS: neutral / happy / sad / angry / tense / excited / emotional / wise / gentle / urgent / fearful / romantic / humorous

voice_id options: NARRATOR FATHER MOTHER SON DAUGHTER OLD_MAN OLD_WOMAN
  YOUNG_MAN YOUNG_WOMAN GIRL BOY VILLAIN HERO HEROINE CHAR_A CHAR_B CHAR_C CHAR_D

IMPORTANT FOR VOICE PLAN:
- Include ALL distinct characters, not just narrator
- Specify accurate gender for EVERY voice (critical for voice matching)
- Include personality traits that help differentiate voices:
  "personality": "gruff warrior" vs "gentle priest" vs "sarcastic merchant"
- Include age_hint for better voice matching:
  "tone": "deep_authoritative_middle_aged" vs "bright_youthful_energetic"
"""


def analyze(segments, work_dir, user_description="", whisper_words=None, audio_path=None):
    """
    Analyze video content for dubbing direction.
    
    Enhanced with:
    - Word-level pause analysis for speaker diarization
    - Audio energy analysis for mood detection
    """
    c = _gc()
    
    # Build context with pause analysis
    seg_lines = []
    for i, s in enumerate(segments):
        gap = ""
        if i > 0:
            pause = round(s["start"] - segments[i-1]["end"], 2)
            if pause > 0.5:
                gap = f" [PAUSE {pause}s]"
        seg_lines.append(f'[{s["id"]}|{s["start"]:.1f}s]{gap} {s["text"][:80]}')
    
    seg_text = "\n".join(seg_lines)
    total_dur = segments[-1]["end"] if segments else 0
    
    # Word-level analysis
    word_analysis = ""
    if whisper_words:
        word_analysis = _analyze_word_patterns(whisper_words, segments)
    
    # Audio energy analysis
    energy_info = ""
    if audio_path and os.path.exists(audio_path):
        try:
            energy_map = _analyze_audio_energy(audio_path, segments)
            if energy_map:
                high_energy = [sid for sid, e in energy_map.items() if e == "high"]
                low_energy = [sid for sid, e in energy_map.items() if e == "low"]
                if high_energy or low_energy:
                    energy_info = "\nAUDIO ENERGY ANALYSIS:\n"
                    if high_energy:
                        energy_info += f"  HIGH energy segments (loud/shouting): {', '.join(high_energy[:15])}\n"
                    if low_energy:
                        energy_info += f"  LOW energy segments (quiet/whisper): {', '.join(low_energy[:15])}\n"
        except Exception as e:
            logger.debug(f"[DIRECTOR] Energy analysis failed: {e}")
    
    msg = f"""Video: {len(segments)} segments, {total_dur:.0f}s duration
User description: {user_description or 'Not provided'}
Note: [PAUSE Xs] indicates silence gaps between segments — these often indicate speaker changes.

{word_analysis}
{energy_info}

TRANSCRIPT:
{seg_text}

Return ONLY valid JSON. Include ALL {len(segments)} segment IDs in speaker_map AND segment_moods.
Make voice_plan comprehensive — include ALL distinct characters with detailed personality traits."""

    logger.info(f"[DIRECTOR] Analyzing {len(segments)} segments ({total_dur:.0f}s)...")
    if word_analysis:
        logger.info(f"[DIRECTOR] Using word-level pause analysis ✓")
    if energy_info:
        logger.info(f"[DIRECTOR] Using audio energy analysis ✓")
    
    resp = c.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": msg}],
        temperature=0.1, max_tokens=8000,
        response_format={"type": "json_object"}
    )
    plan = json.loads(resp.choices[0].message.content)
    
    # Fill missing segment IDs
    sm = plan.get("speaker_map", {})
    moods = plan.get("segment_moods", {})
    for s in segments:
        sid = str(s["id"])
        if sid not in sm:
            sm[sid] = "NARRATOR"
        if sid not in moods:
            moods[sid] = "neutral"
    plan["speaker_map"] = sm
    plan["segment_moods"] = moods
    
    # Apply to segments
    for s in segments:
        sid = str(s["id"])
        s["speaker"] = sm.get(sid, "NARRATOR")
        s["mood"] = moods.get(sid, "neutral")
    plan["segments"] = segments
    
    # Log
    logger.info(f"[DIRECTOR] {'━' * 55}")
    logger.info(f"[DIRECTOR] CONTENT TYPE:  {plan.get('content_type', '?')}")
    logger.info(f"[DIRECTOR] SPEAKERS:      {plan.get('real_speaker_count', '?')}")
    logger.info(f"[DIRECTOR] STORY:         {plan.get('narrative_summary', '')[:80]}")
    logger.info(f"[DIRECTOR] MOOD ARC:      {plan.get('mood_arc', '?')}")
    logger.info(f"[DIRECTOR] TRANSLATION:   {plan.get('translation_style', '?')}")
    logger.info(f"[DIRECTOR] VOICE PLAN:")
    for v in plan.get("voice_plan", []):
        logger.info(
            f"[DIRECTOR]   {v['voice_id']} → {v['role']} | "
            f"{v['gender']} | {v['tone']} | {v.get('personality', '')}"
        )
    
    # Count speakers per segment
    speaker_counts = {}
    for sid, speaker in sm.items():
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    for spk, cnt in sorted(speaker_counts.items(), key=lambda x: -x[1]):
        logger.info(f"[DIRECTOR]   {spk}: {cnt} segments")
    logger.info(f"[DIRECTOR] {'━' * 55}")
    
    with open(os.path.join(work_dir, "director_plan.json"), "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    return plan
