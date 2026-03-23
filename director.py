"""
DIRECTOR v2 — LLM content analysis + enhanced speaker diarization.
Uses Whisper word-level timestamps + pause patterns for better speaker separation.
"""
import os, json, logging
from groq import Groq
logger = logging.getLogger(__name__)
_client = None

def _gc():
    global _client
    if not _client: _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

PROMPT = """You are a professional dubbing director analyzing a video transcript for dubbing.

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
- segment_moods: assign emotion per segment (neutral/happy/sad/angry/tense/excited/emotional/wise/gentle/urgent)

voice_id options: NARRATOR FATHER MOTHER SON DAUGHTER OLD_MAN OLD_WOMAN
  YOUNG_MAN YOUNG_WOMAN GIRL BOY VILLAIN HERO HEROINE CHAR_A CHAR_B CHAR_C CHAR_D"""

def analyze(segments, work_dir, user_description="", whisper_words=None):
    c = _gc()
    
    # Build context with pause analysis for diarization
    seg_lines = []
    for i, s in enumerate(segments):
        gap = ""
        if i > 0:
            pause = round(s["start"] - segments[i-1]["end"], 2)
            if pause > 0.5: gap = f" [PAUSE {pause}s]"
        seg_lines.append(f'[{s["id"]}|{s["start"]:.1f}s]{gap} {s["text"][:80]}')
    
    seg_text = "\n".join(seg_lines)
    total_dur = segments[-1]["end"] if segments else 0

    msg = f"""Video: {len(segments)} segments, {total_dur:.0f}s duration
User description: {user_description or 'Not provided'}
Note: [PAUSE Xs] indicates silence gaps between segments — these often indicate speaker changes.

TRANSCRIPT:
{seg_text}

Return ONLY valid JSON. Include ALL {len(segments)} segment IDs in speaker_map AND segment_moods."""

    logger.info(f"[DIRECTOR] Analyzing {len(segments)} segments ({total_dur:.0f}s)...")
    
    resp = c.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":PROMPT}, {"role":"user","content":msg}],
        temperature=0.1, max_tokens=8000,
        response_format={"type":"json_object"}
    )
    plan = json.loads(resp.choices[0].message.content)

    # Fill missing segment IDs
    sm = plan.get("speaker_map", {})
    moods = plan.get("segment_moods", {})
    for s in segments:
        sid = str(s["id"])
        if sid not in sm: sm[sid] = "NARRATOR"
        if sid not in moods: moods[sid] = "neutral"
    plan["speaker_map"] = sm
    plan["segment_moods"] = moods

    # Apply to segments
    for s in segments:
        sid = str(s["id"])
        s["speaker"] = sm.get(sid, "NARRATOR")
        s["mood"] = moods.get(sid, "neutral")
    plan["segments"] = segments

    # Log
    logger.info(f"[DIRECTOR] {'━'*45}")
    logger.info(f"[DIRECTOR] CONTENT TYPE:  {plan.get('content_type','?')}")
    logger.info(f"[DIRECTOR] SPEAKERS:      {plan.get('real_speaker_count','?')}")
    logger.info(f"[DIRECTOR] STORY:         {plan.get('narrative_summary','')[:80]}")
    logger.info(f"[DIRECTOR] MOOD ARC:      {plan.get('mood_arc','?')}")
    logger.info(f"[DIRECTOR] TRANSLATION:   {plan.get('translation_style','?')}")
    logger.info(f"[DIRECTOR] VOICE PLAN:")
    for v in plan.get("voice_plan", []):
        logger.info(f"[DIRECTOR]   {v['voice_id']} → {v['role']} | {v['gender']} | {v['tone']} | {v.get('personality','')}")
    logger.info(f"[DIRECTOR] {'━'*45}")

    with open(os.path.join(work_dir, "director_plan.json"), "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return plan
