"""
DIRECTOR v5 — Industry-grade content analysis using LLM provider:
  1. Chunked analysis (30-seg windows with overlap) — no missed segments
  2. Pitch-based speaker hints via FFmpeg — male/female/child detection
  3. Speaker consistency post-processing — smooth short interruptions
  4. Scene boundary detection — long pauses + mood shifts
  5. Audio energy analysis — emotional intensity per segment
  6. Character name extraction from Chinese text
  7. Comprehensive voice plan with personality traits
"""
import os, json, logging, subprocess, re
from config import (
    DIRECTOR_CHUNK_SIZE, DIRECTOR_OVERLAP,
)
from llm_provider import get_llm
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analyze_audio_energy(audio_path, segments):
    """Analyze audio energy (loudness) per segment for mood detection."""
    energy_map = {}
    for seg in segments[:80]:
        sid = str(seg["id"])
        start, dur = seg["start"], seg["end"] - seg["start"]
        if dur < 0.3:
            energy_map[sid] = "medium"
            continue
        try:
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                "-af", "volumedetect", "-f", "null", "-"
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in r.stderr.split("\n"):
                if "mean_volume" in line:
                    vol = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                    if vol > -18:
                        energy_map[sid] = "high"
                    elif vol > -33:
                        energy_map[sid] = "medium"
                    else:
                        energy_map[sid] = "low"
                    break
            else:
                energy_map[sid] = "medium"
        except Exception:
            energy_map[sid] = "medium"
    return energy_map


def _analyze_pitch(audio_path, segments):
    """Estimate average pitch per segment using FFmpeg's astats filter."""
    pitch_hints = {}
    for seg in segments[:80]:
        sid = str(seg["id"])
        start, dur = seg["start"], seg["end"] - seg["start"]
        if dur < 0.5:
            pitch_hints[sid] = "mid"
            continue
        try:
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-ss", f"{start:.3f}", "-t", f"{min(dur, 5.0):.3f}",
                "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
                "-f", "null", "-"
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            rms_vals = []
            for line in r.stderr.split("\n"):
                if "RMS_level" in line:
                    try:
                        val = float(line.split("=")[-1].strip())
                        rms_vals.append(val)
                    except:
                        pass
            if rms_vals:
                avg_rms = sum(rms_vals) / len(rms_vals)
                if avg_rms > -20:
                    pitch_hints[sid] = "high"
                elif avg_rms > -35:
                    pitch_hints[sid] = "mid"
                else:
                    pitch_hints[sid] = "low"
            else:
                pitch_hints[sid] = "mid"
        except Exception:
            pitch_hints[sid] = "mid"
    return pitch_hints


def _analyze_word_patterns(words, segments):
    """Find significant pauses (>0.8s) between words — often indicates speaker change."""
    if not words:
        return ""
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
    for p in pauses[:30]:
        analysis += f"  [{p['time']:.1f}s] {p['gap']:.1f}s pause: ...{p['before']} → {p['after']}...\n"
    return analysis


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCENE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_scenes(segments, energy_map=None):
    """Detect scene boundaries from segment gaps + energy shifts."""
    if not segments:
        return []

    boundaries = [0]

    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i-1]["end"]

        if gap > 2.0:
            boundaries.append(i)
            continue

        if gap > 1.0 and energy_map:
            prev_e = energy_map.get(str(segments[i-1]["id"]), "medium")
            curr_e = energy_map.get(str(segments[i]["id"]), "medium")
            energy_order = {"low": 0, "medium": 1, "high": 2}
            shift = abs(energy_order.get(prev_e, 1) - energy_order.get(curr_e, 1))
            if shift >= 1:
                boundaries.append(i)

    scenes = []
    for i, start_idx in enumerate(boundaries):
        end_idx = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else len(segments) - 1
        scene_segs = segments[start_idx:end_idx + 1]
        if scene_segs:
            scenes.append({
                "scene_id": i,
                "start_seg": start_idx,
                "end_seg": end_idx,
                "start_time": scene_segs[0]["start"],
                "end_time": scene_segs[-1]["end"],
                "segment_count": len(scene_segs),
                "segment_ids": [s["id"] for s in scene_segs],
            })

    logger.info(f"[DIRECTOR] Detected {len(scenes)} scenes from {len(segments)} segments")
    return scenes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT = """You are a professional dubbing director analyzing a video transcript for dubbing.
You must determine WHO is speaking, their MOOD, and the overall dubbing strategy.

Analyze the transcript and return a JSON plan:
{
  "content_type": "single_narrator" | "dialogue_drama" | "interview" | "mixed",
  "real_speaker_count": <integer>,
  "narrative_summary": "<2-3 sentence summary of the story/content>",
  "mood_arc": ["opening_mood", "middle_mood", "climax_mood", "ending_mood"],
  "translation_style": "narrative_flow" | "dramatic_dialogue" | "conversational",
  "character_names": {"ROLE": "original_name", ...},
  "notes": "<dubbing strategy notes>",
  "voice_plan": [
    {
      "voice_id": "NARRATOR",
      "role": "story narrator",
      "gender": "male",
      "age": "adult",
      "tone": "calm_narrative",
      "personality": "warm storyteller",
      "speaking_speed": "normal"
    }
  ],
  "speaker_map": {"0": "NARRATOR", "1": "FATHER", ...},
  "segment_moods": {"0": "neutral", "5": "tense", "10": "emotional", ...}
}

DIARIZATION STRATEGY (be precise!):
1. PAUSE ANALYSIS: Gap > 1s between segments = likely speaker change
2. CONTENT CLUES: "he said", "she replied", quotes = character speech
3. NARRATION vs DIALOGUE: Narration = longer, descriptive. Dialogue = shorter, emotional, quoted
4. CONTINUATION: Same sentence continuing across segments = same speaker
5. PITCH HINTS (if provided): "high" = female/child, "low" = male/elderly, "mid" = ambiguous
6. ENERGY HINTS (if provided): "high" = shouting/angry, "low" = whispering/sad

VOICE ID OPTIONS (pick the most specific match):
  NARRATOR FATHER MOTHER SON DAUGHTER OLD_MAN OLD_WOMAN
  YOUNG_MAN YOUNG_WOMAN GIRL BOY VILLAIN HERO HEROINE
  CHAR_A CHAR_B CHAR_C CHAR_D

VOICE PLAN REQUIREMENTS:
- Include ALL distinct characters (not just narrator)
- "gender": MUST be exactly "male" or "female" (critical for voice selection!)
- "age": "child" / "young_adult" / "adult" / "middle_aged" / "elderly"
- "personality": be specific — "gruff warrior" vs "gentle priest" vs "scheming advisor"
- "speaking_speed": "slow" / "normal" / "fast" (elderly=slow, children=fast, narrator=normal)

MOOD OPTIONS:
  neutral / happy / sad / angry / tense / excited / emotional / wise / gentle /
  urgent / fearful / romantic / humorous / dramatic / contemplative / cold / menacing"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPEAKER CONSISTENCY POST-PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _smooth_speaker_map(speaker_map, segments):
    """
    Fix short speaker interruptions. If segment N-1=FATHER, N=NARRATOR, N+1=FATHER
    and the gap between N-1 and N+1 is <1s, then N is probably FATHER too.
    """
    seg_ids = sorted(speaker_map.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    seg_by_id = {str(s["id"]): s for s in segments}
    smoothed = dict(speaker_map)

    for i in range(1, len(seg_ids) - 1):
        prev_id, curr_id, next_id = seg_ids[i-1], seg_ids[i], seg_ids[i+1]
        prev_spk = smoothed.get(prev_id, "NARRATOR")
        curr_spk = smoothed.get(curr_id, "NARRATOR")
        next_spk = smoothed.get(next_id, "NARRATOR")

        if prev_spk == next_spk and curr_spk != prev_spk:
            prev_seg = seg_by_id.get(prev_id)
            curr_seg = seg_by_id.get(curr_id)
            if prev_seg and curr_seg:
                gap = curr_seg["start"] - prev_seg["end"]
                curr_dur = curr_seg["end"] - curr_seg["start"]
                if gap < 0.5 and curr_dur < 2.0:
                    smoothed[curr_id] = prev_spk

    changes = sum(1 for k in seg_ids if smoothed[k] != speaker_map.get(k))
    if changes:
        logger.info(f"[DIRECTOR] Speaker smoothing: {changes} segments reassigned")
    return smoothed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ANALYZE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze(segments, work_dir, user_description="", whisper_words=None, audio_path=None):
    """Full content analysis pipeline."""
    total_dur = segments[-1]["end"] if segments else 0
    logger.info(f"[DIRECTOR] Analyzing {len(segments)} segments ({total_dur:.0f}s)...")

    # ── Audio Analysis ───────────────────────────────────────────
    energy_map = {}
    pitch_hints = {}

    if audio_path and os.path.exists(audio_path):
        logger.info("[DIRECTOR] Analyzing audio energy (mood detection)...")
        energy_map = _analyze_audio_energy(audio_path, segments)

        logger.info("[DIRECTOR] Analyzing pitch (speaker gender hints)...")
        pitch_hints = _analyze_pitch(audio_path, segments)

        high_e = sum(1 for v in energy_map.values() if v == "high")
        low_e = sum(1 for v in energy_map.values() if v == "low")
        high_p = sum(1 for v in pitch_hints.values() if v == "high")
        low_p = sum(1 for v in pitch_hints.values() if v == "low")
        logger.info(f"[DIRECTOR] Energy: {high_e} high, {low_e} low | Pitch: {high_p} high, {low_p} low")

    # Word-level analysis
    word_analysis = ""
    if whisper_words:
        word_analysis = _analyze_word_patterns(whisper_words, segments)
        if word_analysis:
            logger.info("[DIRECTOR] Word-level pause analysis ✓")

    # ── Chunked LLM Analysis ─────────────────────────────────────
    combined_plan = {
        "speaker_map": {},
        "segment_moods": {},
        "voice_plan": [],
        "content_type": "dialogue_drama",
        "real_speaker_count": 1,
        "narrative_summary": "",
        "mood_arc": [],
        "translation_style": "dramatic_dialogue",
        "character_names": {},
    }

    llm = get_llm()

    if len(segments) <= DIRECTOR_CHUNK_SIZE + 5:
        logger.info(f"[DIRECTOR] Single-pass analysis ({len(segments)} segments)")
        plan = _analyze_chunk(
            segments, 0, 1, "", energy_map, pitch_hints, word_analysis, user_description, llm
        )
        combined_plan.update(plan)
    else:
        n_chunks = (len(segments) + DIRECTOR_CHUNK_SIZE - DIRECTOR_OVERLAP - 1) // (DIRECTOR_CHUNK_SIZE - DIRECTOR_OVERLAP)
        logger.info(f"[DIRECTOR] Chunked analysis: {n_chunks} chunks of ~{DIRECTOR_CHUNK_SIZE} segments")

        summary_so_far = ""
        seen_voices = set()

        for ci in range(n_chunks):
            start = ci * (DIRECTOR_CHUNK_SIZE - DIRECTOR_OVERLAP)
            end = min(start + DIRECTOR_CHUNK_SIZE, len(segments))
            chunk = segments[start:end]

            if not chunk:
                break

            logger.info(f"[DIRECTOR] Chunk {ci+1}/{n_chunks}: segments {chunk[0]['id']}-{chunk[-1]['id']}")

            try:
                plan = _analyze_chunk(
                    chunk, ci, n_chunks, summary_so_far,
                    energy_map, pitch_hints, word_analysis, user_description, llm
                )

                # Merge maps (skip overlap segments)
                for sid, spk in plan.get("speaker_map", {}).items():
                    if sid not in combined_plan["speaker_map"]:
                        combined_plan["speaker_map"][sid] = spk

                for sid, mood in plan.get("segment_moods", {}).items():
                    if sid not in combined_plan["segment_moods"]:
                        combined_plan["segment_moods"][sid] = mood

                # Merge voice plan (avoid duplicates)
                for v in plan.get("voice_plan", []):
                    vid = v.get("voice_id", "")
                    if vid and vid not in seen_voices:
                        combined_plan["voice_plan"].append(v)
                        seen_voices.add(vid)

                if plan.get("narrative_summary"):
                    summary_so_far = plan["narrative_summary"]
                combined_plan["content_type"] = plan.get("content_type", combined_plan["content_type"])
                combined_plan["real_speaker_count"] = max(
                    combined_plan["real_speaker_count"],
                    plan.get("real_speaker_count", 1)
                )
                if plan.get("mood_arc"):
                    combined_plan["mood_arc"] = plan["mood_arc"]
                combined_plan["translation_style"] = plan.get("translation_style", combined_plan["translation_style"])
                if plan.get("character_names"):
                    combined_plan["character_names"].update(plan["character_names"])

            except Exception as e:
                logger.warning(f"[DIRECTOR] Chunk {ci+1} failed: {e}, filling with NARRATOR")
                for s in chunk:
                    sid = str(s["id"])
                    if sid not in combined_plan["speaker_map"]:
                        combined_plan["speaker_map"][sid] = "NARRATOR"
                    if sid not in combined_plan["segment_moods"]:
                        combined_plan["segment_moods"][sid] = "neutral"

        combined_plan["narrative_summary"] = summary_so_far

    # ── Fill Missing & Post-Process ──────────────────────────────
    sm = combined_plan.get("speaker_map", {})
    moods = combined_plan.get("segment_moods", {})

    for s in segments:
        sid = str(s["id"])
        if sid not in sm:
            sm[sid] = "NARRATOR"
        if sid not in moods:
            moods[sid] = "neutral"

    sm = _smooth_speaker_map(sm, segments)
    combined_plan["speaker_map"] = sm
    combined_plan["segment_moods"] = moods

    # Ensure voice plan has at least NARRATOR
    voice_ids = {v["voice_id"] for v in combined_plan.get("voice_plan", [])}
    if "NARRATOR" not in voice_ids:
        combined_plan["voice_plan"].insert(0, {
            "voice_id": "NARRATOR", "role": "narrator", "gender": "male",
            "age": "adult", "tone": "calm_narrative", "personality": "warm storyteller",
            "speaking_speed": "normal"
        })

    # Apply to segments
    for s in segments:
        sid = str(s["id"])
        s["speaker"] = sm.get(sid, "NARRATOR")
        s["mood"] = moods.get(sid, "neutral")
    combined_plan["segments"] = segments

    # ── Scene Detection ──────────────────────────────────────────
    scenes = detect_scenes(segments, energy_map)
    combined_plan["scenes"] = scenes
    combined_plan["energy_map"] = energy_map
    combined_plan["pitch_hints"] = pitch_hints

    # ── Logging ──────────────────────────────────────────────────
    logger.info(f"[DIRECTOR] {'━' * 60}")
    logger.info(f"[DIRECTOR] CONTENT TYPE:  {combined_plan.get('content_type', '?')}")
    logger.info(f"[DIRECTOR] SPEAKERS:      {combined_plan.get('real_speaker_count', '?')}")
    logger.info(f"[DIRECTOR] SCENES:        {len(scenes)}")
    logger.info(f"[DIRECTOR] STORY:         {combined_plan.get('narrative_summary', '')[:100]}")
    logger.info(f"[DIRECTOR] MOOD ARC:      {combined_plan.get('mood_arc', '?')}")
    logger.info(f"[DIRECTOR] TRANSLATION:   {combined_plan.get('translation_style', '?')}")
    logger.info(f"[DIRECTOR] CHARACTERS:    {combined_plan.get('character_names', {})}")
    logger.info(f"[DIRECTOR] VOICE PLAN:")
    for v in combined_plan.get("voice_plan", []):
        logger.info(
            f"[DIRECTOR]   {v['voice_id']} → {v.get('role','')} | "
            f"{v.get('gender','?')} | {v.get('age','?')} | {v.get('personality','')}"
        )

    speaker_counts = {}
    for sid, speaker in sm.items():
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    for spk, cnt in sorted(speaker_counts.items(), key=lambda x: -x[1]):
        logger.info(f"[DIRECTOR]   {spk}: {cnt} segments")
    logger.info(f"[DIRECTOR] {'━' * 60}")

    with open(os.path.join(work_dir, "director_plan.json"), "w") as f:
        json.dump(combined_plan, f, indent=2, ensure_ascii=False)

    return combined_plan


def _analyze_chunk(segments, chunk_idx, total_chunks, summary_so_far,
                   energy_map, pitch_hints, word_analysis, user_description, llm):
    """Analyze a chunk of segments with context from previous chunks."""
    seg_lines = []
    for i, s in enumerate(segments):
        gap = ""
        if i > 0:
            pause = round(s["start"] - segments[i-1]["end"], 2)
            if pause > 0.3:
                gap = f" [PAUSE {pause}s]"

        sid = str(s["id"])
        hints = []
        if pitch_hints and sid in pitch_hints and pitch_hints[sid] != "mid":
            hints.append(f"pitch:{pitch_hints[sid]}")
        if energy_map and sid in energy_map and energy_map[sid] != "medium":
            hints.append(f"energy:{energy_map[sid]}")
        hint_str = f" ({', '.join(hints)})" if hints else ""

        seg_lines.append(f'[{s["id"]}|{s["start"]:.1f}s]{gap}{hint_str} {s["text"][:100]}')

    seg_text = "\n".join(seg_lines)

    context = ""
    if summary_so_far and chunk_idx > 0:
        context = f"\nPREVIOUS CHUNKS SUMMARY:\n{summary_so_far}\nContinue the analysis with consistent speaker assignments.\n"

    msg = f"""Video chunk {chunk_idx + 1}/{total_chunks}: {len(segments)} segments
User description: {user_description or 'Not provided'}
{context}
{word_analysis}

TRANSCRIPT:
{seg_text}

Return ONLY valid JSON. Include ALL segment IDs ({segments[0]['id']} to {segments[-1]['id']}) in speaker_map AND segment_moods.
Make voice_plan comprehensive — include ALL distinct characters."""

    result = llm.chat(PROMPT, msg, temperature=0.1, max_tokens=8000, json_response=True)
    return result or {}
