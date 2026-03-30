"""
VOICE CASTER v5 — Aggressive character differentiation using Edge TTS prosody.
Each character archetype gets a DISTINCT voice profile so father doesn't sound like son.

Strategy:
  - Edge TTS only has 2 voices per language (1 male, 1 female)
  - We differentiate using SSML prosody: rate + pitch + volume
  - Father = deep/slow, Child = high/fast, Villain = deep/menacing, etc.
  - Voice profiles are LOCKED per character for the entire video
"""
import logging
from config import VOICE_PROFILES, VOICE_MAP, ROLE_GENDER, get_voice_map
logger = logging.getLogger(__name__)


def cast(dir_result, target_lang="Hindi"):
    """
    Build cast map with LOCKED voice profiles per character.
    Returns: {voice_id: {"voice": "edge_tts_voice", "rate": "+5%", "pitch": "-10Hz", "volume": "+0%"}}
    """
    vm = get_voice_map(target_lang)
    cast_map = {}

    for v in dir_result.get("voice_plan", []):
        vid = v["voice_id"]

        # Get profile (or build one from voice_plan metadata)
        profile = VOICE_PROFILES.get(vid)

        if not profile:
            # Build from voice_plan metadata
            gender = v.get("gender", ROLE_GENDER.get(vid, "male"))
            age = v.get("age", "adult")
            speed = v.get("speaking_speed", "normal")

            rate = "+0%"
            pitch = "+0Hz"
            if age == "child":
                rate, pitch = "+15%", "+20Hz"
            elif age == "elderly":
                rate, pitch = "-15%", "-20Hz"
            elif age == "young_adult":
                rate, pitch = "+8%", "+5Hz"

            if speed == "slow":
                rate = str(int(rate.replace('%','').replace('+','')) - 8) + "%"
                if not rate.startswith("-"):
                    rate = "+" + rate
            elif speed == "fast":
                rate = str(int(rate.replace('%','').replace('+','')) + 8) + "%"
                if not rate.startswith("-"):
                    rate = "+" + rate

            profile = {"gender": gender, "rate": rate, "pitch": pitch, "volume": "+0%"}

        # Get base Edge TTS voice
        base_voice = vm.get(profile["gender"], vm.get("male"))

        cast_map[vid] = {
            "voice": base_voice,
            "rate": profile["rate"],
            "pitch": profile["pitch"],
            "volume": profile.get("volume", "+0%"),
            "gender": profile["gender"],
        }

        logger.info(
            f"[CAST] {vid} → {base_voice} | "
            f"rate={profile['rate']} pitch={profile['pitch']} vol={profile.get('volume','+0%')}"
        )

    # Ensure NARRATOR exists (neutral profile)
    if not cast_map:
        default_profile = VOICE_PROFILES["NARRATOR"]
        cast_map["NARRATOR"] = {
            "voice": vm.get("male", list(vm.values())[0]),
            "rate": default_profile["rate"],
            "pitch": default_profile["pitch"],
            "volume": default_profile.get("volume", "+0%"),
            "gender": "male",
        }

    # Log pitch consistency stats
    pitches = [v["pitch"] for v in cast_map.values()]
    logger.info(f"[CAST] ✓ Cast complete — {len(cast_map)} voices | pitches: {pitches}")

    return cast_map


def get_supported_languages():
    return list(VOICE_MAP.keys())
