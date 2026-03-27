"""
VOICE CASTER v4 — Aggressive character differentiation using Edge TTS prosody.
Each character archetype gets a DISTINCT voice profile so father doesn't sound like son.

Strategy:
  - Edge TTS only has 2 voices per language (1 male, 1 female)
  - We differentiate using SSML prosody: rate + pitch + volume
  - Father = deep/slow, Child = high/fast, Villain = deep/menacing, etc.
  - Voice profiles are LOCKED per character for the entire video
"""
import logging
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VOICE PROFILES — rate, pitch, volume per character archetype
# These create DISTINCT voices from just 2 base Edge TTS voices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VOICE_PROFILES = {
    # Male characters (base: MadhurNeural / male voice)
    "NARRATOR":    {"gender": "male",   "rate": "+3%",   "pitch": "+0Hz",   "volume": "+0%"},
    "FATHER":      {"gender": "male",   "rate": "-10%",  "pitch": "-15Hz",  "volume": "+5%"},
    "OLD_MAN":     {"gender": "male",   "rate": "-18%",  "pitch": "-25Hz",  "volume": "-5%"},
    "HERO":        {"gender": "male",   "rate": "+5%",   "pitch": "+5Hz",   "volume": "+5%"},
    "VILLAIN":     {"gender": "male",   "rate": "-8%",   "pitch": "-20Hz",  "volume": "+10%"},
    "YOUNG_MAN":   {"gender": "male",   "rate": "+12%",  "pitch": "+8Hz",   "volume": "+0%"},
    "CHAR_A":      {"gender": "male",   "rate": "+0%",   "pitch": "-5Hz",   "volume": "+0%"},
    "CHAR_C":      {"gender": "male",   "rate": "-5%",   "pitch": "-10Hz",  "volume": "+0%"},
    
    # Female characters (base: SwaraNeural / female voice)
    "MOTHER":      {"gender": "female", "rate": "-3%",   "pitch": "+0Hz",   "volume": "+0%"},
    "OLD_WOMAN":   {"gender": "female", "rate": "-12%",  "pitch": "-10Hz",  "volume": "-5%"},
    "HEROINE":     {"gender": "female", "rate": "+5%",   "pitch": "+8Hz",   "volume": "+5%"},
    "YOUNG_WOMAN": {"gender": "female", "rate": "+8%",   "pitch": "+10Hz",  "volume": "+0%"},
    "DAUGHTER":    {"gender": "female", "rate": "+5%",   "pitch": "+5Hz",   "volume": "+0%"},
    "GIRL":        {"gender": "female", "rate": "+15%",  "pitch": "+18Hz",  "volume": "+5%"},
    "CHAR_B":      {"gender": "female", "rate": "+0%",   "pitch": "+3Hz",   "volume": "+0%"},
    "CHAR_D":      {"gender": "female", "rate": "+8%",   "pitch": "+12Hz",  "volume": "+0%"},
    
    # Child characters (use female voice + high pitch)
    "BOY":         {"gender": "female", "rate": "+15%",  "pitch": "+25Hz",  "volume": "+5%"},
    "SON":         {"gender": "female", "rate": "+10%",  "pitch": "+18Hz",  "volume": "+0%"},
}

# Edge TTS base voices per language
VOICE_MAP = {
    "Hindi":      {"male": "hi-IN-MadhurNeural",    "female": "hi-IN-SwaraNeural"},
    "Tamil":      {"male": "ta-IN-ValluvarNeural",   "female": "ta-IN-PallaviNeural"},
    "Telugu":     {"male": "te-IN-MohanNeural",      "female": "te-IN-ShrutiNeural"},
    "Bengali":    {"male": "bn-IN-BashkarNeural",    "female": "bn-IN-TanishaaNeural"},
    "Marathi":    {"male": "mr-IN-ManoharNeural",    "female": "mr-IN-AarohiNeural"},
    "Gujarati":   {"male": "gu-IN-NiranjanNeural",   "female": "gu-IN-DhwaniNeural"},
    "Kannada":    {"male": "kn-IN-GaganNeural",      "female": "kn-IN-SapnaNeural"},
    "Malayalam":  {"male": "ml-IN-MidhunNeural",     "female": "ml-IN-SobhanaNeural"},
    "Urdu":       {"male": "ur-IN-SalmanNeural",     "female": "ur-IN-GulNeural"},
    "Nepali":     {"male": "hi-IN-MadhurNeural",     "female": "hi-IN-SwaraNeural"},
    "English":    {"male": "en-US-AndrewNeural",     "female": "en-US-AvaNeural"},
    "Spanish":    {"male": "es-MX-JorgeNeural",      "female": "es-MX-DaliaNeural"},
    "French":     {"male": "fr-FR-RemyMultilingualNeural", "female": "fr-FR-VivienneMultilingualNeural"},
    "Portuguese": {"male": "pt-BR-AntonioNeural",    "female": "pt-BR-FranciscaNeural"},
    "German":     {"male": "de-DE-FlorianMultilingualNeural", "female": "de-DE-SeraphinaMultilingualNeural"},
    "Japanese":   {"male": "ja-JP-KeitaNeural",      "female": "ja-JP-NanamiNeural"},
    "Korean":     {"male": "ko-KR-InJoonNeural",     "female": "ko-KR-SunHiNeural"},
    "Arabic":     {"male": "ar-SA-HamedNeural",      "female": "ar-SA-ZariyahNeural"},
    "Turkish":    {"male": "tr-TR-AhmetNeural",       "female": "tr-TR-EmelNeural"},
}

ROLE_GENDER = {
    "NARRATOR": "male", "FATHER": "male", "MOTHER": "female", "SON": "female",
    "DAUGHTER": "female", "OLD_MAN": "male", "OLD_WOMAN": "female",
    "YOUNG_MAN": "male", "YOUNG_WOMAN": "female", "GIRL": "female", "BOY": "female",
    "VILLAIN": "male", "HERO": "male", "HEROINE": "female",
    "CHAR_A": "male", "CHAR_B": "female", "CHAR_C": "male", "CHAR_D": "female",
}


def cast(dir_result, target_lang="Hindi"):
    """
    Build cast map with LOCKED voice profiles per character.
    Returns: {voice_id: {"voice": "edge_tts_voice", "rate": "+5%", "pitch": "-10Hz", "volume": "+0%"}}
    """
    vm = VOICE_MAP.get(target_lang, VOICE_MAP.get("English"))
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
    
    # Ensure NARRATOR exists
    if not cast_map:
        default_profile = VOICE_PROFILES["NARRATOR"]
        cast_map["NARRATOR"] = {
            "voice": vm.get("male", list(vm.values())[0]),
            "rate": default_profile["rate"],
            "pitch": default_profile["pitch"],
            "volume": default_profile.get("volume", "+0%"),
            "gender": "male",
        }
    
    return cast_map


def get_supported_languages():
    return list(VOICE_MAP.keys())
