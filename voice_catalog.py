"""
VOICE CATALOG v2 — Smart voice matching for LOCAL Fish Speech inference.
Matches character profiles to native-sounding regional voices. No Chinese
voice cloning — picks voices that sound authentically Indian/regional.

Architecture:
1. Curated regional voice catalog with metadata (age, gender, tone, personality)
2. Score-based matching: Director's character profile → best voice from catalog
3. Fish Speech LOCAL on Colab GPU (no API key, no cloud)
4. Edge TTS fallback if Fish Speech not available
"""
import os, json, logging
from config import ROLE_GENDER
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CURATED VOICE CATALOG
# Each voice has metadata for intelligent matching.
# reference_id = Fish Audio model ID (from fish.audio/models/<id>)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VOICE_CATALOG = {
    "Hindi": {
        "voices": [
            # ── Male Voices ──────────────────────────────────────
            {
                "id": "hindi_male_narrator",
                "name": "Anurag",
                "gender": "male",
                "age": "adult",
                "tone": "calm_measured",
                "personality": "narrator",
                "best_for": ["NARRATOR", "CHAR_A"],
                "description": "Deep, calm male voice. Perfect for narration and storytelling.",
                "energy": "medium",
            },
            {
                "id": "hindi_male_authoritative",
                "name": "Rajesh",
                "gender": "male",
                "age": "middle_aged",
                "tone": "deep_authoritative",
                "personality": "father_figure",
                "best_for": ["FATHER", "OLD_MAN", "VILLAIN"],
                "description": "Authoritative, commanding male voice with gravitas.",
                "energy": "medium_low",
            },
            {
                "id": "hindi_male_young",
                "name": "Arjun",
                "gender": "male",
                "age": "young_adult",
                "tone": "energetic_bright",
                "personality": "hero",
                "best_for": ["HERO", "YOUNG_MAN", "SON"],
                "description": "Youthful, energetic male voice with enthusiasm.",
                "energy": "high",
            },
            {
                "id": "hindi_male_elderly",
                "name": "Pandit Ji",
                "gender": "male",
                "age": "elderly",
                "tone": "wise_gentle",
                "personality": "elder",
                "best_for": ["OLD_MAN", "CHAR_C"],
                "description": "Warm, wise elderly male voice with measured pace.",
                "energy": "low",
            },
            # ── Female Voices ────────────────────────────────────
            {
                "id": "hindi_female_narrator",
                "name": "Priya",
                "gender": "female",
                "age": "adult",
                "tone": "warm_pleasant",
                "personality": "narrator",
                "best_for": ["NARRATOR", "MOTHER", "CHAR_B"],
                "description": "Warm, pleasant female voice. Great for narration and motherly roles.",
                "energy": "medium",
            },
            {
                "id": "hindi_female_young",
                "name": "Ananya",
                "gender": "female",
                "age": "young_adult",
                "tone": "bright_expressive",
                "personality": "heroine",
                "best_for": ["HEROINE", "YOUNG_WOMAN", "DAUGHTER", "GIRL"],
                "description": "Bright, expressive young female voice with range.",
                "energy": "high",
            },
            {
                "id": "hindi_female_mature",
                "name": "Meera Ji",
                "gender": "female",
                "age": "middle_aged",
                "tone": "authoritative_warm",
                "personality": "mother_figure",
                "best_for": ["MOTHER", "OLD_WOMAN"],
                "description": "Mature, warm female voice with authority.",
                "energy": "medium",
            },
            {
                "id": "hindi_male_child",
                "name": "Chintu",
                "gender": "male",
                "age": "child",
                "tone": "innocent_playful",
                "personality": "child",
                "best_for": ["BOY", "SON"],
                "description": "Childlike, innocent voice for young boy characters.",
                "energy": "high",
            },
        ],
        # Fish Audio voice reference IDs for Hindi (from fish.audio)
        # These are real voice model IDs on the Fish Audio platform
        "fish_audio_refs": {
            "hindi_male_narrator": None,       # Will be populated during setup
            "hindi_male_authoritative": None,
            "hindi_male_young": None,
            "hindi_male_elderly": None,
            "hindi_female_narrator": None,
            "hindi_female_young": None,
            "hindi_female_mature": None,
            "hindi_male_child": None,
        },
        # Edge TTS fallback voices (used when Fish Audio unavailable)
        "edge_fallback": {
            "male": "hi-IN-MadhurNeural",
            "female": "hi-IN-SwaraNeural",
        }
    },
    "English": {
        "voices": [
            {"id": "en_male_narrator", "name": "James", "gender": "male", "age": "adult",
             "tone": "professional", "personality": "narrator", "best_for": ["NARRATOR", "HERO"],
             "description": "Clear, professional male narrator.", "energy": "medium"},
            {"id": "en_male_deep", "name": "Morgan", "gender": "male", "age": "middle_aged",
             "tone": "deep_gravitas", "personality": "father_figure", "best_for": ["FATHER", "OLD_MAN", "VILLAIN"],
             "description": "Deep, resonant male voice with gravitas.", "energy": "medium_low"},
            {"id": "en_male_young", "name": "Alex", "gender": "male", "age": "young_adult",
             "tone": "casual_energetic", "personality": "hero", "best_for": ["YOUNG_MAN", "SON", "HERO"],
             "description": "Casual, energetic young male.", "energy": "high"},
            {"id": "en_female_narrator", "name": "Sarah", "gender": "female", "age": "adult",
             "tone": "warm_clear", "personality": "narrator", "best_for": ["NARRATOR", "MOTHER", "HEROINE"],
             "description": "Warm, clear female narrator.", "energy": "medium"},
            {"id": "en_female_young", "name": "Emma", "gender": "female", "age": "young_adult",
             "tone": "bright_expressive", "personality": "heroine", "best_for": ["YOUNG_WOMAN", "DAUGHTER", "GIRL"],
             "description": "Bright, expressive young female.", "energy": "high"},
        ],
        "fish_audio_refs": {},
        "edge_fallback": {"male": "en-US-AndrewNeural", "female": "en-US-AvaNeural"}
    },
    "Tamil": {
        "voices": [
            {"id": "ta_male_narrator", "name": "Karthik", "gender": "male", "age": "adult",
             "tone": "theatrical_deep", "personality": "narrator", "best_for": ["NARRATOR", "HERO", "FATHER"],
             "description": "Deep, theatrical Tamil male voice.", "energy": "medium"},
            {"id": "ta_female_narrator", "name": "Divya", "gender": "female", "age": "adult",
             "tone": "melodic_warm", "personality": "narrator", "best_for": ["NARRATOR", "MOTHER", "HEROINE"],
             "description": "Melodic, warm Tamil female voice.", "energy": "medium"},
        ],
        "fish_audio_refs": {},
        "edge_fallback": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"}
    },
    "Telugu": {
        "voices": [
            {"id": "te_male_narrator", "name": "Ravi", "gender": "male", "age": "adult",
             "tone": "energetic_bold", "personality": "narrator", "best_for": ["NARRATOR", "HERO", "FATHER"],
             "description": "Energetic, bold Telugu male voice.", "energy": "high"},
            {"id": "te_female_narrator", "name": "Lakshmi", "gender": "female", "age": "adult",
             "tone": "graceful_warm", "personality": "narrator", "best_for": ["NARRATOR", "MOTHER", "HEROINE"],
             "description": "Graceful, warm Telugu female voice.", "energy": "medium"},
        ],
        "fish_audio_refs": {},
        "edge_fallback": {"male": "te-IN-MohanNeural", "female": "te-IN-ShrutiNeural"}
    },
    "Bengali": {
        "voices": [
            {"id": "bn_male_narrator", "name": "Subho", "gender": "male", "age": "adult",
             "tone": "intellectual_measured", "personality": "narrator", "best_for": ["NARRATOR", "FATHER"],
             "description": "Intellectual, measured Bengali male voice.", "energy": "medium"},
            {"id": "bn_female_narrator", "name": "Rani", "gender": "female", "age": "adult",
             "tone": "expressive_warm", "personality": "narrator", "best_for": ["NARRATOR", "MOTHER", "HEROINE"],
             "description": "Expressive, warm Bengali female voice.", "energy": "medium"},
        ],
        "fish_audio_refs": {},
        "edge_fallback": {"male": "bn-IN-BashkarNeural", "female": "bn-IN-TanishaaNeural"}
    },
    "Nepali": {
        "voices": [
            {"id": "ne_male_narrator", "name": "Bikash", "gender": "male", "age": "adult",
             "tone": "warm_storyteller", "personality": "narrator", "best_for": ["NARRATOR", "FATHER", "HERO"],
             "description": "Warm storyteller Nepali male voice.", "energy": "medium"},
            {"id": "ne_female_narrator", "name": "Sita", "gender": "female", "age": "adult",
             "tone": "gentle_melodic", "personality": "narrator", "best_for": ["NARRATOR", "MOTHER", "HEROINE"],
             "description": "Gentle, melodic Nepali female voice.", "energy": "medium"},
        ],
        "fish_audio_refs": {},
        "edge_fallback": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"}  # Hindi fallback
    },
}


def match_voices(voice_plan, target_lang="Hindi"):
    """
    Intelligently match Director's voice_plan to the best catalog voices.
    
    Args:
        voice_plan: From Director — list of dicts with voice_id, role, gender, tone, personality
        target_lang: Target language for voice selection
    
    Returns:
        cast_map: {voice_id: voice_catalog_entry} for each character
    """
    lang_data = VOICE_CATALOG.get(target_lang, VOICE_CATALOG.get("Hindi"))
    voices = lang_data["voices"]
    
    cast_map = {}
    used_voices = set()  # Track used voices to maximize differentiation
    
    for character in voice_plan:
        vid = character.get("voice_id", "NARRATOR")
        char_gender = character.get("gender", ROLE_GENDER.get(vid, "male"))
        char_tone = character.get("tone", "")
        char_personality = character.get("personality", "")
        
        best_voice = None
        best_score = -1
        
        for voice in voices:
            # Skip if gender doesn't match
            if voice["gender"] != char_gender:
                continue
            
            score = 0
            
            # Direct role match (highest priority)
            if vid in voice["best_for"]:
                score += 50
            
            # Tone similarity
            if char_tone and char_tone.lower() in voice["tone"].lower():
                score += 20
            
            # Personality match
            if char_personality and char_personality.lower() in voice["personality"].lower():
                score += 15
            
            # Age-appropriate matching
            age_map = {
                "FATHER": ["middle_aged", "adult"], "MOTHER": ["middle_aged", "adult"],
                "OLD_MAN": ["elderly"], "OLD_WOMAN": ["elderly"],
                "SON": ["young_adult", "child"], "DAUGHTER": ["young_adult", "child"],
                "BOY": ["child", "young_adult"], "GIRL": ["child", "young_adult"],
                "YOUNG_MAN": ["young_adult"], "YOUNG_WOMAN": ["young_adult"],
                "HERO": ["young_adult", "adult"], "HEROINE": ["young_adult", "adult"],
                "VILLAIN": ["adult", "middle_aged"],
            }
            if vid in age_map and voice["age"] in age_map[vid]:
                score += 10
            
            # Diversity bonus — prefer unused voices
            if voice["id"] not in used_voices:
                score += 8
            
            if score > best_score:
                best_score = score
                best_voice = voice
        
        if best_voice is None:
            # Fallback: just use first voice of matching gender
            gender_voices = [v for v in voices if v["gender"] == char_gender]
            best_voice = gender_voices[0] if gender_voices else voices[0]
        
        used_voices.add(best_voice["id"])
        cast_map[vid] = best_voice
        
        logger.info(
            f"[VOICE CATALOG] {vid} ({char_gender}/{char_tone}) "
            f"→ {best_voice['name']} ({best_voice['tone']}) "
            f"[score: {best_score}]"
        )
    
    # Ensure we have at least a NARRATOR
    if "NARRATOR" not in cast_map:
        narrator_voices = [v for v in voices if "NARRATOR" in v["best_for"]]
        cast_map["NARRATOR"] = narrator_voices[0] if narrator_voices else voices[0]
    
    return cast_map


def get_edge_fallback(voice_id, target_lang="Hindi"):
    """Get Edge TTS fallback voice for a character role."""
    lang_data = VOICE_CATALOG.get(target_lang, VOICE_CATALOG.get("Hindi"))
    gender = ROLE_GENDER.get(voice_id, "male")
    return lang_data["edge_fallback"].get(gender, "hi-IN-MadhurNeural")


def get_fish_ref(voice_entry, target_lang="Hindi"):
    """Get Fish Audio reference ID for a voice catalog entry."""
    lang_data = VOICE_CATALOG.get(target_lang, VOICE_CATALOG.get("Hindi"))
    refs = lang_data.get("fish_audio_refs", {})
    return refs.get(voice_entry["id"])


def list_available_languages():
    """List all languages with voice catalogs."""
    return list(VOICE_CATALOG.keys())


def get_voice_count(target_lang="Hindi"):
    """Get number of available voices for a language."""
    lang_data = VOICE_CATALOG.get(target_lang, {})
    return len(lang_data.get("voices", []))
