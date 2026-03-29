"""Tests for voice_caster module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_caster import cast, get_supported_languages
from config import VOICE_PROFILES, VOICE_MAP


def test_cast_with_empty_voice_plan():
    """Empty voice plan should return default NARRATOR."""
    dir_result = {"voice_plan": []}
    result = cast(dir_result, "Hindi")
    assert "NARRATOR" in result
    assert result["NARRATOR"]["voice"] == "hi-IN-MadhurNeural"


def test_cast_with_known_voices():
    """Known voice IDs should get correct profiles."""
    dir_result = {
        "voice_plan": [
            {"voice_id": "FATHER", "gender": "male", "age": "adult"},
            {"voice_id": "MOTHER", "gender": "female", "age": "adult"},
        ]
    }
    result = cast(dir_result, "Hindi")
    assert "FATHER" in result
    assert "MOTHER" in result
    # Should use correct gender voices
    assert result["FATHER"]["gender"] == "male"
    assert result["MOTHER"]["gender"] == "female"


def test_cast_includes_prosody_params():
    """Cast should include rate, pitch, volume."""
    dir_result = {
        "voice_plan": [
            {"voice_id": "HERO", "gender": "male", "age": "adult"},
        ]
    }
    result = cast(dir_result, "Hindi")
    hero = result["HERO"]
    assert "rate" in hero
    assert "pitch" in hero
    assert "volume" in hero
    assert "voice" in hero


def test_get_supported_languages():
    """Test that supported languages list is not empty."""
    langs = get_supported_languages()
    assert len(langs) > 0
    assert "Hindi" in langs
    assert "English" in langs


def test_cast_different_languages():
    """Test casting to different languages."""
    dir_result = {
        "voice_plan": [
            {"voice_id": "NARRATOR", "gender": "male", "age": "adult"},
        ]
    }
    # Hindi
    result_hi = cast(dir_result, "Hindi")
    assert result_hi["NARRATOR"]["voice"] == "hi-IN-MadhurNeural"

    # English
    result_en = cast(dir_result, "English")
    assert result_en["NARRATOR"]["voice"] == "en-US-AndrewNeural"


def test_unknown_voice_id_uses_fallback():
    """Unknown voice ID should use NARRATOR profile."""
    dir_result = {
        "voice_plan": [
            {"voice_id": "CHAR_X", "gender": "male", "age": "adult"},
        ]
    }
    result = cast(dir_result, "Hindi")
    # Should have CHAR_X entry
    assert "CHAR_X" in result
    # NARRATOR is not auto-added when other voices exist
    # (NARRATOR only auto-added for empty voice_plan)


if __name__ == "__main__":
    test_cast_with_empty_voice_plan()
    test_cast_with_known_voices()
    test_cast_includes_prosody_params()
    test_get_supported_languages()
    test_cast_different_languages()
    test_unknown_voice_id_uses_fallback()
    print("All voice_caster tests passed!")
