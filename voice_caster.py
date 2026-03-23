"""VOICE CASTER v3 — Better differentiation: use SwaraNeural for young characters."""
import logging
logger = logging.getLogger(__name__)

VOICE_MAP = {
    "Hindi": {
        "male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural",
        "NARRATOR":"hi-IN-MadhurNeural",
        "FATHER":"hi-IN-MadhurNeural", "HERO":"hi-IN-MadhurNeural",
        "VILLAIN":"hi-IN-MadhurNeural", "OLD_MAN":"hi-IN-MadhurNeural",
        "YOUNG_MAN":"hi-IN-MadhurNeural", "CHAR_A":"hi-IN-MadhurNeural",
        "CHAR_C":"hi-IN-MadhurNeural",
        # Female + young characters → SwaraNeural for differentiation
        "MOTHER":"hi-IN-SwaraNeural", "DAUGHTER":"hi-IN-SwaraNeural",
        "HEROINE":"hi-IN-SwaraNeural", "OLD_WOMAN":"hi-IN-SwaraNeural",
        "GIRL":"hi-IN-SwaraNeural", "YOUNG_WOMAN":"hi-IN-SwaraNeural",
        "CHAR_B":"hi-IN-SwaraNeural", "CHAR_D":"hi-IN-SwaraNeural",
        # SON/BOY use SwaraNeural — young boy voice in Hindi dubbing
        # sounds better with female neural voice at higher pitch
        "SON":"hi-IN-SwaraNeural", "BOY":"hi-IN-SwaraNeural",
    },
    "Tamil": {
        "male":"ta-IN-ValluvarNeural","female":"ta-IN-PallaviNeural",
        "NARRATOR":"ta-IN-ValluvarNeural","FATHER":"ta-IN-ValluvarNeural",
        "MOTHER":"ta-IN-PallaviNeural","OLD_MAN":"ta-IN-ValluvarNeural",
        "SON":"ta-IN-PallaviNeural","BOY":"ta-IN-PallaviNeural",
    },
    "Telugu": {
        "male":"te-IN-MohanNeural","female":"te-IN-ShrutiNeural",
        "NARRATOR":"te-IN-MohanNeural","FATHER":"te-IN-MohanNeural",
        "MOTHER":"te-IN-ShrutiNeural","SON":"te-IN-ShrutiNeural",
    },
    "Bengali": {
        "male":"bn-IN-BashkarNeural","female":"bn-IN-TanishaaNeural",
        "NARRATOR":"bn-IN-BashkarNeural","FATHER":"bn-IN-BashkarNeural",
        "MOTHER":"bn-IN-TanishaaNeural","SON":"bn-IN-TanishaaNeural",
    },
    "Marathi": {
        "male":"mr-IN-ManoharNeural","female":"mr-IN-AarohiNeural",
        "NARRATOR":"mr-IN-ManoharNeural","FATHER":"mr-IN-ManoharNeural",
        "MOTHER":"mr-IN-AarohiNeural","SON":"mr-IN-AarohiNeural",
    },
    "Gujarati": {
        "male":"gu-IN-NiranjanNeural","female":"gu-IN-DhwaniNeural",
        "NARRATOR":"gu-IN-NiranjanNeural","FATHER":"gu-IN-NiranjanNeural",
        "MOTHER":"gu-IN-DhwaniNeural","SON":"gu-IN-DhwaniNeural",
    },
    "Kannada": {
        "male":"kn-IN-GaganNeural","female":"kn-IN-SapnaNeural",
        "NARRATOR":"kn-IN-GaganNeural","FATHER":"kn-IN-GaganNeural",
        "MOTHER":"kn-IN-SapnaNeural","SON":"kn-IN-SapnaNeural",
    },
    "Malayalam": {
        "male":"ml-IN-MidhunNeural","female":"ml-IN-SobhanaNeural",
        "NARRATOR":"ml-IN-MidhunNeural","FATHER":"ml-IN-MidhunNeural",
        "MOTHER":"ml-IN-SobhanaNeural","SON":"ml-IN-SobhanaNeural",
    },
    "Urdu": {
        "male":"ur-IN-SalmanNeural","female":"ur-IN-GulNeural",
        "NARRATOR":"ur-IN-SalmanNeural","FATHER":"ur-IN-SalmanNeural",
        "MOTHER":"ur-IN-GulNeural","SON":"ur-IN-GulNeural",
    },
    "English": {
        "male":"en-US-AndrewNeural","female":"en-US-AvaNeural",
        "NARRATOR":"en-US-AndrewNeural","FATHER":"en-US-AndrewNeural",
        "MOTHER":"en-US-AvaNeural","OLD_MAN":"en-US-AndrewNeural",
        "HERO":"en-US-AndrewNeural","HEROINE":"en-US-AvaNeural",
        "SON":"en-US-AvaNeural","BOY":"en-US-AvaNeural",
        "VILLAIN":"en-US-AndrewNeural","GIRL":"en-US-AvaNeural",
    },
    "Spanish": {
        "male":"es-MX-JorgeNeural","female":"es-MX-DaliaNeural",
        "NARRATOR":"es-MX-JorgeNeural","FATHER":"es-MX-JorgeNeural",
        "MOTHER":"es-MX-DaliaNeural","SON":"es-MX-DaliaNeural",
    },
    "French": {
        "male":"fr-FR-RemyMultilingualNeural","female":"fr-FR-VivienneMultilingualNeural",
        "NARRATOR":"fr-FR-RemyMultilingualNeural","FATHER":"fr-FR-RemyMultilingualNeural",
        "MOTHER":"fr-FR-VivienneMultilingualNeural","SON":"fr-FR-VivienneMultilingualNeural",
    },
    "Portuguese": {
        "male":"pt-BR-AntonioNeural","female":"pt-BR-FranciscaNeural",
        "NARRATOR":"pt-BR-AntonioNeural","SON":"pt-BR-FranciscaNeural",
    },
    "German": {
        "male":"de-DE-FlorianMultilingualNeural","female":"de-DE-SeraphinaMultilingualNeural",
        "NARRATOR":"de-DE-FlorianMultilingualNeural","SON":"de-DE-SeraphinaMultilingualNeural",
    },
    "Japanese": {
        "male":"ja-JP-KeitaNeural","female":"ja-JP-NanamiNeural",
        "NARRATOR":"ja-JP-KeitaNeural","SON":"ja-JP-NanamiNeural",
    },
    "Korean": {
        "male":"ko-KR-InJoonNeural","female":"ko-KR-SunHiNeural",
        "NARRATOR":"ko-KR-InJoonNeural","SON":"ko-KR-SunHiNeural",
    },
    "Arabic": {
        "male":"ar-SA-HamedNeural","female":"ar-SA-ZariyahNeural",
        "NARRATOR":"ar-SA-HamedNeural","SON":"ar-SA-ZariyahNeural",
    },
    "Turkish": {
        "male":"tr-TR-AhmetNeural","female":"tr-TR-EmelNeural",
        "NARRATOR":"tr-TR-AhmetNeural","SON":"tr-TR-EmelNeural",
    },
}

ROLE_GENDER = {
    "NARRATOR":"male","FATHER":"male","MOTHER":"female","SON":"female",
    "DAUGHTER":"female","OLD_MAN":"male","OLD_WOMAN":"female",
    "YOUNG_MAN":"male","YOUNG_WOMAN":"female","GIRL":"female","BOY":"female",
    "VILLAIN":"male","HERO":"male","HEROINE":"female",
    "CHAR_A":"male","CHAR_B":"female","CHAR_C":"male","CHAR_D":"female",
}

def cast(dir_result, target_lang="Hindi"):
    vm = VOICE_MAP.get(target_lang, VOICE_MAP.get("English"))
    if not vm:
        vm = VOICE_MAP["English"]
    
    cast_map = {}
    for v in dir_result.get("voice_plan", []):
        vid = v["voice_id"]
        gender = v.get("gender", ROLE_GENDER.get(vid, "male"))
        voice = vm.get(vid, vm.get(gender, vm.get("male", list(vm.values())[0])))
        cast_map[vid] = voice
        logger.info(f"[CAST] {vid} → {voice}")
    
    if not cast_map:
        cast_map["NARRATOR"] = vm.get("male", list(vm.values())[0])
    return cast_map

def get_supported_languages():
    return list(VOICE_MAP.keys())
