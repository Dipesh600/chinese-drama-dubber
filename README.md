# 🎬 Chinese Drama Dubber v6.0

**AI-powered dubbing pipeline that translates and dubs Chinese/Asian drama videos into 18+ languages.**

Uses LLM Director for content understanding, Groq Whisper for transcription, Llama-3.3-70B for translation with scene-aware parallelism, creative dialogue rewriting, and **Fish Speech (local GPU) as primary TTS** with Edge TTS as fallback.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Languages](https://img.shields.io/badge/Languages-18+-orange)
![TTS](https://img.shields.io/badge/TTS-Fish_Speech_Primary-purple)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Director** | Analyzes full transcript to understand content type, identify real speakers, assign moods per segment |
| 🌐 **18 Languages** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Urdu, English, Spanish, French, Portuguese, German, Japanese, Korean, Arabic, Turkish |
| 🎤 **Fish Speech Primary TTS** | Local GPU TTS (best quality, zero cost) → Edge TTS fallback |
| ⚡ **Parallel LLM Processing** | Scene-based parallel translation and dialogue writing |
| 🗣️ **Smart Voice Cast** | Different Edge TTS voices + rate/pitch per character (NARRATOR, FATHER, SON, OLD_MAN, etc.) |
| ✍️ **Dialogue Writer** | Creative rewrite pass — transforms literal translations into natural, emotional, timing-aware dialogue |
| 🎵 **Smart Demucs** | Skips slow Demucs separation if background is already clean (saves 5-10 min) |
| 🎵 **Background Audio** | Intelligent ducking — original music/ambient at 35% in gaps, 8% during speech |
| 🔤 **Romanizer** | Auto-converts Devanagari script to Roman for consistent TTS pronunciation |
| ⚡ **Quality Validator** | Pre-TTS checks: word count limits, hallucination detection, auto-trimming |
| 📝 **Subtitles** | Auto-generated .srt files with per-line timing |
| 📊 **Structured Logging** | JSON logs with correlation IDs for pipeline tracking |
| 🧪 **Test Suite** | Unit tests for core modules |

---

## 🏗️ Pipeline Architecture (v6.0)

```
┌───────────────────────────────────────────────────────────────────────┐
│                    13-STAGE PIPELINE (v6.0)                          │
├───────────────────────────────────────────────────────────────────────┤
│  1. DOWNLOAD         yt-dlp → video.mp4 + audio.mp3                │
│  2. SMART DETECT     Vocal bleed check (skip Demucs if clean)       │
│  3. DEMUCS           Vocal/music separation (if needed)              │
│  4. TRANSCRIBE       Groq Whisper large-v3 → word-level segments    │
│  5. DIRECTOR         Parallel scene LLM analysis + speaker diarize   │
│  6. PRE-PROCESS      Merge micro-segments (<1.5s)                   │
│  7. TRANSLATE        Two-pass: Draft → Polish (SCENE-PARALLEL)      │
│  7b. ROMANIZE        Devanagari → Roman script                      │
│  8. DIALOGUE WRITER  Creative rewrite with mood + timing (PARALLEL)  │
│  8b. VALIDATE        Quality gate before TTS                          │
│  9. SENTENCE MERGE   Combine into natural TTS units                  │
│  10. VOICE CAST       Assign TTS voices per character                │
│  11. TTS              Fish Speech PRIMARY → Edge TTS fallback         │
│                       (PARALLEL clip generation, 5 workers)           │
│  12. ALIGN            Timestamp alignment (rubberband stretch)       │
│  13. ASSEMBLE         Mix + smart ducking + subs + merge             │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Dipesh600/chinese-drama-dubber.git
cd chinese-drama-dubber
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com/keys)
```

### 3. Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 4. Optional: Fish Speech TTS (recommended for best quality)

```bash
# Install Fish Speech: https://github.com/fishaudio/fish-speech
# Run the server:
python -m fish_audio.main --listen localhost:8080
```

### 5. Run

```python
from orchestrator import run_agent

result = run_agent(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    target_lang="Hindi",        # Any of 18 supported languages
    source_lang="zh",           # "zh", "en", "auto"
    user_description="A story about a father teaching his son",
    preserve_bg=True            # Keep original background music
)

if result["success"]:
    print(f"✅ Done! Video: {result['video_path']}")
    print(f"   Subtitles: {result['srt_path']}")
    print(f"   Content: {result['content_type']} | {result['real_speaker_count']} speakers")
```

---

## 🆕 v6.0 Improvements

| Feature | Before | After |
|---------|--------|-------|
| **TTS Priority** | Edge TTS primary | Fish Speech PRIMARY → Edge TTS fallback |
| **LLM Parallelism** | Sequential batches | Scene-based parallel (2-3x faster) |
| **Demucs** | Always runs (slow) | Smart skip if background clean |
| **TTS Generation** | Sequential | Parallel (5 workers) |
| **Config** | Duplicated across modules | Single `config.py` |
| **LLM Abstraction** | Groq only | `llm_provider.py` (Groq/Ollama/Gemini ready) |
| **Logging** | Plain text | Structured JSON + human-readable |
| **Tests** | None | Unit tests for core modules |

---

## 🌐 Supported Languages

| Language | Male Voice | Female Voice |
|----------|-----------|--------------|
| 🇮🇳 Hindi | MadhurNeural | SwaraNeural |
| 🇮🇳 Tamil | ValluvarNeural | PallaviNeural |
| 🇮🇳 Telugu | MohanNeural | ShrutiNeural |
| 🇮🇳 Bengali | BashkarNeural | TanishaaNeural |
| 🇮🇳 Marathi | ManoharNeural | AarohiNeural |
| 🇮🇳 Gujarati | NiranjanNeural | DhwaniNeural |
| 🇮🇳 Kannada | GaganNeural | SapnaNeural |
| 🇮🇳 Malayalam | MidhunNeural | SobhanaNeural |
| 🇮🇳 Urdu | SalmanNeural | GulNeural |
| 🇺🇸 English | AndrewNeural | AvaNeural |
| 🇪🇸 Spanish | JorgeNeural | DaliaNeural |
| 🇫🇷 French | RemyMultilingualNeural | VivienneMultilingualNeural |
| 🇧🇷 Portuguese | AntonioNeural | D-AlmeidaNeural |
| 🇩🇪 German | FlorianMultilingualNeural | SeraphinaMultilingualNeural |
| 🇯🇵 Japanese | KeitaNeural | NanamiNeural |
| 🇰🇷 Korean | InJoonNeural | SunHiNeural |
| 🇸🇦 Arabic | HamedNeural | ZariyahNeural |
| 🇹🇷 Turkish | AhmetNeural | EmelNeural |

---

## 📁 Module Reference

| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Main pipeline orchestrator — runs all 13 stages |
| `config.py` | **NEW** — Single source of truth for all configuration |
| `llm_provider.py` | **NEW** — LLM abstraction layer (Groq/Ollama/Gemini) |
| `logging_utils.py` | **NEW** — Structured logging with correlation IDs |
| `director.py` | LLM content analysis, speaker diarization, mood tracking |
| `preprocessor.py` | Merges micro-segments to prevent translation hallucination |
| `translator.py` | **Parallel** two-pass translation (scene-based) |
| `dialogue_writer.py` | **Parallel** creative dialogue polishing |
| `romanizer.py` | Devanagari → Roman script conversion |
| `validator.py` | Pre-TTS quality gate |
| `sentence_merger.py` | Combines short segments into natural TTS units |
| `voice_caster.py` | Maps characters to TTS voices per language |
| `tts_engine.py` | Fish Speech PRIMARY → Edge TTS fallback, parallel generation |
| `audio_separator.py` | **Smart** Demucs with vocal bleed detection |
| `assembler.py` | FFmpeg assembly with smart audio ducking + subtitles |
| `timestamp_aligner.py` | Rubberband time-stretch for lip sync |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Your Groq API key ([get free](https://console.groq.com/keys)) |
| `FISH_SPEECH_URL` | Optional | Fish Speech server URL (default: http://localhost:8080) |
| `TTS_WORKERS` | Optional | Parallel TTS workers (default: 5) |
| `OUTPUT_DIR` | Optional | Output directory (default: /content/drive/MyDrive/DubbedVideos) |

### Groq Models Used

| Model | Purpose | Rate Limit |
|-------|---------|------------|
| `whisper-large-v3` | Audio transcription | Separate quota |
| `llama-3.3-70b-versatile` | Director + Dialogue Writer | 100K tokens/day |
| `llama-4-scout-17b-16e` | Translation | Separate quota |
| `llama-3.1-8b-instant` | Fallback translation | Separate quota |

---

## 🧪 Running Tests

```bash
python tests/test_config.py
python tests/test_sentence_merger.py
python tests/test_preprocessor.py
python tests/test_romanizer.py
python tests/test_voice_caster.py
```

---

## 🤝 Contributing

PRs welcome! Key areas for improvement:
- [ ] Ollama local LLM integration
- [ ] Lip-sync for live-action content
- [ ] Batch processing for multi-episode series
- [ ] Web UI for non-technical users
- [ ] ElevenLabs premium TTS integration

---

## 📄 License

MIT License — use freely for personal and commercial projects.

---

<p align="center">
  Built with ❤️ using <a href="https://groq.com">Groq</a> + <a href="https://github.com/fishaudio/fish-speech">Fish Speech</a>
</p>
