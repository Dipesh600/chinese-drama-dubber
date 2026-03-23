# 🎬 Chinese Drama Dubber

**AI-powered dubbing pipeline that translates and dubs Chinese/Asian drama videos into 18+ languages.**

Uses LLM Director for content understanding, Groq Whisper for transcription, Llama-4 for translation, creative dialogue rewriting, and Microsoft Edge Neural TTS for free voice generation.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Languages](https://img.shields.io/badge/Languages-18+-orange)
![TTS](https://img.shields.io/badge/TTS-Microsoft_Edge-purple)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Director** | Analyzes full transcript to understand content type, identify real speakers, assign moods per segment |
| 🌐 **18 Languages** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Urdu, English, Spanish, French, Portuguese, German, Japanese, Korean, Arabic, Turkish |
| 🗣️ **Smart Voice Cast** | Different Edge TTS voices + rate/pitch per character (NARRATOR, FATHER, SON, OLD_MAN, etc.) |
| ✍️ **Dialogue Writer** | Creative rewrite pass — transforms literal translations into natural, emotional, timing-aware dialogue |
| 🎵 **Background Audio** | Intelligent ducking — original music/ambient at 30% in gaps, 8% during speech |
| 🔤 **Romanizer** | Auto-converts Devanagari script to Roman for consistent TTS pronunciation |
| ⚡ **Quality Validator** | Pre-TTS checks: word count limits, hallucination detection, auto-trimming |
| 📝 **Subtitles** | Auto-generated .srt files with per-line timing |
| 🆓 **100% Free** | Only needs a free Groq API key — Edge TTS requires no API key |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    10-STAGE PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│  1. DOWNLOAD        yt-dlp → video.mp4 + audio.mp3          │
│  2. TRANSCRIBE      Groq Whisper large-v3 → segments        │
│  3. DIRECTOR        LLM content analysis + speaker diarize  │
│  4. PRE-PROCESS     Merge micro-segments (<1.5s)            │
│  5. TRANSLATE       Llama-4-Scout → raw translation         │
│  5b. ROMANIZE       Devanagari → Roman script               │
│  6. DIALOGUE WRITER Creative rewrite with mood + timing     │
│  6b. VALIDATE       Quality gate before TTS                 │
│  7. SENTENCE MERGE  Combine into natural TTS units          │
│  8. VOICE CAST      Assign Edge TTS voices per character    │
│  9. TTS             Edge Neural TTS + overlap prevention    │
│ 10. ASSEMBLE        Mix + smart ducking + subs + merge      │
└─────────────────────────────────────────────────────────────┘
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

# Or download static binary
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ
sudo cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffprobe /usr/local/bin/
```

### 4. Run

```python
from orchestrator import run_agent

result = run_agent(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    target_lang="Hindi",        # Any of 18 supported languages
    source_lang="zh",           # "zh", "en", "auto"
    user_description="A story about a father teaching his son",  # Helps the Director
    preserve_bg=True            # Keep original background music
)

if result["success"]:
    print(f"✅ Done! Video: {result['video_path']}")
    print(f"   Subtitles: {result['srt_path']}")
    print(f"   Content: {result['content_type']} | {result['real_speaker_count']} speakers")
```

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
| 🇫🇷 French | RemyNeural | VivienneNeural |
| 🇧🇷 Portuguese | AntonioNeural | FranciscaNeural |
| 🇩🇪 German | FlorianNeural | SeraphinaNeural |
| 🇯🇵 Japanese | KeitaNeural | NanamiNeural |
| 🇰🇷 Korean | InJoonNeural | SunHiNeural |
| 🇸🇦 Arabic | HamedNeural | ZariyahNeural |
| 🇹🇷 Turkish | AhmetNeural | EmelNeural |

---

## 📁 Module Reference

| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Main pipeline orchestrator — runs all 10 stages |
| `director.py` | LLM content analysis, speaker diarization, mood tracking |
| `preprocessor.py` | Merges micro-segments to prevent translation hallucination |
| `translator.py` | Llama-4-Scout translation with auto-fallback to 8b |
| `romanizer.py` | Devanagari → Roman script conversion |
| `dialogue_writer.py` | Creative dialogue polishing with mood + word limits |
| `validator.py` | Pre-TTS quality gate (word count, hallucination, empty text) |
| `sentence_merger.py` | Combines short segments into natural TTS units |
| `voice_caster.py` | Maps characters to Edge TTS voices per language |
| `tts_engine.py` | Edge TTS generation with rate/pitch per character |
| `assembler.py` | FFmpeg assembly with smart audio ducking + subtitles |

---

## 🔧 How It Works

### LLM Director
The Director reads the full transcript and determines:
- **Content type**: `single_narrator` / `dialogue_drama` / `mixed`
- **Real speaker count**: Based on dialogue patterns, not audio gaps
- **Mood per segment**: neutral, happy, sad, angry, tense, wise, etc.
- **Voice plan**: Which characters need which voice type

### Dialogue Writer
Takes raw translations and rewrites them:
```
RAW:   "Usne halke se kaha, Bete, cupboard mein ek ghadi rakhi hai."
FINAL: "Bete, jao cupboard se ghadi nikaal kar laao."
```
- Respects word count limits (2.8 words/sec) to prevent speed-up
- Adds fillers only to character dialogue, never narrator
- Matches mood: angry=sharp, sad=soft, wise=measured

### Smart Audio Ducking
Instead of flat background volume:
- **30% volume** during gaps (original music comes through)
- **8% volume** during dubbed speech (voice is clear)
- Smooth transitions between levels

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Your Groq API key ([get free](https://console.groq.com/keys)) |

### Groq Models Used

| Model | Purpose | Rate Limit |
|-------|---------|------------|
| `whisper-large-v3` | Audio transcription | Separate quota |
| `llama-3.3-70b-versatile` | Director analysis | 100K tokens/day |
| `llama-4-scout-17b` | Translation + dialogue | Separate quota |
| `llama-3.1-8b-instant` | Fallback translation | Separate quota |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| 3.8 min video | ~160-240s processing |
| 15 min video | ~400-600s processing |
| Output quality | Near-broadcast for narration |
| Cost | $0 (all free APIs) |

---

## 🤝 Contributing

PRs welcome! Key areas for improvement:
- [ ] ElevenLabs/Azure TTS integration for premium voice quality
- [ ] Lip-sync consideration for live-action content
- [ ] Batch processing for multi-episode series
- [ ] Web UI for non-technical users

---

## 📄 License

MIT License — use freely for personal and commercial projects.

---

<p align="center">
  Built with ❤️ using <a href="https://groq.com">Groq</a> + <a href="https://github.com/rany2/edge-tts">Edge TTS</a>
</p>
