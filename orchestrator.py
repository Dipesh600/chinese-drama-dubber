"""
ORCHESTRATOR v6.0 — Industry-standard dubbing pipeline:
  1. Demucs audio separation (clean background, no vocal bleed)
  2. Whisper transcription with word-level timestamps (Groq API)
  3. Director v5: chunked LLM analysis, pitch hints, scene detection, speaker smoothing
  4. Two-pass translation: Draft → Polish (scene-aware) via LLM provider
  5. Dialogue Writer v5: voice bible, scene-grouped, pronunciation hints
  6. Sentence Merger: combines short segments for natural TTS
  7. Voice Caster: SSML prosody profiles per character
  8. Fish Speech TTS PRIMARY → Edge TTS fallback (parallel generation)
  9. Timestamp Aligner: rubberband time-stretch, silence padding
  10. Professional assembly: batch FFmpeg, smart ducking, EBU R128
  11. Subtitle generation with aligned timestamps
  12. State persistence for crash recovery
  13. Configuration via config.py (single source of truth)
"""
import os, sys, json, logging, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import director, preprocessor, translator, dialogue_writer, sentence_merger
import voice_caster, tts_engine, assembler, romanizer, validator
import audio_separator, timestamp_aligner
from logging_utils import setup_logging, StageTracker, PipelineContext, generate_correlation_id

setup_logging(structured=False, level=logging.INFO)
log = logging.getLogger(__name__)


class DubberV6:
    """Industry-standard dubbing pipeline."""

    def __init__(self, url, target_lang="Hindi", source_lang="zh",
                 user_description="", output_dir="/content/drive/MyDrive/DubbedVideos",
                 preserve_bg=True, use_demucs=True):
        self.url = url
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.user_description = user_description
        self.preserve_bg = preserve_bg
        self.use_demucs = use_demucs
        self.correlation_id = generate_correlation_id()

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.work_dir = os.path.join(output_dir, f"run_{ts}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.sp = os.path.join(self.work_dir, "state.json")
        self.state = {}
        self.ctx = PipelineContext(url, target_lang, source_lang)
    
    def _done(self, k):
        return self.state.get(k, {}).get("done", False)
    
    def _save(self, k, v):
        self.state[k] = {"done": True, **v}
        with open(self.sp, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def run(self):
        t0 = time.time()

        log.info("=" * 70)
        log.info(f"  🎬 DUBBER v6.0 — Industry-Standard Pipeline")
        log.info(f"  Correlation: {self.correlation_id[:8]}")
        log.info(f"  Language:    {self.target_lang}")
        log.info(f"  Source:      {self.source_lang}")
        log.info(f"  URL:         {self.url}")
        log.info(f"  BG Audio:    {'Demucs + Smart Ducking' if self.use_demucs else 'Smart Ducking'}")
        log.info(f"  TTS Engine:  Fish Speech PRIMARY → Edge TTS fallback")
        log.info(f"  Features:    Two-pass translation | Scene detection | Voice bible")
        log.info(f"               Parallel TTS | Batch FFmpeg assembly")
        log.info("=" * 70)
        
        try:
            video_path = os.path.join(self.work_dir, "video.mp4")
            audio_path = os.path.join(self.work_dir, "audio.mp3")
            
            # ━━ STEP 1: DOWNLOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not self._done("s1"):
                log.info(f"[1/13] 📥 Downloading: {self.url}")
                import yt_dlp
                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": video_path,
                    "quiet": True, "no_warnings": True
                }
                
                # YouTube anti-bot: try cookie authentication
                cookie_file = os.environ.get("COOKIE_FILE", "")
                cookie_paths = [
                    cookie_file,
                    os.path.join(os.path.dirname(__file__), "cookies.txt"),
                    "/content/cookies.txt",
                    os.path.expanduser("~/cookies.txt"),
                ]
                for cp in cookie_paths:
                    if cp and os.path.exists(cp):
                        ydl_opts["cookiefile"] = cp
                        log.info(f"[1/13] 🍪 Using cookies: {cp}")
                        break
                
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([self.url])
                except Exception as dl_err:
                    if "Sign in" in str(dl_err) or "bot" in str(dl_err).lower():
                        log.error(
                            "[1/13] ❌ YouTube blocked the download (anti-bot).\n"
                            "       FIX: Export your YouTube cookies and upload cookies.txt\n"
                            "       HOW:\n"
                            "         1. Install browser extension: 'Get cookies.txt LOCALLY'\n"
                            "         2. Go to youtube.com (make sure you're logged in)\n"
                            "         3. Click the extension → Export cookies\n"
                            "         4. Upload the cookies.txt file to Colab (/content/cookies.txt)\n"
                            "         5. Re-run this cell"
                        )
                    raise
                
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000",
                    "-ac", "1", "-b:a", "64k", audio_path
                ], capture_output=True, timeout=300)
                
                audio_mb = os.path.getsize(audio_path) / 1024 / 1024
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    capture_output=True, text=True
                )
                dur = float(r.stdout.strip())
                self._save("s1", {"audio_mb": round(audio_mb, 1), "duration": round(dur, 1)})
                log.info(f"[1/13] ✓ Audio: {audio_mb:.1f}MB | {dur:.0f}s ({dur/60:.1f}min)")
            
            # ━━ STEP 1b: AUDIO SEPARATION (Demucs) ━━━━━━━━━━━━━━━━━━
            bg_audio_path = None
            if self.use_demucs and self.preserve_bg:
                if not self._done("s1b"):
                    try:
                        log.info(f"[1b/13] 🎵 Separating audio (Demucs)...")
                        stems = audio_separator.separate(audio_path, self.work_dir)
                        bg_audio_path = stems.get("bg_path")
                        self._save("s1b", {"bg_path": bg_audio_path or "", "success": True})
                        log.info(f"[1b/13] ✓ Clean background extracted!")
                    except Exception as e:
                        log.warning(f"[1b/13] ⚠ Demucs failed ({e}), using original audio")
                        self._save("s1b", {"success": False, "error": str(e)})
                else:
                    bg_audio_path = self.state.get("s1b", {}).get("bg_path")
            
            # ━━ STEP 2: TRANSCRIBE (Whisper) ━━━━━━━━━━━━━━━━━━━━━━━━
            whisper_path = os.path.join(self.work_dir, "whisper.json")
            if not self._done("s2"):
                log.info(f"[2/13] 🎤 Transcribing (Groq Whisper large-v3)...")
                from llm_provider import get_llm
                llm = get_llm()
                result = llm.transcribe(audio_path, language=self.source_lang)
                if result:
                    segs = result["segments"]
                    words = result.get("words", [])
                else:
                    raise RuntimeError("Transcription failed")
                with open(whisper_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segs, "words": words}, f, indent=2, ensure_ascii=False)
                self._save("s2", {"segments": len(segs), "words": len(words)})
                log.info(f"[2/13] ✓ {len(segs)} segments, {len(words)} word timestamps")
            
            with open(whisper_path, encoding="utf-8") as f:
                whisper_data = json.load(f)
            segments = whisper_data["segments"]
            whisper_words = whisper_data.get("words", [])
            
            # ━━ STEP 3: DIRECTOR v4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            director_path = os.path.join(self.work_dir, "director_plan.json")
            if not self._done("s3"):
                log.info(f"[3/13] 🎬 DIRECTOR analyzing content...")
                log.info(f"[3/13]    + Chunked analysis (30-seg windows)")
                log.info(f"[3/13]    + Pitch-based speaker hints")
                log.info(f"[3/13]    + Audio energy analysis")
                log.info(f"[3/13]    + Scene boundary detection")
                log.info(f"[3/13]    + Speaker consistency smoothing")
                dir_result = director.analyze(
                    segments, self.work_dir,
                    user_description=self.user_description,
                    whisper_words=whisper_words,
                    audio_path=audio_path
                )
                self._save("s3", {
                    "content_type": dir_result["content_type"],
                    "speakers": dir_result["real_speaker_count"],
                    "scenes": len(dir_result.get("scenes", []))
                })
            
            with open(director_path, encoding="utf-8") as f:
                dir_result = json.load(f)
            
            sm = dir_result.get("speaker_map", {})
            moods = dir_result.get("segment_moods", {})
            for s in segments:
                sid = str(s["id"])
                s["speaker"] = sm.get(sid, "NARRATOR")
                s["mood"] = moods.get(sid, "neutral")
            dir_result["segments"] = segments
            
            n_scenes = len(dir_result.get("scenes", []))
            log.info(
                f"[3/13] ✓ Type={dir_result['content_type']} | "
                f"Speakers={dir_result['real_speaker_count']} | "
                f"Scenes={n_scenes}"
            )
            
            # ━━ STEP 4: PRE-PROCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[4/13] 🔧 Pre-processing: merge micro-segments...")
            merged = preprocessor.merge_short_segments(dir_result["segments"])
            dir_result["segments"] = merged
            
            # ━━ STEP 5: TRANSLATE (Two-Pass) ━━━━━━━━━━━━━━━━━━━━━━━━
            raw_path = os.path.join(self.work_dir, "translated_raw.json")
            if not self._done("s5"):
                log.info(f"[5/13] 🌐 Translating → {self.target_lang} (two-pass)...")
                log.info(f"[5/13]    Pass 1: Draft translation with context")
                log.info(f"[5/13]    Pass 2: Scene-aware polish for coherence")
                translator.translate(dir_result, self.work_dir, self.target_lang)
                self._save("s5", {"raw_path": raw_path})
            with open(raw_path, encoding="utf-8") as f:
                raw_script = json.load(f)
            
            # ━━ STEP 5b: ROMANIZE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[5b/13] ✏️ Romanizing Devanagari → Roman script...")
            romanizer.romanize_segments(raw_script["segments"])
            
            # ━━ STEP 6: DIALOGUE WRITER v4 ━━━━━━━━━━━━━━━━━━━━━━━━━━
            dubbed_path = os.path.join(self.work_dir, "dubbed_script.json")
            if not self._done("s6"):
                log.info(f"[6/13] ✍️ DIALOGUE WRITER (voice bible + scene-grouped)...")
                dialogue_writer.rewrite(
                    raw_script["segments"], self.work_dir, self.target_lang,
                    narrative_summary=dir_result.get("narrative_summary", ""),
                    mood_arc=dir_result.get("mood_arc"),
                    voice_plan=dir_result.get("voice_plan"),
                    scenes=dir_result.get("scenes"),
                )
                self._save("s6", {"dubbed_path": dubbed_path})
            with open(dubbed_path, encoding="utf-8") as f:
                dubbed_script = json.load(f)
            
            # ━━ STEP 6b: ROMANIZE (post-writer) ━━━━━━━━━━━━━━━━━━━━━
            rom_fixed = romanizer.romanize_segments(dubbed_script["segments"])
            if rom_fixed:
                with open(dubbed_path, "w", encoding="utf-8") as f:
                    json.dump(dubbed_script, f, indent=2, ensure_ascii=False)
            
            # Expand merged segments
            expanded = preprocessor.expand_dubbed_to_subsegments(dubbed_script["segments"])
            
            # ━━ STEP 6c: VALIDATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[6c/13] ✅ Validating script quality...")
            validator.validate(expanded, auto_fix=True)
            
            # ━━ STEP 7: SENTENCE MERGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[7/13] 📝 Merging into sentence-level TTS units...")
            tts_segments = sentence_merger.merge_for_tts(expanded)
            
            # ━━ STEP 8: VOICE CASTING (SSML Differentiation) ━━━━━━━━
            log.info(f"[8/13] 🎤 Voice casting (SSML prosody differentiation)...")
            cast_map = voice_caster.cast(dir_result, self.target_lang)
            log.info(f"[8/13] ✓ {len(cast_map)} voice profiles locked")
            
            # ━━ STEP 9: TTS GENERATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            tts_manifest_path = os.path.join(self.work_dir, "tts_manifest.json")
            if not self._done("s9"):
                log.info(f"[9/13] 🔊 Generating TTS ({len(tts_segments)} units)...")
                tts_manifest = tts_engine.generate(
                    tts_segments, self.work_dir, cast_map, self.target_lang
                )
                self._save("s9", {"clips": len(tts_manifest)})
            with open(tts_manifest_path) as f:
                tts_manifest = json.load(f)
            log.info(f"[9/13] ✓ {len(tts_manifest)} clips generated")
            
            # ━━ STEP 10: TIMESTAMP ALIGNMENT ━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[10/13] ⏱️ Timestamp alignment (rubberband stretch + padding)...")
            tts_manifest = timestamp_aligner.align(tts_manifest, self.work_dir)
            
            # ━━ STEP 11: ASSEMBLE (Professional Mix) ━━━━━━━━━━━━━━━━
            if not self._done("s11"):
                log.info(f"[11/13] 🎵 Assembling (professional mix)...")
                
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    capture_output=True, text=True
                )
                total_dur = float(r.stdout.strip())
                
                if bg_audio_path and os.path.exists(bg_audio_path):
                    log.info(f"[11/13] Using Demucs clean background ✓")
                else:
                    bg_audio_path = None
                
                asm_r = assembler.assemble(
                    tts_manifest, self.work_dir, video_path, total_dur,
                    preserve_bg=self.preserve_bg,
                    bg_audio_path=bg_audio_path
                )
                self._save("s11", asm_r)
            
            with open(self.sp) as f:
                final = json.load(f)
            asm_r = {k: v for k, v in final.get("s11", {}).items() if k != "done"}
            
            # ━━ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elapsed = round(time.time() - t0, 1)
            size_mb = asm_r.get("size_mb", 0)
            
            log.info("=" * 70)
            log.info(f"  ✅ DONE in {elapsed}s ({elapsed/60:.1f}min) | {size_mb}MB")
            log.info(f"  📹 Video:      {asm_r.get('video_path', '?')}")
            log.info(f"  📝 Subtitles:  {asm_r.get('srt_path', '?')}")
            log.info(f"  🌐 Language:   {self.target_lang}")
            log.info(f"  🎬 Content:    {dir_result.get('content_type', '?')} | {dir_result.get('real_speaker_count', '?')} speakers")
            log.info(f"  🎵 BG Audio:   {'Demucs separation' if bg_audio_path else 'smart ducking'}")
            log.info(f"  🎭 Scenes:     {n_scenes}")
            log.info("=" * 70)
            
            return {
                "success": True,
                "video_path": asm_r.get("video_path", ""),
                "srt_path": asm_r.get("srt_path", ""),
                "size_mb": size_mb,
                "processing_time": elapsed,
                "content_type": dir_result.get("content_type", "?"),
                "real_speaker_count": dir_result.get("real_speaker_count", 1),
                "target_lang": self.target_lang,
                "work_dir": self.work_dir,
                "scenes": n_scenes,
            }
            
        except Exception as e:
            import traceback
            log.error(f"❌ Pipeline error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "work_dir": self.work_dir,
                "stage": self._get_current_stage()
            }
    
    def _get_current_stage(self):
        stages = ["s1", "s1b", "s2", "s3", "s5", "s6", "s9", "s11"]
        for s in stages:
            if not self._done(s):
                return s
        return "unknown"


def run_agent(url, target_lang="Hindi", source_lang="zh",
              user_description="", output_dir="/content/drive/MyDrive/DubbedVideos",
              preserve_bg=True, use_demucs=True):
    """
    Run the industry-standard dubbing pipeline.
    
    Args:
        url: YouTube video URL
        target_lang: Target language (Hindi, Tamil, Telugu, Bengali, Nepali, English, etc.)
        source_lang: Source language code (zh, en, ja, ko, etc.)
        user_description: Optional description of the video content
        output_dir: Base directory for output files
        preserve_bg: Keep background music/ambient audio
        use_demucs: Use Demucs to separate vocals from background
    
    Returns:
        dict with success status, paths, and metadata
    """
    d = DubberV6(
        url=url, target_lang=target_lang, source_lang=source_lang,
        user_description=user_description, output_dir=output_dir,
        preserve_bg=preserve_bg, use_demucs=use_demucs
    )
    return d.run()
