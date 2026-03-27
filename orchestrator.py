"""
ORCHESTRATOR v5.0 — Production-grade dubbing pipeline with:
1. Demucs audio separation (clean background, no Chinese vocal bleed)
2. Word-level timestamp analysis for speaker diarization
3. Audio energy analysis for mood detection
4. Context-aware translation with rolling window
5. Emotion-aware dialogue rewriting with character consistency
6. Fish Audio TTS with regional voice catalog (2M+ voices)
7. EBU R128 loudness normalization
8. Professional audio mixing with smart ducking
9. Subtitle generation
10. State persistence for crash recovery
"""
import os, sys, json, logging, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import director, preprocessor, translator, dialogue_writer, sentence_merger
import voice_catalog, tts_engine, assembler, romanizer, validator, audio_separator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


class DubberV5:
    """Production-grade dubbing pipeline."""
    
    def __init__(self, url, target_lang="Hindi", source_lang="zh",
                 user_description="", output_dir="/content/drive/MyDrive/DubbedVideos",
                 preserve_bg=True, use_fish_audio=True, use_demucs=True):
        self.url = url
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.user_description = user_description
        self.preserve_bg = preserve_bg
        self.use_fish_audio = use_fish_audio
        self.use_demucs = use_demucs
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.work_dir = os.path.join(output_dir, f"run_{ts}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.sp = os.path.join(self.work_dir, "state.json")
        self.state = {}
    
    def _done(self, k):
        return self.state.get(k, {}).get("done", False)
    
    def _save(self, k, v):
        self.state[k] = {"done": True, **v}
        with open(self.sp, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def run(self):
        t0 = time.time()
        
        # Check Fish Speech LOCAL server (runs on Colab GPU, no API key)
        fish_local = False
        if self.use_fish_audio:
            try:
                import httpx
                r = httpx.get("http://localhost:8080/", timeout=2.0)
                fish_local = r.status_code < 500
            except Exception:
                pass
        
        log.info("=" * 70)
        log.info(f"  🎬 DUBBER v5.0 — Production Pipeline")
        log.info(f"  Language:    {self.target_lang}")
        log.info(f"  Source:      {self.source_lang}")
        log.info(f"  URL:         {self.url}")
        log.info(f"  BG Audio:    {'Demucs Separation + Smart Ducking' if self.use_demucs else 'Smart Ducking'}")
        log.info(f"  TTS Engine:  {'🐟 Fish Speech LOCAL (GPU, no API key)' if fish_local else '🔊 Edge TTS (fallback)'}")
        log.info("=" * 70)
        
        try:
            video_path = os.path.join(self.work_dir, "video.mp4")
            audio_path = os.path.join(self.work_dir, "audio.mp3")
            
            # ━━ 1. DOWNLOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not self._done("s1"):
                log.info(f"[1/12] 📥 Downloading: {self.url}")
                import yt_dlp
                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": video_path,
                    "quiet": True, "no_warnings": True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([self.url])
                
                # Extract audio
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
                log.info(f"[1/12] ✓ Audio: {audio_mb:.1f}MB | {dur:.0f}s ({dur/60:.1f}min)")
            
            # ━━ 1b. AUDIO SEPARATION (Demucs) ━━━━━━━━━━━━━━━━━━━━━━
            bg_audio_path = None
            if self.use_demucs and self.preserve_bg:
                if not self._done("s1b"):
                    try:
                        log.info(f"[1b/12] 🎵 Separating audio (Demucs — removes Chinese vocals)...")
                        stems = audio_separator.separate(audio_path, self.work_dir)
                        bg_audio_path = stems.get("bg_path")
                        self._save("s1b", {
                            "bg_path": bg_audio_path or "",
                            "success": True
                        })
                        log.info(f"[1b/12] ✓ Clean background extracted!")
                    except Exception as e:
                        log.warning(f"[1b/12] ⚠ Demucs failed ({e}), using original audio")
                        self._save("s1b", {"success": False, "error": str(e)})
                else:
                    bg_audio_path = self.state.get("s1b", {}).get("bg_path")
            
            # ━━ 2. TRANSCRIBE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            whisper_path = os.path.join(self.work_dir, "whisper.json")
            if not self._done("s2"):
                log.info(f"[2/12] 🎤 Transcribing (Groq Whisper)...")
                from groq import Groq
                c = Groq(api_key=os.environ["GROQ_API_KEY"])
                with open(audio_path, "rb") as af:
                    resp = c.audio.transcriptions.create(
                        model="whisper-large-v3", file=af,
                        language=self.source_lang if self.source_lang != "auto" else None,
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"]
                    )
                segs = [{
                    "id": i,
                    "start": round(s["start"] if isinstance(s, dict) else s.start, 3),
                    "end": round(s["end"] if isinstance(s, dict) else s.end, 3),
                    "text": (s["text"] if isinstance(s, dict) else s.text).strip()
                } for i, s in enumerate(resp.segments)]
                
                words = []
                if hasattr(resp, 'words') and resp.words:
                    words = [{
                        "word": w["word"] if isinstance(w, dict) else w.word,
                        "start": w["start"] if isinstance(w, dict) else w.start,
                        "end": w["end"] if isinstance(w, dict) else w.end
                    } for w in resp.words]
                
                with open(whisper_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segs, "words": words}, f, indent=2, ensure_ascii=False)
                self._save("s2", {"segments": len(segs), "words": len(words)})
                log.info(f"[2/12] ✓ {len(segs)} segments, {len(words)} word-level timestamps")
            
            with open(whisper_path, encoding="utf-8") as f:
                whisper_data = json.load(f)
            segments = whisper_data["segments"]
            whisper_words = whisper_data.get("words", [])
            log.info(f"[2/12] ✓ {len(segments)} segments loaded")
            
            # ━━ 3. DIRECTOR (Enhanced) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            director_path = os.path.join(self.work_dir, "director_plan.json")
            if not self._done("s3"):
                log.info(f"[3/12] 🎬 DIRECTOR analyzing content...")
                log.info(f"[3/12]    + Word-level pause analysis")
                log.info(f"[3/12]    + Audio energy analysis")
                dir_result = director.analyze(
                    segments, self.work_dir,
                    user_description=self.user_description,
                    whisper_words=whisper_words,
                    audio_path=audio_path
                )
                self._save("s3", {
                    "content_type": dir_result["content_type"],
                    "speakers": dir_result["real_speaker_count"]
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
            log.info(
                f"[3/12] ✓ Type={dir_result['content_type']} | "
                f"Speakers={dir_result['real_speaker_count']}"
            )
            
            # ━━ 4. PRE-PROCESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[4/12] 🔧 Pre-processing: merge micro-segments...")
            merged = preprocessor.merge_short_segments(dir_result["segments"])
            dir_result["segments"] = merged
            
            # ━━ 5. TRANSLATE (Context-Aware) ━━━━━━━━━━━━━━━━━━━━━━━
            raw_path = os.path.join(self.work_dir, "translated_raw.json")
            if not self._done("s5"):
                log.info(f"[5/12] 🌐 Translating → {self.target_lang} (context-aware)...")
                translator.translate(dir_result, self.work_dir, self.target_lang)
                self._save("s5", {"raw_path": raw_path})
            with open(raw_path, encoding="utf-8") as f:
                raw_script = json.load(f)
            
            # ━━ 5b. ROMANIZE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[5b/12] ✏️ Romanizing Devanagari → Roman script...")
            romanizer.romanize_segments(raw_script["segments"])
            
            # ━━ 6. DIALOGUE WRITER (Emotion-Aware) ━━━━━━━━━━━━━━━━━
            dubbed_path = os.path.join(self.work_dir, "dubbed_script.json")
            if not self._done("s6"):
                log.info(f"[6/12] ✍️ DIALOGUE WRITER polishing (emotion + character consistency)...")
                dialogue_writer.rewrite(
                    raw_script["segments"], self.work_dir, self.target_lang,
                    narrative_summary=dir_result.get("narrative_summary", ""),
                    mood_arc=dir_result.get("mood_arc")
                )
                self._save("s6", {"dubbed_path": dubbed_path})
            with open(dubbed_path, encoding="utf-8") as f:
                dubbed_script = json.load(f)
            
            # ━━ 6b. ROMANIZE (post-writer) ━━━━━━━━━━━━━━━━━━━━━━━━
            rom_fixed = romanizer.romanize_segments(dubbed_script["segments"])
            if rom_fixed:
                with open(dubbed_path, "w", encoding="utf-8") as f:
                    json.dump(dubbed_script, f, indent=2, ensure_ascii=False)
            
            # Expand merged segments
            expanded = preprocessor.expand_dubbed_to_subsegments(dubbed_script["segments"])
            
            # ━━ 6c. VALIDATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[6c/12] ✅ Validating quality...")
            validator.validate(expanded, auto_fix=True)
            
            # ━━ 7. SENTENCE MERGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            log.info(f"[7/12] 📝 Merging into sentence-level TTS units...")
            tts_segments = sentence_merger.merge_for_tts(expanded)
            
            # ━━ 8. VOICE MATCHING (Smart Catalog) ━━━━━━━━━━━━━━━━━━
            log.info(f"[8/12] 🎤 Smart voice matching...")
            voice_plan = dir_result.get("voice_plan", [{"voice_id": "NARRATOR", "gender": "male", "tone": "calm", "personality": "narrator"}])
            
            # Use voice catalog for intelligent matching
            voice_catalog_map = voice_catalog.match_voices(voice_plan, self.target_lang)
            
            # Also build Edge TTS cast_map for fallback
            from voice_caster import cast as edge_cast
            edge_cast_map = edge_cast(dir_result, self.target_lang)
            
            log.info(f"[8/12] ✓ {len(voice_catalog_map)} characters cast with regional voices")
            
            # ━━ 9. TTS GENERATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            tts_manifest_path = os.path.join(self.work_dir, "tts_manifest.json")
            if not self._done("s9"):
                log.info(f"[9/12] 🔊 Generating TTS ({len(tts_segments)} units)...")
                tts_r = tts_engine.generate(
                    tts_segments, edge_cast_map, self.work_dir, self.target_lang,
                    voice_catalog_map=voice_catalog_map,
                    use_fish_audio=self.use_fish_audio
                )
                self._save("s9", {
                    "clips": len(tts_r["manifest"]),
                    "fish_used": tts_r.get("fish_used", 0),
                    "edge_used": tts_r.get("edge_used", 0)
                })
            with open(tts_manifest_path) as f:
                tts_manifest = json.load(f)
            log.info(f"[9/12] ✓ {len(tts_manifest)} clips generated")
            
            # ━━ 10. ASSEMBLE (Professional Mix) ━━━━━━━━━━━━━━━━━━━━
            if not self._done("s10"):
                log.info(f"[10/12] 🎵 Assembling (professional mix)...")
                
                # Get total duration
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    capture_output=True, text=True
                )
                total_dur = float(r.stdout.strip())
                
                # Use Demucs-separated background if available
                if bg_audio_path and os.path.exists(bg_audio_path):
                    log.info(f"[10/12] Using Demucs-separated clean background ✓")
                else:
                    bg_audio_path = None
                    log.info(f"[10/12] Using original audio as background")
                
                asm_r = assembler.assemble(
                    tts_manifest, self.work_dir, video_path, total_dur,
                    preserve_bg=self.preserve_bg,
                    bg_audio_path=bg_audio_path
                )
                self._save("s10", asm_r)
            
            with open(self.sp) as f:
                final = json.load(f)
            asm_r = {k: v for k, v in final.get("s10", {}).items() if k != "done"}
            
            # ━━ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elapsed = round(time.time() - t0, 1)
            size_mb = asm_r.get("size_mb", 0)
            
            log.info("=" * 70)
            log.info(f"  ✅ DONE in {elapsed}s ({elapsed/60:.1f}min) | {size_mb}MB")
            log.info(f"  📹 Video:      {asm_r.get('video_path', '?')}")
            log.info(f"  📝 Subtitles:  {asm_r.get('srt_path', '?')}")
            log.info(f"  🌐 Language:   {self.target_lang}")
            log.info(f"  🎬 Content:    {dir_result.get('content_type', '?')} | {dir_result.get('real_speaker_count', '?')} speakers")
            log.info(f"  🎵 BG Audio:   {'Demucs separation' if bg_audio_path else 'smart ducking'}")
            
            s9_info = self.state.get("s9", {})
            if s9_info.get("fish_used", 0) > 0:
                log.info(f"  🐟 TTS:        Fish Audio ({s9_info['fish_used']} clips) + Edge ({s9_info.get('edge_used', 0)} clips)")
            else:
                log.info(f"  🔊 TTS:        Edge TTS ({s9_info.get('edge_used', 0)} clips)")
            
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
                "mood_arc": dir_result.get("mood_arc", []),
                "preserve_bg": self.preserve_bg,
                "tts_engine": "fish_audio" if s9_info.get("fish_used", 0) > 0 else "edge_tts",
                "demucs_used": bool(bg_audio_path),
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
        """Determine which stage failed for error reporting."""
        stages = ["s1", "s1b", "s2", "s3", "s5", "s6", "s9", "s10"]
        for s in stages:
            if not self._done(s):
                return s
        return "unknown"


def run_agent(url, target_lang="Hindi", source_lang="zh",
              user_description="", output_dir="/content/drive/MyDrive/DubbedVideos",
              preserve_bg=True, use_fish_audio=True, use_demucs=True):
    """
    Run the dubbing pipeline.
    
    Args:
        url: YouTube video URL
        target_lang: Target language (Hindi, Tamil, Telugu, Bengali, Nepali, English, etc.)
        source_lang: Source language code (zh, en, ja, ko, etc.)
        user_description: Optional description of the video content
        output_dir: Base directory for output files
        preserve_bg: Keep background music/ambient audio
        use_fish_audio: Try Fish Audio API for premium TTS
        use_demucs: Use Demucs to separate vocals from background
    
    Returns:
        dict with success status, paths, and metadata
    """
    d = DubberV5(
        url=url, target_lang=target_lang, source_lang=source_lang,
        user_description=user_description, output_dir=output_dir,
        preserve_bg=preserve_bg, use_fish_audio=use_fish_audio,
        use_demucs=use_demucs
    )
    return d.run()
