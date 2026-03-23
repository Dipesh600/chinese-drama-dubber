"""
ORCHESTRATOR v4.1 — 10-stage pipeline with all 6 fixes:
1. Romanizer (Devanagari → Roman)
2. Word count limits (prevents speed-up)
3. Intelligent audio ducking
4. Better voice differentiation (SON→SwaraNeural)
5. No narrator fillers
6. Quality validation before TTS
"""
import os, sys, json, logging, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import director, preprocessor, translator, dialogue_writer, sentence_merger
import voice_caster, tts_engine, assembler, romanizer, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

class DubberV4:
    def __init__(self, url, target_lang="Hindi", source_lang="zh",
                 user_description="", output_dir="/home/user/dub_v3_runs",
                 preserve_bg=True):
        self.url = url
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.user_description = user_description
        self.preserve_bg = preserve_bg
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.work_dir = os.path.join(output_dir, f"run_{ts}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.sp = os.path.join(self.work_dir, "state.json")
        self.state = {}

    def _done(self, k):
        return self.state.get(k, {}).get("done", False)
    def _save(self, k, v):
        self.state[k] = {"done": True, **v}
        with open(self.sp, "w") as f: json.dump(self.state, f, indent=2)

    def run(self):
        t0 = time.time()
        log.info("=" * 65)
        log.info(f"  DUBBER v4.1 — {self.target_lang} | All Fixes Applied")
        log.info(f"  URL: {self.url}")
        log.info(f"  BG Audio: {'Smart Ducking' if self.preserve_bg else 'OFF'}")
        log.info("=" * 65)

        try:
            video_path = os.path.join(self.work_dir, "video.mp4")
            audio_path = os.path.join(self.work_dir, "audio.mp3")

            # ── 1. DOWNLOAD ─────────────────────────────────────────
            if not self._done("s1"):
                log.info(f"[1/10] Downloading: {self.url}")
                import yt_dlp
                ydl_opts = {
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": video_path, "quiet": True, "no_warnings": True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([self.url])
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000",
                    "-ac", "1", "-b:a", "64k", audio_path
                ], capture_output=True, timeout=300)
                audio_mb = os.path.getsize(audio_path) / 1024 / 1024
                r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=noprint_wrappers=1:nokey=1",audio_path],
                    capture_output=True, text=True)
                dur = float(r.stdout.strip())
                self._save("s1", {"audio_mb": round(audio_mb,1), "duration": round(dur,1)})
                log.info(f"[1/10] ✓ Audio: {audio_mb:.1f}MB | {dur:.0f}s ({dur/60:.1f}min)")

            # ── 2. TRANSCRIBE ───────────────────────────────────────
            whisper_path = os.path.join(self.work_dir, "whisper.json")
            if not self._done("s2"):
                log.info(f"[2/10] Transcribing (Groq Whisper)...")
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
                    "start": round(s["start"] if isinstance(s,dict) else s.start, 3),
                    "end": round(s["end"] if isinstance(s,dict) else s.end, 3),
                    "text": (s["text"] if isinstance(s,dict) else s.text).strip()
                } for i, s in enumerate(resp.segments)]
                words = []
                if hasattr(resp, 'words') and resp.words:
                    words = [{"word": w["word"] if isinstance(w,dict) else w.word,
                              "start": w["start"] if isinstance(w,dict) else w.start,
                              "end": w["end"] if isinstance(w,dict) else w.end} for w in resp.words]
                with open(whisper_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segs, "words": words}, f, indent=2, ensure_ascii=False)
                self._save("s2", {"segments": len(segs), "words": len(words)})
                log.info(f"[2/10] ✓ {len(segs)} segments, {len(words)} words")
            with open(whisper_path, encoding="utf-8") as f:
                whisper_data = json.load(f)
            segments = whisper_data["segments"]
            log.info(f"[2/10] ✓ {len(segments)} segments loaded")

            # ── 3. DIRECTOR ─────────────────────────────────────────
            director_path = os.path.join(self.work_dir, "director_plan.json")
            if not self._done("s3"):
                log.info(f"[3/10] DIRECTOR analyzing content...")
                dir_result = director.analyze(segments, self.work_dir, self.user_description)
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
            log.info(f"[3/10] ✓ Type={dir_result['content_type']} | Speakers={dir_result['real_speaker_count']}")

            # ── 4. PRE-PROCESS ──────────────────────────────────────
            log.info(f"[4/10] Pre-processing: merge micro-segments...")
            merged = preprocessor.merge_short_segments(dir_result["segments"])
            dir_result["segments"] = merged

            # ── 5. TRANSLATE (Llama-4-Scout) ────────────────────────
            raw_path = os.path.join(self.work_dir, "translated_raw.json")
            if not self._done("s5"):
                log.info(f"[5/10] Translating → {self.target_lang} (Llama-4-Scout)...")
                translator.translate(dir_result, self.work_dir, self.target_lang)
                self._save("s5", {"raw_path": raw_path})
            with open(raw_path, encoding="utf-8") as f:
                raw_script = json.load(f)

            # ── 5b. ROMANIZE after translation ──────────────────────
            log.info(f"[5b] Romanizing Devanagari → Roman script...")
            romanizer.romanize_segments(raw_script["segments"])

            # ── 6. DIALOGUE WRITER (with word limits) ───────────────
            dubbed_path = os.path.join(self.work_dir, "dubbed_script.json")
            if not self._done("s6"):
                log.info(f"[6/10] DIALOGUE WRITER polishing (word limits active)...")
                dialogue_writer.rewrite(
                    raw_script["segments"], self.work_dir, self.target_lang,
                    narrative_summary=dir_result.get("narrative_summary", ""),
                    mood_arc=dir_result.get("mood_arc")
                )
                self._save("s6", {"dubbed_path": dubbed_path})
            with open(dubbed_path, encoding="utf-8") as f:
                dubbed_script = json.load(f)

            # ── 6b. ROMANIZE after dialogue writer too ──────────────
            rom_fixed = romanizer.romanize_segments(dubbed_script["segments"])
            if rom_fixed:
                with open(dubbed_path, "w", encoding="utf-8") as f:
                    json.dump(dubbed_script, f, indent=2, ensure_ascii=False)

            # Expand merged segments
            expanded = preprocessor.expand_dubbed_to_subsegments(dubbed_script["segments"])

            # ── 6c. VALIDATE before TTS ─────────────────────────────
            log.info(f"[6c] Validating quality...")
            validator.validate(expanded, auto_fix=True)

            # ── 7. SENTENCE MERGE ───────────────────────────────────
            log.info(f"[7/10] Merging into sentence-level TTS units...")
            tts_segments = sentence_merger.merge_for_tts(expanded)

            # ── 8. VOICE CAST ───────────────────────────────────────
            cast_map = voice_caster.cast(dir_result, self.target_lang)
            log.info(f"[8/10] Voice cast: {cast_map}")

            # ── 9. TTS ─────────────────────────────────────────────
            tts_manifest_path = os.path.join(self.work_dir, "tts_manifest.json")
            if not self._done("s9"):
                log.info(f"[9/10] Generating TTS ({len(tts_segments)} units)...")
                tts_r = tts_engine.generate(tts_segments, cast_map, self.work_dir, self.target_lang)
                self._save("s9", {"clips": len(tts_r["manifest"])})
            with open(tts_manifest_path) as f:
                tts_manifest = json.load(f)
            log.info(f"[9/10] ✓ {len(tts_manifest)} clips")

            # ── 10. ASSEMBLE (smart ducking) ────────────────────────
            if not self._done("s10"):
                log.info(f"[10/10] Assembling + {'smart ducking' if self.preserve_bg else 'no BG'}...")
                r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                    "-of","default=noprint_wrappers=1:nokey=1",audio_path],
                    capture_output=True, text=True)
                total_dur = float(r.stdout.strip())
                asm_r = assembler.assemble(
                    tts_manifest, self.work_dir, video_path, total_dur,
                    preserve_bg=self.preserve_bg
                )
                self._save("s10", asm_r)
            with open(self.sp) as f:
                final = json.load(f)
            asm_r = {k:v for k,v in final.get("s10",{}).items() if k!="done"}

            elapsed = round(time.time()-t0, 1)
            size_mb = asm_r.get("size_mb", 0)
            log.info("=" * 65)
            log.info(f"  ✅ DONE in {elapsed}s | {size_mb}MB | {self.target_lang}")
            log.info(f"  Video: {asm_r.get('video_path','?')}")
            log.info(f"  Subs:  {asm_r.get('srt_path','?')}")
            log.info(f"  BG:    {'smart ducking (30% gaps / 8% speech)' if self.preserve_bg else 'none'}")
            log.info("=" * 65)

            return {
                "success": True,
                "video_path": asm_r.get("video_path",""),
                "srt_path": asm_r.get("srt_path",""),
                "size_mb": size_mb, "processing_time": elapsed,
                "content_type": dir_result.get("content_type","?"),
                "real_speaker_count": dir_result.get("real_speaker_count",1),
                "cast": cast_map, "target_lang": self.target_lang,
                "work_dir": self.work_dir,
                "mood_arc": dir_result.get("mood_arc",[]),
                "preserve_bg": self.preserve_bg,
            }
        except Exception as e:
            import traceback
            log.error(f"Pipeline error: {e}"); traceback.print_exc()
            return {"success": False, "error": str(e), "work_dir": self.work_dir}

def run_agent(url, target_lang="Hindi", source_lang="zh",
              user_description="", output_dir="/home/user/dub_v3_runs",
              preserve_bg=True):
    d = DubberV4(url=url, target_lang=target_lang, source_lang=source_lang,
                 user_description=user_description, output_dir=output_dir,
                 preserve_bg=preserve_bg)
    return d.run()
