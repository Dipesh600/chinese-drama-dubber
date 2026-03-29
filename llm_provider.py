"""
LLM PROVIDER — Abstraction layer for LLM API calls.
Supports Groq (current) with easy extension to Ollama, Gemini, etc.

Usage:
    from llm_provider import LLM
    llm = LLM()
    result = llm.chat("What is 2+2?", model="llama-3.3-70b-versatile")
"""
import os, json, logging, time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("[LLM] groq not installed")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "fast": "llama-3.1-8b-instant",
}

OLLAMA_MODELS = {
    "primary": "llama3.3:70b",
    "scout": "llama3.2-vision:11b",
    "fast": "llama3.2:3b",
}

GEMINI_MODELS = {
    "primary": "gemini-1.5-pro",
    "fast": "gemini-1.5-flash",
}

PROVIDER_MODELS = {
    "groq": GROQ_MODELS,
    "ollama": OLLAMA_MODELS,
    "gemini": GEMINI_MODELS,
}

FALLBACK_CHAINS = {
    "groq": ["primary", "scout", "fast"],
    "ollama": ["primary", "fast"],
    "gemini": ["primary", "fast"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LLM:
    """
    Unified LLM provider with pluggable backends.
    Currently supports: Groq (default)
    Future: Ollama (local), Gemini (cloud)
    """

    def __init__(self, provider: str = "groq", api_key: str = None):
        """
        Initialize LLM provider.

        Args:
            provider: "groq" (default), "ollama", or "gemini"
            api_key: API key (defaults to env var)
        """
        self.provider = provider.lower()
        self.models = PROVIDER_MODELS.get(self.provider, GROQ_MODELS)
        self.fallback_chain = FALLBACK_CHAINS.get(self.provider, ["primary"])

        if self.provider == "groq":
            self._init_groq(api_key)
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "gemini":
            self._init_gemini(api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def _init_groq(self, api_key: str = None):
        """Initialize Groq client."""
        if not HAS_GROQ:
            raise RuntimeError("groq package not installed. Run: pip install groq")
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = Groq(api_key=key)
        logger.info(f"[LLM] Groq initialized ✓")

    def _init_ollama(self):
        """Initialize Ollama client."""
        if not HAS_HTTPX:
            raise RuntimeError("httpx required for Ollama. Run: pip install httpx")
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.client = None  # Ollama uses httpx directly
        logger.info(f"[LLM] Ollama initialized at {self.ollama_url} ✓")

    def _init_gemini(self, api_key: str = None):
        """Initialize Gemini client."""
        # Future: implement Gemini
        raise NotImplementedError("Gemini provider coming soon")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN CHAT METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        message: str,
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        json_response: bool = True,
        system_override: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a chat completion request.

        Args:
            prompt: System prompt
            message: User message
            model: Model name (auto-selects from chain if not provided)
            temperature: Randomness (0=deterministic, 1=creative)
            max_tokens: Max output tokens
            json_response: Expect JSON response
            system_override: Override system prompt

        Returns:
            Parsed JSON response or raw text
        """
        # Select model
        if model is None:
            model = self.models["primary"]

        # Build messages
        system = prompt if system_override is None else system_override
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]

        # Route to correct provider
        if self.provider == "groq":
            return self._chat_groq(model, messages, temperature, max_tokens, json_response)
        elif self.provider == "ollama":
            return self._chat_ollama(model, messages, temperature, max_tokens, json_response)

    def _chat_groq(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        json_response: bool,
    ) -> Optional[Dict[str, Any]]:
        """Execute chat via Groq API with automatic fallback."""
        max_retries = int(os.environ.get("MAX_RETRIES", "4"))

        for attempt in range(max_retries):
            try:
                response_format = {"type": "json_object"} if json_response else {"type": "text"}

                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                content = resp.choices[0].message.content

                if json_response:
                    return json.loads(content)
                return content

            except Exception as e:
                err = str(e)
                logger.warning(f"[LLM] Groq error (attempt {attempt+1}/{max_retries}): {err}")

                # Try fallback model
                if "429" in err or "rate" in err.lower() or "model" in err.lower():
                    next_model = self._get_next_model(model)
                    if next_model:
                        logger.info(f"[LLM] Falling back to {next_model}")
                        model = next_model
                        continue

                # Rate limit wait
                wait = min(2 ** attempt, 12)
                if attempt < max_retries - 1:
                    logger.info(f"[LLM] Retrying in {wait}s...")
                    time.sleep(wait)

        logger.error(f"[LLM] All {max_retries} attempts failed")
        return None

    def _chat_ollama(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        json_response: bool,
    ) -> Optional[Dict[str, Any]]:
        """Execute chat via Ollama API."""
        try:
            import httpx

            # Convert messages to Ollama format
            ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            resp = httpx.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=120.0,
            )
            resp.raise_for_status()

            result = resp.json()
            content = result.get("message", {}).get("content", "")

            if json_response:
                return json.loads(content)
            return content

        except Exception as e:
            logger.error(f"[LLM] Ollama error: {e}")
            return None

    def _get_next_model(self, current_model: str) -> Optional[str]:
        """Get next model in fallback chain."""
        chain = FALLBACK_CHAINS.get(self.provider, [])
        for i, m in enumerate(chain):
            if self.models.get(m) == current_model:
                # Return next in chain
                if i + 1 < len(chain):
                    return self.models.get(chain[i + 1])
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSCRIPTION (Whisper via Groq)
    # ─────────────────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            language: Source language code (e.g., "zh", "en", "ja")

        Returns:
            Dict with segments and words
        """
        if self.provider != "groq":
            raise NotImplementedError("Transcription only available via Groq")

        if not HAS_GROQ:
            raise RuntimeError("groq package required for transcription")

        try:
            with open(audio_path, "rb") as f:
                resp = self.client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    language=language if language != "auto" else None,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
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

            return {"segments": segs, "words": words}

        except Exception as e:
            logger.error(f"[LLM] Transcription failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_llm_instance = None

def get_llm(provider: str = "groq") -> LLM:
    """Get or create singleton LLM instance."""
    global _llm_instance
    if _llm_instance is None or _llm_instance.provider != provider:
        _llm_instance = LLM(provider=provider)
    return _llm_instance
