"""
voice.py

Speech-to-text and text-to-speech through the Gemini API.

Both calls share the chat model's API-key pool and failover rules:
a key that is out of quota or rejected is parked and the next key is
tried straight away, so voice never stalls on one spent key.

STT  - a Gemini Flash model listens to the recording and returns the
       transcript in the speaker's own script plus an English
       translation. The English goes through routing and retrieval,
       because the router keywords and the knowledge base are English.
TTS  - a Gemini TTS model speaks the answer. It returns raw 16-bit PCM,
       which is wrapped in a WAV header so any browser can play it.
"""

import io
import json
import os
import re
import wave
from typing import Callable, Dict, Optional, Tuple, TypeVar

from google import genai
from google.genai import types

from app.llm.api_keys import key_pool
from app.llm.gemini import (
    LLM_MODEL,
    REQUEST_TIMEOUT,
    _classify,
    _is_daily_quota,
    _retry_after,
    _seconds_until_daily_reset,
)
from app.utils.errors import LLMUnavailableError, QuotaExhaustedError
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

STT_MODEL = os.getenv("STT_MODEL") or LLM_MODEL
TTS_MODEL = os.getenv("TTS_MODEL") or "gemini-3.8-flash-tts"
TTS_VOICE = os.getenv("TTS_VOICE") or "Kore"

# Spoken replies are asked to be short, but cap what is sent to TTS
# anyway - audio output is billed per second, and it dominates cost.
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "700"))

# Languages offered in the UI. The code is a BCP-47 tag the browser
# also understands, for the speechSynthesis fallback.
LANGUAGES: Dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
}

STT_PROMPT = """You are transcribing a short voice question from a \
student or parent asking a college help assistant.

The speaker is expected to speak {language}, possibly mixed with \
English words (course names, "B.Com", "hostel", etc.).

Return JSON with exactly these keys:
- "transcript": the exact words spoken, written in the script of the \
language spoken (e.g. Tamil script for Tamil). Keep English words in \
English.
- "english": a faithful English translation of the question. If the \
speaker spoke English, repeat the transcript.

If there is no intelligible speech, return empty strings for both."""

T = TypeVar("T")

_clients: Dict[str, genai.Client] = {}


def _client(label: str, api_key: str) -> genai.Client:

    if label not in _clients:
        _clients[label] = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(REQUEST_TIMEOUT * 1000)
            ),
        )

    return _clients[label]


def _with_failover(call: Callable[[genai.Client], T], what: str) -> T:
    """
    Run `call` against the pool's keys until one succeeds.

    Mirrors RotatingGeminiChat.invoke: only quota and credential
    errors move to the next key; anything else is a real problem
    with the request and is raised immediately.
    """

    if key_pool.size == 0:
        raise LLMUnavailableError()

    attempted = []
    last_error: Optional[Exception] = None
    daily = False

    for _ in range(key_pool.size):

        key = key_pool.current()

        if key is None or key.label in attempted:
            break

        attempted.append(key.label)

        try:

            result = call(_client(key.label, key.value))

            key_pool.report_success(key)

            return result

        except Exception as error:

            reason = _classify(error)

            if reason is None:
                raise

            last_error = error

            logger.warning(
                f"{what}: key '{key.label}' hit a {reason} error; "
                f"failing over to the next key."
            )

            if reason == "quota" and _is_daily_quota(error):
                daily = True
                cooldown = _seconds_until_daily_reset()
            elif reason == "quota":
                cooldown = _retry_after(error)
            else:
                cooldown = None

            key_pool.penalise(key, retry_after=cooldown)

    if last_error is None:
        raise LLMUnavailableError()

    raise QuotaExhaustedError(
        retry_after=(
            _seconds_until_daily_reset()
            if daily else _retry_after(last_error)
        ),
        daily=daily,
    ) from last_error


# --------------------------------------------------
# Speech-to-text
# --------------------------------------------------

def transcribe(
    audio: bytes,
    mime_type: str,
    language: str,
) -> Tuple[str, str]:
    """
    Transcribe a recording.

    Returns (transcript_in_spoken_script, english_translation).
    Both are empty when nothing intelligible was said.
    """

    language_name = LANGUAGES.get(language, "English")

    def call(client: genai.Client):
        return client.models.generate_content(
            model=STT_MODEL,
            contents=[
                types.Part.from_bytes(data=audio, mime_type=mime_type),
                STT_PROMPT.format(language=language_name),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    result = _with_failover(call, "STT")

    try:
        data = json.loads(result.text or "{}")
    except json.JSONDecodeError:
        logger.warning(f"STT returned non-JSON output: {result.text!r}")
        data = {}

    transcript = str(data.get("transcript") or "").strip()
    english = str(data.get("english") or "").strip() or transcript

    logger.info(
        f"STT ({language_name}): {transcript!r} -> {english!r}"
    )

    return transcript, english


# --------------------------------------------------
# Text-to-speech
# --------------------------------------------------

MARKDOWN_PATTERNS = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),   # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),    # links
    (re.compile(r"https?://\S+"), ""),               # bare URLs
    (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1"),      # code
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),     # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),          # bullets
    (re.compile(r"^\s*\|?[-:\s|]+\|?\s*$", re.M), ""),  # table rules
    (re.compile(r"^[ \t]*\||\|[ \t]*$", re.M), ""),   # table row edges
    (re.compile(r"[ \t]*\|[ \t]*"), ", "),            # table cells
    (re.compile(r"[*_~>]+"), ""),                     # emphasis, quotes
    (re.compile(r"\s*\n+\s*"), ". "),                 # line breaks -> pauses
    (re.compile(r"([.!?:;,।])\s*\.(?=\s|$)"), r"\1"),   # no doubled stops
    (re.compile(r"\s+"), " "),
]


def speakable(text: str) -> str:
    """
    Strip markdown so TTS doesn't read symbols aloud, and cap length.
    """

    for pattern, replacement in MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)

    text = text.strip()

    if len(text) > TTS_MAX_CHARS:
        cut = text[:TTS_MAX_CHARS]
        # End on a sentence boundary if there is one reasonably close.
        boundary = max(cut.rfind(". "), cut.rfind("। "), cut.rfind("? "))
        text = cut[: boundary + 1] if boundary > TTS_MAX_CHARS // 2 else cut

    return text


def _pcm_rate(mime_type: str) -> int:

    match = re.search(r"rate=(\d+)", mime_type or "")

    return int(match.group(1)) if match else 24000


def _to_wav(pcm: bytes, rate: int) -> bytes:

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(rate)
        wav.writeframes(pcm)

    return buffer.getvalue()


def synthesize(text: str) -> Optional[bytes]:
    """
    Speak `text`. Returns WAV bytes, or None if there is nothing to say.
    """

    text = speakable(text)

    if not text:
        return None

    def call(client: genai.Client):
        return client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=TTS_VOICE
                        )
                    )
                ),
            ),
        )

    result = _with_failover(call, "TTS")

    for candidate in result.candidates or []:
        for part in (candidate.content.parts if candidate.content else []) or []:
            blob = part.inline_data
            if blob and blob.data:
                mime = blob.mime_type or ""
                if "wav" in mime:
                    return blob.data
                return _to_wav(blob.data, _pcm_rate(mime))

    logger.warning("TTS returned no audio.")

    return None
