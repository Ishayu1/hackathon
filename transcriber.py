"""Fast audio transcription plus military-message triage.

This module owns speech-to-text, message segment classification, and severity
scoring. Deepfake and lie-detection models can be merged later through the
``external_signals`` argument without changing this pipeline's public shape.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


class SegmentCategory(str, Enum):
    ADMINISTRATIVE = "administrative"
    COMMAND = "command"
    INTELLIGENCE = "intelligence"
    LOGISTICS = "logistics"
    MEDICAL = "medical"
    EMERGENCY = "emergency"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str
    category: SegmentCategory
    severity: SeverityLevel
    severity_score: int
    matched_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionResult:
    audio_path: str
    transcript: str
    category: SegmentCategory
    severity: SeverityLevel
    severity_score: int
    chunks: list[TranscriptChunk]
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    processing_seconds: float
    real_time_factor: float | None
    external_signals: dict[str, Any] = field(default_factory=dict)


CATEGORY_KEYWORDS: dict[SegmentCategory, tuple[str, ...]] = {
    SegmentCategory.EMERGENCY: (
        "mayday",
        "sos",
        "need help",
        "please help",
        "send help",
        "request help",
        "requesting help",
        "urgent",
        "emergency",
        "casualty",
        "casualties",
        "troops in contact",
        "tic",
        "taking fire",
        "under fire",
        "ambush",
        "medevac",
        "casevac",
        "medical evacuation",
        "evacuation requested",
        "contact front",
        "contact rear",
        "contact left",
        "contact right",
        "get out of here",
        "run away",
        "get back",
        "under cover",
        "take cover",
        "chemical alert",
        "chemical",
        "biological",
        "nuclear",
    ),
    SegmentCategory.COMMAND: (
        "execute",
        "advance",
        "hold position",
        "withdraw",
        "retreat",
        "engage",
        "fire mission",
        "cease fire",
        "rally point",
        "objective",
        "orders",
        "command",
        "move to",
        "secure",
        "defend",
        "open fire",
        "take cover",
        "get back",
    ),
    SegmentCategory.INTELLIGENCE: (
        "enemy",
        "hostile",
        "insurgent",
        "sighting",
        "recon",
        "surveillance",
        "intel",
        "unknown vehicle",
        "movement",
        "coordinates",
        "grid",
        "drone",
        "uav",
        "ied",
        "mine",
    ),
    SegmentCategory.LOGISTICS: (
        "ammo",
        "ammunition",
        "fuel",
        "water",
        "rations",
        "resupply",
        "convoy",
        "maintenance",
        "transport",
        "equipment",
        "supply",
        "repair",
    ),
    SegmentCategory.MEDICAL: (
        "wounded",
        "injured",
        "bleeding",
        "medic",
        "medical",
        "triage",
        "evacuation",
        "evac",
        "litter",
        "aid station",
    ),
    SegmentCategory.AUTHENTICATION: (
        "authenticate",
        "challenge",
        "password",
        "callsign",
        "call sign",
        "codeword",
        "code word",
        "verification",
        "confirm identity",
    ),
    SegmentCategory.ADMINISTRATIVE: (
        "status report",
        "sitrep",
        "routine",
        "briefing",
        "roll call",
        "personnel",
        "schedule",
        "training",
        "formation",
        "accountability",
        "admin",
    ),
}


SEVERITY_KEYWORDS: dict[SeverityLevel, tuple[str, ...]] = {
    SeverityLevel.CRITICAL: (
        "mayday",
        "sos",
        "troops in contact",
        "taking fire",
        "under fire",
        "casualties",
        "kia",
        "wounded",
        "medevac",
        "casevac",
        "ambush",
        "ied",
        "chemical",
        "biological",
        "nuclear",
        "breach",
        "overrun",
        "get out of here",
        "run away",
        "under cover",
        "take cover",
    ),
    SeverityLevel.HIGH: (
        "need help",
        "please help",
        "send help",
        "request help",
        "requesting help",
        "urgent",
        "hostile",
        "enemy",
        "engage",
        "fire mission",
        "withdraw",
        "lost comms",
        "compromised",
        "mine",
        "drone",
        "uav",
        "casualty",
        "evacuation requested",
        "open fire",
        "cease fire",
        "get back",
        "move to",
    ),
    SeverityLevel.MEDIUM: (
        "movement",
        "sighting",
        "resupply",
        "ammo",
        "fuel",
        "medical",
        "execute",
        "secure",
        "hold position",
        "checkpoint",
        "delay",
    ),
    SeverityLevel.LOW: (
        "routine",
        "training",
        "briefing",
        "roll call",
        "schedule",
        "status report",
        "sitrep",
    ),
}


ACTION_VERBS = (
    "execute",
    "engage",
    "advance",
    "withdraw",
    "secure",
    "defend",
    "evacuate",
)

ASSISTANCE_OFFER_PATTERN = re.compile(
    r"\b(?:i|we)\s*(?:can|could|will|'ll|am able to|are able to|able to)\s+(?:go\s+)?help\b"
    r"|\b(?:i|we)\s+can\s+assist\b"
)

BENIGN_HELP_CONTEXT = (
    "box",
    "boxes",
    "homework",
    "furniture",
    "groceries",
    "bag",
    "bags",
    "carry",
    "lift",
)

BENIGN_TERM_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "get back": (
        re.compile(r"\bget\s+back\s+to\s+(?:me|you|us|them|him|her)\b"),
    ),
    "open fire": (
        re.compile(r"\bopen\s+fire\s+(?:hydrant|place|pit|door|station)\b"),
    ),
    "under cover": (
        re.compile(r"\bunder\s+cover\s+of\s+(?:darkness|night)\b"),
    ),
    "please help": (
        re.compile(r"\bplease\s+help\s+(?:me\s+)?(?:carry|lift|move)\b"),
    ),
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    normalized = normalize_text(text)
    found: list[str] = []
    for term in terms:
        if not term or not re.search(rf"\b{re.escape(term)}\b", normalized):
            continue
        if any(pattern.search(normalized) for pattern in BENIGN_TERM_PATTERNS.get(term, ())):
            continue
        found.append(term)
    return found


def parse_custom_keywords(custom_keywords: Iterable[str] | str | None) -> list[str]:
    if custom_keywords is None:
        return []
    if isinstance(custom_keywords, str):
        items = re.split(r"[,;\n]", custom_keywords)
    else:
        items = custom_keywords
    return [normalize_text(item) for item in items if normalize_text(item)]


def has_benign_help_context(text: str) -> bool:
    return bool(find_terms(text, BENIGN_HELP_CONTEXT))


def classify_segment(text: str) -> tuple[SegmentCategory, list[str]]:
    scores: dict[SegmentCategory, int] = {}
    matches: dict[SegmentCategory, list[str]] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        found = find_terms(text, keywords)
        if found:
            matches[category] = found
            scores[category] = len(found)

    normalized = normalize_text(text)
    if "status report" in matches.get(SegmentCategory.ADMINISTRATIVE, []):
        scores[SegmentCategory.ADMINISTRATIVE] = scores.get(SegmentCategory.ADMINISTRATIVE, 0) + 1
    if any(re.search(rf"\b{verb}\b", normalized) for verb in ACTION_VERBS):
        scores[SegmentCategory.COMMAND] = scores.get(SegmentCategory.COMMAND, 0) + 1

    if not scores:
        return SegmentCategory.UNKNOWN, []

    category = max(scores, key=lambda item: scores[item])
    return category, matches.get(category, [])


def score_severity(
    text: str,
    external_signals: dict[str, Any] | None = None,
    custom_keywords: Iterable[str] | str | None = None,
    custom_keyword_weight: int = 20,
) -> tuple[SeverityLevel, int, list[str]]:
    score = 0
    matched_terms: list[str] = []
    matched_levels: set[SeverityLevel] = set()

    weights = {
        SeverityLevel.CRITICAL: 45,
        SeverityLevel.HIGH: 30,
        SeverityLevel.MEDIUM: 15,
        SeverityLevel.LOW: 5,
    }
    for level, keywords in SEVERITY_KEYWORDS.items():
        found = find_terms(text, keywords)
        if found:
            matched_levels.add(level)
            matched_terms.extend(found)
            score += weights[level] * len(found)

    normalized = normalize_text(text)
    distress_request = re.search(r"\b(?:need|request(?:ing)?|send)\s+help\b|\bplease\s+help\b", normalized)
    if distress_request and not has_benign_help_context(normalized):
        score += 25
        matched_terms.append("distress request")
    elif ASSISTANCE_OFFER_PATTERN.search(normalized) and not has_benign_help_context(normalized):
        score += 20
        matched_terms.append("assistance offer")
    if re.search(r"\bnow\b|\bimmediate(?:ly)?\b|\basap\b", normalized):
        score += 15
        matched_terms.append("immediacy")
    if re.search(r"\bmultiple\b|\bmass\b|\bheavy\b", normalized):
        score += 10
        matched_terms.append("scale")
    weapon_commands = find_terms(text, ("open fire", "cease fire"))
    if weapon_commands:
        score += 15
        matched_terms.append("weapons command")

    custom_matches = find_terms(text, parse_custom_keywords(custom_keywords))
    if custom_matches:
        score += custom_keyword_weight * len(custom_matches)
        matched_terms.extend(f"custom: {match}" for match in custom_matches)

    for signal, value in (external_signals or {}).items():
        if signal in {"deepfake_probability", "lie_probability", "duress_probability"} and isinstance(value, (int, float)):
            if value >= 0.85:
                score += 20
            elif value >= 0.65:
                score += 10
            if signal == "duress_probability" and value >= 0.5:
                matched_terms.append("acoustic duress signal")

    score = min(score, 100)
    if matched_levels and matched_levels <= {SeverityLevel.LOW}:
        return SeverityLevel.LOW, min(score, 14), matched_terms
    if score >= 75:
        return SeverityLevel.CRITICAL, score, matched_terms
    if score >= 45:
        return SeverityLevel.HIGH, score, matched_terms
    if score >= 15:
        return SeverityLevel.MEDIUM, score, matched_terms
    return SeverityLevel.LOW, score, matched_terms


def classify_message(
    text: str,
    external_signals: dict[str, Any] | None = None,
    custom_keywords: Iterable[str] | str | None = None,
    custom_keyword_weight: int = 20,
) -> tuple[SegmentCategory, SeverityLevel, int, list[str]]:
    category, category_terms = classify_segment(text)
    severity, severity_score, severity_terms = score_severity(
        text,
        external_signals,
        custom_keywords=custom_keywords,
        custom_keyword_weight=custom_keyword_weight,
    )
    terms = sorted(set(category_terms + severity_terms))
    return category, severity, severity_score, terms


@lru_cache(maxsize=4)
def load_whisper_model(model_size: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install it with "
            "`pip install faster-whisper` and make sure ffmpeg is available."
        ) from exc

    return WhisperModel(model_size, device=device, compute_type=compute_type)


class FastMilitaryTranscriber:
    """Reusable transcriber that keeps the Whisper model warm in memory."""

    def __init__(
        self,
        *,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        language: str | None = "en",
        vad_filter: bool = False,
        custom_keywords: Iterable[str] | str | None = None,
        custom_keyword_weight: int = 20,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter
        self.custom_keywords = parse_custom_keywords(custom_keywords)
        self.custom_keyword_weight = custom_keyword_weight
        self.model = load_whisper_model(model_size, device, compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        external_signals: dict[str, Any] | None = None,
        custom_keywords: Iterable[str] | str | None = None,
    ) -> TranscriptionResult:
        return _transcribe_with_model(
            self.model,
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
            external_signals=external_signals,
            custom_keywords=self.custom_keywords if custom_keywords is None else custom_keywords,
            custom_keyword_weight=self.custom_keyword_weight,
        )

    def stream(
        self,
        audio_path: str | Path,
        *,
        external_signals: dict[str, Any] | None = None,
        custom_keywords: Iterable[str] | str | None = None,
    ) -> Iterable[dict[str, Any]]:
        return _stream_with_model(
            self.model,
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=self.vad_filter,
            external_signals=external_signals,
            custom_keywords=self.custom_keywords if custom_keywords is None else custom_keywords,
            custom_keyword_weight=self.custom_keyword_weight,
        )


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_size: str = "tiny.en",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 1,
    language: str | None = "en",
    vad_filter: bool = False,
    external_signals: dict[str, Any] | None = None,
    custom_keywords: Iterable[str] | str | None = None,
    custom_keyword_weight: int = 20,
) -> TranscriptionResult:
    model = load_whisper_model(model_size, device, compute_type)
    return _transcribe_with_model(
        model,
        audio_path,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        external_signals=external_signals,
        custom_keywords=custom_keywords,
        custom_keyword_weight=custom_keyword_weight,
    )


def _transcribe_with_model(
    model: Any,
    audio_path: str | Path,
    *,
    beam_size: int,
    language: str | None,
    vad_filter: bool,
    external_signals: dict[str, Any] | None = None,
    custom_keywords: Iterable[str] | str | None = None,
    custom_keyword_weight: int = 20,
) -> TranscriptionResult:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    started = time.perf_counter()
    segments, info = model.transcribe(
        str(path),
        beam_size=beam_size,
        vad_filter=vad_filter,
        language=language,
        condition_on_previous_text=False,
    )

    chunks: list[TranscriptChunk] = []
    transcript_parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        transcript_parts.append(text)
        category, severity, severity_score, terms = classify_message(
            text,
            external_signals,
            custom_keywords=custom_keywords,
            custom_keyword_weight=custom_keyword_weight,
        )
        chunks.append(
            TranscriptChunk(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
                category=category,
                severity=severity,
                severity_score=severity_score,
                matched_terms=terms,
            )
        )

    transcript = " ".join(transcript_parts)
    category, severity, severity_score, _ = classify_message(
        transcript,
        external_signals,
        custom_keywords=custom_keywords,
        custom_keyword_weight=custom_keyword_weight,
    )
    processing_seconds = time.perf_counter() - started
    duration = getattr(info, "duration", None)
    real_time_factor = processing_seconds / duration if duration else None

    return TranscriptionResult(
        audio_path=str(path),
        transcript=transcript,
        category=category,
        severity=severity,
        severity_score=severity_score,
        chunks=chunks,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration_seconds=duration,
        processing_seconds=processing_seconds,
        real_time_factor=real_time_factor,
        external_signals=external_signals or {},
    )


def transcribe_then_emit(audio_path: str | Path, **kwargs: Any) -> Iterable[dict[str, Any]]:
    """Yield classification chunks during transcription, then the final result."""

    model = load_whisper_model(
        kwargs.get("model_size", "tiny.en"),
        kwargs.get("device", "cpu"),
        kwargs.get("compute_type", "int8"),
    )
    return _stream_with_model(
        model,
        audio_path,
        beam_size=kwargs.get("beam_size", 1),
        language=kwargs.get("language", "en"),
        vad_filter=kwargs.get("vad_filter", False),
        external_signals=kwargs.get("external_signals"),
        custom_keywords=kwargs.get("custom_keywords"),
        custom_keyword_weight=kwargs.get("custom_keyword_weight", 20),
    )


def _stream_with_model(
    model: Any,
    audio_path: str | Path,
    *,
    beam_size: int,
    language: str | None,
    vad_filter: bool,
    external_signals: dict[str, Any] | None = None,
    custom_keywords: Iterable[str] | str | None = None,
    custom_keyword_weight: int = 20,
) -> Iterable[dict[str, Any]]:
    path = Path(audio_path)
    started = time.perf_counter()
    segments, info = model.transcribe(
        str(path),
        beam_size=beam_size,
        vad_filter=vad_filter,
        language=language,
        condition_on_previous_text=False,
    )

    transcript_parts: list[str] = []
    chunks: list[TranscriptChunk] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        transcript_parts.append(text)
        category, severity, severity_score, terms = classify_message(
            text,
            external_signals,
            custom_keywords=custom_keywords,
            custom_keyword_weight=custom_keyword_weight,
        )
        chunk = TranscriptChunk(
            start=float(segment.start),
            end=float(segment.end),
            text=text,
            category=category,
            severity=severity,
            severity_score=severity_score,
            matched_terms=terms,
        )
        chunks.append(chunk)
        yield {"type": "segment", "chunk": asdict(chunk)}

    transcript = " ".join(transcript_parts)
    category, severity, severity_score, _ = classify_message(
        transcript,
        external_signals,
        custom_keywords=custom_keywords,
        custom_keyword_weight=custom_keyword_weight,
    )
    processing_seconds = time.perf_counter() - started
    duration = getattr(info, "duration", None)
    yield {
        "type": "final",
        "result": asdict(
            TranscriptionResult(
                audio_path=str(path),
                transcript=transcript,
                category=category,
                severity=severity,
                severity_score=severity_score,
                chunks=chunks,
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
                duration_seconds=duration,
                processing_seconds=processing_seconds,
                real_time_factor=processing_seconds / duration if duration else None,
                external_signals=external_signals or {},
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe and triage military audio messages.")
    parser.add_argument("audio_path", help="Path to an audio file supported by ffmpeg.")
    parser.add_argument("--model-size", default="tiny.en", help="faster-whisper model name.")
    parser.add_argument("--device", default="cpu", help="auto, cuda, cpu, or other CTranslate2 device.")
    parser.add_argument("--compute-type", default="int8", help="auto, float16, int8_float16, int8, etc.")
    parser.add_argument("--beam-size", type=int, default=1, help="Use 1 for speed; higher improves accuracy.")
    parser.add_argument("--language", default="en", help="Set to empty string for language detection.")
    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD for longer clips with silence/noise.")
    parser.add_argument("--stream", action="store_true", help="Emit per-segment JSON lines before final output.")
    parser.add_argument(
        "--custom-keywords",
        default="",
        help="Comma-, semicolon-, or newline-separated operator watch terms.",
    )
    parser.add_argument(
        "--custom-keyword-weight",
        type=int,
        default=20,
        help="Severity points added for each custom keyword match.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kwargs = {
        "model_size": args.model_size,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "language": args.language or None,
        "vad_filter": args.vad_filter,
        "custom_keywords": args.custom_keywords,
        "custom_keyword_weight": args.custom_keyword_weight,
    }

    if args.stream:
        for event in transcribe_then_emit(args.audio_path, **kwargs):
            print(json.dumps(event, default=str), flush=True)
        return

    result = transcribe_audio(args.audio_path, **kwargs)
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
