#!/usr/bin/env python3
"""Rule-based harmony labeling and tokenization for Pianalysis."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from closed_loop_v1 import TOKEN
from token_protocol_v1 import (
    LEGACY_VERSION, TYPED_VERSION, CHORD_BASE, BASS_BASE, BASS_NC, CONF_BASE,
    canonical_version, event_token_map,
)


ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

QUALITY_INTERVALS = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "dim": {0, 3, 6},
    "aug": {0, 4, 8},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "m7b5": {0, 3, 6, 10},
}

QUALITY_ORDER = list(QUALITY_INTERVALS)
CHORD_TOKEN_BASE = 900
BASS_TOKEN_BASE = 1100
CONF_TOKEN_BASE = 1220
CONF_BUCKETS = 10


def chord_token(root: int | None, quality: str | None) -> int:
    if root is None or quality is None:
        return TOKEN["CHORD_NC"]
    return CHORD_TOKEN_BASE + (root % 12) * len(QUALITY_ORDER) + QUALITY_ORDER.index(quality)


def bass_token(bass_pitch: int | None) -> int:
    if bass_pitch is None:
        return BASS_TOKEN_BASE
    return BASS_TOKEN_BASE + max(0, min(127, int(bass_pitch)))


def confidence_token(confidence: float) -> int:
    bucket = int(round(max(0.0, min(1.0, confidence)) * (CONF_BUCKETS - 1)))
    return CONF_TOKEN_BASE + bucket


def chord_name(root: int | None, quality: str | None) -> str:
    if root is None or quality is None:
        return "NC"
    return f"{ROOT_NAMES[root % 12]}:{quality}"


def token_maps(tokenizer_version: str = LEGACY_VERSION) -> dict[str, int]:
    if canonical_version(tokenizer_version) != LEGACY_VERSION:
        mapping = event_token_map(tokenizer_version)
        for root in range(12):
            for index, quality in enumerate(QUALITY_ORDER):
                mapping[f"CHORD_{ROOT_NAMES[root]}:{quality}"] = CHORD_BASE + root * len(QUALITY_ORDER) + index
        mapping.update({f"BASS_{pitch}": BASS_BASE + pitch for pitch in range(128)})
        mapping["BASS_NC"] = BASS_NC
        mapping.update({f"CONF_{bucket}": CONF_BASE + bucket for bucket in range(CONF_BUCKETS)})
        return mapping
    mapping = dict(TOKEN)
    for root in range(12):
        for quality in QUALITY_ORDER:
            mapping[f"CHORD_{ROOT_NAMES[root]}:{quality}"] = chord_token(root, quality)
    for pitch in range(128):
        mapping[f"BASS_{pitch}"] = bass_token(pitch)
    mapping["BASS_NC"] = bass_token(None)
    for bucket in range(CONF_BUCKETS):
        mapping[f"CONF_{bucket}"] = CONF_TOKEN_BASE + bucket
    return mapping


def notes_in_span(notes: list[dict[str, Any]], start: float, end: float, role: str | None = None) -> list[dict[str, Any]]:
    return [
        note
        for note in notes
        if (role is None or note.get("role") == role)
        and float(note["start"]) < end
        and float(note["end"]) > start
    ]


def overlap_weight(note: dict[str, Any], start: float, end: float) -> float:
    overlap = max(0.0, min(float(note["end"]), end) - max(float(note["start"]), start))
    duration = max(0.001, float(note["end"]) - float(note["start"]))
    velocity = max(1, min(127, int(note.get("velocity", 80)))) / 127.0
    return (overlap / duration) * overlap * (0.5 + velocity)


def pitch_class_weights(notes: list[dict[str, Any]], start: float, end: float) -> Counter[int]:
    weights: Counter[int] = Counter()
    for note in notes:
        weights[int(note["pitch"]) % 12] += overlap_weight(note, start, end)
    return weights


def bass_pitch(notes: list[dict[str, Any]]) -> int | None:
    if not notes:
        return None
    return min(int(note["pitch"]) for note in notes)


def score_candidate(
    root: int,
    quality: str,
    accomp_weights: Counter[int],
    melody_weights: Counter[int],
    bass: int | None,
) -> float:
    intervals = QUALITY_INTERVALS[quality]
    chord_pcs = {(root + interval) % 12 for interval in intervals}
    total_accomp = sum(accomp_weights.values())
    total_melody = sum(melody_weights.values())
    if total_accomp <= 0:
        return -999.0

    chord_weight = sum(weight for pc, weight in accomp_weights.items() if pc in chord_pcs)
    outside_weight = sum(weight for pc, weight in accomp_weights.items() if pc not in chord_pcs)
    melody_fit = sum(weight for pc, weight in melody_weights.items() if pc in chord_pcs)
    root_weight = accomp_weights[root]
    third_or_fifth = sum(accomp_weights[(root + interval) % 12] for interval in (3, 4, 7))

    score = 2.4 * (chord_weight / total_accomp)
    score -= 1.3 * (outside_weight / total_accomp)
    score += 0.45 * (melody_fit / total_melody) if total_melody > 0 else 0.0
    score += 0.30 * (root_weight / total_accomp)
    score += 0.20 * (third_or_fifth / total_accomp)

    if bass is not None:
        bass_pc = bass % 12
        if bass_pc == root:
            score += 0.40
        elif bass_pc in chord_pcs:
            score += 0.12
        else:
            score -= 0.30

    if quality in {"maj7", "min7", "7", "m7b5"}:
        score -= 0.05
    return score


def confidence_from_scores(best_score: float, second_score: float, coverage: float, note_count: int) -> float:
    margin = max(0.0, best_score - second_score)
    density = min(1.0, note_count / 5.0)
    confidence = 0.45 * coverage + 0.35 * min(1.0, margin / 0.85) + 0.20 * density
    return max(0.0, min(1.0, confidence))


def label_span(
    notes: list[dict[str, Any]],
    start: float,
    end: float,
    min_confidence: float,
) -> dict[str, Any]:
    accomp_notes = notes_in_span(notes, start, end, "accompaniment")
    melody_notes = notes_in_span(notes, start, end, "melody")
    if not accomp_notes:
        return {
            "start": start,
            "end": end,
            "chord": "NC",
            "root": None,
            "quality": None,
            "bass": None,
            "confidence": 0.0,
            "note_count": 0,
            "coverage": 0.0,
        }

    accomp_weights = pitch_class_weights(accomp_notes, start, end)
    melody_weights = pitch_class_weights(melody_notes, start, end)
    bass = bass_pitch(accomp_notes)
    scored: list[tuple[float, int, str]] = []
    for root in range(12):
        for quality in QUALITY_ORDER:
            scored.append((score_candidate(root, quality, accomp_weights, melody_weights, bass), root, quality))
    scored.sort(reverse=True, key=lambda item: item[0])
    best_score, root, quality = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else best_score
    chord_pcs = {(root + interval) % 12 for interval in QUALITY_INTERVALS[quality]}
    total_weight = sum(accomp_weights.values()) or 1.0
    coverage = sum(weight for pc, weight in accomp_weights.items() if pc in chord_pcs) / total_weight
    confidence = confidence_from_scores(best_score, second_score, coverage, len(accomp_notes))

    if confidence < min_confidence:
        root = None
        quality = None

    return {
        "start": start,
        "end": end,
        "chord": chord_name(root, quality),
        "root": root,
        "quality": quality,
        "bass": bass,
        "confidence": round(confidence, 4),
        "note_count": len(accomp_notes),
        "coverage": round(coverage, 4),
    }


def label_piece_harmony(
    piece: dict[str, Any],
    segment_seconds: float,
    min_confidence: float,
) -> list[dict[str, Any]]:
    duration = float(piece.get("duration", 0.0))
    if duration <= 0:
        return []
    segment_count = max(1, int(math.ceil(duration / segment_seconds)))
    labels: list[dict[str, Any]] = []
    for index in range(segment_count):
        start = index * segment_seconds
        end = min(duration, start + segment_seconds)
        label = label_span(piece["notes"], start, end, min_confidence)
        label["index"] = index
        labels.append(label)
    return labels


def harmony_tokens_for_labels(labels: list[dict[str, Any]], tokenizer_version: str = LEGACY_VERSION) -> list[int]:
    if canonical_version(tokenizer_version) != LEGACY_VERSION:
        mapping = token_maps(tokenizer_version)
        tokens = []
        for label in labels:
            root, quality = label.get("root"), label.get("quality")
            chord = "CHORD_NC" if root is None or quality is None else f"CHORD_{ROOT_NAMES[root % 12]}:{quality}"
            bass = label.get("bass")
            bass_name = "BASS_NC" if bass is None else f"BASS_{max(0, min(127, int(bass)))}"
            bucket = confidence_token(float(label.get("confidence", 0.0))) - CONF_TOKEN_BASE
            tokens.extend([TOKEN["BAR"], mapping[chord], mapping[bass_name], mapping[f"CONF_{bucket}"]])
        return tokens
    tokens: list[int] = []
    for label in labels:
        tokens.append(TOKEN["BAR"])
        root = label.get("root")
        quality = label.get("quality")
        tokens.append(chord_token(root, quality))
        tokens.append(bass_token(label.get("bass")))
        tokens.append(confidence_token(float(label.get("confidence", 0.0))))
    return tokens


def slice_harmony_labels(
    labels: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    sliced: list[dict[str, Any]] = []
    for label in labels:
        label_start = float(label["start"])
        label_end = float(label["end"])
        if label_start < end and label_end > start:
            item = dict(label)
            item["start"] = max(0.0, label_start - start)
            item["end"] = min(end - start, label_end - start)
            sliced.append(item)
    return sliced
