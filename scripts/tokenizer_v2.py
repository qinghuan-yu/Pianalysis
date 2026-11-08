"""TSD-style explicit durations on Pianalysis' existing 10 ms grid.

Representation reference: https://github.com/Natooz/MidiTok (TSD).
This is a local implementation, not MidiTok's beat-quantized codec.
"""

from __future__ import annotations

from token_protocol_v1 import TOKEN, VALUE_RANGES, MAX_TIME_DELTA, parameter_token

DURATION_BASE = 1378
MAX_DURATION = 800
V2_RANGES = {**VALUE_RANGES, "duration": (DURATION_BASE, MAX_DURATION, 1)}
V2_FIELDS = {TOKEN["TIME"]: ("time",), TOKEN["NOTE_ON_MELODY"]: ("pitch", "velocity", "duration"),
             TOKEN["NOTE_ON_ACCOMP"]: ("pitch", "velocity", "duration")}


def duration_token(ticks: int) -> int:
    if not 1 <= ticks <= MAX_DURATION:
        raise ValueError(f"Note duration must be 1..{MAX_DURATION} ticks, got {ticks}; use shorter windows")
    return DURATION_BASE + ticks - 1


def encode_notes(notes: list[dict], role: str, quantum_ms: int) -> list[int]:
    if quantum_ms <= 0:
        raise ValueError("quantum_ms must be positive")
    op = TOKEN["NOTE_ON_MELODY"] if role == "melody" else TOKEN["NOTE_ON_ACCOMP"]
    selected = sorted((n for n in notes if n["role"] == role), key=lambda n: (n["start"], n["pitch"], n["end"]))
    tokens = []
    current = 0
    for note in selected:
        start = max(0, round(float(note["start"]) * 1000 / quantum_ms))
        end = max(start + 1, round(float(note["end"]) * 1000 / quantum_ms))
        # 与 V1.1 一样拆分长停顿；不截短持续时值。
        delta = start - current
        while delta:
            step = min(delta, MAX_TIME_DELTA)
            tokens.extend([TOKEN["TIME"], parameter_token("time", step)])
            delta -= step
        current = start
        tokens.extend([op, parameter_token("pitch", int(note["pitch"])),
                       parameter_token("velocity", max(1, min(127, int(note.get("velocity", 80))))),
                       duration_token(end - start)])
    return tokens


def parse_duration_events(tokens: list[int]):
    events = []
    report = {"unknown_tokens": 0, "invalid_parameter_events": 0,
              "incomplete_events": 0, "eos_index": None}
    i = 0
    while i < len(tokens):
        op = tokens[i]
        if op == TOKEN["EOS"]:
            report["eos_index"] = i
            break
        fields = V2_FIELDS.get(op)
        if fields is None:
            report["unknown_tokens"] += 1
            i += 1
            continue
        if i + len(fields) >= len(tokens):
            report["incomplete_events"] += 1
            break
        values = []
        for field, token in zip(fields, tokens[i + 1:i + 1 + len(fields)]):
            base, count, minimum = V2_RANGES[field]
            if not base <= token < base + count:
                break
            values.append(token - base + minimum)
        if len(values) != len(fields):
            report["invalid_parameter_events"] += 1
            i += 1
            continue
        events.append((op, values))
        i += len(fields) + 1
    return events, report


def decode_notes(tokens: list[int], quantum_ms: int) -> list[dict]:
    if quantum_ms <= 0:
        raise ValueError("quantum_ms must be positive")
    events, _ = parse_duration_events(tokens)
    current = 0
    notes = []
    for op, values in events:
        if op == TOKEN["TIME"]:
            current += values[0]
        else:
            pitch, velocity, duration = values
            notes.append({"id": len(notes), "start": current * quantum_ms / 1000,
                          "end": (current + duration) * quantum_ms / 1000,
                          "pitch": pitch, "velocity": velocity,
                          "role": "melody" if op == TOKEN["NOTE_ON_MELODY"] else "accompaniment"})
    notes.sort(key=lambda n: (n["start"], n["pitch"], n["end"]))
    return notes
