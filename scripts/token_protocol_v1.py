"""Versioned V1 event parameters; V1.1 keeps the original note-on/off grammar."""

from __future__ import annotations

LEGACY_VERSION = "pianalysis_numeric_v1"
TYPED_VERSION = "pianalysis_typed_v1_1"
TSD_VERSION = "pianalysis_tsd_v2"

TOKEN = {
    "PAD": 0, "BOS": 1, "SEP": 2, "EOS": 3, "TIME": 4,
    "NOTE_ON_MELODY": 10, "NOTE_OFF_MELODY": 11,
    "NOTE_ON_ACCOMP": 20, "NOTE_OFF_ACCOMP": 21,
    "HARMONY": 30, "BAR": 31, "CHORD_NC": 32,
}

EVENT_FIELDS = {4: ("time",), 10: ("pitch", "velocity"), 11: ("pitch",),
                20: ("pitch", "velocity"), 21: ("pitch",)}
# 每项为起始 ID、取值数量、最小数值；长时间间隔拆成多个 TIME 事件。
VALUE_RANGES = {"pitch": (64, 128, 0), "velocity": (192, 127, 1), "time": (319, 800, 1)}
MAX_TIME_DELTA = 800
CHORD_BASE = 1119
BASS_BASE = 1239
BASS_NC = 1367
CONF_BASE = 1368


def canonical_version(version: str | None) -> str:
    if version in (None, LEGACY_VERSION, "pianalysis_numeric_v1_harmony_prompt"):
        return LEGACY_VERSION
    if version in (TYPED_VERSION, TSD_VERSION):
        return version
    raise ValueError(f"Unsupported tokenizer_version: {version!r}")


def parameter_token(field: str, value: int) -> int:
    base, count, minimum = VALUE_RANGES[field]
    if not minimum <= value < minimum + count:
        raise ValueError(f"Invalid {field}: {value}")
    return base + value - minimum


def event_token_map(tokenizer_version: str = TYPED_VERSION) -> dict[str, int]:
    mapping = dict(TOKEN)
    for field, (base, count, minimum) in VALUE_RANGES.items():
        mapping.update({f"{field.upper()}_{minimum + i}": base + i for i in range(count)})
    if canonical_version(tokenizer_version) == TSD_VERSION:
        from tokenizer_v2 import DURATION_BASE, MAX_DURATION
        mapping.update({f"DURATION_{i + 1}": DURATION_BASE + i for i in range(MAX_DURATION)})
    return mapping


def parse_events(tokens: list[int], tokenizer_version: str = LEGACY_VERSION):
    """Return complete events and diagnostics; EOS is meaningful only at a boundary."""
    version = canonical_version(tokenizer_version)
    if version == TSD_VERSION:
        from tokenizer_v2 import parse_duration_events
        return parse_duration_events(tokens)
    events = []
    report = {"unknown_tokens": 0, "invalid_parameter_events": 0,
              "incomplete_events": 0, "eos_index": None}
    i = 0
    while i < len(tokens):
        op = tokens[i]
        if op == TOKEN["EOS"]:
            report["eos_index"] = i
            break
        fields = EVENT_FIELDS.get(op)
        if fields is None:
            report["unknown_tokens"] += 1
            i += 1
            continue
        if i + len(fields) >= len(tokens):
            report["incomplete_events"] += 1
            break
        values = []
        valid = True
        for field, raw in zip(fields, tokens[i + 1:i + 1 + len(fields)]):
            base, count, minimum = VALUE_RANGES[field]
            value = raw - base + minimum if version == TYPED_VERSION else raw
            if version == TYPED_VERSION:
                valid &= minimum <= value < minimum + count
            else:
                valid &= value >= 0 if field == "time" else minimum <= value < minimum + count
            values.append(value)
        if valid:
            events.append((op, values))
        else:
            report["invalid_parameter_events"] += 1
        # 类型化参数不可能是事件指令；遇到坏事件时从下一 token 重新同步。
        i += 1 if not valid and version == TYPED_VERSION else 1 + len(fields)
    return events, report


def encode_legacy_events(tokens: list[int], tokenizer_version: str) -> list[int]:
    if canonical_version(tokenizer_version) == TSD_VERSION:
        raise ValueError("TSD encoding requires notes with explicit start/end, not NOTE_ON/OFF events")
    if canonical_version(tokenizer_version) == LEGACY_VERSION:
        return tokens
    events, report = parse_events(tokens)
    if any(report[key] for key in ("unknown_tokens", "invalid_parameter_events", "incomplete_events")) or report["eos_index"] is not None:
        raise ValueError(f"Cannot encode invalid event stream: {report}")
    encoded = []
    for op, values in events:
        if op == TOKEN["TIME"]:
            remaining = values[0]
            while remaining:
                delta = min(remaining, MAX_TIME_DELTA)
                encoded.extend([op, parameter_token("time", delta)])
                remaining -= delta
        else:
            encoded.append(op)
            encoded.extend(parameter_token(field, value) for field, value in zip(EVENT_FIELDS[op], values))
    return encoded


def decode_to_legacy_events(tokens: list[int], tokenizer_version: str) -> list[int]:
    if canonical_version(tokenizer_version) == TSD_VERSION:
        raise ValueError("Use tokenizer_v2.decode_notes for explicit-duration tokens")
    if canonical_version(tokenizer_version) == LEGACY_VERSION:
        return tokens
    events, _ = parse_events(tokens, tokenizer_version)
    return [token for op, values in events for token in (op, *values)]
