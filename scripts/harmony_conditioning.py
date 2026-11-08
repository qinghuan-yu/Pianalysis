"""Timed harmony prompts for TSD notes; annotation review remains independent."""

from bisect import bisect_right
import math

from token_protocol_v1 import TOKEN, TSD_VERSION, parameter_token, VALUE_RANGES
from tokenizer_v2 import DURATION_BASE, MAX_DURATION, duration_token

VERSION = "pianalysis_timed_harmony_v2"
CHORD, NO_CHORD, UNKNOWN = range(DURATION_BASE + MAX_DURATION, DURATION_BASE + MAX_DURATION + 3)
ROOT_BASE, BASS_BASE, PC_BASE = UNKNOWN + 1, UNKNOWN + 13, UNKNOWN + 25


def token_map():
    return {"H_CHORD": CHORD, "H_NO_CHORD": NO_CHORD, "H_UNKNOWN": UNKNOWN,
            **{f"H_{name}_{pc}": base + pc for name, base in (("ROOT", ROOT_BASE), ("BASS", BASS_BASE), ("PC", PC_BASE))
               for pc in range(12)}}


def plan_labels_seconds(plan, timeline):
    from harmony_plan import validate_plan
    validate_plan(plan)
    if timeline["ppq"] != plan["ppq"] or plan.get("origin_tick", 0):
        raise ValueError("Harmony requires a global plan and matching tempo-map PPQ")
    tempos = timeline["tempo_changes"]
    if not tempos or tempos[0]["tick"] != 0 or any(b["tick"] <= a["tick"] for a, b in zip(tempos, tempos[1:])):
        raise ValueError("Invalid harmony tempo map")
    if any(not math.isfinite(p["bpm"]) or p["bpm"] <= 0 for p in tempos):
        raise ValueError("Invalid harmony tempo")
    ticks = [p["tick"] for p in tempos]

    def seconds(tick):
        point = tempos[bisect_right(ticks, tick) - 1]
        return point["seconds"] + (tick - point["tick"]) * 60 / (point["bpm"] * plan["ppq"])

    return [{**label, "start": seconds(label["start_tick"]), "end": seconds(label["end_tick"])}
            for label in plan["labels"]]


def window_labels(labels, start, end):
    if not 0 <= start < end:
        raise ValueError("Invalid harmony window")
    result, cursor = [], 0.0
    keys = ("status", "root_pc", "bass_pc", "pitch_classes", "source", "review_status")

    def unknown(a, b):
        return {"start": a, "end": b, "symbol": "X", "status": "unknown", "root_pc": None,
                "bass_pc": None, "pitch_classes": [], "source": "unannotated_gap", "review_status": "unreviewed"}

    for label in labels:
        if label["start"] >= end or label["end"] <= start:
            continue
        a, b = max(start, label["start"]) - start, min(end, label["end"]) - start
        if a < cursor - 1e-8:
            raise ValueError("Overlapping or unsorted harmony labels")
        if a > cursor + 1e-8:
            result.append(unknown(cursor, a))
        item = {**label, "start": a, "end": b}
        # 原始 tick 留在完整标注中；切窗后的条件统一使用相对秒数，避免混用坐标。
        item.pop("start_tick", None)
        item.pop("end_tick", None)
        if result and abs(result[-1]["end"] - a) < 1e-8 and all(result[-1].get(k) == item.get(k) for k in keys):
            result[-1]["end"] = b
        else:
            result.append(item)
        cursor = b
    if cursor < end - start - 1e-8:
        result.append(unknown(cursor, end - start))
    return result


def encode_labels(labels, quantum_ms=10):
    if quantum_ms <= 0:
        raise ValueError("Harmony quantum_ms must be positive")
    tokens, current, previous_end = [], 0, 0
    for label in labels:
        start, end = (round(label[k] * 1000 / quantum_ms) for k in ("start", "end"))
        if start < previous_end or end < start:
            raise ValueError("Invalid harmony interval")
        # 小于量化网格且两端落在同一点的片段没有可表示的持续时值。
        if start == end:
            continue
        delta = start - current
        while delta:
            step = min(delta, 800)
            tokens.extend([TOKEN["TIME"], parameter_token("time", step)])
            delta -= step
        current, previous_end = start, end
        if label["status"] == "chord":
            root, bass, pcs = label["root_pc"], label["bass_pc"], label["pitch_classes"]
            if root not in range(12) or bass not in range(12) or not pcs or pcs != sorted(set(pcs)) or any(p not in range(12) for p in pcs):
                raise ValueError("Invalid harmony pitch classes")
            tokens.extend([CHORD, ROOT_BASE + root, BASS_BASE + bass, *[PC_BASE + pc for pc in pcs]])
        elif label["status"] in ("no_chord", "unknown"):
            tokens.append(NO_CHORD if label["status"] == "no_chord" else UNKNOWN)
        else:
            raise ValueError("Invalid harmony status")
        tokens.append(duration_token(end - start))
    return tokens


def decode_labels(tokens, quantum_ms=10):
    if quantum_ms <= 0:
        raise ValueError("Harmony quantum_ms must be positive")
    labels, i, current, previous_end = [], 0, 0, 0
    while i < len(tokens):
        op = tokens[i]
        i += 1
        if op == TOKEN["TIME"]:
            base, count, minimum = VALUE_RANGES["time"]
            if i >= len(tokens) or not base <= tokens[i] < base + count:
                raise ValueError("Invalid harmony TIME")
            current += tokens[i] - base + minimum
            i += 1
            if i == len(tokens):
                raise ValueError("Harmony TIME must precede a span")
            continue
        root, bass, pcs = None, None, []
        if op == CHORD:
            if i + 2 >= len(tokens) or not ROOT_BASE <= tokens[i] < ROOT_BASE + 12 or not BASS_BASE <= tokens[i + 1] < BASS_BASE + 12:
                raise ValueError("Invalid harmony root or bass")
            root, bass = tokens[i] - ROOT_BASE, tokens[i + 1] - BASS_BASE
            i += 2
            while i < len(tokens) and PC_BASE <= tokens[i] < PC_BASE + 12:
                pcs.append(tokens[i] - PC_BASE)
                i += 1
            if not pcs or pcs != sorted(set(pcs)):
                raise ValueError("Invalid harmony pitch-class set")
            status = "chord"
        elif op in (NO_CHORD, UNKNOWN):
            status = "no_chord" if op == NO_CHORD else "unknown"
        else:
            raise ValueError("Invalid harmony event")
        if i >= len(tokens) or not DURATION_BASE <= tokens[i] < DURATION_BASE + MAX_DURATION:
            raise ValueError("Invalid harmony duration")
        end = current + tokens[i] - DURATION_BASE + 1
        if current < previous_end:
            raise ValueError("Overlapping harmony tokens")
        labels.append({"start": current * quantum_ms / 1000, "end": end * quantum_ms / 1000,
                       "status": status, "root_pc": root, "bass_pc": bass, "pitch_classes": pcs})
        previous_end = end
        i += 1
    return labels


def condition_sample(sample, labels):
    if sample["tokenizer_version"] != TSD_VERSION:
        raise ValueError("Timed harmony conditioning requires TSD V2")
    tokens = encode_labels(labels, sample.get("quantum_ms", 10))
    sample.update({"harmony_tokens": tokens, "harmony_labels": labels, "harmony_tokenizer_version": VERSION,
                   "harmony_length": len(tokens), "target_start_index": len(sample["source_tokens"]) + len(tokens) + 3})
    sample["training_sequence"] = [TOKEN["BOS"], *sample["source_tokens"], TOKEN["HARMONY"], *tokens,
                                   TOKEN["SEP"], *sample["target_tokens"], TOKEN["EOS"]]
    sample["total_length"] = len(sample["training_sequence"])


def validate_model_conditioning(model, version):
    expected = getattr(model.config, "pianalysis_harmony_version", None)
    if expected != version:
        raise ValueError(f"Model harmony protocol {expected!r} differs from input {version!r}; use a matching trained model")
