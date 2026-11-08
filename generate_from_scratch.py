#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional accompaniment generation.

Despite the historical filename, this script no longer generates from scratch.
It uses a melody source prompt and asks the trained model to generate the
accompaniment target after SEP.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2LMHeadModel, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, load_midi_notes, piece_to_sample, tokens_to_notes, write_notes_midi  # noqa: E402
from harmony_v1 import harmony_tokens_for_labels, token_maps  # noqa: E402
import harmony_conditioning as hc
from token_protocol_v1 import LEGACY_VERSION, TSD_VERSION, canonical_version, parse_events  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate accompaniment from a melody prompt.")
    parser.add_argument("--model-path", default="model_output/accompaniment_gpt2_tsd_v2_e20/final_model")
    parser.add_argument("--dataset", default="data/training_windows_tsd_v2/dataset_windows_v1.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-json", default="", help="Optional JSON file containing source_tokens.")
    parser.add_argument("--melody-midi", default="", help="Optional melody-only MIDI file to use as the prompt.")
    parser.add_argument("--harmony-json", default="", help="Optional JSON containing harmony labels or harmony_tokens.")
    parser.add_argument("--out-json", default="generated_accompaniment.json")
    parser.add_argument("--out-midi", default="generated_accompaniment.mid")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantum-ms", type=int, default=10, help="Quantization used when --melody-midi is provided.")
    parser.add_argument("--min-accomp-pitch", type=int, default=36)
    parser.add_argument("--max-accomp-pitch", type=int, default=84)
    parser.add_argument("--min-note-seconds", type=float, default=0.06)
    parser.add_argument("--max-note-seconds", type=float, default=3.0)
    parser.add_argument("--max-simultaneous-notes", type=int, default=4)
    parser.add_argument("--melody-clearance", type=int, default=1, help="Keep accompaniment at least this many semitones below active melody.")
    parser.add_argument("--bar-seconds", type=float, default=2.0, help="Approximate bar length for empty-bar checks/fills.")
    parser.add_argument("--no-fill-empty-bars", action="store_true", help="Do not add a conservative bass note in empty accompaniment bars.")
    parser.add_argument("--no-grammar-constraints", action="store_true", help="Disable TSD V2 grammar constraints for ablation experiments.")
    parser.add_argument("--window-seconds", type=float, default=None, help="Melody MIDI window length; TSD defaults to its training window, 0 disables windowing.")
    return parser.parse_args()


def load_harmony_tokens(path_value: str, tokenizer_version: str = LEGACY_VERSION) -> tuple[list[int], dict[str, Any]]:
    if not path_value:
        return [], {}
    path = Path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("version") == "harmony_plan_v2":
        raise ValueError("Use windowed --melody-midi generation for a timed harmony plan and its MIDI tempo map")
    if isinstance(data.get("harmony_tokens"), list):
        version = data.get("tokenizer_version", data.get("metadata", {}).get("tokenizer_version"))
        if canonical_version(version) != canonical_version(tokenizer_version):
            raise ValueError("Harmony tokenizer_version differs from model; provide harmony labels to re-encode.")
        return [int(token) for token in data["harmony_tokens"]], {"harmony_source": str(path)}

    labels = data.get("labels")
    if labels is None and isinstance(data.get("harmony"), dict):
        labels = data["harmony"].get("labels")
    if labels is None and isinstance(data.get("metadata"), dict):
        labels = data["metadata"].get("harmony_labels")
    if not isinstance(labels, list):
        raise ValueError(f"{path} must contain harmony_tokens or harmony.labels")

    return harmony_tokens_for_labels(labels, tokenizer_version), {"harmony_source": str(path), "harmony_label_count": len(labels)}


def load_source(args: argparse.Namespace, tokenizer_version: str = LEGACY_VERSION) -> tuple[list[int], list[int], dict[str, Any], dict[str, Any]]:
    extra_harmony_tokens, harmony_info = load_harmony_tokens(args.harmony_json, tokenizer_version)

    if args.melody_midi:
        midi_path = Path(args.melody_midi)
        piece = load_midi_notes(midi_path)
        for note in piece["notes"]:
            note["role"] = "melody"
        sample = piece_to_sample(piece, args.quantum_ms, tokenizer_version)
        return sample["source_tokens"], extra_harmony_tokens, {
            "tokenizer_version": tokenizer_version,
            "token_ids": token_maps(tokenizer_version),
            "quantum_ms": args.quantum_ms,
        }, {
            "source": str(midi_path),
            "source_type": "melody_midi",
            "duration": piece["duration"],
            "melody_note_count": sample["melody_note_count"],
            "source_length": sample["source_length"],
            **harmony_info,
        }

    if args.source_json:
        path = Path(args.source_json)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        source_tokens = data.get("source_tokens")
        if not isinstance(source_tokens, list):
            raise ValueError(f"{path} must contain source_tokens")
        harmony_tokens = extra_harmony_tokens or [int(x) for x in data.get("harmony_tokens", [])]
        metadata = dict(data.get("metadata", {}))
        metadata.setdefault("tokenizer_version", data.get("tokenizer_version", LEGACY_VERSION))
        metadata.setdefault("quantum_ms", data.get("quantum_ms", 10))
        return [int(x) for x in source_tokens], harmony_tokens, metadata, {"source": str(path), **harmony_info}

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"No samples in {dataset_path}")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"--sample-index must be between 0 and {len(samples) - 1}")

    sample = samples[args.sample_index]
    source_tokens = sample.get("source_tokens")
    if not isinstance(source_tokens, list):
        if "source_length" not in sample:
            raise ValueError("Dataset sample needs source_tokens or source_length; numeric SEP is ambiguous in V1.")
        source_tokens = sample["training_sequence"][1:1 + int(sample["source_length"])]
    harmony_tokens = extra_harmony_tokens or [int(x) for x in sample.get("harmony_tokens", [])]
    return [int(x) for x in source_tokens], harmony_tokens, data.get("metadata", {}), {**sample, **harmony_info}


def first_eos_position(tokens: list[int], eos_token_id: int, tokenizer_version: str = LEGACY_VERSION) -> int | None:
    if eos_token_id != TOKEN["EOS"]:
        raise ValueError("Unsupported EOS token ID")
    if canonical_version(tokenizer_version) != LEGACY_VERSION:
        return tokens.index(eos_token_id) if eos_token_id in tokens else None
    return parse_events(tokens, tokenizer_version)[1]["eos_index"]


def validate_source_metadata(metadata: dict[str, Any], tokenizer_version: str, quantum_ms: int) -> None:
    if canonical_version(metadata.get("tokenizer_version")) != canonical_version(tokenizer_version):
        raise ValueError("Source tokenizer_version differs from model. Rebuild the dataset with the model's tokenizer version.")
    if int(metadata.get("quantum_ms", 10)) != quantum_ms or quantum_ms <= 0:
        raise ValueError("Source quantum_ms differs from model or is not positive.")


def model_protocol(model_path: Path, model: GPT2LMHeadModel) -> tuple[str, int]:
    metadata_path = model_path.parent / "training_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("dataset_metadata", {})
    version = canonical_version(getattr(model.config, "pianalysis_tokenizer_version", metadata.get("tokenizer_version")))
    quantum_ms = int(getattr(model.config, "pianalysis_quantum_ms", metadata.get("quantum_ms", 10)))
    return version, quantum_ms


class EventEOSCriteria(StoppingCriteria):
    def __init__(self, prompt_length: int, tokenizer_version: str):
        self.prompt_length = prompt_length
        self.tokenizer_version = tokenizer_version

    def __call__(self, input_ids, scores, **kwargs):
        finished = [first_eos_position(row[self.prompt_length:].tolist(), TOKEN["EOS"], self.tokenizer_version) is not None for row in input_ids]
        return torch.tensor(finished, dtype=torch.bool, device=input_ids.device)


def event_diagnostics(tokens: list[int], tokenizer_version: str) -> dict[str, Any]:
    events, report = parse_events(tokens, tokenizer_version)
    active: dict[tuple[int, int], int] = {}
    orphan_offs = 0
    elapsed = 0
    wrong_role_events = 0
    for op, values in events:
        if op == TOKEN["TIME"]:
            elapsed += values[0]
            continue
        wrong_role_events += op in (TOKEN["NOTE_ON_MELODY"], TOKEN["NOTE_OFF_MELODY"])
        if canonical_version(tokenizer_version) == TSD_VERSION:
            continue
        role = 0 if op in (10, 11) else 1
        key = (role, values[0])
        if op in (10, 20):
            active[key] = active.get(key, 0) + 1
        elif active.get(key, 0):
            active[key] -= 1
        else:
            orphan_offs += 1
    return {**report, "orphan_note_offs": orphan_offs, "unclosed_notes": sum(active.values()),
            "wrong_role_note_events": wrong_role_events, "elapsed_ticks": elapsed}


def generate_target(model, prompt: list[int], args, tokenizer_version: str, logits_processor=None, target_duration=None):
    tokenizer_version = canonical_version(tokenizer_version)
    available = int(model.config.n_positions) - len(prompt)
    if available <= 0:
        raise ValueError(f"Prompt has {len(prompt)} tokens; model context is {model.config.n_positions}. Use a shorter melody window.")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if any(token < 0 or token >= model.config.vocab_size for token in prompt):
        raise ValueError("Prompt token exceeds model vocabulary; check tokenizer_version and harmony inputs.")
    budget = min(args.max_new_tokens, available)
    processors = LogitsProcessorList(list(logits_processor or []))
    grammar = None
    if tokenizer_version == TSD_VERSION and not getattr(args, "no_grammar_constraints", False):
        from duration_constraints import DurationGrammar
        quantum_ms = int(getattr(model.config, "pianalysis_quantum_ms", 10))
        horizon = None if target_duration is None else round(target_duration * 1000 / quantum_ms)
        grammar = DurationGrammar(len(prompt), budget, horizon)
        processors.append(grammar)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=model.device)
    legacy = tokenizer_version == LEGACY_VERSION
    # 旧协议的数值 3 也会出现在参数位，必须关闭 Hugging Face 的逐 token EOS 判断。
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=budget, do_sample=True, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, pad_token_id=TOKEN["PAD"],
            eos_token_id=None if legacy else TOKEN["EOS"], forced_eos_token_id=None,
            stopping_criteria=StoppingCriteriaList([EventEOSCriteria(len(prompt), tokenizer_version)]) if legacy else None,
            logits_processor=processors,
        )
    generated = output[0, len(prompt):].detach().cpu().tolist()
    eos_at = first_eos_position(generated, TOKEN["EOS"], tokenizer_version)
    if eos_at is not None:
        stop_reason = "eos"
    elif len(generated) >= budget:
        stop_reason = "context_limit" if budget < args.max_new_tokens else "max_new_tokens"
    else:
        stop_reason = "unexpected_end"
    target = generated if eos_at is None else generated[:eos_at]
    report = {
        **event_diagnostics(target, tokenizer_version),
        "stop_reason": stop_reason, "requested_max_new_tokens": args.max_new_tokens,
        "effective_max_new_tokens": budget, "generated_tokens_including_eos": len(generated),
        "eos_index": eos_at, "prompt_length": len(prompt),
        "grammar_constraints": grammar is not None,
        "forced_eos_reason": grammar.forced_eos_reason if grammar is not None else None,
        "horizon_ticks": grammar.horizon_ticks if grammar is not None else None,
    }
    return target, report


def generate_windowed(model, piece: dict, args, tokenizer_version: str):
    from build_training_windows_v1 import slice_piece
    from tokenizer_v2 import MAX_DURATION

    quantum_ms = int(getattr(model.config, "pianalysis_quantum_ms", 10))
    window_seconds = args.window_seconds
    if window_seconds is None:
        window_seconds = float(getattr(model.config, "pianalysis_window_seconds", 8.0))
    if window_seconds <= 0 or (tokenizer_version == TSD_VERSION and window_seconds > MAX_DURATION * quantum_ms / 1000):
        raise ValueError("Window length must be positive and fit the tokenizer duration range")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    melody = [{**note, "role": "melody"} for note in piece["notes"]]
    if not melody:
        raise ValueError("Melody MIDI contains no notes")
    duration = max(note["end"] for note in melody)
    source = {**piece, "notes": melody, "duration": duration}
    hc.validate_model_conditioning(model, hc.VERSION if "timed_harmony" in source else None)
    source_budget = model.config.n_positions - min(args.max_new_tokens, model.config.n_positions // 2)
    intervals = [(i * window_seconds, min(duration, (i + 1) * window_seconds))
                 for i in range(math.ceil(duration / window_seconds))]
    planned = []

    def add_interval(start, end):
        interval = slice_piece(source, start, end)
        tokens = piece_to_sample(interval, quantum_ms, tokenizer_version)["source_tokens"]
        harmony = hc.encode_labels(interval["timed_harmony"], quantum_ms) if "timed_harmony" in interval else []
        if len(tokens) + len(harmony) + 2 + bool(harmony) > source_budget:
            # 密集旋律按时间继续细分；绝不截掉 source token。
            midpoint = round((start + end) * 500 / quantum_ms) * quantum_ms / 1000
            if midpoint <= start or midpoint >= end:
                raise ValueError("Melody density cannot fit model context even at the minimum time window")
            add_interval(start, midpoint)
            add_interval(midpoint, end)
        else:
            planned.append((start, end, tokens, harmony))

    for start, end in intervals:
        add_interval(start, end)
    accompaniment, reports = [], []
    for start, end, source_tokens, harmony in planned:
        if not source_tokens:
            reports.append({"start": start, "end": end, "skipped_silent": True,
                            "source_tokens": [], "generated_target_tokens": [],
                            "generation_report": {**event_diagnostics([], tokenizer_version), "stop_reason": "silent_window"}})
            continue
        prompt = [TOKEN["BOS"], *source_tokens] + ([TOKEN["HARMONY"], *harmony] if harmony else []) + [TOKEN["SEP"]]
        target, report = generate_target(model, prompt, args,
                                        tokenizer_version, target_duration=end - start)
        for note in tokens_to_notes(target, quantum_ms, tokenizer_version):
            accompaniment.append({**note, "id": len(accompaniment), "role": "accompaniment",
                                  "start": note["start"] + start, "end": note["end"] + start})
        reports.append({"start": start, "end": end, "skipped_silent": False,
                        "source_tokens": source_tokens, "harmony_tokens": harmony,
                        "generated_target_tokens": target, "generation_report": report})
    return melody, accompaniment, reports


def note_duration(note: dict[str, Any]) -> float:
    return float(note["end"]) - float(note["start"])


def active_melody_pitch(melody_notes: list[dict[str, Any]], start: float, end: float) -> int | None:
    active = [
        int(note["pitch"])
        for note in melody_notes
        if float(note["start"]) < end and float(note["end"]) > start
    ]
    if not active:
        return None
    return max(active)


def clamp_accompaniment_note(
    note: dict[str, Any],
    melody_notes: list[dict[str, Any]],
    args: argparse.Namespace,
    report: dict[str, int],
) -> dict[str, Any] | None:
    cleaned = dict(note)
    cleaned["role"] = "accompaniment"
    cleaned["start"] = max(0.0, float(cleaned["start"]))
    cleaned["end"] = max(cleaned["start"], float(cleaned["end"]))

    if note_duration(cleaned) < args.min_note_seconds:
        report["removed_too_short"] += 1
        return None

    if note_duration(cleaned) > args.max_note_seconds:
        cleaned["end"] = cleaned["start"] + args.max_note_seconds
        report["capped_too_long"] += 1

    pitch = int(cleaned["pitch"])
    while pitch > args.max_accomp_pitch:
        pitch -= 12
        report["transposed_into_range"] += 1
    while pitch < args.min_accomp_pitch:
        pitch += 12
        report["transposed_into_range"] += 1

    melody_pitch = active_melody_pitch(melody_notes, cleaned["start"], cleaned["end"])
    if melody_pitch is not None:
        ceiling = melody_pitch - args.melody_clearance
        while pitch > ceiling and pitch - 12 >= args.min_accomp_pitch:
            pitch -= 12
            report["transposed_below_melody"] += 1
        if pitch > ceiling:
            report["removed_above_melody"] += 1
            return None

    if pitch < args.min_accomp_pitch or pitch > args.max_accomp_pitch:
        report["removed_out_of_range"] += 1
        return None

    cleaned["pitch"] = pitch
    cleaned["velocity"] = max(24, min(96, int(cleaned.get("velocity", 72))))
    return cleaned


def merge_same_pitch_overlaps(notes: list[dict[str, Any]], report: dict[str, int]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_pitch: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        by_pitch.setdefault(int(note["pitch"]), []).append(note)

    for pitch, items in by_pitch.items():
        items.sort(key=lambda item: (float(item["start"]), float(item["end"])))
        current: dict[str, Any] | None = None
        for item in items:
            if current is None:
                current = dict(item)
                continue
            if float(item["start"]) <= float(current["end"]) + 0.02:
                current["end"] = max(float(current["end"]), float(item["end"]))
                current["velocity"] = max(int(current.get("velocity", 72)), int(item.get("velocity", 72)))
                report["merged_same_pitch_overlap"] += 1
            else:
                merged.append(current)
                current = dict(item)
        if current is not None:
            current["pitch"] = pitch
            merged.append(current)

    merged.sort(key=lambda item: (float(item["start"]), int(item["pitch"]), float(item["end"])))
    return merged


def limit_simultaneous_notes(
    notes: list[dict[str, Any]],
    max_notes: int,
    report: dict[str, int],
) -> list[dict[str, Any]]:
    if max_notes <= 0:
        return []

    kept: list[dict[str, Any]] = []
    for note in sorted(notes, key=lambda item: (float(item["start"]), int(item["pitch"]))):
        start = float(note["start"])
        active = [
            item
            for item in kept
            if float(item["start"]) <= start < float(item["end"])
        ]
        if len(active) >= max_notes:
            report["removed_excess_polyphony"] += 1
            continue
        kept.append(note)
    return kept


def fill_empty_bars(
    notes: list[dict[str, Any]],
    melody_notes: list[dict[str, Any]],
    duration: float,
    args: argparse.Namespace,
    report: dict[str, int],
) -> list[dict[str, Any]]:
    if args.no_fill_empty_bars or args.bar_seconds <= 0:
        return notes

    filled = list(notes)
    bar_count = int(duration / args.bar_seconds) + 1
    next_id = len(filled)
    for bar in range(bar_count):
        start = bar * args.bar_seconds
        end = min(duration, start + args.bar_seconds)
        if end <= start:
            continue
        has_accompaniment = any(float(note["start"]) < end and float(note["end"]) > start for note in filled)
        if has_accompaniment:
            continue
        melody_pitch = active_melody_pitch(melody_notes, start, end)
        if melody_pitch is None:
            continue
        pitch = melody_pitch - 24
        while pitch < args.min_accomp_pitch:
            pitch += 12
        while pitch > min(args.max_accomp_pitch, melody_pitch - args.melody_clearance):
            pitch -= 12
        if pitch < args.min_accomp_pitch:
            continue
        filled.append(
            {
                "id": next_id,
                "start": start,
                "end": min(end, start + max(args.min_note_seconds, min(args.max_note_seconds, args.bar_seconds * 0.75))),
                "pitch": pitch,
                "velocity": 48,
                "role": "accompaniment",
            }
        )
        next_id += 1
        report["filled_empty_bars"] += 1
    filled.sort(key=lambda item: (float(item["start"]), int(item["pitch"]), float(item["end"])))
    return filled


def sanitize_accompaniment(
    melody_notes: list[dict[str, Any]],
    accompaniment_notes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    report: dict[str, int] = {
        "raw_accompaniment_notes": len(accompaniment_notes),
        "removed_too_short": 0,
        "capped_too_long": 0,
        "transposed_into_range": 0,
        "transposed_below_melody": 0,
        "removed_above_melody": 0,
        "removed_out_of_range": 0,
        "merged_same_pitch_overlap": 0,
        "removed_excess_polyphony": 0,
        "truncated_after_melody": 0,
        "filled_empty_bars": 0,
    }
    if melody_notes:
        melody_end = max(float(note["end"]) for note in melody_notes)
    else:
        melody_end = 0.0

    cleaned: list[dict[str, Any]] = []
    for note in accompaniment_notes:
        if melody_end and float(note["start"]) >= melody_end:
            report["truncated_after_melody"] += 1
            continue
        adjusted = dict(note)
        if melody_end:
            adjusted["end"] = min(float(adjusted["end"]), melody_end)
        filtered = clamp_accompaniment_note(adjusted, melody_notes, args, report)
        if filtered is not None:
            cleaned.append(filtered)

    cleaned = merge_same_pitch_overlaps(cleaned, report)
    cleaned = limit_simultaneous_notes(cleaned, args.max_simultaneous_notes, report)
    cleaned = fill_empty_bars(cleaned, melody_notes, melody_end, args, report)
    cleaned = merge_same_pitch_overlaps(cleaned, report)
    cleaned = limit_simultaneous_notes(cleaned, args.max_simultaneous_notes, report)

    for idx, note in enumerate(cleaned):
        note["id"] = idx
        note["role"] = "accompaniment"

    final_report: dict[str, int | float] = dict(report)
    final_report["sanitized_accompaniment_notes"] = len(cleaned)
    final_report["melody_duration"] = round(melody_end, 4)
    return cleaned, final_report


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path)
    print(f"Loading model: {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer_version, model_quantum_ms = model_protocol(model_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    window_seconds = args.window_seconds
    if window_seconds is None:
        window_seconds = float(getattr(model.config, "pianalysis_window_seconds", 8.0)) if tokenizer_version == TSD_VERSION else 0
    if window_seconds < 0:
        raise ValueError("window_seconds must not be negative")
    if window_seconds and args.melody_midi:
        validate_source_metadata({"tokenizer_version": tokenizer_version, "quantum_ms": args.quantum_ms}, tokenizer_version, model_quantum_ms)
        args.window_seconds = window_seconds
        piece = load_midi_notes(Path(args.melody_midi))
        if args.harmony_json:
            from harmony_plan import midi_timeline
            import pretty_midi
            import hashlib
            plan = json.loads(Path(args.harmony_json).read_text(encoding="utf-8"))
            if plan.get("melody_sha256") and plan["melody_sha256"] != hashlib.sha256(Path(args.melody_midi).read_bytes()).hexdigest():
                raise ValueError("Harmony plan does not match the melody MIDI hash")
            timeline = midi_timeline(pretty_midi.PrettyMIDI(args.melody_midi))
            if plan["duration_ticks"] != timeline["duration_ticks"]:
                raise ValueError("Harmony plan and MIDI duration differ")
            piece["timed_harmony"] = hc.plan_labels_seconds(plan, timeline)
        melody, raw, windows = generate_windowed(model, piece, args, tokenizer_version)
        cleaned, sanitation = sanitize_accompaniment(melody, raw, args)
        payload = {"model_path": str(model_path), "tokenizer_version": tokenizer_version, "quantum_ms": model_quantum_ms,
                   "harmony_tokenizer_version": getattr(model.config, "pianalysis_harmony_version", None),
                   "harmony_source": args.harmony_json or None,
                   "source_info": {"source": args.melody_midi, "duration": max(n["end"] for n in melody)},
                   "generation_params": vars(args), "windows": windows,
                   "melody_note_count": len(melody), "raw_accompaniment_note_count": len(raw),
                   "accompaniment_note_count": len(cleaned), "raw_accompaniment_notes": raw,
                   "sanitation_report": sanitation, "out_midi": args.out_midi}
        write_notes_midi(melody + cleaned, Path(args.out_midi))
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Generated {len(windows)} windows; raw notes={len(raw)}; sanitized notes={len(cleaned)}")
        print(f"Wrote {args.out_json}")
        print(f"Wrote {args.out_midi}")
        return
    if window_seconds and args.window_seconds is not None and not args.melody_midi:
        raise ValueError("--window-seconds requires --melody-midi")
    source_tokens, harmony_tokens, metadata, source_info = load_source(args, tokenizer_version)
    validate_source_metadata(metadata, tokenizer_version, model_quantum_ms)
    hc.validate_model_conditioning(model, metadata.get("harmony_tokenizer_version"))
    token_ids = dict(TOKEN)
    token_ids.update(token_maps(tokenizer_version))
    token_ids.update(metadata.get("token_ids", {}))
    token_ids = {key: int(value) for key, value in token_ids.items()}
    quantum_ms = int(metadata.get("quantum_ms", 10))
    if metadata.get("harmony_tokenizer_version") == hc.VERSION:
        if not harmony_tokens:
            raise ValueError("Timed harmony model requires harmony tokens")
        hc.decode_labels(harmony_tokens, quantum_ms)

    prompt = [token_ids["BOS"]] + source_tokens
    if harmony_tokens:
        prompt += [token_ids["HARMONY"]] + harmony_tokens
    prompt += [token_ids["SEP"]]
    source_duration = source_info.get("window_duration", source_info.get("duration"))
    target_tokens, generation_report = generate_target(model, prompt, args, tokenizer_version, target_duration=source_duration)

    melody_notes = tokens_to_notes(source_tokens, quantum_ms, tokenizer_version)
    raw_accompaniment_notes = tokens_to_notes(target_tokens, quantum_ms, tokenizer_version)
    for note in melody_notes:
        note["role"] = "melody"
    for note in raw_accompaniment_notes:
        note["role"] = "accompaniment"
    accompaniment_notes, sanitation_report = sanitize_accompaniment(
        melody_notes,
        raw_accompaniment_notes,
        args,
    )

    write_notes_midi(melody_notes + accompaniment_notes, Path(args.out_midi))

    payload = {
        "model_path": str(model_path),
        "tokenizer_version": tokenizer_version,
        "quantum_ms": quantum_ms,
        "generation_report": generation_report,
        "source_info": source_info,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "prompt": prompt,
        "source_tokens": source_tokens,
        "harmony_tokens": harmony_tokens,
        "generated_target_tokens": target_tokens,
        "melody_note_count": len(melody_notes),
        "raw_accompaniment_note_count": len(raw_accompaniment_notes),
        "accompaniment_note_count": len(accompaniment_notes),
        "sanitation_report": sanitation_report,
        "out_midi": args.out_midi,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_json).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Generated target tokens: {len(target_tokens)}")
    print(
        f"Melody notes: {len(melody_notes)}; "
        f"raw accompaniment notes: {len(raw_accompaniment_notes)}; "
        f"sanitized accompaniment notes: {len(accompaniment_notes)}"
    )
    print(f"Sanitation report: {sanitation_report}")
    print(f"Generation report: {generation_report}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_midi}")


if __name__ == "__main__":
    main()
