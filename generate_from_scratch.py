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
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2LMHeadModel


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, load_midi_notes, piece_to_sample, tokens_to_notes, write_notes_midi  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate accompaniment from a melody prompt.")
    parser.add_argument("--model-path", default="model_output/accompaniment_gpt2/final_model")
    parser.add_argument("--dataset", default="data/training_windows_v1/dataset_windows_v1.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-json", default="", help="Optional JSON file containing source_tokens.")
    parser.add_argument("--melody-midi", default="", help="Optional melody-only MIDI file to use as the prompt.")
    parser.add_argument("--out-json", default="generated_accompaniment.json")
    parser.add_argument("--out-midi", default="generated_accompaniment.mid")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
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
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    if args.melody_midi:
        midi_path = Path(args.melody_midi)
        piece = load_midi_notes(midi_path)
        for note in piece["notes"]:
            note["role"] = "melody"
        sample = piece_to_sample(piece, args.quantum_ms)
        return sample["source_tokens"], {"token_ids": TOKEN, "quantum_ms": args.quantum_ms}, {
            "source": str(midi_path),
            "source_type": "melody_midi",
            "duration": piece["duration"],
            "melody_note_count": sample["melody_note_count"],
            "source_length": sample["source_length"],
        }

    if args.source_json:
        path = Path(args.source_json)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        source_tokens = data.get("source_tokens")
        if not isinstance(source_tokens, list):
            raise ValueError(f"{path} must contain source_tokens")
        return [int(x) for x in source_tokens], data.get("metadata", {}), {"source": str(path)}

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
        seq = sample["training_sequence"]
        sep_index = seq.index(data["metadata"]["token_ids"]["SEP"])
        source_tokens = seq[1:sep_index]
    return [int(x) for x in source_tokens], data.get("metadata", {}), sample


def first_eos_position(tokens: list[int], eos_token_id: int) -> int | None:
    for idx, token in enumerate(tokens):
        if token == eos_token_id:
            return idx
    return None


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

    source_tokens, metadata, source_info = load_source(args)
    token_ids = dict(TOKEN)
    token_ids.update(metadata.get("token_ids", {}))
    token_ids = {key: int(value) for key, value in token_ids.items()}
    quantum_ms = int(metadata.get("quantum_ms", 10))

    prompt = [token_ids["BOS"]] + source_tokens + [token_ids["SEP"]]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    model_path = Path(args.model_path)
    print(f"Loading model: {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=token_ids["PAD"],
            eos_token_id=token_ids["EOS"],
        )

    generated = output[0].detach().cpu().tolist()
    target_tokens = generated[len(prompt) :]
    eos_at = first_eos_position(target_tokens, token_ids["EOS"])
    if eos_at is not None:
        target_tokens = target_tokens[:eos_at]

    melody_notes = tokens_to_notes(source_tokens, quantum_ms)
    raw_accompaniment_notes = tokens_to_notes(target_tokens, quantum_ms)
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
        "generated_target_tokens": target_tokens,
        "melody_note_count": len(melody_notes),
        "raw_accompaniment_note_count": len(raw_accompaniment_notes),
        "accompaniment_note_count": len(accompaniment_notes),
        "sanitation_report": sanitation_report,
        "out_midi": args.out_midi,
    }
    with Path(args.out_json).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Generated target tokens: {len(target_tokens)}")
    print(
        f"Melody notes: {len(melody_notes)}; "
        f"raw accompaniment notes: {len(raw_accompaniment_notes)}; "
        f"sanitized accompaniment notes: {len(accompaniment_notes)}"
    )
    print(f"Sanitation report: {sanitation_report}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_midi}")


if __name__ == "__main__":
    main()
