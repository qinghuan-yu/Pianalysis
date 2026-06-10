#!/usr/bin/env python3
"""
Build trainable fixed-duration windows from annotated note JSON files.

Input is expected from scripts/dp_melody_cleaning_v1.py:
  data/dp_cleaned_v1/annotated_notes/*.json

Output samples are short conditional sequences:
  [BOS] source_melody [SEP] target_accompaniment [EOS]

Each sample includes target_start_index so train_v2.py can mask source loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, piece_to_sample, tokens_to_notes, write_notes_midi  # noqa: E402


def slice_piece(piece: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    sliced_notes: list[dict[str, Any]] = []
    for note in piece["notes"]:
        if note["start"] < end and note["end"] > start:
            adjusted = dict(note)
            adjusted["start"] = max(0.0, float(note["start"]) - start)
            adjusted["end"] = min(end - start, float(note["end"]) - start)
            if adjusted["end"] > adjusted["start"]:
                sliced_notes.append(adjusted)

    return {
        "piece_id": piece["piece_id"],
        "source_file": piece.get("source_file", ""),
        "duration": end - start,
        "instrument_count": piece.get("instrument_count", 0),
        "notes": sliced_notes,
    }


def make_window_sample(
    piece: dict[str, Any],
    start: float,
    end: float,
    quantum_ms: int,
    max_length: int,
    min_duration: float,
    depth: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return accepted samples and rejected diagnostics for this interval."""
    interval_piece = slice_piece(piece, start, end)
    if not interval_piece["notes"]:
        return [], [
            {
                "piece_id": piece["piece_id"],
                "start_time": start,
                "end_time": end,
                "reason": "empty",
            }
        ]

    sample = piece_to_sample(interval_piece, quantum_ms)
    melody_count = sample["melody_note_count"]
    accompaniment_count = sample["accompaniment_note_count"]

    if melody_count == 0 or accompaniment_count == 0:
        return [], [
            {
                "piece_id": piece["piece_id"],
                "start_time": start,
                "end_time": end,
                "reason": "missing_source_or_target",
                "melody_note_count": melody_count,
                "accompaniment_note_count": accompaniment_count,
            }
        ]

    if sample["total_length"] <= max_length:
        sample.update(
            {
                "slice_id": None,
                "start_time": start,
                "end_time": end,
                "window_duration": end - start,
                "source_piece_id": piece["piece_id"],
            }
        )
        return [sample], []

    duration = end - start
    if duration <= min_duration or depth >= 8:
        return [], [
            {
                "piece_id": piece["piece_id"],
                "start_time": start,
                "end_time": end,
                "reason": "too_long",
                "total_length": sample["total_length"],
                "melody_note_count": melody_count,
                "accompaniment_note_count": accompaniment_count,
            }
        ]

    mid = start + duration / 2.0
    left_samples, left_rejects = make_window_sample(
        piece, start, mid, quantum_ms, max_length, min_duration, depth + 1
    )
    right_samples, right_rejects = make_window_sample(
        piece, mid, end, quantum_ms, max_length, min_duration, depth + 1
    )
    return left_samples + right_samples, left_rejects + right_rejects


def build_windows_for_piece(
    piece: dict[str, Any],
    quantum_ms: int,
    window_seconds: float,
    overlap_seconds: float,
    max_length: int,
    min_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duration = float(piece.get("duration", 0.0))
    if duration <= 0:
        return [], [{"piece_id": piece.get("piece_id", "unknown"), "reason": "zero_duration"}]

    step = window_seconds - overlap_seconds
    if step <= 0:
        raise ValueError("window_seconds must be greater than overlap_seconds")

    samples: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        accepted, rejected = make_window_sample(piece, start, end, quantum_ms, max_length, min_duration)
        samples.extend(accepted)
        rejects.extend(rejected)
        if end >= duration:
            break
        start += step

    for index, sample in enumerate(samples):
        sample["piece_id"] = f"{piece['piece_id']}__slice_{index:04d}"
        sample["slice_id"] = index
    return samples, rejects


def write_preview_midis(samples: list[dict[str, Any]], out_dir: Path, quantum_ms: int, limit: int) -> None:
    preview_dir = out_dir / "preview_midi"
    for sample in samples[:limit]:
        notes = tokens_to_notes(sample["source_tokens"] + sample["target_tokens"], quantum_ms)
        write_notes_midi(notes, preview_dir / f"{sample['piece_id']}.mid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short training windows from annotated note JSON.")
    parser.add_argument("--annotated-dir", default="data/dp_cleaned_v1/annotated_notes")
    parser.add_argument("--out-dir", default="data/training_windows_v1")
    parser.add_argument("--quantum-ms", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--preview-limit", type=int, default=20)
    args = parser.parse_args()

    annotated_dir = ROOT / args.annotated_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(annotated_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No annotated note JSON files found in {annotated_dir}. "
            "Run scripts/dp_melody_cleaning_v1.py first."
        )

    all_samples: list[dict[str, Any]] = []
    all_rejects: list[dict[str, Any]] = []
    piece_reports: list[dict[str, Any]] = []

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            piece = json.load(handle)
        samples, rejects = build_windows_for_piece(
            piece,
            args.quantum_ms,
            args.window_seconds,
            args.overlap_seconds,
            args.max_length,
            args.min_duration,
        )
        all_samples.extend(samples)
        all_rejects.extend(rejects)
        piece_reports.append(
            {
                "piece_id": piece["piece_id"],
                "duration": piece.get("duration", 0),
                "accepted_windows": len(samples),
                "rejected_windows": len(rejects),
                "max_window_length": max((sample["total_length"] for sample in samples), default=0),
                "avg_window_length": math.floor(sum(sample["total_length"] for sample in samples) / len(samples))
                if samples
                else 0,
            }
        )
        print(f"OK {piece['piece_id']}: accepted={len(samples)} rejected={len(rejects)}")

    dataset = {
        "metadata": {
            "source": str(annotated_dir),
            "tokenizer_version": "pianalysis_numeric_v1",
            "format": "[BOS] source_melody [SEP] target_accompaniment [EOS]",
            "quantum_ms": args.quantum_ms,
            "window_seconds": args.window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "max_length": args.max_length,
            "min_duration": args.min_duration,
            "sample_count": len(all_samples),
            "rejected_count": len(all_rejects),
            "token_ids": TOKEN,
            "loss_rule": "labels before target_start_index and padding positions must be -100",
        },
        "samples": all_samples,
    }
    with (out_dir / "dataset_windows_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    lengths = [sample["total_length"] for sample in all_samples]
    report = {
        "processed_pieces": len(files),
        "accepted_windows": len(all_samples),
        "rejected_windows": len(all_rejects),
        "avg_window_length": math.floor(sum(lengths) / len(lengths)) if lengths else 0,
        "max_window_length": max(lengths, default=0),
        "min_window_length": min(lengths, default=0),
        "piece_reports": piece_reports,
        "rejects": all_rejects,
    }
    with (out_dir / "window_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    write_preview_midis(all_samples, out_dir, args.quantum_ms, args.preview_limit)

    print(f"Wrote {out_dir / 'dataset_windows_v1.json'}")
    print(f"Wrote {out_dir / 'window_report.json'}")


if __name__ == "__main__":
    main()
