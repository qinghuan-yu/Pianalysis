#!/usr/bin/env python3
"""
Minimal closed-loop MIDI pipeline for Pianalysis.

This script intentionally avoids model training. It validates the data path that
must be trustworthy before training:

MIDI -> note JSON -> conditional tokens -> MIDI reconstruction.

The target sequence is accompaniment-only. At inference time the original input
melody should be merged with generated accompaniment, so the model does not need
to rewrite the melody.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import pretty_midi  # noqa: E402


TOKEN = {
    "PAD": 0,
    "BOS": 1,
    "SEP": 2,
    "EOS": 3,
    "TIME": 4,
    "NOTE_ON_MELODY": 10,
    "NOTE_OFF_MELODY": 11,
    "NOTE_ON_ACCOMP": 20,
    "NOTE_OFF_ACCOMP": 21,
}


def quantize_time(seconds: float, quantum_ms: int) -> int:
    return max(0, int(round(seconds * 1000.0 / quantum_ms)))


def tick_to_seconds(tick: int, quantum_ms: int) -> float:
    return tick * quantum_ms / 1000.0


def load_midi_notes(midi_path: Path) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[dict[str, Any]] = []
    note_id = 0

    for track_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if note.end <= note.start:
                continue

            notes.append(
                {
                    "id": note_id,
                    "track": track_index,
                    "program": int(instrument.program),
                    "instrument_name": instrument.name or f"track_{track_index}",
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "role": "unknown",
                }
            )
            note_id += 1

    notes.sort(key=lambda item: (item["start"], item["pitch"], item["end"]))

    return {
        "piece_id": midi_path.stem,
        "source_file": str(midi_path.as_posix()),
        "duration": float(midi.get_end_time()),
        "instrument_count": len(midi.instruments),
        "notes": notes,
    }


def apply_skyline_roles(piece: dict[str, Any], time_window: float = 0.05) -> None:
    """Assign initial melody/accompaniment roles using skyline as a weak label."""
    notes = piece["notes"]
    for note in notes:
        note["role"] = "accompaniment"

    if not notes:
        return

    max_time = max(note["end"] for note in notes)
    current = 0.0
    melody_ids: set[int] = set()

    while current < max_time:
        end = current + time_window
        active = [
            note
            for note in notes
            if note["start"] < end and note["end"] > current
        ]
        if active:
            max_pitch = max(note["pitch"] for note in active)
            for note in active:
                if note["pitch"] == max_pitch:
                    melody_ids.add(note["id"])
        current = end

    for note in notes:
        if note["id"] in melody_ids:
            note["role"] = "melody"


def make_events(notes: list[dict[str, Any]], role: str, quantum_ms: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for note in notes:
        if note["role"] != role:
            continue

        start_tick = quantize_time(note["start"], quantum_ms)
        end_tick = max(start_tick + 1, quantize_time(note["end"], quantum_ms))
        velocity = max(1, min(127, int(note.get("velocity", 80))))

        events.append(
            {
                "tick": start_tick,
                "kind": "on",
                "pitch": int(note["pitch"]),
                "velocity": velocity,
                "role": role,
            }
        )
        events.append(
            {
                "tick": end_tick,
                "kind": "off",
                "pitch": int(note["pitch"]),
                "velocity": 0,
                "role": role,
            }
        )

    # Close notes before opening new notes at the same tick to reduce overlaps.
    events.sort(key=lambda event: (event["tick"], 0 if event["kind"] == "off" else 1, event["pitch"]))
    return events


def events_to_tokens(events: list[dict[str, Any]], quantum_ms: int) -> list[int]:
    del quantum_ms
    tokens: list[int] = []
    current_tick = 0

    for event in events:
        delta = event["tick"] - current_tick
        if delta > 0:
            tokens.extend([TOKEN["TIME"], delta])
            current_tick = event["tick"]

        if event["role"] == "melody":
            if event["kind"] == "on":
                tokens.extend([TOKEN["NOTE_ON_MELODY"], event["pitch"], event["velocity"]])
            else:
                tokens.extend([TOKEN["NOTE_OFF_MELODY"], event["pitch"]])
        else:
            if event["kind"] == "on":
                tokens.extend([TOKEN["NOTE_ON_ACCOMP"], event["pitch"], event["velocity"]])
            else:
                tokens.extend([TOKEN["NOTE_OFF_ACCOMP"], event["pitch"]])

    return tokens


def piece_to_sample(piece: dict[str, Any], quantum_ms: int) -> dict[str, Any]:
    notes = piece["notes"]
    source = events_to_tokens(make_events(notes, "melody", quantum_ms), quantum_ms)
    target = events_to_tokens(make_events(notes, "accompaniment", quantum_ms), quantum_ms)
    training_sequence = [TOKEN["BOS"]] + source + [TOKEN["SEP"]] + target + [TOKEN["EOS"]]
    target_start_index = len(source) + 2

    return {
        "piece_id": piece["piece_id"],
        "source_file": piece["source_file"],
        "duration": piece["duration"],
        "quantum_ms": quantum_ms,
        "source_tokens": source,
        "target_tokens": target,
        "training_sequence": training_sequence,
        "target_start_index": target_start_index,
        "source_length": len(source),
        "target_length": len(target),
        "total_length": len(training_sequence),
        "note_count": len(notes),
        "melody_note_count": sum(1 for note in notes if note["role"] == "melody"),
        "accompaniment_note_count": sum(1 for note in notes if note["role"] == "accompaniment"),
    }


def tokens_to_notes(tokens: list[int], quantum_ms: int) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    active: dict[tuple[str, int], list[dict[str, Any]]] = {}
    current_tick = 0
    note_id = 0
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in (TOKEN["PAD"], TOKEN["BOS"], TOKEN["SEP"], TOKEN["EOS"]):
            i += 1
            continue

        if token == TOKEN["TIME"]:
            if i + 1 >= len(tokens):
                break
            delta = tokens[i + 1]
            if delta < 0:
                delta = 0
            current_tick += delta
            i += 2
            continue

        if token in (TOKEN["NOTE_ON_MELODY"], TOKEN["NOTE_ON_ACCOMP"]):
            if i + 2 >= len(tokens):
                break
            role = "melody" if token == TOKEN["NOTE_ON_MELODY"] else "accompaniment"
            pitch = int(tokens[i + 1])
            velocity = max(1, min(127, int(tokens[i + 2])))
            active.setdefault((role, pitch), []).append(
                {"start_tick": current_tick, "velocity": velocity}
            )
            i += 3
            continue

        if token in (TOKEN["NOTE_OFF_MELODY"], TOKEN["NOTE_OFF_ACCOMP"]):
            if i + 1 >= len(tokens):
                break
            role = "melody" if token == TOKEN["NOTE_OFF_MELODY"] else "accompaniment"
            pitch = int(tokens[i + 1])
            stack = active.get((role, pitch), [])
            if stack:
                started = stack.pop(0)
                start_tick = started["start_tick"]
                end_tick = max(start_tick + 1, current_tick)
                notes.append(
                    {
                        "id": note_id,
                        "start": tick_to_seconds(start_tick, quantum_ms),
                        "end": tick_to_seconds(end_tick, quantum_ms),
                        "pitch": pitch,
                        "velocity": started["velocity"],
                        "role": role,
                    }
                )
                note_id += 1
            i += 2
            continue

        i += 1

    for (role, pitch), stack in active.items():
        for started in stack:
            start_tick = started["start_tick"]
            end_tick = max(start_tick + 1, current_tick)
            notes.append(
                {
                    "id": note_id,
                    "start": tick_to_seconds(start_tick, quantum_ms),
                    "end": tick_to_seconds(end_tick, quantum_ms),
                    "pitch": pitch,
                    "velocity": started["velocity"],
                    "role": role,
                }
            )
            note_id += 1

    notes.sort(key=lambda note: (note["start"], note["pitch"], note["end"]))
    return notes


def write_notes_midi(notes: list[dict[str, Any]], out_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    melody = pretty_midi.Instrument(program=0, name="Melody")
    accompaniment = pretty_midi.Instrument(program=0, name="Accompaniment")

    for item in notes:
        note = pretty_midi.Note(
            velocity=int(item.get("velocity", 80)),
            pitch=int(item["pitch"]),
            start=float(item["start"]),
            end=max(float(item["end"]), float(item["start"]) + 0.01),
        )
        if item.get("role") == "melody":
            melody.notes.append(note)
        else:
            accompaniment.notes.append(note)

    if melody.notes:
        midi.instruments.append(melody)
    if accompaniment.notes:
        midi.instruments.append(accompaniment)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def build_report(piece: dict[str, Any], sample: dict[str, Any], reconstructed: list[dict[str, Any]]) -> dict[str, Any]:
    original_counts = {
        "melody": sample["melody_note_count"],
        "accompaniment": sample["accompaniment_note_count"],
        "total": sample["note_count"],
    }
    reconstructed_counts = {
        "melody": sum(1 for note in reconstructed if note["role"] == "melody"),
        "accompaniment": sum(1 for note in reconstructed if note["role"] == "accompaniment"),
        "total": len(reconstructed),
    }
    return {
        "piece_id": piece["piece_id"],
        "source_file": piece["source_file"],
        "duration": piece["duration"],
        "original_counts": original_counts,
        "reconstructed_counts": reconstructed_counts,
        "source_length": sample["source_length"],
        "target_length": sample["target_length"],
        "total_length": sample["total_length"],
        "target_start_index": sample["target_start_index"],
        "closed_loop_note_count_match": original_counts["total"] == reconstructed_counts["total"],
    }


def process_file(midi_path: Path, out_dir: Path, quantum_ms: int, write_midi: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    piece = load_midi_notes(midi_path)
    apply_skyline_roles(piece)
    sample = piece_to_sample(piece, quantum_ms)
    reconstructed_notes = tokens_to_notes(sample["source_tokens"] + sample["target_tokens"], quantum_ms)
    report = build_report(piece, sample, reconstructed_notes)

    notes_dir = out_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    with (notes_dir / f"{piece['piece_id']}.json").open("w", encoding="utf-8") as handle:
        json.dump(piece, handle, ensure_ascii=False, indent=2)

    if write_midi:
        write_notes_midi(reconstructed_notes, out_dir / "roundtrip_midi" / f"{piece['piece_id']}_roundtrip.mid")
        accompaniment_notes = tokens_to_notes(sample["target_tokens"], quantum_ms)
        write_notes_midi(accompaniment_notes, out_dir / "target_accompaniment_midi" / f"{piece['piece_id']}_accompaniment.mid")

    return sample, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIDI -> token -> MIDI closed-loop validation.")
    parser.add_argument("--midi-dir", default="MIDI", help="Directory containing MIDI files.")
    parser.add_argument("--out-dir", default="data/closed_loop_v1", help="Output directory.")
    parser.add_argument("--quantum-ms", type=int, default=10, help="Time quantization in milliseconds.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of MIDI files, 0 means all.")
    parser.add_argument("--write-midi", action="store_true", help="Write reconstructed MIDI files.")
    args = parser.parse_args()

    midi_dir = ROOT / args.midi_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = (
        list(midi_dir.glob("*.mid"))
        + list(midi_dir.glob("*.midi"))
        + list(midi_dir.glob("*.MID"))
        + list(midi_dir.glob("*.MIDI"))
    )
    by_path = {str(path.resolve()).lower(): path for path in discovered}
    midi_files = sorted(by_path.values(), key=lambda path: path.name.lower())
    if args.limit > 0:
        midi_files = midi_files[: args.limit]

    if not midi_files:
        raise SystemExit(f"No MIDI files found in {midi_dir}")

    samples: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for midi_path in midi_files:
        try:
            sample, report = process_file(midi_path, out_dir, args.quantum_ms, args.write_midi)
            samples.append(sample)
            reports.append(report)
            print(
                f"OK {midi_path.name}: notes={sample['note_count']} "
                f"melody={sample['melody_note_count']} accomp={sample['accompaniment_note_count']} "
                f"tokens={sample['total_length']}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(midi_path), "error": str(exc)})
            print(f"FAIL {midi_path.name}: {exc}")

    dataset = {
        "metadata": {
            "tokenizer_version": "pianalysis_numeric_v1",
            "format": "[BOS] source_melody [SEP] target_accompaniment [EOS]",
            "quantum_ms": args.quantum_ms,
            "sample_count": len(samples),
            "failure_count": len(failures),
            "token_ids": TOKEN,
            "loss_rule": "labels before target_start_index and padding positions must be -100",
        },
        "samples": samples,
    }

    with (out_dir / "dataset_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    summary = {
        "midi_dir": str(midi_dir),
        "out_dir": str(out_dir),
        "quantum_ms": args.quantum_ms,
        "processed": len(samples),
        "failed": failures,
        "reports": reports,
        "avg_total_length": math.floor(sum(item["total_length"] for item in samples) / len(samples)) if samples else 0,
        "max_total_length": max((item["total_length"] for item in samples), default=0),
    }
    with (out_dir / "roundtrip_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {out_dir / 'dataset_v1.json'}")
    print(f"Wrote {out_dir / 'roundtrip_report.json'}")


if __name__ == "__main__":
    main()
