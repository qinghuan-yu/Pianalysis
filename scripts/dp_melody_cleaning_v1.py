#!/usr/bin/env python3
"""
Enhanced Skyline + dynamic programming melody annotation.

This is a data-cleaning step, not a model. It turns raw piano MIDI files into
weakly annotated melody/accompaniment data that can be reviewed in vue-piano or
used for a first training experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import (  # noqa: E402
    TOKEN,
    load_midi_notes,
    piece_to_sample,
    tick_to_seconds,
    tokens_to_notes,
    write_notes_midi,
)


@dataclass(frozen=True)
class Candidate:
    note_id: int
    onset_tick: int
    start: float
    end: float
    pitch: int
    velocity: int
    duration: float
    local_score: float
    rank_from_top: int
    chord_size: int


def quantize_onset(seconds: float, quantum_ms: int) -> int:
    return max(0, int(round(seconds * 1000.0 / quantum_ms)))


def group_notes_by_onset(notes: list[dict[str, Any]], quantum_ms: int) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        tick = quantize_onset(note["start"], quantum_ms)
        groups.setdefault(tick, []).append(note)
    return dict(sorted(groups.items()))


def metric_weight(seconds: float) -> float:
    """A light metrical prior using common 0.5s/1.0s grid hints."""
    # MIDI files in this dataset often lack reliable meter extraction. This
    # weak prior only helps tied scores and should not dominate pitch/continuity.
    frac_1s = abs(seconds - round(seconds))
    frac_05s = abs((seconds * 2.0) - round(seconds * 2.0)) / 2.0
    if frac_1s < 0.025:
        return 0.18
    if frac_05s < 0.025:
        return 0.10
    return 0.0


def candidate_local_score(note: dict[str, Any], rank_from_top: int, chord_size: int) -> float:
    pitch = int(note["pitch"])
    velocity = int(note.get("velocity", 80))
    duration = max(0.0, float(note["end"]) - float(note["start"]))

    pitch_height = (pitch - 21) / (108 - 21)
    duration_score = min(duration / 0.75, 1.2)
    velocity_score = velocity / 127.0
    rank_score = 1.0 / (rank_from_top + 1)
    density_penalty = min(max(chord_size - 1, 0) * 0.08, 0.45)

    score = 0.95 * pitch_height
    score += 0.62 * duration_score
    score += 0.22 * velocity_score
    score += 0.48 * rank_score
    score += metric_weight(float(note["start"]))
    score -= density_penalty

    if duration < 0.055:
        score -= 1.20
    elif duration < 0.10:
        score -= 0.55
    elif duration < 0.16:
        score -= 0.20

    if pitch < 48:
        score -= 0.70
    elif pitch < 55:
        score -= 0.30

    return score


def build_candidates(
    notes: list[dict[str, Any]],
    quantum_ms: int,
    top_k: int,
    min_local_score: float,
) -> tuple[list[int], dict[int, list[Candidate]]]:
    groups = group_notes_by_onset(notes, quantum_ms)
    candidates_by_tick: dict[int, list[Candidate]] = {}

    for tick, onset_notes in groups.items():
        sorted_notes = sorted(onset_notes, key=lambda item: (-int(item["pitch"]), -float(item["end"])))
        selected = sorted_notes[:top_k]
        candidates: list[Candidate] = []
        for rank, note in enumerate(selected):
            score = candidate_local_score(note, rank, len(onset_notes))
            if score < min_local_score:
                continue
            candidates.append(
                Candidate(
                    note_id=int(note["id"]),
                    onset_tick=tick,
                    start=float(note["start"]),
                    end=float(note["end"]),
                    pitch=int(note["pitch"]),
                    velocity=int(note.get("velocity", 80)),
                    duration=max(0.0, float(note["end"]) - float(note["start"])),
                    local_score=score,
                    rank_from_top=rank,
                    chord_size=len(onset_notes),
                )
            )
        candidates_by_tick[tick] = candidates

    return list(groups.keys()), candidates_by_tick


def transition_score(prev: Candidate | None, cur: Candidate | None) -> float:
    if cur is None:
        return 0.0
    if prev is None:
        return 0.0

    interval = abs(cur.pitch - prev.pitch)
    onset_gap = max(0.0, cur.start - prev.start)
    rest_gap = max(0.0, cur.start - prev.end)

    score = 0.0
    score -= 0.055 * interval

    if interval > 12:
        score -= 0.45
    if interval > 19:
        score -= 0.85
    if interval <= 2 and onset_gap <= 1.2:
        score += 0.18
    if interval <= 5 and onset_gap <= 1.2:
        score += 0.12
    if onset_gap < 0.08 and interval > 7:
        score -= 0.50
    if rest_gap > 2.5:
        score -= 0.15
    if cur.start < prev.end and interval > 0:
        score -= 0.18

    return score


def select_melody_path(
    notes: list[dict[str, Any]],
    quantum_ms: int,
    top_k: int,
    min_local_score: float,
) -> tuple[set[int], dict[str, Any]]:
    ticks, candidates_by_tick = build_candidates(notes, quantum_ms, top_k, min_local_score)
    if not ticks:
        return set(), {"selected": 0, "candidate_count": 0}

    # Each layer has state 0 = skip, states 1..n = candidates.
    prev_scores: list[float] = [0.0]
    prev_states: list[Candidate | None] = [None]
    backptrs: list[list[int]] = []
    state_layers: list[list[Candidate | None]] = []

    candidate_count = 0
    for tick in ticks:
        candidates = candidates_by_tick.get(tick, [])
        candidate_count += len(candidates)
        states: list[Candidate | None] = [None] + candidates
        state_layers.append(states)
        scores: list[float] = []
        ptrs: list[int] = []

        for state in states:
            local = 0.0 if state is None else state.local_score
            if state is None:
                # Skipping is allowed, but a long all-skip path should not win
                # over a coherent melody path when candidates are plausible.
                local = -0.015

            best_score = -1e12
            best_prev = 0
            for prev_index, prev_state in enumerate(prev_states):
                score = prev_scores[prev_index] + local + transition_score(prev_state, state)
                if score > best_score:
                    best_score = score
                    best_prev = prev_index
            scores.append(best_score)
            ptrs.append(best_prev)

        prev_scores = scores
        prev_states = states
        backptrs.append(ptrs)

    best_index = max(range(len(prev_scores)), key=lambda idx: prev_scores[idx])
    selected: list[Candidate] = []

    for layer_index in range(len(state_layers) - 1, -1, -1):
        state = state_layers[layer_index][best_index]
        if state is not None:
            selected.append(state)
        best_index = backptrs[layer_index][best_index]

    selected.reverse()
    selected_ids = postprocess_selected(notes, selected)

    diagnostics = {
        "selected": len(selected_ids),
        "candidate_count": candidate_count,
        "onset_count": len(ticks),
        "top_k": top_k,
        "min_local_score": min_local_score,
    }
    return selected_ids, diagnostics


def postprocess_selected(notes: list[dict[str, Any]], selected: list[Candidate]) -> set[int]:
    if not selected:
        return set()

    by_id = {int(note["id"]): note for note in notes}
    keep: list[Candidate] = []

    for index, cur in enumerate(selected):
        prev = selected[index - 1] if index > 0 else None
        nxt = selected[index + 1] if index + 1 < len(selected) else None

        isolated_high_jump = False
        if cur.duration < 0.14 and prev and nxt:
            if abs(cur.pitch - prev.pitch) > 12 and abs(cur.pitch - nxt.pitch) > 12:
                isolated_high_jump = True
        if isolated_high_jump:
            continue

        keep.append(cur)

    selected_ids = {item.note_id for item in keep}

    # Collapse same-onset octave melodies to a single note for conditioning.
    # Keep the upper note by default, because the generated accompaniment will
    # be merged with the original melody later.
    by_onset: dict[int, list[Candidate]] = {}
    for item in keep:
        by_onset.setdefault(item.onset_tick, []).append(item)
    for same_onset in by_onset.values():
        if len(same_onset) <= 1:
            continue
        sorted_items = sorted(same_onset, key=lambda item: item.pitch, reverse=True)
        top = sorted_items[0]
        for item in sorted_items[1:]:
            if abs(top.pitch - item.pitch) == 12:
                selected_ids.discard(item.note_id)

    # Guard against a pathological empty result.
    if not selected_ids:
        best = max(notes, key=lambda note: (int(note["pitch"]), float(note["end"]) - float(note["start"])))
        selected_ids.add(int(best["id"]))

    # Keep ids valid after postprocessing.
    return {note_id for note_id in selected_ids if note_id in by_id}


def annotate_piece(piece: dict[str, Any], quantum_ms: int, top_k: int, min_local_score: float) -> dict[str, Any]:
    selected_ids, diagnostics = select_melody_path(piece["notes"], quantum_ms, top_k, min_local_score)
    for note in piece["notes"]:
        note["role"] = "melody" if int(note["id"]) in selected_ids else "accompaniment"
        note["annotation_method"] = "enhanced_skyline_dp_v1"
    piece["annotation"] = {
        "method": "enhanced_skyline_dp_v1",
        "quantum_ms": quantum_ms,
        **diagnostics,
    }
    return piece


def write_melody_and_accomp(piece: dict[str, Any], out_dir: Path) -> None:
    melody_notes = [note for note in piece["notes"] if note["role"] == "melody"]
    accompaniment_notes = [note for note in piece["notes"] if note["role"] == "accompaniment"]
    write_notes_midi(melody_notes, out_dir / "melody_midi" / f"{piece['piece_id']}_melody.mid")
    write_notes_midi(accompaniment_notes, out_dir / "accompaniment_midi" / f"{piece['piece_id']}_accompaniment.mid")
    write_notes_midi(piece["notes"], out_dir / "annotated_midi" / f"{piece['piece_id']}_annotated.mid")


def process_file(
    midi_path: Path,
    out_dir: Path,
    quantum_ms: int,
    top_k: int,
    min_local_score: float,
    write_midi: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    piece = load_midi_notes(midi_path)
    annotate_piece(piece, quantum_ms, top_k, min_local_score)
    sample = piece_to_sample(piece, quantum_ms)

    out_notes = out_dir / "annotated_notes"
    out_notes.mkdir(parents=True, exist_ok=True)
    with (out_notes / f"{piece['piece_id']}.json").open("w", encoding="utf-8") as handle:
        json.dump(piece, handle, ensure_ascii=False, indent=2)

    if write_midi:
        write_melody_and_accomp(piece, out_dir)
        reconstructed = tokens_to_notes(sample["source_tokens"] + sample["target_tokens"], quantum_ms)
        write_notes_midi(reconstructed, out_dir / "roundtrip_midi" / f"{piece['piece_id']}_roundtrip.mid")

    report = {
        "piece_id": piece["piece_id"],
        "source_file": piece["source_file"],
        "duration": piece["duration"],
        "note_count": sample["note_count"],
        "melody_note_count": sample["melody_note_count"],
        "accompaniment_note_count": sample["accompaniment_note_count"],
        "melody_ratio": sample["melody_note_count"] / sample["note_count"] if sample["note_count"] else 0,
        "source_length": sample["source_length"],
        "target_length": sample["target_length"],
        "total_length": sample["total_length"],
        "target_start_index": sample["target_start_index"],
        "annotation": piece["annotation"],
    }
    return sample, report


def discover_midi_files(midi_dir: Path) -> list[Path]:
    discovered = (
        list(midi_dir.glob("*.mid"))
        + list(midi_dir.glob("*.midi"))
        + list(midi_dir.glob("*.MID"))
        + list(midi_dir.glob("*.MIDI"))
    )
    by_path = {str(path.resolve()).lower(): path for path in discovered}
    return sorted(by_path.values(), key=lambda path: path.name.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MIDI dataset using enhanced Skyline + DP melody annotation.")
    parser.add_argument("--midi-dir", default="MIDI")
    parser.add_argument("--out-dir", default="data/dp_cleaned_v1")
    parser.add_argument("--quantum-ms", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-local-score", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-midi", action="store_true")
    args = parser.parse_args()

    midi_dir = ROOT / args.midi_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_files = discover_midi_files(midi_dir)
    if args.limit > 0:
        midi_files = midi_files[: args.limit]
    if not midi_files:
        raise SystemExit(f"No MIDI files found in {midi_dir}")

    samples: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for midi_path in midi_files:
        try:
            sample, report = process_file(
                midi_path,
                out_dir,
                args.quantum_ms,
                args.top_k,
                args.min_local_score,
                args.write_midi,
            )
            samples.append(sample)
            reports.append(report)
            print(
                f"OK {midi_path.name}: notes={sample['note_count']} "
                f"melody={sample['melody_note_count']} ratio={report['melody_ratio']:.2%} "
                f"tokens={sample['total_length']}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(midi_path), "error": str(exc)})
            print(f"FAIL {midi_path.name}: {exc}")

    dataset = {
        "metadata": {
            "tokenizer_version": "pianalysis_numeric_v1",
            "annotation_method": "enhanced_skyline_dp_v1",
            "format": "[BOS] source_melody [SEP] target_accompaniment [EOS]",
            "quantum_ms": args.quantum_ms,
            "top_k": args.top_k,
            "min_local_score": args.min_local_score,
            "sample_count": len(samples),
            "failure_count": len(failures),
            "token_ids": TOKEN,
            "loss_rule": "labels before target_start_index and padding positions must be -100",
        },
        "samples": samples,
    }
    with (out_dir / "dataset_dp_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)

    ratios = [item["melody_ratio"] for item in reports]
    summary = {
        "midi_dir": str(midi_dir),
        "out_dir": str(out_dir),
        "processed": len(samples),
        "failed": failures,
        "avg_total_length": math.floor(sum(item["total_length"] for item in samples) / len(samples)) if samples else 0,
        "max_total_length": max((item["total_length"] for item in samples), default=0),
        "avg_melody_ratio": sum(ratios) / len(ratios) if ratios else 0,
        "min_melody_ratio": min(ratios) if ratios else 0,
        "max_melody_ratio": max(ratios) if ratios else 0,
        "reports": reports,
    }
    with (out_dir / "cleaning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {out_dir / 'dataset_dp_v1.json'}")
    print(f"Wrote {out_dir / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
