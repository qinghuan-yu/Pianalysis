#!/usr/bin/env python3
"""Add rule-based harmony labels to annotated note JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from harmony_v1 import label_piece_harmony, token_maps  # noqa: E402


def summarize_labels(labels: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [label for label in labels if label.get("chord") != "NC"]
    confidences = [float(label.get("confidence", 0.0)) for label in labels]
    chord_counts: dict[str, int] = {}
    for label in labels:
        chord = str(label.get("chord", "NC"))
        chord_counts[chord] = chord_counts.get(chord, 0) + 1
    return {
        "segments": len(labels),
        "valid_segments": len(valid),
        "nc_segments": len(labels) - len(valid),
        "valid_ratio": round(len(valid) / len(labels), 4) if labels else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "chord_counts": dict(sorted(chord_counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label annotated note JSON files with coarse harmony.")
    parser.add_argument("--annotated-dir", default="data/dp_cleaned_v1/annotated_notes")
    parser.add_argument("--out-dir", default="data/harmony_labeled_v1")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=float, default=0.42)
    args = parser.parse_args()

    annotated_dir = ROOT / args.annotated_dir
    out_dir = ROOT / args.out_dir
    notes_out = out_dir / "annotated_notes"
    notes_out.mkdir(parents=True, exist_ok=True)

    files = sorted(annotated_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No annotated note JSON files found in {annotated_dir}")

    piece_reports: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            piece = json.load(handle)

        labels = label_piece_harmony(piece, args.segment_seconds, args.min_confidence)
        piece["harmony"] = {
            "version": "harmony_v1",
            "segment_seconds": args.segment_seconds,
            "min_confidence": args.min_confidence,
            "labels": labels,
        }

        out_path = notes_out / path.name
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(piece, handle, ensure_ascii=False, indent=2)

        summary = summarize_labels(labels)
        summary["piece_id"] = piece.get("piece_id", path.stem)
        summary["source_file"] = piece.get("source_file", "")
        piece_reports.append(summary)
        print(
            f"OK {path.name}: segments={summary['segments']} "
            f"valid={summary['valid_segments']} avg_conf={summary['avg_confidence']}"
        )

    aggregate = {
        "source": str(annotated_dir),
        "out_dir": str(out_dir),
        "version": "harmony_v1",
        "segment_seconds": args.segment_seconds,
        "min_confidence": args.min_confidence,
        "token_ids": token_maps(),
        "processed_pieces": len(piece_reports),
        "segments": sum(item["segments"] for item in piece_reports),
        "valid_segments": sum(item["valid_segments"] for item in piece_reports),
        "nc_segments": sum(item["nc_segments"] for item in piece_reports),
        "piece_reports": piece_reports,
    }
    aggregate["valid_ratio"] = (
        round(aggregate["valid_segments"] / aggregate["segments"], 4)
        if aggregate["segments"]
        else 0.0
    )

    with (out_dir / "harmony_report.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {notes_out}")
    print(f"Wrote {out_dir / 'harmony_report.json'}")


if __name__ == "__main__":
    main()
