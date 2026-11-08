"""Import POP909's named tracks into Pianalysis annotated notes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from closed_loop_v1 import load_midi_notes

UPSTREAM_URL = "https://github.com/music-x-lab/POP909-Dataset"


def load_piece(path, dataset_root):
    path, dataset_root = Path(path).resolve(), Path(dataset_root).resolve()
    relative = path.relative_to(dataset_root)
    if len(relative.parts) < 2 or not relative.parts[0].isdigit():
        raise ValueError("Expected POP909/<numeric song ID>/<song or version>.mid")
    piece = load_midi_notes(path)
    selected = []
    excluded_bridge = 0
    for note in piece["notes"]:
        name = note["instrument_name"].strip().upper()
        if name == "BRIDGE":
            excluded_bridge += 1
            continue
        if name not in ("MELODY", "PIANO"):
            raise ValueError(f"Unrecognized POP909 track {name!r}; expected MELODY, BRIDGE, PIANO")
        note["role"] = "melody" if name == "MELODY" else "accompaniment"
        selected.append(note)
    for role, track_name in (("melody", "MELODY"), ("accompaniment", "PIANO")):
        if not any(note["role"] == role for note in selected):
            raise ValueError(f"Missing nonempty {track_name} track in {path.name}")
    # 同一首歌的不同编配版本共享划分身份，避免训练/验证泄漏。
    piece["source_piece_id"] = f"pop909_{relative.parts[0]}"
    piece["piece_id"] = "pop909_" + "__".join(relative.with_suffix("").parts)
    piece["notes"] = selected
    for index, note in enumerate(selected):
        note["id"] = index
    piece["duration"] = max(note["end"] for note in selected)
    song_dir = dataset_root / relative.parts[0]
    piece["import_info"] = {
        "dataset": "POP909", "upstream": UPSTREAM_URL, "relative_file": relative.as_posix(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_source": "named_tracks", "bridge_policy": "exclude",
        "excluded_bridge_notes": excluded_bridge,
        "sidecars": {name: str(song_dir / name) for name in ("beat_midi.txt", "chord_midi.txt", "key_audio.txt") if (song_dir / name).exists()},
    }
    return piece


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Directory containing POP909's numbered song folders")
    parser.add_argument("--out-dir", default="data/pop909_imported")
    parser.add_argument("--include-versions", action="store_true")
    args = parser.parse_args()
    root, out_dir = Path(args.dataset_dir), Path(args.out_dir)
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in (".mid", ".midi")
                   and (args.include_versions or "versions" not in path.relative_to(root).parts))
    if not files:
        raise SystemExit("No POP909 MIDI files found")
    notes_dir = out_dir / "annotated_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for path in files:
        try:
            piece = load_piece(path, root)
            destination = notes_dir / f"{piece['piece_id']}.json"
            destination.write_text(json.dumps(piece, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({"piece_id": piece["piece_id"], "source_piece_id": piece["source_piece_id"],
                            "notes": len(piece["notes"]), "excluded_bridge_notes": piece["import_info"]["excluded_bridge_notes"]})
            print(f"OK {piece['piece_id']}: {len(piece['notes'])} notes")
        except (ValueError, OSError) as exc:
            failures.append({"file": str(path), "error": str(exc)})
    report = {"upstream": UPSTREAM_URL, "bridge_policy": "exclude", "include_versions": args.include_versions,
              "processed": len(records), "songs": len({r['source_piece_id'] for r in records}),
              "files": records, "failures": failures}
    (out_dir / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Imported {len(records)} files; failed {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
