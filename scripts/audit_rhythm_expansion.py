"""Build source-evidence candidates; this does not grant symbolic or listening acceptance."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from build_harmony_review import write_json
from build_trusted_harmony import evidence
import harmony_plan as hp
from import_pop909 import load_piece
import local_voicing as lv
import rhythm_v4 as rv


def main():
    root = Path("data/pop909_sample")
    results = []
    for number in range(5, 13):
        song = f"{number:03d}"
        folder = root / song
        midi = hp.pretty_midi.PrettyMIDI(str(folder / f"{song}.mid"))
        piece = load_piece(folder / f"{song}.mid", root)
        labels = hp.read_chord_sidecar(folder / "chord_midi.txt", midi)
        beats = [float(line.split()[0]) for line in (folder / "beat_midi.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        notes = [{**n, "start_tick": int(midi.time_to_tick(n["start"])), "end_tick": int(midi.time_to_tick(n["end"]))} for n in piece["notes"]]
        melody = [n for n in notes if n["role"] == "melody"]
        candidates, rejected = [], []
        for index, label in enumerate(labels):
            reasons = []
            if label["status"] != "chord" or label.get("quality") not in ("maj", "min"):
                reasons.append("outside_major_minor_triads")
            if midi.tick_to_time(label["start_tick"]) >= 90:
                reasons.append("outside_first_90_seconds")
            if reasons:
                rejected.append({"index": index, "symbol": label["symbol"], "reasons": reasons})
                continue
            proof = evidence(piece, midi, label, beats)
            reasons += [k for k, v in proof["checks"].items() if not v]
            active = [n for n in melody if n["start_tick"] < label["end_tick"] and n["end_tick"] > label["start_tick"]]
            if not active:
                reasons.append("no_named_melody")
            try:
                rv.steps_for(label, midi.resolution)
                summary = lv.melody_summary(melody, label)
                lv.teacher_voicing(label, summary, lv.INITIAL)
                frames, projection = rv.project_reference(label, notes, midi.resolution)
                if projection["max_onset_error_quarters"] > .0625:
                    reasons.append("off_grid_onsets_over_1_64_quarter")
            except ValueError as error:
                reasons.append(str(error))
            if reasons:
                rejected.append({"index": index, "symbol": label["symbol"], "reasons": reasons})
            else:
                candidates.append({"index": index, "symbol": label["symbol"], "label": label,
                                   "evidence": proof, "melody": summary, "projection": projection,
                                   "onsets_per_quarter": projection["projected_upper_notes"] / (len(frames) / 4)})
        hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in (f"{song}.mid", "chord_midi.txt", "beat_midi.txt")}
        record = {"song": song, "split": "train" if number <= 10 else "validation" if number == 11 else "test",
                  "hashes": hashes, "status": "candidates_pending_symbolic_review", "candidates": candidates, "rejected": rejected}
        results.append(record)
        print(json.dumps({"song": song, "eligible": len(candidates), "first16": [
            {"i": c["index"], "chord": c["symbol"], "seconds": round(c["evidence"]["start_seconds"], 2),
             "fit": round(c["evidence"]["chord_fit"], 4), "pcs": {k: round(v, 2) for k, v in c["evidence"]["pitch_class_note_seconds"].items()},
             "bass_kind": c["evidence"]["bass_anchor_kind"], "melody_min": c["melody"]["min"], "density": c["onsets_per_quarter"]}
             for c in candidates[:16]], "rejection_reasons": dict(Counter(reason for r in rejected for reason in r["reasons"]))}, ensure_ascii=False))
    write_json(Path("data/rhythm_v5_audit/candidates.json"), {"selection": "first_16_eligible_per_song_within_first_90_seconds", "pieces": results})


if __name__ == "__main__":
    main()
