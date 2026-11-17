"""独立核验标准数据的时间、标签、划分，输出可复核的训练前报告。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_song(song):
    errors = []
    for name in ("song_id", "work_id", "source"):
        if not song.get(name):
            errors.append("missing_" + name)
    if song.get("schema_version") != 1:
        errors.append("schema_version")
    if song.get("split") not in ("train", "validation", "test"):
        errors.append("invalid_split")
    duration = song.get("duration_beats")
    if not finite(duration) or duration <= 0:
        return errors + ["invalid_duration"]
    if not song.get("melody"):
        errors.append("empty_melody")
    for role in ("melody", "accompaniment"):
        notes = song.get(role)
        if not isinstance(notes, list):
            errors.append(role + "_not_list")
            continue
        for note in notes:
            if not all(finite(note.get(k)) for k in ("pitch", "start", "end", "velocity")):
                errors.append(role + "_nonfinite_note")
                continue
            if not (0 <= note["pitch"] <= 127 and int(note["pitch"]) == note["pitch"] and 1 <= note["velocity"] <= 127):
                errors.append(role + "_invalid_pitch_velocity")
            if note["end"] <= note["start"] or note["end"] > duration + 1e-6:
                errors.append(role + "_invalid_note_time")
    previous_end = -math.inf
    chords = song.get("chords")
    if not isinstance(chords, list) or not chords:
        errors.append("missing_chord_timeline")
        chords = []
    for chord in chords:
        if not all(finite(chord.get(k)) for k in ("start", "end")):
            errors.append("nonfinite_chord_time")
            continue
        if chord["end"] <= chord["start"] or chord["end"] > duration + 1e-6:
            errors.append("invalid_chord_time")
        if chord["start"] < previous_end - 1e-6:
            errors.append("overlapping_or_unsorted_chords")
        previous_end = chord["end"]
        if chord.get("status") not in ("known", "unknown", "no_chord"):
            errors.append("invalid_chord_status")
        if not chord.get("provenance"):
            errors.append("missing_chord_provenance")
        if chord.get("status") == "known":
            root = chord.get("root")
            pcs = chord.get("pitch_classes")
            if not isinstance(root, int) or not 0 <= root <= 11 or not chord.get("quality"):
                errors.append("invalid_known_chord")
            if not isinstance(pcs, list) or not pcs or any(not isinstance(pc, int) or not 0 <= pc <= 11 for pc in pcs):
                errors.append("invalid_chord_pitch_classes")
        if chord.get("bass") is not None and (not isinstance(chord["bass"], int) or not 0 <= chord["bass"] <= 11):
            errors.append("invalid_chord_bass")
    for name, field in (("tempos", "beat"), ("time_signatures", "beat")):
        events = song.get(name, [])
        if name == "tempos" and events == [] and not song.get("accompaniment") and song.get("metadata", {}).get("tempo_status") == "unknown":
            # 纯符号和声监督不需要秒坐标；未知速度必须显式声明，不能填假 BPM。
            continue
        if not events or any(not finite(e.get(field)) for e in events):
            errors.append("invalid_" + name)
            continue
        if any(events[i][field] < events[i-1][field] for i in range(1, len(events))):
            errors.append("unsorted_" + name)
    if any(not finite(t.get("bpm")) or t["bpm"] <= 0 for t in song.get("tempos", [])):
        errors.append("invalid_tempo")
    for signature in song.get("time_signatures", []):
        n, d = signature.get("numerator"), signature.get("denominator")
        if not isinstance(n, int) or n <= 0 or d not in (1, 2, 4, 8, 16, 32, 64):
            errors.append("invalid_meter")
    return sorted(set(errors))


def melody_fingerprint(song):
    # 完整旋律的移调不变指纹仅查确定性重复，不冒充全面近似去重。
    notes = sorted(song["melody"], key=lambda n: (n["start"], n["pitch"], n["end"]))
    if len(notes) < 32:
        return None
    first = notes[0]
    values = [(n["pitch"] - first["pitch"], round((n["start"]-first["start"])*96), round((n["end"]-n["start"])*96)) for n in notes]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def audit(path):
    counts, errors, statuses = Counter(), [], Counter()
    works, identities, fingerprints = defaultdict(set), set(), defaultdict(list)
    examples = []
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for row_number, raw in enumerate(source, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                song = json.loads(raw.decode("utf-8"))
                problems = validate_song(song)
            except (ValueError, TypeError, KeyError) as error:
                errors.append({"line": row_number, "errors": ["parse_or_schema:" + str(error)]})
                continue
            identity = song.get("song_id")
            if identity in identities:
                problems.append("duplicate_song_id")
            identities.add(identity)
            if problems:
                errors.append({"line": row_number, "song_id": identity, "errors": problems})
                continue
            counts[f"{song['source']}:{song['split']}"] += 1
            works[song["work_id"]].add(song["split"])
            fp = melody_fingerprint(song)
            if fp:
                fingerprints[fp].append((identity, song["split"]))
            for chord in song["chords"]:
                statuses[chord["status"]] += chord["end"] - chord["start"]
            if len(examples) < 18 and song["accompaniment"]:
                examples.append({"song_id": identity, "split": song["split"], "metadata": song.get("metadata", {}), "first_chords": song["chords"][:8]})
    leaks = {work: sorted(splits) for work, splits in works.items() if len(splits) > 1}
    duplicate_leaks = [entries for entries in fingerprints.values() if len({x[1] for x in entries}) > 1]
    return {"schema_version": 1, "samples_path": path.as_posix(), "samples_sha256": digest.hexdigest(),
            "passed": bool(counts) and not errors and not leaks and not duplicate_leaks,
            "song_counts": dict(sorted(counts.items())), "validated_songs": sum(counts.values()),
            "schema_errors": errors, "work_split_leaks": leaks, "exact_transposed_melody_split_leaks": duplicate_leaks,
            "chord_duration_quarter_notes_by_status": dict(statuses), "examples": examples,
            "scope": "Independent structural validation. Does not establish musical correctness, human listening approval, absence of approximate duplicates, or unseen pretraining status."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/auto_accompaniment/processed/samples.jsonl")
    parser.add_argument("--out-dir", default="data/auto_accompaniment/verification")
    args = parser.parse_args()
    report = audit(Path(args.samples))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {k: v for k, v in report.items() if k != "examples"}
    cards = "".join("<section><h2>" + html.escape(str(e["song_id"])) + "</h2><pre>" + html.escape(json.dumps(e, ensure_ascii=False, indent=2)) + "</pre></section>" for e in report["examples"])
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>训练数据核验</title><style>body{max-width:1000px;margin:32px auto;font:15px/1.6 system-ui;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f6;padding:16px}section{border-top:1px solid #ccd}</style><h1>训练数据核验</h1><p>本页记录时间与标签结构检查，不代表人工听审。原始来源、未知标签和例外保留在报告中。</p><pre>" + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre>" + cards + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("passed", "validated_songs", "song_counts")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("Data gate failed; see validation.json before training")


if __name__ == "__main__":
    main()
