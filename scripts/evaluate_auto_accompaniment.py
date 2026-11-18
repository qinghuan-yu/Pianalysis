"""Evaluate symbolic harmony plans and accompaniment integrity without listening claims.

The canonical schema uses quarter-note beats. Missing predictions and predicted
unknown spans fail against evaluable reference time. Reference unknown spans are
masked explicitly. Texture metrics are diagnostics, not an aesthetic score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


CANONICAL_SHA256 = "bf3e8705a2b977eccb32d94e54d6130d4c3e0168c73cf35d86c4965658299d60"
BOUNDARY_TOLERANCE_BEATS = 0.25
EPS = 1e-8
NOTE_FIELDS = ("pitch", "start", "end", "velocity")
HARMONY_METRICS = ("root_accuracy", "pitch_classes_accuracy", "quality_accuracy", "bass_accuracy", "no_chord_accuracy")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pc(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 12


def _duration(song: dict) -> float:
    duration = song.get("duration_beats")
    if not _finite(duration) or duration <= 0:
        raise ValueError("Reference duration_beats must be finite and positive")
    return float(duration)


def _metric(correct: float, total: float) -> dict:
    return {"correct_beats": correct, "reference_beats": total,
            "accuracy": correct / total if total > 0 else None}


def _load_chords(song: dict | None, duration: float) -> tuple[list[dict], list[str]]:
    if song is None:
        return [], []
    events = song.get("chords")
    if not isinstance(events, list):
        return [], ["chords_must_be_a_list"]
    output, errors = [], []
    for index, chord in enumerate(events):
        if not isinstance(chord, dict):
            errors.append(f"chord_{index}:not_an_object")
            continue
        start, end = chord.get("start"), chord.get("end")
        if not _finite(start) or not _finite(end) or start < 0 or end <= start or end > duration + EPS:
            errors.append(f"chord_{index}:invalid_interval")
            continue
        status = chord.get("status")
        if status not in ("known", "unknown", "no_chord"):
            errors.append(f"chord_{index}:invalid_status")
            continue
        pcs = chord.get("pitch_classes", [])
        if not isinstance(pcs, list) or any(not _pc(p) for p in pcs):
            errors.append(f"chord_{index}:invalid_pitch_classes")
            continue
        root, bass = chord.get("root"), chord.get("bass")
        if root is not None and not _pc(root):
            errors.append(f"chord_{index}:invalid_root")
            continue
        if bass is not None and not _pc(bass):
            errors.append(f"chord_{index}:invalid_bass")
            continue
        if status == "no_chord" and (root is not None or bass is not None or pcs):
            errors.append(f"chord_{index}:no_chord_has_pitches")
            continue
        if status == "known" and not pcs:
            errors.append(f"chord_{index}:known_chord_has_no_pitch_classes")
            continue
        output.append({**chord, "start": float(start), "end": min(float(end), duration),
                       "root": root, "bass": bass, "pitch_classes": sorted(set(pcs))})
    output.sort(key=lambda c: (c["start"], c["end"]))
    for before, after in zip(output, output[1:]):
        if after["start"] < before["end"] - EPS:
            errors.append("overlapping_chord_intervals")
            break
    return output, errors


def _quality(chord: dict | None) -> tuple[int, ...] | None:
    if chord is None or chord["status"] != "known" or not _pc(chord.get("root")):
        return None
    pcs = chord.get("pitch_classes", [])
    if not pcs:
        return None
    return tuple(sorted({(pitch - chord["root"]) % 12 for pitch in pcs}))


def _signature(chord: dict) -> tuple:
    if chord["status"] != "known":
        return (chord["status"],)
    return (chord["status"], chord["root"], tuple(chord["pitch_classes"]), chord["bass"])


def _at(events: list[dict], time: float) -> dict | None:
    # 评估端只在边界邻域调用；主积分使用单调游标避免逐段全表扫描。
    for event in events:
        if event["start"] <= time < event["end"]:
            return event
        if event["start"] > time:
            break
    return None


def _evaluable(chord: dict | None) -> bool:
    return chord is not None and chord["status"] in ("known", "no_chord")


def _boundaries(events: list[dict]) -> list[float]:
    values = []
    for previous, current in zip(events, events[1:]):
        if abs(previous["end"] - current["start"]) > EPS:
            continue
        if _evaluable(previous) and _evaluable(current) and _signature(previous) != _signature(current):
            values.append(current["start"])
    return values


def _boundary_scores(tp: int, fp: int, fn: int, complete_empty_case: bool = False) -> dict:
    precision = tp / (tp + fp) if tp + fp else (1.0 if complete_empty_case else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if complete_empty_case else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (1.0 if complete_empty_case else 0.0)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _match_boundaries(reference: list[float], predicted: list[float]) -> tuple[int, int, int, list[dict]]:
    left = right = 0
    matches = []
    # 排序后的容差区间用双指针做一对一最大基数匹配，预测边界不可重复领分。
    while left < len(reference) and right < len(predicted):
        ref, pred = reference[left], predicted[right]
        if pred < ref - BOUNDARY_TOLERANCE_BEATS - EPS:
            right += 1
        elif ref < pred - BOUNDARY_TOLERANCE_BEATS - EPS:
            left += 1
        else:
            matches.append({"reference": ref, "prediction": pred, "absolute_error_beats": abs(ref - pred)})
            left += 1
            right += 1
    tp = len(matches)
    return tp, len(predicted) - tp, len(reference) - tp, matches


def evaluate_harmony(reference_song: dict, predicted_song: dict | None) -> dict:
    """Compare harmony over reference time; missing/unknown predictions score zero."""
    duration = _duration(reference_song)
    reference, reference_errors = _load_chords(reference_song, duration)
    if reference_errors:
        raise ValueError("Invalid reference chords: " + "; ".join(reference_errors))
    predicted, errors = _load_chords(predicted_song, duration)
    if predicted_song is not None and predicted_song.get("song_id") != reference_song.get("song_id"):
        errors.append("song_id_mismatch")
    prediction_present = predicted_song is not None
    prediction_valid = prediction_present and not errors
    # 不允许通过舍弃非法重叠事件提高分数；结构不合法的预测全段计失败。
    scoring_prediction = predicted if prediction_valid else []
    points = sorted({0.0, duration} | {c[k] for c in reference + scoring_prediction for k in ("start", "end")})
    correct, denominators = Counter(), Counter()
    coverage = Counter()
    quality_uncomparable = 0.0
    root_unavailable = bass_unavailable = 0.0
    r_index = p_index = 0
    for start, end in zip(points, points[1:]):
        length = end - start
        if length <= 0:
            continue
        mid = (start + end) / 2
        while r_index < len(reference) and reference[r_index]["end"] <= mid:
            r_index += 1
        while p_index < len(scoring_prediction) and scoring_prediction[p_index]["end"] <= mid:
            p_index += 1
        ref = reference[r_index] if r_index < len(reference) and reference[r_index]["start"] <= mid else None
        pred = scoring_prediction[p_index] if p_index < len(scoring_prediction) and scoring_prediction[p_index]["start"] <= mid else None
        if ref is None:
            coverage["reference_missing_beats"] += length
            continue
        if ref["status"] == "unknown":
            coverage["reference_unknown_beats"] += length
            continue
        coverage["reference_evaluable_beats"] += length
        coverage[f"reference_{ref['status']}_beats"] += length
        if pred is None:
            coverage["prediction_missing_on_evaluable_beats"] += length
        elif pred["status"] == "unknown":
            coverage["prediction_unknown_on_evaluable_beats"] += length
        else:
            coverage["prediction_evaluable_on_evaluable_beats"] += length
        if ref["status"] == "no_chord":
            denominators["no_chord_accuracy"] += length
            if pred is not None and pred["status"] == "no_chord":
                correct["no_chord_accuracy"] += length
            continue
        known_prediction = pred is not None and pred["status"] == "known"
        if _pc(ref["root"]):
            denominators["root_accuracy"] += length
            if known_prediction and pred["root"] == ref["root"]:
                correct["root_accuracy"] += length
        else:
            root_unavailable += length
        denominators["pitch_classes_accuracy"] += length
        if known_prediction and pred["pitch_classes"] == ref["pitch_classes"]:
            correct["pitch_classes_accuracy"] += length
        quality = _quality(ref)
        if quality is not None:
            denominators["quality_accuracy"] += length
            if known_prediction and _quality(pred) == quality:
                correct["quality_accuracy"] += length
        else:
            quality_uncomparable += length
        if _pc(ref["bass"]):
            denominators["bass_accuracy"] += length
            if known_prediction and pred["bass"] == ref["bass"]:
                correct["bass_accuracy"] += length
        else:
            bass_unavailable += length
    coverage_keys = ("reference_missing_beats", "reference_unknown_beats", "reference_evaluable_beats",
                     "reference_known_beats", "reference_no_chord_beats",
                     "prediction_missing_on_evaluable_beats", "prediction_unknown_on_evaluable_beats",
                     "prediction_evaluable_on_evaluable_beats")
    coverage = {k: float(coverage[k]) for k in coverage_keys}
    coverage["reference_duration_beats"] = duration
    coverage["prediction_evaluable_fraction"] = (
        coverage["prediction_evaluable_on_evaluable_beats"] / coverage["reference_evaluable_beats"]
        if coverage["reference_evaluable_beats"] else None)
    reference_boundaries = _boundaries(reference)
    predicted_candidates = _boundaries(predicted) if prediction_valid else []
    prediction_boundaries, ignored = [], []
    for beat in predicted_candidates:
        if _evaluable(_at(reference, beat - EPS)) and _evaluable(_at(reference, beat + EPS)):
            prediction_boundaries.append(beat)
        else:
            ignored.append(beat)
    tp, fp, fn, matches = _match_boundaries(reference_boundaries, prediction_boundaries)
    complete_empty = prediction_valid and coverage["reference_evaluable_beats"] > 0 and (
        coverage["prediction_evaluable_on_evaluable_beats"] >= coverage["reference_evaluable_beats"] - EPS)
    boundaries = {**_boundary_scores(tp, fp, fn, complete_empty),
                  "tolerance_beats": BOUNDARY_TOLERANCE_BEATS,
                  "reference_count": len(reference_boundaries), "prediction_count": len(prediction_boundaries),
                  "ignored_prediction_boundaries_in_reference_unknown": len(ignored),
                  "matches": matches}
    return {"prediction_present": prediction_present, "prediction_valid": prediction_valid,
            "errors": errors, "coverage": coverage,
            "metrics": {key: _metric(correct[key], denominators[key]) for key in HARMONY_METRICS},
            "quality_comparison": "exact root-relative pitch-class set; quality label spelling including custom is ignored",
            "reference_quality_uncomparable_beats": quality_uncomparable,
            "reference_root_unavailable_beats": root_unavailable,
            "reference_bass_unavailable_beats": bass_unavailable,
            "boundaries": boundaries}


def _notes(song: dict | None, role: str, duration: float) -> tuple[list[dict], list[str]]:
    if song is None:
        return [], []
    events = song.get(role)
    if not isinstance(events, list):
        return [], [f"{role}_must_be_a_list"]
    valid, errors = [], []
    for index, note in enumerate(events):
        if not isinstance(note, dict) or any(field not in note for field in NOTE_FIELDS):
            errors.append(f"{role}_{index}:missing_fields")
            continue
        pitch, velocity = note["pitch"], note["velocity"]
        if not isinstance(pitch, int) or isinstance(pitch, bool) or not 0 <= pitch <= 127:
            errors.append(f"{role}_{index}:invalid_pitch")
            continue
        if not isinstance(velocity, int) or isinstance(velocity, bool) or not 1 <= velocity <= 127:
            errors.append(f"{role}_{index}:invalid_velocity")
            continue
        start, end = note["start"], note["end"]
        if not _finite(start) or not _finite(end) or not 0 <= start < end <= duration + EPS:
            errors.append(f"{role}_{index}:invalid_interval")
            continue
        valid.append({field: note[field] for field in NOTE_FIELDS})
    return valid, errors


def _union(notes: list[dict]) -> list[tuple[float, float]]:
    result = []
    for start, end in sorted((n["start"], n["end"]) for n in notes):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _intersection_length(a: list[tuple], b: list[tuple]) -> float:
    left = right = 0
    overlap = 0.0
    while left < len(a) and right < len(b):
        overlap += max(0.0, min(a[left][1], b[right][1]) - max(a[left][0], b[right][0]))
        if a[left][1] < b[right][1]:
            left += 1
        else:
            right += 1
    return overlap


def _diagnostics(notes: list[dict], duration: float) -> dict:
    groups = defaultdict(list)
    for note in notes:
        groups[round(note["start"], 8)].append(note["pitch"])
    bass = [min(pitches) for _, pitches in sorted(groups.items())]
    leaps = [abs(after - before) for before, after in zip(bass, bass[1:])]
    pitches = [n["pitch"] for n in notes]
    intervals = _union(notes)
    active = sum(end - start for start, end in intervals)
    duplicates = sum(count - 1 for count in Counter(
        (n["pitch"], round(n["start"], 8)) for n in notes).values() if count > 1)
    exact_duplicates = sum(count - 1 for count in Counter(
        tuple(n[field] for field in NOTE_FIELDS) for n in notes).values() if count > 1)
    return {"note_count": len(notes), "onset_group_count": len(groups),
            "notes_per_quarter_beat": len(notes) / duration,
            "first_onset_beat": min((n["start"] for n in notes), default=None),
            "last_offset_beat": max((n["end"] for n in notes), default=None),
            "active_beats": active, "active_fraction": active / duration,
            "pitch_min": min(pitches) if pitches else None,
            "pitch_max": max(pitches) if pitches else None,
            "pitch_median": statistics.median(pitches) if pitches else None,
            "outside_88_key_piano_note_count": sum(p < 21 or p > 108 for p in pitches),
            "duplicate_pitch_onset_events": duplicates,
            "exact_duplicate_note_events": exact_duplicates,
            "duplicate_onset_rounding_decimals": 8,
            "bass_onset_leaps": {"definition": "successive lowest pitches among note-onset groups, not inferred voices",
                                  "count": len(leaps), "mean_semitones": statistics.mean(leaps) if leaps else None,
                                  "max_semitones": max(leaps) if leaps else None,
                                  "over_octave_count": sum(leap > 12 for leap in leaps),
                                  "over_octave_fraction": sum(leap > 12 for leap in leaps) / len(leaps) if leaps else None}}


def evaluate_texture(reference_song: dict, predicted_song: dict | None) -> dict:
    """Check melody preservation and MIDI events; report texture diagnostics only."""
    duration = _duration(reference_song)
    reference_melody, ref_melody_errors = _notes(reference_song, "melody", duration)
    reference_notes, ref_note_errors = _notes(reference_song, "accompaniment", duration)
    if ref_melody_errors or ref_note_errors:
        raise ValueError("Invalid reference notes: " + "; ".join(ref_melody_errors + ref_note_errors))
    predicted_melody, melody_errors = _notes(predicted_song, "melody", duration)
    predicted_notes, note_errors = _notes(predicted_song, "accompaniment", duration)
    reference_counts = Counter(tuple(n[k] for k in NOTE_FIELDS) for n in reference_melody)
    predicted_counts = Counter(tuple(n[k] for k in NOTE_FIELDS) for n in predicted_melody)
    reference_diag, predicted_diag = _diagnostics(reference_notes, duration), _diagnostics(predicted_notes, duration)
    present = predicted_song is not None
    declared_duration = predicted_song.get("duration_beats") if present else None
    checks = {"prediction_present": present,
              "song_id_matches": present and predicted_song.get("song_id") == reference_song.get("song_id"),
              "melody_preserved_exactly": present and not melody_errors and reference_counts == predicted_counts,
              "events_legal": present and not melody_errors and not note_errors,
              "duration_preserved": _finite(declared_duration) and abs(declared_duration - duration) <= EPS,
              "no_duplicate_pitch_onset": present and predicted_diag["duplicate_pitch_onset_events"] == 0,
              "nonempty_when_reference_has_accompaniment": present and (not reference_notes or bool(predicted_notes))}
    ref_active = reference_diag["active_beats"]
    overlap = _intersection_length(_union(reference_notes), _union(predicted_notes))
    return {"applicable": bool(reference_notes), "hard_checks": checks,
            "hard_checks_pass": all(checks.values()),
            "melody_difference": {"missing_or_changed_note_events": sum((reference_counts - predicted_counts).values()),
                                  "added_or_changed_note_events": sum((predicted_counts - reference_counts).values())},
            "errors": melody_errors + note_errors,
            "coverage_diagnostics": {"reference_active_beats": ref_active,
                                     "predicted_active_beats": predicted_diag["active_beats"],
                                     "intersection_active_beats": overlap,
                                     "reference_active_time_recall": overlap / ref_active if ref_active else None},
            "reference_diagnostics": reference_diag, "prediction_diagnostics": predicted_diag,
            "aesthetic_quality_assessed": False,
            "interpretation": "Density, register, activity coverage and bass leaps describe events; they do not score musical naturalness, harmony or style."}


def _aggregate(rows: list[dict]) -> dict:
    totals = {}
    for key in HARMONY_METRICS:
        totals[key] = _metric(sum(r["harmony"]["metrics"][key]["correct_beats"] for r in rows),
                              sum(r["harmony"]["metrics"][key]["reference_beats"] for r in rows))
    tp = sum(r["harmony"]["boundaries"]["true_positive"] for r in rows)
    fp = sum(r["harmony"]["boundaries"]["false_positive"] for r in rows)
    fn = sum(r["harmony"]["boundaries"]["false_negative"] for r in rows)
    complete_empty = bool(rows) and all(r["harmony"]["boundaries"]["f1"] == 1 for r in rows)
    texture_rows = [r for r in rows if r["texture"]["applicable"]]
    return {"song_count": len(rows), "predicted_song_count": sum(r["prediction_status"] == "available" for r in rows),
            "harmony": totals, "boundaries": _boundary_scores(tp, fp, fn, complete_empty),
            "texture": {"applicable_song_count": len(texture_rows),
                        "hard_pass_song_count": sum(r["texture"]["hard_checks_pass"] for r in texture_rows),
                        "hard_pass_fraction": (sum(r["texture"]["hard_checks_pass"] for r in texture_rows) / len(texture_rows)
                                               if texture_rows else None),
                        "failed_check_counts": dict(Counter(k for r in texture_rows for k, value in r["texture"]["hard_checks"].items() if not value))}}


def _records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected an object")
                    yield value
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, (list, dict)):
            raise ValueError("Expected JSON song object or list")
        values = value if isinstance(value, list) else value.get("predictions", [value])
        if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
            raise ValueError("Expected a song object, list of song objects, or predictions list")
        yield from values


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def evaluate_files(reference_path: Path, prediction_path: Path, splits: set[str]) -> dict:
    """Evaluate all selected references, including references with no prediction."""
    reference_hash = _file_hash(reference_path)
    prediction_hash = _file_hash(prediction_path)
    predictions = {}
    duplicates = set()
    prediction_count = 0
    for item in _records(prediction_path):
        song_id = item.get("song_id")
        if not isinstance(song_id, str) or not song_id:
            raise ValueError("Every prediction must have a nonempty song_id")
        prediction_count += 1
        if song_id in predictions:
            duplicates.add(song_id)
        predictions[song_id] = item
    rows, selected_ids = [], set()
    for reference in _records(reference_path):
        if reference.get("split") not in splits:
            continue
        song_id = reference.get("song_id")
        if not isinstance(song_id, str) or not song_id or song_id in selected_ids:
            raise ValueError("Reference song_id must be unique and nonempty")
        selected_ids.add(song_id)
        prediction = predictions.get(song_id) if song_id not in duplicates else None
        status = "duplicate_rejected" if song_id in duplicates else "available" if prediction is not None else "missing"
        rows.append({"song_id": song_id, "work_id": reference.get("work_id") or song_id,
                     "split": reference["split"], "source": reference.get("source"),
                     "prediction_status": status,
                     "harmony": evaluate_harmony(reference, prediction),
                     "texture": evaluate_texture(reference, prediction)})
    if not rows:
        raise ValueError("No reference songs matched the requested evaluation split")
    if _file_hash(reference_path) != reference_hash or _file_hash(prediction_path) != prediction_hash:
        raise ValueError("Evaluation inputs changed during evaluation")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["work_id"]].append(row)
    works = [{"work_id": work, "song_ids": [r["song_id"] for r in group], **_aggregate(group)}
             for work, group in sorted(grouped.items())]
    macro = {}
    for key in HARMONY_METRICS:
        values = [w["harmony"][key]["accuracy"] for w in works if w["harmony"][key]["accuracy"] is not None]
        macro[key] = {"mean_accuracy": statistics.mean(values) if values else None, "evaluable_work_count": len(values)}
    missing = [r["song_id"] for r in rows if r["prediction_status"] != "available"]
    sources = {str(r["source"]) for r in rows}
    return {"schema_version": 1, "evaluator": "auto_accompaniment_symbolic_v1",
            "provenance": {"reference_path": str(reference_path.resolve()),
                           "reference_sha256": reference_hash, "frozen_canonical_sha256": CANONICAL_SHA256,
                           "reference_matches_frozen_canonical": reference_hash == CANONICAL_SHA256,
                           "predictions_path": str(prediction_path.resolve()), "predictions_sha256": prediction_hash,
                           "evaluator_sha256": _file_hash(Path(__file__)),
                           "selected_splits": sorted(splits)},
            "coverage": {"reference_song_count": len(rows), "prediction_record_count": prediction_count,
                         "matched_unique_prediction_count": len(rows) - len(missing),
                         "prediction_fraction": (len(rows) - len(missing)) / len(rows),
                         "missing_or_duplicate_prediction_song_ids": missing,
                         "duplicate_prediction_song_ids": sorted(duplicates),
                         "prediction_song_ids_outside_selected_reference": sorted(set(predictions) - selected_ids),
                         "reference_work_count": len(works)},
            "micro_time_weighted": _aggregate(rows), "macro_work_harmony": macro,
            "per_source": {source: _aggregate([r for r in rows if str(r["source"]) == source]) for source in sorted(sources)},
            "per_split": {split: _aggregate([r for r in rows if r["split"] == split]) for split in sorted(splits)},
            "macro_work_boundary_f1": statistics.mean(w["boundaries"]["f1"] for w in works),
            "per_work": works, "per_song": rows,
            "listening_performed": False, "aesthetic_quality_assessed": False,
            "metric_policy": {"reference_unknown": "masked and reported, including missing reference intervals",
                              "predicted_unknown_or_missing": "failure on every evaluable reference beat",
                              "invalid_prediction": "all harmony time fails; no invalid event is silently discarded for a higher score",
                              "quality": "compare exact root-relative pitch-class sets, not raw quality names",
                              "no_chord": "only explicit no_chord; evaluated separately from pitch-bearing chords",
                              "boundary": "semantic chord changes only, one-to-one matching within 0.25 quarter-note beats; unknown reference regions excluded",
                              "texture": "hard invariants and descriptive diagnostics only; no chord-tone or aesthetic score",
                              "missing_predictions": "retained as failed rows and included in song/work aggregation"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"),
                        help="Defaults to both validation and test; training rows are never evaluated by this CLI")
    args = parser.parse_args()
    if args.output.resolve() in (args.reference.resolve(), args.predictions.resolve()):
        parser.error("--output must not overwrite either input")
    report = evaluate_files(args.reference, args.predictions, {args.split} if args.split else {"validation", "test"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "coverage": {
        key: report["coverage"][key] for key in ("reference_song_count", "matched_unique_prediction_count", "prediction_fraction")},
        "harmony": report["micro_time_weighted"]["harmony"],
        "texture": report["micro_time_weighted"]["texture"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
