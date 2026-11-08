"""Beat-relative rhythm events projected from reviewed PIANO notes."""

from collections import Counter
import math
import statistics

import local_voicing as lv

VERSION = "local_rhythm_v4"
PAD, BOS, SEP, EOS, PREVIOUS, MELODY, MAJOR, MINOR = range(8)
LENGTH, STEP, MASK, DURATION, VELOCITY, ONSETS, RELATIVE_PC = 16, 32, 48, 56, 72, 76, 80
VOCAB = 93
SUBDIVISIONS = 4
MAX_STEPS = 16
VELOCITY_GAINS = (.75, .9, 1.05, 1.2)


def steps_for(label, ppq):
    steps = (label["end_tick"] - label["start_tick"]) * SUBDIVISIONS / ppq
    if abs(steps - round(steps)) > 1e-7 or not 1 <= round(steps) <= MAX_STEPS:
        raise ValueError("Rhythm requires an exact 1/16 grid and at most four quarters per harmony span")
    return round(steps)


def degrees(label):
    result = sorted(label["pitch_classes"], key=lambda p: (p - label["root_pc"]) % 12)
    if len(result) != 3 or label["quality"] not in ("maj", "min"):
        raise ValueError("The first learned-rhythm experiment supports major/minor triads only")
    return result


def melody_tokens(label, melody, ppq):
    result = []
    unit = ppq / SUBDIVISIONS
    for step in range(steps_for(label, ppq)):
        left = label["start_tick"] + step * unit
        right = left + unit
        active = [n for n in melody if n["start_tick"] < right and n["end_tick"] > left]
        onset_count = sum(left <= n["start_tick"] < right for n in melody)
        pc = (max(n["pitch"] for n in active) - label["root_pc"]) % 12 if active else 12
        result.extend([ONSETS + min(3, onset_count), RELATIVE_PC + pc])
    return result


def prompt(label, melody, ppq, previous):
    degrees(label)
    steps = steps_for(label, ppq)
    if any(not 0 <= frame["mask"] <= 7 for frame in previous):
        raise ValueError("Invalid previous rhythm")
    return [BOS, LENGTH + steps - 1, MAJOR if label["quality"] == "maj" else MINOR,
            PREVIOUS, *[MASK + f["mask"] for f in previous[-MAX_STEPS:]], MELODY,
            *melody_tokens(label, melody, ppq), SEP]


def empty_frames(count):
    return [{"mask": 0, "durations": [None] * 3, "velocities": [None] * 3} for _ in range(count)]


def project_reference(label, notes, ppq):
    """保留原PIANO的起音/时值，投影到根、三、五音三条单音声部。"""
    pcs = degrees(label)
    count, origin, end = steps_for(label, ppq), label["start_tick"], label["end_tick"]
    unit = ppq / SUBDIVISIONS
    active = [n for n in notes if n["role"] == "accompaniment" and n["start_tick"] < end and n["end_tick"] > origin]
    bass_candidates = [n["pitch"] for n in active if n["pitch"] % 12 == label["bass_pc"]]
    if not bass_candidates:
        raise ValueError("Reviewed bass has no source evidence")
    bass_pitch = min(bass_candidates)
    frames, events, audit = empty_frames(count), {}, Counter()
    mean_velocity = statistics.mean(n["velocity"] for n in active)
    onset_errors = []
    for note in active:
        audit["source_overlapping_notes"] += 1
        if note["pitch"] == bass_pitch:
            audit["bass_notes_separated"] += 1
            continue
        if note["start_tick"] < origin:
            audit["carried_upper_notes_excluded"] += 1
            continue
        pc = note["pitch"] % 12
        if pc not in pcs:
            audit["nonchord_notes_excluded"] += 1
            continue
        step = round((note["start_tick"] - origin) / unit)
        if step >= count:
            audit["onsets_rounded_to_next_span_excluded"] += 1
            continue
        stop = max(step + 1, min(count, round((min(end, note["end_tick"]) - origin) / unit)))
        lane = pcs.index(pc)
        velocity = min(range(4), key=lambda i: abs(VELOCITY_GAINS[i] - note["velocity"] / mean_velocity))
        onset_errors.append(abs(note["start_tick"] - (origin + step * unit)) / ppq)
        if (step, lane) in events:
            audit["octave_unisons_collapsed"] += 1
            old = events[step, lane]
            stop, velocity = max(stop, old[0]), max(velocity, old[1])
        events[step, lane] = (stop, velocity)
    for lane in range(3):
        onsets = sorted(step for step, voice in events if voice == lane)
        for index, step in enumerate(onsets):
            stop, velocity = events[step, lane]
            next_onset = onsets[index + 1] if index + 1 < len(onsets) else count
            if stop > next_onset:
                audit["same_degree_retrigger_cuts"] += 1
                stop = next_onset
            frames[step]["mask"] |= 1 << lane
            frames[step]["durations"][lane] = stop - step
            frames[step]["velocities"][lane] = velocity
    audit["projected_upper_notes"] = sum(f["mask"].bit_count() for f in frames)
    return frames, {**audit, "max_onset_error_quarters": max(onset_errors, default=0),
                    "bass_pitch_separated": bass_pitch, "target_source": "projected_real_PIANO_notes"}


def training_sample(label, melody, ppq, previous, frames, identity):
    prefix = prompt(label, melody, ppq, previous)
    tokens, labels = list(prefix), [-100] * len(prefix)
    for step, frame in enumerate(frames):
        tokens.append(STEP + step)
        labels.append(-100)  # 已知拍位是条件，不伪装成模型学会预测的节拍。
        values = [MASK + frame["mask"]]
        for lane in range(3):
            if frame["mask"] & (1 << lane):
                duration, velocity = frame["durations"][lane], frame["velocities"][lane]
                if not 1 <= duration <= len(frames) - step or not 0 <= velocity < 4:
                    raise ValueError("Invalid projected rhythm event")
                values.extend([DURATION + duration - 1, VELOCITY + velocity])
        tokens.extend(values)
        labels.extend(values)
    tokens.append(EOS)
    labels.append(EOS)
    if len(tokens) > 256:
        raise ValueError("Rhythm context exceeds the configured model")
    return {"tokens": tokens, "labels": labels, "source_id": identity, "prefix_length": len(prefix)}


def build_dataset(annotation):
    records = []
    for piece in annotation["pieces"]:
        melody = [n for n in piece["notes"] if n["role"] == "melody"]
        previous, previous_end = [], None
        for label in piece["labels"]:
            if label["start_tick"] != previous_end:
                previous = []  # 未审阅区间不凭空连成一段已知的节奏历史。
            frames, audit = project_reference(label, piece["notes"], piece["ppq"])
            identity = f"{piece['song']}:{label['source_label_index']}"
            records.append({"song": piece["song"], "split": piece["split"], "source_id": identity,
                            "label": label, "previous": previous, "frames": frames, "projection_audit": audit,
                            "sample": training_sample(label, melody, piece["ppq"], previous, frames, identity)})
            previous, previous_end = frames, label["end_tick"]
    return {"version": VERSION, "subdivisions_per_quarter": SUBDIVISIONS,
            "target_source": "projected_real_PIANO_notes", "augmentation": "none", "records": records}


def render(label, voicing, frames, ppq, melody_velocity):
    if len(frames) != steps_for(label, ppq):
        raise ValueError("Predicted rhythm has the wrong horizon")
    pcs = degrees(label)
    mapping = {p % 12: p for p in voicing[1:]}
    if set(mapping) != set(pcs):
        raise ValueError("Frozen voicing does not cover the requested triad; do not hide this with pitch correction")
    origin, end = label["start_tick"], label["end_tick"]
    base_velocity = max(16, min(50, round(melody_velocity * .45)))
    notes = [{"pitch": voicing[0], "start_tick": origin, "end_tick": end,
              "velocity": max(20, min(58, round(melody_velocity * .52))), "role": "accompaniment"}]
    for lane, pc in enumerate(pcs):
        onsets = [i for i, f in enumerate(frames) if f["mask"] & (1 << lane)]
        for index, step in enumerate(onsets):
            frame = frames[step]
            duration = frame["durations"][lane]
            velocity = frame["velocities"][lane]
            if duration is None or not 1 <= duration <= len(frames) - step or velocity not in range(4):
                raise ValueError("Generated rhythm event violates its field/horizon")
            next_step = onsets[index + 1] if index + 1 < len(onsets) else len(frames)
            stop = min(step + duration, next_step)
            notes.append({"pitch": mapping[pc], "start_tick": origin + round(step * ppq / SUBDIVISIONS),
                          "end_tick": min(end, origin + round(stop * ppq / SUBDIVISIONS)),
                          "velocity": max(1, min(64, round(base_velocity * VELOCITY_GAINS[velocity]))),
                          "role": "accompaniment"})
    return notes


def onset_metrics(predicted, reference):
    event_set = lambda frames: {(i, lane) for i, f in enumerate(frames) for lane in range(3) if f["mask"] & (1 << lane)}
    a, b = event_set(predicted), event_set(reference)
    intersection = a & b
    errors = [abs(predicted[i]["durations"][lane] - reference[i]["durations"][lane]) / SUBDIVISIONS for i, lane in intersection]
    return {"tp": len(intersection), "fp": len(a - b), "fn": len(b - a),
            "predicted_notes": len(a), "reference_notes": len(b),
            "duration_error_sum_quarters": sum(errors), "duration_matches": len(errors)}


def aggregate_metrics(rows):
    counts = Counter()
    for row in rows:
        counts.update(row)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {**counts, "onset_precision": tp / (tp + fp) if tp + fp else 0,
            "onset_recall": tp / (tp + fn) if tp + fn else 0,
            "onset_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "duration_mae_quarters_on_matched_onsets": counts["duration_error_sum_quarters"] / counts["duration_matches"] if counts["duration_matches"] else None}


def baseline_frames(label, voicing, ppq):
    pcs = degrees(label)
    frames = empty_frames(steps_for(label, ppq))
    for index, step in enumerate(range(0, len(frames), 2)):
        pitch = voicing[1 + (0, 1, 2, 1)[index % 4]]
        lane = pcs.index(pitch % 12)
        frames[step]["mask"] = 1 << lane
        frames[step]["durations"][lane] = min(2, len(frames) - step)
        frames[step]["velocities"][lane] = 1
    return frames
