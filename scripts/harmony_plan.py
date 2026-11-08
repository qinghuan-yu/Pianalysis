"""Timed harmony annotations and deterministic block-chord review tools."""

import copy
from bisect import bisect_right
from itertools import product
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps" / "harmony"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from mir_eval import chord as mir_chord
import pretty_midi
from chorder import Dechorder
import miditoolkit
import mido
import numpy as np


def parse_symbol(symbol):
    symbol = str(symbol).strip()
    if symbol in ("N", "X"):
        return {"symbol": symbol, "status": "no_chord" if symbol == "N" else "unknown",
                "root_pc": None, "quality": None, "bass_pc": None, "pitch_classes": []}
    try:
        root, relative_chroma, bass_degree = mir_chord.encode(symbol)
        quality = mir_chord.split(symbol)[1]
    except (ValueError, TypeError, mir_chord.InvalidChordException) as exc:
        raise ValueError(f"Invalid Harte chord symbol: {symbol}") from exc
    return {"symbol": symbol, "status": "chord", "root_pc": int(root), "quality": quality,
            "bass_pc": int((root + bass_degree) % 12),
            "pitch_classes": sorted((int(root) + i) % 12 for i, value in enumerate(relative_chroma) if value)}


def read_chord_sidecar(path, midi):
    labels = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        start, end, symbol = line.split()
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f"Invalid sidecar interval on line {index}")
        labels.append({**parse_symbol(symbol), "start_tick": int(midi.time_to_tick(start)),
                       "end_tick": int(midi.time_to_tick(end)), "source": "pop909_chord_midi",
                       "source_seconds": [start, end], "review_status": "unreviewed",
                       "confidence": None})
    return labels


def midi_timeline(midi):
    times, tempi = midi.get_tempo_changes()
    point = lambda time: {"tick": int(midi.time_to_tick(float(time))), "seconds": float(time)}
    return {"ppq": int(midi.resolution), "duration_ticks": int(midi.time_to_tick(midi.get_end_time())),
            "tempo_changes": [{**point(t), "bpm": float(b)} for t, b in zip(times, tempi)],
            "time_signatures": [{**point(e.time), "numerator": e.numerator, "denominator": e.denominator}
                                for e in midi.time_signature_changes],
            "key_signatures": [{**point(e.time), "key_number": e.key_number} for e in midi.key_signature_changes],
            "beats": [point(t) for t in midi.get_beats()],
            "downbeats": [point(t) for t in midi.get_downbeats()],
            "beat_source": "pretty_midi_tempo_meter_grid",
            "meter_assumed_4_4": not bool(midi.time_signature_changes),
            "review_status": "unreviewed", "warning": "MIDI grid is not verified score alignment"}


def inspect_midi_metadata(path):
    source = mido.MidiFile(path)
    primary, others = {}, []
    for track_index, track in enumerate(source.tracks):
        tick = 0
        for event in track:
            tick += event.time
            if event.type not in ("set_tempo", "time_signature", "key_signature"):
                continue
            value = {k: v for k, v in event.dict().items() if k != "time"}
            if track_index == 0:
                primary[tick, event.type] = value
            else:
                others.append((track_index, tick, event.type, value))
    duplicates, conflicts = 0, []
    for track, tick, kind, value in others:
        reference = primary.get((tick, kind))
        if reference == value:
            duplicates += 1
        else:
            conflicts.append({"track": track, "tick": tick, "event_type": kind,
                              "track_zero_value": reference, "other_track_value": value})
    return {"duplicate_events": duplicates, "conflicts": conflicts,
            "timing_conflict": any(e["event_type"] in ("set_tempo", "time_signature") for e in conflicts),
            "policy": "Preserve track-zero interpretation; unresolved conflicts require review"}


def validate_plan(plan):
    if plan.get("version") != "harmony_plan_v2" or plan.get("ppq", 0) <= 0 or plan.get("duration_ticks", 0) <= 0:
        raise ValueError("Invalid harmony plan version, ppq or duration")
    previous = 0
    for label in plan["labels"]:
        start, end = label["start_tick"], label["end_tick"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= plan["duration_ticks"]:
            raise ValueError("Invalid harmony tick interval")
        if start < previous:
            raise ValueError("Harmony intervals overlap or are not ordered")
        parsed = parse_symbol(label["symbol"])
        for key in ("status", "root_pc", "bass_pc", "pitch_classes"):
            if label.get(key) != parsed[key]:
                raise ValueError(f"Harmony {key} disagrees with chord symbol {label['symbol']}")
        previous = end


def slice_plan(plan, start_tick, end_tick):
    validate_plan(plan)
    if not 0 <= start_tick < end_tick <= plan["duration_ticks"]:
        raise ValueError("Invalid harmony slice interval")
    result = {**copy.deepcopy(plan), "origin_tick": plan.get("origin_tick", 0) + start_tick,
              "duration_ticks": end_tick - start_tick, "labels": []}
    for label in plan["labels"]:
        if label["start_tick"] < end_tick and label["end_tick"] > start_tick:
            result["labels"].append({**copy.deepcopy(label),
                                     "start_tick": max(start_tick, label["start_tick"]) - start_tick,
                                     "end_tick": min(end_tick, label["end_tick"]) - start_tick})
    validate_plan(result)
    return result


def block_chord_notes(plan):
    validate_plan(plan)
    notes, previous = [], [55, 60, 64]
    for index, label in enumerate(plan["labels"]):
        if label["status"] != "chord":
            continue
        choices = [[p for p in range(48, 72) if p % 12 == pc] for pc in label["pitch_classes"]]
        voicings = [sorted(v) for v in product(*choices)]
        # 保留全部和弦内音，选相邻和弦间移动较小的紧凑配位。
        def cost(voicing):
            return sum(min(abs(p - old) for old in previous) for p in voicing) + 0.3 * (max(voicing) - min(voicing))
        upper = min(voicings, key=lambda v: (cost(v), v))
        bass = 36 + label["bass_pc"]
        for pitch in [bass, *upper]:
            notes.append({"pitch": pitch, "start_tick": label["start_tick"], "end_tick": label["end_tick"],
                          "velocity": 52 if pitch == bass else 45, "role": "accompaniment", "label_index": index})
        previous = upper
    return notes


def write_block_midi(midi, melody_notes, plan, destination):
    if midi.resolution != plan["ppq"] or plan.get("origin_tick", 0):
        raise ValueError("Export requires a global plan matching the source MIDI ppq")
    result = mido.MidiFile(ticks_per_beat=midi.resolution)

    def add_track(events):
        track, previous = mido.MidiTrack(), 0
        for tick, priority, message in sorted(events, key=lambda e: (e[0], e[1])):
            track.append(message.copy(time=tick - previous))
            previous = tick
        result.tracks.append(track)

    metadata = []
    for time, bpm in zip(*midi.get_tempo_changes()):
        # 四舍五入恢复原始整数微秒，避免 pretty_midi 写出时截掉 1 微秒。
        metadata.append((int(midi.time_to_tick(time)), 0, mido.MetaMessage("set_tempo", tempo=round(60_000_000 / bpm))))
    for event in midi.time_signature_changes:
        metadata.append((int(midi.time_to_tick(event.time)), 1, mido.MetaMessage("time_signature", numerator=event.numerator, denominator=event.denominator)))
    for event in midi.key_signature_changes:
        name, mode = pretty_midi.key_number_to_key_name(event.key_number).split()
        metadata.append((int(midi.time_to_tick(event.time)), 2, mido.MetaMessage("key_signature", key=name + ("m" if mode == "minor" else ""))))
    for label in plan["labels"]:
        metadata.append((label["start_tick"], 3, mido.MetaMessage("text", text=label["symbol"])))
    add_track(metadata)
    melody = [{**n, "start_tick": int(midi.time_to_tick(n["start"])), "end_tick": int(midi.time_to_tick(n["end"]))} for n in melody_notes]
    for channel, (name, notes) in enumerate((("MELODY", melody), ("BLOCK_CHORDS", block_chord_notes(plan)))):
        events = [(0, -2, mido.MetaMessage("track_name", name=name))]
        used, active = set(), {}
        channels = [c for c in range(16) if c not in (1, 9)] if name == "MELODY" else [1]
        for note in sorted(notes, key=lambda n: (n["start_tick"], n["pitch"], n["end_tick"])):
            # 相同音高重叠必须分 MIDI channel，避免后一个 note-off 截短前一个音。
            available = [c for c in channels if active.get((c, note["pitch"]), -1) <= note["start_tick"]]
            if not available:
                raise ValueError("Too many overlapping same-pitch melody notes for available MIDI channels")
            voice = available[0]
            active[voice, note["pitch"]] = note["end_tick"]
            used.add(voice)
            events.append((note["start_tick"], 1, mido.Message("note_on", channel=voice, note=int(note["pitch"]), velocity=int(note.get("velocity", 80)))))
            events.append((note["end_tick"], 0, mido.Message("note_off", channel=voice, note=int(note["pitch"]), velocity=0)))
        events.extend((0, -1, mido.Message("program_change", channel=c, program=0)) for c in sorted(used))
        add_track(events)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(destination))
    return pretty_midi.PrettyMIDI(str(destination))


def annotate_accompaniment(piece, midi):
    notes = [n for n in piece["notes"] if n.get("role") == "accompaniment"]
    if not notes:
        raise ValueError("Harmony annotation requires reference accompaniment; melody-only needs a separate plan")
    duration = int(midi.time_to_tick(midi.get_end_time()))
    source = miditoolkit.MidiFile(ticks_per_beat=midi.resolution)
    instrument = miditoolkit.Instrument(0)
    instrument.notes = [miditoolkit.Note(int(n.get("velocity", 80)), int(n["pitch"]),
                                       int(midi.time_to_tick(n["start"])), int(midi.time_to_tick(n["end"]))) for n in notes]
    instrument.notes = [n for n in instrument.notes if n.end > n.start]
    source.instruments = [instrument]
    source.max_tick = duration
    chords = Dechorder.dechord(source)
    quality_map = {"M": "maj", "m": "min", "o": "dim", "+": "aug", "7": "7", "M7": "maj7",
                   "m7": "min7", "o7": "dim7", "/o7": "hdim7", "sus2": "sus2", "sus4": "sus4"}
    names = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    degrees = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")
    labels = []
    for index, start in enumerate(range(0, duration, midi.resolution)):
        end = min(duration, start + midi.resolution)
        active = [n for n in instrument.notes if n.start < end and n.end > start]
        chord = chords[index] if index < len(chords) else None
        symbol = "X" if active else "N"
        if active and chord is not None and chord.root_pc is not None and chord.quality in quality_map:
            symbol = f"{names[chord.root_pc]}:{quality_map[chord.quality]}"
            if chord.bass_pc is not None and chord.bass_pc != chord.root_pc:
                symbol += "/" + degrees[(chord.bass_pc - chord.root_pc) % 12]
        parsed = parse_symbol(symbol)
        weights = [0.0] * 12
        for n in active:
            weights[n.pitch % 12] += min(end, n.end) - max(start, n.start)
        total = sum(weights)
        coverage = sum(weights[pc] for pc in parsed["pitch_classes"]) / total if total else None
        reasons = []
        if parsed["status"] == "unknown":
            reasons.append("unrecognized_harmony")
        if active and sum(x > 0 for x in weights) < 3:
            reasons.append("sparse_pitch_class_evidence")
        if coverage is not None and coverage < 0.7:
            reasons.append("low_chord_tone_coverage")
        labels.append({**parsed, "start_tick": start, "end_tick": end, "source": "chorder_0.1.4",
                       "confidence": None, "review_status": "unreviewed", "review_reasons": reasons,
                       "evidence": {"note_count": len(active), "chord_tone_coverage": coverage,
                                    "pitch_class_weights_ticks": weights,
                                    "persistent_bass_pc": Dechorder.get_bass_pc(active, start, end) if active else None}})
    plan = {"version": "harmony_plan_v2", "piece_id": piece.get("piece_id", "unknown"),
            "ppq": midi.resolution, "duration_ticks": duration,
            "evidence_scope": "reference_accompaniment", "review_status": "unreviewed", "labels": labels}
    validate_plan(plan)
    return plan


def complete_sidecar_plan(piece, midi, path):
    duration = int(midi.time_to_tick(midi.get_end_time()))
    labels, previous = [], 0
    for label in read_chord_sidecar(path, midi):
        if label["start_tick"] < previous:
            raise ValueError("Sidecar chord intervals overlap")
        if label["start_tick"] >= duration:
            break
        if label["start_tick"] > previous:
            labels.append({**parse_symbol("X"), "start_tick": previous, "end_tick": label["start_tick"],
                           "source": "sidecar_gap", "confidence": None, "review_status": "unreviewed",
                           "review_reasons": ["missing_annotation"]})
        reasons = []
        if label["end_tick"] > duration:
            label["end_tick"] = duration
            reasons.append("clipped_to_midi_end")
        label["review_reasons"] = reasons
        labels.append(label)
        previous = label["end_tick"]
    if previous < duration:
        labels.append({**parse_symbol("X"), "start_tick": previous, "end_tick": duration,
                       "source": "sidecar_gap", "confidence": None, "review_status": "unreviewed",
                       "review_reasons": ["missing_annotation"]})
    result = {"version": "harmony_plan_v2", "piece_id": piece.get("piece_id", "unknown"), "ppq": midi.resolution,
              "duration_ticks": duration, "evidence_scope": "upstream_extracted_annotation",
              "review_status": "unreviewed", "labels": labels}
    validate_plan(result)
    return result


def annotate_music(piece, midi, chord_path=None, beat_path=None, key_path=None):
    result = copy.deepcopy(piece)
    timeline = midi_timeline(midi)
    bars = [p["tick"] for p in timeline["downbeats"]]
    beats = [p["tick"] for p in timeline["beats"]]
    histogram = np.zeros(12)
    for note in result["notes"]:
        start, end = int(midi.time_to_tick(note["start"])), int(midi.time_to_tick(note["end"]))
        bar_index, beat_index = bisect_right(bars, start) - 1, bisect_right(beats, start) - 1
        note["rhythm"] = {"start_tick": start, "end_tick": end,
                          "bar_index": bar_index if bar_index >= 0 else None,
                          "beat_index": beat_index if beat_index >= 0 else None,
                          "quarter_offset_in_bar": (start - bars[bar_index]) / midi.resolution if bar_index >= 0 else None,
                          "grid_source": timeline["beat_source"]}
        histogram[note["pitch"] % 12] += max(0, note["end"] - note["start"]) * note.get("velocity", 80) / 127
    # Krumhansl-Kessler 调性轮廓只给候选，不把相关系数冒充正确概率。
    profiles = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    candidates = []
    if np.std(histogram) > 0:
        for mode, profile in profiles.items():
            for root in range(12):
                candidates.append({"tonic_pc": root, "mode": mode,
                                   "correlation": float(np.corrcoef(histogram, np.roll(profile, root))[0, 1])})
        candidates.sort(key=lambda c: c["correlation"], reverse=True)
    annotation = {"version": "music_annotation_v2", "timeline": timeline,
                  "roles": {"source": piece.get("annotation", {}).get("method", piece.get("import_info", {}).get("label_source", "provided")),
                            "review_status": "unreviewed"},
                  "key_estimate": {"method": "krumhansl_kessler_duration_velocity", "status": "candidate_not_verified",
                                   "candidates": candidates[:3], "pitch_class_histogram": histogram.tolist()},
                  "phrase_segments": {"status": "not_annotated", "segments": []}}
    if beat_path:
        points = []
        for line in Path(beat_path).read_text(encoding="utf-8").splitlines():
            values = [float(x) for x in line.split()]
            if not values:
                continue
            if any(not math.isfinite(x) for x in values) or values[0] < 0 or (points and values[0] <= points[-1]["seconds"]):
                raise ValueError("Invalid POP909 beat sidecar")
            # 原列完整保存，避免把上游 0/1 标记误读为小节内拍号。
            points.append({"seconds": values[0], "tick": int(midi.time_to_tick(values[0])), "raw_flags": values[1:]})
        annotation["pop909_beat_midi"] = {"source": str(beat_path), "points": points, "review_status": "unreviewed"}
    if key_path:
        annotation["pop909_key_audio"] = {"source": str(key_path), "time_domain": "audio_seconds",
                                          "review_status": "unreviewed", "segments": []}
        for line in Path(key_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                start, end, key = line.split()
                annotation["pop909_key_audio"]["segments"].append({"start": float(start), "end": float(end), "symbol": key})
    result["music_annotation"] = annotation
    result["harmony_plan"] = complete_sidecar_plan(piece, midi, chord_path) if chord_path else annotate_accompaniment(piece, midi)
    return result
