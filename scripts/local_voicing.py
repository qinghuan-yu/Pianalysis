"""Local chord-to-voicing curriculum; rhythm is scheduled from verified ticks."""

from itertools import product
from pathlib import Path
import copy
import statistics

import harmony_plan as hp

VERSION = "local_voicing_v3"
PAD, BOS, SEP, EOS = range(4)
PITCH = 4
ROOT, BASS, PC = 132, 144, 156
CHORD, MELODY, PREVIOUS = 168, 169, 170
VOCAB = 171
INITIAL = [36, 48, 52, 55]


def melody_summary(notes, label):
    active = [n for n in notes if n["start_tick"] < label["end_tick"] and n["end_tick"] > label["start_tick"]]
    pitches = [int(n["pitch"]) for n in active]
    return {"min": min(pitches, default=76), "max": max(pitches, default=76),
            "pcs": sorted({p % 12 for p in pitches}),
            "velocity": statistics.median(n.get("velocity", 80) for n in active) if active else 80}


def prompt(label, melody, previous):
    if label["status"] != "chord" or len(label["pitch_classes"]) != 3:
        raise ValueError("The first curriculum only supports known triads")
    if len(previous) != 4 or any(not 0 <= p <= 127 for p in previous):
        raise ValueError("Invalid previous voicing")
    return [BOS, CHORD, ROOT + label["root_pc"], BASS + label["bass_pc"],
            *[PC + p for p in label["pitch_classes"]], MELODY,
            PITCH + melody["min"], PITCH + melody["max"], *[PC + p for p in melody["pcs"]],
            PREVIOUS, *[PITCH + p for p in previous], SEP]


def teacher_voicing(label, melody, previous):
    """规则教师只构造合成练习目标，不冒充原曲的真实配位。"""
    ceiling = min(67, melody["min"] - 3)
    choices = [[p for p in range(36, ceiling + 1) if p % 12 == pc] for pc in label["pitch_classes"]]
    candidates = []
    for pitches in product(*choices):
        upper = sorted(pitches)
        if upper[-1] - upper[0] > 12 or upper[0] < 42:
            continue
        basses = [p for p in range(24, 48) if p % 12 == label["bass_pc"] and p <= upper[0] - 3]
        if not basses:
            continue
        bass = min(basses, key=lambda p: (abs(p - 40), p))
        # 保持内声部连续，同时避免一直向低音区塌缩。
        cost = sum(abs(p - q) for p, q in zip(upper, previous[1:]))
        cost += .25 * abs(sum(upper) / 3 - 54) + .05 * abs(bass - previous[0])
        candidates.append((cost, [bass, *upper]))
    if not candidates:
        raise ValueError("Melody register cannot accommodate this simple voicing")
    return min(candidates, key=lambda c: (c[0], c[1]))[1]


def assess(voicing, label, melody):
    upper = voicing[1:]
    return {"bass_correct": voicing[0] % 12 == label["bass_pc"],
            "chord_complete": set(p % 12 for p in upper) == set(label["pitch_classes"]),
            "chord_tone_ratio": sum(p % 12 in label["pitch_classes"] for p in upper) / 3,
            "voices_ordered": all(a < b for a, b in zip(voicing, voicing[1:])),
            "melody_clear": max(voicing) <= melody["min"] - 3,
            "unique_pitches": len(set(voicing)) == 4,
            "upper_span_ok": max(upper) - min(upper) <= 12}


def schedule(label, voicing, ppq, style, velocity=100):
    """拍点/时值/力度是明确的渲染规则，不计为模型学到的能力。"""
    if style not in ("block", "arpeggio"):
        raise ValueError("Unknown simple texture")
    a, b = label["start_tick"], label["end_tick"]
    upper_velocity = max(16, min(50, round(velocity * .45)))
    bass_velocity = max(20, min(58, round(velocity * .52)))
    make = lambda p, left, right, v: {"pitch": p, "start_tick": left, "end_tick": right,
                                      "velocity": v, "role": "accompaniment"}
    notes = [make(voicing[0], a, b, bass_velocity)]
    if style == "block":
        notes.extend(make(p, a, b, upper_velocity) for p in voicing[1:])
    else:
        index = 0
        while True:
            start = a + round(index * ppq / 2)
            if start >= b:
                break
            end = min(b, a + round((index + 1) * ppq / 2))
            notes.append(make(voicing[1 + (0, 1, 2, 1)[index % 4]], start, end, upper_velocity))
            index += 1
    return notes


def write_midi(source, melody, accompaniment, path, origin=0, end=None):
    """保留速度图；裁切时继承起点的有效元数据，同音重叠分channel。"""
    end = end if end is not None else int(source.time_to_tick(source.get_end_time()))
    if not 0 <= origin < end:
        raise ValueError("Invalid MIDI interval")
    result = hp.mido.MidiFile(ticks_per_beat=source.resolution)

    def add(events):
        track, previous = hp.mido.MidiTrack(), 0
        for tick, priority, msg in sorted(events, key=lambda x: (x[0], x[1])):
            track.append(msg.copy(time=tick - previous))
            previous = tick
        result.tracks.append(track)

    metadata = []
    families = [[(int(source.time_to_tick(t)), hp.mido.MetaMessage("set_tempo", tempo=round(60_000_000 / bpm)))
                 for t, bpm in zip(*source.get_tempo_changes())]]
    families.append([(int(source.time_to_tick(e.time)), hp.mido.MetaMessage("time_signature", numerator=e.numerator, denominator=e.denominator))
                     for e in source.time_signature_changes])
    for family in families:
        prior = [event for event in family if event[0] <= origin]
        if prior:
            metadata.append((0, 0, prior[-1][1]))
        metadata.extend((tick - origin, 0, msg) for tick, msg in family if origin < tick < end)
    for family in [[(int(source.time_to_tick(e.time)), e.key_number) for e in source.key_signature_changes]]:
        active = [event for event in family if event[0] <= origin]
        points = ([(origin, active[-1][1])] if active else []) + [x for x in family if origin < x[0] < end]
        for tick, key in points:
            name, mode = hp.pretty_midi.key_number_to_key_name(key).split()
            metadata.append((tick - origin, 1, hp.mido.MetaMessage("key_signature", key=name + ("m" if mode == "minor" else ""))))
    add(metadata)
    for role, notes, channels in (("MELODY", melody, [0, 2, 3, 4, 5]),
                                  ("ACCOMPANIMENT", accompaniment, [1, 6, 7, 8, 10, 11, 12, 13, 14, 15])):
        events = [(0, -2, hp.mido.MetaMessage("track_name", name=role))]
        active, used = {}, set()
        for note in sorted(notes, key=lambda n: (n["start_tick"], n["pitch"], n["end_tick"])):
            a, b = max(origin, note["start_tick"]), min(end, note["end_tick"])
            if a >= b:
                continue
            available = [c for c in channels if active.get((c, note["pitch"]), -1) <= a]
            if not available:
                raise ValueError("Too many overlapping notes of the same pitch")
            channel = available[0]
            active[channel, note["pitch"]] = b
            used.add(channel)
            events.extend([(a - origin, 1, hp.mido.Message("note_on", channel=channel, note=note["pitch"], velocity=int(note["velocity"]))),
                           (b - origin, 0, hp.mido.Message("note_off", channel=channel, note=note["pitch"], velocity=0))])
        events.extend((0, -1, hp.mido.Message("program_change", channel=c, program=0)) for c in sorted(used))
        add(events)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(path))
    return hp.pretty_midi.PrettyMIDI(str(path))


def transpose(label, melody, semitones, octave=0, inversion=None):
    label, melody = copy.deepcopy(label), copy.deepcopy(melody)
    for key in ("root_pc", "bass_pc"):
        label[key] = (label[key] + semitones) % 12
    label["pitch_classes"] = sorted((p + semitones) % 12 for p in label["pitch_classes"])
    if inversion is not None:
        ordered = sorted(label["pitch_classes"], key=lambda p: (p - label["root_pc"]) % 12)
        label["bass_pc"] = ordered[inversion]
    label.pop("symbol", None)
    for key in ("min", "max"):
        melody[key] += semitones + octave
    melody["pcs"] = sorted((p + semitones) % 12 for p in melody["pcs"])
    return label, melody
