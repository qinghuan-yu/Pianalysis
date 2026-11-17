"""Normalize pinned melody/harmony/piano sources with explicit quality gates.

All output times are quarter-note beats. Missing chord annotation is unknown,
never no-chord. This script performs structural checks, not human listening.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
import zipfile

import mido

from fetch_auto_accompaniment_data import PROJECT, verify, write_json


VERSION = "auto_accompaniment_data_v1"
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6),
    "aug": (0, 4, 8), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10), "hdim7": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9), "minmaj7": (0, 3, 7, 11),
    "aug7": (0, 4, 8, 10), "5": (0, 7), "6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9), "add9": (0, 2, 4, 7),
    "minadd9": (0, 2, 3, 7), "9": (0, 2, 4, 7, 10),
    "maj9": (0, 2, 4, 7, 11), "min9": (0, 2, 3, 7, 10),
    "11": (0, 2, 4, 5, 7, 10), "min11": (0, 2, 3, 5, 7, 10),
    "7sus4": (0, 5, 7, 10), "7sus2": (0, 2, 7, 10),
}
POP_QUALITIES = {"M": "maj", "m": "min", "o": "dim", "+": "aug",
                "D7": "7", "M7": "maj7", "m7": "min7",
                "/o7": "hdim7", "o7": "dim7", "mM7": "minmaj7",
                "+7": "aug7", "sus2": "sus2", "sus4": "sus4"}
QUALITY_BY_SET = {tuple(sorted(value)): key for key, value in INTERVALS.items()}
SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}


class Rejected(ValueError):
    pass


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def number(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Rejected("non_finite_time")
    return result


def unknown(start: float, end: float, reason: str) -> dict:
    return {"start": start, "end": end, "root": None, "quality": "unknown",
            "bass": None, "pitch_classes": [], "symbol": "X",
            "status": "unknown", "provenance": reason}


def chord_timeline(chords: list[dict], duration: float) -> list[dict]:
    output, cursor = [], 0.0
    for chord in sorted(chords, key=lambda c: (c["start"], c["end"])):
        start, end = number(chord["start"]), number(chord["end"])
        if start < -1e-8 or end <= start or end > duration + 1e-6:
            raise Rejected("invalid_chord_interval")
        if start < cursor - 1e-6:
            raise Rejected("overlapping_chord_annotations")
        if start > cursor + 1e-8:
            output.append(unknown(cursor, start, "missing_annotation"))
        output.append(chord)
        cursor = end
    if cursor < duration - 1e-8:
        output.append(unknown(cursor, duration, "missing_annotation"))
    return output


def parse_notes(track: mido.MidiTrack, ppq: int) -> tuple[list[dict], list[dict]]:
    active = defaultdict(deque)
    notes, controls = [], []
    tick = 0
    for message in track:
        tick += message.time
        if message.time < 0:
            raise Rejected("negative_delta_tick")
        if message.type == "control_change" and message.control in (64, 66, 67):
            controls.append({"beat": tick / ppq, "control": message.control,
                             "value": message.value})
        if message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((tick, message.velocity))
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            key = (message.channel, message.note)
            if not active[key]:
                raise Rejected(f"unmatched_note_off:{track.name}")
            start, velocity = active[key].popleft()
            if tick <= start:
                raise Rejected(f"non_positive_note_duration:{track.name}")
            notes.append({"pitch": message.note, "start": start / ppq,
                          "end": tick / ppq, "velocity": velocity})
    if any(active.values()):
        raise Rejected(f"unterminated_note:{track.name}")
    return sorted(notes, key=lambda n: (n["start"], n["pitch"], n["end"])), controls


def pc(name: str) -> int:
    match = re.fullmatch(r"([A-G])([#b]{0,2})", name)
    if not match:
        raise Rejected(f"invalid_pitch_name:{name}")
    return (dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[match[1]]
            + match[2].count("#") - match[2].count("b")) % 12


def marker_chord(symbol: str) -> tuple[int | None, str, int | None]:
    if symbol in ("N", "NC", "N.C."):
        return None, "no_chord", None
    if ":" not in symbol:
        return None, "unknown", None
    root_name, tail = symbol.split(":", 1)
    bass_name = root_name
    # 半减七和弦的 /o7 是性质名，不是转位斜线。
    if "/" in tail and re.fullmatch(r"[A-G][#b]{0,2}", tail.rsplit("/", 1)[-1]):
        tail, bass_name = tail.rsplit("/", 1)
    quality = POP_QUALITIES.get(tail, "unknown")
    try:
        return pc(root_name), quality, pc(bass_name)
    except Rejected:
        return None, "unknown", None


def parse_pop(path: Path, meta: dict) -> dict:
    midi = mido.MidiFile(path)
    if midi.type != 1 or midi.ticks_per_beat <= 0:
        raise Rejected("unsupported_midi_type_or_ppq")
    ppq = midi.ticks_per_beat
    names = [track.name for track in midi.tracks]
    if len(names) != 5 or names[1:] != ["MELODY", "BRIDGE", "PIANO", "CHORDS"]:
        raise Rejected(f"unexpected_track_roles:{names}")
    if int(meta["ppq"]) != ppq:
        raise Rejected("metadata_ppq_mismatch")
    roles, controls = {}, {}
    for track in midi.tracks[1:]:
        roles[track.name], controls[track.name] = parse_notes(track, ppq)
    if not roles["MELODY"] or not roles["PIANO"] or not roles["CHORDS"]:
        raise Rejected("empty_required_role")
    meters, tempos, keys, markers = [], [], [], {}
    tick = 0
    for message in midi.tracks[0]:
        tick += message.time
        beat = tick / ppq
        if message.type == "time_signature":
            meters.append({"beat": beat, "numerator": message.numerator,
                           "denominator": message.denominator})
        elif message.type == "set_tempo":
            tempos.append({"beat": beat, "bpm": mido.tempo2bpm(message.tempo)})
        elif message.type == "key_signature":
            keys.append({"beat": beat, "key": message.key})
        elif message.type == "marker":
            if beat in markers and markers[beat] != message.text:
                raise Rejected("conflicting_chord_markers")
            markers[beat] = message.text
    if not meters or meters[0]["beat"] != 0 or not tempos or tempos[0]["beat"] != 0:
        raise Rejected("missing_initial_meter_or_tempo")
    musical_roles = ("MELODY", "BRIDGE", "PIANO")
    duration = max(n["end"] for role in musical_roles for n in roles[role])
    blocks = defaultdict(list)
    for note in roles["CHORDS"]:
        blocks[note["start"]].append(note)
    chords, mismatches = [], []
    for start, notes in sorted(blocks.items()):
        ends = {n["end"] for n in notes}
        if len(ends) != 1:
            raise Rejected("chord_tones_have_different_offsets")
        end = next(iter(ends))
        if start >= duration:
            continue
        end = min(end, duration)
        pcs = sorted({n["pitch"] % 12 for n in notes})
        bass = min(n["pitch"] for n in notes) % 12
        # 上游模板未识别的和弦音块不一定有 marker；保留真实音级并屏蔽名称监督。
        symbol = markers.get(start, "X")
        root, quality, marker_bass = marker_chord(symbol)
        status = "known" if quality in INTERVALS else "unknown"
        if status == "known":
            expected = sorted((root + interval) % 12 for interval in INTERVALS[quality])
            if expected != pcs or marker_bass != bass:
                mismatches.append({"beat": start, "symbol": symbol, "pitch_classes": pcs,
                                   "bass": bass, "marker_bass": marker_bass})
        chords.append({"start": start, "end": end, "root": root,
                       "quality": quality, "bass": bass, "pitch_classes": pcs,
                       "symbol": symbol, "status": status,
                       "provenance": "upstream_algorithmic_unreviewed" if "algorithmic" in meta["flags"] else "upstream_expert_chord_tones_template_symbol"})
    if mismatches:
        raise Rejected(f"marker_tone_mismatch:{len(mismatches)}")
    # 上游已完成时序修订；这里只除以 PPQ，绝不再次加 total_shift_ticks。
    accompaniment = sorted(roles["PIANO"],
                           key=lambda n: (n["start"], n["pitch"], n["end"]))
    item = {"schema_version": 1, "song_id": f"pop909_cl_{path.stem}",
            "work_id": f"pop909:{path.stem}", "source": "pop909_cl",
            "split": None, "ppq": ppq, "melody": roles["MELODY"],
            "accompaniment": accompaniment,
            "chords": chord_timeline(chords, duration),
            "time_signatures": meters, "tempos": tempos, "duration_beats": duration,
            "metadata": {"title": meta["name"], "artist": meta["artist"],
                         "upstream_piece_id": path.stem, "key_signatures": keys,
                         "source_total_shift_ticks": int(meta["total_shift_ticks"]),
                         "applied_additional_shift_ticks": 0,
                         "bar_origin_beat": 0.0,
                         "bar_origin_provenance": "POP909-CL corrected downbeat grid",
                         "pickup_beats": None,
                         "first_melody_beat": roles["MELODY"][0]["start"],
                         "source_track_note_counts": {r: len(n) for r, n in roles.items()},
                         "target_roles": ["PIANO"], "excluded_target_roles": ["CHORDS", "MELODY", "BRIDGE"],
                         "tempo_status": "source_midi",
                         "pedal_events_by_role": {r: controls[r] for r in musical_roles},
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "upstream_flags": meta["flags"], "human_listening_verified": False,
                         "train_eligible": not bool(meta["flags"])}}
    return item


def verify_pop_extraction(raw: Path) -> None:
    """Verify every extracted input against the checksum-pinned archive."""
    with zipfile.ZipFile(raw / "POP909_processed.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith(".mid") or n.endswith("metadata.csv")]
        for name in names:
            path = raw / "pop909_cl" / name
            if not path.is_file() or path.read_bytes() != archive.read(name):
                raise ValueError(f"Extracted POP909-CL source differs from archive: {name}")
        expected = {Path(n).name for n in names if n.endswith(".mid")}
        actual = {p.name for p in (raw / "pop909_cl/POP909_processed").glob("*.mid")}
        if expected != actual:
            raise ValueError("Extracted POP909-CL file set differs from pinned archive")


def parse_hook(uid: str, raw: dict) -> dict:
    ann = raw["annotations"]
    if not ann.get("melody") or not ann.get("harmony"):
        raise Rejected("missing_melody_or_harmony")
    if "NO_SWING" not in raw["tags"]:
        raise Rejected("unsupported_swing_annotation")
    if not ann["meters"] or ann["meters"][0]["beat"] != 0:
        raise Rejected("missing_initial_meter")
    if any(m["beat_unit"] != 4 for m in ann["meters"]):
        raise Rejected("unsupported_hooktheory_nonquarter_beat_unit")
    duration = number(ann["num_beats"])
    if duration <= 0:
        raise Rejected("non_positive_duration")
    melody = [{"pitch": (5 + int(n["octave"])) * 12 + int(n["pitch_class"]),
               "start": number(n["onset"]), "end": number(n["offset"]), "velocity": 80}
              for n in ann["melody"]]
    chords = []
    for source in ann["harmony"]:
        root = source["root_pitch_class"]
        if not isinstance(root, int) or not 0 <= root < 12:
            raise Rejected("invalid_hook_root")
        offsets, total = [0], 0
        for interval in source["root_position_intervals"]:
            total += int(interval)
            offsets.append(total)
        inversion = source["inversion"]
        if not isinstance(inversion, int) or not 0 <= inversion < len(offsets):
            raise Rejected("invalid_hook_inversion")
        degrees = tuple(sorted({i % 12 for i in offsets}))
        quality = QUALITY_BY_SET.get(degrees, "custom")
        pcs = sorted({(root + i) % 12 for i in offsets})
        bass = (root + offsets[inversion]) % 12
        symbol = f"{NAMES[root]}:{quality}" + (f"/{NAMES[bass]}" if bass != root else "")
        chords.append({"start": number(source["onset"]), "end": number(source["offset"]),
                       "root": root, "quality": quality, "bass": bass,
                       "pitch_classes": pcs, "symbol": symbol,
                       "status": "known",
                       "provenance": "hooktheory_community_symbolic_intervals"})
    meters = [{"beat": number(m["beat"]), "numerator": m["beats_per_bar"],
               "denominator": m["beat_unit"]} for m in ann["meters"]]
    tempos = []
    alignment = raw.get("alignment", {}).get("user")
    if alignment and len(alignment["beats"]) == len(alignment["times"]):
        for i in range(len(alignment["beats"]) - 1):
            a, b = alignment["beats"][i:i + 2]
            ta, tb = alignment["times"][i:i + 2]
            if b <= a or tb <= ta:
                raise Rejected("non_monotonic_hook_alignment")
            tempos.append({"beat": number(a), "bpm": 60.0 * (b - a) / (tb - ta)})
    identity = raw["hooktheory"]
    return {"schema_version": 1, "song_id": f"hooktheory_{uid}",
            "work_id": f"hooktheory:{identity['artist']}/{identity['song']}",
            "source": "hooktheory", "split": {"TRAIN": "train", "VALID": "validation", "TEST": "test"}[raw["split"]],
            "ppq": 480, "melody": sorted(melody, key=lambda n: (n["start"], n["pitch"], n["end"])),
            "accompaniment": [], "chords": chord_timeline(chords, duration),
            "time_signatures": meters, "tempos": tempos, "duration_beats": duration,
            "metadata": {"title": identity["song"], "artist": identity["artist"],
                         "hooktheory_id": uid, "upstream_split": raw["split"],
                         "youtube_id": raw.get("youtube", {}).get("id"),
                         "key_signatures": ann["keys"], "bar_origin_beat": None,
                         "bar_origin_provenance": "not independently established for annotation clip",
                         "pickup_beats": None, "first_melody_beat": min(n["start"] for n in melody),
                         "source_tags": raw["tags"], "source_sha256": digest(raw),
                         "velocity_provenance": "constant_80_for_symbolic_notes_not_performance",
                         "ppq_provenance": "normalization_resolution_not_upstream_MIDI",
                         "tempo_provenance": "user_alignment_derived" if tempos else "unavailable",
                         "tempo_status": "source_alignment_derived" if tempos else "unknown",
                         "annotation_level": "song_segment", "human_listening_verified": False,
                         "train_eligible": True}}


def validate(item: dict) -> None:
    duration = item["duration_beats"]
    for role in ("melody", "accompaniment"):
        for note in item[role]:
            if not isinstance(note["pitch"], int) or not 0 <= note["pitch"] <= 127:
                raise Rejected(f"invalid_pitch:{role}")
            if not 1 <= note["velocity"] <= 127:
                raise Rejected(f"invalid_velocity:{role}")
            if not 0 <= note["start"] < note["end"] <= duration + 1e-6:
                raise Rejected(f"invalid_note_interval:{role}")
    if not any(c["status"] == "known" for c in item["chords"]):
        raise Rejected("no_supported_known_chord")
    for meter in item["time_signatures"]:
        if not 0 <= meter["beat"] < duration or meter["numerator"] <= 0:
            raise Rejected("invalid_meter")
    if any(t["bpm"] <= 0 or not math.isfinite(t["bpm"]) for t in item["tempos"]):
        raise Rejected("invalid_tempo")
    item["metadata"]["known_chord_coverage"] = sum(
        c["end"] - c["start"] for c in item["chords"] if c["status"] == "known") / duration
    item["metadata"]["unknown_chord_beats"] = sum(
        c["end"] - c["start"] for c in item["chords"] if c["status"] == "unknown")


def normalized_name(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def melody_fingerprint(item: dict) -> str:
    notes = item["melody"]
    origin, pitch = notes[0]["start"], notes[0]["pitch"]
    return digest([(n["pitch"] - pitch, round((n["start"] - origin) * 24),
                    round((n["end"] - n["start"]) * 24)) for n in notes])


def group_and_split(items: list[dict], config: dict) -> dict:
    parents = list(range(len(items)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    identifiers, reasons = {}, Counter()
    for i, item in enumerate(items):
        meta = item["metadata"]
        title, artist = normalized_name(meta["title"]), normalized_name(meta["artist"])
        keys = [("source_work", item["work_id"]),
                ("title_artist", title + "/" + artist),
                ("melody_full_transposition_invariant", melody_fingerprint(item))]
        if title and len(title) >= 4:
            keys.append(("long_title_conservative_group", title))
        if meta.get("youtube_id"):
            keys.append(("youtube_source", meta["youtube_id"]))
        for kind, value in keys:
            key = (kind, value)
            if key in identifiers:
                union(i, identifiers[key])
                reasons[kind] += 1
            else:
                identifiers[key] = i
    # 32 音片段用于发现全曲和片段跨库重复；命中两段才归为同组。
    shingles = defaultdict(list)
    for i, item in enumerate(items):
        notes = item["melody"]
        if len(notes) < 48:
            continue
        seen = set()
        for start in range(0, len(notes) - 31, 8):
            part = notes[start:start + 32]
            key = digest([(n["pitch"] - part[0]["pitch"],
                           round((n["start"] - part[0]["start"]) * 24),
                           round((n["end"] - n["start"]) * 24)) for n in part])
            if key not in seen:
                shingles[key].append(i)
                seen.add(key)
    hits = Counter()
    for ids in shingles.values():
        if len(ids) > 8:
            continue
        for pos, a in enumerate(ids):
            for b in ids[pos + 1:]:
                if items[a]["source"] != items[b]["source"]:
                    hits[(a, b)] += 1
    cross_candidates = []
    for (a, b), count in sorted(hits.items()):
        if count >= 2:
            union(a, b)
            cross_candidates.append({"a": items[a]["song_id"], "b": items[b]["song_id"],
                                     "shared_32_note_shingles": count,
                                     "status": "conservatively_grouped_not_human_verified"})
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    frozen = {f"pop909_cl_{piece}": split for split, pieces in config["legacy_pop909_splits"].items() for piece in pieces}
    manifest = {"schema_version": 1, "seed": config["seed"], "legacy_frozen": frozen,
                "grouping_rule_counts": dict(reasons), "cross_source_melody_candidates": cross_candidates,
                "leakage_guarantee": "No group crosses splits. Title/source identity and exact transposition-invariant fingerprints are screened; aliases, translations and approximate cover matches remain unresolved.",
                "works": []}
    for indices in groups.values():
        members = [items[i] for i in indices]
        source_ids = sorted(m["song_id"] for m in members)
        work = "work:" + digest(source_ids)[:20]
        locks = {frozen[m["song_id"]] for m in members if m["song_id"] in frozen}
        if len(locks) > 1:
            for m in members:
                m["metadata"]["train_eligible"] = False
                m["metadata"]["rejection_reason"] = "conflicting_legacy_split_work_group"
            split = max(locks, key=SPLIT_RANK.get)
        elif locks:
            split = next(iter(locks))
        elif any(m["split"] for m in members):
            split = max((m["split"] for m in members if m["split"]), key=SPLIT_RANK.get)
        else:
            value = int(hashlib.sha256(f"{config['seed']}:{work}".encode()).hexdigest()[:8], 16) / 2**32
            split = "train" if value < 0.8 else "validation" if value < 0.9 else "test"
        for m in members:
            m["metadata"]["source_work_id"] = m["work_id"]
            m["work_id"], m["split"] = work, split
        manifest["works"].append({"work_id": work, "split": split, "song_ids": source_ids})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "examples/auto_accompaniment/data_sources.json")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--only-source", choices=("pop909_cl", "hooktheory"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = (args.root or PROJECT / config["root"]).resolve()
    if not root.is_relative_to(PROJECT):
        raise ValueError("Dataset root must remain inside the project")
    raw, out = root / "raw", root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    sources = [s for s in config["sources"] if not args.only_source or s["id"] == args.only_source]
    verified = [verify(raw / s["filename"], s) for s in sources]
    records, items = [], []
    def accept(song_id, source, loader, source_details=None):
        try:
            item = loader()
            validate(item)
            items.append(item)
            records.append({"song_id": song_id, "source": source,
                            "status": "accepted" if item["metadata"]["train_eligible"] else "quarantined",
                            "reason": None if item["metadata"]["train_eligible"] else "upstream_flag_requires_review",
                            "upstream_flags": item["metadata"].get("upstream_flags"),
                            "melody_notes": len(item["melody"]), "accompaniment_notes": len(item["accompaniment"]),
                            "known_chord_coverage": item["metadata"]["known_chord_coverage"],
                            "duration_beats": item["duration_beats"], "ppq": item["ppq"],
                            "bar_origin_beat": item["metadata"]["bar_origin_beat"],
                            "first_melody_beat": item["metadata"]["first_melody_beat"],
                            "source_sha256": item["metadata"]["source_sha256"]})
        except (Rejected, ValueError, KeyError, TypeError, IndexError) as error:
            records.append({"song_id": song_id, "source": source, "status": "rejected",
                            "reason": str(error)})
        records[-1]["train_eligible"] = records[-1]["status"] == "accepted"
        if source_details:
            records[-1]["source_metadata"] = source_details
    if any(s["id"] == "pop909_cl" for s in sources):
        verify_pop_extraction(raw)
        pop = raw / "pop909_cl/POP909_processed"
        with (pop / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            metadata = {r["piece_id"]: r for r in csv.DictReader(handle)}
        paths = sorted(pop.glob("*.mid"))
        for path in paths[:args.limit]:
            accept(f"pop909_cl_{path.stem}", "pop909_cl", lambda p=path: parse_pop(p, metadata[p.stem]),
                   {**metadata[path.stem], "midi_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for uid, meta in metadata.items():
            if meta["status"] != "released":
                records.append({"song_id": f"pop909_cl_{uid}", "source": "pop909_cl",
                                "status": "upstream_missing", "reason": meta.get("flags") or meta["status"]})
        print(f"Checked POP909-CL: {len(paths[:args.limit])} MIDI files", flush=True)
    if any(s["id"] == "hooktheory" for s in sources):
        with gzip.open(raw / "Hooktheory.json.gz", "rt", encoding="utf-8") as handle:
            corpus = json.load(handle)
        for uid in sorted(corpus)[:args.limit]:
            accept(f"hooktheory_{uid}", "hooktheory", lambda k=uid: parse_hook(k, corpus[k]))
        print(f"Checked Hooktheory: {len(corpus) if args.limit is None else min(len(corpus), args.limit)} annotations", flush=True)
        del corpus
    splits = group_and_split(items, config)
    accepted = [item for item in items if item["metadata"]["train_eligible"]]
    temporary = out / "samples.jsonl.part"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in accepted:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(out / "samples.jsonl")
    write_json(out / "split_manifest.json", splits)
    summary = Counter((item["source"], item["split"]) for item in accepted)
    audit = {"schema_version": 1, "human_listening_performed": False,
             "counts": dict(Counter(f"{r['source']}:{r['status']}" for r in records)),
             "rejection_reasons": dict(Counter(r["reason"] for r in records if r.get("reason"))),
             "split_counts": {f"{k[0]}:{k[1]}": v for k, v in summary.items()},
             "unknown_policy": "Unannotated or unsupported spans are unknown and must be masked; no absence is converted to no_chord.",
             "target_policy": "POP909-CL target is PIANO only; CHORDS, MELODY and BRIDGE are excluded. Hooktheory has no accompaniment target.",
             "records": records}
    write_json(out / "audit.json", audit)
    manifest = {"schema_version": 1, "preprocessor": VERSION, "sources": verified,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "samples_sha256": hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest(),
                "sample_count": len(accepted), "work_count": len({i["work_id"] for i in accepted}),
                "time_unit": "quarter_note", "split_before_slicing_and_augmentation": True,
                "partial_build": bool(args.limit or args.only_source),
                "human_listening_performed": False}
    write_json(out / "manifest.json", manifest)
    write_json(out / "schema_examples.json", [{**i, "melody": i["melody"][:3],
               "accompaniment": i["accompaniment"][:3], "chords": i["chords"][:3],
               "example_truncated": True} for i in [next((a for a in accepted if a["source"] == s["id"]), None) for s in sources] if i])
    print(json.dumps({"sample_count": len(accepted), "split_counts": audit["split_counts"],
                      "rejection_reasons": audit["rejection_reasons"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
