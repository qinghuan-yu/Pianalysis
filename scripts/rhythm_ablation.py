"""Counterfactual note edits with explicit invariants; these are not training targets."""

from collections import Counter
import copy
import statistics


def key(note):
    return note["start_tick"], note["pitch"]


def index_notes(notes):
    indexed = {key(n): n for n in notes}
    if len(indexed) != len(notes):
        raise ValueError("Duplicate pitch/onset pairs make the intervention ambiguous")
    return indexed


def containing_label(note, labels):
    return next(l for l in labels if l["start_tick"] <= note["start_tick"] < l["end_tick"])


def swap_common_field(base, donor, labels, field):
    if field not in ("end_tick", "velocity"):
        raise ValueError("Only duration or velocity can be exchanged here")
    index_notes(base)
    donor_index = index_notes(donor)
    result, edits, capped, common = copy.deepcopy(base), [], 0, 0
    for note in result:
        identity = key(note)
        if identity not in donor_index:
            continue
        common += 1
        target = donor_index[identity][field]
        if field == "end_tick":
            label = containing_label(note, labels)
            next_onset = min((n["start_tick"] for n in base if n["pitch"] == note["pitch"] and n["start_tick"] > note["start_tick"]), default=label["end_tick"])
            bounded = min(target, next_onset, label["end_tick"])
            capped += bounded != target
            target = bounded
            if target <= note["start_tick"]:
                raise ValueError("Donor duration is not positive")
        elif not 1 <= target <= 127:
            raise ValueError("Invalid donor velocity")
        if note[field] != target:
            edits.append({"start_tick": note["start_tick"], "pitch": note["pitch"], "old": note[field], "new": target})
            note[field] = target
    return result, {"field": field, "common_onsets": common, "changed_notes": len(edits),
                    "donor_endings_capped_at_base_retrigger": capped, "edits": edits}


def thin_to_count(base, donor, labels):
    """只删V4音符；保留共同起音及每和声段各音高的首次起音。"""
    base_index, donor_index = index_notes(base), index_notes(donor)
    target = len(donor)
    if target > len(base):
        raise ValueError("Thinning cannot add notes")
    protected = set(base_index) & set(donor_index)
    for label in labels:
        group = [n for n in base if label["start_tick"] <= n["start_tick"] < label["end_tick"]]
        for pitch in {n["pitch"] for n in group}:
            protected.add(key(min((n for n in group if n["pitch"] == pitch), key=lambda n: n["start_tick"])))
    if len(protected) > target:
        raise ValueError("The requested count conflicts with protected chord attacks")

    def distance(note):
        label = containing_label(note, labels)
        same = [abs(n["start_tick"] - note["start_tick"]) for n in donor if n["pitch"] == note["pitch"]
                and label["start_tick"] <= n["start_tick"] < label["end_tick"]]
        return min(same, default=label["end_tick"] - label["start_tick"])

    removable = sorted((n for n in base if key(n) not in protected),
                       key=lambda n: (-distance(n), n["velocity"], -n["start_tick"], n["pitch"]))
    removed = removable[:len(base) - target]
    removed_keys = {key(n) for n in removed}
    result = [copy.deepcopy(n) for n in base if key(n) not in removed_keys]
    return result, {"target_upper_notes": target, "removed_notes": len(removed), "protected_notes": len(protected),
                    "removed_events": removed, "selection": "farthest_from_donor_same_pitch_attack_then_quieter_then_later",
                    "scope": "global_count_matched; local density and rhythmic positions are not matched"}


def standardize(notes, labels, ppq, velocity=45):
    result = copy.deepcopy(notes)
    for note in result:
        label = containing_label(note, labels)
        note["end_tick"] = min(label["end_tick"], note["start_tick"] + round(ppq / 4))
        note["velocity"] = velocity
    return result


def describe(notes, labels, ppq, melody):
    quarters = sum(l["end_tick"] - l["start_tick"] for l in labels) / ppq
    unit = ppq / 4
    counts, patterns, full_patterns = [], [], []
    for label in labels:
        pcs = sorted(label["pitch_classes"], key=lambda p: (p - label["root_pc"]) % 12)
        active = [n for n in notes if label["start_tick"] <= n["start_tick"] < label["end_tick"]]
        counts.append(len(active))
        pattern = tuple(sorted((round((n["start_tick"] - label["start_tick"]) / unit), pcs.index(n["pitch"] % 12)) for n in active))
        patterns.append(pattern)
        if label["end_tick"] - label["start_tick"] == ppq * 4:
            full_patterns.append(pattern)
    melody_ticks = {n["start_tick"] for n in melody}
    durations = [(n["end_tick"] - n["start_tick"]) / ppq for n in notes]
    pattern_counts = Counter(full_patterns)
    return {"upper_notes": len(notes), "upper_onsets_per_quarter": len(notes) / quarters,
            "duration_mean_quarters": statistics.mean(durations) if durations else 0,
            "duration_median_quarters": statistics.median(durations) if durations else 0,
            "velocity_mean": statistics.mean(n["velocity"] for n in notes) if notes else 0,
            "velocity_sd": statistics.pstdev(n["velocity"] for n in notes) if notes else 0,
            "onsets_at_melody_onset": sum(n["start_tick"] in melody_ticks for n in notes),
            "notes_by_chord": counts, "four_quarter_spans": len(full_patterns),
            "unique_four_quarter_onset_patterns": len(pattern_counts),
            "most_repeated_four_quarter_pattern_count": max(pattern_counts.values(), default=0),
            "onset_patterns": patterns}
