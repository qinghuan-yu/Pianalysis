"""Verify rhythm checkpoint replay, source provenance and actual MIDI/audio exports."""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from build_harmony_review import write_json
from import_pop909 import load_piece
import harmony_plan as hp
import local_voicing as lv
import rhythm_v4 as rv
from train_local_voicing import validate_dataset, predict as predict_voicing
from train_rhythm_v4 import predict


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tuples(notes, origin, end):
    return sorted((n["pitch"], int(n["velocity"]), max(origin, n["start_tick"]) - origin,
                   min(end, n["end_tick"]) - origin)
                  for n in notes if n["start_tick"] < end and n["end_tick"] > origin)


def metadata(midi, origin, end):
    families = {
        "tempo": [(int(midi.time_to_tick(t)), round(60_000_000 / b)) for t, b in zip(*midi.get_tempo_changes())],
        "meter": [(int(midi.time_to_tick(x.time)), x.numerator, x.denominator) for x in midi.time_signature_changes],
        "key": [(int(midi.time_to_tick(x.time)), x.key_number) for x in midi.key_signature_changes],
    }
    result = {}
    for name, values in families.items():
        prior = [x for x in values if x[0] <= origin]
        result[name] = ([(0, *prior[-1][1:])] if prior else []) + [(x[0] - origin, *x[1:]) for x in values if origin < x[0] < end]
    return result


def realized_frames(label, notes, ppq):
    # 从导出事件的时值反算指标；原始预测的长音可能已被同声部重触发截短。
    frames, pcs = rv.empty_frames(rv.steps_for(label, ppq)), rv.degrees(label)
    for n in notes:
        step = round((n["start_tick"] - label["start_tick"]) * rv.SUBDIVISIONS / ppq)
        lane = pcs.index(n["pitch"] % 12)
        frames[step]["mask"] |= 1 << lane
        frames[step]["durations"][lane] = (n["end_tick"] - n["start_tick"]) * rv.SUBDIVISIONS / ppq
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/trusted_harmony_v4")
    parser.add_argument("--model-dir", default="model_output/local_rhythm_v4")
    args = parser.parse_args()
    out, model_dir = Path(args.data_dir), Path(args.model_dir)
    annotation, evaluation = read(out / "annotations.json"), read(out / "evaluation/evaluation.json")
    training = read(model_dir / "training_metadata.json")
    validate_dataset(annotation)
    assert digest(out / "annotations.json") == training["source_sha256"]
    dataset = rv.build_dataset(annotation)
    assert dataset == read(out / "rhythm_dataset.json")
    # 删除核对页的合成配位也应得到同一个真实节奏数据集。
    without_teacher = copy.deepcopy(annotation)
    for piece in without_teacher["pieces"]:
        for label in piece["labels"]:
            label.pop("synthetic_teacher_voicing")
    rebuilt = rv.build_dataset(without_teacher)
    assert [(r["frames"], r["sample"], r["projection_audit"]) for r in rebuilt["records"]] == [
        (r["frames"], r["sample"], r["projection_audit"]) for r in dataset["records"]]
    records = {r["source_id"]: r for r in dataset["records"]}
    assert Counter(r["split"] for r in records.values()) == training["splits"]

    accepted = read("examples/harmony/local_voicing_v3_acceptance.json")
    assert digest(accepted["model_file"]) == accepted["model_sha256"] == training["frozen_voicing_sha256"]
    for item in accepted["files"]:
        assert digest(item["path"]) == item["sha256"]
    old = read("examples/harmony/test1_d_major_4563.json")
    for path, expected in [("test/test1.mid", old["melody_sha256"]),
                           ("data/harmony_v2_review/block_midis/test1_pickup_4563.mid", old["audition_acceptance"]["midi_sha256"]),
                           ("data/harmony_v2_review/review/test1_pickup_4563.wav", old["audition_acceptance"]["audio_sha256"])]:
        assert digest(path) == expected
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(model_dir / "final_model").to(device).eval()
    voicing_model = GPT2LMHeadModel.from_pretrained(training["args"]["voicing_model"]).to(device).eval()
    cases = {c["song"]: c for c in evaluation["cases"]}
    checks, artifact_hashes, cut_counts = [], {}, Counter()
    pooled = {split: {mode: [] for mode in ("learned", "fixed_arpeggio")} for split in ("train", "validation", "test")}
    raw_pooled = copy.deepcopy(pooled)

    for piece in annotation["pieces"] + [{"song": "test1"}]:
        song = piece["song"]
        case = cases[song]
        if song == "test1":
            source = hp.pretty_midi.PrettyMIDI("test/test1.mid")
            plan = read("data/harmony_v2_review/plans/test1_pickup_4563.json")
            assert digest("test/test1.mid") == plan["melody_sha256"]
            labels = [l for l in plan["labels"] if l["status"] == "chord"]
            melody = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
                       "end_tick": int(source.time_to_tick(n.end))} for i in source.instruments if not i.is_drum for n in i.notes]
            origin, end = 0, plan["duration_ticks"]
        else:
            source = hp.pretty_midi.PrettyMIDI(piece["source_file"])
            original = load_piece(Path(piece["source_file"]), Path(piece["source_file"]).parent.parent)
            origin, end, labels = piece["origin_tick"], piece["end_tick"], piece["labels"]
            for role in ("melody", "accompaniment"):
                source_notes = [{**n, "start_tick": int(source.time_to_tick(n["start"])), "end_tick": int(source.time_to_tick(n["end"]))}
                                for n in original["notes"] if n["role"] == role]
                assert tuples(source_notes, origin, end) == tuples([n for n in piece["notes"] if n["role"] == role], origin, end)
            melody = [n for n in piece["notes"] if n["role"] == "melody"]
        assert (case["origin_tick"], case["end_tick"]) == (origin, end)
        assert len(case["rows"]) == len(labels)
        previous, previous_voicing, previous_end = [], lv.INITIAL, None
        generated, baseline = [], []
        for row, label in zip(case["rows"], labels):
            assert (row["start_tick"], row["end_tick"], row["symbol"]) == (label["start_tick"], label["end_tick"], label["symbol"])
            if previous_end != label["start_tick"]:
                previous, previous_voicing = [], lv.INITIAL
            summary = lv.melody_summary(melody, label)
            assert predict_voicing(voicing_model, label, summary, previous_voicing) == row["voicing"]
            assert predict(model, label, melody, source.resolution, previous) == row["frames"]
            pitch = lv.assess(row["voicing"], label, summary)
            assert pitch == row["pitch_metrics"] and all(pitch.values())
            learned_notes = rv.render(label, row["voicing"], row["frames"], source.resolution, summary["velocity"])
            fixed_notes = lv.schedule(label, row["voicing"], source.resolution, "arpeggio", summary["velocity"])
            generated.extend(learned_notes)
            baseline.extend(fixed_notes)
            actual_frames = realized_frames(label, learned_notes[1:], source.resolution)
            cut_counts[song] += sum(actual_frames[i]["durations"][lane] < f["durations"][lane]
                                    for i, f in enumerate(row["frames"]) for lane in range(3) if f["mask"] & (1 << lane))
            if song != "test1":
                reference = records[f"{song}:{label['source_label_index']}"]["frames"]
                for mode, frames, rendered in (("learned", row["frames"], actual_frames),
                                                ("fixed_arpeggio", rv.baseline_frames(label, row["voicing"], source.resolution),
                                                 realized_frames(label, fixed_notes[1:], source.resolution))):
                    raw = rv.onset_metrics(frames, reference)
                    assert raw == row[mode + "_metrics"]
                    raw_pooled[piece["split"]][mode].append(raw)
                    pooled[piece["split"]][mode].append(rv.onset_metrics(rendered, reference))
            previous, previous_voicing, previous_end = row["frames"], row["voicing"], label["end_tick"]
        assert generated == case["generated_notes"] and baseline == case["baseline_notes"]
        for style, notes in (("learned", generated), ("fixed", baseline)):
            path = out / "evaluation" / f"{song}_{style}.mid"
            actual = hp.pretty_midi.PrettyMIDI(str(path))
            for name, expected in (("MELODY", melody), ("ACCOMPANIMENT", notes)):
                exported = sorted((n.pitch, n.velocity, int(actual.time_to_tick(n.start)), int(actual.time_to_tick(n.end)))
                                  for i in actual.instruments if i.name == name for n in i.notes)
                assert tuples(expected, origin, end) == exported, (song, style, name)
            assert actual.resolution == source.resolution
            assert metadata(source, origin, end) == metadata(actual, 0, end - origin)
            assert all(any(l["start_tick"] <= n["start_tick"] < n["end_tick"] <= l["end_tick"] for l in labels) for n in notes)
            with wave.open(str(path.with_suffix(".wav")), "rb") as audio:
                assert audio.getnchannels() == 2 and audio.getsampwidth() == 2 and audio.getnframes() > 0
                samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                assert np.any(samples)
                seconds = audio.getnframes() / audio.getframerate()
            for artifact in (path, path.with_suffix(".wav")):
                artifact_hashes[artifact.as_posix()] = digest(artifact)
            checks.append({"song": song, "style": style, "melody_exact": True, "raw_predictions_replayed": True,
                           "export_matches_prediction_and_render_rules": True, "tempo_meter_key_and_ppq_exact": True,
                           "unknown_gaps_silent": True, "audio_seconds": seconds})
    assert (out / "evaluation/test1_fixed.mid").read_bytes() == Path("data/trusted_harmony_v3/evaluation/test1_arpeggio.mid").read_bytes()
    recomputed = {s: {m: rv.aggregate_metrics(v) for m, v in modes.items()} for s, modes in raw_pooled.items()}
    assert recomputed == evaluation["metrics"]
    for folder in ("review", "evaluation"):
        page = out / folder / "index.html"
        for link in re.findall(r"(?:src|href)='([^']+)'", page.read_text(encoding="utf-8")):
            assert (page.parent / link).is_file(), link
    projection = Counter()
    for record in records.values():
        projection.update({k: v for k, v in record["projection_audit"].items() if k not in ("max_onset_error_quarters", "bass_pitch_separated", "target_source")})
    audition_status = "pending"
    acceptance_path = Path("examples/harmony/local_rhythm_v4_acceptance.json")
    if acceptance_path.is_file():
        audition = read(acceptance_path)
        # 试听结论仅绑定这次检查点和产物，不能自动转移给另一次训练。
        if Path(audition["model_file"]).resolve() == (model_dir / "final_model/model.safetensors").resolve():
            assert digest(audition["model_file"]) == audition["model_sha256"]
            assert digest(audition["frozen_voicing_file"]) == audition["frozen_voicing_sha256"]
            for item in audition["files"] + audition["comparison_files"]:
                assert item["path"] in artifact_hashes and artifact_hashes[item["path"]] == item["sha256"]
            audition_status = "test1_learned_accepted_other_cases_pending"
    result = {"protocol": rv.VERSION, "source_and_annotation_hashes_verified": True, "source_note_tracks_exact": True,
              "real_targets_independent_of_synthetic_teacher": True, "v3_accepted_media_and_model_unchanged": True,
              "test1_fixed_midi_identical_to_accepted_v3": True, "local_page_links_valid": True,
              "rhythm_model_sha256": digest(model_dir / "final_model/model.safetensors"),
              "projection_audit": dict(projection), "rendered_prediction_retrigger_cuts": dict(cut_counts),
              "rendered_duration_metrics": {s: {m: rv.aggregate_metrics(v) for m, v in modes.items()} for s, modes in pooled.items()},
              "midi_audio_cases": checks, "artifact_sha256": artifact_hashes, "human_audition_status": audition_status}
    write_json(out / "validation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("artifact_sha256", "midi_audio_cases")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
