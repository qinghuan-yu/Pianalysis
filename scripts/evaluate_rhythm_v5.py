"""Compare frozen V4, expanded V4 and three V5 density settings on identical inputs."""

from collections import Counter
import html
import json
from pathlib import Path
import wave

import numpy as np
import torch
from transformers import GPT2LMHeadModel
from build_harmony_review import write_json, write_preview_audio
from import_pop909 import load_piece
import harmony_plan as hp
import local_voicing as lv
import rhythm_v4 as v4
import rhythm_v5 as v5
from train_local_voicing import validate_dataset, predict as predict_voicing
from train_rhythm_v4 import predict as predict_v4
from verify_rhythm_v4 import digest, metadata, tuples, realized_frames

METHODS = ("accepted_v4", "expanded_v4", "v5_sparse", "v5_medium", "v5_dense", "fixed")
TITLES = {"accepted_v4": "已验收V4节奏", "expanded_v4": "仅扩充数据", "v5_sparse": "新模型 · 稀疏",
          "v5_medium": "新模型 · 中等", "v5_dense": "新模型 · 密集", "fixed": "固定分解"}


def summarize(rows):
    duration = sum(r["quarters"] for r in rows)
    return {"raw": v4.aggregate_metrics([r["raw"] for r in rows]),
            "rendered": v4.aggregate_metrics([r["rendered"] for r in rows]),
            "upper_onsets_per_quarter": sum(r["raw"]["predicted_notes"] for r in rows) / duration,
            "reference_onsets_per_quarter": sum(r["raw"]["reference_notes"] for r in rows) / duration,
            "per_span_density_mae": sum(abs(r["raw"]["predicted_notes"] - r["raw"]["reference_notes"]) / r["quarters"] for r in rows) / len(rows),
            "retrigger_cuts": sum(r["cuts"] for r in rows)}


def verify_export(source, melody, notes, path, origin, end, labels):
    actual = hp.pretty_midi.PrettyMIDI(str(path))
    for name, expected in (("MELODY", melody), ("ACCOMPANIMENT", notes)):
        exported = sorted((n.pitch, n.velocity, int(actual.time_to_tick(n.start)), int(actual.time_to_tick(n.end)))
                          for i in actual.instruments if i.name == name for n in i.notes)
        assert tuples(expected, origin, end) == exported, (path, name)
    assert actual.resolution == source.resolution
    assert metadata(source, origin, end) == metadata(actual, 0, end - origin)
    assert all(any(l["start_tick"] <= n["start_tick"] < n["end_tick"] <= l["end_tick"] for l in labels) for n in notes)
    return actual


def main():
    out = Path("data/trusted_harmony_v5/evaluation")
    if out.exists():
        raise ValueError("Choose a fresh evaluation output; do not overwrite results")
    data_path = Path("data/trusted_harmony_v5/annotations.json")
    annotation = json.loads(data_path.read_text(encoding="utf-8"))
    validate_dataset(annotation)
    records = {r["source_id"]: r for r in v4.build_dataset(annotation)["records"]}
    plan = json.loads(Path("examples/harmony/rhythm_v5_experiment.json").read_text(encoding="utf-8"))
    models_paths = {"accepted_v4": "model_output/local_rhythm_v4/final_model",
                    "expanded_v4": "model_output/local_rhythm_v4_expanded/final_model", "v5": "model_output/local_rhythm_v5/final_model"}
    for key in ("expanded_v4", "v5"):
        info = json.loads((Path(models_paths[key]).parent / "training_metadata.json").read_text(encoding="utf-8"))
        assert info["annotations_sha256"] == digest(data_path)
        assert info["experiment_sha256"] == digest("examples/harmony/rhythm_v5_experiment.json")
    accepted_files = []
    for file in ("local_voicing_v3_acceptance.json", "local_rhythm_v4_acceptance.json"):
        accepted = json.loads((Path("examples/harmony") / file).read_text(encoding="utf-8"))
        assert digest(accepted["model_file"]) == accepted["model_sha256"]
        for item in accepted["files"] + accepted.get("comparison_files", []):
            assert digest(item["path"]) == item["sha256"]
            accepted_files.append(item)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {key: GPT2LMHeadModel.from_pretrained(path).to(device).eval() for key, path in models_paths.items()}
    voicing_model = GPT2LMHeadModel.from_pretrained(plan["frozen_voicing"]).to(device).eval()
    source = hp.pretty_midi.PrettyMIDI("test/test1.mid")
    external = json.loads(Path("data/harmony_v2_review/plans/test1_pickup_4563.json").read_text(encoding="utf-8"))
    assert digest("test/test1.mid") == external["melody_sha256"]
    cases = list(annotation["pieces"]) + [{"song": "test1", "split": "external", "source_file": "test/test1.mid",
        "ppq": source.resolution, "origin_tick": 0, "end_tick": external["duration_ticks"], "labels": external["labels"],
        "notes": [{"role": "melody", "pitch": n.pitch, "velocity": n.velocity,
                   "start_tick": int(source.time_to_tick(n.start)), "end_tick": int(source.time_to_tick(n.end))}
                  for i in source.instruments if not i.is_drum for n in i.notes]}]
    reports, hashes, checks = [], {}, []
    pooled = {split: {m: [] for m in METHODS} for split in ("train", "validation", "test")}
    out.mkdir(parents=True)
    for piece in cases:
        song, split = piece["song"], piece["split"]
        source = hp.pretty_midi.PrettyMIDI(piece["source_file"])
        melody = [n for n in piece["notes"] if n["role"] == "melody"]
        labels = [l for l in piece["labels"] if l["status"] == "chord"]
        origin, end = piece["origin_tick"], piece["end_tick"]
        if split != "external":
            raw = load_piece(Path(piece["source_file"]), Path(piece["source_file"]).parent.parent)
            for role in ("melody", "accompaniment"):
                original = [{**n, "start_tick": int(source.time_to_tick(n["start"])), "end_tick": int(source.time_to_tick(n["end"]))}
                            for n in raw["notes"] if n["role"] == role]
                assert tuples(original, origin, end) == tuples([n for n in piece["notes"] if n["role"] == role], origin, end)
        previous = {m: [] for m in METHODS}
        previous_voicing, previous_end = lv.INITIAL, None
        notes = {m: [] for m in METHODS}
        metrics = {m: [] for m in METHODS}
        rows, profile_counts = [], {m: 0 for m in METHODS}
        quarters = sum((l["end_tick"] - l["start_tick"]) / piece["ppq"] for l in labels)
        for label in labels:
            if label["start_tick"] != previous_end:
                previous, previous_voicing = {m: [] for m in METHODS}, lv.INITIAL
            summary = lv.melody_summary(melody, label)
            voicing = predict_voicing(voicing_model, label, summary, previous_voicing)
            assert all(lv.assess(voicing, label, summary).values())
            row = {"start_tick": label["start_tick"], "end_tick": label["end_tick"], "symbol": label["symbol"],
                   "voicing": voicing, "methods": {}}
            for method in METHODS:
                if method.startswith("v5_"):
                    frames = v5.predict(models["v5"], label, melody, piece["ppq"], previous[method], method[3:])
                elif method == "fixed":
                    frames = v4.baseline_frames(label, voicing, piece["ppq"])
                else:
                    frames = predict_v4(models[method], label, melody, piece["ppq"], previous[method])
                rendered = (lv.schedule(label, voicing, piece["ppq"], "arpeggio", summary["velocity"]) if method == "fixed"
                            else v4.render(label, voicing, frames, piece["ppq"], summary["velocity"]))
                actual = realized_frames(label, rendered[1:], piece["ppq"])
                cuts = sum(actual[i]["durations"][lane] < f["durations"][lane]
                           for i, f in enumerate(frames) for lane in range(3) if f["mask"] & (1 << lane))
                if method.startswith("v5_"):
                    assert cuts == 0
                profile_counts[method] += len(rendered) - 1
                row["methods"][method] = {"frames": frames, "notes": rendered, "retrigger_cuts": cuts}
                if split != "external":
                    ref = records[f"{song}:{label['source_label_index']}"]["frames"]
                    values = {"raw": v4.onset_metrics(frames, ref), "rendered": v4.onset_metrics(actual, ref),
                              "quarters": (label["end_tick"] - label["start_tick"]) / piece["ppq"], "cuts": cuts}
                    metrics[method].append(values)
                    pooled[split][method].append(values)
                    row["methods"][method]["metrics"] = values
                notes[method].extend(rendered)
                previous[method] = frames
            rows.append(row)
            previous_voicing, previous_end = voicing, label["end_tick"]
        for method in METHODS:
            path = out / f"{song}_{method}.mid"
            lv.write_midi(source, melody, notes[method], path, origin, end)
            actual = verify_export(source, melody, notes[method], path, origin, end, labels)
            hashes[path.as_posix()] = digest(path)
            if song in ("test1", "011", "012"):
                write_preview_audio(actual, path.with_suffix(".wav"))
                with wave.open(str(path.with_suffix(".wav")), "rb") as stream:
                    assert stream.getnchannels() == 2 and stream.getsampwidth() == 2
                    assert np.any(np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2"))
                hashes[path.with_suffix(".wav").as_posix()] = digest(path.with_suffix(".wav"))
            checks.append({"song": song, "method": method, "melody_exact": True, "accompaniment_exact": True,
                           "tempo_meter_key_ppq_exact": True, "unknown_gaps_silent": True})
        density = {m: profile_counts[m] / quarters for m in METHODS}
        reports.append({"song": song, "split": split, "origin_tick": origin, "end_tick": end, "rows": rows,
                        "metrics": {m: summarize(values) for m, values in metrics.items()} if split != "external" else {},
                        "upper_onsets_per_quarter": density,
                        "density_ordered": density["v5_sparse"] <= density["v5_medium"] <= density["v5_dense"],
                        "density_strict": density["v5_sparse"] < density["v5_medium"] < density["v5_dense"]})
        print(json.dumps({"song": song, "segments": len(labels), "density": density,
                          "f1": {m: summarize(v)["raw"]["onset_f1"] for m, v in metrics.items()} if split != "external" else None}), flush=True)
    assert (out / "test1_accepted_v4.mid").read_bytes() == Path("data/trusted_harmony_v4/evaluation/test1_learned.mid").read_bytes()
    assert (out / "test1_fixed.mid").read_bytes() == Path("data/trusted_harmony_v3/evaluation/test1_arpeggio.mid").read_bytes()
    result = {"version": "rhythm_v5_comparison", "annotations_sha256": digest(data_path),
              "experiment_sha256": digest("examples/harmony/rhythm_v5_experiment.json"),
              "models": {k: {"path": p, "sha256": digest(Path(p) / "model.safetensors")} for k, p in models_paths.items()},
              "metrics": {s: {m: summarize(v) for m, v in methods.items()} for s, methods in pooled.items()},
              "cases": reports, "inference_density_source": "fixed_caller_profiles_never_reference", "human_audition_status": "pending"}
    write_json(out / "evaluation.json", result)
    write_json(out.parent / "validation.json", {"accepted_media_and_weights_unchanged": True, "test1_old_outputs_identical": True,
        "source_notes_and_hashes_verified": True, "midi_cases": checks, "artifact_sha256": hashes,
        "v5_raw_duration_equals_rendered_duration": True, "density_monotone_cases": sum(c["density_ordered"] for c in reports),
        "density_strict_cases": sum(c["density_strict"] for c in reports), "total_cases": len(reports), "human_audition_status": "pending"})
    sections = []
    for song in ("test1", "012", "011"):
        case = next(c for c in reports if c["song"] == song)
        audio = "".join(f"<div><h3>{TITLES[m]} · 上方每拍{case['upper_onsets_per_quarter'][m]:.2f}音</h3><audio controls preload='none' src='{song}_{m}.wav'></audio><a href='{song}_{m}.mid'>MIDI</a></div>" for m in METHODS)
        sections.append(f"<section><h2>{song} · {case['split']}</h2>{audio}</section>")
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>扩充曲目与节奏控制</title><style>body{max-width:960px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}section{border-top:1px solid #aaa;margin-top:30px}audio{width:100%}h3{font-size:17px}</style><h1>扩充曲目与节奏控制</h1><p>已验收V4作为对照保留。新模型先预测全部起音，再预测时值；稀疏、中等、密集是输入偏好，不保证每段严格达到指定密度。和声配位、弱起和原旋律沿用已核验方案。测试曲目不参与训练，试听仍待确认。</p><p><a href='../review/index.html'>原编配与195段和声核对</a></p>" + "".join(sections) + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({s: {m: {"f1": v["raw"]["onset_f1"], "duration_mae": v["rendered"]["duration_mae_quarters_on_matched_onsets"], "density_mae": v["per_span_density_mae"]} for m, v in methods.items()} for s, methods in result["metrics"].items()}), flush=True)


if __name__ == "__main__":
    main()
