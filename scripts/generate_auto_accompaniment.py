"""仅使用标准样本的旋律和拍速信息，串联自动和声与伴奏模型。"""

import argparse
import copy
import json
from pathlib import Path
import time

import torch

import auto_harmony as harmony
import auto_texture as texture


def melody_excerpt(row, start_bar, bars):
    origin = row.get("metadata", {}).get("bar_origin_beat")
    if origin is None or start_bar < 0 or bars < 1:
        raise ValueError("Known bar origin and a positive excerpt length are required")
    if any((e["numerator"], e["denominator"]) != (4, 4) for e in row["time_signatures"]):
        raise ValueError("The current texture adapter supports 4/4 only")
    start, end = origin + start_bar * 4, origin + (start_bar + bars) * 4
    if start < 0 or end > row["duration_beats"] + 1e-7:
        raise ValueError("Excerpt exceeds source timeline")
    if any(n["start"] < start - 1e-7 < n["end"] - 1e-7 or n["start"] < end - 1e-7 < n["end"] - 1e-7 for n in row["melody"]):
        raise ValueError("Choose an excerpt without a melody note crossing its boundary; notes are not clipped")
    result = {k: copy.deepcopy(row[k]) for k in ("schema_version", "song_id", "work_id", "source", "split", "ppq")}
    result["melody"] = [{**n, "start": n["start"] - start, "end": n["end"] - start} for n in row["melody"] if start <= n["start"] < end]
    if not result["melody"]:
        raise ValueError("No melody in the selected excerpt")
    # 不复制参考和声、PIANO、调性标签或伴奏派生特征；仅传入旋律与拍速信息。
    result["accompaniment"] = []
    result["chords"] = [{"start": 0.0, "end": end-start, "root": None, "bass": None,
                         "quality": "unknown", "symbol": "X", "pitch_classes": [],
                         "status": "unknown", "provenance": "withheld_reference_for_melody_only_inference"}]
    for field in ("tempos", "time_signatures"):
        events = sorted(row[field], key=lambda e: e["beat"])
        previous = [e for e in events if e["beat"] <= start]
        if not previous:
            raise ValueError("Missing " + field + " at excerpt start")
        result[field] = [{**previous[-1], "beat": 0.0}] + [{**e, "beat": e["beat"] - start} for e in events if start < e["beat"] < end]
    result["duration_beats"] = end - start
    result["metadata"] = {"bar_origin_beat": 0.0, "excerpt_source_start_beat": start,
                          "reference_harmony_withheld": True, "reference_accompaniment_withheld": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--start-bar", type=int, default=0)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--harmony-checkpoint")
    parser.add_argument("--texture-checkpoint", required=True)
    parser.add_argument("--texture-config", default=str(texture.DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bars > 16:
        raise ValueError("First pipeline supports at most 16 bars; longer structure is not validated")
    torch.set_num_threads(4)
    samples_hash = harmony.sha256_file(Path(args.samples))
    rows = harmony.read_samples(args.samples)
    if harmony.sha256_file(Path(args.samples)) != samples_hash:
        raise ValueError("Input samples changed while loading")
    row = next((r for r in rows if r["song_id"] == args.song_id), None)
    if row is None:
        raise ValueError("Unknown song_id")
    clean = melody_excerpt(row, args.start_bar, args.bars)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    harmony.write_json(out.resolve() / "input.json", clean)
    started = time.monotonic()
    texture.seed_all(args.seed)
    model, vocabulary, metadata, payload = harmony.load_model(checkpoint=args.harmony_checkpoint)
    del payload
    model.to(args.device)
    plan = harmony.generate_plan(model, clean, vocabulary, texture_compatible_chords=True)
    plan["metadata"]["auto_harmony"]["checkpoint_sha256"] = metadata["checkpoint_sha256"]
    harmony.write_json(out.resolve() / "harmony.json", plan)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    config = texture.read_config(args.texture_config)
    vocab = texture.Vocabulary(config["upstream_dir"])
    renderer, loading = texture.load_model(vocab, args.texture_checkpoint, args.device)
    if loading.get("protocol") != texture.PROTOCOL or loading.get("optimizer_steps", 0) < 1:
        raise ValueError("Require an accompaniment-only adapted checkpoint, not the original full-performance checkpoint")
    config = loading.get("config", config)
    encoded = texture.Encoder(vocab, config).encode(plan, 0.0, args.bars, require_target=False)
    notes, generation = texture.generate(renderer, vocab, plan, encoded, seed=args.seed)
    preserved = texture.export_midi(plan, notes, 0.0, args.bars*4, out / "output.mid")
    if preserved != clean["melody"]:
        raise RuntimeError("Original melody was changed during generation")
    prediction = copy.deepcopy(plan)
    prediction["accompaniment"] = notes
    prediction["metadata"]["auto_texture"] = {"checkpoint_sha256": loading["checkpoint_sha256"], "seed": args.seed,
                                                "bass_condition_supported": False, "generation": generation}
    texture.write_json(out / "prediction.json", prediction)
    texture.write_json(out / "report.json", {"song_id": args.song_id, "source_split": row["split"],
                         "source_samples_sha256": samples_hash,
                         "seconds": time.monotonic() - started, "melody_notes": len(preserved),
                         "accompaniment_notes": len(notes), "canonical_melody_preserved": True,
                         "reference_harmony_withheld": True, "reference_accompaniment_withheld": True,
                         "harmony_vocabulary": "explicitly restricted to exact pitch-class sets supported by texture",
                         "quality_status": "Unreviewed; short-step smoke checkpoints do not demonstrate quality"})
    print(json.dumps({"output": str(out), "accompaniment_notes": len(notes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
