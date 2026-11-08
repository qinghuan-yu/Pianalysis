"""Paired, local V1/V1.1 generation experiment on held-out melodies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import GPT2LMHeadModel
from closed_loop_v1 import TOKEN, load_midi_notes, piece_to_sample, tokens_to_notes, write_notes_midi
from generate_from_scratch import (
    event_diagnostics, first_eos_position, generate_target, model_protocol, sanitize_accompaniment,
)
from token_protocol_v1 import LEGACY_VERSION, TYPED_VERSION, decode_to_legacy_events


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coverage(notes, duration):
    if duration <= 0:
        return 0.0
    covered = 0.0
    right = 0.0
    for note in sorted(notes, key=lambda item: item["start"]):
        start, end = max(0.0, note["start"]), min(duration, note["end"])
        if end > max(right, start):
            covered += end - max(right, start)
            right = end
    return covered / duration


def generate_original(model, prompt, args):
    """Reproduce the original token-wise EOS behavior as an explicit control."""
    budget = min(args.max_new_tokens, model.config.n_positions - len(prompt))
    if budget <= 0:
        raise ValueError("Original control prompt exceeds context")
    inputs = torch.tensor([prompt], device=model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=budget,
            do_sample=True, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            pad_token_id=TOKEN["PAD"], eos_token_id=TOKEN["EOS"],
        )
    generated = output[0, len(prompt):].tolist()
    eos_at = generated.index(TOKEN["EOS"]) if TOKEN["EOS"] in generated else None
    target = generated if eos_at is None else generated[:eos_at]
    boundary_eos = first_eos_position(generated, TOKEN["EOS"])
    return target, {
        **event_diagnostics(target, LEGACY_VERSION),
        "stop_reason": "eos" if eos_at is not None else "context_limit" if budget < args.max_new_tokens else "max_new_tokens",
        "premature_numeric_eos": eos_at is not None and boundary_eos != eos_at,
        "eos_index": eos_at, "generated_tokens_including_eos": len(generated),
        "requested_max_new_tokens": args.max_new_tokens, "effective_max_new_tokens": budget,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", default="model_output/accompaniment_gpt2_e20/final_model")
    parser.add_argument("--candidate-model", default="model_output/accompaniment_gpt2_v1_1_e20/final_model")
    parser.add_argument("--baseline-dataset", default="data/training_windows_v1/dataset_windows_v1.json")
    parser.add_argument("--candidate-dataset", default="data/training_windows_v1_1/dataset_windows_v1.json")
    parser.add_argument("--melody-midi", default="test/test1.mid")
    parser.add_argument("--out-dir", default="data/protocol_v1_1_audit/evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42])
    parser.add_argument("--windows-per-piece", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    if args.windows_per_piece <= 0:
        parser.error("--windows-per-piece must be positive")
    # 与既有 conservative 示例一致；原始生成指标在任何后处理之前计算。
    args.min_accomp_pitch = 36
    args.max_accomp_pitch = 84
    args.min_note_seconds = 0.06
    args.max_note_seconds = 3.0
    args.max_simultaneous_notes = 4
    args.melody_clearance = 1
    args.bar_seconds = 2.0
    args.no_fill_empty_bars = False

    baseline = read_json(args.baseline_dataset)
    candidate = read_json(args.candidate_dataset)
    baseline_meta = read_json(Path(args.baseline_model).parent / "training_metadata.json")
    candidate_meta = read_json(Path(args.candidate_model).parent / "training_metadata.json")
    if baseline_meta["split"] != candidate_meta["split"]:
        raise ValueError("Model train/eval piece splits differ")
    eval_pieces = baseline_meta["split"]["eval_pieces"]
    candidate_by_id = {sample["piece_id"]: sample for sample in candidate["samples"]}
    groups = defaultdict(list)
    for sample in baseline["samples"]:
        if sample["source_piece_id"] in eval_pieces:
            groups[sample["source_piece_id"]].append(sample)
    cases = []
    for piece in eval_pieces:
        samples = groups[piece]
        for index in sorted({min(len(samples) - 1, n * len(samples) // args.windows_per_piece) for n in range(args.windows_per_piece)}):
            old = samples[index]
            new = candidate_by_id[old["piece_id"]]
            for field in ("source_tokens", "target_tokens"):
                if old[field] != decode_to_legacy_events(new[field], TYPED_VERSION):
                    raise ValueError(f"Unpaired sample: {old['piece_id']}")
            if old.get("harmony_tokens") or new.get("harmony_tokens"):
                raise ValueError("This experiment controls for melody-only conditioning")
            cases.append({"name": old["piece_id"], "group": "heldout_window", "legacy": old, "typed": new})
    if args.melody_midi:
        piece = load_midi_notes(Path(args.melody_midi))
        for note in piece["notes"]:
            note["role"] = "melody"
        cases.append({"name": "standalone_melody", "group": "long_melody",
                      "legacy": piece_to_sample(piece, 10), "typed": piece_to_sample(piece, 10, TYPED_VERSION)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists():
        raise ValueError("Experiment already exists; choose another --out-dir")
    rows = []
    for path, conditions in ((args.baseline_model, ["legacy_original", "legacy_event_eos"]),
                             (args.candidate_model, ["typed_v1_1"])):
        model = GPT2LMHeadModel.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        version, quantum_ms = model_protocol(Path(path), model)
        expected = TYPED_VERSION if conditions == ["typed_v1_1"] else LEGACY_VERSION
        if version != expected or quantum_ms != 10:
            raise ValueError("Unexpected experiment model protocol or time quantum")
        for condition in conditions:
            condition_dir = out_dir / condition
            condition_dir.mkdir(exist_ok=True)
            for case in cases:
                sample = case["typed" if version == TYPED_VERSION else "legacy"]
                source = sample["source_tokens"]
                melody = tokens_to_notes(source, quantum_ms, version)
                duration = max(note["end"] for note in melody)
                prompt = [TOKEN["BOS"], *source, TOKEN["SEP"]]
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    tokens, report = generate_original(model, prompt, args) if condition == "legacy_original" else generate_target(model, prompt, args, version)
                    raw = tokens_to_notes(tokens, quantum_ms, version)
                    for note in raw:
                        note["role"] = "accompaniment"
                    cleaned, sanitation = sanitize_accompaniment(melody, raw, args)
                    last_end = max((note["end"] for note in raw), default=0.0)
                    row = {
                        "condition": condition, "case": case["name"], "group": case["group"], "seed": seed,
                        "tokenizer_version": version, "melody_duration": duration, "generated_tokens": len(tokens),
                        "raw_notes": len(raw), "raw_coverage": coverage(raw, duration),
                        "raw_tail_gap_seconds": max(0.0, duration - last_end),
                        "raw_overshoot_seconds": max(0.0, last_end - duration),
                        "raw_notes_over_3s": sum(note["end"] - note["start"] > 3.0 for note in raw),
                        "sanitized_coverage": coverage(cleaned, duration),
                        "generation_report": report, "sanitation_report": sanitation,
                    }
                    rows.append(row)
                    stem = condition_dir / f"{case['name']}_seed{seed}"
                    payload = {**row, "generated_target_tokens": tokens, "raw_accompaniment_notes": raw,
                               "sanitized_accompaniment_notes": cleaned, "melody_notes": melody}
                    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    write_notes_midi(melody + cleaned, stem.with_suffix(".mid"))
                    print(f"{condition} {case['name']} seed={seed} stop={report['stop_reason']} coverage={row['raw_coverage']:.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregates = []
    for condition in ("legacy_original", "legacy_event_eos", "typed_v1_1"):
        for group in ("heldout_window", "long_melody"):
            selected = [row for row in rows if row["condition"] == condition and row["group"] == group]
            if not selected:
                continue
            item = {"condition": condition, "group": group, "runs": len(selected),
                    "stop_reasons": dict(Counter(row["generation_report"]["stop_reason"] for row in selected)),
                    "premature_numeric_eos": sum(row["generation_report"].get("premature_numeric_eos", False) for row in selected)}
            for key in ("raw_notes", "raw_coverage", "raw_tail_gap_seconds", "raw_overshoot_seconds", "raw_notes_over_3s", "sanitized_coverage"):
                item[f"mean_{key}"] = statistics.mean(row[key] for row in selected)
            for key in ("unknown_tokens", "invalid_parameter_events", "incomplete_events", "unclosed_notes", "orphan_note_offs"):
                item[f"mean_{key}"] = statistics.mean(row["generation_report"][key] for row in selected)
            for key in ("sanitized_accompaniment_notes", "filled_empty_bars", "merged_same_pitch_overlap", "truncated_after_melody"):
                item[f"mean_{key}"] = statistics.mean(row["sanitation_report"][key] for row in selected)
            aggregates.append(item)
    summary = {"settings": vars(args), "eval_pieces": eval_pieces, "aggregates": aggregates, "runs": rows,
               "limitations": ["One training seed; only four held-out pieces.",
                               "Coverage includes sustained notes and does not measure musical quality.",
                               "Long melody is outside the eight-second training distribution.",
                               "Legacy decoding is permissive; typed invalid parameters are rejected.",
                               "No listening evaluation was performed."]}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
