"""Paired V1.1 / TSD / constrained TSD evaluation, with long-melody controls."""

from __future__ import annotations

import argparse
from collections import Counter
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
from generate_from_scratch import generate_target, generate_windowed, sanitize_accompaniment, model_protocol
from evaluate_token_protocol_v1 import coverage, read_json
from token_protocol_v1 import TYPED_VERSION, TSD_VERSION


def same_pitch_overlaps(notes):
    ends = {}
    count = 0
    for note in sorted(notes, key=lambda n: n["start"]):
        pitch = note["pitch"]
        count += ends.get(pitch, -1) > note["start"]
        ends[pitch] = max(ends.get(pitch, -1), note["end"])
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", default="model_output/accompaniment_gpt2_v1_1_e20/final_model")
    parser.add_argument("--candidate-model", default="model_output/accompaniment_gpt2_tsd_v2_e20/final_model")
    parser.add_argument("--baseline-dataset", default="data/training_windows_v1_1/dataset_windows_v1.json")
    parser.add_argument("--candidate-dataset", default="data/training_windows_tsd_v2/dataset_windows_v1.json")
    parser.add_argument("--out-dir", default="data/tsd_v2_audit/evaluation")
    parser.add_argument("--melody-midi", default="test/test1.mid")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42])
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    args.temperature, args.top_k, args.top_p = 0.7, 30, 0.9
    args.min_accomp_pitch, args.max_accomp_pitch = 36, 84
    args.min_note_seconds, args.max_note_seconds = 0.06, 3.0
    args.max_simultaneous_notes, args.melody_clearance = 4, 1
    args.bar_seconds, args.no_fill_empty_bars = 2.0, False
    args.window_seconds = 8.0
    budget = args.max_new_tokens
    baseline, candidate = read_json(args.baseline_dataset), read_json(args.candidate_dataset)
    old_meta = read_json(Path(args.baseline_model).parent / "training_metadata.json")
    new_meta = read_json(Path(args.candidate_model).parent / "training_metadata.json")
    if old_meta["split"] != new_meta["split"]:
        raise ValueError("Training/evaluation piece split changed")
    candidate_by_id = {s["piece_id"]: s for s in candidate["samples"]}
    cases = []
    for piece_id in old_meta["split"]["eval_pieces"]:
        samples = [s for s in baseline["samples"] if s["source_piece_id"] == piece_id]
        for index in (0, len(samples) // 2):
            old = samples[index]
            new = candidate_by_id[old["piece_id"]]
            for key in ("start_time", "end_time", "melody_note_count", "accompaniment_note_count"):
                if old[key] != new[key]:
                    raise ValueError(f"Unpaired reference window: {old['piece_id']}")
            cases.append({"name": old["piece_id"], "group": "heldout_window", "old": old, "new": new})
    piece = load_midi_notes(Path(args.melody_midi))
    for note in piece["notes"]:
        note["role"] = "melody"
    cases.append({"name": "standalone_melody", "group": "long_melody", "piece": piece,
                  "old": piece_to_sample(piece, 10, TYPED_VERSION), "new": piece_to_sample(piece, 10, TSD_VERSION)})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists():
        raise ValueError("Choose a new output directory; existing results will not be overwritten")
    rows = []
    for path, conditions in ((args.baseline_model, ["v1_1"]),
                             (args.candidate_model, ["tsd_unconstrained", "tsd_constrained", "tsd_larger_budget", "tsd_windowed"])):
        model = GPT2LMHeadModel.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        version, quantum = model_protocol(Path(path), model)
        if version != (TYPED_VERSION if conditions == ["v1_1"] else TSD_VERSION) or quantum != 10:
            raise ValueError("Unexpected model protocol")
        for condition in conditions:
            args.no_grammar_constraints = condition == "tsd_unconstrained"
            args.max_new_tokens = budget * 2 if condition == "tsd_larger_budget" else budget
            folder = out_dir / condition
            folder.mkdir(exist_ok=True)
            for case in cases:
                if condition in ("tsd_larger_budget", "tsd_windowed") and case["group"] != "long_melody":
                    continue
                sample = case["old" if version == TYPED_VERSION else "new"]
                source = sample["source_tokens"]
                melody = tokens_to_notes(source, 10, version)
                duration = max(n["end"] for n in melody)
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    windows = None
                    if condition == "tsd_windowed":
                        _, raw, windows = generate_windowed(model, case["piece"], args, version)
                        tokens = [token for w in windows for token in w["generated_target_tokens"]]
                        reports = [w["generation_report"] for w in windows]
                    else:
                        tokens, report = generate_target(model, [TOKEN["BOS"], *source, TOKEN["SEP"]], args,
                                                         version, target_duration=duration)
                        reports = [report]
                        raw = tokens_to_notes(tokens, 10, version)
                    for note in raw:
                        note["role"] = "accompaniment"
                    cleaned, sanitation = sanitize_accompaniment(melody, raw, args)
                    last_end = max((n["end"] for n in raw), default=0.0)
                    invalid = any(r.get(key, 0) for r in reports for key in
                                  ("unknown_tokens", "invalid_parameter_events", "incomplete_events", "wrong_role_note_events"))
                    row = {"condition": condition, "case": case["name"], "group": case["group"], "seed": seed,
                           "melody_duration": duration, "raw_coverage": coverage(raw, duration),
                           "raw_notes": len(raw), "raw_notes_over_3s": sum(n["end"] - n["start"] > 3 for n in raw),
                           "raw_same_pitch_overlaps": same_pitch_overlaps(raw),
                           "raw_tail_gap_seconds": max(0.0, duration - last_end),
                           "raw_overshoot_seconds": max(0.0, last_end - duration),
                           "syntax_valid": not invalid, "generated_tokens": len(tokens),
                           "generation_reports": reports, "sanitation_report": sanitation,
                           "total_effective_token_budget": sum(r.get("effective_max_new_tokens", 0) for r in reports)}
                    rows.append(row)
                    payload = {**row, "tokenizer_version": version, "generated_target_tokens": tokens,
                               "windows": windows, "melody_notes": melody, "raw_accompaniment_notes": raw,
                               "sanitized_accompaniment_notes": cleaned}
                    stem = folder / f"{case['name']}_seed{seed}"
                    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    write_notes_midi(melody + cleaned, stem.with_suffix(".mid"))
                    print(f"{condition} {case['name']} seed={seed} coverage={row['raw_coverage']:.3f} valid={not invalid}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregates = []
    for condition, group in sorted({(r["condition"], r["group"]) for r in rows}):
        selected = [r for r in rows if r["condition"] == condition and r["group"] == group]
        result = {"condition": condition, "group": group, "runs": len(selected),
                  "syntax_valid_runs": sum(r["syntax_valid"] for r in selected),
                  "stop_reasons": dict(Counter(g["stop_reason"] for r in selected for g in r["generation_reports"])),
                  "forced_eos": dict(Counter(g["forced_eos_reason"] for r in selected for g in r["generation_reports"] if g.get("forced_eos_reason")))}
        for key in ("raw_coverage", "raw_notes", "raw_notes_over_3s", "raw_same_pitch_overlaps", "raw_tail_gap_seconds", "raw_overshoot_seconds", "generated_tokens", "total_effective_token_budget"):
            result[f"mean_{key}"] = statistics.mean(r[key] for r in selected)
        aggregates.append(result)
    args.max_new_tokens = budget
    summary = {"settings": vars(args), "aggregates": aggregates, "runs": rows,
               "limitations": ["One training seed; four held-out pieces and one long melody.",
                               "Windowed generation uses a separate token budget per window; larger single-pass budget is a separate control.",
                               "Constrained TSD enforces both event syntax and the requested time horizon.",
                               "Zero missing NOTE_OFF events follows from duration representation, not proven musical quality.",
                               "Coverage is inflated by sustained notes; no subjective listening evaluation."]}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
