"""Post-hoc prompt probes on frozen checkpoints; no training or promotion."""

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel
from build_harmony_review import write_json
import harmony_plan as hp
import rhythm_v5 as v5
from train_rhythm_v4 import predict as predict_v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/rhythm_naturalness_audit/context_probes.json")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError("Do not overwrite the recorded probes")
    evaluation = json.loads(Path("data/trusted_harmony_v5/evaluation/evaluation.json").read_text(encoding="utf-8"))
    case = next(c for c in evaluation["cases"] if c["song"] == "test1")
    plan = json.loads(Path("data/harmony_v2_review/plans/test1_pickup_4563.json").read_text(encoding="utf-8"))
    labels = [l for l in plan["labels"] if l["status"] == "chord"]
    source = hp.pretty_midi.PrettyMIDI("test/test1.mid")
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
               "end_tick": int(source.time_to_tick(n.end))} for i in source.instruments if not i.is_drum for n in i.notes]
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports = {}
    for method in ("accepted_v4", "expanded_v4", "v5_medium"):
        model_key = "v5" if method == "v5_medium" else method
        model = GPT2LMHeadModel.from_pretrained(evaluation["models"][model_key]["path"]).to(device).eval()
        predict = v5.predict if method == "v5_medium" else predict_v4
        rows = []
        for index, (label, row) in enumerate(zip(labels, case["rows"])):
            history = case["rows"][index - 1]["methods"][method]["frames"] if index else []
            old_history = case["rows"][index - 1]["methods"]["accepted_v4"]["frames"] if index else []
            baseline = row["methods"][method]["frames"]
            assert predict(model, label, melody, source.resolution, history) == baseline
            record = {"start_tick": label["start_tick"], "symbol": label["symbol"], "probes": {}}
            for probe, current_melody, previous in (("remove_melody_condition", [], history),
                                                     ("empty_previous_history", melody, []),
                                                     ("silent_same_length_history", melody, [{**f, "mask": 0} for f in history]),
                                                     ("reverse_same_length_history", melody, list(reversed(history))),
                                                     ("dense_same_length_history", melody, [{**f, "mask": 7} for f in history]),
                                                     ("accepted_v4_previous_history", melody, old_history)):
                changed = predict(model, label, current_melody, source.resolution, previous)
                record["probes"][probe] = {"changed_mask_steps": sum(a["mask"] != b["mask"] for a, b in zip(baseline, changed)),
                    "changed_frames": sum(a != b for a, b in zip(baseline, changed)),
                    "upper_notes": sum(f["mask"].bit_count() for f in changed), "frames": changed}
            rows.append(record)
        totals = {probe: {"changed_mask_steps": sum(r["probes"][probe]["changed_mask_steps"] for r in rows),
                           "spans_with_onset_changes": sum(r["probes"][probe]["changed_mask_steps"] > 0 for r in rows),
                           "spans_with_any_changes": sum(r["probes"][probe]["changed_frames"] > 0 for r in rows),
                           "upper_notes": sum(r["probes"][probe]["upper_notes"] for r in rows)} for probe in rows[0]["probes"]}
        reports[method] = {"summary": totals, "rows": rows}
        print(json.dumps({"model": method, "summary": totals}), flush=True)
    write_json(out, {"scope": "post_hoc_test1_prompt_diagnostics; not independent validation or musical quality evidence",
        "history_policy": "Each span uses the saved original rollout history unless that field is the intervention; probe outputs are not fed forward.",
        "position_caveat": "Empty history shortens the prefix and shifts absolute positions. Same-length probes separate content changes from this confound.",
        "melody_policy": "Remove only the rhythm model's melody tokens; actual melody and frozen voicing are not regenerated.",
        "models": evaluation["models"], "reports": reports})


if __name__ == "__main__":
    main()
