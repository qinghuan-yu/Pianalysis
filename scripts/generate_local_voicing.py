"""Render a melody and explicit tick-based triad plan with the local curriculum model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from build_harmony_review import write_json, write_preview_audio
import harmony_plan as hp
import local_voicing as lv
from train_local_voicing import rollout, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model_output/local_voicing_v3/final_model")
    parser.add_argument("--melody-midi", default="test/test1.mid")
    parser.add_argument("--harmony-plan", default="data/harmony_v2_review/plans/test1_pickup_4563.json")
    parser.add_argument("--out-dir", default="data/trusted_harmony_v3/cli_test1")
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError("Choose a fresh output directory")
    plan = json.loads(Path(args.harmony_plan).read_text(encoding="utf-8"))
    hp.validate_plan(plan)
    digest = hashlib.sha256(Path(args.melody_midi).read_bytes()).hexdigest()
    if plan.get("melody_sha256") and plan["melody_sha256"] != digest:
        raise ValueError("Melody differs from the reviewed harmony plan")
    source = hp.pretty_midi.PrettyMIDI(args.melody_midi)
    if source.resolution != plan["ppq"] or plan.get("origin_tick", 0) or int(source.time_to_tick(source.get_end_time())) != plan["duration_ticks"]:
        raise ValueError("Harmony plan and melody timeline differ")
    if any(l["status"] == "chord" and len(l["pitch_classes"]) != 3 for l in plan["labels"]):
        raise ValueError("This curriculum supports triads only; do not silently discard extensions")
    model = GPT2LMHeadModel.from_pretrained(args.model).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
               "end_tick": int(source.time_to_tick(n.end)), "role": "melody"}
              for instrument in source.instruments if not instrument.is_drum for n in instrument.notes]
    rows, notes = rollout(model, plan["labels"], melody, source.resolution)
    out.mkdir(parents=True)
    for style, accompaniment in notes.items():
        rendered = lv.write_midi(source, melody, accompaniment, out / (style + ".mid"))
        write_preview_audio(rendered, out / (style + ".wav"))
    result = {"protocol": lv.VERSION, "model": args.model, "melody_sha256": digest,
              "harmony_plan_sha256": hashlib.sha256(Path(args.harmony_plan).read_bytes()).hexdigest(),
              "summary": summarize(rows), "rows": rows, "pitch_correction": False,
              "timing_and_dynamics": "provided by plan and deterministic simple renderer",
              "unknown_policy": "silent accompaniment; not a prediction of the unknown harmony"}
    write_json(out / "result.json", result)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
