"""Generate from an explicit harmony plan using a caller-selected density profile."""

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel
from build_harmony_review import write_json, write_preview_audio
import harmony_plan as hp
import local_voicing as lv
import rhythm_v4 as v4
import rhythm_v5 as v5
from train_local_voicing import predict as predict_voicing
from verify_rhythm_v4 import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model_output/local_rhythm_v5/final_model")
    parser.add_argument("--voicing-model", default="model_output/local_voicing_v3/final_model")
    parser.add_argument("--density", choices=v5.PROFILES, default="medium")
    parser.add_argument("--melody-midi", default="test/test1.mid")
    parser.add_argument("--harmony-plan", default="data/harmony_v2_review/plans/test1_pickup_4563.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError("Choose a fresh output directory")
    plan = json.loads(Path(args.harmony_plan).read_text(encoding="utf-8"))
    hp.validate_plan(plan)
    if plan.get("melody_sha256") and plan["melody_sha256"] != digest(args.melody_midi):
        raise ValueError("Melody differs from the reviewed plan")
    source = hp.pretty_midi.PrettyMIDI(args.melody_midi)
    if source.resolution != plan["ppq"] or plan.get("origin_tick", 0) or int(source.time_to_tick(source.get_end_time())) != plan["duration_ticks"]:
        raise ValueError("Melody and harmony timeline differ")
    for label in plan["labels"]:
        if label["status"] == "chord":
            v4.degrees(label)
            v4.steps_for(label, source.resolution)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    voicing_model = GPT2LMHeadModel.from_pretrained(args.voicing_model).to(device).eval()
    melody = [{"role": "melody", "pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
               "end_tick": int(source.time_to_tick(n.end))} for i in source.instruments if not i.is_drum for n in i.notes]
    previous, previous_voicing, previous_end = [], lv.INITIAL, None
    notes, rows = [], []
    for label in plan["labels"]:
        if label["status"] != "chord":
            previous_end = None
            continue
        if previous_end != label["start_tick"]:
            previous, previous_voicing = [], lv.INITIAL
        summary = lv.melody_summary(melody, label)
        voicing = predict_voicing(voicing_model, label, summary, previous_voicing)
        if not all(lv.assess(voicing, label, summary).values()):
            raise ValueError("Frozen voicing failed; no hidden pitch correction is applied")
        frames = v5.predict(model, label, melody, source.resolution, previous, args.density)
        events = v4.render(label, voicing, frames, source.resolution, summary["velocity"])
        notes.extend(events)
        rows.append({"start_tick": label["start_tick"], "end_tick": label["end_tick"], "symbol": label["symbol"],
                     "voicing": voicing, "frames": frames, "notes": events})
        previous, previous_voicing, previous_end = frames, voicing, label["end_tick"]
    out.mkdir(parents=True)
    rendered = lv.write_midi(source, melody, notes, out / "output.mid", end=plan["duration_ticks"])
    write_preview_audio(rendered, out / "output.wav")
    result = {"protocol": v5.VERSION, "density": args.density, "args": vars(args),
              "model_sha256": digest(Path(args.model) / "model.safetensors"),
              "voicing_model_sha256": digest(Path(args.voicing_model) / "model.safetensors"),
              "melody_sha256": digest(args.melody_midi), "plan_sha256": digest(args.harmony_plan),
              "rows": rows, "notes": notes, "density_is_soft_condition": True}
    write_json(out / "result.json", result)
    print(json.dumps({"segments": len(rows), "accompaniment_notes": len(notes), "density": args.density}))


if __name__ == "__main__":
    main()
