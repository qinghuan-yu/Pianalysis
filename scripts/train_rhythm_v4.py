"""Learn reviewed real accompaniment rhythms while freezing accepted harmony voicings."""

import argparse
from collections import Counter
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from build_harmony_review import write_json, write_preview_audio
import harmony_plan as hp
import local_voicing as lv
import rhythm_v4 as rv
from train_local_voicing import validate_dataset, predict as predict_voicing


def batch(samples, device):
    length = max(len(s["tokens"]) for s in samples)
    ids = torch.full((len(samples), length), rv.PAD, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, sample in enumerate(samples):
        count = len(sample["tokens"])
        ids[i, :count] = torch.tensor(sample["tokens"], device=device)
        labels[i, :count] = torch.tensor(sample["labels"], device=device)
        mask[i, :count] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def nll(model, samples):
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), 16):
            inputs = batch(samples[start:start + 16], model.device)
            valid = int((inputs["labels"][:, 1:] != -100).sum())
            total += float(model(**inputs).loss) * valid
            count += valid
    return total / count


def predict(model, label, melody, ppq, previous):
    if getattr(model.config, "pianalysis_protocol", None) != rv.VERSION:
        raise ValueError("Rhythm model protocol mismatch")
    tokens = rv.prompt(label, melody, ppq, previous)
    frames = rv.empty_frames(rv.steps_for(label, ppq))

    def choose(base, count):
        logits = model(input_ids=torch.tensor([tokens], device=model.device)).logits[0, -1]
        value = int(logits[base:base + count].argmax())
        tokens.append(base + value)
        return value

    with torch.inference_mode():
        for step, frame in enumerate(frames):
            tokens.append(rv.STEP + step)
            frame["mask"] = choose(rv.MASK, 8)
            for lane in range(3):
                if frame["mask"] & (1 << lane):
                    frame["durations"][lane] = choose(rv.DURATION, len(frames) - step) + 1
                    frame["velocities"][lane] = choose(rv.VELOCITY, 4)
    return frames


def evaluate(model, voicing_model, annotation, dataset, out):
    indexed = {r["source_id"]: r for r in dataset["records"]}
    reports, sections = [], []
    pooled = {split: {"learned": [], "fixed_arpeggio": []} for split in ("train", "validation", "test")}
    case_inputs = list(annotation["pieces"])
    plan = json.loads(Path("data/harmony_v2_review/plans/test1_pickup_4563.json").read_text(encoding="utf-8"))
    source = hp.pretty_midi.PrettyMIDI("test/test1.mid")
    if hashlib.sha256(Path("test/test1.mid").read_bytes()).hexdigest() != plan["melody_sha256"]:
        raise ValueError("Accepted external melody changed")
    case_inputs.append({"song": "test1", "split": "external", "source_file": "test/test1.mid", "ppq": source.resolution,
                        "origin_tick": 0, "end_tick": plan["duration_ticks"], "labels": plan["labels"],
                        "notes": [{"pitch": n.pitch, "velocity": n.velocity, "role": "melody", "start_tick": int(source.time_to_tick(n.start)),
                                   "end_tick": int(source.time_to_tick(n.end))} for i in source.instruments if not i.is_drum for n in i.notes]})
    for piece in case_inputs:
        melody = [n for n in piece["notes"] if n["role"] == "melody"]
        source = hp.pretty_midi.PrettyMIDI(piece["source_file"])
        previous_rhythm, previous_voicing, previous_end = [], lv.INITIAL, None
        generated, baseline, rows = [], [], []
        for label in piece["labels"]:
            if label["status"] != "chord":
                previous_rhythm, previous_end = [], None
                continue
            if label["start_tick"] != previous_end:
                previous_rhythm, previous_voicing = [], lv.INITIAL
            summary = lv.melody_summary(melody, label)
            voicing = predict_voicing(voicing_model, label, summary, previous_voicing)
            pitch_metrics = lv.assess(voicing, label, summary)
            if not all(v for k, v in pitch_metrics.items() if k != "chord_tone_ratio"):
                raise ValueError(f"Frozen voicing failed on {piece['song']} at {label['start_tick']}: {pitch_metrics}")
            frames = predict(model, label, melody, piece["ppq"], previous_rhythm)
            generated.extend(rv.render(label, voicing, frames, piece["ppq"], summary["velocity"]))
            baseline.extend(lv.schedule(label, voicing, piece["ppq"], "arpeggio", summary["velocity"]))
            row = {"start_tick": label["start_tick"], "end_tick": label["end_tick"], "symbol": label["symbol"],
                   "voicing": voicing, "frames": frames, "pitch_metrics": pitch_metrics}
            if piece["split"] != "external":
                reference = indexed[f"{piece['song']}:{label['source_label_index']}"]["frames"]
                row["learned_metrics"] = rv.onset_metrics(frames, reference)
                row["fixed_arpeggio_metrics"] = rv.onset_metrics(rv.baseline_frames(label, voicing, piece["ppq"]), reference)
                for mode in pooled[piece["split"]]:
                    pooled[piece["split"]][mode].append(row[mode + "_metrics"])
            rows.append(row)
            previous_rhythm, previous_voicing, previous_end = frames, voicing, label["end_tick"]
        stem = piece["song"]
        for style, notes in (("learned", generated), ("fixed", baseline)):
            rendered = lv.write_midi(source, melody, notes, out / f"{stem}_{style}.mid", piece["origin_tick"], piece["end_tick"])
            write_preview_audio(rendered, out / f"{stem}_{style}.wav")
        reports.append({"song": stem, "split": piece["split"], "origin_tick": piece["origin_tick"], "end_tick": piece["end_tick"],
                        "rows": rows, "generated_notes": generated, "baseline_notes": baseline})
        sections.append((stem == "test1", f"<section><h2>{html.escape(stem)} · {piece['split']}</h2><p>新模型：学习的节奏、时值与相对力度</p><audio controls preload='none' src='{stem}_learned.wav'></audio><p><a href='{stem}_learned.mid'>新MIDI</a></p><p>对照：相同配位的固定分解节奏</p><audio controls preload='none' src='{stem}_fixed.wav'></audio></section>"))
        print(f"Rendered {stem}: {len(rows)} spans, {len(generated)} notes", flush=True)
    metrics = {split: {mode: rv.aggregate_metrics(values) for mode, values in groups.items()} for split, groups in pooled.items()}
    result = {"protocol": rv.VERSION, "metrics": metrics, "cases": reports,
              "learned": ["upper-note onset mask", "quantized durations", "relative velocity bucket"],
              "provided": ["V3 frozen voicing pitches", "verified harmony boundaries", "quarter subdivision grid", "sustained bass anchor", "melody-relative volume scale", "same-lane retrigger release"],
              "scope": "Projected chord-tone rhythm experiment; not full arrangement reconstruction or Animenz style learning.",
              "human_audition_status": "pending"}
    write_json(out / "evaluation.json", result)
    table = "".join(f"<tr><td>{split}</td><td>{values['fixed_arpeggio']['onset_f1']:.3f}</td><td>{values['learned']['onset_f1']:.3f}</td></tr>" for split, values in metrics.items())
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>真实节奏学习对照</title><style>body{max-width:920px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}section{border-top:1px solid #bbb;padding:20px 0}audio{width:100%}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #ccc;padding:8px}</style><h1>保持和声，开始学习真实节奏</h1><p>新模型预测上方三声部的起音、时值和相对力度。音高来自已验收V3模型，低音锚点、拍网格和音量尺度仍由规则提供。训练目标来自原PIANO轨，经去八度重复、去装饰音和网格化处理，不是原曲完整复刻。新版本待试听，旧验收结果保留。</p><table><tr><th>划分</th><th>固定分解起音F1</th><th>学习节奏起音F1</th></tr>" + table + "</table>" + "".join(section for _, section in sorted(sections, key=lambda x: not x[0])) + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/trusted_harmony_v4/annotations.json")
    parser.add_argument("--output-dir", default="model_output/local_rhythm_v4")
    parser.add_argument("--evaluation-dir", default="data/trusted_harmony_v4/evaluation")
    parser.add_argument("--voicing-model", default="model_output/local_voicing_v3/final_model")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    out, evaluation = Path(args.output_dir), Path(args.evaluation_dir)
    if out.exists() or evaluation.exists():
        raise ValueError("Choose fresh experiment directories")
    annotation = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    validate_dataset(annotation)
    dataset = rv.build_dataset(annotation)
    train = [r["sample"] for r in dataset["records"] if r["split"] == "train"]
    validation = [r["sample"] for r in dataset["records"] if r["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Empty reviewed rhythm split")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(vocab_size=rv.VOCAB, n_positions=256, n_ctx=256, n_layer=2, n_head=4, n_embd=96,
                        resid_pdrop=.2, embd_pdrop=.2, attn_pdrop=.2, pad_token_id=rv.PAD, eos_token_id=rv.EOS,
                        bos_token_id=rv.BOS, pianalysis_protocol=rv.VERSION)
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.02)
    max_steps = math.ceil(len(train) / 16) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step + 1) / 20) * (.2 + .8 * (1 + math.cos(math.pi * min(step, max_steps) / max_steps)) / 2))
    out.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    write_json(evaluation.parent / "rhythm_dataset.json", dataset)
    metadata = {"args": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
                "source_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                "frozen_voicing_sha256": hashlib.sha256((Path(args.voicing_model) / "model.safetensors").read_bytes()).hexdigest(),
                "splits": dict(Counter(r["split"] for r in dataset["records"])),
                "target_source": dataset["target_source"], "augmentation": "none", "test1_in_training": False}
    write_json(out / "training_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    best, bad, best_epoch, history = float("inf"), 0, 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        running, count = 0., 0
        for start in range(0, len(train), 16):
            inputs = batch(train[start:start + 16], device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            valid = int((inputs["labels"][:, 1:] != -100).sum())
            running += float(loss.detach()) * valid
            count += valid
        if epoch % 10 == 0 or epoch == args.epochs:
            score = nll(model, validation)
            row = {"epoch": epoch, "train_nll": running / count, "validation_nll": score}
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best - 1e-4:
                best, best_epoch, bad = score, epoch, 0
                model.save_pretrained(out / "final_model")
            else:
                bad += 1
            if bad >= 8:
                break
    model = GPT2LMHeadModel.from_pretrained(out / "final_model").to(device).eval()
    shuffled = copy.deepcopy(validation)
    for sample in shuffled:
        start, end = sample["tokens"].index(rv.MELODY) + 1, sample["prefix_length"] - 1
        pairs = [sample["tokens"][i:i + 2] for i in range(start, end, 2)]
        sample["tokens"][start:end] = [token for pair in reversed(pairs) for token in pair]
    training = {"history": history, "best_epoch": best_epoch, "best_validation_nll": best,
                "seconds": time.monotonic() - started, "melody_timing_control": {
                    "correct_nll": nll(model, validation), "reversed_melody_frame_nll": nll(model, shuffled)}}
    write_json(out / "training_results.json", training)
    voicing = GPT2LMHeadModel.from_pretrained(args.voicing_model).to(device).eval()
    metrics = evaluate(model, voicing, annotation, dataset, evaluation)
    write_json(out / "evaluation_summary.json", metrics)
    print(json.dumps({"training": training, "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
