"""Train a small local voicing model on explicitly synthetic, reviewed harmony exercises."""

import argparse
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
from build_trusted_harmony import check_hashes
import harmony_plan as hp
import local_voicing as lv


def validate_dataset(data):
    if data.get("review_status") != "assistant_symbolic_reviewed" or not data.get("training_gate") or not all(data["training_gate"].values()):
        raise ValueError("The symbolic review gate has not passed")
    songs = set()
    for piece in data["pieces"]:
        if piece["song"] in songs or piece["split"] not in ("train", "validation", "test"):
            raise ValueError("Song leakage or invalid split")
        songs.add(piece["song"])
        check_hashes(Path(piece["source_file"]).parent, piece["source_hashes"])
        previous = -1
        for label in piece["labels"]:
            if label.get("review_status") != "assistant_symbolic_reviewed" or label["status"] != "chord" or not all(label["evidence"]["checks"].values()):
                raise ValueError("Unreviewed or unknown label reached training")
            if label["start_tick"] < previous or label["end_tick"] <= label["start_tick"]:
                raise ValueError("Overlapping or invalid reviewed labels")
            previous = label["end_tick"]


def exercise(label, melody, previous, identity):
    target = lv.teacher_voicing(label, melody, previous)
    prefix = lv.prompt(label, melody, previous)
    seq = [*prefix, *[lv.PITCH + p for p in target], lv.EOS]
    return {"sequence": seq, "target_start": len(prefix), "source_id": identity,
            "label": label, "melody": melody, "previous": previous, "target": target,
            "target_source": "synthetic_rule_teacher"}


def make_exercises(data, split):
    samples = []
    contexts = [lv.INITIAL, [40, 52, 55, 59], [43, 55, 59, 62], [33, 45, 48, 52]]
    for piece in data["pieces"]:
        if piece["split"] != split:
            continue
        for label in piece["labels"]:
            for shift in range(-6, 6):
                for octave in (-12, 0, 12):
                    for inversion in range(3):
                        changed, melody = lv.transpose(label, label["melody_summary"], shift, octave, inversion)
                        if not 48 <= melody["min"] <= melody["max"] <= 108:
                            continue
                        for context in contexts:
                            previous = [p + shift for p in context]
                            try:
                                sample = exercise(changed, melody, previous, f"{piece['song']}:{label['source_label_index']}")
                            except ValueError:
                                continue
                            sample["augmentation"] = {"transpose": shift, "melody_octave": octave,
                                                      "synthetic_inversion": inversion, "synthetic_context": previous}
                            samples.append(sample)
    return samples


def batch(samples, device):
    length = max(len(s["sequence"]) for s in samples)
    ids = torch.full((len(samples), length), lv.PAD, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, sample in enumerate(samples):
        count = len(sample["sequence"])
        ids[i, :count] = torch.tensor(sample["sequence"], device=device)
        labels[i, sample["target_start"]:count] = ids[i, sample["target_start"]:count]
        mask[i, :count] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def nll(model, samples, size=128):
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), size):
            inputs = batch(samples[start:start + size], model.device)
            valid = int((inputs["labels"][:, 1:] != -100).sum())
            total += float(model(**inputs).loss) * valid
            count += valid
    return total / count


def predict(model, label, melody, previous):
    if getattr(model.config, "pianalysis_protocol", None) != lv.VERSION:
        raise ValueError("Model protocol mismatch")
    tokens = lv.prompt(label, melody, previous)
    result = []
    with torch.inference_mode():
        for _ in range(4):
            logits = model(input_ids=torch.tensor([tokens], device=model.device)).logits[0, -1]
            # 仅约束输出字段是MIDI音高；不按和弦筛音，不事后改音或补低音。
            pitch = int(logits[lv.PITCH:lv.PITCH + 128].argmax())
            result.append(pitch)
            tokens.append(lv.PITCH + pitch)
    return result


def summarize(rows):
    checks = [r["metrics"] for r in rows]
    if not checks:
        return {"segments": 0, "bass_correct": None, "chord_complete": None,
                "all_requirements_passed_segments": 0}
    keys = checks[0].keys()
    return {"segments": len(rows), **{k: sum(c[k] for c in checks) / len(checks) for k in keys},
            "all_requirements_passed_segments": sum(all(v for k, v in c.items() if k != "chord_tone_ratio") for c in checks)}


def rollout(model, labels, melody, ppq):
    previous, rows, notes = lv.INITIAL, [], {"block": [], "arpeggio": []}
    for label in labels:
        if label["status"] != "chord":
            continue
        summary = lv.melody_summary(melody, label)
        voicing = predict(model, label, summary, previous)
        rows.append({"start_tick": label["start_tick"], "end_tick": label["end_tick"],
                     "symbol": label.get("symbol"), "root_pc": label["root_pc"], "bass_pc": label["bass_pc"],
                     "voicing": voicing, "previous": previous, "metrics": lv.assess(voicing, label, summary)})
        for style in notes:
            notes[style].extend(lv.schedule(label, voicing, ppq, style, summary["velocity"]))
        previous = voicing
    return rows, notes


def evaluate_outputs(model, data, out):
    cases, audio_sections, counter_correct, counter_wrong = [], [], [], []
    for piece in data["pieces"]:
        melody = [n for n in piece["notes"] if n["role"] == "melody"]
        source = hp.pretty_midi.PrettyMIDI(piece["source_file"])
        rows, notes = rollout(model, piece["labels"], melody, piece["ppq"])
        case = {"song": piece["song"], "split": piece["split"], "summary": summarize(rows), "rows": rows}
        cases.append(case)
        for style in notes:
            stem = f"pop909_{piece['song']}_{style}"
            rendered = lv.write_midi(source, melody, notes[style], out / (stem + ".mid"), piece["origin_tick"], piece["end_tick"])
            write_preview_audio(rendered, out / (stem + ".wav"))
        audio_sections.append((f"POP909 {piece['song']} · {piece['split']}", f"pop909_{piece['song']}"))
        if piece["split"] in ("validation", "test"):
            previous = lv.INITIAL
            changed = copy.deepcopy(piece["labels"])
            for target, replacement in zip(changed, piece["labels"][1:] + piece["labels"][:1]):
                for key in ("root_pc", "bass_pc", "pitch_classes"):
                    target[key] = copy.deepcopy(replacement[key])
                target.pop("symbol", None)
            for label, wrong in zip(piece["labels"], changed):
                summary = lv.melody_summary(melody, label)
                good_sample = exercise(label, summary, previous, piece["song"])
                bad_sample = copy.deepcopy(good_sample)
                wrong_prompt = lv.prompt(wrong, summary, previous)
                bad_sample["sequence"] = wrong_prompt + good_sample["sequence"][good_sample["target_start"]:]
                bad_sample["target_start"] = len(wrong_prompt)
                counter_correct.append(good_sample)
                counter_wrong.append(bad_sample)
                previous = good_sample["target"]
            altered_rows, _ = rollout(model, changed, melody, piece["ppq"])
            case["rotated_condition"] = {"summary_against_new_chords": summarize(altered_rows), "rows": altered_rows,
                "changed_voicings": sum(a["voicing"] != b["voicing"] for a, b in zip(rows, altered_rows))}

    # 外部test1始终留出，不使用它选择检查点或构造练习。
    plan_path = Path("data/harmony_v2_review/plans/test1_pickup_4563.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    melody_path = Path("test/test1.mid")
    if hashlib.sha256(melody_path.read_bytes()).hexdigest() != plan["melody_sha256"]:
        raise ValueError("Accepted test1 melody changed")
    source = hp.pretty_midi.PrettyMIDI(str(melody_path))
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
               "end_tick": int(source.time_to_tick(n.end)), "role": "melody"}
              for instrument in source.instruments if not instrument.is_drum for n in instrument.notes]
    rows, notes = rollout(model, plan["labels"], melody, source.resolution)
    cases.append({"song": "test1", "split": "external", "summary": summarize(rows), "rows": rows})
    for style in notes:
        rendered = lv.write_midi(source, melody, notes[style], out / ("test1_" + style + ".mid"))
        write_preview_audio(rendered, out / ("test1_" + style + ".wav"))
    audio_sections.insert(0, ("test1 · 花人局尾奏 · 外部检查", "test1"))
    temporal = {"segments": len(counter_correct), "correct_nll": nll(model, counter_correct),
                "rotated_nll": nll(model, counter_wrong),
                "scope": "同一合成目标，替换当前段和弦；与旧模型复杂目标的NLL数值不可直接比较。"}
    write_json(out / "evaluation.json", {"cases": cases, "temporal_condition_test": temporal,
        "generation": "greedy raw pitches, syntax-only pitch field restriction; no harmonic mask or pitch correction",
        "provided_by_rules": ["chord timing", "beat phase", "durations", "block/arpeggio pattern", "velocity balance", "unknown-span silence"],
        "human_audition_status": "pending"})
    sections = "".join(f"<section><h2>{html.escape(title)}</h2>" + "".join(
        f"<p>{name}</p><audio controls preload='none' src='{stem}_{style}.wav'></audio><p><a href='{stem}_{style}.mid'>MIDI</a></p>"
        for style, name in (("block", "模型配位 · 柱式"), ("arpeggio", "相同模型配位 · 简单分解"))) + "</section>" for title, stem in audio_sections)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>局部和声训练试听</title><style>body{max-width:900px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}section{border-top:1px solid #bbb;padding:20px 0}audio{width:100%}</style><h1>先检验简单和声</h1><p>模型预测低音与三音配位，音高未经和弦纠正。换和弦时机、分解节奏及力度由已核验计划和明确规则提供；这一步没有学习复杂织体。模型只训练规则合成练习，不代表已学会原曲风格。以下输出待用户试听。</p><p><a href='../review/index.html'>查看首批标注与原编配对照</a></p>" + sections + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    return {"cases": [{k: c[k] for k in ("song", "split", "summary")} for c in cases], "temporal_condition_test": temporal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/trusted_harmony_v3/annotations.json")
    parser.add_argument("--output-dir", default="model_output/local_voicing_v3")
    parser.add_argument("--evaluation-dir", default="data/trusted_harmony_v3/evaluation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out, evaluation = Path(args.output_dir), Path(args.evaluation_dir)
    if out.exists() or evaluation.exists():
        raise ValueError("Use fresh experiment directories")
    data_path = Path(args.annotations)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    validate_dataset(data)
    train, validation = make_exercises(data, "train"), make_exercises(data, "validation")
    if not train or not validation:
        raise ValueError("Empty curriculum split")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(vocab_size=lv.VOCAB, n_positions=64, n_ctx=64, n_embd=128, n_layer=2, n_head=4,
                        resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1, pad_token_id=lv.PAD,
                        bos_token_id=lv.BOS, eos_token_id=lv.EOS, pianalysis_protocol=lv.VERSION)
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=.01)
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step + 1) / 80) * (.15 + .85 * (1 + math.cos(math.pi * min(step, steps) / steps)) / 2))
    metadata = {"args": vars(args), "protocol": lv.VERSION, "annotations_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "parameters": sum(p.numel() for p in model.parameters()), "train_exercises": len(train), "validation_exercises": len(validation),
                "source_label_counts": {split: sum(len(p["labels"]) for p in data["pieces"] if p["split"] == split) for split in ("train", "validation", "test")},
                "splits": {split: [p["song"] for p in data["pieces"] if p["split"] == split] for split in ("train", "validation", "test")},
                "target_source": "synthetic_rule_teacher", "augmentation": "transpose -6..5, melody octave -12/0/12, synthetic triad inversions and previous contexts; split before augmentation",
                "test1_in_training": False, "device": device}
    out.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    write_json(out / "training_metadata.json", metadata)
    write_json(evaluation.parent / "curriculum_manifest.json", {**metadata, "train_source_ids": sorted({s['source_id'] for s in train}),
                                                              "validation_source_ids": sorted({s['source_id'] for s in validation})})
    best, best_epoch, bad, history = float("inf"), 0, 0, []
    started = time.monotonic()
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train)
        running = 0.0
        for start in range(0, len(train), args.batch_size):
            selected = train[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch(selected, device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(selected)
        if epoch % 5 == 0 or epoch == args.epochs:
            loss = nll(model, validation)
            record = {"epoch": epoch, "train_loss": running / len(train), "validation_loss": loss}
            history.append(record)
            print(json.dumps(record), flush=True)
            if loss < best - 1e-5:
                best, best_epoch, bad = loss, epoch, 0
                model.save_pretrained(out / "final_model")
            else:
                bad += 1
            if bad >= 5:
                break
    model = GPT2LMHeadModel.from_pretrained(out / "final_model").to(device).eval()
    training = {"history": history, "best_epoch": best_epoch, "best_validation_loss": best, "seconds": time.monotonic() - started}
    write_json(out / "training_results.json", training)
    result = evaluate_outputs(model, data, evaluation)
    write_json(out / "evaluation_summary.json", result)
    print(json.dumps({"training": training, "evaluation": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
