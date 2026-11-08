"""Compare unconditioned TSD with timed harmony and deliberately shifted chords."""
from argparse import ArgumentParser, Namespace
import copy
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from transformers import GPT2LMHeadModel
import harmony_conditioning as hc
import harmony_plan as hp
from build_harmony_review import write_json, write_preview_audio
from closed_loop_v1 import load_midi_notes, tokens_to_notes, write_notes_midi
from evaluate_token_protocol_v1 import coverage
from generate_from_scratch import generate_target, generate_windowed
from token_protocol_v1 import TSD_VERSION
from train_v2 import ConditionalMusicDataset, load_dataset


def shifted(labels):
    result = copy.deepcopy(labels)
    for label in result:
        if label["status"] == "chord":
            label["root_pc"] = (label["root_pc"] + 1) % 12
            label["bass_pc"] = (label["bass_pc"] + 1) % 12
            label["pitch_classes"] = sorted((p + 1) % 12 for p in label["pitch_classes"])
            # 扰动只作用于诊断条件；原始 Harte 名称不再描述新音级。
            label.pop("symbol", None)
    return result


def chord_fit(notes, labels):
    matched, total = 0.0, 0.0
    for note in notes:
        for label in labels:
            if label["status"] != "chord":
                continue
            duration = max(0, min(note["end"], label["end"]) - max(note["start"], label["start"]))
            total += duration
            if note["pitch"] % 12 in set(label["pitch_classes"]) | {label["bass_pc"]}:
                matched += duration
    return matched / total if total else None


def target_nll(model, samples):
    dataset = ConditionalMusicDataset(samples, model.config.n_positions, 0)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(dataset), 8):
            batch = [dataset[i] for i in range(start, min(len(dataset), start + 8))]
            inputs = {key: torch.stack([row[key] for row in batch]).to(model.device) for key in batch[0]}
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, :-1]
            targets = inputs["labels"][:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
            counts = (targets != -100).sum(dim=1)
            rows.extend({"loss_sum": float(a), "tokens": int(b)} for a, b in zip(loss.sum(dim=1).cpu(), counts.cpu()))
    return rows


def aggregate_runs(rows):
    aggregates = []
    for group in ("heldout_window", "external_melody"):
        valid_keys = []
        for condition in ("baseline", "timed_harmony", "shifted_harmony"):
            valid_keys.append({(r["case"], r["seed"]) for r in rows if r["group"] == group
                               and r["condition"] == condition and r["raw_chord_fit_to_reference"] is not None})
        paired = set.intersection(*valid_keys)
        for condition in ("baseline", "timed_harmony", "shifted_harmony"):
            selected = [r for r in rows if r["group"] == group and r["condition"] == condition]
            scores = [r["raw_chord_fit_to_reference"] for r in selected if (r["case"], r["seed"]) in paired]
            row = {"group": group, "condition": condition, "runs": len(selected),
                   "syntax_valid": sum(r["syntax_valid"] for r in selected),
                   "chord_fit_valid_runs": sum(r["raw_chord_fit_to_reference"] is not None for r in selected),
                   "paired_chord_fit_runs": len(scores),
                   "paired_chord_fit_to_reference": statistics.mean(scores) if scores else None}
            for key in ("raw_chord_fit_to_reference", "raw_chord_fit_to_shifted", "raw_coverage", "raw_note_count", "tail_gap_seconds"):
                values = [r[key] for r in selected if r[key] is not None]
                row[key] = statistics.mean(values) if values else None
            aggregates.append(row)
    return aggregates


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/training_windows_harmony_timed_v2/dataset_windows_v1.json")
    parser.add_argument("--model", default="model_output/accompaniment_gpt2_harmony_timed_v2/final_model")
    parser.add_argument("--baseline-model", default="model_output/accompaniment_gpt2_tsd_v2_e20/final_model")
    parser.add_argument("--melody-midi", default="test/test1.mid")
    parser.add_argument("--harmony-plan", default="data/harmony_v2_review/plans/test1_pickup_4563.json")
    parser.add_argument("--out-dir", default="data/harmony_conditioning_v2/evaluation")
    args = parser.parse_args()
    out = Path(args.out_dir)
    if (out / "summary.json").exists():
        raise ValueError("Choose a fresh evaluation directory")
    out.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    metadata, samples, _ = load_dataset(Path(args.dataset), 1024)
    trained = read(Path(args.model).parent / "training_metadata.json")
    baseline_meta = read(Path(args.baseline_model).parent / "training_metadata.json")
    if trained["split"] != baseline_meta["split"] or metadata["harmony_tokenizer_version"] != hc.VERSION:
        raise ValueError("Unpaired data split or harmony protocol")
    eval_ids = trained["split"]["eval_pieces"]
    heldout = [sample for sample in samples if sample["source_piece_id"] in eval_ids]
    selected = []
    for piece_id in eval_ids:
        group = [sample for sample in heldout if sample["source_piece_id"] == piece_id]
        selected.extend([group[0], group[len(group) // 2]])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    hc.validate_model_conditioning(model, hc.VERSION)
    wrong = copy.deepcopy(heldout)
    for sample in wrong:
        hc.condition_sample(sample, shifted(sample["harmony_labels"]))
    correct_nll, shifted_nll = target_nll(model, heldout), target_nll(model, wrong)
    nll = {"windows": len(heldout),
           "correct": sum(x["loss_sum"] for x in correct_nll) / sum(x["tokens"] for x in correct_nll),
           "shifted": sum(x["loss_sum"] for x in shifted_nll) / sum(x["tokens"] for x in shifted_nll),
           "correct_lower_loss_windows": sum(a["loss_sum"] < b["loss_sum"] for a, b in zip(correct_nll, shifted_nll))}
    write_json(out / "target_nll.json", {**nll, "paired_windows": [
        {"piece_id": s["piece_id"], "correct": a, "shifted": b} for s, a, b in zip(heldout, correct_nll, shifted_nll)]})
    print(f"Held-out target NLL: {nll}", flush=True)
    baseline = GPT2LMHeadModel.from_pretrained(args.baseline_model).to(device).eval()
    plan = read(args.harmony_plan)
    if hashlib.sha256(Path(args.melody_midi).read_bytes()).hexdigest() != plan["melody_sha256"]:
        raise ValueError("Audition MIDI differs from the accepted plan")
    midi = hp.pretty_midi.PrettyMIDI(args.melody_midi)
    full_labels = hc.plan_labels_seconds(plan, hp.midi_timeline(midi))
    piece = load_midi_notes(Path(args.melody_midi))
    params = Namespace(window_seconds=8.0, max_new_tokens=384, temperature=0.7, top_k=30, top_p=0.9,
                       no_grammar_constraints=False)
    # 手算校准：C 与 D 两段各两秒，持续 C 音只匹配其中一半。
    check_labels = [{**hp.parse_symbol(s), "start": a, "end": b} for s, a, b in [("C:maj", 0, 2), ("D:maj", 2, 4)]]
    assert chord_fit([{"pitch": 60, "start": 0, "end": 4}], check_labels) == 0.5
    rows, audio_cases = [], []
    for condition in ("baseline", "timed_harmony", "shifted_harmony"):
        active_model = baseline if condition == "baseline" else model
        for sample in [*selected, None]:
            for seed in (7, 42):
                torch.manual_seed(seed)
                if sample is None:
                    source = copy.deepcopy(piece)
                    if condition != "baseline":
                        source["timed_harmony"] = shifted(full_labels) if condition == "shifted_harmony" else full_labels
                    melody, raw, windows = generate_windowed(active_model, source, params, TSD_VERSION)
                    reports = [w["generation_report"] for w in windows]
                    tokens = [t for w in windows for t in w["generated_target_tokens"]]
                    labels, name, group, duration = full_labels, "test1", "external_melody", midi.get_end_time()
                else:
                    labels, name, group = sample["harmony_labels"], sample["piece_id"], "heldout_window"
                    harmony = [] if condition == "baseline" else hc.encode_labels(shifted(labels) if condition == "shifted_harmony" else labels)
                    prompt = [1, *sample["source_tokens"]] + ([30, *harmony] if harmony else []) + [2]
                    duration = sample["window_duration"]
                    tokens, report = generate_target(active_model, prompt, params, TSD_VERSION, target_duration=duration)
                    reports, windows = [report], None
                    raw = tokens_to_notes(tokens, 10, TSD_VERSION)
                    melody = tokens_to_notes(sample["source_tokens"], 10, TSD_VERSION)
                invalid = any(r.get(k, 0) for r in reports for k in ("unknown_tokens", "invalid_parameter_events", "incomplete_events", "wrong_role_note_events"))
                row = {"condition": condition, "case": name, "group": group, "seed": seed,
                       "raw_chord_fit_to_reference": chord_fit(raw, labels),
                       "raw_chord_fit_to_shifted": chord_fit(raw, shifted(labels)),
                       "raw_coverage": coverage(raw, duration), "raw_note_count": len(raw),
                       "tail_gap_seconds": max(0, duration - max((n["end"] for n in raw), default=0)),
                       "syntax_valid": not invalid, "generation_reports": reports}
                rows.append(row)
                stem = f"{condition}_{name}_seed{seed}"
                write_json(out / (stem + ".json"), {**row, "windows": windows, "generated_target_tokens": tokens,
                                                     "melody_notes": melody, "raw_accompaniment_notes": raw})
                if sample is None:
                    path = out / (stem + ".mid")
                    write_notes_midi(melody + raw, path)
                    if seed == 7:
                        rendered = hp.pretty_midi.PrettyMIDI(str(path))
                        for instrument in rendered.instruments:
                            if "melody" in instrument.name.lower():
                                instrument.name = "MELODY"
                        write_preview_audio(rendered, out / (stem + ".wav"), end=duration)
                        audio_cases.append((condition, stem))
                print(f"{condition} {name} seed={seed} fit={row['raw_chord_fit_to_reference']} coverage={row['raw_coverage']:.3f}", flush=True)
    aggregates = aggregate_runs(rows)
    write_json(out / "summary.json", {"settings": vars(args), "generation_settings": vars(params), "seeds": [7, 42],
                                      "target_nll": nll, "aggregates": aggregates, "runs": rows,
                                      "limitations": ["Corpus harmony is unreviewed reference-derived weak supervision.",
                                                      "One training seed; 4 held-out pieces; 1 externally accepted harmony plan.",
                                                      "Chord fit is duration-weighted membership, not musical quality; nonchord tones can be intentional.",
                                                      "Raw model outputs: no pitch correction, fallback bass or sanitation.",
                                                      "Shifted condition raises chord pitch classes and bass by a semitone; melody is unchanged."]})
    titles = {"baseline": "旧模型：仅旋律条件", "timed_harmony": "新模型：已确认的旋律＋和声条件", "shifted_harmony": "诊断对照：和弦整体升半音，旋律不变"}
    page = "<!doctype html><meta charset='utf-8'><title>和声条件织体对照</title><style>body{max-width:900px;margin:40px auto;padding:20px;font:16px/1.7 system-ui}audio{width:100%}section{border-top:1px solid #ccc;padding:20px 0}</style><h1>和声条件织体对照</h1><p>先比较柱式基线与新模型。以下为 seed 7 的原始生成，没有自动纠正音高或补低音；模型织体尚未通过试听。未知弱起段的生成也需要核对。</p>"
    page += "<section><h2>已通过试听：柱式和弦基线</h2><audio controls src='../../harmony_v2_review/review/test1_pickup_4563.wav'></audio></section>"
    for condition, stem in audio_cases:
        page += f"<section><h2>{html.escape(titles[condition])}</h2><audio controls src='{stem}.wav'></audio><p><a href='{stem}.mid'>MIDI</a> · <a href='{stem}.json'>生成记录</a></p></section>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    main()
