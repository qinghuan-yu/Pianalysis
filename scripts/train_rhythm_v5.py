"""Train the preregistered expanded-data and density-conditioned comparisons."""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from build_harmony_review import write_json
import rhythm_v4 as v4
import rhythm_v5 as v5
from train_local_voicing import validate_dataset
from train_rhythm_v4 import batch, nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("v4", "v5"), required=True)
    parser.add_argument("--annotations", default="data/trusted_harmony_v5/annotations.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    plan_path = Path("examples/harmony/rhythm_v5_experiment.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    module = v4 if args.protocol == "v4" else v5
    out = Path(args.output_dir)
    if out.exists():
        raise ValueError("Choose a fresh training directory")
    annotation = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    validate_dataset(annotation)
    dataset = module.build_dataset(annotation)
    train = [r["sample"] for r in dataset["records"] if r["split"] == "train"]
    validation = [r["sample"] for r in dataset["records"] if r["split"] == "validation"]
    if not train or not validation:
        raise ValueError("Empty training/validation split")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(4)
    rng = random.Random(plan["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=module.VOCAB, n_positions=256, n_ctx=256, n_layer=2, n_head=4,
        n_embd=96, resid_pdrop=.2, embd_pdrop=.2, attn_pdrop=.2, pad_token_id=v4.PAD, eos_token_id=v4.EOS,
        bos_token_id=v4.BOS, pianalysis_protocol=module.VERSION)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.02)
    epochs = plan["max_epochs"]
    max_steps = math.ceil(len(train) / 16) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, (step + 1) / 20) *
        (.2 + .8 * (1 + math.cos(math.pi * min(step, max_steps) / max_steps)) / 2))
    out.mkdir(parents=True)
    write_json(out / "rhythm_dataset.json", dataset)
    metadata = {"protocol": module.VERSION, "args": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
        "seed": plan["seed"], "annotations_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
        "experiment_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "frozen_voicing_sha256": hashlib.sha256((Path(plan["frozen_voicing"]) / "model.safetensors").read_bytes()).hexdigest(),
        "splits": dict(Counter(r["split"] for r in dataset["records"])), "test_used_for_selection": False,
        "max_sequence_length": max(len(r["sample"]["tokens"]) for r in dataset["records"]),
        "density_profiles_train": dict(Counter(v5.profile_for(r["frames"]) for r in dataset["records"] if r["split"] == "train"))}
    write_json(out / "training_metadata.json", metadata)
    print(json.dumps(metadata), flush=True)
    best, bad, history, started = float("inf"), 0, [], time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train)
        total, count = 0., 0
        for start in range(0, len(train), 16):
            inputs = batch(train[start:start + 16], device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            valid = int((inputs["labels"][:, 1:] != -100).sum())
            total += float(loss.detach()) * valid
            count += valid
        if epoch % 10 == 0 or epoch == epochs:
            score = nll(model, validation)
            row = {"epoch": epoch, "train_nll": total / count, "validation_nll": score}
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
    altered = copy.deepcopy(validation)
    for sample in altered:
        start, end = sample.get("melody_slice", (sample["tokens"].index(v4.MELODY) + 1, sample["prefix_length"] - 1))
        pairs = [sample["tokens"][i:i + 2] for i in range(start, end, 2)]
        sample["tokens"][start:end] = [t for pair in reversed(pairs) for t in pair]
    result = {"best_epoch": best_epoch, "best_validation_nll": best, "history": history,
              "seconds": time.monotonic() - started, "melody_timing_control": {
                  "correct_nll": nll(model, validation), "reversed_nll": nll(model, altered)},
              "validation_density": "reference_profile_for_teacher_forced_loss_only" if args.protocol == "v5" else "not_conditioned"}
    write_json(out / "training_results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
