#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional accompaniment generation.

Despite the historical filename, this script no longer generates from scratch.
It uses a melody source prompt and asks the trained model to generate the
accompaniment target after SEP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2LMHeadModel


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, tokens_to_notes, write_notes_midi  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate accompaniment from a melody prompt.")
    parser.add_argument("--model-path", default="model_output/accompaniment_gpt2/final_model")
    parser.add_argument("--dataset", default="data/training_windows_v1/dataset_windows_v1.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-json", default="", help="Optional JSON file containing source_tokens.")
    parser.add_argument("--out-json", default="generated_accompaniment.json")
    parser.add_argument("--out-midi", default="generated_accompaniment.mid")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    if args.source_json:
        path = Path(args.source_json)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        source_tokens = data.get("source_tokens")
        if not isinstance(source_tokens, list):
            raise ValueError(f"{path} must contain source_tokens")
        return [int(x) for x in source_tokens], data.get("metadata", {}), {"source": str(path)}

    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"No samples in {dataset_path}")
    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"--sample-index must be between 0 and {len(samples) - 1}")

    sample = samples[args.sample_index]
    source_tokens = sample.get("source_tokens")
    if not isinstance(source_tokens, list):
        seq = sample["training_sequence"]
        sep_index = seq.index(data["metadata"]["token_ids"]["SEP"])
        source_tokens = seq[1:sep_index]
    return [int(x) for x in source_tokens], data.get("metadata", {}), sample


def first_eos_position(tokens: list[int], eos_token_id: int) -> int | None:
    for idx, token in enumerate(tokens):
        if token == eos_token_id:
            return idx
    return None


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    source_tokens, metadata, source_info = load_source(args)
    token_ids = dict(TOKEN)
    token_ids.update(metadata.get("token_ids", {}))
    token_ids = {key: int(value) for key, value in token_ids.items()}
    quantum_ms = int(metadata.get("quantum_ms", 10))

    prompt = [token_ids["BOS"]] + source_tokens + [token_ids["SEP"]]
    input_ids = torch.tensor([prompt], dtype=torch.long)

    model_path = Path(args.model_path)
    print(f"Loading model: {model_path}")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=token_ids["PAD"],
            eos_token_id=token_ids["EOS"],
        )

    generated = output[0].detach().cpu().tolist()
    target_tokens = generated[len(prompt) :]
    eos_at = first_eos_position(target_tokens, token_ids["EOS"])
    if eos_at is not None:
        target_tokens = target_tokens[:eos_at]

    melody_notes = tokens_to_notes(source_tokens, quantum_ms)
    accompaniment_notes = tokens_to_notes(target_tokens, quantum_ms)
    for note in melody_notes:
        note["role"] = "melody"
    for note in accompaniment_notes:
        note["role"] = "accompaniment"

    write_notes_midi(melody_notes + accompaniment_notes, Path(args.out_midi))

    payload = {
        "model_path": str(model_path),
        "source_info": source_info,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "prompt": prompt,
        "source_tokens": source_tokens,
        "generated_target_tokens": target_tokens,
        "melody_note_count": len(melody_notes),
        "accompaniment_note_count": len(accompaniment_notes),
        "out_midi": args.out_midi,
    }
    with Path(args.out_json).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Generated target tokens: {len(target_tokens)}")
    print(f"Melody notes: {len(melody_notes)}; accompaniment notes: {len(accompaniment_notes)}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_midi}")


if __name__ == "__main__":
    main()
