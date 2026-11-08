#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional symbolic accompaniment training.

Expected dataset format:
    [BOS] source_melody [SEP] target_accompaniment [EOS]

Only target_accompaniment contributes to loss. Source melody, BOS/SEP, and PAD
positions are masked with -100.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from token_protocol_v1 import LEGACY_VERSION, canonical_version, parse_events
import harmony_conditioning as hc


TOKEN_DEFAULTS = {
    "PAD": 0,
    "BOS": 1,
    "SEP": 2,
    "EOS": 3,
}


class ConditionalMusicDataset(Dataset):
    """Dataset with source and padding masked out of labels."""

    def __init__(self, samples: list[dict[str, Any]], max_length: int, pad_token_id: int):
        self.samples = samples
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        token_ids = list(sample["training_sequence"])
        target_start = int(sample["target_start_index"])

        if len(token_ids) > self.max_length:
            raise ValueError(
                f"Sample {sample.get('piece_id', idx)} length {len(token_ids)} exceeds max_length={self.max_length}. "
                "Build shorter windows before training."
            )

        labels = token_ids.copy()
        for i in range(min(target_start, len(labels))):
            labels[i] = -100

        attention_mask = [1] * len(token_ids)
        pad_len = self.max_length - len(token_ids)
        if pad_len > 0:
            token_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a melody-conditioned accompaniment GPT-2 model.")
    parser.add_argument("--data-file", default="data/training_windows_tsd_v2/dataset_windows_v1.json")
    parser.add_argument("--output-dir", default="model_output/accompaniment_gpt2_tsd_v2")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path, max_length: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metadata = data.get("metadata", {})
    tokenizer_version = canonical_version(metadata.get("tokenizer_version"))
    harmony_version = metadata.get("harmony_tokenizer_version")
    if harmony_version not in (None, hc.VERSION):
        raise ValueError("Unsupported harmony protocol")
    if harmony_version == hc.VERSION and tokenizer_version != "pianalysis_tsd_v2":
        raise ValueError("Timed harmony requires the TSD V2 note protocol")
    raw_samples = data.get("samples", [])
    if not raw_samples:
        raise ValueError(f"No samples found in {path}")

    valid: list[dict[str, Any]] = []
    skipped = {
        "missing_training_sequence": 0,
        "missing_target_start_index": 0,
        "empty_target": 0,
        "over_max_length": 0,
    }

    max_token_id = 0
    max_sample_length = 0
    for sample in raw_samples:
        if canonical_version(sample.get("tokenizer_version", tokenizer_version)) != tokenizer_version:
            raise ValueError(f"Sample {sample.get('piece_id')} tokenizer_version differs from dataset metadata")
        if sample.get("harmony_tokenizer_version") != harmony_version:
            raise ValueError("Sample harmony protocol differs from dataset metadata")
        seq = sample.get("training_sequence")
        target_start = sample.get("target_start_index")
        if not isinstance(seq, list) or not seq:
            skipped["missing_training_sequence"] += 1
            continue
        if target_start is None:
            skipped["missing_target_start_index"] += 1
            continue
        target_start = int(target_start)
        if target_start >= len(seq):
            skipped["empty_target"] += 1
            continue
        if len(seq) > max_length:
            skipped["over_max_length"] += 1
            continue

        if tokenizer_version != LEGACY_VERSION:
            if harmony_version == hc.VERSION:
                harmony_tokens = sample.get("harmony_tokens")
                if not harmony_tokens:
                    raise ValueError("Missing timed harmony tokens")
                hc.decode_labels(harmony_tokens, int(metadata.get("quantum_ms", 10)))
                expected = [1, *sample["source_tokens"], 30, *harmony_tokens, 2, *sample["target_tokens"], 3]
                if seq != expected or target_start != len(sample["source_tokens"]) + len(harmony_tokens) + 3:
                    raise ValueError("Invalid timed harmony training_sequence")
            for field in ("source_tokens", "target_tokens"):
                tokens = sample.get(field)
                if not isinstance(tokens, list):
                    raise ValueError(f"Invalid typed sample {sample.get('piece_id')}: missing {field}")
                _, report = parse_events(tokens, tokenizer_version)
                if any(report[key] for key in ("unknown_tokens", "invalid_parameter_events", "incomplete_events")) or report["eos_index"] is not None:
                    raise ValueError(f"Invalid typed {field} in {sample.get('piece_id')}: {report}")
            if (target_start < 2 or seq[0] != TOKEN_DEFAULTS["BOS"]
                    or seq[target_start - 1] != TOKEN_DEFAULTS["SEP"] or seq[-1] != TOKEN_DEFAULTS["EOS"]
                    or seq[target_start:-1] != sample["target_tokens"]
                    or seq[1:1 + len(sample["source_tokens"])] != sample["source_tokens"]):
                raise ValueError(f"Invalid typed training_sequence in {sample.get('piece_id')}")

        max_token_id = max(max_token_id, max(int(x) for x in seq))
        max_sample_length = max(max_sample_length, len(seq))
        valid.append(sample)

    if not valid:
        raise ValueError(f"No valid samples remain after validation. Skipped: {skipped}")

    stats = {
        "loaded_samples": len(raw_samples),
        "valid_samples": len(valid),
        "skipped": skipped,
        "max_token_id": max_token_id,
        "vocab_size": max_token_id + 1,
        "max_sample_length": max_sample_length,
    }
    return metadata, valid, stats


def split_by_piece(
    samples: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    pieces = sorted({sample.get("source_piece_id") or sample.get("piece_id", "") for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(pieces)

    eval_count = max(1, int(round(len(pieces) * eval_ratio))) if len(pieces) > 1 else 0
    eval_pieces = set(pieces[:eval_count])
    train_pieces = set(pieces[eval_count:])

    train_samples = [
        sample for sample in samples if (sample.get("source_piece_id") or sample.get("piece_id", "")) in train_pieces
    ]
    eval_samples = [
        sample for sample in samples if (sample.get("source_piece_id") or sample.get("piece_id", "")) in eval_pieces
    ]

    if not train_samples or not eval_samples:
        raise ValueError("Train/eval split produced an empty split. Adjust --eval-ratio or dataset size.")

    return train_samples, eval_samples, sorted(train_pieces), sorted(eval_pieces)


def create_model(
    args: argparse.Namespace, vocab_size: int, token_ids: dict[str, int],
    tokenizer_version: str = LEGACY_VERSION, quantum_ms: int = 10, window_seconds: float = 8.0,
) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.max_length,
        n_ctx=args.max_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=token_ids["BOS"],
        eos_token_id=token_ids["EOS"],
        pad_token_id=token_ids["PAD"],
        pianalysis_tokenizer_version=canonical_version(tokenizer_version),
        pianalysis_quantum_ms=quantum_ms,
        pianalysis_window_seconds=window_seconds,
    )
    model = GPT2LMHeadModel(config)
    params_m = sum(param.numel() for param in model.parameters()) / 1e6
    print(
        "Model: "
        f"vocab={vocab_size}, max_length={args.max_length}, layers={args.n_layer}, "
        f"heads={args.n_head}, embd={args.n_embd}, params={params_m:.2f}M"
    )
    return model


def training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_dir": str(Path(args.output_dir) / "logs"),
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": torch.cuda.is_available() and not args.no_fp16,
        "dataloader_num_workers": 0,
        "report_to": [] if args.report_to.lower() == "none" else [args.report_to],
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def save_training_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_metadata: dict[str, Any],
    dataset_stats: dict[str, Any],
    train_pieces: list[str],
    eval_pieces: list[str],
    token_ids: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_args": vars(args),
        "dataset_metadata": dataset_metadata,
        "dataset_stats": dataset_stats,
        "split": {
            "train_piece_count": len(train_pieces),
            "eval_piece_count": len(eval_pieces),
            "train_pieces": train_pieces,
            "eval_pieces": eval_pieces,
        },
        "token_ids": token_ids,
        "label_masking": {
            "source_before_target_start_index": -100,
            "padding": -100,
        },
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite_output_dir:
        shutil.rmtree(output_dir)

    dataset_metadata, samples, dataset_stats = load_dataset(data_file, args.max_length)
    token_ids = dict(TOKEN_DEFAULTS)
    token_ids.update(dataset_metadata.get("token_ids", {}))
    token_ids = {key: int(value) for key, value in token_ids.items()}

    vocab_size = max(dataset_stats["vocab_size"], max(token_ids.values()) + 1)
    train_samples, eval_samples, train_pieces, eval_pieces = split_by_piece(samples, args.eval_ratio, args.seed)

    print(f"Dataset: {data_file}")
    print(f"Valid samples: {dataset_stats['valid_samples']} / {dataset_stats['loaded_samples']}")
    print(f"Skipped: {dataset_stats['skipped']}")
    print(f"Train samples: {len(train_samples)} from {len(train_pieces)} pieces")
    print(f"Eval samples: {len(eval_samples)} from {len(eval_pieces)} pieces")
    print(f"Vocab size: {vocab_size}; max sample length: {dataset_stats['max_sample_length']}")

    train_dataset = ConditionalMusicDataset(train_samples, args.max_length, token_ids["PAD"])
    eval_dataset = ConditionalMusicDataset(eval_samples, args.max_length, token_ids["PAD"])
    model = create_model(
        args, vocab_size, token_ids,
        dataset_metadata.get("tokenizer_version", LEGACY_VERSION),
        int(dataset_metadata.get("quantum_ms", 10)),
        float(dataset_metadata.get("window_seconds", 8.0)),
    )
    model.config.pianalysis_harmony_version = dataset_metadata.get("harmony_tokenizer_version")

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_arguments(args),
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    save_training_metadata(
        output_dir,
        args,
        dataset_metadata,
        dataset_stats,
        train_pieces,
        eval_pieces,
        token_ids,
    )

    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final_model"))
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_samples)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(f"Training complete. Final model: {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
