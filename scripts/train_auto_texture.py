"""Fine-tune official Embellish weights on canonical accompaniment-only targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from auto_texture import (DEFAULT_CONFIG, PROTOCOL, Vocabulary, collate, load_model, load_samples,
                          loss_sum, prepare_windows, read_config, seed_all, sha256, write_json)


RESUME_PROTOCOL_KEYS = (
    "max_tokens", "window_bars", "stride_bars", "max_onset_error_beats", "max_duration_error_beats",
    "batch_size", "accum_steps", "learning_rate", "weight_decay", "warmup_steps", "max_steps", "seed",
    "timing_policy", "max_end_error_beats", "precision", "gradient_checkpointing", "gradient_clip",
    "validation_interval", "max_validation_windows", "early_stopping_patience", "early_stopping_min_delta",
    "protocol", "schema_version", "bass_condition_supported",
)


def validate_native_precision(device, precision):
    """T4 的 BF16 仿真不等于硬件支持；仅 Ampere 及更新 CUDA GPU 使用 BF16。"""
    device = torch.device(device)
    if device.type == "cuda" and precision == "bf16":
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8:
            raise RuntimeError(
                f"bf16 requires native CUDA support (compute capability >= 8.0); {device} is {major}.{minor}. "
                "Set precision='fp32' for a new run. Resume must retain the checkpoint precision."
            )


def validate_resume_config(previous, current):
    """锁定更新与验证协议；保留旧版严格时值协议的既定默认值。"""
    previous = dict(previous)
    previous.setdefault("timing_policy", "strict_duration")
    previous.setdefault("max_end_error_beats", 0.125)
    changed = [key for key in RESUME_PROTOCOL_KEYS
               if key not in previous or key not in current or previous[key] != current[key]]
    if changed:
        raise ValueError("Resume changes training/validation configuration: " + ", ".join(changed))


def autocast(device, precision):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") and precision == "bf16" else nullcontext()


@torch.no_grad()
def evaluate(model, windows, vocab, device, batch_size, precision):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        tokens, segments, labels = collate(windows[start:start + batch_size], vocab, device)
        with autocast(device, precision):
            summed, amount = loss_sum(model(tokens, segments), labels)
        total += float(summed)
        count += amount
    if not count:
        raise ValueError("Empty validation targets")
    return {"nll": total / count, "target_tokens": count, "windows": len(windows)}


def rng_state(sampler, cuda):
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if cuda else [],
            "python": random.getstate(), "sampler": sampler.getstate()}


def restore_rng(state, sampler, cuda):
    torch.set_rng_state(state["torch"])
    if state["cuda"] and cuda:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    sampler.setstate(state["sampler"])


def save_checkpoint(path, model, optimizer, scheduler, metadata, sampler, best, bad):
    # 原子替换，预算停止时保留最近一个完整 optimizer step，避免半步状态。
    path = Path(path)
    temporary = path.with_suffix(".partial")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "metadata": metadata, "rng": rng_state(sampler, metadata["device"].startswith("cuda")), "best_validation_nll": best,
                "bad_validations": bad}, temporary)
    temporary.replace(path)


def main():
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum-steps", type=int)
    parser.add_argument("--window-bars", type=int)
    parser.add_argument("--max-validation-windows", type=int)
    parser.add_argument("--allow-partial-data", action="store_true", help="Only for bounded smoke tests; recorded in metadata")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    config = read_config(args.config)
    for name in ("batch_size", "accum_steps", "window_bars", "max_validation_windows"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.window_bars:
        config["stride_bars"] = args.window_bars
    if any(config[k] < 1 for k in ("batch_size", "accum_steps", "window_bars", "max_validation_windows")):
        raise ValueError("Batch/window limits must be positive")
    max_steps = args.max_steps if args.max_steps is not None else config["max_steps"]
    if max_steps < 1 or (args.max_wall_seconds is not None and args.max_wall_seconds <= 0):
        raise ValueError("Training budgets must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    validate_native_precision(args.device, config["precision"])
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Use a fresh output directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.samples).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    samples_hash = sha256(args.samples)
    if not args.allow_partial_data and (not manifest_path.exists() or manifest.get("partial_build") is not False or manifest.get("samples_sha256") != samples_hash):
        raise ValueError("Training requires a complete canonical manifest with matching samples_sha256; bounded smoke may explicitly allow partial data")
    if args.allow_partial_data and args.max_steps is None:
        raise ValueError("Partial-data smoke training requires explicit --max-steps")
    torch.set_num_threads(args.cpu_threads)
    seed_all(config["seed"])
    sampler = random.Random(config["seed"])
    vocab = Vocabulary(config["upstream_dir"])
    rows = load_samples(args.samples)
    if sha256(args.samples) != samples_hash:
        raise ValueError("Canonical dataset changed while loading")
    windows, dataset_report = prepare_windows(rows, vocab, config)
    train = [w for w in windows if w.split == "train"]
    validation = [w for w in windows if w.split == "validation"]
    if not train or not validation:
        write_json(output / "dataset_audit.json", dataset_report)
        raise ValueError(f"Require supported train and validation windows; found {len(train)}, {len(validation)}")
    validation_sampler = random.Random(config["seed"] + 1)
    validation = validation_sampler.sample(validation, min(len(validation), config["max_validation_windows"]))
    grouped = defaultdict(list)
    for window in train:
        grouped[window.work_id].append(window)
    works = sorted(grouped)
    write_json(output / "dataset_audit.json", dataset_report)
    write_json(output / "resolved_config.json", config)
    source_checkpoint = args.resume or config["pretrained_checkpoint"]
    model, loading = load_model(vocab, source_checkpoint, args.device, config["gradient_checkpointing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def rate(step):
        if step < config["warmup_steps"]:
            return (step + 1) / max(1, config["warmup_steps"])
        progress = min(1.0, (step - config["warmup_steps"]) / max(1, config["max_steps"] - config["warmup_steps"]))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    step, best, bad = 0, float("inf"), 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        previous = payload["metadata"]
        if previous.get("protocol") != PROTOCOL or previous.get("samples_sha256") != samples_hash:
            raise ValueError("Resume protocol or dataset hash differs")
        validate_resume_config(previous["config"], config)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step = previous["optimizer_steps"]
        best, bad = payload["best_validation_nll"], payload["bad_validations"]
        restore_rng(payload["rng"], sampler, args.device.startswith("cuda"))
        del payload
    metadata = {"protocol": PROTOCOL, "config": config, "samples_sha256": samples_hash,
                "vocabulary_sha256": vocab.sha256, "source_checkpoint": loading, "optimizer_steps": step,
                "canonical_manifest_partial": manifest.get("partial_build"), "allow_partial_data": args.allow_partial_data,
                "test_used_for_selection": False, "sampling": "uniform work, then uniform supported window",
                "validation_song_ids": sorted(set(w.song_id for w in validation)),
                "validation_windows": len(validation), "validation_target_tokens": sum(w.stats["target_tokens"] for w in validation),
                "bass_condition_supported": False, "target": "PIANO accompaniment only; original melody merged after generation",
                "torch_version": str(torch.__version__), "device": args.device}
    write_json(output / "training_metadata.json", metadata)
    print(json.dumps({"parameters": loading["parameters"], "strict_load": True, "train_windows": len(train),
                      "validation_windows": len(validation), "starting_step": step, "max_steps": max_steps}), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    setup_seconds = started - invocation_started
    validation_seconds_total, checkpoint_seconds_total = 0.0, 0.0
    measurements, effective_tokens = [], []
    stop_reason = "max_steps"
    history_path = output / "history.jsonl"
    while step < max_steps:
        if args.max_wall_seconds is not None and time.monotonic() - started >= args.max_wall_seconds:
            stop_reason = "wall_budget"
            break
        model.train()
        batches = [[sampler.choice(grouped[sampler.choice(works)]) for _ in range(config["batch_size"])] for _ in range(config["accum_steps"])]
        target_count = sum(sum(w.stats["target_tokens"] for w in batch) for batch in batches)
        input_count = sum(sum(len(w.tokens) for w in batch) for batch in batches)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        step_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        summed_nll = 0.0
        for batch in batches:
            tokens, segments, labels = collate(batch, vocab, args.device)
            with autocast(args.device, config["precision"]):
                summed, _ = loss_sum(model(tokens, segments), labels)
                loss = summed / target_count
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; previous checkpoint remains available")
            loss.backward()
            summed_nll += float(summed.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"]))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm; optimizer update rejected")
        optimizer.step()
        scheduler.step()
        step += 1
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.monotonic() - step_start
        measurements.append(seconds)
        effective_tokens.append(input_count)
        metadata["optimizer_steps"] = step
        record = {"step": step, "train_nll": summed_nll / target_count, "optimizer_step_seconds": seconds,
                  "input_tokens": input_count, "target_tokens": target_count, "grad_norm": grad_norm,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        budget_expired = args.max_wall_seconds is not None and time.monotonic() - started >= args.max_wall_seconds
        if not budget_expired and (step % config["validation_interval"] == 0 or step == max_steps):
            validation_start = time.monotonic()
            record["validation"] = evaluate(model, validation, vocab, args.device, config["batch_size"], config["precision"])
            record["validation_seconds"] = time.monotonic() - validation_start
            validation_seconds_total += record["validation_seconds"]
            score = record["validation"]["nll"]
            if score < best - config["early_stopping_min_delta"]:
                best, bad = score, 0
                save_started = time.monotonic()
                save_checkpoint(output / "best.pt", model, optimizer, scheduler, metadata, sampler, best, bad)
                checkpoint_seconds_total += time.monotonic() - save_started
            else:
                bad += 1
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if step % config["checkpoint_interval"] == 0 or step == max_steps or budget_expired:
            save_started = time.monotonic()
            save_checkpoint(output / "last.pt", model, optimizer, scheduler, metadata, sampler, best, bad)
            checkpoint_seconds_total += time.monotonic() - save_started
        if bad >= config["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    save_started = time.monotonic()
    save_checkpoint(output / "last.pt", model, optimizer, scheduler, metadata, sampler, best, bad)
    checkpoint_seconds_total += time.monotonic() - save_started
    trimmed = measurements[config["profile_skip_steps"]:]
    summary = {"protocol": PROTOCOL, "optimizer_steps": step, "steps_this_invocation": len(measurements),
               "stop_reason": stop_reason, "wall_seconds": time.monotonic() - started,
               "max_wall_seconds": args.max_wall_seconds,
               "wall_budget_contract": "training loop budget excludes setup; checked between complete optimizer steps; validation and checkpoint I/O can also exceed deadline",
               "setup_seconds": setup_seconds, "validation_seconds": validation_seconds_total,
               "checkpoint_seconds": checkpoint_seconds_total, "total_invocation_seconds": time.monotonic() - invocation_started,
               "best_validation_nll": best if math.isfinite(best) else None,
               "optimizer_step_seconds_after_warmup": {"excluded_initial_steps": config["profile_skip_steps"],
                   "count": len(trimmed), "mean": statistics.mean(trimmed) if trimmed else None,
                   "median": statistics.median(trimmed) if trimmed else None, "max": max(trimmed) if trimmed else None},
               "input_tokens_per_step": effective_tokens, "batch_size": config["batch_size"], "accum_steps": config["accum_steps"],
               "input_tokens_per_step_mean": statistics.mean(effective_tokens) if effective_tokens else None,
               "input_tokens_per_step_max": max(effective_tokens) if effective_tokens else None,
               "max_tokens": config["max_tokens"], "window_bars": config["window_bars"],
               "precision": config["precision"] if args.device.startswith("cuda") else "float32", "requested_precision": config["precision"],
               "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
               "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if args.device.startswith("cuda") else None,
               "quality_status": "training telemetry, not listening acceptance", "test_used": False}
    write_json(output / "training_summary.json", summary)
    write_json(output / "training_metadata.json", metadata)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
