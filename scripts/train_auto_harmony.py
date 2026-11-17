"""对真实 ReaLchords 离线 checkpoint 进行监督微调；默认仅准备，--run 才更新参数。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from auto_harmony import (
    BOS, EOS, PAD, SILENCE, PRETRAINING_EXPOSURE, HarmonyVocabulary,
    encode_corpus, load_model, make_batch, read_samples, resolve_path,
    sha256_file, torch, np, write_json,
)
import torch.nn.functional as F


RESUME_PROTOCOL_KEYS = (
    "seed", "max_frames", "batch_size", "accumulation_steps", "learning_rate", "weight_decay",
    "warmup_steps", "max_steps", "source_weights", "transpose_semitones", "onset_weight",
    "melody_overlap_policy", "precision", "gradient_clip_norm", "validation_interval",
    "validation_max_windows", "validate_at_end", "early_stopping_patience", "early_stopping_min_delta",
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
    """锁定更新与验证协议；输出路径、GPU、计时及日志选项仍可改变。"""
    changed = [key for key in RESUME_PROTOCOL_KEYS
               if key not in previous or key not in current or previous[key] != current[key]]
    if changed:
        raise ValueError("Resume changes training/validation configuration: " + ", ".join(changed))


def masked_loss(logits, batch, vocabulary: HarmonyVocabulary, onset_weight: float):
    labels = batch["labels"]
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                             ignore_index=-100, reduction="none").view_as(labels)
    weights = batch["label_mask"].float()
    weights = weights * torch.where(labels.ge(vocabulary.on_start), onset_weight, 1.0)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("Batch has no supervised targets")
    return (losses * weights).sum() / denominator, losses


def move_batch(batch, device):
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


def forward(model, batch):
    return model(batch["inputs"], batch["decoder"],
                 enc_mask=batch["enc_mask"], dec_mask=batch["dec_mask"])


class WindowSampler:
    """先按来源、再按歌曲抽样，避免长曲或大数据源淹没其他监督。"""

    def __init__(self, songs, vocabulary, config):
        self.vocabulary, self.config = vocabulary, config
        self.groups = defaultdict(list)
        for song in songs:
            if song.row["split"] == "train" and song.known.any():
                self.groups[song.row["source"]].append(song)
        self.sources = sorted(self.groups)
        self.work_groups = {}
        self.source_draws = Counter()
        self.source_frames = Counter()
        self.sampled_works = set()
        for source, members in self.groups.items():
            works = defaultdict(list)
            for song in members:
                works[song.row["work_id"]].append(song)
            self.work_groups[source] = list(works.values())
        if not self.sources:
            raise ValueError("No training songs with known supported chords")
        supplied = config.get("source_weights", {})
        if supplied and set(supplied) != set(self.sources):
            raise ValueError(f"source_weights must cover exactly available training sources: {self.sources}")
        weights = np.array([float(supplied.get(source, 1.0)) for source in self.sources])
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("Source weights must be finite and positive")
        self.probabilities = weights / weights.sum()

    def sample(self, rng):
        windows = []
        max_frames = self.config["max_frames"]
        attempts = 0
        while len(windows) < self.config["batch_size"]:
            attempts += 1
            if attempts > self.config["batch_size"] * 200:
                raise ValueError("Could not sample enough windows with valid labels")
            source = str(rng.choice(self.sources, p=self.probabilities))
            works = self.work_groups[source]
            members = works[int(rng.integers(len(works)))]
            song = members[int(rng.integers(len(members)))]
            length = min(max_frames, len(song.melody))
            start = int(rng.integers((len(song.melody) - length) // 4 + 1)) * 4
            if not song.known[start:start + length].any():
                continue
            shifts = self.config.get("transpose_semitones", [0])
            shift = int(rng.choice(shifts))
            if shift:
                mapping = self.vocabulary.transpose_map(shift)
                if (mapping[song.melody[start:start + length]] == PAD).any():
                    continue
                shifted_y = mapping[song.chords[start:start + length]]
                if not (song.known[start:start + length] & (shifted_y != PAD)).any():
                    continue
            windows.append((song, start, length, shift))
            self.source_draws[source] += 1
            self.source_frames[source] += length
            self.sampled_works.add(song.row["work_id"])
        return make_batch(windows, self.vocabulary)


def validation_windows(songs, config):
    windows = []
    for song in songs:
        if song.row["split"] != "validation":
            continue
        for start in range(0, len(song.melody), config["max_frames"]):
            length = min(config["max_frames"], len(song.melody) - start)
            if song.known[start:start + length].any():
                windows.append((song, start, length, 0))
    if not windows:
        raise ValueError("No validation windows with known supported chords")
    rng = np.random.default_rng(config["seed"] + 1)
    order = rng.permutation(len(windows))
    limit = config.get("validation_max_windows", 0)
    if limit > 0:
        order = order[:limit]
    return [windows[int(i)] for i in order]


@torch.no_grad()
def evaluate(model, windows, vocabulary, config, device):
    was_training = model.training
    model.eval()
    loss_sum = label_count = correct = frames = invalid = 0
    tp = fp = fn = 0
    started = time.perf_counter()
    for offset in range(0, len(windows), config["batch_size"]):
        batch = move_batch(make_batch(windows[offset:offset + config["batch_size"]], vocabulary), device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=config["precision"] == "bf16" and device.type == "cuda"):
            logits = forward(model, batch)
        _, losses = masked_loss(logits, batch, vocabulary, 1.0)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite validation loss")
        mask = batch["label_mask"]
        frame_mask = batch["frame_mask"]
        predictions = logits.argmax(-1)
        labels = batch["labels"]
        loss_sum += float((losses * mask).sum())
        label_count += int(mask.sum())
        correct += int(((predictions == labels) & frame_mask).sum())
        frames += int(frame_mask.sum())
        invalid += int((((predictions < vocabulary.hold_start) & (predictions != SILENCE)) & frame_mask).sum())
        predicted_boundary = predictions.ge(vocabulary.on_start) & frame_mask
        target_boundary = labels.ge(vocabulary.on_start) & frame_mask
        tp += int((predicted_boundary & target_boundary).sum())
        fp += int((predicted_boundary & ~target_boundary & frame_mask).sum())
        fn += int((~predicted_boundary & target_boundary).sum())
    model.train(was_training)
    return {"teacher_forced_nll": loss_sum / max(label_count, 1),
            "teacher_forced_frame_accuracy": correct / max(frames, 1),
            "teacher_forced_boundary_f1_exact_grid": 2 * tp / max(2 * tp + fp + fn, 1),
            "teacher_forced_invalid_token_rate": invalid / max(frames, 1),
            "known_frames": frames, "supervised_tokens_including_eos": label_count,
            "windows": len(windows), "seconds": time.perf_counter() - started,
            "scope": "development validation; teacher-forced metrics are not free-running harmony quality",
            "pretraining_exposure": PRETRAINING_EXPOSURE}


def save_checkpoint(path, model, optimizer, scheduler, rng, step, config, metadata,
                    best_validation, samples_sha256, bad_validation_checks=0):
    payload = {
        "format": "pianalysis_auto_harmony_v1", "model": model.state_dict(),
        "model_kwargs": metadata["model_kwargs"], "vocabulary_sha256": metadata["vocabulary"]["sha256"],
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "config": config, "best_validation": best_validation,
        "bad_validation_checks": bad_validation_checks,
        "samples_sha256": samples_sha256, "metadata": metadata,
        "numpy_generator_state": rng.bit_generator.state, "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, args):
    started = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    validate_native_precision(device, config["precision"])
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_sha = sha256_file(resolve_path(config["samples"]))
    manifest_path = resolve_path(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    verified_data = (manifest.get("partial_build") is False
                     and manifest.get("samples_sha256") == samples_sha
                     and manifest.get("time_unit") == "quarter_note"
                     and manifest.get("split_before_slicing_and_augmentation") is True)
    if not verified_data:
        bounded_smoke = (args.allow_unverified_data and 0 < args.stop_after_steps <= 100
                         and 0 < config["max_wall_seconds"] <= 600)
        if not bounded_smoke:
            raise ValueError("Require complete manifest with matching samples SHA/time unit/split contract; unverified data only allowed by explicit bounded smoke")
    checkpoint_path = args.resume or config.get("checkpoint")
    model, vocabulary, metadata, loaded = load_model(config["upstream"], checkpoint_path)
    if config["max_frames"] > metadata["max_frames"] or config["max_frames"] < 1:
        raise ValueError(f"max_frames must be in 1..{metadata['max_frames']}")
    rows = read_samples(config["samples"])
    if verified_data and manifest.get("sample_count") != len(rows):
        raise ValueError("Manifest sample count mismatch")
    songs, encoding_audit = encode_corpus(rows, vocabulary, config["melody_overlap_policy"])
    del rows
    write_json(output_dir / "encoding_audit.json", encoding_audit)
    if not songs:
        raise ValueError("All songs were rejected during encoding")
    sampler = WindowSampler(songs, vocabulary, config)
    val_windows = validation_windows(songs, config)
    preparation = {"model": metadata, "config": config, "samples_sha256": samples_sha,
                   "manifest": str(manifest_path), "manifest_verified": verified_data,
                   "encoded_songs": len(songs), "rejected_songs": len(encoding_audit["rejected"]),
                   "train_sources": {source: len(sampler.groups[source]) for source in sampler.sources},
                   "train_works_by_source": {source: len(sampler.work_groups[source]) for source in sampler.sources},
                   "sampling": "source by configured weights, then uniform work, uniform segment, uniform beat-aligned window",
                   "validation_windows": len(val_windows), "execution": "prepare_only" if not args.run else "training"}
    write_json(output_dir / "preparation.json", preparation)
    if not args.run:
        print(json.dumps({"status": "prepared_no_parameter_updates", "report": str(output_dir / 'preparation.json')}, ensure_ascii=False))
        return
    # 保留初始权重供短步数值验证；仅复制一个小张量，避免额外显存占用。
    initial_probe = next(model.parameters()).detach().cpu().flatten()[:4096].clone()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])

    def lr_scale(step):
        warmup = config["warmup_steps"]
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(config["max_steps"] - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step, best_validation, bad_validation_checks = 0, math.inf, 0
    if args.resume:
        if loaded.get("format") != "pianalysis_auto_harmony_v1":
            raise ValueError("--resume requires a project training checkpoint; use checkpoint for upstream initialization")
        if loaded["samples_sha256"] != samples_sha:
            raise ValueError("Resume data hash changed")
        validate_resume_config(loaded["config"], config)
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        step, best_validation = loaded["step"], loaded["best_validation"]
        bad_validation_checks = loaded.get("bad_validation_checks", 0)
        rng.bit_generator.state = loaded["numpy_generator_state"]
        random.setstate(loaded["python_rng_state"])
        torch.set_rng_state(loaded["torch_rng_state"])
        if loaded["cuda_rng_states"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(loaded["cuda_rng_states"])
    del loaded
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_step = step
    step_seconds, losses_history = [], []
    validation_seconds, checkpoint_seconds = 0.0, 0.0
    last_validation = None
    stop_reason = "max_steps"
    training_started = time.perf_counter()
    log_path = output_dir / "metrics.jsonl"
    try:
        with log_path.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as log:
            while step < config["max_steps"]:
                if config["max_wall_seconds"] > 0 and time.perf_counter() - started >= config["max_wall_seconds"]:
                    stop_reason = "max_wall_seconds"
                    break
                if args.stop_after_steps and step - initial_step >= args.stop_after_steps:
                    stop_reason = "stop_after_steps"
                    break
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                total_loss, supervised = 0.0, 0
                for _ in range(config["accumulation_steps"]):
                    batch = move_batch(sampler.sample(rng), device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=config["precision"] == "bf16" and device.type == "cuda"):
                        logits = forward(model, batch)
                        loss, _ = masked_loss(logits, batch, vocabulary, config["onset_weight"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss / config["accumulation_steps"]).backward()
                    total_loss += float(loss.detach()) / config["accumulation_steps"]
                    supervised += int(batch["frame_mask"].sum())
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                step += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - tick
                step_seconds.append(elapsed)
                losses_history.append(total_loss)
                entry = {"step": step, "train_loss": total_loss, "gradient_norm": float(grad_norm),
                         "lr": optimizer.param_groups[0]["lr"], "optimizer_step_seconds": elapsed,
                         "supervised_frames": supervised}
                if step % config["validation_interval"] == 0:
                    last_validation = evaluate(model, val_windows, vocabulary, config, device)
                    validation_seconds += last_validation["seconds"]
                    entry["validation"] = last_validation
                    score = last_validation["teacher_forced_nll"]
                    if score < best_validation - config["early_stopping_min_delta"]:
                        best_validation = score
                        bad_validation_checks = 0
                        checkpoint_tick = time.perf_counter()
                        save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, rng,
                                        step, config, metadata, best_validation, samples_sha, bad_validation_checks)
                        checkpoint_seconds += time.perf_counter() - checkpoint_tick
                    else:
                        bad_validation_checks += 1
                log.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)
                if step % config["checkpoint_interval"] == 0:
                    checkpoint_tick = time.perf_counter()
                    save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, rng,
                                    step, config, metadata, best_validation, samples_sha, bad_validation_checks)
                    checkpoint_seconds += time.perf_counter() - checkpoint_tick
                if config["early_stopping_patience"] > 0 and bad_validation_checks >= config["early_stopping_patience"]:
                    stop_reason = "early_stopping"
                    break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    # wall-clock停止后不再额外运行验证；checkpoint仍必须落盘以支持续训。
    if last_validation is None and config["validate_at_end"] and stop_reason != "max_wall_seconds":
        last_validation = evaluate(model, val_windows, vocabulary, config, device)
        validation_seconds += last_validation["seconds"]
        score = last_validation["teacher_forced_nll"]
        if score < best_validation - config["early_stopping_min_delta"]:
            best_validation = score
            bad_validation_checks = 0
            checkpoint_tick = time.perf_counter()
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, rng,
                            step, config, metadata, best_validation, samples_sha, bad_validation_checks)
            checkpoint_seconds += time.perf_counter() - checkpoint_tick
    checkpoint_tick = time.perf_counter()
    save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, rng,
                    step, config, metadata, best_validation, samples_sha, bad_validation_checks)
    checkpoint_seconds += time.perf_counter() - checkpoint_tick
    probe = next(model.parameters()).detach().cpu().flatten()[:4096]
    measured = step_seconds[config["profile_warmup_steps"]:]
    report = {"status": stop_reason, "optimizer_steps_total": step,
              "optimizer_steps_this_run": step - initial_step, "seconds_total": time.perf_counter() - started,
              "seconds_training_and_finalization": time.perf_counter() - training_started,
              "seconds_setup": training_started - started, "seconds_validation": validation_seconds,
              "seconds_checkpoint_writes": checkpoint_seconds,
              "seconds_optimizer_steps": sum(step_seconds), "optimizer_step_seconds": step_seconds,
              "profile_warmup_steps": config["profile_warmup_steps"],
              "profile_measured_steps": len(measured),
              "mean_step_seconds_excluding_warmup": float(np.mean(measured)) if measured else None,
              "median_step_seconds_excluding_warmup": float(np.median(measured)) if measured else None,
              "sequence_length_frames": config["max_frames"], "microbatch": config["batch_size"],
              "gradient_accumulation": config["accumulation_steps"], "precision": config["precision"],
              "parameters": metadata["parameters"], "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
              "parameter_probe_max_change": float((probe - initial_probe).abs().max()),
              "train_losses": losses_history, "validation": last_validation,
              "sampling_draws_by_source": dict(sampler.source_draws),
              "sampling_frames_by_source": dict(sampler.source_frames),
              "sampling_unique_works": len(sampler.sampled_works),
              "bad_validation_checks": bad_validation_checks,
              "checkpoint": str(output_dir / "last.pt"), "pretraining_exposure": PRETRAINING_EXPOSURE,
              "budget_note": "wall limit checked before each optimizer step; a running step and final checkpoint write can exceed the limit"}
    write_json(output_dir / "run_report.json", report)
    print(json.dumps({"report": str(output_dir / 'run_report.json'), "optimizer_steps": step, "status": stop_reason}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/auto_accompaniment/harmony_train.json")
    parser.add_argument("--samples")
    parser.add_argument("--manifest")
    parser.add_argument("--run", action="store_true", help="明确启用参数更新；默认只准备数据和模型")
    parser.add_argument("--allow-unverified-data", action="store_true", help="仅用于<=100步且<=600秒的显式smoke")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="仅本次运行的短步上限，保留总训练schedule")
    for name in ("max-steps", "batch-size", "accumulation-steps", "max-frames", "profile-warmup-steps"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--validation-max-windows", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    if args.samples:
        config["samples"] = args.samples
        config["manifest"] = str(Path(args.samples).with_name("manifest.json"))
    if args.manifest:
        config["manifest"] = args.manifest
    for key in ("max_steps", "batch_size", "accumulation_steps", "max_frames", "profile_warmup_steps",
                "max_wall_seconds", "validation_max_windows", "output_dir", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("max_steps", "batch_size", "accumulation_steps", "validation_interval", "checkpoint_interval"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["precision"] not in ("fp32", "bf16") or config["onset_weight"] <= 0:
        raise ValueError("Invalid precision or onset weight")
    if config["warmup_steps"] < 0 or config["profile_warmup_steps"] < 0 or args.stop_after_steps < 0:
        raise ValueError("Step counts must be nonnegative")
    if config["early_stopping_patience"] < 0 or config["early_stopping_min_delta"] < 0:
        raise ValueError("Early stopping settings must be nonnegative")
    train(config, args)


if __name__ == "__main__":
    main()
