"""Colab 独立 Python 环境中的训练调度与 Drive 持久化入口。"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from colab_runtime import DriveStore, commit_record, digest, json_hash, read_record, write_json

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_KEYS = {"output_dir", "device", "max_wall_seconds", "samples", "manifest"}


def semantic_config(config):
    return {key: value for key, value in config.items() if key not in RUNTIME_KEYS}


def choose_config(base, stage, hardware, existing=None):
    if existing is not None:
        result = dict(existing)
        if result["precision"] == "bf16" and not hardware["native_bf16"]:
            raise ValueError("上次运行使用 bf16，本次 GPU 不支持原生 bf16。请选择兼容 GPU；不能静默改精度恢复。")
        return result
    result = dict(base)
    accumulation = "accumulation_steps" if stage == "harmony" else "accum_steps"
    effective = int(result["batch_size"]) * int(result[accumulation])
    if not hardware["native_bf16"]:
        result["precision"] = "fp32" if stage == "harmony" else "float32"
    if hardware["memory_gib"] < 24 or not hardware["native_bf16"]:
        result["batch_size"] = 1
    if effective % result["batch_size"]:
        raise ValueError("Cannot preserve effective batch size")
    result[accumulation] = effective // result["batch_size"]
    # 只调整全新 run 的保存频率；已存在 run 严格采用原有效配置。
    result["checkpoint_interval"] = max(500, result["checkpoint_interval"])
    return result


def inspect_gpu():
    import torch
    import transformers
    if sys.version_info[:2] != (3, 11) or not str(torch.__version__).startswith("2.5.1") or transformers.__version__ != "4.57.1":
        raise RuntimeError("请使用 Notebook 创建的 Python 3.11 / torch 2.5.1 / transformers 4.57.1 独立环境")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Colab 没有可用 GPU；请在运行时设置中选择 GPU")
    properties = torch.cuda.get_device_properties(0)
    native_bf16 = properties.major >= 8 and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if native_bf16 else torch.float32
    # capability 之外再做实际矩阵前向/反向，不把模拟 bf16 当硬件支持。
    matrix = torch.randn(128, 128, device="cuda", dtype=dtype, requires_grad=True)
    (matrix @ matrix.T).float().square().mean().backward()
    if not torch.isfinite(matrix.grad).all():
        raise RuntimeError("CUDA smoke produced a non-finite gradient")
    del matrix
    torch.cuda.empty_cache()
    return {"name": properties.name, "compute_capability": [properties.major, properties.minor],
            "memory_gib": properties.total_memory / 1024**3, "native_bf16": native_bf16,
            "python": sys.version, "torch": str(torch.__version__), "transformers": transformers.__version__,
            "cuda_forward_backward_verified": True}


def stage_contract(stage, config, bundle):
    vocabulary = ROOT / ("data/auto_accompaniment/upstream/realchords/weights/chord_names_augmented.json" if stage == "harmony" else
                         "data/auto_accompaniment/upstream/compose_embellish/stage02_embellish/vocab/skyline2midi_vocab.pkl")
    upstream_manifest = ROOT / ("data/auto_accompaniment/upstream/realchords/source_manifest.json" if stage == "harmony" else
                                "data/auto_accompaniment/upstream/compose_embellish/manifest.json")
    return {"stage": stage, "samples_sha256": bundle["samples_sha256"], "vocabulary_sha256": digest(vocabulary),
            "base_config_sha256": json_hash(config), "upstream_manifest_sha256": digest(upstream_manifest),
            "code_sha256": {name: digest(ROOT / "scripts" / name) for name in (f"auto_{stage}.py", f"train_auto_{stage}.py")},
            "requirements_sha256": digest(ROOT / "requirements-auto-accompaniment.txt")}


def select_existing(stage_root, contract):
    candidates, existing = [], False
    for marker in Path(stage_root).glob("run_*/checkpoints/g_*/commit.json"):
        existing = True
        try:
            record = read_record(marker)
            if record["contract"] == contract:
                candidates.append((record["step"], record["created_ns"], marker.parents[2], record))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for _, _, run_root, record in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
        store = DriveStore(run_root, record["run_id"], contract, record["config"])
        if store.latest() is not None:
            return run_root, record["config"]
    if existing:
        raise ValueError("Drive 中已有 checkpoint，但数据/词表/代码/配置不兼容或所有代均损坏；不会静默从头重训。请保留旧目录后明确使用新的项目目录。")
    return None


def validate_restored_checkpoint(path, stage, contract, config):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if stage == "harmony":
        if payload.get("format") != "pianalysis_auto_harmony_v1":
            raise ValueError("Wrong harmony checkpoint protocol")
        actual_config, data_hash, vocabulary_hash, step = payload["config"], payload["samples_sha256"], payload["vocabulary_sha256"], payload["step"]
    else:
        metadata = payload["metadata"]
        if metadata.get("protocol") != "compose_embellish_accompaniment_only_v1":
            raise ValueError("Wrong accompaniment checkpoint protocol")
        actual_config, data_hash, vocabulary_hash, step = metadata["config"], metadata["samples_sha256"], metadata["vocabulary_sha256"], metadata["optimizer_steps"]
    if data_hash != contract["samples_sha256"] or vocabulary_hash != contract["vocabulary_sha256"] or semantic_config(actual_config) != semantic_config(config):
        raise ValueError("Restored payload disagrees with committed data/vocabulary/effective configuration")
    if any(key not in payload for key in ("model", "optimizer", "scheduler")):
        raise ValueError("Checkpoint lacks optimizer/scheduler resume state")
    bad = payload.get("bad_validation_checks", 0) if stage == "harmony" else payload.get("bad_validations", 0)
    early_stopped = config.get("early_stopping_patience", 0) > 0 and bad >= config["early_stopping_patience"]
    del payload
    gc.collect()
    return {"step": int(step), "early_stopped": early_stopped}


def worker(args):
    descriptor = json.loads(Path(args.descriptor).read_text(encoding="utf-8"))
    stage, local = descriptor["stage"], Path(descriptor["local_root"])
    store = DriveStore(descriptor["drive_run"], descriptor["run_id"], descriptor["contract"], descriptor["config"], descriptor["keep_all"])
    native = importlib.import_module("train_auto_" + stage)
    original_save = native.save_checkpoint
    stop, sync_errors = threading.Event(), []

    def sync_loop():
        while not stop.wait(10):
            try:
                store.sync_artifacts(local)
            except Exception as exc:
                sync_errors.append(repr(exc))
                write_json(local / "drive_sync_error.json", {"error": repr(exc), "time_ns": time.time_ns()})
                return

    def durable_save(*positional, **keyword):
        if sync_errors:
            raise RuntimeError("Drive artifact synchronization failed: " + sync_errors[-1])
        bound = inspect.signature(original_save).bind(*positional, **keyword)
        original_save(*positional, **keyword)
        path = Path(bound.arguments["path"])
        step = bound.arguments["step"] if stage == "harmony" else bound.arguments["metadata"]["optimizer_steps"]
        started = time.monotonic()
        folder, record = store.save_checkpoint(path, path.stem, step)
        print(json.dumps({"drive_checkpoint_committed": str(folder), "step": step, "kind": record["kind"],
                          "sync_seconds": time.monotonic() - started}), flush=True)

    native.save_checkpoint = durable_save
    thread = threading.Thread(target=sync_loop, daemon=True)
    thread.start()
    try:
        sys.argv = [str(ROOT / "scripts" / f"train_auto_{stage}.py")] + descriptor["native_arguments"]
        native.main()
    finally:
        stop.set()
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("Drive artifact synchronization did not finish; prior committed checkpoints remain recoverable")
        store.sync_artifacts(local)


def run_phase(stage, phase, local, store, config_path, descriptor_base, resume, current_step, seconds):
    train_dir = local / "train"
    native_arguments = ["--config", str(config_path), "--samples", str(ROOT / "data/auto_accompaniment/processed/samples.jsonl"),
                        "--output-dir", str(train_dir), "--device", "cuda", "--max-wall-seconds", str(max(1, seconds - min(180, seconds * 0.2)))]
    if stage == "harmony":
        native_arguments.append("--run")
    if resume:
        native_arguments += ["--resume", str(resume)]
    if phase == "profile":
        native_arguments += ["--stop-after-steps", "6"] if stage == "harmony" else ["--max-steps", str(min(current_step + 6, descriptor_base["config"]["max_steps"]))]
    stamp = str(time.time_ns())
    descriptor = dict(descriptor_base, native_arguments=native_arguments)
    descriptor_path = local / "sessions" / (stamp + ".descriptor.json")
    write_json(descriptor_path, descriptor)
    log_path = local / "sessions" / (stamp + f".{phase}.log")
    started, last_print = time.monotonic(), time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--descriptor", str(descriptor_path)]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            while process.poll() is None:
                if time.monotonic() - started >= seconds:
                    timed_out = True
                    # 先给 native 保存机会；达到硬界限时终止整个 worker 进程组。
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    break
                if time.monotonic() - last_print >= 15:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    print(stage, phase, lines[-1] if lines else "准备中…", flush=True)
                    last_print = time.monotonic()
                time.sleep(1)
        except KeyboardInterrupt:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            raise
        finally:
            if process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    record = {"stage": stage, "phase": phase, "seconds": time.monotonic() - started, "exit_code": process.returncode,
              "hard_timeout": timed_out, "log": str(log_path), "time_limit_seconds": seconds,
              "deadline_note": "Controller includes a final interrupt/save grace period of up to 60 seconds; Drive FUSE latency can extend finalization."}
    write_json(local / "sessions" / (stamp + f".{phase}.result.json"), record)
    for filename in ("training_summary.json", "run_report.json"):
        source = train_dir / filename
        if source.exists():
            write_json(local / "sessions" / (stamp + f".{phase}." + filename), json.loads(source.read_text(encoding="utf-8")))
    store.sync_artifacts(local)
    if process.returncode and not timed_out:
        raise RuntimeError(f"{stage} {phase} failed. See {log_path} and its Drive artifact snapshots")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root")
    parser.add_argument("--stage", choices=("harmony", "texture", "both"), default="both")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--keep-all-checkpoints", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--descriptor", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not args.drive_root or not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("Provide --drive-root and a positive finite --max-hours")
    drive_root = Path(args.drive_root).resolve()
    if not drive_root.is_dir():
        raise ValueError("Drive project folder is missing; mount Drive and upload inputs first")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    bundle = json.loads((ROOT / ".colab_bundle_receipt.json").read_text(encoding="utf-8"))
    samples = ROOT / "data/auto_accompaniment/processed/samples.jsonl"
    if digest(samples) != bundle["samples_sha256"]:
        raise ValueError("Extracted canonical data differs from the verified bundle")
    from verify_auto_training_data import audit
    gate = audit(samples)
    if not gate["passed"] or gate["samples_sha256"] != bundle["samples_sha256"]:
        raise ValueError("Independent canonical-data gate failed")
    hardware = inspect_gpu()
    print(json.dumps({"hardware": hardware, "hours": args.max_hours, "stage": args.stage}, ensure_ascii=False), flush=True)
    stages = ["harmony", "texture"] if args.stage == "both" else [args.stage]
    for index, stage in enumerate(stages):
        remaining = deadline - time.monotonic()
        if remaining < 120:
            print("总预算不足，已提交的 checkpoint 可在下次继续。", flush=True)
            break
        stage_deadline = time.monotonic() + remaining / (len(stages) - index)
        base = json.loads((ROOT / f"examples/auto_accompaniment/{stage}_train.json").read_text(encoding="utf-8"))
        contract = stage_contract(stage, base, bundle)
        stage_root = drive_root / "runs" / stage
        found = select_existing(stage_root, contract)
        if found:
            drive_run, existing_config = found
            config = choose_config(base, stage, hardware, existing_config)
            run_id = drive_run.name
        else:
            run_id = "run_" + uuid.uuid4().hex
            drive_run = stage_root / run_id
            config = choose_config(base, stage, hardware)
        store = DriveStore(drive_run, run_id, contract, config, args.keep_all_checkpoints)
        local = Path("/content/pianalysis_runs") / run_id
        local.mkdir(parents=True, exist_ok=True)
        store.restore_artifacts(local)
        checkpoint = None
        current_step = 0
        early_stopped = False
        if found:
            restored = local / "restore" / (str(time.time_ns()) + ".pt")
            store.restore(restored)
            progress = validate_restored_checkpoint(restored, stage, contract, config)
            current_step, early_stopped = progress["step"], progress["early_stopped"]
            checkpoint = restored
        config_path = local / "effective_config.json"
        write_json(config_path, config)
        write_json(local / "environment.json", hardware)
        write_json(local / "bundle_receipt.json", bundle)
        write_json(local / "data_gate.json", gate)
        write_json(local / "run_contract.json", {"contract": contract, "config": config, "run_id": run_id,
                   "retention": "latest 2 last + latest 1 best; logs/config/reports retained" if not args.keep_all_checkpoints else "all generations",
                   "resume_step": current_step})
        store.sync_artifacts(local)
        if current_step >= config["max_steps"] or early_stopped:
            print(f"{stage} 已达到最大步数或早停条件；保留现有结果。", flush=True)
            continue
        descriptor = {"stage": stage, "local_root": str(local), "drive_run": str(drive_run), "run_id": run_id,
                      "contract": contract, "config": config, "keep_all": args.keep_all_checkpoints}
        profile_seconds = min(600, max(60, (stage_deadline - time.monotonic()) * 0.2))
        result = run_phase(stage, "profile", local, store, config_path, descriptor, checkpoint, current_step, profile_seconds)
        latest = store.latest()
        if latest is None:
            raise RuntimeError("Profile did not leave a committed checkpoint; formal training was not started")
        next_restore = local / "restore" / (str(time.time_ns()) + ".pt")
        store.restore(next_restore)
        progress = validate_restored_checkpoint(next_restore, stage, contract, config)
        current_step = progress["step"]
        remaining_stage = stage_deadline - time.monotonic()
        if result["hard_timeout"] or remaining_stage < 60 or current_step >= config["max_steps"] or progress["early_stopped"]:
            continue
        run_phase(stage, "train", local, store, config_path, descriptor, next_restore, current_step, remaining_stage)
    print("运行结束。Drive 中 commit.json 校验通过的 checkpoint 可恢复；本次没有自动删除其他 run 或关闭 Colab 会话。", flush=True)


if __name__ == "__main__":
    main()
