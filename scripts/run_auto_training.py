"""在当前机器运行单个训练阶段，核验数据并限制整个子进程时长。"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from verify_auto_training_data import audit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("harmony", "texture"))
    parser.add_argument("--samples", default="data/auto_accompaniment/processed/samples.jsonl")
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("A positive finite process wall-time cap is required")
    start = time.monotonic()
    cap = args.max_hours * 3600
    samples = (ROOT / args.samples).resolve()
    manifest = json.loads(samples.with_name("manifest.json").read_text(encoding="utf-8"))
    report = audit(samples)
    if not report["passed"] or manifest.get("partial_build") is not False or manifest.get("samples_sha256") != report["samples_sha256"]:
        raise ValueError("Dataset gate failed or build/hash differs; training was not started")
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Choose a fresh run directory, or resume explicitly")
    config = args.config or f"examples/auto_accompaniment/{args.stage}_train.json"
    remaining = cap - (time.monotonic() - start)
    if remaining <= 5:
        raise ValueError("Budget exhausted during data verification")
    # 子训练器先软停止；硬停止只影响本进程，不会关闭或删除云机器。
    graceful_budget = max(1, remaining - min(120, remaining * 0.15))
    command = [sys.executable, str(ROOT / f"scripts/train_auto_{args.stage}.py"), "--samples", str(samples),
               "--config", str(ROOT / config), "--output-dir", str(output), "--device", args.device,
               "--max-wall-seconds", str(graceful_budget)]
    if args.stage == "harmony":
        command.append("--run")
    if args.max_steps is not None:
        command += ["--max-steps", str(args.max_steps)]
    if args.resume:
        command += ["--resume", str(ROOT / args.resume)]
    runlogs = ROOT / "data/auto_accompaniment/job_logs"
    runlogs.mkdir(parents=True, exist_ok=True)
    tag = f"{args.stage}_{time.time_ns()}"
    log_path = runlogs / (tag + ".log")
    print(json.dumps({"command": command, "process_cap_hours": args.max_hours, "log": str(log_path)}, ensure_ascii=False), flush=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            returncode = process.wait(timeout=max(1, cap - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=30)
            returncode = 124
    record = {"command": command, "samples_sha256": report["samples_sha256"], "max_hours": args.max_hours,
              "wall_seconds": time.monotonic() - start, "exit_code": returncode, "hard_timeout": timed_out,
              "log": log_path.relative_to(ROOT).as_posix(),
              "billing_note": "This stops only the training process. A rented GPU instance continues billing until explicitly stopped/terminated by its owner.",
              "checkpoint_note": "Soft stop saves last; hard timeout retains the most recent complete checkpoint, not necessarily the latest step."}
    (runlogs / (tag + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
