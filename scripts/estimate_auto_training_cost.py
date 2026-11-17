"""计算明确的云端费用上限；将预算配额与实测训练时长分开。"""

import argparse
import json
import math
from pathlib import Path


def estimate(config, gpu=None, hours=None):
    name = gpu or config["recommended_gpu"]
    rate = config["gpu_options"][name]["usd_per_gpu_hour"]
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("Invalid GPU hourly rate")
    total_hours = sum(x["gpu_hours_cap"] for x in config["phases"]) if hours is None else hours
    if total_hours < 0 or not math.isfinite(total_hours):
        raise ValueError("Hours must be finite and nonnegative")
    disk = config["storage"]
    persistent = disk["network_volume_gb"] * disk["network_usd_per_gb_month"] * disk["retention_months"]
    scratch = disk["container_disk_gb"] * disk["container_usd_per_gb_month"] * total_hours / disk["hours_per_billing_month_assumption"]
    compute = total_hours * rate
    subtotal = compute + persistent + scratch
    reserve = subtotal * config["contingency_fraction"]
    return {
        "gpu": name, "gpu_hours_cap": total_hours, "usd_per_gpu_hour": rate,
        "compute_usd": round(compute, 2), "persistent_storage_usd": round(persistent, 2),
        "container_storage_usd": round(scratch, 2), "contingency_usd": round(reserve, 2),
        "total_budget_usd": round(subtotal + reserve, 2),
        "total_budget_cny_at_assumed_rate": math.ceil((subtotal + reserve) * config["cny_per_usd_budget_assumption"]),
        "is_runtime_prediction": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/auto_accompaniment/cloud_budget.json")
    parser.add_argument("--output", default="data/auto_accompaniment/budget.json")
    parser.add_argument("--gpu")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--seconds-per-step-low", type=float)
    parser.add_argument("--seconds-per-step-high", type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    first_hours = sum(x["gpu_hours_cap"] for x in config["phases"] if x["id"] in config["first_pass_phase_ids"])
    report = {"quote": config, "first_pass": estimate(config, args.gpu, first_hours),
              "full_campaign": estimate(config, args.gpu, args.hours),
              "alternatives_same_hour_allocation": [estimate(config, name, args.hours) for name in config["gpu_options"]],
              "caution": "Equal GPU hours are not equal training work. Compare actual cloud seconds per optimizer step, batch, length and validation overhead before choosing hardware."}
    values = (args.optimizer_steps, args.seconds_per_step_low, args.seconds_per_step_high)
    if any(v is not None for v in values):
        if any(v is None for v in values) or args.optimizer_steps <= 0 or not (0 < args.seconds_per_step_low <= args.seconds_per_step_high):
            parser.error("Supply positive optimizer steps and an ordered seconds/step range together")
        report["user_supplied_throughput_scenario"] = {
            "optimizer_steps": args.optimizer_steps,
            "training_hours_range": [args.optimizer_steps * s / 3600 for s in values[1:]],
            "excludes": "Validation, data preparation, setup, checkpointing and generation unless included in the supplied rate",
            "measurement_provenance": "Caller supplied; not independently verified by this estimator",
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"first_pass": report["first_pass"], "full_campaign": report["full_campaign"], "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
