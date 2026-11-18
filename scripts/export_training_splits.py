"""Export only identifiers from the frozen, accepted samples (no note data)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


SAMPLES_SHA256 = "bf3e8705a2b977eccb32d94e54d6130d4c3e0168c73cf35d86c4965658299d60"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("data/auto_accompaniment/processed/samples.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256()
    records = []
    with args.samples.open("rb") as source:
        for line in source:
            digest.update(line)
            row = json.loads(line.decode("utf-8"))
            records.append([row[field] for field in ("song_id", "work_id", "source", "split")])
    if digest.hexdigest() != SAMPLES_SHA256:
        raise ValueError("This release describes a different sample file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["song_id", "work_id", "source", "split"])
        writer.writerows(records)


if __name__ == "__main__":
    main()
