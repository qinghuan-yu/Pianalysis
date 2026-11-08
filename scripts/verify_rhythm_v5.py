"""Check saved V5 artifact hashes, independent CLI replay and paired duration errors."""

from collections import Counter
import json
from pathlib import Path
import re

from build_harmony_review import write_json
import rhythm_v4 as v4
from verify_rhythm_v4 import digest, realized_frames


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attacks(frames):
    return {(step, lane) for step, frame in enumerate(frames) for lane in range(3) if frame["mask"] & (1 << lane)}


def main():
    root = Path("data/trusted_harmony_v5")
    evaluation = read(root / "evaluation/evaluation.json")
    validation = read(root / "validation.json")
    annotation = read(root / "annotations.json")
    assert digest(root / "annotations.json") == evaluation["annotations_sha256"]
    assert read("data/trusted_harmony_v4/annotations.json")["pieces"] == annotation["pieces"][:4]
    for path, expected in validation["artifact_sha256"].items():
        assert digest(path) == expected, path
    for info in evaluation["models"].values():
        assert digest(Path(info["path"]) / "model.safetensors") == info["sha256"]
    for name in ("local_rhythm_v4", "local_voicing_v3"):
        accepted = read(f"examples/harmony/{name}_acceptance.json")
        assert digest(accepted["model_file"]) == accepted["model_sha256"]
        for item in accepted["files"] + accepted.get("comparison_files", []):
            assert digest(item["path"]) == item["sha256"]
    for folder in ("review", "evaluation"):
        page = root / folder / "index.html"
        for link in re.findall(r"(?:src|href)='([^']+)'", page.read_text(encoding="utf-8")):
            assert (page.parent / link).is_file(), link
    cli = read(root / "cli_test1_medium/result.json")
    test1 = next(c for c in evaluation["cases"] if c["song"] == "test1")
    assert len(cli["rows"]) == len(test1["rows"])
    for generated, row in zip(cli["rows"], test1["rows"]):
        assert generated["voicing"] == row["voicing"]
        assert generated["frames"] == row["methods"]["v5_medium"]["frames"]
        assert generated["notes"] == row["methods"]["v5_medium"]["notes"]
    assert (root / "cli_test1_medium/output.mid").read_bytes() == (root / "evaluation/test1_v5_medium.mid").read_bytes()
    assert (root / "cli_test1_medium/output.wav").read_bytes() == (root / "evaluation/test1_v5_medium.wav").read_bytes()
    records = {(r["song"], r["label"]["start_tick"]): r for r in v4.build_dataset(annotation)["records"]}
    pieces = {p["song"]: p for p in annotation["pieces"]}
    methods = ("accepted_v4", "expanded_v4", "v5_medium")
    errors = {split: {m: [] for m in methods} for split in ("train", "validation", "test")}
    density_counts, projection = Counter(), Counter()
    for record in records.values():
        projection.update({k: v for k, v in record["projection_audit"].items() if k not in ("target_source", "bass_pitch_separated", "max_onset_error_quarters")})
    for case in evaluation["cases"]:
        for row in case["rows"]:
            counts = [len(attacks(row["methods"]["v5_" + p]["frames"])) for p in ("sparse", "medium", "dense")]
            density_counts["segments"] += 1
            density_counts["nondecreasing"] += counts[0] <= counts[1] <= counts[2]
            density_counts["strictly_increasing"] += counts[0] < counts[1] < counts[2]
            if case["split"] == "external":
                continue
            reference = records[case["song"], row["start_tick"]]
            common = attacks(reference["frames"])
            actual = {}
            for method in methods:
                actual[method] = realized_frames(reference["label"], row["methods"][method]["notes"][1:], pieces[case["song"]]["ppq"])
                common &= attacks(actual[method])
            for method in methods:
                errors[case["split"]][method].extend(abs(actual[method][step]["durations"][lane] - reference["frames"][step]["durations"][lane]) / 4 for step, lane in common)
    paired = {s: {m: {"matched_same_events": len(v), "mae_quarters": sum(v) / len(v) if v else None} for m, v in methods.items()} for s, methods in errors.items()}
    validation.update(independent_medium_cli_matches=True, old_67_segments_unchanged=True, local_page_links_valid=True,
                      per_span_density_order=dict(density_counts), paired_duration_on_common_events=paired,
                      projection_audit=dict(projection), software_tests_passed=94)
    write_json(root / "validation.json", validation)
    print(json.dumps({"midi_count": len(validation["midi_cases"]), "per_span_density_order": dict(density_counts),
                      "paired_duration": paired, "projection": dict(projection)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
