"""生成私有云训练包和逐文件校验清单；不上传、不创建云资源。"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = (
    "auto_harmony.py", "auto_texture.py", "colab_runtime.py", "colab_train.py",
    "estimate_auto_training_cost.py", "fetch_auto_accompaniment_data.py", "fetch_auto_models.py",
    "package_auto_training.py", "prepare_auto_accompaniment_data.py", "run_auto_training.py",
    "train_auto_harmony.py", "train_auto_texture.py", "verify_auto_training_data.py",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def split_archive(path, part_bytes):
    """分片用于受大小限制的上传；原始训练包仍保留。"""
    parts = []
    with path.open("rb") as incoming:
        while incoming.tell() < path.stat().st_size:
            target = path.with_name(path.name + f".part{len(parts) + 1:03d}")
            size, checksum = 0, hashlib.sha256()
            with target.open("xb") as outgoing:
                while size < part_bytes:
                    block = incoming.read(min(8 * 1024 * 1024, part_bytes - size))
                    if not block:
                        break
                    outgoing.write(block)
                    checksum.update(block)
                    size += len(block)
            if digest(target) != checksum.hexdigest():
                raise ValueError("Part checksum readback failed: " + str(target))
            parts.append({"name": target.name, "bytes": size, "sha256": checksum.hexdigest()})
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/auto_accompaniment/cloud_training_bundle.tar.gz")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--split-mib", type=int, default=96, help="Upload part size in MiB; 0 disables splitting")
    args = parser.parse_args()
    if args.split_mib < 0 or args.split_mib > 480:
        parser.error("--split-mib must be between 0 and 480")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("Bundle output must stay within project data/")
    index = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or list(output.parent.glob(output.name + ".part*")):
        raise ValueError("Choose a new archive output; existing bundle and manifest are preserved")
    processed = ROOT / "data/auto_accompaniment/processed"
    data_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    check = json.loads((ROOT / "data/auto_accompaniment/verification/validation.json").read_text(encoding="utf-8"))
    samples_hash = digest(processed / "samples.jsonl")
    if data_manifest.get("partial_build") is not False or not check["passed"]:
        raise ValueError("A complete dataset build and passing independent gate are required")
    if samples_hash != data_manifest["samples_sha256"] or samples_hash != check["samples_sha256"]:
        raise ValueError("Data changed since validation; rerun the gate")
    directories = ["colab", "examples/auto_accompaniment", "data/auto_accompaniment/raw", "data/auto_accompaniment/processed", "data/auto_accompaniment/upstream", "data/auto_accompaniment/verification"]
    paths = [ROOT / n for n in ("README.md", "LICENSE", "requirements-auto-accompaniment.txt")]
    paths.extend(ROOT / "scripts" / name for name in TRAINING_SCRIPTS)
    paths.extend(ROOT / "data/auto_accompaniment" / n for n in ("download_manifest.json", "budget.json"))
    for directory in directories:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in ("__pycache__", ".git", "tests", "test") for part in rel.parts) or path.name.startswith("test_") or path.suffix == ".pyc":
                continue
            if directory == "data/auto_accompaniment/verification" and path.name not in {"validation.json", "index.html", "harmony_contract_checks.json", "run_report.json"}:
                # 短步检查的 optimizer/checkpoint 不是正式训练输入，不能打进迁移包。
                continue
            paths.append(path)
    records = []
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Refusing linked or escaping bundle input: " + str(path))
        records.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"schema_version": 1, "purpose": "Private training transfer; source-data licenses still apply; not a public dataset release", "samples_sha256": samples_hash, "files": records, "total_uncompressed_bytes": sum(r["bytes"] for r in records)}
    output.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not args.manifest_only:
        with tarfile.open(output, "w:gz", compresslevel=3) as archive:
            for record in records:
                archive.add(ROOT / record["path"], arcname=record["path"], recursive=False)
            archive.add(index, arcname="BUNDLE_MANIFEST.json", recursive=False)
        manifest["archive_sha256"] = digest(output)
        manifest["archive_bytes"] = output.stat().st_size
        if args.split_mib:
            manifest["archive_parts"] = split_archive(output, args.split_mib * 1024 * 1024)
        index.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"files": len(records), "uncompressed_bytes": manifest["total_uncompressed_bytes"], "manifest": str(index), "archive": str(output) if not args.manifest_only else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
