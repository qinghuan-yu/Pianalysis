"""Colab 标准库工具：安全解包、隔离环境及有提交记录的 Drive 存储。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid

CREATOR = "pianalysis-colab-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        # FUSE 上 fsync/rename 不构成事务；恢复仍须检查内容和提交记录。
        try:
            os.fsync(stream.fileno())
        except OSError:
            pass


def commit_record(path, value):
    record = dict(value, creator=CREATOR)
    record["record_sha256"] = json_hash(record)
    write_json(path, record)
    if read_record(path) != record:
        raise OSError("Commit marker readback failed")
    return record


def read_record(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = value.get("record_sha256")
    body = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("creator") != CREATOR or expected != json_hash(body):
        raise ValueError("Invalid or incomplete commit marker")
    return value


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe archive/artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Unsafe archive/artifact path: " + value)
    return path


def copy_verified(source, destination, expected_hash=None, expected_size=None):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Linked storage files are not supported")
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher, count = hashlib.sha256(), 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            count += len(block)
            if expected_size is not None and count > expected_size:
                raise ValueError("Copy exceeds declared size")
            hasher.update(block)
            outgoing.write(block)
        outgoing.flush()
        try:
            os.fsync(outgoing.fileno())
        except OSError:
            pass
    actual = hasher.hexdigest()
    if (expected_hash is not None and actual != expected_hash) or (expected_size is not None and count != expected_size):
        raise ValueError("Copy source checksum/size mismatch")
    if digest(destination) != actual:
        raise OSError("Destination checksum readback failed")
    return {"sha256": actual, "bytes": count}


def safe_extract_bundle(archive_path, manifest_path, destination):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("archive_sha256"):
        raise ValueError("Use the external archive manifest, not BUNDLE_MANIFEST.json")
    if archive_path.stat().st_size != manifest["archive_bytes"] or digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Archive checksum/size mismatch")
    expected = {}
    for record in manifest["files"]:
        relative = safe_relative(record["path"]).as_posix()
        if relative in expected or record["bytes"] < 0:
            raise ValueError("Duplicate or invalid file manifest")
        expected[relative] = record
    total = sum(record["bytes"] for record in expected.values())
    if total != manifest["total_uncompressed_bytes"] or total > 32 * 1024**3:
        raise ValueError("Archive expansion bound exceeded")
    if destination.exists():
        receipt = destination / ".colab_bundle_receipt.json"
        if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")).get("archive_sha256") != manifest["archive_sha256"]:
            raise ValueError("Existing workspace belongs to another bundle; preserve it and use a fresh Colab runtime")
        for relative, record in expected.items():
            path = destination / relative
            if path.is_symlink() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Existing extracted file changed: " + relative)
        return manifest
    staging = destination.parent / (destination.name + ".extract-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    seen, internal = set(), None
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative(member.name).as_posix()
            if not member.isfile() or relative in seen:
                raise ValueError("Tar links, devices, directories or duplicates are rejected")
            seen.add(relative)
            if relative == "BUNDLE_MANIFEST.json":
                if member.size > 16 * 1024**2:
                    raise ValueError("Oversized internal manifest")
                internal = json.load(archive.extractfile(member))
                continue
            record = expected.get(relative)
            if record is None or member.size != record["bytes"]:
                raise ValueError("Unexpected tar member or size: " + relative)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            hasher, count = hashlib.sha256(), 0
            with archive.extractfile(member) as incoming, target.open("xb") as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    count += len(block)
                    if count > record["bytes"]:
                        raise ValueError("Tar member exceeds manifest size")
                    hasher.update(block)
                    outgoing.write(block)
            if count != record["bytes"] or hasher.hexdigest() != record["sha256"]:
                raise ValueError("Extracted content checksum mismatch: " + relative)
    if seen != set(expected) | {"BUNDLE_MANIFEST.json"} or internal is None or internal.get("files") != manifest["files"] or internal.get("samples_sha256") != manifest.get("samples_sha256"):
        raise ValueError("Internal and external bundle manifests differ")
    write_json(staging / ".colab_bundle_receipt.json", manifest)
    # 此处只对 /content 的新目录使用本地 rename，不用于 Drive 提交协议。
    staging.replace(destination)
    return manifest


def prepare_environment(project, drive_root, content_root="/content"):
    project, drive_root, content_root = Path(project), Path(drive_root), Path(content_root)
    session = drive_root / "runs/_environment" / str(time.time_ns())
    session.mkdir(parents=True)
    bootstrap = content_root / "pianalysis_uv_bootstrap"
    environment = content_root / "pianalysis-venv"
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(content_root / "pianalysis_python"),
               UV_CACHE_DIR=str(content_root / "pianalysis_uv_cache"), PYTHONPATH=str(bootstrap))
    uv = [sys.executable, "-m", "uv"]
    python = environment / "bin/python"
    commands = [[sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target", str(bootstrap), "uv==0.8.22"],
                uv + ["python", "install", "3.11"],
                uv + ["venv", "--python", "3.11", "--managed-python", "--allow-existing", str(environment)],
                uv + ["pip", "install", "--python", str(python), "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"]]
    requirements = content_root / "pianalysis_requirements.runtime.txt"
    lines = (project / "requirements-auto-accompaniment.txt").read_text(encoding="utf-8").splitlines()
    requirements.write_text("\n".join(line for line in lines if not line.strip().startswith("torch==")) + "\n", encoding="utf-8", newline="\n")
    constraints = content_root / "pianalysis_constraints.runtime.txt"
    constraints.write_text("torch==2.5.1+cu121\n", encoding="utf-8", newline="\n")
    commands.append(uv + ["pip", "install", "--python", str(python), "--constraint", str(constraints), "-r", str(requirements)])
    with (session / "bootstrap.log").open("a", encoding="utf-8", newline="\n") as log:
        for command in commands:
            print("安装独立环境：", " ".join(command), flush=True)
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (session / "packages.freeze.txt").open("w", encoding="utf-8", newline="\n") as log:
        subprocess.run(uv + ["pip", "freeze", "--python", str(python)], env=env, stdout=log, check=True)
    metadata = {"python": str(python), "uv_version": "0.8.22", "requirements_sha256": digest(project / "requirements-auto-accompaniment.txt"),
                "system_python_unchanged": True, "cache_policy": "uv/Python/venv caches live only under /content and are rebuilt; logs and environment metadata persist on Drive"}
    commit_record(session / "environment.json", metadata)
    return str(python)


class DriveStore:
    """Checkpoint 永不覆盖写；完整 payload 回读后才建立带自身 hash 的提交记录。"""
    def __init__(self, root, run_id, contract, config, keep_all=False):
        if not re.fullmatch(r"run_[0-9a-f]+", run_id):
            raise ValueError("Invalid run id")
        self.root = Path(root).resolve()
        self.run_id, self.contract, self.config = run_id, contract, config
        self.keep_all = keep_all
        self.generations = self.root / "checkpoints"
        self.generations.mkdir(parents=True, exist_ok=True)
        self.artifact_lock = threading.Lock()
        self.last_artifact_hash = None

    def save_checkpoint(self, source, kind, step):
        if kind not in {"last", "best"}:
            raise ValueError("Unknown checkpoint kind")
        source = Path(source)
        checksum, size = digest(source), source.stat().st_size
        for folder, prior in self.records():
            if prior["kind"] == kind and prior["step"] == step and prior["checkpoint"]["sha256"] == checksum and digest(folder / "checkpoint.pt") == checksum:
                return folder, prior
        name = "g_" + str(time.time_ns()) + "_" + uuid.uuid4().hex
        target = self.generations / name
        target.mkdir()
        copy_verified(source, target / "checkpoint.pt", checksum, size)
        record = commit_record(target / "commit.json", {"schema_version": 1, "run_id": self.run_id, "kind": kind,
                    "step": int(step), "created_ns": time.time_ns(), "checkpoint": {"path": "checkpoint.pt", "bytes": size, "sha256": checksum},
                    "contract": self.contract, "config": self.config})
        self.prune()
        return target, record

    def records(self, verify_payload=False):
        result = []
        for marker in self.generations.glob("g_*/commit.json"):
            try:
                record = read_record(marker)
                if record["run_id"] != self.run_id or record["contract"] != self.contract or record["config"] != self.config or record["kind"] not in {"last", "best"}:
                    continue
                checkpoint = marker.parent / record["checkpoint"]["path"]
                if record["checkpoint"]["path"] != "checkpoint.pt" or checkpoint.is_symlink() or checkpoint.stat().st_size != record["checkpoint"]["bytes"]:
                    continue
                if verify_payload and digest(checkpoint) != record["checkpoint"]["sha256"]:
                    continue
                result.append((marker.parent, record))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda item: (item[1]["step"], item[1]["created_ns"]), reverse=True)

    def latest(self):
        for folder, record in self.records():
            if digest(folder / "checkpoint.pt") == record["checkpoint"]["sha256"]:
                return folder, record
        return None

    def restore(self, destination):
        found = self.latest()
        if found is None:
            raise ValueError("No complete, checksum-valid checkpoint generation")
        folder, record = found
        copy_verified(folder / "checkpoint.pt", destination, record["checkpoint"]["sha256"], record["checkpoint"]["bytes"])
        return record

    def prune(self):
        if self.keep_all:
            return
        kept, counts = set(), {"last": 0, "best": 0}
        records = self.records()
        for folder, record in self.records(verify_payload=True):
            kind = record["kind"]
            if kind in counts and counts[kind] < (2 if kind == "last" else 1):
                kept.add(folder)
                counts[kind] += 1
        for folder, record in records:
            if folder in kept:
                continue
            if counts[record["kind"]] < (2 if record["kind"] == "last" else 1):
                continue
            # 仅删除本 run 自己生成、具有有效 marker 的已知两文件目录；不递归清理 Drive。
            resolved = folder.resolve()
            if resolved.parent != self.generations.resolve() or folder.is_symlink() or not re.fullmatch(r"g_[0-9]+_[0-9a-f]{32}", folder.name):
                raise ValueError("Unsafe pruning path")
            children = list(folder.iterdir())
            if {child.name for child in children} != {"checkpoint.pt", "commit.json"} or any(child.is_symlink() for child in children):
                continue
            for filename in ("checkpoint.pt", "commit.json"):
                (folder / filename).unlink()
            folder.rmdir()

    def sync_artifacts(self, local_root):
        with self.artifact_lock:
            local_root = Path(local_root)
            objects = self.root / "artifacts/objects"
            objects.mkdir(parents=True, exist_ok=True)
            entries = []
            for path in sorted(local_root.rglob("*")):
                if not path.is_file() or path.is_symlink() or path.suffix in {".pt", ".tmp", ".partial", ".pyc"}:
                    continue
                before = path.stat()
                checksum = digest(path)
                if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                    continue
                target = objects / checksum
                if not target.exists():
                    try:
                        copy_verified(path, target, checksum, before.st_size)
                    except ValueError:
                        # 日志在复制时继续追加；未提交对象不参与恢复，下轮重试。
                        if target.exists() and not target.is_symlink() and target.parent.resolve() == objects.resolve():
                            target.unlink()
                        continue
                if target.stat().st_size != before.st_size or digest(target) != checksum:
                    raise OSError("Corrupt artifact object")
                entries.append({"path": path.relative_to(local_root).as_posix(), "sha256": checksum, "bytes": before.st_size})
            fingerprint = json_hash(entries)
            if fingerprint == self.last_artifact_hash:
                return
            commit_record(self.root / "artifacts/snapshots" / (str(time.time_ns()) + ".json"),
                          {"run_id": self.run_id, "created_ns": time.time_ns(), "files": entries})
            # latest 仅供人查看，断线时可能不完整；恢复从不依赖这些覆盖写的副本。
            for item in entries:
                latest = self.root / "latest" / item["path"]
                if latest.exists() and digest(latest) == item["sha256"]:
                    continue
                latest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(objects / item["sha256"], latest)
            self.last_artifact_hash = fingerprint

    def restore_artifacts(self, local_root):
        local_root = Path(local_root)
        for path in sorted((self.root / "artifacts/snapshots").glob("*.json"), reverse=True):
            try:
                record = read_record(path)
                if record["run_id"] != self.run_id:
                    continue
                for item in record["files"]:
                    safe_relative(item["path"])
                    source = self.root / "artifacts/objects" / item["sha256"]
                    if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or digest(source) != item["sha256"]:
                        raise ValueError("Artifact checksum mismatch")
                for item in record["files"]:
                    target = local_root / item["path"]
                    if not target.exists():
                        copy_verified(self.root / "artifacts/objects" / item["sha256"], target, item["sha256"], item["bytes"])
                return
            except (OSError, ValueError, KeyError, TypeError):
                continue
