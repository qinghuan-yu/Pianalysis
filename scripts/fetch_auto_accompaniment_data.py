"""Fetch immutable upstream datasets, verify hashes, and safely unpack POP909-CL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


PROJECT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify(path: Path, source: dict) -> dict:
    payload = path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if not (source.get("sha256") or source.get("git_blob_sha1")):
        raise ValueError(f"No pinned checksum for {source['id']}")
    if source.get("sha256") and sha256 != source["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path}")
    if source.get("git_blob_sha1") and blob != source["git_blob_sha1"]:
        raise ValueError(f"Git blob mismatch for {path}")
    return {**source, "local_path": path.relative_to(PROJECT).as_posix(),
            "actual_sha256": sha256, "actual_git_blob_sha1": blob,
            "size_bytes": len(payload)}


def safe_extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink rejected: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            count += target.suffix.lower() == ".mid"
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "examples/auto_accompaniment/data_sources.json")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = (args.root or PROJECT / config["root"]).resolve()
    if not root.is_relative_to(PROJECT):
        raise ValueError("Dataset root must remain inside the project")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "sources": [], "audio_downloaded": False}
    for source in config["sources"]:
        path = raw / source["filename"]
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(source["url"], headers={"User-Agent": "Pianalysis-dataset-preparation/1"})
            print(f"Downloading {source['id']} from pinned commit...", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            verify(temporary, source)
            temporary.replace(path)
        entry = verify(path, source)
        if source["id"] == "pop909_cl":
            count = safe_extract(path, raw / "pop909_cl")
            if count != source["expected_midi_files"]:
                raise ValueError(f"Expected {source['expected_midi_files']} MIDI files, got {count}")
            entry["extracted_midi_count"] = count
        manifest["sources"].append(entry)
        print(f"Verified {source['id']}: {entry['size_bytes']} bytes, SHA256 {entry['actual_sha256']}", flush=True)
    write_json(root / "download_manifest.json", manifest)


if __name__ == "__main__":
    main()
