"""获取固定版本的两套上游权重与最小源码，逐文件核验后才供训练加载。"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise ValueError("Pinned asset mismatch: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/auto_accompaniment/model_sources.json")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    total = 0
    for model in config["models"]:
        destination = (ROOT / model["directory"]).resolve()
        if not destination.is_relative_to(ROOT / "data/auto_accompaniment/upstream"):
            raise ValueError("Model assets must stay inside the ignored upstream directory")
        for item in model["manifest"]["files"]:
            relative = item.get("path", item.get("relpath"))
            path = (destination / relative).resolve()
            if not path.is_relative_to(destination) or path.is_symlink():
                raise ValueError("Unsafe asset path")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                # Git 在 Windows 上可能转换行尾。只有还原后完全匹配上游哈希才接受。
                if path.stat().st_size < 2_000_000 and sha256(path) != item["sha256"]:
                    normalized = path.read_bytes().replace(b"\r\n", b"\n")
                    if len(normalized) == item["bytes"] and hashlib.sha256(normalized).hexdigest() == item["sha256"]:
                        path.write_bytes(normalized)
                verify(path, item)
            else:
                if args.offline:
                    raise FileNotFoundError(path)
                temporary = path.with_suffix(path.suffix + ".part")
                request = urllib.request.Request(item["url"], headers={"User-Agent": "Pianalysis-model-preparation/1"})
                print("Downloading " + model["id"] + "/" + relative, flush=True)
                received = 0
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        received += len(chunk)
                        if received > item["bytes"]:
                            raise ValueError("Download exceeds pinned size")
                        stream.write(chunk)
                verify(temporary, item)
                temporary.replace(path)
            total += item["bytes"]
        manifest_path = destination / model["manifest_filename"]
        manifest_path.write_text(json.dumps(model["manifest"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print("Verified " + model["id"], flush=True)
    print(json.dumps({"verified_bytes": total, "models": len(config["models"])}))


if __name__ == "__main__":
    main()
