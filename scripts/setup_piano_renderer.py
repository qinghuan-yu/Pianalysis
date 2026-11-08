"""Fetch a pinned portable FluidSynth and SoundFont into project-local ignored data."""

import hashlib
import json
from pathlib import Path
import stat
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "data/audio_runtime"
ASSETS = [
    {"name": "fluidsynth-v2.6.2-win10-x64-cpp11.zip", "size": 2729665,
     "url": "https://github.com/FluidSynth/fluidsynth/releases/download/v2.6.2/fluidsynth-v2.6.2-win10-x64-cpp11.zip",
     "fallback_url": "https://api.github.com/repos/FluidSynth/fluidsynth/releases/assets/698253322",
     "sha256": "808295345d43c9c428f3fa87d29fbf97a9926f356df95559ff2689f3bd4072d7"},
    {"name": "GeneralUser-GS.sf2", "size": 32319396,
     "url": "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/684543d5e5efaef08d02be50dcda8d552478fa60/GeneralUser-GS.sf2",
     "git_blob_sha": "298b552d2e9d1307e03e5c5c99d2c046aaed9ec3"},
    {"name": "GeneralUser-GS-LICENSE.txt",
     "url": "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt",
     "git_blob_sha": "f8fc9c58d7574f2b0ce6cea86ff49192540bcc33"},
]


def verify(content, asset):
    if "size" in asset and len(content) != asset["size"]:
        raise ValueError("Unexpected download size: " + asset["name"])
    sha256 = hashlib.sha256(content).hexdigest()
    if "sha256" in asset and sha256 != asset["sha256"]:
        raise ValueError("Release checksum mismatch: " + asset["name"])
    blob = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
    if "git_blob_sha" in asset and blob != asset["git_blob_sha"]:
        raise ValueError("Pinned repository blob mismatch: " + asset["name"])
    return sha256


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for asset in ASSETS:
        path = ROOT / asset["name"]
        if path.exists():
            content = path.read_bytes()
        else:
            urls = [asset["url"]]
            if "fallback_url" in asset:
                urls.append(asset["fallback_url"])
            for index, url in enumerate(urls):
                print(f"Downloading {asset['name']} (source {index + 1})", flush=True)
                request = urllib.request.Request(url, headers={
                    "User-Agent": "Pianalysis-piano-preview/1", "Accept": "application/octet-stream"})
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        content = response.read(40_000_001)
                    verify(content, asset)
                    break
                except (urllib.error.URLError, TimeoutError, ValueError) as error:
                    if index == len(urls) - 1:
                        raise
                    print(f"Primary download failed: {error}; trying official API", flush=True)
            path.write_bytes(content)
        checksum = verify(content, asset)
        records.append({**asset, "sha256": checksum, "path": path.as_posix()})
        print(f"Verified {asset['name']}: {len(content)} bytes", flush=True)
    destination = (ROOT / "fluidsynth-2.6.2").resolve()
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / ASSETS[0]["name"]) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive entry escapes portable runtime directory")
        archive.extractall(destination)
    executables = list(destination.rglob("fluidsynth.exe"))
    if len(executables) != 1:
        raise ValueError("Expected one official FluidSynth executable")
    manifest = {"engine": "FluidSynth 2.6.2", "soundfont": "GeneralUser GS 2.0.3", "preset": "bank 0 / program 0 Grand Piano",
                "executable": executables[0].as_posix(), "files": records,
                "installation_scope": "project_local_portable_no_system_changes"}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"executable": manifest["executable"], "soundfont": manifest["soundfont"]}), flush=True)


if __name__ == "__main__":
    main()
