"""Render existing piano MIDIs with one pinned SoundFont and a shared audio gain."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import wave

import mido
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def validate_piano(midi):
    """不改写 MIDI；拒绝会切换到其他音色的消息。"""
    for track in midi.tracks:
        for message in track:
            if message.type == "sysex":
                raise ValueError("SysEx instrument changes are not supported by this piano preview")
            if message.type == "program_change" and message.program != 0:
                raise ValueError("Expected GM acoustic grand piano (program 0)")
            if message.type == "control_change" and message.control in (0, 32) and message.value != 0:
                raise ValueError("Expected sound bank 0")
            if message.type == "note_on" and message.velocity and message.channel == 9:
                raise ValueError("Percussion channels are not supported by this piano preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-midi", nargs="+", default=[
        "data/register_review_v4/original.mid", "data/register_review_v4/shifted.mid"])
    parser.add_argument("--out-dir", default="data/piano_review_v4")
    args = parser.parse_args()
    out = (ROOT / args.out_dir).resolve()
    if not out.is_relative_to(ROOT) or not out.relative_to(ROOT).as_posix().isascii():
        raise ValueError("Choose an ASCII output path inside the project for the native renderer")
    if out.exists():
        raise ValueError("Choose a fresh output directory; existing auditions are preserved")
    runtime = json.loads((ROOT / "data/audio_runtime/manifest.json").read_text(encoding="utf-8"))
    for asset in runtime["files"]:
        if digest(asset["path"]) != asset["sha256"]:
            raise ValueError("Runtime asset checksum mismatch: " + asset["name"])
    executable = Path(runtime["executable"])
    soundfont = next(Path(a["path"]) for a in runtime["files"] if a["name"].endswith(".sf2"))
    inputs = []
    for index, filename in enumerate(args.input_midi):
        source = (ROOT / filename).resolve()
        midi = mido.MidiFile(str(source))
        validate_piano(midi)
        name = source.stem if source.stem.isascii() else f"case_{index + 1}"
        if name in [item["name"] for item in inputs]:
            raise ValueError("Input MIDI filenames must have unique stems")
        inputs.append({"name": name, "source": source, "sha256": digest(source), "duration": midi.length})
    out.mkdir(parents=True)
    samples, reports = {}, {}
    for item in inputs:
        name = item["name"]
        copied = out / f"{name}.mid"
        copied.write_bytes(item["source"].read_bytes())
        raw = out / f"{name}_raw.f32"
        command = [str(executable), "-ni", "-R", "0", "-C", "0", "-r", str(RATE),
                   "-g", "0.2", "-o", "synth.limiter.active=0", "-o", "synth.polyphony=256",
                   "-o", "player.timing-source=sample", "-T", "raw", "-O", "float", "-E", "little",
                   "-F", raw.relative_to(ROOT).as_posix(), soundfont.relative_to(ROOT).as_posix(),
                   copied.relative_to(ROOT).as_posix()]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=120)
        log = (process.stdout + process.stderr).decode("utf-8", errors="replace")
        (out / f"{name}_render.log").write_text(log, encoding="utf-8", newline="\n")
        if process.returncode or "error" in log.lower() or "warning" in log.lower():
            raise RuntimeError(f"Review FluidSynth log for {name}: {log}")
        audio = np.fromfile(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)
        if not audio.size or not np.all(np.isfinite(audio)) or not np.any(audio):
            raise ValueError("Invalid or silent render")
        if len(audio) / RATE < item["duration"]:
            raise ValueError("Render ended before the MIDI timeline")
        samples[name] = audio
        reports[name] = {"source": item["source"].as_posix(), "source_sha256": item["sha256"],
                         "midi_duration_seconds": item["duration"], "raw_duration_seconds": len(audio) / RATE,
                         "raw_peak": float(np.max(np.abs(audio))), "command": command,
                         "raw_tail_100ms_peak": float(np.max(np.abs(audio[-RATE // 10:])))}
        print(f"Rendered {name}: {len(audio) / RATE:.2f} seconds", flush=True)
    # 两版共用同一增益与长度，不逐条归一化，不添加踏板或动态压缩。
    gain = .9 / max(report["raw_peak"] for report in reports.values())
    frames = max(len(audio) for audio in samples.values())
    hashes = {}
    for item in inputs:
        name = item["name"]
        audio = samples[name]
        pcm = np.rint(np.pad(audio, ((0, frames - len(audio)), (0, 0))) * gain * 32767).astype("<i2")
        path = out / f"{name}_piano.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
            stream.writeframes(pcm.tobytes())
        with wave.open(str(path), "rb") as stream:
            decoded = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
            if not np.array_equal(pcm, decoded) or np.max(np.abs(decoded.astype(np.int32))) >= 32767:
                raise ValueError("PCM verification failed")
        if digest(item["source"]) != item["sha256"] or digest(out / f"{name}.mid") != item["sha256"]:
            raise ValueError("MIDI changed during rendering")
        reports[name]["output_peak"] = float(np.max(np.abs(decoded.astype(np.float64)))) / 32768
        hashes[path.name] = digest(path)
        hashes[f"{name}.mid"] = item["sha256"]
    report = {"version": "sampled_piano_preview_v1", "engine": runtime["engine"],
              "soundfont": runtime["soundfont"], "preset": runtime["preset"],
              "soundfont_sha256": digest(soundfont), "sample_rate": RATE,
              "synth_gain": .2, "shared_post_gain": gain, "duration_seconds": frames / RATE,
              "reverb": False, "chorus": False, "limiter": False, "added_pedal": False,
              "input_midi_unchanged": True, "cases": reports, "artifact_sha256": hashes,
              "human_audition_status": "pending_sampled_piano_listening"}
    write_json(out / "result.json", report)
    labels = {"shifted": "伴奏升八度 · 采样钢琴", "original": "原音区 · 采样钢琴"}
    cards = []
    for item in sorted(inputs, key=lambda item: item["name"] != "shifted"):
        name = item["name"]
        title = labels.get(name, name)
        cards.append(f"<section><h2>{html.escape(title)}</h2><audio controls preload='metadata' src='{html.escape(name)}_piano.wav'></audio>"
                     f"<p><a href='{html.escape(name)}.mid'>MIDI</a> · <a href='{html.escape(name)}_piano.wav'>WAV</a></p></section>")
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>采样钢琴试听</title>
<style>body{max-width:860px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}audio{width:100%}section{border-top:1px solid #bbb;padding:20px 0}a{color:#226466}</style>
<h1>采样钢琴试听</h1><p>FluidSynth + GeneralUser GS Grand Piano。旋律与伴奏均使用采样钢琴，保留原 MIDI 的音符、节奏、力度与换和弦时刻。</p>
<p>两版使用相同增益，关闭额外混响与合唱，不添加踏板。</p>""" + "".join(cards) + "</html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"page": (out / "index.html").as_posix(), "shared_gain": gain,
                      "peaks": {name: value["output_peak"] for name, value in reports.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
