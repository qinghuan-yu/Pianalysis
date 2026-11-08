"""Audition one newly authored melody using the frozen accepted V3/V4 checkpoints."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import torch
from transformers import GPT2LMHeadModel

import harmony_plan as hp
import local_voicing as lv
import rhythm_v4 as rv
from train_local_voicing import predict as predict_voicing
from train_rhythm_v4 import predict as predict_rhythm

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def event_key(note):
    return tuple(note[k] for k in ("pitch", "start_tick", "end_tick", "velocity"))


def read_role(path, role):
    midi = hp.pretty_midi.PrettyMIDI(str(path))
    return sorted((n.pitch, int(midi.time_to_tick(n.start)), int(midi.time_to_tick(n.end)), n.velocity)
                  for instrument in midi.instruments if instrument.name == role for n in instrument.notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", default="examples/harmony/original_melody_v4_probe.json")
    parser.add_argument("--out-dir", default="data/original_melody_v4")
    args = parser.parse_args()
    score_path = (ROOT / args.score).resolve()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    out = (ROOT / args.out_dir).resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        raise ValueError("Choose a fresh project output directory")
    if score["meter"] != [4, 4] or score["version"] != "authored_melody_probe_v1":
        raise ValueError("This audition expects an authored four-quarter bar score")
    acceptance = json.loads((ROOT / "examples/harmony/local_rhythm_v4_acceptance.json").read_text(encoding="utf-8"))
    model_files = {"rhythm": ROOT / acceptance["model_file"], "voicing": ROOT / acceptance["frozen_voicing_file"]}
    expected = {"rhythm": acceptance["model_sha256"], "voicing": acceptance["frozen_voicing_sha256"]}
    if any(digest(path) != expected[name] for name, path in model_files.items()):
        raise ValueError("Checkpoint differs from the accepted V4 pair")
    score_hash = digest(score_path)
    ppq, melody, labels = score["ppq"], [], []
    for index, bar in enumerate(score["bars"]):
        if sum(event[1] for event in bar["events"]) != 4:
            raise ValueError(f"Bar {index + 1} does not contain four quarters")
        tick = index * 4 * ppq
        labels.append({**hp.parse_symbol(bar["chord"]), "start_tick": tick, "end_tick": tick + 4 * ppq,
                       "source": "assistant_authored_composition", "review_status": "authored_not_user_auditioned"})
        for pitch, quarters, velocity in bar["events"]:
            length = quarters * ppq
            if length <= 0 or length != int(length):
                raise ValueError("Event duration must have an exact positive tick length")
            if pitch is not None:
                melody.append({"pitch": hp.pretty_midi.note_name_to_number(pitch), "start_tick": tick,
                               "end_tick": tick + int(length), "velocity": velocity, "role": "melody"})
            tick += int(length)
    end = len(score["bars"]) * 4 * ppq
    plan = {"version": "harmony_plan_v2", "ppq": ppq, "duration_ticks": end, "labels": labels,
            "key_context": score["key"] + ":maj", "evidence_scope": "assistant_authored_not_inferred_from_midi",
            "review_status": "authored_not_user_auditioned"}
    hp.validate_plan(plan)
    out.mkdir(parents=True)
    (out / "score.json").write_bytes(score_path.read_bytes())
    source = hp.pretty_midi.PrettyMIDI(resolution=ppq, initial_tempo=score["tempo_bpm"])
    source.time_signature_changes = [hp.pretty_midi.TimeSignature(4, 4, 0)]
    source.key_signature_changes = [hp.pretty_midi.KeySignature(hp.pretty_midi.key_name_to_key_number(score["key"]), 0)]
    # 旋律与和声先落盘，再推理；不根据模型结果挑旋律或更改和弦。
    source = lv.write_midi(source, melody, [], out / "melody.mid", end=end)
    plan["melody_sha256"] = digest(out / "melody.mid")
    write_json(out / "harmony_plan.json", plan)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {name: GPT2LMHeadModel.from_pretrained(path.parent, local_files_only=True).to(device).eval()
              for name, path in model_files.items()}
    for model in models.values():
        model.requires_grad_(False)
    previous_voicing, previous_rhythm = lv.INITIAL, []
    rows, learned, fixed = [], [], []
    for label in labels:
        summary = lv.melody_summary(melody, label)
        voicing = predict_voicing(models["voicing"], label, summary, previous_voicing)
        metrics = lv.assess(voicing, label, summary)
        frames = predict_rhythm(models["rhythm"], label, melody, ppq, previous_rhythm)
        rows.append({"symbol": label["symbol"], "start_tick": label["start_tick"], "end_tick": label["end_tick"],
                     "voicing": voicing, "voicing_metrics": metrics, "frames": frames})
        write_json(out / "raw_predictions.json", rows)
        if not all(value for key, value in metrics.items() if key != "chord_tone_ratio"):
            raise ValueError("Frozen voicing failed; raw output saved without pitch correction")
        learned.extend(rv.render(label, voicing, frames, ppq, summary["velocity"]))
        fixed.extend(lv.schedule(label, voicing, ppq, "arpeggio", summary["velocity"]))
        previous_voicing, previous_rhythm = voicing, frames
    cases = {}
    for name, raw_notes in (("learned", learned), ("fixed", fixed)):
        lv.write_midi(source, melody, raw_notes, out / f"{name}_original_register.mid", end=end)
        shifted = [{**note, "pitch": note["pitch"] + score["accompaniment_semitones"]} for note in raw_notes]
        if any(not 0 <= note["pitch"] <= 127 for note in shifted):
            raise ValueError("Octave shift exceeds MIDI range; no clipping applied")
        lv.write_midi(source, melody, shifted, out / f"{name}.mid", end=end)
        if read_role(out / f"{name}.mid", "MELODY") != sorted(event_key(note) for note in melody):
            raise ValueError("Melody round trip changed events")
        if read_role(out / f"{name}.mid", "ACCOMPANIMENT") != sorted(event_key(note) for note in shifted):
            raise ValueError("Accompaniment round trip changed events")
        for note in shifted:
            label = next(label for label in labels if label["start_tick"] <= note["start_tick"] < label["end_tick"])
            if note["pitch"] % 12 not in label["pitch_classes"] or note["end_tick"] > label["end_tick"]:
                raise ValueError("Accompaniment leaves the supplied chord or crosses its boundary")
        pairs = [(a, m) for a in shifted for m in melody if a["start_tick"] < m["end_tick"] and a["end_tick"] > m["start_tick"]]
        cases[name] = {"notes": len(shifted), "pitch_range": [min(n["pitch"] for n in shifted), max(n["pitch"] for n in shifted)],
                       "minimum_simultaneous_melody_clearance": min(m["pitch"] - a["pitch"] for a, m in pairs),
                       "crossing_or_unison_pairs": sum(a["pitch"] >= m["pitch"] for a, m in pairs),
                       "source_notes": raw_notes, "audition_notes": shifted}
    # 对照仅改变节奏、时值和相对力度；两个版本共享配位与低音。
    basses = lambda notes: sorted(event_key(n) for n in notes if any(
        n["pitch"] == row["voicing"][0] and n["start_tick"] == row["start_tick"] for row in rows))
    if basses(learned) != basses(fixed):
        raise ValueError("Bass differs between compared methods")
    if any(digest(path) != expected[name] for name, path in model_files.items()) or digest(score_path) != score_hash:
        raise ValueError("Frozen inputs changed")
    result = {"case_id": score["case_id"], "score_sha256": score_hash, "model_sha256": expected,
              "decoding": "single deterministic greedy rollout; no retraining or candidate selection",
              "input_melody_notes": len(melody), "duration_seconds": end / ppq * 60 / score["tempo_bpm"],
              "melody_range": [min(n["pitch"] for n in melody), max(n["pitch"] for n in melody)],
              "accompaniment_semitones": score["accompaniment_semitones"], "training_use": False,
              "melody_roundtrip_exact": True, "same_bass_and_voicing": True, "chord_and_boundaries_valid": True,
              "cases": cases, "rows": rows, "human_audition_status": "pending",
              "reference_target_available": False, "scope": score["interpretation_limit"]}
    write_json(out / "generation.json", result)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/render_piano_preview.py"),
                    "--input-midi", str(out / "learned.mid"), str(out / "fixed.mid"), str(out / "melody.mid"),
                    "--out-dir", str(out / "piano")], cwd=ROOT, check=True)
    chords = " · ".join(bar["chord"].replace(":maj", "").replace(":min", "m") for bar in score["bars"])
    cards = "".join(f"<section><h2>{title}</h2><audio controls preload='metadata' src='piano/{name}_piano.wav'></audio>"
                    f"<p><a href='{name}.mid'>MIDI</a> · <a href='piano/{name}_piano.wav'>WAV</a></p></section>"
                    for name, title in (("learned", "V4 模型伴奏"), ("fixed", "固定分解对照"), ("melody", "单独旋律")))
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(score['title'])} · 新旋律试听</title>
<style>body{{max-width:860px;margin:36px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}}audio{{width:100%}}section{{border-top:1px solid #bbb;padding:20px 0}}a{{color:#226466}}details{{margin:24px 0}}</style>
<h1>{html.escape(score['title'])}</h1><p>新写的 8 小节旋律 · G 大调 · 4/4 · 100 BPM · 约 19 秒</p>
<p>和声：{html.escape(chords)}</p><p>两版伴奏均升高一个八度，旋律不移调。采用同一采样钢琴音色与共同增益。</p>
{cards}<details><summary>本次试验说明</summary><p>旋律与和声先写好，再用已验收 V3 配位模型和 V4 节奏模型生成一次；未重训、未挑选多个候选，也未手工修饰模型输出。固定分解使用同一配位与低音。升八度是统一试听处理。</p>
<p>这是单个简单新案例，尚待试听。和声由助手给定，不是模型识别结果；没有参考伴奏，故不计算“准确率”。</p>
<p><a href='score.json'>旋律与和声</a> · <a href='generation.json'>逐段生成记录</a> · <a href='piano/result.json'>音频核验</a></p></details></html>"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    result["artifact_sha256"] = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.suffix in (".mid", ".wav")}
    write_json(out / "generation.json", result)
    print(json.dumps({"page": str(out / "index.html"), "melody_notes": len(melody),
                      "cases": {name: {k: v for k, v in case.items() if k not in ("source_notes", "audition_notes")} for name, case in cases.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
