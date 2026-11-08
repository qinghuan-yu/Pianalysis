"""Build a bounded, source-pinned symbolic review set and simple auditions."""

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path

import harmony_plan as hp
from import_pop909 import load_piece
from build_harmony_review import write_json, write_preview_audio
import local_voicing as lv

ROOT = Path(__file__).resolve().parents[1]


def check_hashes(folder, expected):
    for name, digest in expected.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reviewed source changed: {folder / name}")


def evidence(piece, midi, label, beat_times):
    a, b = (float(midi.tick_to_time(label[k])) for k in ("start_tick", "end_tick"))
    active = [n for n in piece["notes"] if n["role"] == "accompaniment" and n["start"] < b and n["end"] > a]
    weights = Counter()
    for n in active:
        weights[n["pitch"] % 12] += max(0, min(b, n["end"]) - max(a, n["start"]))
    total = sum(weights.values())
    fit = sum(weights[p] for p in label["pitch_classes"]) / total if total else 0
    bass_notes = [n for n in active if n["pitch"] % 12 == label["bass_pc"]]
    bass_onset = min((abs(n["start"] - a) for n in bass_notes), default=float("inf"))
    sounding = [n for n in active if n["start"] <= a and n["end"] >= a + .12]
    sustained_bass = bool(sounding) and min(sounding, key=lambda n: n["pitch"])["pitch"] % 12 == label["bass_pc"]
    grid_error = max(min(abs(t - x) for t in beat_times) for x in (a, b))
    checks = {"complete_triad_evidence": len(label["pitch_classes"]) == 3 and all(weights[p] > .06 for p in label["pitch_classes"]),
              "accompaniment_coverage": fit >= .98,
              "bass_anchor": bass_onset <= .03 or sustained_bass,
              "beat_boundary_alignment": grid_error <= .004}
    return {"start_seconds": a, "end_seconds": b, "pitch_class_note_seconds": dict(weights),
            "chord_fit": fit, "bass_onset_error_seconds": bass_onset, "bass_anchor_kind": "onset" if bass_onset <= .03 else "sustained" if sustained_bass else "missing",
            "beat_grid_error_seconds": grid_error,
            "checks": checks, "accompaniment_note_ids": [n["id"] for n in active]}


def build(config, source_root, out):
    if config.get("review_status") != "assistant_symbolic_reviewed" or config.get("reviewer") != "assistant":
        raise ValueError("An explicit symbolic review record is required")
    out = Path(out)
    if (out / "annotations.json").exists():
        raise ValueError("Choose a fresh annotation output directory")
    records, rows, sections = [], [], []
    identities = {}
    for selection in config["pieces"]:
        song, split = selection["song"], selection["split"]
        if song in identities or split not in ("train", "validation", "test"):
            raise ValueError("Song must appear in exactly one valid split")
        identities[song] = split
        folder = Path(source_root) / song
        check_hashes(folder, selection["hashes"])
        midi = hp.pretty_midi.PrettyMIDI(str(folder / (song + ".mid")))
        piece = load_piece(folder / (song + ".mid"), source_root)
        labels = hp.read_chord_sidecar(folder / "chord_midi.txt", midi)
        beats = [float(line.split()[0]) for line in (folder / "beat_midi.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(selection["indices"]) != len(selection["symbols"]):
            raise ValueError("Review indices and symbols disagree")
        selected = []
        for index, symbol in zip(selection["indices"], selection["symbols"]):
            label = labels[index]
            if label["symbol"] != symbol:
                raise ValueError("Upstream label differs from the reviewed decision")
            proof = evidence(piece, midi, label, beats)
            if not all(proof["checks"].values()):
                raise ValueError(f"Review gate failed for {song}/{index}: {proof}")
            label = {**label, "review_status": config["review_status"], "reviewer": "assistant",
                     "source_label_index": index, "evidence": proof}
            selected.append(label)
            rows.append({"song": song, "split": split, "label_index": index, "symbol": symbol,
                         "start_seconds": proof["start_seconds"], "end_seconds": proof["end_seconds"],
                         "chord_fit": proof["chord_fit"], "bass_error_ms": 1000 * proof["bass_onset_error_seconds"],
                         "bass_anchor_kind": proof["bass_anchor_kind"],
                         "beat_error_ms": 1000 * proof["beat_grid_error_seconds"], "review_status": label["review_status"]})
        origin, end = selected[0]["start_tick"], selected[-1]["end_tick"]
        notes = [{**n, "start_tick": int(midi.time_to_tick(n["start"])), "end_tick": int(midi.time_to_tick(n["end"]))}
                 for n in piece["notes"] if n["start"] < midi.tick_to_time(end) and n["end"] > midi.tick_to_time(origin)]
        melody = [n for n in notes if n["role"] == "melody"]
        if not melody:
            raise ValueError("Reviewed excerpt has no named melody")
        previous = lv.INITIAL
        for label in selected:
            label["melody_summary"] = lv.melody_summary(melody, label)
            label["synthetic_teacher_voicing"] = lv.teacher_voicing(label, label["melody_summary"], previous)
            previous = label["synthetic_teacher_voicing"]
        record = {"song": song, "source_piece_id": piece["source_piece_id"], "split": split,
                  "source_file": str(folder / (song + ".mid")), "source_hashes": selection["hashes"],
                  "review_note": selection["review_note"], "review_status": config["review_status"],
                  "human_audition_status": "pending", "ppq": midi.resolution, "origin_tick": origin, "end_tick": end,
                  "labels": selected, "notes": notes, "roles_source": "POP909_named_MELODY_and_PIANO",
                  "beat_times": [t for t in beats if midi.tick_to_time(origin) - .004 <= t <= midi.tick_to_time(end) + .004],
                  "target_source": "synthetic_curriculum_rule_teacher"}
        records.append(record)
        audio_dir = out / "review"
        previews = {}
        for style in ("reference", "block", "arpeggio"):
            if style == "reference":
                accompaniment = [n for n in notes if n["role"] == "accompaniment"]
            else:
                accompaniment = [n for label in selected for n in lv.schedule(label, label["synthetic_teacher_voicing"], midi.resolution,
                                  style, label["melody_summary"]["velocity"])]
            stem = f"pop909_{song}_{style}"
            rendered = lv.write_midi(midi, melody, accompaniment, audio_dir / (stem + ".mid"), origin, end)
            write_preview_audio(rendered, audio_dir / (stem + ".wav"))
            previews[style] = stem
        chord_rows = "".join(f"<tr><td>{l['evidence']['start_seconds']:.3f}–{l['evidence']['end_seconds']:.3f}</td><td>{html.escape(l['symbol'])}</td><td>{l['evidence']['chord_fit']:.1%}</td><td>{'低音延续' if l['evidence']['bass_anchor_kind'] == 'sustained' else format(l['evidence']['bass_onset_error_seconds']*1000, '.1f') + ' ms'}</td></tr>" for l in selected)
        audio = "".join(f"<p>{title}</p><audio controls preload='none' src='{previews[style]}.wav'></audio>" for style, title in
                        (("reference", "原编配参考"), ("block", "规则柱式核对"), ("arpeggio", "规则简单分解核对")))
        sections.append(f"<section><h2>POP909 {song} · {split}</h2><p>{html.escape(selection['review_note'])}</p>{audio}<table><tr><th>原曲秒数</th><th>和弦</th><th>原伴奏内音覆盖</th><th>低音起音偏差</th></tr>{chord_rows}</table></section>")
    dataset = {"version": config["version"], "review_status": config["review_status"], "reviewer": "assistant",
               "human_audition_status": "pending", "scope": config["scope"], "target_policy": config["target_policy"],
               "training_gate": {"source_hashes_verified": True, "explicit_excerpt_review": True,
                                 "note_and_beat_evidence_passed": True, "song_disjoint_splits": True,
                                 "test1_excluded": True}, "pieces": records, "excluded_examples": config["excluded_examples"]}
    write_json(out / "annotations.json", dataset)
    with (out / "evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>可信标注核对</title><style>body{max-width:900px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}section{border-top:1px solid #bbb;padding:20px 0}audio{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd}</style>" + f"<h1>本批{len(rows)}段：和声与声部核对</h1>" + "<p>这些片段已逐段核对MIDI音符证据，仍待用户试听。原旋律来自命名旋律轨；柱式和分解版是规则生成的练习目标，不是模型生成，也不是原编曲。未选中的时间区间没有标注，不据此推断原曲无和声。</p>" + "".join(sections) + "</html>"
    (out / "review/index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"pieces": len(records), "segments": len(rows), "by_split": dict(Counter(r['split'] for r in rows)),
                      "min_chord_fit": min(r["chord_fit"] for r in rows), "gate": dataset["training_gate"]}, ensure_ascii=False), flush=True)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/harmony/trusted_subset_v3.json")
    parser.add_argument("--source-root", default="data/pop909_sample")
    parser.add_argument("--out-dir", default="data/trusted_harmony_v3")
    args = parser.parse_args()
    build(json.loads(Path(args.config).read_text(encoding="utf-8")), args.source_root, args.out_dir)


if __name__ == "__main__":
    main()
