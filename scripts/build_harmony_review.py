"""Build musical annotations, deterministic block MIDIs and an audition pack."""

from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import html
import json
import sys
import warnings
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harmony_plan as hp
import numpy as np
import pretty_midi
from import_pop909 import load_piece as load_pop909_piece


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_preview_audio(midi, destination, start=0.0, end=None):
    end = midi.get_end_time() if end is None else end
    if not 0 <= start < end:
        raise ValueError("Invalid audition excerpt")
    rate = 22050
    audio = np.zeros((round((end - start) * rate), 2), dtype=np.float64)
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        melody = instrument.name == "MELODY"
        pan = np.array([0.80, 0.65]) if melody else np.array([0.65, 0.80])
        for note in instrument.notes:
            left, right = max(start, note.start), min(end, note.end)
            if right <= left:
                continue
            a = max(0, round((left - start) * rate))
            b = min(len(audio), round((right - start) * rate))
            t = np.arange(b - a) / rate + left - note.start
            duration = note.end - note.start
            frequency = pretty_midi.note_number_to_hz(note.pitch)
            sound = np.zeros(len(t))
            # 固定的柔和加法合成音色；只用于和声试听，不模拟真实钢琴音质。
            for harmonic, weight in ((1, 1.0), (2, 0.35), (3, 0.16), (4, 0.08)):
                if frequency * harmonic < rate / 2:
                    sound += weight * np.sin(2 * np.pi * frequency * harmonic * t) * np.exp(-t * harmonic / 2.5)
            envelope = np.minimum(1.0, t / 0.012) * np.minimum(1.0, np.maximum(0, duration - t) / 0.04)
            sound *= envelope * (note.velocity / 127) * (0.6 if melody else 0.4)
            audio[a:b] += sound[:, None] * pan
    peak = np.max(np.abs(audio)) if audio.size else 0
    if peak:
        audio *= 0.90 / max(1.0, peak)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((audio * 32767).astype("<i2").tobytes())


def build_piece(piece, midi, out_dir, **sidecars):
    out_dir = Path(out_dir)
    annotated = hp.annotate_music(piece, midi, **sidecars)
    plan = annotated["harmony_plan"]
    melody = [n for n in piece["notes"] if n["role"] == "melody"]
    destination = out_dir / "block_midis" / f"{piece['piece_id']}.mid"
    rendered = hp.write_block_midi(midi, melody, plan, destination)
    validation = validate_render(midi, melody, plan, destination)
    statuses = Counter(x["status"] for x in plan["labels"])
    reasons = Counter(reason for x in plan["labels"] for reason in x.get("review_reasons", []))
    total_ticks = plan["duration_ticks"]
    unknown_ticks = sum(x["end_tick"] - x["start_tick"] for x in plan["labels"] if x["status"] == "unknown")
    row = {"piece_id": piece["piece_id"], "source_piece_id": piece.get("source_piece_id", piece["piece_id"]),
           "note_count": len(piece["notes"]), "labels": len(plan["labels"]), "statuses": dict(statuses),
           "unknown_time_ratio": unknown_ticks / total_ticks, "review_reasons": dict(reasons),
           "needs_priority_review": any(x.get("review_reasons") for x in plan["labels"]),
           "evidence_scope": plan["evidence_scope"], "review_status": "unreviewed",
           "render_validation": validation, "block_midi": str(destination),
           "key_candidates": annotated["music_annotation"]["key_estimate"]["candidates"]}
    row["source_midi_warnings"] = piece.get("source_midi_warnings", [])
    row["source_midi_metadata_audit"] = piece.get("source_midi_metadata_audit", {})
    row["needs_priority_review"] |= bool(row["source_midi_metadata_audit"].get("conflicts"))
    row["training_eligible"] = False
    source = Path(piece.get("source_file", ""))
    if source.is_file():
        annotated["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(out_dir / "annotated_notes" / f"{piece['piece_id']}.json", annotated)
    write_json(out_dir / "plans" / f"{piece['piece_id']}.json", plan)
    return row, annotated, rendered


def validate_render(midi, melody, plan, path):
    actual = pretty_midi.PrettyMIDI(str(path))
    tick = lambda t: int(actual.time_to_tick(t))
    expected_melody = sorted((int(n["pitch"]), int(n.get("velocity", 80)), int(midi.time_to_tick(n["start"])), int(midi.time_to_tick(n["end"]))) for n in melody)
    actual_melody = sorted((n.pitch, n.velocity, tick(n.start), tick(n.end)) for i in actual.instruments if i.name == "MELODY" for n in i.notes)
    blocks = [n for i in actual.instruments if i.name == "BLOCK_CHORDS" for n in i.notes]
    by_start = {}
    for note in blocks:
        by_start.setdefault(tick(note.start), []).append(note)
    for label in plan["labels"]:
        column = by_start.get(label["start_tick"], [])
        if label["status"] == "chord":
            if not column or {n.pitch % 12 for n in column} != set(label["pitch_classes"]) | {label["bass_pc"]}:
                raise ValueError("Rendered chord pitch classes differ from annotation")
            if min(n.pitch for n in column) % 12 != label["bass_pc"]:
                raise ValueError("Rendered inversion bass differs from annotation")
            if any(tick(n.end) != label["end_tick"] for n in column):
                raise ValueError("Rendered chord time differs from annotation")
        elif column:
            raise ValueError("Renderer filled an unknown or no-chord interval")
    expected_blocks = sorted((n["pitch"], n["velocity"], n["start_tick"], n["end_tick"]) for n in hp.block_chord_notes(plan))
    actual_blocks = sorted((n.pitch, n.velocity, tick(n.start), tick(n.end)) for n in blocks)
    tempo = lambda m: [(int(m.time_to_tick(t)), round(float(b), 4)) for t, b in zip(*m.get_tempo_changes())]
    meter = lambda m: [(int(m.time_to_tick(t.time)), t.numerator, t.denominator) for t in m.time_signature_changes]
    # pretty_midi 会在缺少拍号时写入默认 4/4；默认假设已在时间轴标注中披露。
    meter_ok = meter(midi) == meter(actual) if midi.time_signature_changes else meter(actual) in ([], [(0, 4, 4)])
    result = {"melody_roundtrip_exact": expected_melody == actual_melody,
              "block_chords_roundtrip_exact": expected_blocks == actual_blocks,
              "tempo_preserved": tempo(midi) == tempo(actual), "meter_preserved_or_declared_default": meter_ok,
              "chord_classes_bass_and_timing_match_plan": True}
    if not all(result.values()):
        raise ValueError(f"MIDI render validation failed: {result}")
    return result


def load_midi(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        midi = pretty_midi.PrettyMIDI(str(path))
    return midi, list(dict.fromkeys(str(w.message) for w in caught))


def make_manual_plan(midi, identifier, events, source="assistant_proposal"):
    duration = int(midi.time_to_tick(midi.get_end_time()))
    labels = []
    for start, end, symbol in events:
        labels.append({**hp.parse_symbol(symbol), "start_tick": round(start * midi.resolution),
                       "end_tick": min(duration, round(end * midi.resolution)), "source": source,
                       "confidence": None, "review_status": "unreviewed", "review_reasons": []})
    plan = {"version": "harmony_plan_v2", "piece_id": identifier, "ppq": midi.resolution,
            "duration_ticks": duration, "evidence_scope": source, "review_status": "unreviewed", "labels": labels}
    hp.validate_plan(plan)
    return plan


def plans_from_config(midi, config):
    plans = []
    for candidate in config["candidates"]:
        source = config.get("source", "assistant_proposal")
        anchored = "timed_events" in candidate
        if anchored:
            if "events" in candidate:
                raise ValueError("Choose timed_events or legacy events, not both")
            plan = make_timed_plan(midi, candidate["id"], candidate["timed_events"], source)
        else:
            plan = make_manual_plan(midi, candidate["id"], candidate["events"], source)
        for key in ("key_context", "review_status", "user_feedback", "timing_assumption",
                    "progression", "musical_timing", "references", "melody_sha256", "audition_acceptance"):
            if key in config:
                plan[key] = config[key]
        if not anchored:
            for label in plan["labels"]:
                label["review_status"] = plan["review_status"]
        plans.append(plan)
    return plans


def make_timed_plan(midi, identifier, events, source):
    # 编号从 1 开始，只接受无交叠的单音旋律，避免把复调排序冒充旋律音序。
    notes = sorted((int(midi.time_to_tick(n.start)), n.pitch, int(midi.time_to_tick(n.end)))
                   for instrument in midi.instruments if not instrument.is_drum for n in instrument.notes)
    if any("onset_between_notes" in event for event in events) and any(b[0] < a[2] for a, b in zip(notes, notes[1:])):
        raise ValueError("Note anchors require a monophonic melody")
    plan = make_manual_plan(midi, identifier, [], source)

    def unknown(start, end):
        return {**hp.parse_symbol("X"), "start_tick": start, "end_tick": end,
                "source": "unannotated_gap", "confidence": None, "review_status": "unreviewed",
                "review_reasons": ["harmony_not_specified"]}

    cursor = 0
    for event in events:
        start, end = (round(event[key] * midi.resolution) for key in ("start_quarter", "end_quarter"))
        onset_range = event.get("onset_between_notes")
        if onset_range is not None:
            first, last = onset_range
            if type(first) is not int or type(last) is not int or not 1 <= first < last <= len(notes):
                raise ValueError("Invalid one-based note range")
            # “两音之间”约束起音范围；前一音可能仍在持续，不能吸附到 note-off。
            if not notes[first - 1][0] < start < notes[last - 1][0]:
                raise ValueError("Chord must start strictly between note onsets")
        if start > cursor:
            plan["labels"].append(unknown(cursor, start))
        plan["labels"].append({**hp.parse_symbol(event["symbol"]), "start_tick": start, "end_tick": end,
                               "source": event.get("source", source), "confidence": None,
                               "review_status": event.get("review_status", "unreviewed"), "review_reasons": [],
                               "onset_between_notes": onset_range})
        cursor = end
    if cursor < plan["duration_ticks"]:
        plan["labels"].append(unknown(cursor, plan["duration_ticks"]))
    hp.validate_plan(plan)
    return plan


def add_audition(items, out_dir, identifier, title, midi, plan, reference=None, start=0.0, end=None):
    end = min(midi.get_end_time(), start + 16.0) if end is None else end
    folder = out_dir / "review"
    write_preview_audio(midi, folder / f"{identifier}.wav", start, end)
    if reference is not None:
        write_preview_audio(reference, folder / f"{identifier}_reference.wav", start, end)
    # 显示相对试听片段的位置，保留原曲秒数便于反馈定位。
    events = []
    for label in plan["labels"]:
        a, b = midi.tick_to_time(label["start_tick"]), midi.tick_to_time(label["end_tick"])
        if a < end and b > start:
            item = {"start": max(a, start) - start, "end": min(b, end) - start,
                    "symbol": label["symbol"], "bass_pc": label["bass_pc"], "status": label["status"],
                    "source": label["source"], "review_status": label.get("review_status", "unreviewed"),
                    "onset_between_notes": label.get("onset_between_notes")}
            if (events and all(events[-1][key] == item[key] for key in ("symbol", "source", "review_status", "onset_between_notes"))
                    and abs(events[-1]["end"] - item["start"]) < 1e-6):
                events[-1]["end"] = item["end"]
            else:
                events.append(item)
    items.append({"id": identifier, "title": title, "start_seconds": start, "end_seconds": end,
                  "audio": f"{identifier}.wav", "reference_audio": f"{identifier}_reference.wav" if reference is not None else None,
                  "midi": f"../block_midis/{plan['piece_id']}.mid", "plan": f"../plans/{plan['piece_id']}.json",
                  "scope": plan["evidence_scope"], "events": events, "review_status": plan["review_status"],
                  "timing_assumption": plan.get("timing_assumption", "")})


def write_review_page(items, out_dir):
    sections = []
    for item in items:
        esc = html.escape
        rows = "".join(f"<tr><td>{e['start']:.3f}–{e['end']:.3f}s</td><td>{esc(e['symbol'])}</td><td>{e['bass_pc'] if e['bass_pc'] is not None else '—'}</td>"
                       f"<td>{esc(e.get('review_status', 'unreviewed'))}</td>"
                       f"<td>{str(e['onset_between_notes'][0]) + '→' + str(e['onset_between_notes'][1]) + ' 起音之间' if e.get('onset_between_notes') else '—'}</td></tr>" for e in item["events"])
        reference = f"<p>原编配参考（同一合成音色）</p><audio controls preload='none' src='{esc(item['reference_audio'])}'></audio>" if item["reference_audio"] else ""
        sections.append(f"""<section data-case="{esc(item['id'])}"><h2>{esc(item['title'])}</h2>
<p>原曲 {item['start_seconds']:.2f}–{item['end_seconds']:.2f}s · 来源：{esc(item['scope'])} · {esc(item['review_status'])}</p>
<p>{esc(item['timing_assumption'])}</p>
<p>旋律 + 柱式和弦</p><audio controls preload="none" src="{esc(item['audio'])}"></audio>{reference}
<p><a href="{esc(item['midi'])}">完整 MIDI</a> · <a href="{esc(item['plan'])}">和声标注 JSON</a></p>
<details><summary>查看和弦时间表（相对片段）</summary><table><tr><th>时间</th><th>和弦</th><th>低音音级（C=0）</th><th>复核状态</th><th>旋律音符编号</th></tr>{rows}</table></details>
<label>判断 <select><option value="pending">待试听</option><option value="accept">这一段可接受</option><option value="harmony_issue">和声不合适</option><option value="timing_issue">换和弦时间不对</option><option value="role_issue">旋律/声部标注不对</option></select></label>
<textarea placeholder="例如：第 5.2 秒的低音不合适；建议换成……"></textarea></section>""")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pianalysis 和声试听</title><style>body{max-width:960px;margin:48px auto;padding:0 24px;font:16px/1.7 system-ui;color:#202b33;background:#f5f3ed}h1{font-size:32px}section{padding:24px 0;border-top:1px solid #ccc}audio{width:100%;max-width:650px}table{border-collapse:collapse;width:100%}td,th{padding:6px;text-align:left;border-bottom:1px solid #ddd}textarea{display:block;width:95%;height:72px;margin:12px 0}button,select{padding:8px}a{color:#125d6a}small{color:#5b6266}</style>
<h1>先听和声，再做织体</h1><p>优先听 test1，并查看该段的音符锚点与复核状态。文件起点不代表音乐强拍；推定拍点仍需试听核对，X 区间不补和弦。再对照留出曲目与 POP909。
这些音频使用固定的柔和合成音色；MIDI 可用钢琴音源播放。请判断和弦、低音与变化时机，不评价音色。</p>
<p>自动标注尚未人工验收。N 表示无和弦，X 表示未知；未知片段保持安静。库内曲目利用了原伴奏识别和声，不代表仅凭旋律自动配和声的能力。</p>
<button id="export">下载试听反馈 JSON</button> <small id="state">反馈保存在当前页面的浏览器中；也可下载带回项目。</small>
""" + "\n".join(sections) + """
<script>
const key='pianalysis-harmony-v2-review'; let saved={};
try { saved=JSON.parse(localStorage.getItem(key)||'{}'); } catch(e) {}
const sections=[...document.querySelectorAll('section[data-case]')];
function collect(){return Object.fromEntries(sections.map(s=>[s.dataset.case,{judgment:s.querySelector('select').value,comment:s.querySelector('textarea').value}]))}
for(const s of sections){const v=saved[s.dataset.case];if(v){s.querySelector('select').value=v.judgment;s.querySelector('textarea').value=v.comment;}s.addEventListener('change',()=>{try{localStorage.setItem(key,JSON.stringify(collect()))}catch(e){document.querySelector('#state').textContent='浏览器不允许本地保存，请下载反馈。'}})}
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify({version:'harmony_review_v2',reviews:collect()},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='harmony_review_feedback.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>"""
    path = out_dir / "review/index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(page)
    write_json(out_dir / "review/cases.json", items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotated-dir", default="data/dp_cleaned_v1/annotated_notes")
    parser.add_argument("--pop909-dir", default="data/pop909_sample")
    parser.add_argument("--out-dir", default="data/harmony_v2_review")
    parser.add_argument("--manual-plans", default="examples/harmony/test1_d_major_4563.json")
    parser.add_argument("--reference-windows", default="data/training_windows_v1_1/dataset_windows_v1.json")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, auditions, review_rows = [], [], []
    cache = {}
    files = sorted(Path(args.annotated_dir).glob("*.json"))
    if not files:
        raise ValueError("No annotated MIDI pieces found")
    for path in files:
        piece = json.loads(path.read_text(encoding="utf-8"))
        midi, messages = load_midi(piece["source_file"])
        piece["source_midi_warnings"] = messages
        piece["source_midi_metadata_audit"] = hp.inspect_midi_metadata(piece["source_file"])
        row, annotated, rendered = build_piece(piece, midi, out_dir)
        records.append(row)
        cache[piece["piece_id"]] = (annotated, midi, rendered)
        print(f"Annotated {piece['piece_id']}: {row['labels']} spans, unknown={row['unknown_time_ratio']:.1%}", flush=True)
    pop_root = Path(args.pop909_dir)
    for path in sorted(pop_root.glob("[0-9][0-9][0-9]/*.mid")):
        piece = load_pop909_piece(path, pop_root)
        midi, messages = load_midi(path)
        piece["source_midi_warnings"] = messages
        piece["source_midi_metadata_audit"] = hp.inspect_midi_metadata(path)
        row, annotated, rendered = build_piece(piece, midi, out_dir, chord_path=path.parent / "chord_midi.txt",
                                               beat_path=path.parent / "beat_midi.txt", key_path=path.parent / "key_audio.txt")
        records.append(row)
        cache[piece["piece_id"]] = (annotated, midi, rendered)
        start = min(n["start"] for n in piece["notes"] if n["role"] == "melody")
        add_audition(auditions, out_dir, piece["piece_id"], f"POP909 {path.stem} · 官方提取标注", rendered, annotated["harmony_plan"], midi, start)
        print(f"Imported POP909 {path.stem}: {row['labels']} timed chord spans", flush=True)

    references = json.loads(Path(args.reference_windows).read_text(encoding="utf-8"))["samples"]
    heldout = ("apple-seed", "frieren-beyond-journeys-end-medley", "gurenge", "vogel-im-kafig")
    for name in heldout:
        annotated, original, rendered = cache[name]
        windows = [w for w in references if w["source_piece_id"] == name]
        for index in (0, len(windows) // 2):
            window = windows[index]
            add_audition(auditions, out_dir, window["piece_id"], f"留出曲目 · {name} · {window['start_time']:.1f}s",
                         rendered, annotated["harmony_plan"], original, window["start_time"], window["end_time"])

    manual = json.loads(Path(args.manual_plans).read_text(encoding="utf-8"))
    melody_path = Path(manual["melody_file"])
    if hashlib.sha256(melody_path.read_bytes()).hexdigest() != manual["melody_sha256"]:
        raise ValueError("Manual harmony candidates do not match this melody file")
    midi, _ = load_midi(melody_path)
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start": n.start, "end": n.end} for i in midi.instruments for n in i.notes]
    for candidate, plan in zip(manual["candidates"], plans_from_config(midi, manual)):
        path = out_dir / "block_midis" / f"{candidate['id']}.mid"
        rendered = hp.write_block_midi(midi, melody, plan, path)
        validation = validate_render(midi, melody, plan, path)
        write_json(out_dir / "plans" / f"{candidate['id']}.json", {**plan, "render_validation": validation})
        add_audition(auditions, out_dir, candidate["id"], candidate["title"], rendered, plan, end=midi.get_end_time())

    # 已知的 I-IV-V7-I 与转位，用于把渲染正确性和真实曲目和声适配分开检验。
    synthetic = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
    synthetic.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0)]
    instrument = pretty_midi.Instrument(0, name="MELODY")
    instrument.notes = [pretty_midi.Note(76, pitch, i * 2, (i + 1) * 2) for i, pitch in enumerate((76, 77, 74, 72))]
    synthetic.instruments = [instrument]
    plan = make_manual_plan(synthetic, "known_cadence", [(0, 4, "C:maj/3"), (4, 8, "F:maj/3"), (8, 12, "G:7/3"), (12, 16, "C:maj")], "synthetic_known_harmony")
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start": n.start, "end": n.end} for n in instrument.notes]
    path = out_dir / "block_midis/known_cadence.mid"
    rendered = hp.write_block_midi(synthetic, melody, plan, path)
    validation = validate_render(synthetic, melody, plan, path)
    write_json(out_dir / "plans/known_cadence.json", {**plan, "render_validation": validation})
    add_audition(auditions, out_dir, "known_cadence", "已知进行 · C–F–G7–C（含转位）", rendered, plan, end=8)
    auditions.sort(key=lambda x: (0 if x["id"].startswith("test1_") else 1 if x["id"] == "known_cadence" else 2, x["id"]))
    write_review_page(auditions, out_dir)

    for name, (annotated, midi, _) in cache.items():
        has_conflict = bool(annotated.get("source_midi_metadata_audit", {}).get("conflicts"))
        for index, label in enumerate(annotated["harmony_plan"]["labels"]):
            if label.get("review_reasons") or has_conflict:
                review_rows.append({"piece_id": name, "label_index": index, "start_seconds": midi.tick_to_time(label["start_tick"]),
                                    "end_seconds": midi.tick_to_time(label["end_tick"]), "symbol": label["symbol"],
                                    "reasons": ";".join(label.get("review_reasons", []) + (["source_midi_metadata_conflict"] if has_conflict else [])),
                                    "review_status": "unreviewed"})
    with (out_dir / "priority_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["piece_id", "label_index", "start_seconds", "end_seconds", "symbol", "reasons", "review_status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)
    summary = {"version": "harmony_review_v2", "pieces": len(records), "notes": sum(r["note_count"] for r in records),
               "harmony_spans": sum(r["labels"] for r in records), "audition_cases": len(auditions),
               "priority_review_spans": len(review_rows), "all_renders_passed": all(all(r["render_validation"].values()) for r in records),
               "pieces_with_source_midi_warnings": sum(bool(r["source_midi_warnings"]) for r in records),
               "pieces_with_metadata_conflicts": sum(bool(r["source_midi_metadata_audit"].get("conflicts")) for r in records),
               "human_accepted_cases": sum(x["review_status"] == "user_audition_accepted" for x in auditions),
               "ready_for_texture_training": False,
               "user_progression_cases": sum(x["scope"] == "user_specified_progression" for x in auditions),
               "notes_on_scope": ["Automatic and upstream annotations remain unreviewed.",
                                  "Corpus harmony uses original accompaniment, not a melody-only planner.",
                                  "Manual harmony uses explicit timing and review status; MIDI zero is not assumed to be a musical downbeat.",
                                  "MIDI metadata warnings are retained and need review; no silent tempo repair."],
               "piece_reports": records}
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "piece_reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
