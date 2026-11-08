"""Build bounded test1 counterfactual auditions; do not train or promote a model."""

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import wave

import numpy as np

from build_harmony_review import write_json, write_preview_audio, synthesize_preview_audio
import harmony_plan as hp
import local_voicing as lv
import rhythm_ablation as ab


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def note_tuples(notes, field=None):
    keys = [k for k in ("pitch", "start_tick", "end_tick", "velocity") if k != field]
    return sorted(tuple(n[k] for k in keys) for n in notes)


def verify_midi(source, melody, notes, path, labels):
    actual = hp.pretty_midi.PrettyMIDI(str(path))
    for role, expected in (("MELODY", melody), ("ACCOMPANIMENT", notes)):
        exported = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(actual.time_to_tick(n.start)),
                     "end_tick": int(actual.time_to_tick(n.end))} for i in actual.instruments if i.name == role for n in i.notes]
        assert note_tuples(expected) == note_tuples(exported), (path, role)
    assert actual.resolution == source.resolution
    assert tuple(map(list, actual.get_tempo_changes())) == tuple(map(list, source.get_tempo_changes()))
    assert [(x.time, x.numerator, x.denominator) for x in actual.time_signature_changes] == [(x.time, x.numerator, x.denominator) for x in source.time_signature_changes]
    assert [(x.time, x.key_number) for x in actual.key_signature_changes] == [(x.time, x.key_number) for x in source.key_signature_changes]
    assert all(any(l["start_tick"] <= n["start_tick"] < n["end_tick"] <= l["end_tick"] for l in labels) for n in notes)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/rhythm_naturalness_audit")
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError("Choose a fresh diagnostic directory")
    config_path = Path("examples/harmony/naturalness_ablation.json")
    config = read(config_path)
    audition = read("examples/harmony/rhythm_v5_audition.json")
    for item in list(audition["models"].values()) + audition["files"]:
        assert digest(item["path"]) == item["sha256"]
    evaluation_path = Path("data/trusted_harmony_v5/evaluation/evaluation.json")
    evaluation = read(evaluation_path)
    case = next(c for c in evaluation["cases"] if c["song"] == "test1")
    plan = read("data/harmony_v2_review/plans/test1_pickup_4563.json")
    hp.validate_plan(plan)
    assert digest("test/test1.mid") == plan["melody_sha256"]
    labels = [l for l in plan["labels"] if l["status"] == "chord"]
    source = hp.pretty_midi.PrettyMIDI("test/test1.mid")
    ppq = source.resolution
    melody = [{"pitch": n.pitch, "velocity": n.velocity, "start_tick": int(source.time_to_tick(n.start)),
               "end_tick": int(source.time_to_tick(n.end)), "role": "melody"}
              for i in source.instruments if not i.is_drum for n in i.notes]
    methods = [config["anchor"], *config["donors"]]
    upper = {m: [n for row in case["rows"] for n in row["methods"][m]["notes"][1:]] for m in methods}
    bass = [r["methods"]["accepted_v4"]["notes"][0] for r in case["rows"]]
    assert len(case["rows"]) == len(labels)
    for row, label in zip(case["rows"], labels):
        assert (row["start_tick"], row["end_tick"]) == (label["start_tick"], label["end_tick"])
        assert all(lv.assess(row["voicing"], label, lv.melody_summary(melody, label)).values())
        assert all(row["methods"][m]["notes"][0] == row["methods"]["accepted_v4"]["notes"][0] for m in methods)
    base = upper["accepted_v4"]
    variants = {"reference": copy.deepcopy(base), "standard_reference": ab.standardize(base, labels, ppq)}
    audits = {}
    for donor_name in config["donors"]:
        donor = upper[donor_name]
        variants[donor_name + "_original"] = copy.deepcopy(donor)
        thinned, thinning = ab.thin_to_count(base, donor, labels)
        duration, duration_audit = ab.swap_common_field(base, donor, labels, "end_tick")
        velocity, velocity_audit = ab.swap_common_field(base, donor, labels, "velocity")
        variants[donor_name + "_thin"] = thinned
        variants[donor_name + "_duration"] = duration
        variants[donor_name + "_velocity"] = velocity
        variants[donor_name + "_standard"] = ab.standardize(donor, labels, ppq)
        variants[donor_name + "_thin_standard"] = ab.standardize(thinned, labels, ppq)
        assert len(thinned) == len(donor)
        assert all(n in base for n in thinned)
        assert note_tuples(duration, "end_tick") == note_tuples(base, "end_tick")
        assert note_tuples(velocity, "velocity") == note_tuples(base, "velocity")
        common = set(ab.index_notes(base)) & set(ab.index_notes(donor))
        audits[donor_name] = {"common_upper_onsets": len(common), "anchor_only_onsets": len(base) - len(common),
                              "donor_only_onsets": len(donor) - len(common), "thin": thinning,
                              "duration": duration_audit, "velocity": velocity_audit}
    out.mkdir(parents=True)
    midis, raw_peaks, artifact_hashes, variant_reports = {}, {}, {}, {}
    horizon = float(source.tick_to_time(plan["duration_ticks"]))
    for name, notes in variants.items():
        path = out / f"{name}.mid"
        lv.write_midi(source, melody, bass + notes, path, end=plan["duration_ticks"])
        midis[name] = verify_midi(source, melody, bass + notes, path, labels)
        raw_audio, rate = synthesize_preview_audio(midis[name], end=horizon)
        raw_peaks[name] = float(np.max(np.abs(raw_audio)))
        artifact_hashes[path.name] = digest(path)
        variant_reports[name] = {"notes": notes, "statistics": ab.describe(notes, labels, ppq, melody)}
    shared_gain = .9 / max(1.0, max(raw_peaks.values()))
    # 先按旧默认渲染校验音色与归一化逻辑未变，再生成统一增益的新诊断音频。
    for name, accepted_name in (("reference", "test1_accepted_v4"), ("v5_medium_original", "test1_v5_medium"),
                                ("expanded_v4_original", "test1_expanded_v4")):
        legacy = out / f"legacy_check_{name}.wav"
        write_preview_audio(midis[name], legacy)
        assert legacy.read_bytes() == Path(f"data/trusted_harmony_v5/evaluation/{accepted_name}.wav").read_bytes()
        assert (out / f"{name}.mid").read_bytes() == Path(f"data/trusted_harmony_v5/evaluation/{accepted_name}.mid").read_bytes()
    waveform_length = None
    pickup_pcm = None
    for name, midi in midis.items():
        path = out / f"{name}.wav"
        write_preview_audio(midi, path, end=horizon, gain=shared_gain)
        with wave.open(str(path), "rb") as stream:
            assert stream.getsampwidth() == 2 and stream.getnchannels() == 2 and stream.getframerate() == rate
            pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
            assert np.any(pcm) and np.max(np.abs(pcm.astype(np.int32))) < 32767
            if waveform_length is None:
                waveform_length = len(pcm)
                pickup_pcm = pcm[:round(source.tick_to_time(labels[0]["start_tick"]) * rate)].copy()
            assert len(pcm) == waveform_length
            assert np.array_equal(pickup_pcm, pcm[:len(pickup_pcm)])
        artifact_hashes[path.name] = digest(path)
        variant_reports[name].update(raw_peak=raw_peaks[name], old_individual_gain=.9 / max(1., raw_peaks[name]),
                                     shared_gain=shared_gain, pcm_peak=float(np.max(np.abs(pcm.astype(np.int32)))) / 32767)
    result = {"version": config["version"], "config_sha256": digest(config_path), "input_evaluation_sha256": digest(evaluation_path),
              "case_id": "test1", "scope": config["scope"], "human_audition_status": "pending", "training_run": False,
              "audits": audits, "variants": variant_reports, "shared_audio_gain": shared_gain, "artifact_sha256": artifact_hashes}
    write_json(out / "analysis.json", result)
    write_json(out / "validation.json", {"midi_cases": len(variants), "audio_cases": len(variants),
        "melody_bass_harmony_timing_and_metadata_exact": True, "single_field_edits_verified": True,
        "thinning_preserves_all_retained_notes": True, "identical_pickup_pcm": True, "all_audio_same_gain_and_length": True,
        "legacy_default_audio_and_original_midi_byte_exact": True, "prior_models_and_audition_files_unchanged": True,
        "new_audio_pcm_frames": waveform_length, "sample_rate": rate, "human_audition_status": "pending"})
    card = lambda name, title, note="": f"<div class='card'><h3>{html.escape(title)}</h3><p>{html.escape(note)}</p><audio controls preload='none' src='{name}.wav'></audio><p><a href='{name}.mid'>MIDI</a></p></div>"
    primary = "v5_medium"
    audit = audits[primary]
    sections = [card("reference", "参照 · 已验收V4", "旋律、低音、和声边界全程相同；本页所有音频使用统一播放增益。"),
        card(primary + "_thin", "A · 只删音：110 → 71", "只移除39个上方音符，保留音的时值和力度不改；局部节奏不等同于V5。"),
        card(primary + "_duration", "B · 只换共同音的时值", f"共同起音{audit['common_upper_onsets']}个，实际改动{audit['duration']['changed_notes']}个音；其他起音、音高与力度不改。"),
        card(primary + "_velocity", "C · 只换共同音的力度", f"实际改动{audit['velocity']['changed_notes']}个音；所有音高、起音与结束时刻不改。")]
    optional = card("standard_reference", "D1 · V4起音分布", "上方音统一为1/4拍、力度45。")
    optional += card(primary + "_thin_standard", "D2 · 删至71音后的V4分布", "同样1/4拍、力度45；与D3总音数一致，局部密度仍可能不同。")
    optional += card(primary + "_standard", "D3 · V5中等档起音分布", "同样1/4拍、力度45；比较的是起音位置和音级分布，不是完整版本的听感。")
    optional += card(primary + "_original", "V5中等档原版复听")
    expanded = "".join(card("expanded_v4_" + suffix, title) for suffix, title in
                       (("thin", "只删音：110 → 86"), ("duration", "只换共同音时值"), ("velocity", "只换共同音力度"),
                        ("thin_standard", "统一时值力度：删音后的V4"), ("standard", "统一时值力度：扩充版起音"), ("original", "扩充版原版复听")))
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>test1 自然度变化排查</title><style>body{max-width:920px;margin:36px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}h3{font-size:19px;margin-bottom:4px}.card{border-top:1px solid #c6c0b3;padding:18px 0}audio{width:100%}details{margin:24px 0;padding:16px;background:#eee9df}summary{cursor:pointer;font-weight:600}p{margin:8px 0;color:#465257}</style><h1>test1：自然度变化排查</h1><p>先听参照与A/B/C，判断哪种变化影响更明显；不明显或不确定也有价值。这些是诊断用的音符编辑，没有训练新模型。</p><p>时值和力度只比较共同起音的子集；未覆盖的音符与因素交互仍可能影响整体听感。统一增益避免各条音频单独调音量。</p>" + "".join(sections) + "<details><summary>进一步比较：数量与起音分布</summary>" + optional + "</details><details><summary>补充对照：仅扩充数据版</summary>" + expanded + "</details></html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    for link in re.findall(r"(?:src|href)='([^']+)'", page):
        assert (out / link).is_file()
    print(json.dumps({"variants": len(variants), "shared_gain": shared_gain,
        "source_statistics": {name: variant_reports[variant]["statistics"] for name, variant in
            (("accepted_v4", "reference"), ("expanded_v4", "expanded_v4_original"), ("v5_medium", "v5_medium_original"))},
        "edits": {name: {"common": a["common_upper_onsets"], "thin_removed": a["thin"]["removed_notes"],
            "duration_changed": a["duration"]["changed_notes"], "duration_capped": a["duration"]["donor_endings_capped_at_base_retrigger"],
            "velocity_changed": a["velocity"]["changed_notes"]} for name, a in audits.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
