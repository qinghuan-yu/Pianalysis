"""Audition an octave shift of named accompaniment without altering the melody."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import wave

import mido
import numpy as np
import pretty_midi
from build_harmony_review import write_json, write_preview_audio, synthesize_preview_audio


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-midi", default="data/trusted_harmony_v4/evaluation/test1_learned.mid")
    parser.add_argument("--octaves", type=int, default=1)
    parser.add_argument("--out-dir", default="data/register_review_v4")
    args = parser.parse_args()
    source_path, out = Path(args.input_midi), Path(args.out_dir)
    if out.exists():
        raise ValueError("Choose a fresh audition directory")
    source_hash = digest(source_path)
    original = mido.MidiFile(str(source_path))
    changed = copy.deepcopy(original)
    shift = args.octaves * 12
    shifted_messages = 0
    for track in changed.tracks:
        if track.name != "ACCOMPANIMENT":
            continue
        for msg in track:
            if msg.type in ("note_on", "note_off", "polytouch"):
                if not 0 <= msg.note + shift <= 127:
                    raise ValueError("Requested octave shift exceeds MIDI pitch range; no clipping applied")
                msg.note += shift
                shifted_messages += 1
    if not shifted_messages:
        raise ValueError("No named ACCOMPANIMENT note track")
    out.mkdir(parents=True)
    (out / "original.mid").write_bytes(source_path.read_bytes())
    changed.save(str(out / "shifted.mid"))
    exported = mido.MidiFile(str(out / "shifted.mid"))
    assert original.type == exported.type and original.ticks_per_beat == exported.ticks_per_beat
    assert len(original.tracks) == len(exported.tracks)
    for old_track, new_track in zip(original.tracks, exported.tracks):
        assert len(old_track) == len(new_track)
        for old, new in zip(old_track, new_track):
            expected = old.dict()
            if old_track.name == "ACCOMPANIMENT" and old.type in ("note_on", "note_off", "polytouch"):
                expected["note"] += shift
            assert new.dict() == expected
    midis = {name: pretty_midi.PrettyMIDI(str(out / f"{name}.mid")) for name in ("original", "shifted")}
    horizon = midis["original"].get_end_time()
    assert midis["shifted"].get_end_time() == horizon
    peaks = {name: float(np.max(np.abs(synthesize_preview_audio(midi, end=horizon)[0]))) for name, midi in midis.items()}
    gain = .9 / max(1., max(peaks.values()))
    reports, hashes, pickup = {}, {}, None
    for name, midi in midis.items():
        melody = [n for i in midi.instruments if i.name == "MELODY" for n in i.notes]
        accompaniment = [n for i in midi.instruments if i.name == "ACCOMPANIMENT" for n in i.notes]
        overlaps = [(a, m) for a in accompaniment for m in melody if a.start < m.end and a.end > m.start]
        reports[name] = {"accompaniment_notes": len(accompaniment), "accompaniment_pitch_range": [min(n.pitch for n in accompaniment), max(n.pitch for n in accompaniment)],
            "melody_pitch_range": [min(n.pitch for n in melody), max(n.pitch for n in melody)] if melody else None,
            "overlapping_melody_pairs_at_or_above_melody": sum(a.pitch >= m.pitch for a, m in overlaps),
            "minimum_pitch_clearance_during_overlap": min((m.pitch - a.pitch for a, m in overlaps), default=None),
            "raw_audio_peak": peaks[name]}
        write_preview_audio(midi, out / f"{name}.wav", end=horizon, gain=gain)
        with wave.open(str(out / f"{name}.wav"), "rb") as stream:
            pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
            assert np.any(pcm) and np.max(np.abs(pcm.astype(np.int32))) < 32767
            intro = pcm[:round(min(n.start for n in accompaniment) * stream.getframerate())]
            if pickup is None:
                pickup = intro.copy()
            assert np.array_equal(intro, pickup)
        for suffix in ("mid", "wav"):
            hashes[f"{name}.{suffix}"] = digest(out / f"{name}.{suffix}")
    assert digest(source_path) == source_hash
    result = {"version": "accompaniment_register_preview_v1", "source_file": source_path.as_posix(), "source_sha256": source_hash,
        "octaves": args.octaves, "semitones": shift, "includes_bass": True, "melody_shifted": False,
        "only_named_accompaniment_pitch_messages_changed": True, "shared_audio_gain": gain,
        "rhythm_duration_velocity_and_metadata_unchanged": True, "pickup_pcm_identical": True,
        "cases": reports, "artifact_sha256": hashes, "human_audition_status": "pending"}
    write_json(out / "result.json", result)
    title = f"伴奏音区对照：{args.octaves:+d} 个八度"
    page = f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{title}</title><style>body{{max-width:860px;margin:40px auto;padding:20px;background:#f7f4ed;color:#263136;font:16px/1.7 system-ui}}audio{{width:100%}}section{{border-top:1px solid #bbb;padding:20px 0}}</style><h1>{title}</h1><p>沿用已验收V4的节奏与力度；整个伴奏（含低音）移调，旋律、弱起和换和弦时刻保留。两版使用同一播放增益，尚待试听。</p><section><h2>原音区</h2><audio controls preload='none' src='original.wav'></audio><p><a href='original.mid'>MIDI</a></p></section><section><h2>伴奏升八度</h2><audio controls preload='none' src='shifted.wav'></audio><p><a href='shifted.mid'>MIDI</a></p></section></html>"
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
