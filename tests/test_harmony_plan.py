import copy
from pathlib import Path
import sys
import tempfile
import unittest
import json
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import harmony_plan as hp
import mido
import pretty_midi
import build_harmony_review as review


def make_midi(path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    track.append(mido.Message("note_on", note=72, velocity=80, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=960))
    track.append(mido.Message("note_off", note=72, velocity=0, time=960))
    midi.save(path)
    return pretty_midi.PrettyMIDI(str(path))


def plan_fixture():
    return {"version": "harmony_plan_v2", "ppq": 480, "duration_ticks": 1920,
            "labels": [{"start_tick": 0, "end_tick": 960, "symbol": "C:maj", "status": "chord",
                        "root_pc": 0, "bass_pc": 0, "pitch_classes": [0, 4, 7], "source": "test"},
                       {"start_tick": 960, "end_tick": 1920, "symbol": "G:7/3", "status": "chord",
                        "root_pc": 7, "bass_pc": 11, "pitch_classes": [2, 5, 7, 11], "source": "test"}]}


class TimedHarmonyTests(unittest.TestCase):
    def test_duplicate_metadata_is_separated_from_conflicting_key_signatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000), mido.MetaMessage("key_signature", key="C")]))
            midi.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000), mido.MetaMessage("key_signature", key="D")]))
            path = Path(tmp) / "metadata.mid"
            midi.save(path)
            audit = hp.inspect_midi_metadata(path)
        self.assertEqual(audit["duplicate_events"], 1)
        self.assertEqual(len(audit["conflicts"]), 1)
        self.assertEqual(audit["conflicts"][0]["event_type"], "key_signature")
        self.assertFalse(audit["timing_conflict"])

    def test_harte_inversion_uses_relative_degree_not_midi_pitch(self):
        chord = hp.parse_symbol("F#:maj7/5")
        self.assertEqual(chord["root_pc"], 6)
        self.assertEqual(chord["bass_pc"], 1)
        self.assertEqual(chord["pitch_classes"], [1, 5, 6, 10])

    def test_no_chord_unknown_and_invalid_symbols_are_distinct(self):
        self.assertEqual(hp.parse_symbol("N")["status"], "no_chord")
        self.assertEqual(hp.parse_symbol("X")["status"], "unknown")
        with self.assertRaises(ValueError):
            hp.parse_symbol("invalid-chord")

    def test_sidecar_seconds_convert_through_tempo_map_without_bar_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "tempo.mid")
            path = Path(tmp) / "chord_midi.txt"
            path.write_text("0.25 1.5 C:maj\n1.5 3.0 G:7/3\n", encoding="utf-8")
            labels = hp.read_chord_sidecar(path, midi)
            self.assertEqual([(x["start_tick"], x["end_tick"]) for x in labels], [(240, 1200), (1200, 1920)])

    def test_timeline_preserves_tempo_meter_and_real_downbeat_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            timeline = hp.midi_timeline(make_midi(Path(tmp) / "tempo.mid"))
        self.assertEqual(timeline["ppq"], 480)
        self.assertEqual([x["tick"] for x in timeline["tempo_changes"]], [0, 960])
        self.assertEqual(timeline["time_signatures"][0]["numerator"], 3)
        self.assertIn(1440, [x["tick"] for x in timeline["downbeats"]])

    def test_slicing_clips_and_offsets_both_chords_without_mutating_original(self):
        plan = plan_fixture()
        original = copy.deepcopy(plan)
        sliced = hp.slice_plan(plan, 480, 1440)
        self.assertEqual([(x["start_tick"], x["end_tick"]) for x in sliced["labels"]], [(0, 480), (480, 960)])
        self.assertEqual(sliced["origin_tick"], 480)
        self.assertEqual(plan, original)

    def test_overlap_and_symbol_pitch_disagreement_fail_validation(self):
        plan = plan_fixture()
        self.assertIsNone(hp.validate_plan(plan))
        plan["labels"][1]["start_tick"] = 900
        with self.assertRaisesRegex(ValueError, "overlap"):
            hp.validate_plan(plan)
        plan = plan_fixture()
        plan["labels"][0]["bass_pc"] = 2
        with self.assertRaisesRegex(ValueError, "symbol"):
            hp.validate_plan(plan)


class BlockChordTests(unittest.TestCase):
    def test_overlapping_same_pitch_melody_notes_keep_independent_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "source.mid")
            notes = [{"pitch": 72, "velocity": 80, "start": 0.0, "end": 3.0},
                     {"pitch": 72, "velocity": 60, "start": 1.0, "end": 2.0}]
            path = Path(tmp) / "blocks.mid"
            hp.write_block_midi(midi, notes, plan_fixture(), path)
            result = pretty_midi.PrettyMIDI(str(path))
            melody = sorted((n.start, n.end, n.velocity) for i in result.instruments if i.name == "MELODY" for n in i.notes)
        self.assertEqual(melody, [(0.0, 3.0, 80), (1.0, 2.0, 60)])

    def test_non_round_tempo_microseconds_are_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = mido.MidiFile(ticks_per_beat=480)
            source.tracks.append(mido.MidiTrack([
                mido.MetaMessage("set_tempo", tempo=437956, time=0),
                mido.Message("note_on", note=72, velocity=80, time=0),
                mido.Message("note_off", note=72, velocity=0, time=1920)]))
            source.save(Path(tmp) / "source.mid")
            midi = pretty_midi.PrettyMIDI(str(Path(tmp) / "source.mid"))
            path = Path(tmp) / "blocks.mid"
            hp.write_block_midi(midi, [], plan_fixture(), path)
            tempi = [message.tempo for track in mido.MidiFile(path).tracks for message in track if message.type == "set_tempo"]
        self.assertEqual(tempi, [437956])

    def test_columns_use_exact_chord_classes_and_correct_inversion_bass(self):
        plan = plan_fixture()
        notes = hp.block_chord_notes(plan)
        for label in plan["labels"]:
            column = [n for n in notes if n["start_tick"] == label["start_tick"]]
            self.assertTrue(column)
            self.assertEqual({n["pitch"] % 12 for n in column}, set(label["pitch_classes"]))
            self.assertEqual(min(n["pitch"] for n in column) % 12, label["bass_pc"])
            self.assertEqual({n["end_tick"] for n in column}, {label["end_tick"]})

    def test_unknown_and_no_chord_intervals_stay_silent(self):
        plan = plan_fixture()
        for label, symbol in zip(plan["labels"], ("X", "N")):
            label.update(hp.parse_symbol(symbol))
        self.assertEqual(hp.block_chord_notes(plan), [])

    def test_export_preserves_melody_tempo_meter_and_inversion_after_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "original.mid")
            original = midi.instruments[0].notes[0]
            melody = [{"pitch": original.pitch, "start": original.start, "end": original.end, "velocity": original.velocity}]
            path = Path(tmp) / "block.mid"
            hp.write_block_midi(midi, melody, plan_fixture(), path)
            exported = pretty_midi.PrettyMIDI(str(path))
            self.assertEqual(list(exported.get_tempo_changes()[1]), [120.0, 60.0])
            self.assertEqual(exported.time_signature_changes[0].numerator, 3)
            note = exported.instruments[0].notes[0]
            self.assertEqual((note.pitch, note.start, note.end, note.velocity), (72, 0.0, 3.0, 80))
            second = [n for n in exported.instruments[1].notes if abs(n.start - 1.0) < 1e-6]
            self.assertEqual(min(n.pitch for n in second) % 12, 11)
            self.assertTrue(all(abs(n.end - 3.0) < 1e-6 for n in second))


class AnnotationTests(unittest.TestCase):
    def test_full_accompaniment_c_major_is_labeled_from_accompaniment_not_melody(self):
        midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
        instrument = pretty_midi.Instrument(0)
        midi.instruments.append(instrument)
        notes = [{"pitch": p, "start": 0.0, "end": 2.0, "velocity": 70, "role": "accompaniment"} for p in (48, 60, 64, 67)]
        notes.append({"pitch": 66, "start": 0.0, "end": 2.0, "velocity": 100, "role": "melody"})
        instrument.notes = [pretty_midi.Note(n["velocity"], n["pitch"], n["start"], n["end"]) for n in notes]
        plan = hp.annotate_accompaniment({"piece_id": "fixture", "notes": notes}, midi)
        self.assertTrue(plan["labels"])
        self.assertTrue(all(n["root_pc"] == 0 and n["pitch_classes"] == [0, 4, 7] for n in plan["labels"]))
        self.assertEqual(plan["evidence_scope"], "reference_accompaniment")

    def test_melody_only_is_not_silently_used_as_reference_accompaniment(self):
        midi = pretty_midi.PrettyMIDI()
        with self.assertRaisesRegex(ValueError, "accompaniment"):
            hp.annotate_accompaniment({"notes": [{"role": "melody", "pitch": 60, "start": 0, "end": 1}]}, midi)

    def test_sidecar_gaps_remain_unknown_and_tail_is_clipped_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "source.mid")
            path = Path(tmp) / "chord_midi.txt"
            path.write_text("0.25 1.0 C:maj\n1.5 4.0 G:7\n", encoding="utf-8")
            plan = hp.complete_sidecar_plan({"piece_id": "fixture"}, midi, path)
        self.assertEqual([n["symbol"] for n in plan["labels"]], ["X", "C:maj", "X", "G:7"])
        self.assertEqual(plan["labels"][-1]["end_tick"], 1920)
        self.assertIn("clipped_to_midi_end", plan["labels"][-1]["review_reasons"])
        self.assertTrue(all(n["review_status"] == "unreviewed" for n in plan["labels"]))

    def test_music_annotation_preserves_roles_and_records_rhythm_and_uncertain_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "source.mid")
            path = Path(tmp) / "chord_midi.txt"
            path.write_text("0 3 C:maj\n", encoding="utf-8")
            piece = {"piece_id": "fixture", "notes": [{"id": 1, "role": "melody", "pitch": 72,
                                                       "start": 0, "end": 3, "velocity": 80}]}
            original = copy.deepcopy(piece)
            result = hp.annotate_music(piece, midi, chord_path=path)
        self.assertEqual(piece, original)
        self.assertEqual(result["notes"][0]["role"], "melody")
        self.assertEqual(result["notes"][0]["rhythm"]["end_tick"], 1920)
        self.assertEqual(result["music_annotation"]["key_estimate"]["status"], "candidate_not_verified")
        self.assertEqual(result["harmony_plan"]["labels"][0]["symbol"], "C:maj")


class ReviewArtifactTests(unittest.TestCase):
    def test_user_progression_preserves_d_major_4563_and_is_not_claimed_fully_reviewed(self):
        midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
        instrument = pretty_midi.Instrument(0)
        instrument.notes = [pretty_midi.Note(80, 74, 0, 8)]
        midi.instruments = [instrument]
        config = {"source": "user_specified_progression", "key_context": "D:maj",
                  "review_status": "progression_confirmed_timing_pending", "candidates": [
                      {"id": "user", "events": [[0, 4, "G:maj"], [4, 8, "A:maj"], [8, 12, "B:min"], [12, 16, "F#:min"]]}]}
        plan = review.plans_from_config(midi, config)[0]
        self.assertEqual([x["root_pc"] for x in plan["labels"]], [7, 9, 11, 6])
        self.assertEqual([x["quality"] for x in plan["labels"]], ["maj", "maj", "min", "min"])
        self.assertEqual(plan["evidence_scope"], "user_specified_progression")
        self.assertEqual(plan["key_context"], "D:maj")
        self.assertEqual(plan["review_status"], "progression_confirmed_timing_pending")

    def test_annotation_and_block_midi_are_built_and_verified_from_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "source.mid")
            path = Path(tmp) / "chords.txt"
            path.write_text("0 3 C:maj\n", encoding="utf-8")
            piece = {"piece_id": "fixture", "notes": [{"id": 1, "role": "melody", "pitch": 72,
                                                       "start": 0, "end": 3, "velocity": 80}]}
            row, _, _ = review.build_piece(piece, midi, Path(tmp) / "out", chord_path=path)
            saved = json.loads((Path(tmp) / "out/annotated_notes/fixture.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["harmony_plan"]["labels"][0]["end_tick"], 1920)
            self.assertTrue(row["render_validation"]["melody_roundtrip_exact"])
            self.assertTrue(row["render_validation"]["block_chords_roundtrip_exact"])

    def test_audio_excerpt_is_nonempty_pcm_with_requested_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi = make_midi(Path(tmp) / "source.mid")
            output = Path(tmp) / "preview.wav"
            review.write_preview_audio(midi, output, start=0.5, end=1.5)
            with wave.open(str(output)) as audio:
                self.assertEqual(audio.getnchannels(), 2)
                self.assertEqual(audio.getnframes(), audio.getframerate())
                self.assertTrue(any(audio.readframes(audio.getnframes())))


if __name__ == "__main__":
    unittest.main()
