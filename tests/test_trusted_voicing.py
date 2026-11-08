import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import harmony_plan as hp
import local_voicing as lv
from build_trusted_harmony import evidence, check_hashes
from train_local_voicing import batch, exercise, validate_dataset, summarize


class TrustedVoicingTests(unittest.TestCase):
    def setUp(self):
        self.label = {**hp.parse_symbol("G:maj"), "start_tick": 288, "end_tick": 672}
        self.melody = {"min": 74, "max": 78, "pcs": [2, 6], "velocity": 100}

    def test_source_and_padding_are_not_targets(self):
        sample = exercise(self.label, self.melody, lv.INITIAL, "fixture")
        longer = copy.deepcopy(sample)
        longer["sequence"].insert(1, lv.CHORD)
        longer["target_start"] += 1
        result = batch([sample, longer], "cpu")
        self.assertTrue((result["labels"][0, :sample["target_start"]] == -100).all())
        self.assertEqual(int((result["labels"][0] != -100).sum()), 5)
        self.assertEqual(int(result["labels"][0, -1]), -100)

    def test_unknown_does_not_become_a_chord(self):
        with self.assertRaises(ValueError):
            lv.prompt(hp.parse_symbol("X"), self.melody, lv.INITIAL)

    def test_empty_known_harmony_is_not_reported_as_perfect(self):
        result = summarize([])
        self.assertEqual(result["segments"], 0)
        self.assertIsNone(result["bass_correct"])

    def test_inversion_has_independent_bass(self):
        label = {**self.label, **hp.parse_symbol("G:maj/3")}
        result = lv.teacher_voicing(label, self.melody, lv.INITIAL)
        self.assertEqual(result[0] % 12, 11)
        self.assertEqual(set(p % 12 for p in result[1:]), {7, 11, 2})
        self.assertTrue(all(v for k, v in lv.assess(result, label, self.melody).items() if k != "chord_tone_ratio"))

    def test_held_melody_limits_the_new_chord(self):
        melody = [{"pitch": 60, "start_tick": 0, "end_tick": 400},
                  {"pitch": 79, "start_tick": 400, "end_tick": 672}]
        summary = lv.melody_summary(melody, self.label)
        self.assertEqual(summary["min"], 60)
        target = lv.teacher_voicing(self.label, summary, lv.INITIAL)
        self.assertLessEqual(max(target), 57)

    def test_arpeggio_preserves_pickup_phase_and_chord_boundary(self):
        notes = lv.schedule(self.label, [43, 55, 59, 62], 96, "arpeggio")
        self.assertEqual(min(n["start_tick"] for n in notes), 288)
        self.assertEqual([n["start_tick"] for n in notes[1:]], list(range(288, 672, 48)))
        self.assertEqual(max(n["end_tick"] for n in notes), 672)
        self.assertTrue(all(n["end_tick"] <= 672 for n in notes))

    def test_sustained_bass_is_valid_without_reattack(self):
        midi = hp.pretty_midi.PrettyMIDI(initial_tempo=60, resolution=480)
        label = {**hp.parse_symbol("C:min"), "start_tick": 480, "end_tick": 960}
        piece = {"notes": [{"id": 0, "pitch": 48, "start": 0., "end": 2., "role": "accompaniment"},
                            {"id": 1, "pitch": 63, "start": 1., "end": 2., "role": "accompaniment"},
                            {"id": 2, "pitch": 67, "start": 1., "end": 2., "role": "accompaniment"}]}
        proof = evidence(piece, midi, label, [0, 1, 2])
        self.assertTrue(all(proof["checks"].values()))
        self.assertEqual(proof["bass_anchor_kind"], "sustained")

    def test_power_chord_does_not_prove_major_quality(self):
        midi = hp.pretty_midi.PrettyMIDI(initial_tempo=60, resolution=480)
        label = {**hp.parse_symbol("C:maj"), "start_tick": 0, "end_tick": 480}
        piece = {"notes": [{"id": i, "pitch": p, "start": 0., "end": 1., "role": "accompaniment"} for i, p in enumerate([48, 55])]}
        self.assertFalse(evidence(piece, midi, label, [0, 1])["checks"]["complete_triad_evidence"])

    def test_source_change_invalidates_review(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "x.mid").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                check_hashes(Path(folder), {"x.mid": "0" * 64})

    def test_unreviewed_data_cannot_train(self):
        with self.assertRaises(ValueError):
            validate_dataset({"review_status": "unreviewed", "training_gate": {"ok": True}})

    def test_song_leakage_cannot_train(self):
        data = {"review_status": "assistant_symbolic_reviewed", "training_gate": {"ok": True}, "pieces": [
            {"song": "001", "split": split, "source_file": "x.mid", "source_hashes": {}, "labels": []}
            for split in ("train", "validation")]}
        with patch("train_local_voicing.check_hashes"):
            with self.assertRaises(ValueError):
                validate_dataset(data)

    def test_export_retains_tempo_ppq_and_overlapping_notes(self):
        source = hp.pretty_midi.PrettyMIDI(initial_tempo=130, resolution=96)
        melody = [{"pitch": 74, "velocity": 100, "start_tick": 0, "end_tick": 96},
                  {"pitch": 74, "velocity": 90, "start_tick": 48, "end_tick": 144}]
        with tempfile.TemporaryDirectory() as folder:
            result = lv.write_midi(source, melody, [], Path(folder, "clip.mid"), origin=24, end=144)
        self.assertEqual(result.resolution, 96)
        self.assertAlmostEqual(result.get_tempo_changes()[1][0], 130, places=3)
        actual = sorted((n.pitch, int(result.time_to_tick(n.start)), int(result.time_to_tick(n.end))) for i in result.instruments for n in i.notes)
        self.assertEqual(actual, [(74, 0, 72), (74, 24, 120)])


if __name__ == "__main__":
    unittest.main()
