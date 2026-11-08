import copy
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rhythm_ablation as ab
from build_harmony_review import synthesize_preview_audio, write_preview_audio


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.labels = [{"start_tick": 0, "end_tick": 96}]
        self.base = [{"pitch": 60, "start_tick": i, "end_tick": i + 12, "velocity": 70, "role": "accompaniment"} for i in (0, 24, 48, 72)]

    def test_duration_swap_keeps_unmatched_notes_and_bounds_next_attack(self):
        donor = [{**self.base[0], "end_tick": 60}]
        original = copy.deepcopy(self.base)
        notes, report = ab.swap_common_field(self.base, donor, self.labels, "end_tick")
        self.assertEqual(notes[0]["end_tick"], 24)
        self.assertEqual(notes[1:], self.base[1:])
        self.assertEqual(self.base, original)
        self.assertEqual(report["donor_endings_capped_at_base_retrigger"], 1)

    def test_velocity_swap_cannot_move_onsets_or_endings(self):
        donor = [{**self.base[1], "end_tick": 80, "velocity": 43}]
        notes, report = ab.swap_common_field(self.base, donor, self.labels, "velocity")
        expected = copy.deepcopy(self.base)
        expected[1]["velocity"] = 43
        self.assertEqual(notes, expected)
        self.assertEqual(report["changed_notes"], 1)

    def test_thinning_removes_events_without_rewriting_survivors(self):
        donor = [self.base[0], self.base[-1]]
        notes, report = ab.thin_to_count(self.base, donor, self.labels)
        self.assertEqual(notes, donor)
        self.assertEqual(report["removed_notes"], 2)

    def test_thinning_cannot_sacrifice_protected_first_chord_tones(self):
        notes = self.base + [{**self.base[0], "pitch": 64}]
        with self.assertRaises(ValueError):
            ab.thin_to_count(notes, [self.base[0]], self.labels)

    def test_duplicate_pitch_onset_cannot_ambiguously_match(self):
        with self.assertRaises(ValueError):
            ab.swap_common_field([self.base[0], self.base[0]], self.base, self.labels, "velocity")

    def test_standard_onset_pair_preserves_attacks_and_clips_at_harmony_end(self):
        notes = ab.standardize([{**self.base[-1], "start_tick": 84}], self.labels, 96)
        self.assertEqual(notes[0]["start_tick"], 84)
        self.assertEqual(notes[0]["end_tick"], 96)
        self.assertEqual(notes[0]["velocity"], 45)


class SharedGainTests(unittest.TestCase):
    @staticmethod
    def midi():
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(0, name="MELODY")
        instrument.notes.append(pretty_midi.Note(80, 72, 0, .5))
        midi.instruments.append(instrument)
        return midi

    def test_shared_gain_is_applied_without_individual_normalization(self):
        midi = self.midi()
        audio, rate = synthesize_preview_audio(midi)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared.wav"
            write_preview_audio(midi, path, gain=.25)
            with wave.open(str(path), "rb") as stream:
                actual = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").reshape(-1, 2)
                self.assertEqual(stream.getframerate(), rate)
            np.testing.assert_array_equal(actual, (audio * .25 * 32767).astype("<i2"))

    def test_clipping_or_invalid_shared_gain_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.wav"
            for gain in (0, float("nan"), 100):
                with self.assertRaises(ValueError):
                    write_preview_audio(self.midi(), path, gain=gain)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
