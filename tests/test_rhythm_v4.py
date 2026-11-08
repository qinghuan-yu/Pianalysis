from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harmony_plan as hp
import rhythm_v4 as rv


class RhythmTests(unittest.TestCase):
    def setUp(self):
        self.label = {**hp.parse_symbol("C:maj"), "start_tick": 288, "end_tick": 672}
        self.ppq = 96
        self.notes = [self.note(36, 288, 672), self.note(60, 312, 360),
                      self.note(64, 336, 384), self.note(67, 384, 432)]

    @staticmethod
    def note(pitch, start, end, role="accompaniment"):
        return {"pitch": pitch, "start_tick": start, "end_tick": end, "velocity": 80, "role": role}

    def test_targets_keep_real_offbeat_onsets(self):
        frames, audit = rv.project_reference(self.label, self.notes, self.ppq)
        self.assertEqual(frames[0]["mask"], 0)
        self.assertEqual(frames[1]["mask"], 1)
        self.assertEqual(frames[1]["durations"][0], 2)
        self.assertEqual(audit["bass_notes_separated"], 1)
        self.assertEqual(audit["target_source"], "projected_real_PIANO_notes")

    def test_carried_upper_note_is_not_a_new_attack(self):
        frames, audit = rv.project_reference(self.label, self.notes + [self.note(64, 280, 300)], self.ppq)
        self.assertEqual(frames[0]["mask"], 0)
        self.assertEqual(audit["carried_upper_notes_excluded"], 1)

    def test_octave_unisons_collapse_with_evidence(self):
        frames, audit = rv.project_reference(self.label, self.notes + [self.note(72, 312, 360)], self.ppq)
        self.assertEqual(frames[1]["mask"], 1)
        self.assertEqual(audit["octave_unisons_collapsed"], 1)

    def test_near_end_attack_does_not_shift_back_a_grid_cell(self):
        frames, audit = rv.project_reference(self.label, self.notes + [self.note(64, 671, 680)], self.ppq)
        self.assertEqual(frames[-1]["mask"], 0)
        self.assertEqual(audit["onsets_rounded_to_next_span_excluded"], 1)

    def test_nonchord_notes_are_reported_not_renamed(self):
        _, audit = rv.project_reference(self.label, self.notes + [self.note(66, 400, 420)], self.ppq)
        self.assertEqual(audit["nonchord_notes_excluded"], 1)

    def test_given_positions_and_source_are_not_loss_targets(self):
        frames, _ = rv.project_reference(self.label, self.notes, self.ppq)
        sample = rv.training_sample(self.label, [], self.ppq, [], frames, "fixture")
        self.assertTrue(all(x == -100 for x in sample["labels"][:sample["prefix_length"]]))
        for token, target in zip(sample["tokens"], sample["labels"]):
            if rv.STEP <= token < rv.STEP + 16:
                self.assertEqual(target, -100)
        self.assertIn(rv.MASK + 1, sample["labels"])

    def test_render_keeps_pickup_phase_and_pitch_mapping(self):
        frames, _ = rv.project_reference(self.label, self.notes, self.ppq)
        notes = rv.render(self.label, [36, 52, 55, 60], frames, self.ppq, 100)
        self.assertIn((60, 312, 360), [(n["pitch"], n["start_tick"], n["end_tick"]) for n in notes])
        self.assertEqual(notes[0]["start_tick"], 288)
        self.assertEqual(notes[0]["end_tick"], 672)

    def test_renderer_rejects_wrong_frozen_harmony(self):
        with self.assertRaises(ValueError):
            rv.render(self.label, [36, 52, 55, 61], rv.empty_frames(16), self.ppq, 100)

    def test_same_lane_retrigger_releases_previous_note(self):
        frames = rv.empty_frames(16)
        for index, duration in [(0, 8), (2, 2)]:
            frames[index] = {"mask": 1, "durations": [duration, None, None], "velocities": [1, None, None]}
        notes = rv.render(self.label, [36, 48, 52, 55], frames, self.ppq, 100)
        roots = [n for n in notes if n["pitch"] == 48]
        self.assertEqual(roots[0]["end_tick"], roots[1]["start_tick"])

    def test_grid_error_and_extended_chords_fail_loudly(self):
        with self.assertRaises(ValueError):
            rv.steps_for({**self.label, "end_tick": 673}, self.ppq)
        with self.assertRaises(ValueError):
            rv.degrees(hp.parse_symbol("C:maj7"))

    def test_empty_prediction_is_not_perfect_recall(self):
        frames, _ = rv.project_reference(self.label, self.notes, self.ppq)
        result = rv.aggregate_metrics([rv.onset_metrics(rv.empty_frames(16), frames)])
        self.assertEqual(result["onset_recall"], 0)
        self.assertEqual(result["onset_f1"], 0)


if __name__ == "__main__":
    unittest.main()
