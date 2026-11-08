from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harmony_plan as hp
import rhythm_v4 as v4
import rhythm_v5 as v5


class ControlledRhythmTests(unittest.TestCase):
    def setUp(self):
        self.label = {**hp.parse_symbol("C:maj"), "start_tick": 288, "end_tick": 672}
        self.frames = v4.empty_frames(16)
        for step in (0, 3, 8):
            self.frames[step] = {"mask": 1, "durations": [2, None, None], "velocities": [1, None, None]}

    def test_future_onsets_precede_duration_targets(self):
        sample = v5.training_sample(self.label, [], 96, [], self.frames, "test")
        phase = sample["tokens"].index(v5.DETAILS_PHASE)
        self.assertEqual(sum(v4.MASK <= t < v4.MASK + 8 for t in sample["tokens"][sample["prefix_length"]:phase]), 16)
        self.assertFalse(any(v4.DURATION <= t < v4.DURATION + 16 for t in sample["tokens"][:phase]))
        for token, target in zip(sample["tokens"], sample["labels"]):
            if token in (v5.ONSETS_PHASE, v5.DETAILS_PHASE) or v4.STEP <= token < v4.STEP + 16:
                self.assertEqual(target, -100)

    def test_next_attack_bounds_duration_without_posthoc_trimming(self):
        self.assertEqual(v5.duration_limit(self.frames, 0, 0), 3)
        self.frames[0]["durations"][0] = 3
        notes = v4.render(self.label, [36, 48, 52, 55], self.frames, 96, 100)
        roots = [n for n in notes if n["pitch"] == 48]
        self.assertEqual(roots[0]["end_tick"] - roots[0]["start_tick"], 72)
        self.assertEqual(roots[0]["end_tick"], roots[1]["start_tick"])

    def test_reference_overlap_is_rejected(self):
        self.frames[0]["durations"][0] = 4
        with self.assertRaises(ValueError):
            v5.training_sample(self.label, [], 96, [], self.frames, "test")

    def test_silent_lane_has_no_duration(self):
        with self.assertRaises(ValueError):
            v5.duration_limit(self.frames, 0, 1)

    def test_inference_profile_is_explicit_and_no_target_is_required(self):
        sparse = v5.prompt(self.label, [], 96, [], "sparse")
        dense = v5.prompt(self.label, [], 96, [], "dense")
        self.assertEqual(sparse[:-2], dense[:-2])
        self.assertNotEqual(sparse[-2], dense[-2])
        with self.assertRaises(ValueError):
            v5.prompt(self.label, [], 96, [], "automatic_from_test_target")

    def test_density_labels_use_quarters_not_seconds(self):
        frames = v4.empty_frames(4)
        frames[0]["mask"] = 3
        self.assertEqual(v5.profile_for(frames), "sparse")
        frames[0]["mask"] = 7
        self.assertEqual(v5.profile_for(frames), "medium")
        frames[1]["mask"] = 1
        self.assertEqual(v5.profile_for(frames), "dense")

    def test_no_new_attacks_does_not_fabricate_duration_targets(self):
        sample = v5.training_sample(self.label, [], 96, [], v4.empty_frames(16), "silent")
        self.assertFalse(any(v4.DURATION <= t < v4.DURATION + 16 for t in sample["tokens"]))
        self.assertTrue(all(x == -100 for x in sample["labels"][:sample["prefix_length"]]))

    def test_decoder_bounds_overlong_model_preference_before_export(self):
        class LongDurationModel:
            config = SimpleNamespace(pianalysis_protocol=v5.VERSION)
            device = "cpu"

            def __call__(self, input_ids):
                tokens = input_ids[0].tolist()
                logits = torch.zeros(1, len(tokens), v5.VOCAB)
                if v5.DETAILS_PHASE not in tokens:
                    step = tokens[-1] - v4.STEP
                    logits[0, -1, v4.MASK + (1 if step in (0, 3, 8) else 0)] = 100
                else:
                    logits[0, -1, v4.DURATION:v4.DURATION + 16] = torch.arange(16)
                return SimpleNamespace(logits=logits)

        frames = v5.predict(LongDurationModel(), self.label, [], 96, [], "medium")
        self.assertEqual([frames[s]["durations"][0] for s in (0, 3, 8)], [3, 5, 8])
        notes = v4.render(self.label, [36, 48, 52, 55], frames, 96, 100)
        self.assertEqual([n["end_tick"] - n["start_tick"] for n in notes[1:]], [72, 120, 192])


if __name__ == "__main__":
    unittest.main()
