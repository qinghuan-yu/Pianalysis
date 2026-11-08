import sys
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, piece_to_sample, tokens_to_notes
from harmony_v1 import token_maps
from token_protocol_v1 import parse_events
import build_training_windows_v1 as windows
import generate_from_scratch as generation
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList

VERSION = "pianalysis_tsd_v2"


def piece():
    return {
        "piece_id": "duration-test", "source_file": "synthetic.mid", "duration": 17.0,
        "notes": [
            {"role": "melody", "pitch": 60, "velocity": 80, "start": 0.0, "end": 0.5},
            {"role": "accompaniment", "pitch": 48, "velocity": 3, "start": 0.03, "end": 0.53},
            {"role": "accompaniment", "pitch": 48, "velocity": 90, "start": 0.1, "end": 0.2},
            {"role": "accompaniment", "pitch": 55, "velocity": 70, "start": 16.03, "end": 16.53},
        ],
    }


class DurationEncodingTests(unittest.TestCase):
    def test_explicit_duration_roundtrip_preserves_overlapping_same_pitch_notes(self):
        sample = piece_to_sample(piece(), 10, VERSION)
        notes = tokens_to_notes(sample["target_tokens"], 10, VERSION)
        expected = [n for n in piece()["notes"] if n["role"] == "accompaniment"]
        self.assertEqual(len(notes), len(expected))
        for actual, original in zip(notes, expected):
            for key in ("pitch", "velocity", "role"):
                self.assertEqual(actual[key], original[key])
            for key in ("start", "end"):
                self.assertAlmostEqual(actual[key], original[key], places=6)

    def test_duration_stream_has_no_note_off_events_or_special_parameters(self):
        sample = piece_to_sample(piece(), 10, VERSION)
        tokens = sample["target_tokens"]
        self.assertFalse(set(tokens) & {0, 1, 2, 3, 11, 21})
        events, report = parse_events(tokens, VERSION)
        self.assertEqual(sum(op == 20 for op, _ in events), 3)
        self.assertFalse(any(report.values()))

    def test_duration_and_harmony_namespaces_are_disjoint(self):
        mapping = token_maps(VERSION)
        self.assertEqual(len(mapping), len(set(mapping.values())))
        self.assertIn("DURATION_800", mapping)

    def test_incomplete_note_never_becomes_a_fabricated_sustained_note(self):
        sample = piece_to_sample(piece(), 10, VERSION)
        tokens = sample["source_tokens"][:-1]
        self.assertEqual(tokens_to_notes(tokens, 10, VERSION), [])
        self.assertEqual(parse_events(tokens, VERSION)[1]["incomplete_events"], 1)

    def test_duration_outside_supported_window_fails_instead_of_silent_clipping(self):
        source = piece()
        source["notes"][0]["end"] = 8.01
        with self.assertRaisesRegex(ValueError, "duration.*800"):
            piece_to_sample(source, 10, VERSION)

    def test_reference_windows_keep_original_ids_and_boundaries(self):
        source = piece()
        refs = [{"piece_id": "kept-id", "source_piece_id": source["piece_id"], "slice_id": 7,
                 "start_time": 0.0, "end_time": 0.4}]
        samples = windows.build_reference_windows(source, refs, 10, 1024, VERSION)
        self.assertEqual(samples[0]["piece_id"], "kept-id")
        self.assertEqual(samples[0]["slice_id"], 7)
        self.assertEqual(samples[0]["window_duration"], 0.4)

    def test_training_validates_duration_parameters(self):
        from train_v2 import load_dataset
        sample = piece_to_sample(piece(), 10, VERSION)
        sample["source_tokens"][-1] = 0
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps({"metadata": {"tokenizer_version": VERSION}, "samples": [sample]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid typed"):
                load_dataset(path, 1024)


class PreferInvalidTokens(LogitsProcessor):
    def __init__(self, valid_sequence):
        self.sequence = valid_sequence

    def __call__(self, input_ids, scores):
        i = input_ids.shape[1] - 2
        scores.fill_(-100.0)
        scores[:, self.sequence[min(i, len(self.sequence) - 1)]] = 900
        scores[:, 21 if i == 0 else 3] = 1000
        return scores


class DurationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.mapping = token_maps(VERSION)
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=max(self.mapping.values()) + 1,
            n_positions=32, n_embd=8, n_layer=1, n_head=1, eos_token_id=3, pad_token_id=0))
        self.model.eval()
        self.args = Namespace(max_new_tokens=8, temperature=1.0, top_k=1, top_p=1.0)

    def test_grammar_rejects_note_off_and_eos_inside_a_note(self):
        expected = [20, self.mapping["PITCH_48"], self.mapping["VELOCITY_80"], self.mapping["DURATION_50"], 3]
        tokens, report = generation.generate_target(self.model, [1, 2], self.args, VERSION,
            LogitsProcessorList([PreferInvalidTokens(expected)]))
        self.assertEqual(tokens, expected[:-1])
        self.assertEqual(report["unclosed_notes"], 0)
        self.assertTrue(report["grammar_constraints"])

    def test_budget_cannot_cut_a_note_in_half(self):
        self.args.max_new_tokens = 4
        tokens, report = generation.generate_target(self.model, [1, 2], self.args, VERSION)
        self.assertEqual(report["incomplete_events"], 0)
        self.assertEqual(report["unknown_tokens"], 0)
        self.assertEqual(report["invalid_parameter_events"], 0)

    def test_window_horizon_limits_note_duration(self):
        expected = [20, self.mapping["PITCH_48"], self.mapping["VELOCITY_80"], self.mapping["DURATION_800"], 3]
        tokens, report = generation.generate_target(self.model, [1, 2], self.args, VERSION,
            LogitsProcessorList([PreferInvalidTokens(expected)]), target_duration=0.5)
        notes = tokens_to_notes(tokens, 10, VERSION)
        self.assertTrue(notes)
        self.assertLessEqual(max(n["end"] for n in notes), 0.5)


class WindowedGenerationTests(unittest.TestCase):
    def test_long_source_is_windowed_and_generated_notes_have_global_offsets(self):
        source = piece()
        source["notes"] = [{"pitch": 60, "velocity": 80, "start": 0.0, "end": 9.0, "role": "melody"},
                           {"pitch": 64, "velocity": 80, "start": 16.0, "end": 17.0, "role": "melody"}]
        mapping = token_maps(VERSION)
        model = Namespace(config=Namespace(n_positions=1024, pianalysis_quantum_ms=10))
        args = Namespace(window_seconds=8.0, max_new_tokens=384)
        complete_note = [20, mapping["PITCH_48"], mapping["VELOCITY_80"], mapping["DURATION_50"]]
        with patch.object(generation, "generate_target", return_value=(complete_note, {"stop_reason": "eos"})) as call:
            melody, accompaniment, windows = generation.generate_windowed(model, source, args, VERSION)
        self.assertEqual(melody, source["notes"])
        self.assertEqual([n["start"] for n in accompaniment], [0.0, 8.0, 16.0])
        self.assertEqual([(w["start"], w["end"]) for w in windows], [(0.0, 8.0), (8.0, 16.0), (16.0, 17.0)])
        self.assertEqual(call.call_count, 3)
        self.assertEqual([c.kwargs["target_duration"] for c in call.call_args_list], [8.0, 8.0, 1.0])

    def test_silent_windows_remain_silent_without_hallucinated_filler(self):
        source = piece()
        source["notes"] = [{"pitch": 64, "velocity": 80, "start": 16.0, "end": 17.0, "role": "melody"}]
        model = Namespace(config=Namespace(n_positions=1024, pianalysis_quantum_ms=10))
        args = Namespace(window_seconds=8.0, max_new_tokens=384)
        with patch.object(generation, "generate_target", return_value=([], {"stop_reason": "eos"})) as call:
            _, accompaniment, windows = generation.generate_windowed(model, source, args, VERSION)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(accompaniment, [])
        self.assertEqual(sum(w.get("skipped_silent", False) for w in windows), 2)




if __name__ == "__main__":
    unittest.main()
