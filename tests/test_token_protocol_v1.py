import sys
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop_v1 import TOKEN, piece_to_sample, tokens_to_notes
from generate_from_scratch import first_eos_position
from harmony_v1 import harmony_tokens_for_labels, token_maps
import generate_from_scratch as generation
import build_training_windows_v1 as windows
import torch
import pretty_midi
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList


TYPED_VERSION = "pianalysis_typed_v1_1"


def example_piece():
    return {
        "piece_id": "time-three",
        "source_file": "synthetic.mid",
        "duration": 16.04,
        "notes": [
            {"role": "melody", "pitch": 60, "velocity": 80, "start": 0.0, "end": 0.03},
            {"role": "accompaniment", "pitch": 3, "velocity": 3, "start": 0.03, "end": 0.06},
            {"role": "accompaniment", "pitch": 60, "velocity": 127, "start": 16.03, "end": 16.04},
        ],
    }


class EventProtocolTests(unittest.TestCase):
    def test_legacy_time_pitch_and_velocity_three_do_not_end_generation(self):
        tokens = [4, 3, 20, 3, 3, 4, 1, 21, 3, 3]
        self.assertEqual(first_eos_position(tokens, TOKEN["EOS"]), 9)

    def test_legacy_numeric_three_without_end_marker_is_not_eos(self):
        self.assertIsNone(first_eos_position([4, 3, 20, 60, 3], TOKEN["EOS"]))

    def test_typed_parameters_never_alias_control_tokens(self):
        sample = piece_to_sample(example_piece(), 10, tokenizer_version=TYPED_VERSION)
        target = sample["target_tokens"]
        self.assertFalse(set(target) & {TOKEN[k] for k in ("PAD", "BOS", "SEP", "EOS")})
        self.assertEqual(sample["tokenizer_version"], TYPED_VERSION)

    def test_typed_roundtrip_preserves_notes_across_long_silence(self):
        piece = example_piece()
        sample = piece_to_sample(piece, 10, tokenizer_version=TYPED_VERSION)
        for role, field in (("melody", "source_tokens"), ("accompaniment", "target_tokens")):
            notes = tokens_to_notes(sample[field], 10, tokenizer_version=TYPED_VERSION)
            expected = [n for n in piece["notes"] if n["role"] == role]
            self.assertEqual(len(notes), len(expected))
            for actual, original in zip(notes, expected):
                for key in ("pitch", "velocity", "role"):
                    self.assertEqual(actual[key], original[key])
                for key in ("start", "end"):
                    self.assertAlmostEqual(actual[key], original[key], places=6)

    def test_typed_harmony_has_unique_ids_even_for_high_bass_and_unknown(self):
        mapping = token_maps(tokenizer_version=TYPED_VERSION)
        self.assertEqual(len(mapping), len(set(mapping.values())))
        labels = [{"root": 0, "quality": "maj", "bass": 127, "confidence": 1.0}]
        tokens = harmony_tokens_for_labels(labels, tokenizer_version=TYPED_VERSION)
        self.assertEqual(tokens, [mapping["BAR"], mapping["CHORD_C:maj"], mapping["BASS_127"], mapping["CONF_9"]])

    def test_legacy_encoding_remains_compatible_with_existing_checkpoints(self):
        sample = piece_to_sample(example_piece(), 10)
        self.assertEqual(sample["source_tokens"], [10, 60, 80, 4, 3, 11, 60])

    def test_training_rejects_legacy_sample_in_typed_dataset(self):
        from train_v2 import load_dataset
        data = {"metadata": {"tokenizer_version": TYPED_VERSION}, "samples": [piece_to_sample(example_piece(), 10)]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer_version"):
                load_dataset(path, 1024)

    def test_training_rejects_raw_values_mislabeled_as_typed(self):
        from train_v2 import load_dataset
        sample = piece_to_sample(example_piece(), 10)
        sample["tokenizer_version"] = TYPED_VERSION
        data = {"metadata": {"tokenizer_version": TYPED_VERSION}, "samples": [sample]}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid typed"):
                load_dataset(path, 1024)

    def test_window_builder_carries_protocol_and_preview_aligns_both_roles(self):
        piece = example_piece()
        samples, rejects = windows.build_windows_for_piece(piece, 10, 8.0, 0.0, 1024, 1.0, tokenizer_version=TYPED_VERSION)
        self.assertTrue(samples)
        self.assertEqual(samples[0]["tokenizer_version"], TYPED_VERSION)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            windows.write_preview_midis(samples[:1], Path(directory), 10, 1)
            midi = pretty_midi.PrettyMIDI(str(Path(directory) / "preview_midi" / f"{samples[0]['piece_id']}.mid"))
            accompaniment = next(i for i in midi.instruments if i.name == "Accompaniment")
            self.assertAlmostEqual(accompaniment.notes[0].start, 0.03, delta=0.003)


class ScheduledTokens(LogitsProcessor):
    def __init__(self, sequence, prompt_length=2):
        self.sequence = sequence
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        scores.fill_(-float("inf"))
        scores[:, self.sequence[input_ids.shape[1] - self.prompt_length]] = 0
        return scores


class GenerationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=1378, n_positions=32, n_ctx=32, n_embd=8, n_layer=1, n_head=1, eos_token_id=3, pad_token_id=0))
        self.model.eval()
        self.args = Namespace(max_new_tokens=20, temperature=1.0, top_k=0, top_p=1.0)

    def run_forced(self, sequence, version="pianalysis_numeric_v1"):
        return generation.generate_target(self.model, [1, 2], self.args, version, LogitsProcessorList([ScheduledTokens(sequence)]))

    def test_huggingface_generation_continues_after_time_three_until_real_eos(self):
        sequence = [4, 3, 20, 60, 3, 4, 1, 21, 60, 3]
        tokens, report = self.run_forced(sequence)
        self.assertEqual(tokens, sequence[:-1])
        self.assertEqual(report["stop_reason"], "eos")

    def test_token_budget_is_reported_separately_from_eos(self):
        self.args.max_new_tokens = 5
        tokens, report = self.run_forced([20, 60, 80, 4, 20])
        self.assertEqual(len(tokens), 5)
        self.assertEqual(report["stop_reason"], "max_new_tokens")
        self.assertEqual(report["unclosed_notes"], 1)

    def test_context_limit_caps_budget_before_model_overflow(self):
        self.args.max_new_tokens = 40
        _, report = self.run_forced([4, 1] * 15)
        self.assertEqual(report["effective_max_new_tokens"], 30)
        self.assertEqual(report["stop_reason"], "context_limit")

    def test_wrong_parameter_type_is_reported_and_next_event_recovers(self):
        from token_protocol_v1 import parameter_token
        pitch = parameter_token("pitch", 60)
        velocity = parameter_token("velocity", 80)
        sequence = [20, pitch, TOKEN["TIME"], parameter_token("time", 3), 20, pitch, velocity, 3]
        _, report = self.run_forced(sequence, TYPED_VERSION)
        self.assertEqual(report["invalid_parameter_events"], 1)
        self.assertEqual(report["unclosed_notes"], 1)

    def test_source_protocol_mismatch_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            generation.validate_source_metadata({"tokenizer_version": "pianalysis_numeric_v1", "quantum_ms": 10}, TYPED_VERSION, 10)

    def test_source_time_quantum_mismatch_fails_before_inference(self):
        with self.assertRaisesRegex(ValueError, "quantum"):
            generation.validate_source_metadata({"tokenizer_version": TYPED_VERSION, "quantum_ms": 20}, TYPED_VERSION, 10)

    def test_typed_eos_interrupting_an_event_reports_eos_and_incomplete_event(self):
        tokens, report = self.run_forced([20, 3], TYPED_VERSION)
        self.assertEqual(tokens, [20])
        self.assertEqual(report["stop_reason"], "eos")
        self.assertEqual(report["incomplete_events"], 1)

    def test_saved_checkpoint_identifies_protocol_without_parent_metadata(self):
        from train_v2 import create_model
        args = Namespace(max_length=32, n_embd=8, n_layer=1, n_head=1, dropout=0.0)
        model = create_model(args, 1378, TOKEN, tokenizer_version=TYPED_VERSION, quantum_ms=10)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            model.save_pretrained(directory)
            restored = GPT2LMHeadModel.from_pretrained(directory)
            self.assertEqual(generation.model_protocol(Path(directory), restored), (TYPED_VERSION, 10))


if __name__ == "__main__":
    unittest.main()
