import copy
import json
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import harmony_conditioning as hc
import harmony_plan as hp
import build_training_windows_v1 as windows
import generate_from_scratch as generation
from closed_loop_v1 import piece_to_sample
from token_protocol_v1 import TSD_VERSION, TOKEN
from train_v2 import ConditionalMusicDataset, load_dataset


def fixture():
    return {"piece_id": "song", "source_piece_id": "song", "source_file": "fixture.mid", "duration": 4.0,
            "notes": [{"pitch": 74, "velocity": 80, "role": "melody", "start": 0, "end": 4},
                      {"pitch": 55, "velocity": 60, "role": "accompaniment", "start": 0.5, "end": 4}],
            "harmony_plan": {"version": "harmony_plan_v2", "ppq": 480, "duration_ticks": 3840,
                "labels": [{**hp.parse_symbol("G:maj/3"), "start_tick": 480, "end_tick": 3840,
                            "source": "fixture", "review_status": "unreviewed"}]},
            "music_annotation": {"timeline": {"ppq": 480,
                "tempo_changes": [{"tick": 0, "seconds": 0.0, "bpm": 120.0}]}}}


class TimedConditioningTests(unittest.TestCase):
    def test_tempo_changes_and_mid_chord_window_boundaries_keep_real_times(self):
        piece = fixture()
        piece["music_annotation"]["timeline"]["tempo_changes"].append({"tick": 960, "seconds": 1.0, "bpm": 60.0})
        labels = hc.plan_labels_seconds(piece["harmony_plan"], piece["music_annotation"]["timeline"])
        self.assertEqual((labels[0]["start"], labels[0]["end"]), (0.5, 7.0))
        crop = hc.window_labels(labels, 2, 4)
        self.assertEqual((crop[0]["start"], crop[0]["end"], crop[0]["bass_pc"]), (0, 2, 11))
        self.assertEqual(labels[0]["start"], 0.5)

    def test_codec_preserves_unknown_no_chord_pitch_classes_bass_and_subbeat_timing(self):
        labels = [{**hp.parse_symbol(symbol), "start": a, "end": b}
                  for symbol, a, b in [("X", 0, 1.38), ("G:maj/3", 1.38, 3.23),
                                       ("N", 3.23, 4), ("F#:hdim7/b7", 4, 8)]]
        tokens = hc.encode_labels(labels)
        actual = hc.decode_labels(tokens)
        keys = ("start", "end", "status", "root_pc", "bass_pc", "pitch_classes")
        self.assertEqual([[x[k] for k in keys] for x in actual], [[x[k] for k in keys] for x in labels])
        with self.assertRaises(ValueError):
            hc.decode_labels(tokens[:-1])

    def test_window_fills_unknown_gaps_and_only_merges_equal_evidence(self):
        labels = [{**hp.parse_symbol("G:maj"), "start": a, "end": b,
                   "source": "test", "review_status": status}
                  for a, b, status in [(1, 2, "unreviewed"), (2, 3, "unreviewed"), (3, 4, "user_audition_accepted")]]
        result = hc.window_labels(labels, 0, 5)
        self.assertEqual([(x["start"], x["end"], x["status"]) for x in result],
                         [(0, 1, "unknown"), (1, 3, "chord"), (3, 4, "chord"), (4, 5, "unknown")])

    def test_training_window_has_timed_prompt_masked_out_of_target_loss(self):
        sample = windows.make_window_sample(fixture(), 1, 4, 10, 1024, 1, tokenizer_version=TSD_VERSION)[0][0]
        self.assertEqual(sample["harmony_tokenizer_version"], hc.VERSION)
        spans = hc.decode_labels(sample["harmony_tokens"])
        self.assertEqual((spans[0]["start"], spans[0]["end"]), (0, 3))
        item = ConditionalMusicDataset([sample], 1024, 0)[0]
        offset = sample["target_start_index"]
        self.assertTrue((item["labels"][:offset] == -100).all())
        self.assertEqual(item["labels"][offset:offset + len(sample["target_tokens"])].tolist(), sample["target_tokens"])

    def test_training_rejects_mismatched_condition_protocol_and_prompt_payload(self):
        sample = piece_to_sample(fixture(), 10, TSD_VERSION)
        hc.condition_sample(sample, hc.window_labels(hc.plan_labels_seconds(fixture()["harmony_plan"],
            fixture()["music_annotation"]["timeline"]), 0, 4))
        metadata = {"tokenizer_version": TSD_VERSION, "harmony_tokenizer_version": hc.VERSION}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            for field, value in (("harmony_tokenizer_version", "wrong"), ("harmony_tokens", [])):
                bad = copy.deepcopy(sample)
                bad[field] = value
                path.write_text(json.dumps({"metadata": metadata, "samples": [bad]}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "harmony|Harmony"):
                    load_dataset(path, 1024)

    def test_windowed_inference_receives_same_clipped_harmony_as_training(self):
        piece = fixture()
        piece["notes"] = [piece["notes"][0]]
        piece["timed_harmony"] = hc.plan_labels_seconds(piece["harmony_plan"], piece["music_annotation"]["timeline"])
        model = Namespace(config=Namespace(n_positions=1024, pianalysis_quantum_ms=10,
                                           pianalysis_harmony_version=hc.VERSION))
        args = Namespace(window_seconds=2, max_new_tokens=64)
        with patch.object(generation, "generate_target", return_value=([], {})) as call:
            melody, _, reports = generation.generate_windowed(model, piece, args, TSD_VERSION)
        self.assertEqual(melody, piece["notes"])
        self.assertEqual(call.call_count, 2)
        for report, invocation in zip(reports, call.call_args_list):
            expected = hc.encode_labels(hc.window_labels(piece["timed_harmony"], report["start"], report["end"]))
            self.assertEqual(report["harmony_tokens"], expected)
            self.assertEqual(invocation.args[1], [1, *report["source_tokens"], 30, *expected, 2])

    def test_old_model_cannot_silently_accept_new_condition_tokens(self):
        with self.assertRaisesRegex(ValueError, "harmony"):
            hc.validate_model_conditioning(Namespace(config=Namespace()), hc.VERSION)
        hc.validate_model_conditioning(Namespace(config=Namespace(pianalysis_harmony_version=hc.VERSION)), hc.VERSION)

    def test_conditioned_model_rejects_missing_plan(self):
        piece = fixture()
        piece.pop("harmony_plan")
        piece["notes"] = [piece["notes"][0]]
        model = Namespace(config=Namespace(n_positions=1024, pianalysis_quantum_ms=10,
                                           pianalysis_harmony_version=hc.VERSION))
        with patch.object(generation, "generate_target", return_value=([], {})):
            with self.assertRaisesRegex(ValueError, "harmony"):
                generation.generate_windowed(model, piece, Namespace(window_seconds=2, max_new_tokens=64), TSD_VERSION)

    def test_global_plan_cannot_be_misread_by_legacy_harmony_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(fixture()["harmony_plan"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "windowed"):
                generation.load_harmony_tokens(str(path), TSD_VERSION)


if __name__ == "__main__":
    unittest.main()
