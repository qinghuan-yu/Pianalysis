import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class Pop909ImportTests(unittest.TestCase):
    def make_midi(self, path, names=("MELODY", "BRIDGE", "PIANO")):
        midi = pretty_midi.PrettyMIDI(initial_tempo=90, resolution=480)
        midi.time_signature_changes = [pretty_midi.TimeSignature(3, 4, 0)]
        for index, name in enumerate(names):
            track = pretty_midi.Instrument(program=0, name=name)
            track.notes.append(pretty_midi.Note(80, 72 - index * 12, 0, 1))
            midi.instruments.append(track)
        path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(path))

    def test_track_names_preserve_roles_even_when_every_track_is_piano(self):
        importer = importlib.import_module("import_pop909")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            path = root / "001" / "001.mid"
            self.make_midi(path)
            result = importer.load_piece(path, root)
            self.assertCountEqual([(n["pitch"], n["role"]) for n in result["notes"]], [(72, "melody"), (48, "accompaniment")])
            self.assertEqual(result["import_info"]["excluded_bridge_notes"], 1)
            self.assertEqual(result["midi_timing"]["resolution"], 480)
            self.assertEqual(result["midi_timing"]["time_signatures"][0]["numerator"], 3)
            self.assertAlmostEqual(result["midi_timing"]["tempo_changes"][0]["bpm"], 90, places=3)

    def test_versions_share_song_identity_for_piece_level_split(self):
        importer = importlib.import_module("import_pop909")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            first = root / "001" / "001.mid"
            second = root / "001" / "versions" / "alternate.mid"
            self.make_midi(first)
            self.make_midi(second)
            a, b = importer.load_piece(first, root), importer.load_piece(second, root)
            self.assertEqual(a["source_piece_id"], b["source_piece_id"])
            self.assertNotEqual(a["piece_id"], b["piece_id"])

    def test_missing_named_melody_fails_instead_of_using_skyline(self):
        importer = importlib.import_module("import_pop909")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            path = root / "001" / "001.mid"
            self.make_midi(path, names=("PIANO",))
            with self.assertRaisesRegex(ValueError, "MELODY"):
                importer.load_piece(path, root)


if __name__ == "__main__":
    unittest.main()
