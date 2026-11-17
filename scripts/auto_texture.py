"""Compose & Embellish-compatible accompaniment data, model and generation CLI."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data/auto_accompaniment/upstream/compose_embellish"
DEFAULT_CONFIG = ROOT / "examples/auto_accompaniment/texture_train.json"
PROTOCOL = "compose_embellish_accompaniment_only_v1"
QUALITY = {"maj": "M", "major": "M", "M": "M", "min": "m", "minor": "m", "m": "m",
           "dim": "o", "o": "o", "aug": "+", "+": "+", "7": "7", "maj7": "M7",
           "M7": "M7", "min7": "m7", "m7": "m7", "dim7": "o7", "o7": "o7",
           "hdim7": "/o7", "/o7": "/o7", "sus2": "sus2", "sus4": "sus4"}
CHORD_INTERVALS = {"M": (0, 4, 7), "m": (0, 3, 7), "o": (0, 3, 6), "+": (0, 4, 8),
                   "7": (0, 4, 7, 10), "M7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
                   "o7": (0, 3, 6, 9), "/o7": (0, 3, 6, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.setdefault("timing_policy", "strict_duration")
    config.setdefault("max_end_error_beats", 0.125)
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Unsupported(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


class Vocabulary:
    def __init__(self, upstream=UPSTREAM):
        self.upstream = Path(upstream)
        path = self.upstream / "stage02_embellish/vocab/skyline2midi_vocab.pkl"
        # 仅解析固定官方下载清单中已校验的词表，不接收任意 pickle。
        manifest = json.loads((self.upstream / "manifest.json").read_text(encoding="utf-8"))
        expected = next(item["sha256"] for item in manifest["files"] if item["path"].endswith("skyline2midi_vocab.pkl"))
        self.sha256 = sha256(path)
        if self.sha256 != expected:
            raise ValueError("Official vocabulary checksum mismatch")
        self.to_id, self.to_event = pickle.loads(path.read_bytes())
        if len(self.to_id) != 366 or self.to_id.get("Track_Midi") != 364:
            raise ValueError("Unexpected upstream vocabulary")
        self.pad = len(self.to_id)
        self.size = self.pad + 1
        self.tempo_values = sorted(int(x.split("_")[1]) for x in self.to_id if x.startswith("Tempo_") and x != "Tempo_Conti")
        self.velocity_values = sorted(int(x.split("_")[-1]) for x in self.to_id if x.startswith("Note_Velocity_"))

    def ids(self, events):
        return [self.to_id[event] for event in events]


class _TokenEmbedding(nn.Module):
    def __init__(self, size, width=512):
        super().__init__()
        self.emb_lookup = nn.Embedding(size, width)
        self.scale = math.sqrt(width)

    def forward(self, tokens):
        return self.emb_lookup(tokens) * self.scale


class _Positions(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        positions = torch.arange(12000, dtype=torch.float32).unsqueeze(1)
        factor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000) / width))
        values = torch.zeros(12000, 1, width)
        values[:, 0, 0::2] = torch.sin(positions * factor)
        values[:, 0, 1::2] = torch.cos(positions * factor)
        self.register_buffer("pe", values)


class TextureModel(nn.Module):
    """Matches every learned upstream GPT-2 parameter; no Hugging Face GPT2LMHead remap."""
    def __init__(self, vocab_size=367, gradient_checkpointing=False):
        super().__init__()
        self.token_emb = _TokenEmbedding(vocab_size)
        self.pe = _Positions()
        self.dec_out_proj = nn.Linear(512, vocab_size)
        config = GPT2Config(n_layer=12, n_head=8, n_embd=512, n_inner=2048,
                            resid_pdrop=0.1, attn_pdrop=0.1, max_position_embeddings=4096)
        config._attn_implementation = "eager"
        self.transformer_decoder = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(12)])
        self.emb_dropout = nn.Dropout(0.1)
        self.segemb = _TokenEmbedding(2)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, tokens, segments):
        if tokens.shape[1] > 2048:
            raise ValueError("Upstream GPT-2 context contract is 2048 tokens")
        hidden = self.token_emb(tokens) + self.segemb(segments)
        hidden = self.emb_dropout(hidden + self.pe.pe[:tokens.shape[1]].transpose(0, 1))
        # 右侧 padding 不影响先前有效查询；损失忽略 padding。
        for block in self.transformer_decoder:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(lambda x, layer=block: layer(x, use_cache=False)[0], hidden, use_reentrant=False)
            else:
                hidden = block(hidden, use_cache=False)[0]
        return self.dec_out_proj(hidden)


def load_model(vocab, checkpoint=None, device="cpu", gradient_checkpointing=False):
    model = TextureModel(vocab.size, gradient_checkpointing)
    official_path = vocab.upstream / "embellish_model_gpt2_pop1k7_loss0.398.bin"
    path = Path(checkpoint) if checkpoint else official_path
    checkpoint_hash = sha256(path)
    official_verified = path.resolve() == official_path.resolve()
    if official_verified:
        manifest = json.loads((vocab.upstream / "manifest.json").read_text(encoding="utf-8"))
        expected = next(item["sha256"] for item in manifest["files"] if item["path"] == official_path.name)
        if checkpoint_hash != expected:
            raise ValueError("Official checkpoint checksum mismatch; weights were not loaded")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload["model"] if "model" in payload else payload
    if metadata and metadata.get("vocabulary_sha256") != vocab.sha256:
        raise ValueError("Checkpoint vocabulary does not match")
    removed = []
    for key in list(state):
        # 旧 transformers 将固定因果掩码持久化；新版本自动重建，不丢弃任何学习参数。
        if key.endswith(".attn.bias") and tuple(state[key].shape) == (1, 1, 4096, 4096):
            removed.append(key)
            del state[key]
        elif key.endswith(".attn.masked_bias") and state[key].numel() == 1:
            removed.append(key)
            del state[key]
    model.load_state_dict(state, strict=True)
    model.to(device)
    metadata = dict(metadata)
    metadata.update({"checkpoint": str(path), "checkpoint_sha256": checkpoint_hash,
                     "official_checkpoint_sha256_verified": official_verified,
                     "removed_legacy_static_buffers": removed, "strict_parameter_load": True,
                     "parameters": sum(p.numel() for p in model.parameters())})
    return model, metadata


def load_samples(path):
    rows, song_ids, groups = [], set(), {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != 1:
                raise ValueError(f"Unsupported canonical schema at line {line_number}")
            identity = (row["source"] if isinstance(row["source"], str) else json.dumps(row["source"], sort_keys=True), row["song_id"])
            if identity in song_ids:
                raise ValueError(f"Duplicate canonical song: {identity}")
            song_ids.add(identity)
            split = row["split"]
            if split not in {"train", "validation", "test"}:
                raise ValueError("Unknown split")
            previous = groups.setdefault(row["work_id"], split)
            if previous != split:
                raise ValueError(f"Work split leakage: {row['work_id']}")
            rows.append(row)
    return rows


@dataclass
class Window:
    song_id: str
    work_id: str
    split: str
    start: float
    end: float
    tokens: list
    segments: list
    labels: list
    stats: dict


class Encoder:
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.config = config

    def _quantize(self, value, kind, stats):
        rounded = math.floor(value * 4 + 0.5) / 4
        error = abs(value - rounded)
        stats[kind + "_quantization_max"] = max(stats.get(kind + "_quantization_max", 0), error)
        stats[kind + "_quantization_sum"] = stats.get(kind + "_quantization_sum", 0) + error
        stats[kind + "_quantization_count"] = stats.get(kind + "_quantization_count", 0) + 1
        limit = self.config["max_duration_error_beats"] if kind == "duration" else self.config["max_onset_error_beats"]
        if error > limit + 1e-8:
            raise Unsupported("off_grid_" + kind, f"{value:.6f} -> {rounded:.6f}, error {error:.6f}")
        return rounded

    def _header(self, row, start, end, stats):
        if not row.get("time_signatures"):
            raise Unsupported("missing_meter")
        if any((t["numerator"], t["denominator"]) != (4, 4) for t in row["time_signatures"]):
            raise Unsupported("unsupported_meter")
        origin = row.get("metadata", {}).get("bar_origin_beat")
        if origin is None:
            raise Unsupported("unknown_bar_origin")
        if abs((start - origin) / 4 - round((start - origin) / 4)) > 1e-8:
            raise Unsupported("unaligned_window")
        if not row.get("melody"):
            raise Unsupported("empty_melody")
        tempos = row.get("tempos", [])
        if not tempos or not any(t["beat"] <= start + 1e-8 for t in tempos):
            raise Unsupported("missing_tempo")
        tempo = next(t["bpm"] for t in reversed(sorted(tempos, key=lambda x: x["beat"])) if t["beat"] <= start + 1e-8)
        if any(start < t["beat"] < end and abs(t["bpm"] - tempo) > 1e-6 for t in tempos):
            raise Unsupported("tempo_change_within_window")
        if not 32 <= tempo <= 224:
            raise Unsupported("tempo_out_of_vocabulary")
        quantized = min(self.vocab.tempo_values, key=lambda x: abs(x - tempo))
        stats["tempo_quantization_bpm"] = abs(quantized - tempo)
        return f"Tempo_{quantized}"

    def chord_event(self, chord, stats):
        status = chord.get("status")
        if status == "unknown":
            raise Unsupported("unknown_harmony")
        if status == "no_chord":
            return "Chord_None_None"
        if status != "known" or chord.get("root") is None:
            raise Unsupported("invalid_harmony")
        root = int(chord["root"])
        if root < 0 or root > 11:
            raise Unsupported("invalid_chord_root")
        pitches = sorted(set(chord.get("pitch_classes", [])))
        quality = QUALITY.get(chord.get("quality"))
        if chord.get("quality") == "custom":
            matches = [name for name, intervals in CHORD_INTERVALS.items() if sorted((root + x) % 12 for x in intervals) == pitches]
            if len(matches) == 1:
                quality = matches[0]
                stats["custom_chords_exactly_normalized"] = stats.get("custom_chords_exactly_normalized", 0) + 1
        if quality is None:
            raise Unsupported("unsupported_chord_quality", str(chord.get("quality")))
        expected = sorted((root + x) % 12 for x in CHORD_INTERVALS[quality])
        if pitches and pitches != expected:
            raise Unsupported("chord_pitch_set_mismatch")
        stats["condition_chords"] = stats.get("condition_chords", 0) + 1
        if chord.get("bass") is not None:
            stats["bass_conditions_omitted"] = stats.get("bass_conditions_omitted", 0) + 1
            if chord["bass"] != root:
                stats["inversion_bass_conditions_omitted"] = stats.get("inversion_bass_conditions_omitted", 0) + 1
        return f"Chord_{root}_{quality}"

    def _notes(self, notes, start, end, stats, role):
        selected = []
        raw = [n for n in notes if start <= n["start"] < end]
        if len({(n["pitch"], n["start"], n["end"], n["velocity"]) for n in raw}) != len(raw):
            raise Unsupported("duplicate_source_notes_" + role)
        if len({(n["pitch"], n["start"]) for n in raw}) != len(raw):
            raise Unsupported("ambiguous_source_unison_" + role)
        for n in notes:
            ns, ne = float(n["start"]), float(n["end"])
            if not (math.isfinite(ns) and math.isfinite(ne)) or ne <= ns:
                raise Unsupported("invalid_note_time")
            if ns < start - 1e-7 and ne > start + 1e-7:
                raise Unsupported("incoming_sustain_" + role)
            if not start <= ns < end:
                continue
            pitch, velocity = int(n["pitch"]), int(n["velocity"])
            if not 21 <= pitch <= 108 or not 1 <= velocity <= 127:
                raise Unsupported("note_out_of_vocabulary")
            onset = self._quantize(ns - start, "onset", stats)
            if not 0 <= onset < end - start:
                raise Unsupported("quantized_onset_outside_window")
            raw_duration = ne - ns
            if raw_duration > 4 + 1e-8:
                raise Unsupported("duration_over_four_beats")
            if self.config.get("timing_policy", "strict_duration") == "bounded_endpoints":
                # 可选有损协议：分别限制起止误差，不将短音无条件截成固定时值。
                quantized_end = math.floor((ne - start) * 4 + 0.5) / 4
                duration = max(0.25, quantized_end - onset)
                end_error = abs(start + onset + duration - ne)
                if end_error > self.config["max_end_error_beats"] + 1e-8:
                    raise Unsupported("off_grid_end")
                if duration > 4:
                    raise Unsupported("quantized_duration_over_four_beats")
                if raw_duration < 0.25:
                    stats["short_notes_expanded"] = stats.get("short_notes_expanded", 0) + 1
            else:
                duration = self._quantize(raw_duration, "duration", stats)
                if duration < 0.25:
                    if abs(raw_duration - 0.25) <= self.config["max_duration_error_beats"] + 1e-8:
                        duration = 0.25
                    else:
                        raise Unsupported("duration_under_grid")
                end_error = abs(start + onset + duration - ne)
            onset_error = abs(start + onset - ns)
            stats["end_quantization_max"] = max(stats.get("end_quantization_max", 0), end_error)
            stats["off_grid_notes_quantized"] = stats.get("off_grid_notes_quantized", 0) + int(onset_error > 1e-7)
            stats["onset_error_over_1_16_beat"] = stats.get("onset_error_over_1_16_beat", 0) + int(onset_error > 1 / 16 + 1e-7)
            vel = min(self.vocab.velocity_values, key=lambda x: abs(x - velocity))
            stats["velocity_quantization_max"] = max(stats.get("velocity_quantization_max", 0), abs(vel - velocity))
            selected.append({"start": onset, "duration": duration, "pitch": pitch, "velocity": vel})
        if len({(n["start"], n["pitch"]) for n in selected}) != len(selected):
            raise Unsupported("quantization_collapses_notes_" + role)
        return sorted(selected, key=lambda n: (n["start"], n["pitch"]))

    def encode(self, row, start, bars, require_target=True):
        if not isinstance(bars, int) or not 1 <= bars <= 128:
            raise Unsupported("unsupported_window_length")
        end = start + 4 * bars
        stats = {"bass_condition_supported": False}
        tempo = self._header(row, start, end, stats)
        melody = self._notes(row["melody"], start, end, stats, "melody")
        target = self._notes(row.get("accompaniment", []), start, end, stats, "accompaniment") if require_target else []
        if require_target and not target:
            raise Unsupported("empty_accompaniment")
        chords = sorted(row.get("chords", []), key=lambda c: c["start"])
        condition_bars, target_bars = [], []
        for bar in range(bars):
            bar_start, bar_end = start + bar * 4, start + (bar + 1) * 4
            matching = [c for c in chords if c["start"] < bar_end and c["end"] > bar_start]
            cursor = bar_start
            chord_at = {}
            for chord in matching:
                left, right = max(bar_start, chord["start"]), min(bar_end, chord["end"])
                if left > cursor + 1e-6:
                    raise Unsupported("unlabeled_harmony_gap")
                if left < cursor - 1e-6:
                    raise Unsupported("overlapping_harmony")
                event = self.chord_event(chord, stats)
                position = int(self._quantize(left - bar_start, "chord_boundary", stats) * 4)
                if position >= 16 or position in chord_at:
                    raise Unsupported("collapsed_chord_boundary")
                self._quantize(right - bar_start, "chord_boundary", stats)
                chord_at[position] = event
                cursor = right
            if cursor < bar_end - 1e-6:
                raise Unsupported("unlabeled_harmony_gap")
            mel_at, target_at = defaultdict(list), defaultdict(list)
            for note in melody:
                if bar * 4 <= note["start"] < (bar + 1) * 4:
                    mel_at[round((note["start"] - bar * 4) * 4)].append(note)
            for note in target:
                if bar * 4 <= note["start"] < (bar + 1) * 4:
                    target_at[round((note["start"] - bar * 4) * 4)].append(note)
            condition = ["Bar_None"]
            for position in sorted(set(chord_at) | set(mel_at)):
                condition.append(f"Beat_{position}")
                if position in chord_at:
                    condition.append(chord_at[position])
                for note in mel_at[position]:
                    condition.extend([f"Note_Pitch_{note['pitch']}", f"Note_Duration_{round(note['duration'] * 480)}"])
            performance = ["Bar_None"]
            for position in sorted(target_at):
                performance.append(f"Beat_{position}")
                for note in target_at[position]:
                    performance.extend([f"Note_Pitch_{note['pitch']}", f"Note_Duration_{round(note['duration'] * 480)}",
                                        f"Note_Velocity_{note['velocity']}"])
            condition_bars.append(condition)
            target_bars.append(performance)
        events, segments, predicted = [tempo], [0], [False]
        for i, (condition, performance) in enumerate(zip(condition_bars, target_bars)):
            events += ["Track_Skyline"] + condition + ["Track_Midi"] + performance
            segments += [0] * (len(condition) + 1) + [1] * (len(performance) + 1)
            predicted += [i > 0] + [False] * (len(condition) + 1) + [True] * len(performance)
        events.append("EOS_None")
        segments.append(1)
        predicted.append(True)
        tokens = self.vocab.ids(events)
        if require_target and len(tokens) - 1 > self.config["max_tokens"]:
            raise Unsupported("context_overflow", str(len(tokens) - 1))
        labels = [token if flag else -100 for token, flag in zip(tokens[1:], predicted[1:])]
        stats.update({"condition_bars": condition_bars, "tempo_event": tempo,
                      "target_notes": len(target), "melody_notes": len(melody), "target_tokens": sum(x != -100 for x in labels)})
        return Window(str(row["song_id"]), str(row["work_id"]), row["split"], start, end, tokens[:-1], segments[:-1], labels, stats)


def prepare_windows(rows, vocab, config, splits=("train", "validation")):
    encoder, windows = Encoder(vocab, config), []
    rejected, attempted, accepted = Counter(), Counter(), Counter()
    examples = []
    for row in rows:
        if row["split"] not in splits:
            continue
        if not row.get("accompaniment"):
            rejected["empty_accompaniment_song"] += 1
            continue
        origin = row.get("metadata", {}).get("bar_origin_beat")
        if origin is None:
            rejected["unknown_bar_origin_song"] += 1
            continue
        bars = config["window_bars"]
        total_bars = math.floor((row["duration_beats"] - origin) / 4 + 1e-7)
        for bar in range(max(0, math.ceil(-origin / 4)), total_bars - bars + 1, config["stride_bars"]):
            attempted[row["split"]] += 1
            try:
                window = encoder.encode(row, origin + 4 * bar, bars)
                windows.append(window)
                accepted[row["split"]] += 1
            except Unsupported as exc:
                rejected[exc.reason] += 1
                if len(examples) < 100:
                    examples.append({"song_id": row["song_id"], "split": row["split"], "bar": bar, "reason": str(exc)})
    statistics = {"attempted_windows": dict(attempted), "accepted_windows": dict(accepted), "rejected": dict(rejected),
                  "rejection_examples": examples, "accepted_songs": dict(Counter(split for split, _ in {(w.split, w.song_id) for w in windows})),
                  "max_tokens": max((len(w.tokens) for w in windows), default=0),
                  "bass_condition_supported": False, "targets": "canonical accompaniment only; melody/chord annotation tracks excluded upstream",
                  "timing_policy": config.get("timing_policy", "strict_duration"),
                  "timing_warning": "bounded_endpoints can change fine rhythmic subdivisions; report separately and listen before adoption" if config.get("timing_policy") == "bounded_endpoints" else None,
                  "test_used": "test" in splits}
    for key in ("onset_quantization_max", "duration_quantization_max", "end_quantization_max", "chord_boundary_quantization_max", "velocity_quantization_max", "tempo_quantization_bpm"):
        statistics[key] = max((w.stats.get(key, 0) for w in windows), default=0)
    for key in ("short_notes_expanded", "off_grid_notes_quantized", "onset_error_over_1_16_beat"):
        statistics[key] = sum(w.stats.get(key, 0) for w in windows)
    statistics["bass_conditions_omitted"] = sum(w.stats.get("bass_conditions_omitted", 0) for w in windows)
    statistics["inversion_bass_conditions_omitted"] = sum(w.stats.get("inversion_bass_conditions_omitted", 0) for w in windows)
    return windows, statistics


def collate(windows, vocab, device):
    width = max(len(w.tokens) for w in windows)
    tokens = torch.full((len(windows), width), vocab.pad, dtype=torch.long)
    segments = torch.zeros_like(tokens)
    labels = torch.full_like(tokens, -100)
    for index, window in enumerate(windows):
        length = len(window.tokens)
        tokens[index, :length] = torch.tensor(window.tokens)
        segments[index, :length] = torch.tensor(window.segments)
        labels[index, :length] = torch.tensor(window.labels)
    return tokens.to(device), segments.to(device), labels.to(device)


def loss_sum(logits, labels):
    count = int((labels != -100).sum())
    if not count:
        raise ValueError("No accompaniment targets in batch")
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"), count


def export_midi(row, accompaniment, start, end, path):
    import mido
    ppq = int(row["ppq"])
    original = [n for n in row["melody"] if start <= n["start"] < end]
    physical_end = max([end] + [n["end"] for n in original + accompaniment])
    midi = mido.MidiFile(ticks_per_beat=ppq)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    messages = []
    for key, kind in (("tempos", "set_tempo"), ("time_signatures", "time_signature")):
        values = sorted(row[key], key=lambda x: x["beat"])
        prior = [x for x in values if x["beat"] <= start]
        if not prior:
            raise ValueError("Missing MIDI timing metadata")
        for value in [dict(prior[-1], beat=start)] + [x for x in values if start < x["beat"] < physical_end]:
            message = mido.MetaMessage(kind, tempo=mido.bpm2tempo(value["bpm"])) if kind == "set_tempo" else mido.MetaMessage(kind, numerator=value["numerator"], denominator=value["denominator"])
            messages.append((round((value["beat"] - start) * ppq), message))
    previous = 0
    for tick, message in sorted(messages, key=lambda x: x[0]):
        meta.append(message.copy(time=tick - previous))
        previous = tick
    for channel, (name, notes) in enumerate((("MELODY", original), ("PIANO_ACCOMPANIMENT", accompaniment))):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=name))
        track.append(mido.Message("program_change", program=0, channel=channel))
        events = []
        for note in notes:
            for beat, on in ((note["start"], True), (note["end"], False)):
                events.append((round((beat - start) * ppq), 1 if on else 0,
                               mido.Message("note_on" if on else "note_off", channel=channel, note=int(note["pitch"]), velocity=int(note["velocity"]) if on else 0)))
        previous = 0
        for tick, _, message in sorted(events, key=lambda x: (x[0], x[1], x[2].note)):
            track.append(message.copy(time=tick - previous))
            previous = tick
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    # 从实际写出的 MIDI 回读，核对主旋律的 tick、力度和配对，而非仅相信内存拷贝。
    reread = mido.MidiFile(path)
    tick, active, decoded = 0, {}, []
    for message in reread.tracks[1]:
        tick += message.time
        if message.type == "note_on" and message.velocity:
            if message.note in active:
                raise ValueError("Overlapping same-pitch melody cannot be paired unambiguously")
            active[message.note] = (tick, message.velocity)
        elif message.type == "note_off" or (message.type == "note_on" and not message.velocity):
            onset, velocity = active.pop(message.note)
            decoded.append((message.note, onset, tick, velocity))
    expected = sorted((int(note["pitch"]), round((note["start"] - start) * ppq),
                       round((note["end"] - start) * ppq), int(note["velocity"])) for note in original)
    if active or sorted(decoded) != expected:
        raise ValueError("Exported MIDI melody differs from original selected melody")
    return original


def prediction_sample(row, accompaniment, start, end, checkpoint):
    original = [copy.deepcopy(n) for n in row["melody"] if start <= n["start"] < end]
    accompaniment = copy.deepcopy(accompaniment)
    physical_end = max([end] + [n["end"] for n in original + accompaniment])
    for note in original + accompaniment:
        note["start"] -= start
        note["end"] -= start
    result = {key: copy.deepcopy(row[key]) for key in ("schema_version", "song_id", "work_id", "source", "split", "ppq")}
    result.update({"melody": original, "accompaniment": accompaniment, "duration_beats": physical_end - start,
                   "chords": [dict(copy.deepcopy(chord), start=max(start, chord["start"]) - start,
                                   end=min(physical_end, chord["end"]) - start)
                              for chord in row["chords"] if chord["start"] < physical_end and chord["end"] > start]})
    for key in ("tempos", "time_signatures"):
        values = sorted(row[key], key=lambda value: value["beat"])
        prior = [value for value in values if value["beat"] <= start]
        result[key] = [dict(copy.deepcopy(prior[-1]), beat=0)] + [dict(copy.deepcopy(value), beat=value["beat"] - start) for value in values if start < value["beat"] < physical_end]
    result["metadata"] = {"bar_origin_beat": 0, "source_start_beat": start, "source_requested_end_beat": end,
                          "source_physical_end_beat": physical_end, "prediction_protocol": PROTOCOL,
                          "checkpoint_sha256": checkpoint["checkpoint_sha256"], "bass_condition_supported": False,
                          "original_melody_preserved_up_to_constant_time_shift": True,
                          "quality_status": "unreviewed", "source_metadata": copy.deepcopy(row.get("metadata", {}))}
    return result


@torch.inference_mode()
def generate(model, vocab, row, encoded, max_events_per_bar=384, temperature=1.0, top_p=0.95, seed=1234):
    seed_all(seed)
    model.eval()
    device = next(model.parameters()).device
    history = []
    notes, generated_events = [], []
    peak_context = 0
    for bar, condition in enumerate(encoded.stats["condition_bars"]):
        current_events = ["Track_Skyline"] + condition + ["Track_Midi"]
        current_segments = [0] * (len(condition) + 1) + [1]
        state, position, last_pitch, active_pitch = "bar", -1, -1, None
        duration = None
        events_in_bar = []
        for _ in range(max_events_per_bar):
            context = [(current_events, current_segments)]
            length = 1 + len(current_events)
            for old_events, old_segments in reversed(history):
                if length + len(old_events) > 2048:
                    break
                context.insert(0, (old_events, old_segments))
                length += len(old_events)
            if length > 2048:
                raise RuntimeError("One complete bar exceeds context; generation rejected")
            event_context = [encoded.stats["tempo_event"]] + [x for block, _ in context for x in block]
            segment_context = [0] + [x for _, block in context for x in block]
            peak_context = max(peak_context, len(event_context))
            if state == "bar":
                allowed = ["Bar_None"]
            elif state == "duration":
                remaining = encoded.end - (encoded.start + bar * 4 + position / 4)
                allowed = [f"Note_Duration_{ticks}" for ticks in range(120, min(1920, math.floor(remaining * 480)) + 1, 120)]
            elif state == "velocity":
                allowed = [f"Note_Velocity_{x}" for x in vocab.velocity_values]
            else:
                boundary = "EOS_None" if bar == len(encoded.stats["condition_bars"]) - 1 else "Track_Skyline"
                allowed = [] if state == "pitch" else [boundary] + [f"Beat_{x}" for x in range(position + 1, 16)]
                if position >= 0:
                    onset = encoded.start + bar * 4 + position / 4
                    occupied = {int(n["pitch"]) for n in row["melody"] + notes if n["start"] <= onset + 1e-8 < n["end"] - 1e-8}
                    allowed += [f"Note_Pitch_{pitch}" for pitch in range(max(21, last_pitch + 1), 109) if pitch not in occupied]
            if not allowed:
                raise RuntimeError("No admissible event in constrained generation")
            logits = model(torch.tensor([vocab.ids(event_context)], device=device), torch.tensor([segment_context], device=device))[0, -1].float()
            allowed_ids = torch.tensor(vocab.ids(allowed), device=device)
            probabilities = torch.softmax(logits[allowed_ids] / temperature, dim=0)
            ordered, order = torch.sort(probabilities, descending=True)
            retained = (ordered.cumsum(0) - ordered) < top_p
            filtered = ordered * retained
            choice = order[torch.multinomial(filtered / filtered.sum(), 1)].item()
            event = allowed[choice]
            if event in {"Track_Skyline", "EOS_None"}:
                break
            current_events.append(event)
            current_segments.append(1)
            events_in_bar.append(event)
            if event == "Bar_None":
                state = "position"
            elif event.startswith("Beat_"):
                position, last_pitch, state = int(event.split("_")[-1]), -1, "pitch"
            elif event.startswith("Note_Pitch_"):
                active_pitch, state = int(event.split("_")[-1]), "duration"
            elif event.startswith("Note_Duration_"):
                duration, state = int(event.split("_")[-1]) / 480, "velocity"
            elif event.startswith("Note_Velocity_"):
                onset = encoded.start + bar * 4 + position / 4
                notes.append({"pitch": active_pitch, "start": onset, "end": onset + duration, "velocity": int(event.split("_")[-1])})
                last_pitch, state = active_pitch, "position"
        else:
            raise RuntimeError(f"Bar {bar} exceeds max_events_per_bar; no truncated MIDI exported")
        history.append((current_events, current_segments))
        generated_events.append(events_in_bar)
    return notes, {"generated_events": generated_events, "peak_context_tokens": peak_context,
                   "melody_unisons_blocked": True, "context_policy": "whole previous bars; no partial-event truncation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "generate"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--song-id")
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--start-bar", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timing-policy", choices=("strict_duration", "bounded_endpoints"))
    parser.add_argument("--max-onset-error-beats", type=float)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    config = read_config(args.config)
    if args.timing_policy:
        config["timing_policy"] = args.timing_policy
    if args.max_onset_error_beats is not None:
        config["max_onset_error_beats"] = args.max_onset_error_beats
    vocab = Vocabulary(config["upstream_dir"])
    source_hash = sha256(args.samples)
    rows = load_samples(args.samples)
    if sha256(args.samples) != source_hash:
        raise ValueError("Canonical dataset changed while loading")
    if args.command == "audit":
        windows, report = prepare_windows(rows, vocab, config)
        report.update({"samples_sha256": source_hash, "vocabulary_sha256": vocab.sha256,
                       "config": config, "target_tokens": sum(w.stats["target_tokens"] for w in windows)})
        write_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"config", "rejection_examples"}}, ensure_ascii=True))
        return
    if not args.checkpoint or not args.song_id:
        parser.error("generate requires --checkpoint from accompaniment training and --song-id")
    row = next(row for row in rows if str(row["song_id"]) == args.song_id)
    origin = row.get("metadata", {}).get("bar_origin_beat")
    if origin is None:
        raise Unsupported("unknown_bar_origin")
    model, metadata = load_model(vocab, args.checkpoint, args.device)
    if metadata.get("protocol") != PROTOCOL or metadata.get("optimizer_steps", 0) < 1:
        raise ValueError("Full-performance upstream checkpoint cannot be merged with original melody; use an accompaniment-only finetuned checkpoint")
    encoded = Encoder(vocab, metadata.get("config", config)).encode(row, origin + 4 * args.start_bar, args.bars, require_target=False)
    started = time.monotonic()
    notes, report = generate(model, vocab, row, encoded, seed=args.seed)
    original = export_midi(row, notes, encoded.start, encoded.end, args.output)
    report.update({"song_id": row["song_id"], "source_split": row["split"], "checkpoint": metadata,
                   "seconds": time.monotonic() - started, "melody_original_notes": original, "accompaniment": notes,
                   "canonical_melody_preserved": True, "seed": args.seed, "bass_condition_supported": False,
                   "quality_status": "unreviewed; smoke training does not demonstrate musical quality"})
    write_json(str(args.output) + ".json", report)
    write_json(str(args.output) + ".prediction.json", prediction_sample(row, notes, encoded.start, encoded.end, metadata))
    print(json.dumps({"output": args.output, "accompaniment_notes": len(notes), "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
