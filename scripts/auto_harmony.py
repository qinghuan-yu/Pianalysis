"""ReaLchords 离线模型适配：保留官方词表、四分之一拍网格及 on/hold 语义。

模型源代码与权重独立存放于忽略目录；本文件不复制或替换上游网络。
未知标签在时间轴上保留位置，但不作为 SILENCE 或监督目标。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import types
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".deps" / "auto_harmony"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import torch
import yaml

DEFAULT_UPSTREAM = ROOT / "data/auto_accompaniment/upstream/realchords"
PAD, BOS, EOS, SILENCE = 0, 1, 2, 3
FRAME_PER_BEAT = 4
EXPECTED_SOURCE_COMMIT = "75a63248a94d2139776ec9ddc20daf4c5be3cd07"
EXPECTED_WEIGHT_SHA256 = "e0e5f43f56c83506d77fa71e17aee5033eff64b513cdb1deba795a5204f78cba"
TEXTURE_INTERVALS = {
    (0, 4, 7), (0, 3, 7), (0, 3, 6), (0, 4, 8), (0, 4, 7, 10),
    (0, 4, 7, 11), (0, 3, 7, 10), (0, 3, 6, 9), (0, 3, 6, 10),
    (0, 2, 7), (0, 5, 7),
}
PRETRAINING_EXPOSURE = {
    "checkpoint": "lukewys/realchords-pytorch/enc_dec_base_chord_3_datasets/step=13000.ckpt",
    "datasets": ["Hooktheory", "POP909", "Nottingham"],
    "warning": "POP909-CL 与 POP909 是相同作品的新标注；重新划分不能消除预训练暴露。上游逐作品训练清单未核实，因此这些来源上的验证不代表未见作品泛化。",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: str | Path, value: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_upstream_files(upstream: Path) -> dict:
    """导入源码前核对最小上游资产；自训练权重也必须使用原始词表。"""
    manifest_path = upstream / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing source_manifest.json; run scripts/fetch_auto_models.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("ReaLchords source commit does not match the pinned adapter version")
    required = {
        "source/realchords/model/gen_model.py", "source/realchords/model/sampling.py",
        "source/realchords/nn/transformers.py", "source/LICENSE",
        "weights/chord_names_augmented.json", "weights/args.yml",
    }
    entries = {entry["relpath"]: entry for entry in manifest.get("files", [])}
    if not required.issubset(entries):
        raise ValueError(f"Incomplete ReaLchords source manifest: {sorted(required - entries.keys())}")
    checked = {}
    for relative in sorted(required):
        path = upstream / relative
        entry = entries[relative]
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream asset: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Upstream asset hash/size mismatch: {relative}; run fetch_auto_models.py --offline to inspect")
        checked[relative] = digest
    return {"manifest": str(manifest_path), "verified_sha256": checked}


@lru_cache(maxsize=1)
def chord_parser():
    """仅加载 note-seq 原版纯 Python 和弦模块，避免无关音频导入。

    临时 namespace 指向已安装包的真实目录，使用原 constants/parser；
    完成后恢复 sys.modules，不给其他调用者留下不完整的 note_seq 包。
    """
    if "note_seq" in sys.modules:
        return importlib.import_module("note_seq.chord_symbols_lib")
    spec = importlib.util.find_spec("note_seq")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install note-seq==0.0.5 (parser only; --no-deps is sufficient).")
    prefix = "note_seq"
    previous = {key: value for key, value in sys.modules.items()
                if key == prefix or key.startswith(prefix + ".")}
    package = types.ModuleType(prefix)
    package.__path__ = list(spec.submodule_search_locations)
    package.__spec__ = spec
    sys.modules[prefix] = package
    try:
        return importlib.import_module("note_seq.chord_symbols_lib")
    finally:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                del sys.modules[key]
        sys.modules.update(previous)


class HarmonyVocabulary:
    """ID 顺序严格遵循官方 HooktheoryTokenizer，绝不重新排序词表。"""

    def __init__(self, path: str | Path):
        self.path = resolve_path(path)
        self.names = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.names, list) or len(set(self.names)) != len(self.names):
            raise ValueError("Expected the original ordered unique chord-name list")
        self.sha256 = sha256_file(self.path)
        self.count = len(self.names)
        self.hold_start = 260
        self.on_start = 260 + self.count
        self.size = 260 + 2 * self.count
        parser = chord_parser()
        self.signatures = []
        self.by_signature: dict[tuple, list[int]] = {}
        for index, name in enumerate(self.names):
            signature = (parser.chord_symbol_root(name) % 12,
                         parser.chord_symbol_bass(name) % 12,
                         tuple(sorted(set(p % 12 for p in parser.chord_symbol_pitches(name)))))
            self.signatures.append(signature)
            self.by_signature.setdefault(signature, []).append(index)
        self.by_name = {name: index for index, name in enumerate(self.names)}
        self.canonical = {signature: min(indices, key=lambda i: (len(self.names[i]), i))
                          for signature, indices in self.by_signature.items()}
        self._transpose_maps: dict[int, np.ndarray] = {}

    def match(self, chord: dict) -> tuple[int | None, str | None]:
        if chord.get("status") == "no_chord":
            return SILENCE, None
        if chord.get("status") != "known":
            return None, "unknown_annotation"
        root, bass, pitches = chord.get("root"), chord.get("bass"), chord.get("pitch_classes")
        if root is None or bass is None or not pitches:
            return None, "missing_root_bass_or_pitches"
        signature = (int(root) % 12, int(bass) % 12, tuple(sorted(set(int(p) % 12 for p in pitches))))
        if signature not in self.by_signature:
            return None, "unsupported_exact_chord"
        # 保留原符号的既有 ID；若只是增强等音拼写，则选择同语义的最短官方名称。
        literal = self.by_name.get(chord.get("symbol", ""))
        index = literal if literal is not None and self.signatures[literal] == signature else self.canonical[signature]
        return self.on_start + index, None

    def transpose_map(self, shift: int) -> np.ndarray:
        if shift in self._transpose_maps:
            return self._transpose_maps[shift]
        mapping = np.zeros(self.size, dtype=np.int64)
        mapping[:4] = np.arange(4)
        for pitch in range(128):
            if 0 <= pitch + shift < 128:
                mapping[4 + pitch] = 4 + pitch + shift
                mapping[132 + pitch] = 132 + pitch + shift
        for index, (root, bass, pitches) in enumerate(self.signatures):
            signature = ((root + shift) % 12, (bass + shift) % 12,
                         tuple(sorted((p + shift) % 12 for p in pitches)))
            target = index if shift == 0 else self.canonical.get(signature)
            if target is not None:
                mapping[self.hold_start + index] = self.hold_start + target
                mapping[self.on_start + index] = self.on_start + target
        self._transpose_maps[shift] = mapping
        return mapping

    def summary(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "chords": self.count,
                "tokens": self.size, "pad": PAD, "bos": BOS, "eos": EOS,
                "silence": SILENCE, "note_hold": [4, 131], "note_on": [132, 259],
                "chord_hold": [260, self.on_start - 1], "chord_on": [self.on_start, self.size - 1]}


@dataclass
class EncodedSong:
    row: dict
    melody: np.ndarray
    chords: np.ndarray
    known: np.ndarray
    origin_beat: float
    audit: dict


def encode_song(row: dict, vocabulary: HarmonyVocabulary, *, overlap_policy: str = "reject") -> EncodedSong:
    if row.get("schema_version") != 1:
        raise ValueError("samples.jsonl schema_version must be 1")
    duration = float(row["duration_beats"])
    events = list(row.get("melody", [])) + list(row.get("chords", []))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_beats must be positive and finite")
    # 对负拍弱起只平移模型数组，原始时间轴通过 origin_beat 完整还原。
    origin = min(0.0, math.floor(min([0.0] + [float(e["start"]) for e in events]) * 4) / 4)
    length = math.ceil((duration - origin) * 4 - 1e-9)
    if length <= 0 or length > 1_000_000:
        raise ValueError("Unsupported song duration")
    melody = np.full(length, SILENCE, dtype=np.int64)
    chords = np.full(length, PAD, dtype=np.int64)
    known = np.zeros(length, dtype=bool)
    occupied = np.zeros(length, dtype=bool)
    chord_occupied = np.zeros(length, dtype=bool)
    errors: list[float] = []
    diagnostics: Counter = Counter()
    active_notes = []

    def span(event: dict) -> tuple[int, int]:
        start, end = float(event["start"]), float(event["end"])
        if not math.isfinite(start + end) or end <= start or end > duration + 1e-5:
            raise ValueError("Invalid/out-of-range event interval")
        left = math.floor((start - origin) * 4 + 1e-9)
        right = math.floor((end - origin) * 4 + 1e-9)
        if right == left:
            right += 1
            diagnostics["subgrid_events_extended"] += 1
        right = min(right, length)
        errors.extend([abs(origin + left / 4 - start), abs(origin + right / 4 - end)])
        return left, right

    for note in sorted(row.get("melody", []), key=lambda n: (n["start"], n["end"], n["pitch"])):
        pitch = int(note["pitch"])
        if not 0 <= pitch < 128:
            raise ValueError("Melody pitch outside MIDI range")
        left, right = span(note)
        collision = int(occupied[left:right].sum())
        diagnostics["melody_overlap_frames"] += collision
        active_notes = [other for other in active_notes if float(other["end"]) > float(note["start"]) + 1e-7]
        category = ("simultaneous_distinct_pitches" if any(abs(float(other["start"]) - float(note["start"])) < 1e-7 and int(other["pitch"]) != pitch for other in active_notes)
                    else "performance_note_overlap" if active_notes else "quantization_only_overlap")
        if collision:
            diagnostics[category + "_events"] += 1
        if collision and overlap_policy == "reject":
            raise ValueError(f"Quantized melody overlaps ({category}); explicit latest_onset policy or upstream repair required")
        if overlap_policy not in ("reject", "latest_onset"):
            raise ValueError("Unknown melody overlap policy")
        occupied[left:right] = True
        melody[left:right] = 4 + pitch
        melody[left] = 132 + pitch
        active_notes.append(note)

    # 冲突标注不能以列表顺序覆盖；最终屏蔽所有冲突帧。
    conflict = np.zeros(length, dtype=bool)
    for chord in sorted(row.get("chords", []), key=lambda c: (c["start"], c["end"])):
        left, right = span(chord)
        conflict[left:right] |= chord_occupied[left:right]
        chord_occupied[left:right] = True
        token, reason = vocabulary.match(chord)
        if reason:
            diagnostics[reason + "_events"] += 1
            continue
        known[left:right] = True
        if token == SILENCE:
            chords[left:right] = SILENCE
        else:
            chords[left:right] = token - vocabulary.count
            chords[left] = token
    known[conflict] = False
    chords[~known] = PAD
    diagnostics["conflicting_chord_frames"] = int(conflict.sum())
    diagnostics["unlabeled_gap_frames"] = int((~chord_occupied).sum())
    audit = {"song_id": row["song_id"], "frames": length, "known_frames": int(known.sum()),
             "origin_beat": origin, "overlap_policy": overlap_policy,
             "quantization": "floor at 4 frames/quarter; subframe spans extended to one frame",
             "quantization_abs_error_mean_beats": float(np.mean(errors)) if errors else 0.0,
             "quantization_abs_error_max_beats": max(errors, default=0.0), **dict(diagnostics)}
    return EncodedSong(row, melody, chords, known, origin, audit)


def read_samples(path: str | Path) -> list[dict]:
    rows, song_ids, split_by_work = [], set(), {}
    with resolve_path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity, work, split = row["song_id"], row["work_id"], row["split"]
            if identity in song_ids:
                raise ValueError(f"Duplicate song_id at line {line_number}: {identity}")
            if split not in ("train", "validation", "test"):
                raise ValueError(f"Invalid split: {split}")
            if work in split_by_work and split_by_work[work] != split:
                raise ValueError(f"Work leaks across splits: {work}")
            song_ids.add(identity)
            split_by_work[work] = split
            rows.append(row)
    return rows


def encode_corpus(rows: list[dict], vocabulary: HarmonyVocabulary, overlap_policy: str) -> tuple[list[EncodedSong], dict]:
    songs, rejected = [], []
    source_coverage: dict[str, Counter] = {}
    unsupported_types: Counter = Counter()
    unsupported_examples = []
    for row in rows:
        coverage = source_coverage.setdefault(row["source"], Counter())
        coverage["input_songs"] += 1
        for chord in row.get("chords", []):
            if chord.get("status") != "known":
                continue
            coverage["known_chord_events"] += 1
            _, reason = vocabulary.match(chord)
            if reason:
                coverage["unsupported_known_chord_events"] += 1
                unsupported_types[row["source"] + ":" + chord.get("quality", "unknown")] += 1
                if len(unsupported_examples) < 20:
                    unsupported_examples.append({"song_id": row["song_id"], "reason": reason,
                                                 "symbol": chord.get("symbol"), "root": chord.get("root"),
                                                 "bass": chord.get("bass"), "pitch_classes": chord.get("pitch_classes")})
        try:
            encoded = encode_song(row, vocabulary, overlap_policy=overlap_policy)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"song_id": row.get("song_id"), "reason": str(exc)})
            coverage["rejected_songs"] += 1
            continue
        songs.append(encoded)
        coverage["encoded_songs"] += 1
        coverage["encoded_frames"] += len(encoded.melody)
        coverage["supervised_frames"] += int(encoded.known.sum())
    summary = {"samples": len(rows), "encoded": len(songs), "rejected": rejected,
               "splits": dict(Counter(s.row["split"] for s in songs)),
               "sources": dict(Counter(s.row["source"] for s in songs)),
               "known_frames": sum(int(s.known.sum()) for s in songs),
               "total_frames": sum(len(s.melody) for s in songs),
               "vocabulary": vocabulary.summary(), "songs": [s.audit for s in songs],
               "coverage_by_source": {key: dict(value) for key, value in source_coverage.items()},
               "unsupported_known_types": dict(unsupported_types), "unsupported_examples": unsupported_examples,
               "pretraining_exposure": PRETRAINING_EXPOSURE}
    return songs, summary


def make_batch(windows: list[tuple[EncodedSong, int, int, int]], vocabulary: HarmonyVocabulary) -> dict:
    """窗口是(song,start_frame,length,transpose)；PAD仅作为未知/填充，占位不压缩。"""
    max_length = max(length for _, _, length, _ in windows)
    inputs = torch.zeros((len(windows), max_length + 2), dtype=torch.long)
    decoder = torch.zeros((len(windows), max_length + 1), dtype=torch.long)
    labels = torch.full_like(decoder, -100)
    label_mask = torch.zeros_like(decoder, dtype=torch.bool)
    frame_mask = torch.zeros_like(decoder, dtype=torch.bool)
    for i, (song, start, length, shift) in enumerate(windows):
        x = song.melody[start:start + length].copy()
        y = song.chords[start:start + length].copy()
        valid = song.known[start:start + length].copy()
        if shift:
            mapping = vocabulary.transpose_map(shift)
            shifted_x = mapping[x]
            if np.any((x != PAD) & (shifted_x == PAD)):
                raise ValueError("Transpose moves melody outside MIDI range")
            x, y = shifted_x, mapping[y]
            valid &= y != PAD
        inputs[i, :length + 2] = torch.as_tensor(np.concatenate(([BOS], x, [EOS])))
        decoder[i, 0] = BOS
        decoder[i, 1:length + 1] = torch.as_tensor(np.where(valid, y, PAD))
        labels[i, :length] = torch.as_tensor(np.where(valid, y, -100))
        labels[i, length] = EOS
        label_mask[i, :length] = torch.as_tensor(valid)
        label_mask[i, length] = True
        frame_mask[i, :length] = torch.as_tensor(valid)
    return {"inputs": inputs, "decoder": decoder, "labels": labels,
            "enc_mask": inputs.ne(PAD), "dec_mask": decoder.ne(PAD),
            "label_mask": label_mask, "frame_mask": frame_mask}


def load_model(upstream: str | Path = DEFAULT_UPSTREAM, checkpoint: str | Path | None = None):
    upstream = resolve_path(upstream)
    upstream_verification = verify_upstream_files(upstream)
    source = upstream / "source"
    if not (source / "realchords/model/gen_model.py").is_file():
        raise FileNotFoundError(f"Missing official ReaLchords source: {source}")
    sys.path.insert(0, str(source))
    from realchords.model.gen_model import EncoderDecoderTransformer

    vocabulary = HarmonyVocabulary(upstream / "weights/chord_names_augmented.json")
    official_checkpoint = upstream / "weights/step=13000.ckpt"
    checkpoint = resolve_path(checkpoint) if checkpoint else official_checkpoint
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint.resolve() == official_checkpoint.resolve() and checkpoint_sha256 != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Official ReaLchords checkpoint SHA256 mismatch; refusing to load")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") == "pianalysis_auto_harmony_v1":
        kwargs = payload["model_kwargs"]
        if payload["vocabulary_sha256"] != vocabulary.sha256:
            raise ValueError("Checkpoint vocabulary order/hash mismatch")
        state = payload["model"]
    else:
        if checkpoint_sha256 != EXPECTED_WEIGHT_SHA256:
            raise ValueError("Non-project checkpoint is not the pinned official ReaLchords weight file")
        config = yaml.safe_load((upstream / "weights/args.yml").read_text(encoding="utf-8"))
        kwargs = {key.split(".", 1)[1]: value for key, value in config.items()
                  if key.startswith("EncoderDecoderTransformer.")}
        kwargs.update(enc_num_tokens=vocabulary.size, dec_num_tokens=vocabulary.size, pad_value=PAD)
        state = {}
        for key, tensor in payload["state_dict"].items():
            key = key.removeprefix("model.").removeprefix("_orig_mod.")
            state[key] = tensor
    model = EncoderDecoderTransformer(**kwargs)
    model.load_state_dict(state, strict=True)
    if kwargs["enc_num_tokens"] != vocabulary.size or kwargs["dec_num_tokens"] != vocabulary.size:
        raise ValueError("Model embedding/output dimensions do not match vocabulary")
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
                "parameters": sum(p.numel() for p in model.parameters()), "model_kwargs": kwargs,
                "max_frames": min(kwargs["enc_max_seq_len"], kwargs["dec_max_seq_len"]) - 2,
                "vocabulary": vocabulary.summary(), "pretraining_exposure": PRETRAINING_EXPOSURE,
                "upstream_source_commit": EXPECTED_SOURCE_COMMIT,
                "upstream_verification": upstream_verification}
    return model, vocabulary, metadata, payload


def decode_plan(tokens: list[int], vocabulary: HarmonyVocabulary, origin: float, end_beat: float) -> list[dict]:
    result = []
    active = None
    for frame, token in enumerate(tokens):
        start = origin + frame / FRAME_PER_BEAT
        if start >= end_beat:
            break
        stop = min(end_beat, start + 1 / FRAME_PER_BEAT)
        onset = token >= vocabulary.on_start
        if token == SILENCE:
            signature = None
            name = "N"
        elif vocabulary.hold_start <= token < vocabulary.size:
            index = (token - vocabulary.hold_start) % vocabulary.count
            signature = vocabulary.signatures[index]
            name = vocabulary.names[index]
        else:
            raise ValueError(f"Invalid chord output token {token}; time axis must not be compacted")
        identity = (signature, name)
        if active is not None and active[0] == identity and not onset:
            result[-1]["end"] = stop
            continue
        if signature is None:
            event = {"start": start, "end": stop, "root": None, "bass": None,
                     "pitch_classes": [], "quality": "none", "symbol": "N", "status": "no_chord"}
        else:
            root, bass, pitches = signature
            event = {"start": start, "end": stop, "root": root, "bass": bass,
                     "pitch_classes": list(pitches), "quality": "custom", "symbol": name, "status": "known"}
        event["provenance"] = {"method": "realchords_offline_prediction", "human_verified": False}
        result.append(event)
        active = (identity, token)
    return result


@torch.no_grad()
def generate_plan(model, row: dict, vocabulary: HarmonyVocabulary, *, max_frames: int = 256,
                  overlap_frames: int = 32, temperature: float = 0.0,
                  overlap_policy: str = "reject", texture_compatible_chords: bool = False) -> dict:
    """固定时长解码，逐帧约束合法类型与HOLD。长曲以前窗尾部条件衔接。"""
    model.eval()
    if not 0 <= overlap_frames < max_frames:
        raise ValueError("overlap_frames must be smaller than max_frames")
    inference_input = dict(row)
    inference_input["chords"] = []
    encoded = encode_song(inference_input, vocabulary, overlap_policy=overlap_policy)
    device = next(model.parameters()).device
    chord_allowed = torch.ones(vocabulary.count, dtype=torch.bool, device=device)
    if texture_compatible_chords:
        chord_allowed = torch.tensor([tuple(sorted((pitch - root) % 12 for pitch in pitches)) in TEXTURE_INTERVALS
                                      for root, _, pitches in vocabulary.signatures], device=device)
    output: list[int] = []
    window_start = 0
    while len(output) < len(encoded.melody):
        stop = min(window_start + max_frames, len(encoded.melody))
        sequence = [BOS] + encoded.melody[window_start:stop].tolist() + [EOS]
        x = torch.tensor([sequence], device=device)
        x_mask = x.ne(PAD)
        context = model.encoder(x, mask=x_mask, return_embeddings=True)
        prefix = output[window_start:]
        decoder = torch.tensor([[BOS] + prefix], device=device)
        previous = prefix[-1] if prefix else None
        for _ in range(len(prefix), stop - window_start):
            logits = model.decoder(decoder, context=context, context_mask=x_mask,
                                   mask=decoder.ne(PAD))[0, -1].float()
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            allowed[SILENCE] = True
            allowed[vocabulary.on_start:] = chord_allowed
            if previous is not None and previous >= vocabulary.hold_start:
                allowed[vocabulary.hold_start + (previous - vocabulary.hold_start) % vocabulary.count] = True
            logits = logits.masked_fill(~allowed, -torch.inf)
            token = int(logits.argmax()) if temperature <= 0 else int(torch.multinomial((logits / temperature).softmax(-1), 1))
            output.append(token)
            decoder = torch.cat((decoder, torch.tensor([[token]], device=device)), dim=1)
            previous = token
        if stop == len(encoded.melody):
            break
        window_start = stop - overlap_frames
    result = copy.deepcopy(row)
    result["accompaniment"] = []
    result["chords"] = decode_plan(output, vocabulary, encoded.origin_beat, row["duration_beats"])
    result.setdefault("metadata", {})["auto_harmony"] = {
        "model": "ReaLchords offline encoder-decoder", "vocabulary_sha256": vocabulary.sha256,
        "frame_per_quarter": FRAME_PER_BEAT, "window_frames": max_frames,
        "overlap_frames": overlap_frames, "constrained_hold_and_fixed_duration": True,
        "texture_compatible_chords": texture_compatible_chords,
        "texture_bass_condition_supported": False if texture_compatible_chords else None,
        "texture_constraint_note": "Exact pitch-class set of 11 C&E chord types only; no chord simplification. Slash bass remains in output, but C&E has no explicit bass-conditioning channel." if texture_compatible_chords else None,
        "encoding_audit": encoded.audit, "pretraining_exposure": PRETRAINING_EXPOSURE}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("encode", help="编码并审计数据；不加载网络")
    audit.add_argument("--samples", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    audit.add_argument("--overlap-policy", choices=["reject", "latest_onset"], default="reject")
    infer = sub.add_parser("infer", help="完整旋律输入，输出带时间和声的同schema JSON")
    infer.add_argument("--samples", required=True)
    infer.add_argument("--song-id", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--checkpoint")
    infer.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--max-frames", type=int, default=256)
    infer.add_argument("--overlap-frames", type=int, default=32)
    infer.add_argument("--temperature", type=float, default=0.0)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--texture-compatible-chords", action="store_true")
    infer.add_argument("--overlap-policy", choices=["reject", "latest_onset"], default="reject")
    args = parser.parse_args()
    rows = read_samples(args.samples)
    if args.command == "encode":
        vocabulary = HarmonyVocabulary(resolve_path(args.upstream) / "weights/chord_names_augmented.json")
        _, report = encode_corpus(rows, vocabulary, args.overlap_policy)
        write_json(args.output, report)
        print(json.dumps({key: report[key] for key in ("samples", "encoded", "known_frames", "total_frames")}, ensure_ascii=False))
        return
    row = next((row for row in rows if row["song_id"] == args.song_id), None)
    if row is None:
        raise ValueError("song-id not found")
    torch.manual_seed(args.seed)
    model, vocabulary, metadata, _ = load_model(args.upstream, args.checkpoint)
    if not 1 <= args.max_frames <= metadata["max_frames"]:
        raise ValueError("max-frames exceeds the pretrained context")
    model.to(args.device)
    result = generate_plan(model, row, vocabulary, max_frames=args.max_frames,
                           overlap_frames=args.overlap_frames, temperature=args.temperature,
                           overlap_policy=args.overlap_policy, texture_compatible_chords=args.texture_compatible_chords)
    result["metadata"]["auto_harmony"]["checkpoint_sha256"] = metadata["checkpoint_sha256"]
    write_json(args.output, result)
    print(json.dumps({"song_id": args.song_id, "chords": len(result["chords"]), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
