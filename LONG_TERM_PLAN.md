# Pianalysis Long-Term Plan

## Goal

Build a system that takes a user-provided melody MIDI, generates a stylistic piano accompaniment texture, and exports a playable MIDI that preserves the original melody while adding model-generated accompaniment.

Target flow:

```text
melody.mid
-> tokenize melody as conditioning input
-> generate accompaniment tokens
-> decode accompaniment tokens to MIDI notes
-> merge original melody + generated accompaniment
-> output arranged_melody_with_accompaniment.mid
```

The model should not rewrite the input melody in the first production version. It should generate accompaniment only.

## Current Situation

There are two related projects:

- `vue-piano`: a melody annotation and tokenization tool.
- `Pianalysis`: a GPT-2 training/generation project.

`vue-piano` is closer to the correct data pipeline than `Pianalysis` because it already uses:

```text
[BOS] Source [SEP] Target [EOS]
```

where:

- `Source` is melody tokens.
- `Target` is full arrangement tokens.

However, both projects need changes before they can reliably train a melody-to-accompaniment model.

## Core Problems

### 1. Melody extraction is not reliable enough

`vue-piano` currently initializes melody labels with a Skyline algorithm: the highest active pitch inside a small time window is treated as melody.

This is useful as a first pass, but it is not reliable for piano arrangements because:

- The melody is not always the highest voice.
- Inner voices can temporarily cross above the melody.
- Arpeggios and ornamental notes may occupy the top register.
- Left-hand or accompaniment figures can leap above the melody.
- Chord voicings often place non-melody tones on top.
- Dense arrangements may contain several simultaneous melodic lines.

Therefore, Skyline output should be treated as a candidate annotation, not ground truth.

### 2. The current training target is not ideal

The current `vue-piano` token format stores:

```text
Source = melody
Target = full arrangement, including melody + accompaniment
```

For the desired product, a better training target is:

```text
Source = melody
Target = accompaniment only
```

At inference time, the original melody should be preserved and merged with the generated accompaniment. This avoids the model changing, dropping, or duplicating the input melody.

### 3. MIDI and token conversion is under-specified

The current numeric token format is compact:

```text
0, delta_time
10, pitch  = melody note on
11, pitch  = melody note off
20, pitch  = accompaniment note on
21, pitch  = accompaniment note off
```

This is workable for a prototype, but it loses or weakly handles important musical information:

- Velocity is dropped during decoding and replaced with a fixed value.
- Track, channel, program, pedal, tempo, and time signature are not preserved.
- Active notes are keyed only by pitch, so two same-pitch overlapping notes cannot be represented cleanly.
- `TIME_SHIFT` values are raw parameters mixed into the same integer stream as event IDs.
- Chord timing, bar position, beat position, and meter are not explicit.
- Round-trip fidelity cannot be guaranteed without strict validation.

### 4. Pianalysis training code does not yet use the conditional format correctly

`train_v2.py` currently trains on the whole `training_sequence` with labels equal to input IDs.

That means the model is trained to predict:

```text
BOS + melody + SEP + target + EOS + PAD
```

For conditional accompaniment generation, labels should ignore:

- `BOS`
- source melody tokens
- `SEP`
- padding tokens

Only target accompaniment tokens should contribute to the loss.

## Recommended Architecture

### Data model

Use a note-level intermediate JSON as the canonical data format before tokenization.

Example:

```json
{
  "piece_id": "song_001",
  "source_file": "song_001.mid",
  "duration": 123.45,
  "ticks_per_beat": 480,
  "tempo_map": [
    {"time": 0.0, "bpm": 92}
  ],
  "time_signatures": [
    {"time": 0.0, "numerator": 4, "denominator": 4}
  ],
  "notes": [
    {
      "id": 0,
      "track": 0,
      "channel": 0,
      "program": 0,
      "start": 0.0,
      "end": 0.5,
      "pitch": 60,
      "velocity": 82,
      "role": "melody"
    }
  ]
}
```

Allowed `role` values:

- `melody`
- `accompaniment`
- `ignore`
- `unknown`

Use this JSON as the bridge between `vue-piano`, preprocessing, training, evaluation, and MIDI reconstruction.

### Token format

Move from ambiguous raw integer streams to a documented event vocabulary.

Recommended first stable version:

```text
<BOS>
<STYLE_ANIME_PIANO>
<TEMPO_90>
<TS_4_4>
<SOURCE>
BAR
POS_0
PITCH_60
DUR_480
VEL_80
<TARGET>
BAR
POS_0
PITCH_48
DUR_960
VEL_70
BAR
POS_480
PITCH_55
DUR_480
VEL_66
<EOS>
```

For model input, these symbolic tokens can be mapped to continuous integer IDs by a tokenizer vocabulary.

Advantages:

- Bar/position tokens reduce timing drift.
- Duration tokens avoid fragile NOTE_ON/NOTE_OFF pairing.
- Velocity can be preserved or bucketed.
- The model learns musical grid structure.
- Round-trip conversion is easier to validate.

For the first version, quantize to musical positions instead of only milliseconds:

- Preserve tempo map from MIDI.
- Convert note start/end to beat ticks.
- Quantize to subdivisions such as 1/16 or 1/24 note.
- Store bar and position.

Keep a millisecond fallback only for unusual files.

## Data Cleaning Strategy

### Phase 1: Human-in-the-loop annotation

Continue using `vue-piano` as an annotation tool, but treat Skyline as a suggestion.

Add annotation quality features:

- Save annotated note JSON, not only exported MIDI.
- Add keyboard shortcuts for marking selected notes as melody/accompaniment/ignore.
- Add "solo melody" and "solo accompaniment" playback.
- Add a quality status field: `raw`, `auto_labeled`, `reviewed`, `approved`.
- Track whether a file has been manually reviewed.
- Show melody coverage statistics per bar.
- Warn when a bar has no melody or too many melody notes.

This is the most important near-term step. A smaller clean dataset is better than a large noisy one.

### Phase 2: Better automatic melody candidates

Replace pure Skyline with a scoring model that combines multiple heuristics:

- Pitch height.
- Note duration.
- Onset strength.
- Velocity.
- Legato continuity.
- Small melodic interval preference.
- Phrase continuity across bars.
- Track/channel names if available.
- Penalize dense chord tones.
- Penalize repeated accompaniment patterns.

Instead of marking all highest notes as melody, compute a melody likelihood score per note. Then use dynamic programming or Viterbi-style path selection to choose a coherent melody line.

Skyline can remain one feature inside this scorer.

### Phase 3: Train a melody role classifier

Once enough manually corrected data exists, train a separate model:

```text
input: full MIDI note graph
output: role per note: melody/accompaniment/ignore
```

This model is not the accompaniment generator. It is a data cleaning assistant. Its job is to reduce manual annotation work.

## Training Strategy

### First production target

Train:

```text
melody tokens -> accompaniment tokens
```

Do not train:

```text
melody tokens -> full arrangement tokens
```

The final MIDI should be:

```text
original melody MIDI + generated accompaniment MIDI
```

This keeps the user melody stable.

### Dataset sample structure

Each sample should contain:

```json
{
  "piece_id": "song_001",
  "slice_id": 0,
  "start_bar": 0,
  "end_bar": 8,
  "style": "anime_piano",
  "source_tokens": ["...melody..."],
  "target_tokens": ["...accompaniment..."],
  "input_ids": [1, "...", 2, "...", 3],
  "target_start_index": 123
}
```

Training labels:

```python
labels = input_ids.copy()
labels[:target_start_index] = -100
labels[padding_positions] = -100
```

This ensures the loss only trains accompaniment generation.

### Slicing

Use musical slicing, not fixed seconds:

- Prefer 4, 8, or 16 bars per sample.
- Include 1-bar context overlap if needed.
- Do not cut sustained notes without marking ties.
- Avoid training on empty or nearly empty melody slices.

Each slice should carry:

- `piece_id`
- `start_bar`
- `end_bar`
- tempo/time signature context
- style tags

### Model choice

Short term:

- Keep GPT-2 causal LM.
- Use `[BOS] controls + source + [SEP] target [EOS]`.
- Mask source loss.

Medium term:

- Move to encoder-decoder architecture.
- Encoder reads melody/control tokens.
- Decoder generates accompaniment tokens.

The short-term GPT-2 approach is easier to adapt from the current code. The encoder-decoder approach is cleaner once the data format is stable.

## MIDI/Token Round-Trip Requirements

Before training, implement round-trip tests.

Required tests:

1. MIDI -> note JSON -> MIDI should preserve note count, pitch, approximate timing, and velocity.
2. note JSON -> tokens -> note JSON should preserve note count, pitch, timing bucket, duration bucket, velocity bucket, and role.
3. training sample -> extract target -> MIDI should not duplicate source duration.
4. sliced samples -> reconstruct MIDI should preserve absolute timing.
5. generated invalid tokens should be repairable or rejected with diagnostics.

Add a report like:

```text
roundtrip_pitch_match: 100%
roundtrip_start_error_mean_ms: 3.2
roundtrip_duration_error_mean_ms: 4.8
missing_note_count: 0
extra_note_count: 0
velocity_bucket_match: 98.7%
```

No dataset should enter training unless it passes validation.

## Implementation Roadmap

### Milestone 1: Stabilize shared data format

- Create `schemas/note_schema.json`.
- Add `role` field instead of `is_melody`.
- Preserve tempo, time signature, track, channel, program, and velocity.
- Add import/export of annotated JSON in `vue-piano`.
- Keep MIDI export as a secondary convenience feature.

Deliverable:

```text
annotated_notes/*.json
```

### Milestone 2: Replace raw token protocol

- Create a shared tokenizer package or module used by both projects.
- Implement `notes_to_tokens()`.
- Implement `tokens_to_notes()`.
- Add vocabulary metadata.
- Add versioning: `tokenizer_version`.
- Add round-trip tests.

Deliverable:

```text
tokenizer/
  vocab.json
  tokenizer_config.json
  token_schema.md
tests/
  test_roundtrip.py
```

### Milestone 3: Build dataset generator

- Convert approved annotated JSON files into training samples.
- Slice by bars.
- Generate `source_tokens`, `target_tokens`, `training_sequence`, and `target_start_index`.
- Store dataset stats.
- Reject low-quality samples.

Deliverable:

```text
data/dataset_v1.json
data/dataset_report_v1.md
```

### Milestone 4: Fix training code

- Load `target_start_index`.
- Mask source and padding labels with `-100`.
- Add seed.
- Add config file support.
- Add tokenizer/data compatibility checks.
- Add train/eval split by piece, not by slice order.
- Save model card and training config with each checkpoint.

Deliverable:

```text
model_output/run_xxx/
  checkpoint-...
  training_config.json
  dataset_report.md
```

### Milestone 5: Build inference pipeline

- Input: melody MIDI.
- Tokenize melody as source.
- Generate accompaniment target.
- Decode accompaniment only.
- Merge with original melody.
- Export final MIDI.

Command:

```bash
python generate_accompaniment.py --melody input.mid --style anime_piano --out arranged.mid
```

Deliverable:

```text
arranged.mid
generation_report.json
```

### Milestone 6: Evaluation

Add automatic checks:

- Note density by bar.
- Pitch range of accompaniment.
- Hand range constraints.
- Chord collision with melody.
- Excessive dissonance flags.
- Empty-bar detection.
- Timing drift detection.

Add human checks:

- Melody preserved.
- Accompaniment fits harmony.
- Texture is playable.
- Style is recognizable.

## Immediate Changes Worth Doing First

1. In `Pianalysis/train_v2.py`, mask loss before `SEP` and on padding.
2. In `vue-piano`, export manually reviewed note JSON with full metadata.
3. Change target from full arrangement to accompaniment-only for model training.
4. Replace fixed velocity decoding with velocity tokens or velocity buckets.
5. Replace pitch-only active note tracking with `(pitch, role)` or note IDs.
6. Add round-trip tests before any new training.
7. Build a small approved dataset of 20-50 carefully corrected pieces before scaling.

## Practical Recommendation

Yes, the idea is implementable.

But the winning order is:

```text
annotation quality
-> reversible token protocol
-> dataset validation
-> masked conditional training
-> generation and MIDI merge
-> model scaling
```

Do not spend more time tuning model hyperparameters until the melody/accompaniment labels and MIDI/token round trip are trustworthy.

