# Pianalysis Tokenizer V2

Tokenizer V1 uses low-level `NOTE_ON`, `NOTE_OFF`, and relative `TIME` events.
It is reversible, but it asks the language model to learn note closure,
duration, bar placement, and polyphonic legality from raw event syntax. That is
why inference can produce hanging notes, very long ties, dense stacks, and odd
rests even when the token stream is technically valid.

Tokenizer V2 should move the model target closer to musical structure.

## Goals

- Encode each note with an explicit duration.
- Anchor events to bars and positions instead of unbounded time deltas.
- Make accompaniment density and rhythm easier to constrain.
- Keep melody conditioning separate from accompaniment targets.
- Preserve a deterministic MIDI -> tokens -> MIDI round trip.

## Proposed Vocabulary Shape

Special tokens:

```text
PAD
BOS
SEP
EOS
BAR
```

Position tokens:

```text
POS_0 ... POS_15
```

V1 uses 10 ms time deltas. V2 should quantize each bar into 16 positions first.
This intentionally favors stable musical rhythm over microscopic timing.

Note tokens:

```text
MEL_NOTE_<pitch>_<duration>_<velocity_bucket>
ACC_NOTE_<pitch>_<duration>_<velocity_bucket>
```

Recommended pitch range:

```text
21..108
```

Recommended duration bins, measured in 16th-note positions:

```text
1, 2, 3, 4, 6, 8, 12, 16, 24, 32
```

Recommended velocity buckets:

```text
pp, p, mp, mf, f, ff
```

Optional harmonic context tokens:

```text
CHORD_C:maj
CHORD_A:min
CHORD_G:7
CHORD_NC
```

Chord tokens should be added only after the note-duration representation is
stable. They are useful, but bad chord labels can hurt more than they help.

## Sequence Format

The conditional training sequence should remain source-target split:

```text
BOS
BAR POS_0 MEL_NOTE_64_2_mf POS_4 MEL_NOTE_67_4_mf ...
SEP
BAR POS_0 ACC_NOTE_48_8_mp ACC_NOTE_55_8_mp POS_8 ACC_NOTE_52_8_mp ...
EOS
```

The loss mask should stay unchanged in spirit:

```text
labels[0:target_start_index] = -100
labels[padding] = -100
```

Only accompaniment target tokens should contribute to training loss.

## Encoding Algorithm

1. Read MIDI notes and assign `role=melody` or `role=accompaniment`.
2. Estimate tempo and bar length from MIDI metadata when possible.
3. Quantize note starts to `(bar_index, pos_index)`.
4. Quantize durations to the closest allowed duration bin.
5. Drop or repair notes shorter than one position.
6. Sort events by `(bar_index, pos_index, role, pitch)`.
7. Emit one `BAR` token per bar, then position and note tokens.

## Decoding Algorithm

1. Maintain the current bar and current position.
2. Decode each `*_NOTE_<pitch>_<duration>_<velocity>` as a complete note.
3. Convert `(bar, pos, duration)` back to seconds or ticks.
4. Apply inference legality filters:
   - pitch range
   - maximum simultaneous notes
   - same-pitch overlap merge
   - melody clearance
   - maximum note duration

## Why This Helps

V1:

```text
TIME 30 NOTE_ON_ACCOMP 60 80 TIME 100 NOTE_OFF_ACCOMP 60
```

The model must remember to close the note later.

V2:

```text
POS_4 ACC_NOTE_60_8_mf
```

The duration is part of the event, so the model cannot create a hanging note by
forgetting the matching `NOTE_OFF`.

## Migration Plan

1. Keep V1 generation working for existing checkpoints.
2. Add `scripts/tokenizer_v2.py` with pure encode/decode functions.
3. Add a V2 dataset builder that emits `source_tokens`, `target_tokens`,
   `training_sequence`, and `target_start_index`.
4. Train a fresh model because V1 checkpoints are not vocabulary-compatible.
5. Compare V1 and V2 with the same melody MIDI prompts and the same legality
   report.
